#include "yolo_producer_cuda.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <stdexcept>

void ck(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
int main() {
    try {
        using namespace saccade;
        constexpr auto profile = ModelProfile::Hr1280x768;
        constexpr auto spec = kHrModelProfile;
        cudaStream_t stream = nullptr;
        ck(cudaSetDevice(0)); ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        float* input = nullptr; float* raw = nullptr;
        PostprocessProducerStatus* producer = nullptr; void* scratch = nullptr;
        ck(cudaMalloc(&input, spec.input_elements()*sizeof(float)));
        ck(cudaMalloc(&raw, spec.raw_elements()*sizeof(float)));
        ck(cudaMalloc(&producer, sizeof(*producer)));
        ck(cudaMalloc(&scratch, yolo_producer_workspace_bytes()));
        PostprocessRawView view;
        std::size_t offset = 0;
        for (unsigned i=0; i<3; ++i) {
            view.cls[i] = raw+offset; view.cls_elements[i] = spec.level_positions(i)*80;
            offset += view.cls_elements[i];
        }
        for (unsigned i=0; i<3; ++i) {
            view.reg[i] = raw+offset; view.reg_elements[i] = spec.level_positions(i)*4;
            offset += view.reg_elements[i];
        }
        if (offset != spec.raw_elements()) throw std::runtime_error("raw extent differs");
        std::cout << yolo_producer_cuda_load_kernels();
        const char* names[] = {"finite", "nan_input_last", "nan_cls_p5_last", "nan_reg_p5_last"};
        const float nan = std::numeric_limits<float>::quiet_NaN();
        for (unsigned test=0; test<4; ++test) {
            ck(cudaMemsetAsync(input,0,spec.input_elements()*sizeof(float),stream));
            ck(cudaMemsetAsync(raw,0,spec.raw_elements()*sizeof(float),stream));
            float* bad = test==1 ? input+spec.input_elements()-1 : test==2 ?
                const_cast<float*>(view.cls[2])+view.cls_elements[2]-1 :
                const_cast<float*>(view.reg[2])+view.reg_elements[2]-1;
            if (test) ck(cudaMemcpyAsync(bad,&nan,sizeof(nan),cudaMemcpyHostToDevice,stream));
            yolo_producer_cuda(input,spec.input_elements(),view,producer,scratch,
                               yolo_producer_workspace_bytes(),stream,profile);
            PostprocessProducerStatus result{};
            ck(cudaMemcpyAsync(&result,producer,sizeof(result),cudaMemcpyDeviceToHost,stream));
            ck(cudaStreamSynchronize(stream));
            if (result.done != kPostprocessProducerDone || result.flags != (test ? kPostprocessNumeric : 0U))
                throw std::runtime_error(std::string("finite gate failed: ")+names[test]);
            std::cout << names[test] << " PASS flags=" << result.flags << " done=" << result.done << '\n';
        }
        ck(cudaFree(scratch)); ck(cudaFree(producer)); ck(cudaFree(raw)); ck(cudaFree(input));
        ck(cudaStreamDestroy(stream));
        std::cout << "PASS four targeted controls; no scan plugin statuses exist\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}

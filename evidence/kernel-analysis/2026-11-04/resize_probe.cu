#include "saccade/resize_cuda.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
// Synthetic, isolated F02. It preserves actual shape and implementation, but
// has no nvJPEG, model, motion, concurrent stream, or real-image timing claim.
int main() {
    try {
        const saccade::ResizeShape shape{1920, 1080, 1280, 768};
        const auto ni = saccade::resize_input_bytes(shape);
        const auto no = saccade::resize_output_elements(shape);
        const auto nw = saccade::resize_workspace_elements(shape);
        std::uint8_t* input = nullptr;
        float* output = nullptr;
        float* workspace = nullptr;
        cudaStream_t stream = nullptr;
        check(cudaSetDevice(0));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaMalloc(&input, ni));
        check(cudaMalloc(&output, no * sizeof(float)));
        check(cudaMalloc(&workspace, nw * sizeof(float)));
        // Avoid uniform input/output compression as an accidental cache variable.
        std::vector<std::uint8_t> rgb(ni);
        std::uint32_t rng = 0x12345678u;
        for (auto& value : rgb) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            value = static_cast<std::uint8_t>(rng);
        }
        check(cudaMemcpyAsync(input, rgb.data(), ni, cudaMemcpyHostToDevice, stream));
        check(cudaMemsetAsync(output, 0, no * sizeof(float), stream));
        check(cudaMemsetAsync(workspace, 0, nw * sizeof(float), stream));
        std::cout << saccade::resize_cuda_load_kernels();
        for (int i = 0; i < 21; ++i)
            saccade::resize_rgb_chw_cuda(shape, input, ni, output, no, workspace, nw, stream);
        check(cudaStreamSynchronize(stream));
        check(cudaFree(workspace));
        check(cudaFree(output));
        check(cudaFree(input));
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

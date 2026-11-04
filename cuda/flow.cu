#include "saccade/flow_cuda.hpp"
#include "saccade/cleanup.hpp"
#include "nvOpticalFlowCuda.h"

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace saccade::flow {
namespace {
void ck(cudaError_t s, const char* op) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(op) + ": CUDA status=" + std::to_string(s) + " " + cudaGetErrorString(s));
}
void driver(CUresult s, const char* op) {
    if (s != CUDA_SUCCESS) throw std::runtime_error(std::string(op) + ": CUresult=" + std::to_string(s));
}
__global__ void grayscale(const std::uint8_t* rgb, std::uint8_t* dst, std::size_t pitch, int w, int h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const auto n = static_cast<std::size_t>(w) * h, i = static_cast<std::size_t>(y) * w + x;
    const float g = (0.299f * float(rgb[i]) + 0.587f * float(rgb[n+i])) + 0.114f * float(rgb[2*n+i]);
    dst[y*pitch+x] = static_cast<std::uint8_t>(fminf(255.f, floorf(g + 0.5f)));
}
__device__ float median4(int a, int b, int c, int d) {
    if (a > b) { const int t=a; a=b; b=t; }
    if (c > d) { const int t=c; c=d; d=t; }
    const int lo = a > c ? a : c, hi = b < d ? b : d;
    return float(lo + hi) / 64.f;
}
__global__ void cells_kernel(const std::uint8_t* raw, std::size_t pitch, Vector* out, int cols, int rows) {
    const int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y;
    if (x>=cols || y>=rows) return;
    const auto* a=reinterpret_cast<const short2*>(raw + 2*y*pitch) + 2*x;
    const auto* b=reinterpret_cast<const short2*>(raw + (2*y+1)*pitch) + 2*x;
    // API input=previous, reference=current: raw vectors already point forward.
    out[y*cols+x] = {median4(a[0].x,a[1].x,b[0].x,b[1].x), median4(a[0].y,a[1].y,b[0].y,b[1].y)};
}
}
struct Engine::Impl {
    int width, height, cols, rows;
    CUdevice device = 0;
    CUcontext context = nullptr;
    cudaStream_t stream = nullptr;
    void* library = nullptr;
    std::string path;
    NV_OF_CUDA_API_FUNCTION_LIST api{};
    NvOFHandle handle = nullptr;
    struct Buffer { NvOFGPUBufferHandle handle = nullptr; std::uint8_t* ptr = nullptr; std::size_t pitch = 0; };
    std::array<Buffer,2> input{};
    Buffer output;
    Vector* reduced = nullptr;
    std::array<cudaEvent_t, 6> events{};
    void of(NV_OF_STATUS status, const char* op) {
        if (status == NV_OF_SUCCESS) return;
        char error[1024]{}; std::uint32_t size = sizeof(error);
        if (handle && api.nvOFGetLastError) api.nvOFGetLastError(handle, error, &size);
        throw std::runtime_error(std::string(op) + ": NV_OF_STATUS=" + std::to_string(status) + " " + error);
    }
    void allocate(Buffer& b, int w, int h, NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT format) {
        const NV_OF_BUFFER_DESCRIPTOR desc{static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(h), usage, format};
        of(api.nvOFCreateGPUBufferCuda(handle, &desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &b.handle), "NvOFCreateGPUBufferCuda");
        b.ptr = reinterpret_cast<std::uint8_t*>(api.nvOFGPUBufferGetCUdeviceptr(b.handle));
        NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
        of(api.nvOFGPUBufferGetStrideInfo(b.handle, &stride), "NvOFGPUBufferGetStrideInfo");
        b.pitch = stride.strideInfo[0].strideXInBytes;
    }
    Impl(int w, int h, Perf perf): width(w), height(h), cols(w/8), rows(h/8) {
        try {
            driver(cuInit(0), "cuInit"); driver(cuDeviceGet(&device, 0), "cuDeviceGet");
            driver(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
            driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
            ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create IO stream");
            library = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW | RTLD_LOCAL);
            if (!library) throw std::runtime_error(std::string("dlopen: ") + dlerror());
            auto create = reinterpret_cast<decltype(&NvOFAPICreateInstanceCuda)>(dlsym(library,"NvOFAPICreateInstanceCuda"));
            if (!create) throw std::runtime_error("missing NvOFAPICreateInstanceCuda");
            Dl_info info{};
            if (!dladdr(reinterpret_cast<void*>(create), &info)) throw std::runtime_error("dladdr NVOF failed");
            path = std::filesystem::canonical(info.dli_fname).string();
            std::cout << "NVOF library=" << path << " API=2.0 width=" << w << " height=" << h
                      << " grid=4 perf=" << static_cast<int>(perf) << std::endl;
            of(create(NV_OF_API_VERSION, &api), "NvOFAPICreateInstanceCuda");
            of(api.nvCreateOpticalFlowCuda(context, &handle), "NvCreateOpticalFlowCuda");
            of(api.nvOFSetIOCudaStreams(handle, stream, stream), "NvOFSetIOCudaStreams");
            NV_OF_INIT_PARAMS init{};
            init.width=w; init.height=h; init.outGridSize=NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
            init.mode=NV_OF_MODE_OPTICALFLOW; init.perfLevel=static_cast<NV_OF_PERF_LEVEL>(perf);
            of(api.nvOFInit(handle, &init), "NvOFInit");
            std::cout << "NvOFInit NV_OF_STATUS=0\n";
            for (auto& b: input) allocate(b,w,h,NV_OF_BUFFER_USAGE_INPUT,NV_OF_BUFFER_FORMAT_GRAYSCALE8);
            allocate(output,(w+3)/4,(h+3)/4,NV_OF_BUFFER_USAGE_OUTPUT,NV_OF_BUFFER_FORMAT_SHORT2);
            ck(cudaMalloc(&reduced, static_cast<std::size_t>(cols)*rows*sizeof(Vector)), "allocate cells");
            for (auto& event: events) ck(cudaEventCreate(&event), "create timing event");
        } catch (...) { release(); throw; }
    }
    CleanupReport release() {
        CleanupReport r;
        if (stream) { int pending=1; r.release(pending,[&](int){ck(cudaStreamSynchronize(stream),"drain IO stream");}); }
        r.release(reduced,[](auto p){ck(cudaFree(p),"free cells");});
        for (auto& event: events) r.release(event,[](auto e){ck(cudaEventDestroy(e),"destroy timing event");});
        for (auto* b: {&input[0],&input[1],&output})
            r.release(b->handle,[&](auto h){of(api.nvOFDestroyGPUBufferCuda(h),"NvOFDestroyGPUBufferCuda");});
        r.release(handle,[&](auto h){of(api.nvOFDestroy(h),"NvOFDestroy");});
        r.release(library,[](auto p){if(dlclose(p)) throw std::runtime_error("dlclose NVOF failed");});
        r.release(stream,[](auto s){ck(cudaStreamDestroy(s),"destroy IO stream");});
        r.release(context,[&](auto){driver(cuDevicePrimaryCtxRelease(device),"cuDevicePrimaryCtxRelease");});
        return r;
    }
    ~Impl() { release(); }
    Frame finish(std::int64_t frame, Timing& timing) {
        Frame result{frame, frame == 0 ? Status::uninitialized : Status::valid,
                     std::vector<Vector>(static_cast<std::size_t>(cols)*rows)};
        if (frame > 0) {
            NV_OF_EXECUTE_INPUT_PARAMS in{};
            in.inputFrame=input[(frame-1)%2].handle; in.referenceFrame=input[frame%2].handle;
            in.disableTemporalHints=NV_OF_TRUE;
            NV_OF_EXECUTE_OUTPUT_PARAMS out{}; out.outputBuffer=output.handle;
            ck(cudaEventRecord(events[2], stream), "record Execute start");
            of(api.nvOFExecute(handle,&in,&out), "NvOFExecute");
            ck(cudaEventRecord(events[3], stream), "record Execute end");
            ck(cudaEventRecord(events[4], stream), "record cell start");
            const dim3 threads(16,16), blocks((cols+15)/16,(rows+15)/16);
            cells_kernel<<<blocks,threads,0,stream>>>(output.ptr,output.pitch,reduced,cols,rows);
            ck(cudaGetLastError(), "cell kernel");
            ck(cudaEventRecord(events[5], stream), "record cell end");
            ck(cudaMemcpyAsync(result.cells.data(), reduced, result.cells.size()*sizeof(Vector),cudaMemcpyDeviceToHost,stream),"copy cells");
        }
        ck(cudaStreamSynchronize(stream), "wait IO stream");
        ck(cudaEventElapsedTime(&timing.grayscale_ms, events[0], events[1]), "gray elapsed");
        if (frame > 0) {
            ck(cudaEventElapsedTime(&timing.execute_ms,events[2],events[3]), "Execute elapsed");
            ck(cudaEventElapsedTime(&timing.cell_ms,events[4],events[5]), "cell elapsed");
        }
        return result;
    }
};
Engine::Engine(int w,int h,Perf p):impl_(std::make_unique<Impl>(w,h,p)) {}
Engine::~Engine() = default;
cudaStream_t Engine::stream() const { return impl_->stream; }
std::string Engine::library_path() const { return impl_->path; }
Frame Engine::push_rgb(const std::uint8_t* rgb,std::int64_t frame,Timing& timing) {
    auto& s=*impl_; auto& b=s.input[frame%2];
    ck(cudaEventRecord(s.events[0],s.stream),"record gray start");
    const dim3 threads(16,16), blocks((s.width+15)/16,(s.height+15)/16);
    grayscale<<<blocks,threads,0,s.stream>>>(rgb,b.ptr,b.pitch,s.width,s.height);
    ck(cudaGetLastError(),"grayscale kernel");
    ck(cudaEventRecord(s.events[1],s.stream),"record gray end");
    return s.finish(frame,timing);
}
Frame Engine::push_gray(std::span<const std::uint8_t> gray,std::int64_t frame,Timing& timing) {
    auto& s=*impl_; auto& b=s.input[frame%2];
    ck(cudaEventRecord(s.events[0],s.stream),"record upload start");
    ck(cudaMemcpy2DAsync(b.ptr,b.pitch,gray.data(),s.width,s.width,s.height,cudaMemcpyHostToDevice,s.stream),"upload gray");
    ck(cudaEventRecord(s.events[1],s.stream),"record upload end");
    return s.finish(frame,timing);
}
void Engine::close() { impl_->release().throw_if_failed(); }
} // namespace saccade::flow

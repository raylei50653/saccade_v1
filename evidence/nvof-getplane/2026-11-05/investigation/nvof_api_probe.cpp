// Standalone API 2.0 reproducer: no Saccade, TensorRT, nvJPEG or CUDA kernels.
// Build and commands are recorded by run_probe.py next to this file.
#include "nvOpticalFlowCuda.h"
#include "array_trace.hpp"
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned width = 1920, height = 1080, grid = 4;
struct Vector { std::int16_t x, y; };
static_assert(sizeof(Vector) == 4);
void cuda(CUresult status, const char* name) {
    if (status != CUDA_SUCCESS)
        throw std::runtime_error(std::string(name) + " CUresult=" + std::to_string(status));
}
template<class F> void of(unsigned session, const std::string& phase, F call) {
    std::fprintf(stderr, "PROBE BEGIN session=%u phase=%s\n", session, phase.c_str());
    const auto status = call();
    std::fprintf(stderr, "PROBE END session=%u phase=%s status=%d\n", session, phase.c_str(), int(status));
    if (status != NV_OF_SUCCESS)
        throw std::runtime_error(phase + " NV_OF_STATUS=" + std::to_string(status));
}
struct Buffer {
    NvOFGPUBufferHandle handle = nullptr;
    CUdeviceptr ptr = 0;
    unsigned pitch = 0;
};
void upload(const Buffer& buffer, const std::vector<std::uint8_t>& image, CUstream stream) {
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST; copy.srcHost = image.data(); copy.srcPitch = width;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE; copy.dstDevice = buffer.ptr; copy.dstPitch = buffer.pitch;
    copy.WidthInBytes = width; copy.Height = height;
    cuda(cuMemcpy2DAsync(&copy, stream), "cuMemcpy2DAsync H2D");
}
void run(unsigned session, unsigned executes, bool init_only, bool default_stream, bool tracing,
         CUcontext context, NV_OF_CUDA_API_FUNCTION_LIST& api) {
    CUstream stream = nullptr;
    NvOFHandle handle = nullptr;
    Buffer input, reference, output;
    auto cleanup = [&] {
        if (stream) cuStreamSynchronize(stream);
        for (auto* buffer : {&input, &reference, &output}) if (buffer->handle) {
            api.nvOFDestroyGPUBufferCuda(buffer->handle); buffer->handle = nullptr;
        }
        if (handle) { api.nvOFDestroy(handle); handle = nullptr; }
        if (stream) { cuStreamDestroy(stream); stream = nullptr; }
    };
    try {
        if (!default_stream) cuda(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        of(session, "nvCreateOpticalFlowCuda", [&] { return api.nvCreateOpticalFlowCuda(context, &handle); });
        of(session, "nvOFSetIOCudaStreams", [&] { return api.nvOFSetIOCudaStreams(handle, stream, stream); });
        NV_OF_INIT_PARAMS init{};
        init.width = width; init.height = height; init.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
        init.mode = NV_OF_MODE_OPTICALFLOW; init.perfLevel = NV_OF_PERF_LEVEL_SLOW;
        of(session, "nvOFInit", [&] { return api.nvOFInit(handle, &init); });
        if (tracing) query_traced_arrays();
        if (!init_only) {
            auto allocate = [&](Buffer& buffer, unsigned w, unsigned h, NV_OF_BUFFER_USAGE usage,
                                NV_OF_BUFFER_FORMAT format, const char* label) {
                const NV_OF_BUFFER_DESCRIPTOR desc{w, h, usage, format};
                of(session, std::string("nvOFCreateGPUBufferCuda-") + label, [&] {
                    return api.nvOFCreateGPUBufferCuda(handle, &desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &buffer.handle);
                });
                buffer.ptr = api.nvOFGPUBufferGetCUdeviceptr(buffer.handle);
                NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
                of(session, std::string("nvOFGPUBufferGetStrideInfo-") + label, [&] {
                    return api.nvOFGPUBufferGetStrideInfo(buffer.handle, &stride);
                });
                buffer.pitch = stride.strideInfo[0].strideXInBytes;
                if (!buffer.ptr || stride.numPlanes != 1 || buffer.pitch < w * (format == NV_OF_BUFFER_FORMAT_SHORT2 ? 4u : 1u))
                    throw std::runtime_error("invalid returned buffer pointer/stride");
            };
            allocate(input, width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, "input");
            allocate(reference, width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, "reference");
            allocate(output, width / grid, height / grid, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, "output");
            std::vector<std::uint8_t> image(width * height), shifted(image.size());
            for (unsigned i = 0; i < image.size(); ++i) {
                std::uint32_t value = i + 0x9e3779b9u;
                value ^= value >> 16; value *= 0x7feb352du; value ^= value >> 15;
                value *= 0x846ca68bu; value ^= value >> 16; image[i] = value >> 24;
            }
            for (int y = 0; y < int(height); ++y) for (int x = 0; x < int(width); ++x)
                shifted[y * width + x] = image[std::max(0, y - 4) * width + std::max(0, x - 8)];
            upload(input, image, stream); upload(reference, shifted, stream);
            cuda(cuMemsetD2D8Async(output.ptr, output.pitch, 0, width / grid * sizeof(Vector), height / grid, stream), "clear output");
            cuda(cuStreamSynchronize(stream), "prepare stream");
            std::vector<Vector> baseline;
            for (unsigned iteration = 0; iteration < executes; ++iteration) {
                NV_OF_EXECUTE_INPUT_PARAMS in{};
                in.inputFrame = input.handle; in.referenceFrame = reference.handle; in.disableTemporalHints = NV_OF_TRUE;
                NV_OF_EXECUTE_OUTPUT_PARAMS out{}; out.outputBuffer = output.handle;
                of(session, "nvOFExecute#" + std::to_string(iteration), [&] { return api.nvOFExecute(handle, &in, &out); });
                if (tracing) query_traced_arrays();
                cuda(cuStreamSynchronize(stream), "complete Execute");
                std::vector<Vector> result(width / grid * (height / grid));
                CUDA_MEMCPY2D copy{};
                copy.srcMemoryType = CU_MEMORYTYPE_DEVICE; copy.srcDevice = output.ptr; copy.srcPitch = output.pitch;
                copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = result.data(); copy.dstPitch = width / grid * sizeof(Vector);
                copy.WidthInBytes = copy.dstPitch; copy.Height = height / grid;
                cuda(cuMemcpy2DAsync(&copy, stream), "read flow");
                cuda(cuStreamSynchronize(stream), "complete read");
                std::vector<int> xs, ys;
                for (unsigned y = 8; y + 8 < height / grid; ++y) for (unsigned x = 8; x + 8 < width / grid; ++x) {
                    const auto value = result[y * (width / grid) + x]; xs.push_back(value.x); ys.push_back(value.y);
                }
                std::nth_element(xs.begin(), xs.begin() + xs.size() / 2, xs.end());
                std::nth_element(ys.begin(), ys.begin() + ys.size() / 2, ys.end());
                const double dx = xs[xs.size() / 2] / 32., dy = ys[ys.size() / 2] / 32.;
                const bool repeated = baseline.empty() || std::memcmp(baseline.data(), result.data(), result.size() * sizeof(Vector)) == 0;
                std::fprintf(stderr, "PROBE FLOW session=%u iteration=%u median_x=%.5f median_y=%.5f repeated_bytes=%u\n",
                             session, iteration, dx, dy, unsigned(repeated));
                if (!repeated || std::abs(dx - 8) > 2 || std::abs(dy - 4) > 2)
                    throw std::runtime_error("flow validation failed");
                baseline = std::move(result);
            }
            for (auto* buffer : {&input, &reference, &output}) {
                of(session, "nvOFDestroyGPUBufferCuda", [&] { return api.nvOFDestroyGPUBufferCuda(buffer->handle); });
                buffer->handle = nullptr;
            }
        }
        of(session, "nvOFDestroy", [&] { return api.nvOFDestroy(handle); }); handle = nullptr;
        if (stream) { cuda(cuStreamDestroy(stream), "cuStreamDestroy"); stream = nullptr; }
    } catch (...) { cleanup(); throw; }
}
}
int main(int argc, char** argv) {
    try {
        unsigned sessions = 1, executes = 2;
        bool init_only = false, tracing = false, default_stream = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--sessions" && i + 1 < argc) sessions = std::stoul(argv[++i]);
            else if (arg == "--executes" && i + 1 < argc) executes = std::stoul(argv[++i]);
            else if (arg == "--init-only") init_only = true;
            else if (arg == "--cupti") tracing = true;
            else if (arg == "--default-stream") default_stream = true;
            else throw std::runtime_error("usage: nvof_api_probe [--sessions N] [--executes N] [--init-only] [--default-stream] [--cupti]");
        }
        if (tracing) start_array_trace();
        cuda(cuInit(0), "cuInit");
        CUdevice device = 0; cuda(cuDeviceGet(&device, 0), "cuDeviceGet");
        CUcontext context = nullptr; cuda(cuDevicePrimaryCtxRetain(&context, device), "retain primary context");
        cuda(cuCtxSetCurrent(context), "cuCtxSetCurrent");
        void* library = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        auto create = reinterpret_cast<decltype(&NvOFAPICreateInstanceCuda)>(dlsym(library, "NvOFAPICreateInstanceCuda"));
        if (!create) throw std::runtime_error("missing NvOFAPICreateInstanceCuda");
        NV_OF_CUDA_API_FUNCTION_LIST api{};
        of(0, "NvOFAPICreateInstanceCuda", [&] { return create(NV_OF_API_VERSION, &api); });
        for (unsigned session = 0; session < sessions; ++session)
            run(session, executes, init_only, default_stream, tracing, context, api);
        if (dlclose(library)) throw std::runtime_error("dlclose failed");
        cuda(cuDevicePrimaryCtxRelease(device), "release primary context");
        if (tracing) stop_array_trace();
        std::fprintf(stderr, "PROBE PASS sessions=%u executes_per_session=%u init_only=%u default_stream=%u\n",
                     sessions, executes, unsigned(init_only), unsigned(default_stream));
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "PROBE FAIL %s\n", error.what()); return 1; }
}

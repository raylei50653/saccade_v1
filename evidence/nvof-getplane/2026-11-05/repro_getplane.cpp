// Minimal repro: valid NVOF API use reports failing cuArrayGetPlane calls under
// Compute Sanitizer. Uses only the CUDA Driver API and the NVOF library, with no
// kernels and no CUDA arrays of its own.
//
// usage: repro_getplane EXECUTES    (0 stops after nvOFInit)
// exit:  0 = every CUDA and NVOF call made here succeeded, 1 = one of them failed
#include "nvOpticalFlowCuda.h"
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static void cu(CUresult status, const char* name) {
    if (status == CUDA_SUCCESS) return;
    std::printf("FAILED %s: CUresult=%d\n", name, int(status));
    std::exit(1);
}

static void of(NV_OF_STATUS status, const char* name) {
    std::printf("%s: %s\n", name, status == NV_OF_SUCCESS ? "NV_OF_SUCCESS" : "FAILED");
    std::fflush(stdout);
    if (status != NV_OF_SUCCESS) std::exit(1);
}

static void* load_create_function() {
#ifdef _WIN32
    HMODULE library = LoadLibraryA("nvofapi64.dll");
    return library ? reinterpret_cast<void*>(GetProcAddress(library, "NvOFAPICreateInstanceCuda")) : nullptr;
#else
    void* library = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW);
    return library ? dlsym(library, "NvOFAPICreateInstanceCuda") : nullptr;
#endif
}

int main(int argc, char** argv) {
    if (argc != 2) { std::printf("usage: repro_getplane EXECUTES\n"); return 1; }
    const int executes = std::atoi(argv[1]);
    const unsigned width = 1920, height = 1080, grid = 4;

    cu(cuInit(0), "cuInit");
    CUdevice device;
    CUcontext context;
    cu(cuDeviceGet(&device, 0), "cuDeviceGet");
    cu(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    cu(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    CUstream stream;
    cu(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");

    auto create = reinterpret_cast<decltype(&NvOFAPICreateInstanceCuda)>(load_create_function());
    if (!create) { std::printf("FAILED to load the NVOF library\n"); return 1; }
    NV_OF_CUDA_API_FUNCTION_LIST api{};
    of(create(NV_OF_API_VERSION, &api), "NvOFAPICreateInstanceCuda");

    NvOFHandle session = nullptr;
    of(api.nvCreateOpticalFlowCuda(context, &session), "nvCreateOpticalFlowCuda");
    of(api.nvOFSetIOCudaStreams(session, stream, stream), "nvOFSetIOCudaStreams");
    NV_OF_INIT_PARAMS init{};
    init.width = width;
    init.height = height;
    init.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
    init.mode = NV_OF_MODE_OPTICALFLOW;
    init.perfLevel = NV_OF_PERF_LEVEL_SLOW;
    std::printf("-- nvOFInit begins\n");
    std::fflush(stdout);
    of(api.nvOFInit(session, &init), "nvOFInit");

    if (executes > 0) {
        const auto allocate = [&](unsigned w, unsigned h, NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT format, unsigned element) {
            const NV_OF_BUFFER_DESCRIPTOR description{w, h, usage, format};
            NvOFGPUBufferHandle buffer = nullptr;
            of(api.nvOFCreateGPUBufferCuda(session, &description, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &buffer), "nvOFCreateGPUBufferCuda");
            NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
            of(api.nvOFGPUBufferGetStrideInfo(buffer, &stride), "nvOFGPUBufferGetStrideInfo");
            cu(cuMemsetD2D8Async(api.nvOFGPUBufferGetCUdeviceptr(buffer), stride.strideInfo[0].strideXInBytes, 0, w * element, h, stream), "cuMemsetD2D8Async");
            return buffer;
        };
        NvOFGPUBufferHandle input = allocate(width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, 1);
        NvOFGPUBufferHandle reference = allocate(width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, 1);
        NvOFGPUBufferHandle output = allocate(width / grid, height / grid, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, 4);
        cu(cuStreamSynchronize(stream), "cuStreamSynchronize");

        for (int i = 0; i < executes; ++i) {
            NV_OF_EXECUTE_INPUT_PARAMS in{};
            in.inputFrame = input;
            in.referenceFrame = reference;
            in.disableTemporalHints = NV_OF_TRUE;
            NV_OF_EXECUTE_OUTPUT_PARAMS out{};
            out.outputBuffer = output;
            std::printf("-- nvOFExecute %d begins\n", i);
            std::fflush(stdout);
            of(api.nvOFExecute(session, &in, &out), "nvOFExecute");
            cu(cuStreamSynchronize(stream), "cuStreamSynchronize");
        }
        of(api.nvOFDestroyGPUBufferCuda(input), "nvOFDestroyGPUBufferCuda");
        of(api.nvOFDestroyGPUBufferCuda(reference), "nvOFDestroyGPUBufferCuda");
        of(api.nvOFDestroyGPUBufferCuda(output), "nvOFDestroyGPUBufferCuda");
    }
    of(api.nvOFDestroy(session), "nvOFDestroy");
    cu(cuStreamDestroy(stream), "cuStreamDestroy");
    cu(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    std::printf("repro finished: every CUDA and NVOF call made here succeeded\n");
    return 0;
}

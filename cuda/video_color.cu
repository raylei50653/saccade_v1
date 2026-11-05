#include "saccade/video_color_cuda.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace saccade {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void cu(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
void drv(CUresult error, const char* operation) {
    if (error != CUDA_SUCCESS) {
        const char* message = nullptr;
        (void)cuGetErrorString(error, &message);
        throw std::runtime_error(std::string(operation) + ": " + (message ? message : "CUDA driver failure"));
    }
}
void pointer(const void* value, int device, CUcontext context) {
    require(value != nullptr, "device video plane/output must not be null");
    cudaPointerAttributes attributes{};
    const auto result = cudaPointerGetAttributes(&attributes, value);
    if (result == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        throw std::invalid_argument("device video requires device pointers");
    }
    cu(result, "video cudaPointerGetAttributes");
    require(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
            "device video pointer must belong to current device");
    CUcontext allocation = nullptr;
    drv(cuPointerGetAttribute(&allocation, CU_POINTER_ATTRIBUTE_CONTEXT, reinterpret_cast<CUdeviceptr>(value)),
        "video pointer context");
    require(allocation == context, "device video pointer belongs to another context");
}
bool overlaps(const void* a, std::size_t na, const void* b, std::size_t nb) {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    require(na <= std::numeric_limits<std::uintptr_t>::max() - x &&
            nb <= std::numeric_limits<std::uintptr_t>::max() - y, "device video pointer range overflow");
    return x < y + nb && y < x + na;
}
std::size_t extent(std::size_t pitch, std::size_t rows, std::size_t row_bytes) {
    require(pitch >= row_bytes, "device video pitch is shorter than a row");
    require(rows > 0 && (rows - 1) <= (std::numeric_limits<std::size_t>::max() - row_bytes) / pitch,
            "device video pitch extent overflow");
    return (rows - 1) * pitch + row_bytes;
}
struct Coefficients { float kr, kb, yoffset, yscale, ccenter, cscale; };
__device__ std::uint8_t quantize(float value) {
    return static_cast<std::uint8_t>(__float2int_rn(fminf(255.0f, fmaxf(0.0f, value * 255.0f))));
}
template<bool TenBit>
__global__ void convert(const std::uint8_t* luma, const std::uint8_t* chroma,
                        std::size_t yp, std::size_t uvp, int width, int height,
                        Coefficients c, std::uint8_t* output) {
    const auto pixels = static_cast<std::size_t>(width) * height;
    for (std::size_t p = blockIdx.x * blockDim.x + threadIdx.x; p < pixels;
         p += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const auto x = p % width, y = p / width;
        float yy, u, v;
        if constexpr (TenBit) {
            yy = reinterpret_cast<const std::uint16_t*>(luma + y * yp)[x] >> 6;
            const auto* row = reinterpret_cast<const std::uint16_t*>(chroma + (y / 2) * uvp);
            u = row[2 * (x / 2)] >> 6; v = row[2 * (x / 2) + 1] >> 6;
        } else {
            yy = luma[y * yp + x];
            u = chroma[(y / 2) * uvp + 2 * (x / 2)]; v = chroma[(y / 2) * uvp + 2 * (x / 2) + 1];
        }
        yy = (yy - c.yoffset) * c.yscale;
        const float cb = (u - c.ccenter) * c.cscale, cr = (v - c.ccenter) * c.cscale;
        const float r = yy + (2.0f - 2.0f * c.kr) * cr;
        const float b = yy + (2.0f - 2.0f * c.kb) * cb;
        const float g = yy - (2.0f * c.kb * (1.0f - c.kb) * cb +
                              2.0f * c.kr * (1.0f - c.kr) * cr) / (1.0f - c.kr - c.kb);
        output[p] = quantize(r); output[pixels + p] = quantize(g); output[2 * pixels + p] = quantize(b);
    }
}
} // namespace

void video_color_cuda(const DeviceVideoFrame& input, std::uint8_t* output,
                      std::size_t capacity, cudaStream_t stream) {
    require(stream && stream != cudaStreamLegacy && stream != cudaStreamPerThread,
            "device video requires explicit nondefault consumer stream");
    require(input.width > 0 && input.width <= 8192 && input.height > 0 && input.height <= 8192,
            "device video dimensions must be in 1..8192");
    require(input.format == VideoSurfaceFormat::Nv12 || input.format == VideoSurfaceFormat::P010,
            "device video format must be NV12 or P010");
    require(input.matrix == VideoColorMatrix::Bt601 || input.matrix == VideoColorMatrix::Bt709 || input.matrix == VideoColorMatrix::Bt2020,
            "unsupported device video color matrix");
    require(input.context && input.ready_event && input.owner, "device video requires context, readiness event and owner");
    int device = -1, stream_device = -1;
    cu(cudaGetDevice(&device), "video cudaGetDevice");
    cu(cudaStreamGetDevice(stream, &stream_device), "video cudaStreamGetDevice");
    require(device == input.device && stream_device == device, "device video device/stream mismatch");
    CUcontext current = nullptr, producer_context = nullptr;
    drv(cuCtxGetCurrent(&current), "video current context");
    require(current && current == input.context, "device video must use current CUDA context");
    drv(cuStreamGetCtx(reinterpret_cast<CUstream>(input.stream), &producer_context), "video producer context");
    require(producer_context == current, "device video producer stream belongs to another context");
    cudaStreamCaptureStatus capture{};
    cu(cudaStreamIsCapturing(stream, &capture), "video capture status");
    require(capture == cudaStreamCaptureStatusNone, "device video preprocessing does not support graph capture");
    const std::size_t sample = input.format == VideoSurfaceFormat::P010 ? 2 : 1;
    const auto ybytes = extent(input.pitches[0], input.height, static_cast<std::size_t>(input.width) * sample);
    const auto uvbytes = extent(input.pitches[1], (input.height + 1) / 2,
                                static_cast<std::size_t>((input.width + 1) / 2) * 2 * sample);
    const auto output_bytes = static_cast<std::size_t>(input.width) * input.height * 3;
    require(capacity >= output_bytes, "device video RGB output is too short");
    if (sample == 2) {
        require(input.pitches[0] % 2 == 0 && input.pitches[1] % 2 == 0 &&
                reinterpret_cast<std::uintptr_t>(input.planes[0]) % 2 == 0 &&
                reinterpret_cast<std::uintptr_t>(input.planes[1]) % 2 == 0,
                "P010 planes and pitches must be word aligned");
    }
    pointer(input.planes[0], device, current); pointer(input.planes[1], device, current); pointer(output, device, current);
    require(!overlaps(input.planes[0], ybytes, output, output_bytes) &&
            !overlaps(input.planes[1], uvbytes, output, output_bytes), "device video RGB output aliases input");
    require(!overlaps(input.planes[0], ybytes, input.planes[1], uvbytes), "device video planes overlap");
    Coefficients c{0.299f, 0.114f, 0, 0, 0, 0};
    if (input.matrix == VideoColorMatrix::Bt709) { c.kr = 0.2126f; c.kb = 0.0722f; }
    if (input.matrix == VideoColorMatrix::Bt2020) { c.kr = 0.2627f; c.kb = 0.0593f; }
    const float scale = sample == 2 ? 4.0f : 1.0f;
    const float maximum = sample == 2 ? 1023.0f : 255.0f;
    c.yoffset = input.full_range ? 0.0f : 16.0f * scale;
    c.yscale = 1.0f / (input.full_range ? maximum : 219.0f * scale);
    c.ccenter = 128.0f * scale;
    c.cscale = 1.0f / (input.full_range ? maximum : 224.0f * scale);
    cu(cudaStreamWaitEvent(stream, reinterpret_cast<cudaEvent_t>(input.ready_event), 0), "wait device video ready");
    const auto pixels = static_cast<std::size_t>(input.width) * input.height;
    const auto blocks = static_cast<unsigned>(std::min<std::size_t>((pixels + 255) / 256, 4096));
    if (sample == 2) convert<true><<<blocks, 256, 0, stream>>>(input.planes[0], input.planes[1], input.pitches[0], input.pitches[1], input.width, input.height, c, output);
    else convert<false><<<blocks, 256, 0, stream>>>(input.planes[0], input.planes[1], input.pitches[0], input.pitches[1], input.width, input.height, c, output);
    cu(cudaGetLastError(), "convert device video color");
}

void video_color_cuda_load_kernels() {
    cudaFuncAttributes attributes{};
    cu(cudaFuncGetAttributes(&attributes, convert<false>), "load NV12 conversion");
    cu(cudaFuncGetAttributes(&attributes, convert<true>), "load P010 conversion");
}
} // namespace saccade

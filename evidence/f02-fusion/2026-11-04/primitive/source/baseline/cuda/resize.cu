#include "saccade/resize_cuda.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifndef SACCADE_BUILD_COMMIT
#define SACCADE_BUILD_COMMIT "unknown"
#endif
#ifndef SACCADE_BUILD_DIRTY
#define SACCADE_BUILD_DIRTY "unknown"
#endif

namespace saccade {
namespace {
constexpr unsigned kThreads = 256;

void ck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

std::size_t count(int a, int b) {
    const auto n = std::size_t{3} * static_cast<unsigned>(a) * static_cast<unsigned>(b);
    if (n > static_cast<std::size_t>(INT_MAX))
        throw std::invalid_argument("resize tensor exceeds INT_MAX elements");
    return n;
}
void shape_check(const ResizeShape& shape) {
    for (const int n : {shape.input_width, shape.input_height, shape.output_width, shape.output_height})
        if (n < 1 || n > 65535) throw std::invalid_argument("resize dimension outside [1,65535]");
    (void)count(shape.input_width, shape.input_height);
    (void)count(shape.output_width, shape.output_height);
    (void)count(shape.output_width, shape.input_height);
}

struct Span { int begin, size; float offset, invscale; };

// Coordinate, edge truncation and FP32 weight construction follow the CUDA
// antialias operator used by T01 (PyTorch v2.11.0 UpSample.cuh). Preserve its
// expression structure, including nvcc's contraction across the center temporary.
__device__ Span span(int index, int input_size, float scale) {
    const float center = scale * (static_cast<float>(index) + 0.5f);
    const float support = fmaxf(scale, 1.0f);
    const int first = max(static_cast<int>(center - support + 0.5f), 0);
    const int last = min(static_cast<int>(center + support + 0.5f), input_size);
    return {first, last - first, static_cast<float>(first) - center,
            scale >= 1.0f ? __fdiv_rn(1.0f, scale) : 1.0f};
}
__device__ float weight(const Span& s, int j) {
    const float x = fabsf(__fmul_rn(__fadd_rn(__fadd_rn(static_cast<float>(j), s.offset), 0.5f), s.invscale));
    return x < 1.0f ? __fsub_rn(1.0f, x) : 0.0f;
}
__device__ float weight_total(const Span& s) {
    float total = 0.0f;
    for (int j = 0; j < s.size; ++j) total = __fadd_rn(total, weight(s, j));
    return total;
}

__global__ void horizontal(const std::uint8_t* input, float* scratch,
                           int width, int height, int output_width, float scale,
                           std::size_t elements) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += std::size_t{blockDim.x} * gridDim.x) {
        const int x = static_cast<int>(index % output_width);
        const std::size_t row = index / output_width;
        const Span s = span(x, width, scale);
        const float total = weight_total(s);
        float sum = 0.0f;
        for (int j = 0; j < s.size; ++j) {
            // CUDA scalar division uses a rounded reciprocal, as in T01.
            const float value = __fmul_rn(static_cast<float>(input[row * width + s.begin + j]), 1.0f / 255.0f);
            const float w = __fdiv_rn(weight(s, j), total);
            sum = j == 0 ? __fmul_rn(value, w) : __fmaf_rn(value, w, sum);
        }
        scratch[index] = sum;
    }
    (void)height;
}
__global__ void vertical(const float* scratch, float* output,
                         int height, int output_width, int output_height,
                         float scale, std::size_t elements) {
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < elements; index += std::size_t{blockDim.x} * gridDim.x) {
        const int x = static_cast<int>(index % output_width);
        const int y = static_cast<int>((index / output_width) % output_height);
        const std::size_t channel = index / (std::size_t{static_cast<unsigned>(output_width)} * output_height);
        const Span s = span(y, height, scale);
        const float total = weight_total(s);
        float sum = 0.0f;
        for (int j = 0; j < s.size; ++j) {
            const float value = scratch[(channel * height + s.begin + j) * output_width + x];
            const float w = __fdiv_rn(weight(s, j), total);
            sum = j == 0 ? __fmul_rn(value, w) : __fmaf_rn(value, w, sum);
        }
        output[index] = sum;
    }
}

void device_pointer(const void* p, int device, const char* name) {
    if (p == nullptr) throw std::invalid_argument(std::string("null resize ") + name);
    cudaPointerAttributes attributes{};
    const auto result = cudaPointerGetAttributes(&attributes, p);
    if (result == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        throw std::invalid_argument(std::string("resize ") + name + " must be device memory");
    }
    ck(result, "cudaPointerGetAttributes");
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        throw std::invalid_argument(std::string("resize ") + name + " must be device memory on current device");
}
bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    if (a_bytes > std::numeric_limits<std::uintptr_t>::max() - x ||
        b_bytes > std::numeric_limits<std::uintptr_t>::max() - y)
        throw std::invalid_argument("resize pointer range overflow");
    return x < y + b_bytes && y < x + a_bytes;
}
unsigned blocks(std::size_t n) {
    return static_cast<unsigned>(std::min<std::size_t>((n + kThreads - 1) / kThreads, 4096));
}
} // namespace

std::size_t resize_input_bytes(const ResizeShape& shape) {
    shape_check(shape); return count(shape.input_width, shape.input_height);
}
std::size_t resize_output_elements(const ResizeShape& shape) {
    shape_check(shape); return count(shape.output_width, shape.output_height);
}
std::size_t resize_workspace_elements(const ResizeShape& shape) {
    shape_check(shape); return count(shape.output_width, shape.input_height);
}

void resize_rgb_chw_cuda(const ResizeShape& shape,
                         const std::uint8_t* input, std::size_t input_bytes,
                         float* output, std::size_t output_elements,
                         float* workspace, std::size_t workspace_elements,
                         cudaStream_t stream) {
    const auto ni = resize_input_bytes(shape), no = resize_output_elements(shape), nw = resize_workspace_elements(shape);
    if (input_bytes < ni || output_elements < no || workspace_elements < nw)
        throw std::invalid_argument("insufficient resize capacity");
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        throw std::invalid_argument("resize requires an explicit non-default stream");
    if (reinterpret_cast<std::uintptr_t>(output) % alignof(float) != 0 ||
        reinterpret_cast<std::uintptr_t>(workspace) % alignof(float) != 0)
        throw std::invalid_argument("unaligned resize FP32 pointer");
    if (overlaps(input, ni, output, no * sizeof(float)) || overlaps(input, ni, workspace, nw * sizeof(float)) ||
        overlaps(output, no * sizeof(float), workspace, nw * sizeof(float)))
        throw std::invalid_argument("overlapping resize buffers");
    int device = -1; ck(cudaGetDevice(&device), "cudaGetDevice");
    device_pointer(input, device, "input");
    device_pointer(output, device, "output");
    device_pointer(workspace, device, "workspace");
    unsigned flags = 0; ck(cudaStreamGetFlags(stream, &flags), "cudaStreamGetFlags");
    int stream_device = -1; ck(cudaStreamGetDevice(stream, &stream_device), "cudaStreamGetDevice");
    if (stream_device != device)
        throw std::invalid_argument("resize stream belongs to another device");
    const float sx = static_cast<float>(shape.input_width) / shape.output_width;
    const float sy = static_cast<float>(shape.input_height) / shape.output_height;
    horizontal<<<blocks(nw), kThreads, 0, stream>>>(input, workspace, shape.input_width,
        shape.input_height, shape.output_width, sx, nw);
    ck(cudaGetLastError(), "resize horizontal launch");
    vertical<<<blocks(no), kThreads, 0, stream>>>(workspace, output, shape.input_height,
        shape.output_width, shape.output_height, sy, no);
    ck(cudaGetLastError(), "resize vertical launch");
}

std::string resize_cuda_load_kernels() {
    std::ostringstream text;
    for (const auto kernel : {reinterpret_cast<const void*>(horizontal), reinterpret_cast<const void*>(vertical)}) {
        cudaFuncAttributes attributes{};
        ck(cudaFuncGetAttributes(&attributes, kernel), "cudaFuncGetAttributes(resize)");
        text << "registers=" << attributes.numRegs << " local_bytes=" << attributes.localSizeBytes
             << " shared_bytes=" << attributes.sharedSizeBytes << '\n';
    }
    return text.str();
}
std::string resize_cuda_identity() {
    std::ostringstream text;
    text << "contract=t01-input/1\nresize=fp32-antialiased-bilinear-half-pixel\n"
         << "source_commit=" << SACCADE_BUILD_COMMIT << "\nsource_dirty=" << SACCADE_BUILD_DIRTY
         << "\nnvcc=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
         << "\nnormalize=fp32-reciprocal-255-before-resize\n";
    return text.str();
}
} // namespace saccade

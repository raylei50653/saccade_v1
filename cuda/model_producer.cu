#include "saccade/model_producer_cuda.hpp"
#include "saccade/scan_frame_kernels.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
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
struct Workspace { ScanStatus input, output; };
struct Inputs {
    const float* tensors[7];
    std::size_t counts[7];
    ScanStatus* plugins[kModelProducerPlugins];
};
constexpr unsigned kThreads = 256;

void ck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

__global__ void initialize(PostprocessProducerStatus* producer, Workspace* work) {
    *producer = PostprocessProducerStatus{};
    *work = Workspace{};
}

__global__ void finite(Inputs inputs, Workspace* work) {
    const unsigned tensor = blockIdx.y;
    bool bad = false;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < inputs.counts[tensor]; i += gridDim.x * blockDim.x)
        bad = bad || !(fabsf(inputs.tensors[tensor][i]) <= FLT_MAX);
    if (bad) {
        ScanStatus* status = tensor == 0 ? &work->input : &work->output;
        atomicOr(&status->flags, tensor == 0 ? kFrameNonFiniteInput : kFrameNonFiniteOutput);
    }
}

__global__ void aggregate(Inputs inputs, PostprocessProducerStatus* producer, Workspace* work) {
    work->input.done = work->output.done = kScanDone;
    std::uint32_t flags = 0;
    // Inspect every source, including after the first failure. Retiring on this
    // stream makes an omitted subsequent plugin write a protocol failure.
    for (unsigned i = 0; i < kModelProducerPlugins + 2; ++i) {
        ScanStatus* source = i < kModelProducerPlugins ? inputs.plugins[i] :
                             i == kModelProducerPlugins ? &work->input : &work->output;
        const std::uint32_t allowed = i < kModelProducerPlugins ?
            (i % 3 == 1 ? kScanInputFailures | kScanRangeFailures : kCrossScanNonFiniteInput) :
            i == kModelProducerPlugins ? kFrameNonFiniteInput : kFrameNonFiniteOutput;
        if (source->flags != 0) flags |= kPostprocessNumeric;
        if (source->done != kScanDone || (source->flags & ~allowed) != 0)
            flags |= kPostprocessProtocol;
        *source = ScanStatus{};
    }
    producer->flags = flags;
    // Completion is meaningful to consumers ordered after this kernel on a
    // stream/event edge; this is not a concurrent polling protocol.
    producer->done = kPostprocessProducerDone;
}

struct Region { const void* pointer; std::size_t bytes, alignment; };
void require_pointer(const Region& region, int device) {
    const auto start = reinterpret_cast<std::uintptr_t>(region.pointer);
    if (region.pointer == nullptr || start % region.alignment != 0 ||
        region.bytes > std::numeric_limits<std::uintptr_t>::max() - start)
        throw std::invalid_argument("null, unaligned or overflowing model producer pointer");
    cudaPointerAttributes attributes{};
    const auto error = cudaPointerGetAttributes(&attributes, region.pointer);
    if (error == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        throw std::invalid_argument("model producer pointer must be device memory");
    }
    ck(error, "cudaPointerGetAttributes(model producer)");
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        throw std::invalid_argument("model producer pointer must be on the current CUDA device");
    CUdeviceptr base = 0;
    std::size_t bytes = 0;
    const auto range_error = cuMemGetAddressRange(&base, &bytes, static_cast<CUdeviceptr>(start));
    if (range_error != CUDA_SUCCESS) {
        if (range_error == CUDA_ERROR_INVALID_VALUE || range_error == CUDA_ERROR_NOT_FOUND ||
            range_error == CUDA_ERROR_NOT_SUPPORTED)
            throw std::invalid_argument("model producer requires a queryable device allocation extent");
        const char* name = nullptr;
        (void)cuGetErrorName(range_error, &name);
        throw std::runtime_error(std::string("cuMemGetAddressRange(model producer): ") +
                                 (name == nullptr ? "unknown CUDA driver error" : name));
    }
    const auto first = static_cast<std::uintptr_t>(base);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - first ||
        start < first || start > first + bytes || region.bytes > first + bytes - start)
        throw std::invalid_argument("model producer range exceeds its device allocation");
}
bool overlaps(const Region& a, const Region& b) {
    const auto first = reinterpret_cast<std::uintptr_t>(a.pointer);
    const auto second = reinterpret_cast<std::uintptr_t>(b.pointer);
    return first < second + b.bytes && second < first + a.bytes;
}
} // namespace

std::size_t model_producer_workspace_bytes() { return sizeof(Workspace); }

void model_producer_cuda(const float* images, std::size_t image_elements,
                         const PostprocessRawView& raw, const ModelProducerPlugins& plugins,
                         PostprocessProducerStatus* producer, void* workspace,
                         std::size_t workspace_bytes, cudaStream_t stream) {
    if (image_elements != kModelProducerInputElements || workspace_bytes < sizeof(Workspace))
        throw std::invalid_argument("incorrect model producer input/workspace capacity");
    // Share the fixed six-output profile validation, without introducing an
    // ImageMapping dependency into this adapter's actual device work.
    validate_postprocess_arguments(raw, PostprocessRequest{});
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        throw std::invalid_argument("model producer requires an explicit non-default stream");
    int device = -1;
    ck(cudaGetDevice(&device), "cudaGetDevice(model producer)");
    unsigned stream_flags = 0;
    ck(cudaStreamGetFlags(stream, &stream_flags), "cudaStreamGetFlags(model producer)");
    int stream_device = -1;
    ck(cudaStreamGetDevice(stream, &stream_device), "cudaStreamGetDevice(model producer)");
    if (stream_device != device)
        throw std::invalid_argument("model producer stream belongs to another CUDA device");
    Region regions[18]{};
    Inputs inputs{};
    inputs.tensors[0] = images; inputs.counts[0] = image_elements;
    for (unsigned i = 0; i < 3; ++i) {
        inputs.tensors[i + 1] = raw.cls[i]; inputs.counts[i + 1] = raw.cls_elements[i];
        inputs.tensors[i + 4] = raw.reg[i]; inputs.counts[i + 4] = raw.reg_elements[i];
    }
    for (unsigned i = 0; i < 7; ++i)
        regions[i] = {inputs.tensors[i], inputs.counts[i] * sizeof(float), alignof(float)};
    for (unsigned i = 0; i < kModelProducerPlugins; ++i) {
        inputs.plugins[i] = plugins[i];
        regions[i + 7] = {plugins[i], sizeof(ScanStatus), alignof(ScanStatus)};
    }
    regions[16] = {producer, sizeof(PostprocessProducerStatus), alignof(PostprocessProducerStatus)};
    regions[17] = {workspace, sizeof(Workspace), alignof(ScanStatus)};
    // There are 18 accessed ranges; unused supplied workspace capacity is not
    // accessed and need not be disjoint from any later arena subrange.
    for (unsigned i = 0; i < 18; ++i) require_pointer(regions[i], device);
    for (unsigned i = 0; i < 18; ++i) for (unsigned j = i + 1; j < 18; ++j)
        if (overlaps(regions[i], regions[j]))
            throw std::invalid_argument("overlapping model producer buffers");
    auto* work = static_cast<Workspace*>(workspace);
    initialize<<<1, 1, 0, stream>>>(producer, work);
    ck(cudaGetLastError(), "model producer initialize launch");
    finite<<<dim3(256, 7), kThreads, 0, stream>>>(inputs, work);
    ck(cudaGetLastError(), "model producer finite launch");
    aggregate<<<1, 1, 0, stream>>>(inputs, producer, work);
    ck(cudaGetLastError(), "model producer aggregate launch");
}

std::string model_producer_cuda_load_kernels() {
    std::ostringstream out;
    for (const auto& kernel : {std::pair{"initialize", reinterpret_cast<const void*>(initialize)},
                              std::pair{"finite", reinterpret_cast<const void*>(finite)},
                              std::pair{"aggregate", reinterpret_cast<const void*>(aggregate)}}) {
        cudaFuncAttributes attr{};
        ck(cudaFuncGetAttributes(&attr, kernel.second), "cudaFuncGetAttributes(model producer)");
        out << kernel.first << " registers=" << attr.numRegs << " local=" << attr.localSizeBytes
            << " shared=" << attr.sharedSizeBytes << " max_threads=" << attr.maxThreadsPerBlock << '\n';
    }
    return out.str();
}

std::string model_producer_cuda_identity() {
    int runtime = 0, driver = 0, device = -1;
    ck(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion(model producer)");
    ck(cudaDriverGetVersion(&driver), "cudaDriverGetVersion(model producer)");
    ck(cudaGetDevice(&device), "cudaGetDevice(model producer)");
    cudaDeviceProp gpu{};
    ck(cudaGetDeviceProperties(&gpu, device), "cudaGetDeviceProperties(model producer)");
    std::ostringstream out;
    out << "policy=" << kModelProducerPolicy << "\nsource_commit=" << SACCADE_BUILD_COMMIT
        << "\nsource_dirty=" << SACCADE_BUILD_DIRTY
        << "\nnvcc=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
        << "\ncuda_flags=--ftz=false --fmad=false -lineinfo\nruntime=" << runtime << "\ndriver=" << driver
        << "\ngpu=" << gpu.name << "\nsm=" << gpu.major << gpu.minor
        << "\nworkspace_bytes=" << sizeof(Workspace) << "\nplugin_statuses=" << kModelProducerPlugins << '\n';
    return out.str();
}
} // namespace saccade

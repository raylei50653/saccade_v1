#include "saccade/postprocess_cuda.hpp"

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
#ifndef SACCADE_POSTPROCESS_CUDA_FLAGS
#define SACCADE_POSTPROCESS_CUDA_FLAGS "unknown"
#endif
#define SACCADE_PP_STRINGIFY_DETAIL(...) #__VA_ARGS__
#define SACCADE_PP_STRINGIFY(...) SACCADE_PP_STRINGIFY_DETAIL(__VA_ARGS__)

namespace saccade {
namespace {
constexpr unsigned kThreads = 128;

// Positions are indexed by the original model ID; top_ids is the stable,
// class-independent score order. No atomically assigned output row exists.
struct Decoded {
    float x1, y1, x2, y2, score;
    std::uint32_t class_id;
};
struct Workspace {
    std::uint32_t run_stage, numeric, qualified;
    Decoded positions[kPostprocessPositions];
    std::uint32_t top_ids[kPostprocessCapacity];
};
struct KernelInputs { const float* cls[3]; const float* reg[3]; };

void ck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

__device__ bool finite32(float x) { return fabsf(x) <= FLT_MAX; }

// std::array is deliberately confined to host descriptors. The row ABI is
// fixed by postprocess.hpp's assertions and has no inter-field padding.
__device__ DetectionCandidate* rows(DetectionCandidates* output) {
    return reinterpret_cast<DetectionCandidate*>(
        reinterpret_cast<unsigned char*>(output) + sizeof(DetectionCandidatesHeader));
}

__device__ void reset(DetectionCandidates* output, const PostprocessRequest& request,
                       std::uint32_t flags) {
    auto* bytes = reinterpret_cast<unsigned char*>(output);
    for (unsigned i = 0; i < sizeof(DetectionCandidates); ++i) bytes[i] = 0;
    output->header.epoch = request.epoch;
    output->header.frame = request.frame;
    output->header.schema = kPostprocessSchema;
    output->header.capacity = request.capacity;
    output->header.flags = flags;
}

__global__ void initialize(const PostprocessProducerStatus* producer,
                           DetectionCandidates* output, Workspace* work,
                           PostprocessRequest request) {
    std::uint32_t flags = 0;
    if (producer->flags != 0) flags |= kPostprocessUpstream;
    if (producer->done != kPostprocessProducerDone) flags |= kPostprocessProtocol;
    reset(output, request, flags);
    work->run_stage = flags == 0 ? 1u : 0u;
    work->numeric = 0;
    work->qualified = 0;
    for (unsigned i = 0; i < kPostprocessCapacity; ++i) work->top_ids[i] = 0;
}

__device__ float sigmoid(float x) {
    const float e = expf(-fabsf(x));
    return __fdiv_rn(x >= 0.0f ? 1.0f : e, __fadd_rn(1.0f, e));
}

__global__ void decode(KernelInputs input, Workspace* work) {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= kPostprocessPositions || work->run_stage == 0) return;
    unsigned level = 0, local = id, side = 80, stride = 8;
    if (id >= 8000) { level = 2; local -= 8000; side = 20; stride = 32; }
    else if (id >= 6400) { level = 1; local -= 6400; side = 40; stride = 16; }
    const unsigned plane = side * side;
    bool bad = false;
    float best_score = -1.0f;
    unsigned best_class = 0;
    for (unsigned c = 0; c < kPostprocessClasses; ++c) {
        const float logit = input.cls[level][c * plane + local];
        if (!finite32(logit)) bad = true;
        // All logits are still visited when one is invalid. A bad value has
        // no score/order meaning: the whole frame is rejected after this pass.
        const float score = finite32(logit) ? sigmoid(logit) : 0.0f;
        if (score > best_score) { best_score = score; best_class = c; }
    }
    float reg[4];
    for (unsigned c = 0; c < 4; ++c) {
        reg[c] = input.reg[level][c * plane + local];
        if (!finite32(reg[c])) bad = true;
    }
    const float ax = __fadd_rn(static_cast<float>(local % side), 0.5f);
    const float ay = __fadd_rn(static_cast<float>(local / side), 0.5f);
    const float raw[4]{__fsub_rn(ax, reg[0]), __fsub_rn(ay, reg[1]),
                       __fadd_rn(ax, reg[2]), __fadd_rn(ay, reg[3])};
    Decoded candidate{};
    float box[4];
    for (unsigned c = 0; c < 4; ++c) {
        box[c] = __fmul_rn(raw[c], static_cast<float>(stride));
        if (!finite32(raw[c]) || !finite32(box[c])) bad = true;
    }
    candidate.x1 = box[0]; candidate.y1 = box[1];
    candidate.x2 = box[2]; candidate.y2 = box[3];
    candidate.score = best_score; candidate.class_id = best_class;
    work->positions[id] = candidate;
    if (bad) atomicOr(&work->numeric, 1u);
    if (best_score >= kPostprocessScoreFloor) atomicAdd(&work->qualified, 1u);
}

// Independent exact ranks are unique under (score descending, model ID
// ascending). There are only 8400 positions, so the bounded quadratic scan is
// simple, deterministic and uses four launches rather than a sorting network's
// many global phase boundaries. No performance claim is attached to F04 v1.
__global__ void rank_top(Workspace* work) {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= kPostprocessPositions || work->run_stage == 0 || work->numeric != 0) return;
    const float score = work->positions[id].score;
    if (score < kPostprocessScoreFloor) return;
    unsigned rank = 0;
    for (unsigned other = 0; other < kPostprocessPositions; ++other) {
        const float rival = work->positions[other].score;
        if (rival > score || (rival == score && other < id)) ++rank;
        if (rank >= kPostprocessCapacity) return;
    }
    work->top_ids[rank] = id;
}

__device__ bool box_area(const DetectionCandidate& box, float& area) {
    const float w = __fsub_rn(box.x2, box.x1), h = __fsub_rn(box.y2, box.y1);
    area = __fmul_rn(w, h);
    return finite32(w) && finite32(h) && finite32(area) && area > 0.0f;
}

__device__ bool iou(const DetectionCandidate& a, const DetectionCandidate& b,
                    float area_a, float area_b, float& value) {
    const float dx = __fsub_rn(fminf(a.x2, b.x2), fmaxf(a.x1, b.x1));
    const float dy = __fsub_rn(fminf(a.y2, b.y2), fmaxf(a.y1, b.y1));
    const float intersect = __fmul_rn(fmaxf(dx, 0.0f), fmaxf(dy, 0.0f));
    const float sum = __fadd_rn(area_a, area_b);
    const float united = __fsub_rn(sum, intersect);
    value = __fdiv_rn(intersect, united);
    return finite32(dx) && finite32(dy) && finite32(intersect) && finite32(sum) &&
           finite32(united) && united > 0.0f && finite32(value);
}

__global__ void finalize(DetectionCandidates* output, Workspace* work,
                         PostprocessRequest request) {
    if (work->run_stage == 0) { output->header.done = kPostprocessDone; return; }
    if (work->numeric != 0) {
        reset(output, request, kPostprocessNumeric);
        output->header.done = kPostprocessDone;
        return;
    }
    auto& header = output->header;
    auto* result = rows(output);
    header.low_score = kPostprocessPositions - work->qualified;
    header.topk_dropped = work->qualified > kPostprocessCapacity
                       ? work->qualified - kPostprocessCapacity : 0u;
    const unsigned selected = work->qualified < kPostprocessCapacity
                            ? work->qualified : kPostprocessCapacity;
    unsigned count = 0;
    for (unsigned i = 0; i < selected; ++i) {
        const unsigned id = work->top_ids[i];
        const auto& candidate = work->positions[id];
        if (candidate.class_id != 0) { ++header.class_dropped; continue; }
        if (candidate.x2 <= candidate.x1 || candidate.y2 <= candidate.y1) {
            ++header.degenerate; continue;
        }
        result[count] = {candidate.x1, candidate.y1, candidate.x2, candidate.y2,
                         candidate.score, candidate.class_id, id, 0};
        ++count;
    }
    if (count > request.capacity) {
        reset(output, request, kPostprocessOverflow);
        output->header.done = kPostprocessDone;
        return;
    }
    float areas[kPostprocessCapacity];
    bool bad = false;
    for (unsigned i = 0; i < count; ++i) {
        if (!box_area(result[i], areas[i])) { bad = true; break; }
        result[i].flags = kDetectionKept;
        for (unsigned j = 0; j < i; ++j) {
            if (result[j].flags == 0) continue;
            float overlap = 0.0f;
            if (!iou(result[i], result[j], areas[i], areas[j], overlap)) { bad = true; break; }
            if (overlap > kPostprocessNmsIou) {
                result[i].flags = kDetectionSuppressed;
                break;
            }
        }
        if (bad) break;
    }
    const float sx = static_cast<float>(request.mapping.scale_x);
    const float sy = static_cast<float>(request.mapping.scale_y);
    if (!bad) for (unsigned i = 0; i < count; ++i) {
        auto& candidate = result[i];
        candidate.x1 = __fdiv_rn(candidate.x1, sx);
        candidate.y1 = __fdiv_rn(candidate.y1, sy);
        candidate.x2 = __fdiv_rn(candidate.x2, sx);
        candidate.y2 = __fdiv_rn(candidate.y2, sy);
        if (!finite32(candidate.x1) || !finite32(candidate.y1) ||
            !finite32(candidate.x2) || !finite32(candidate.y2)) { bad = true; break; }
        if (candidate.flags == kDetectionKept) ++header.kept;
        else ++header.suppressed;
    }
    if (bad) reset(output, request, kPostprocessNumeric);
    else header.count = count;
    output->header.done = kPostprocessDone;
}

struct Region { const void* pointer; std::size_t bytes, alignment; };

void require_pointer(const Region& region, int device) {
    if (region.pointer == nullptr || reinterpret_cast<std::uintptr_t>(region.pointer) % region.alignment != 0)
        throw std::invalid_argument("null or unaligned postprocess pointer");
    const auto start = reinterpret_cast<std::uintptr_t>(region.pointer);
    if (region.bytes > std::numeric_limits<std::uintptr_t>::max() - start)
        throw std::invalid_argument("postprocess pointer range overflow");
    cudaPointerAttributes attributes{};
    const auto error = cudaPointerGetAttributes(&attributes, region.pointer);
    if (error == cudaErrorInvalidValue) {
        (void)cudaGetLastError();
        throw std::invalid_argument("postprocess pointer must be device memory");
    }
    ck(error, "cudaPointerGetAttributes(postprocess)");
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        throw std::invalid_argument("postprocess pointer must be device memory on the current device");
    CUdeviceptr allocation_base = 0;
    std::size_t allocation_bytes = 0;
    const auto range_error = cuMemGetAddressRange(&allocation_base, &allocation_bytes,
                                                static_cast<CUdeviceptr>(start));
    if (range_error != CUDA_SUCCESS) {
        if (range_error == CUDA_ERROR_INVALID_VALUE || range_error == CUDA_ERROR_NOT_FOUND ||
            range_error == CUDA_ERROR_NOT_SUPPORTED)
            throw std::invalid_argument("postprocess requires a device allocation with a queryable extent (cudaMalloc profile)");
        const char* name = nullptr;
        (void)cuGetErrorName(range_error, &name);
        throw std::runtime_error(std::string("cuMemGetAddressRange(postprocess): ") +
                                 (name == nullptr ? "unknown CUDA driver error" : name));
    }
    const auto base = static_cast<std::uintptr_t>(allocation_base);
    if (allocation_bytes > std::numeric_limits<std::uintptr_t>::max() - base)
        throw std::invalid_argument("postprocess allocation range overflow");
    const auto end = base + allocation_bytes;
    if (start < base || start > end || region.bytes > end - start)
        throw std::invalid_argument("postprocess accessed range exceeds its device allocation");
}

bool overlaps(const Region& a, const Region& b) {
    const auto first = reinterpret_cast<std::uintptr_t>(a.pointer);
    const auto second = reinterpret_cast<std::uintptr_t>(b.pointer);
    return first < second + b.bytes && second < first + a.bytes;
}
} // namespace

std::size_t postprocess_workspace_bytes() { return sizeof(Workspace); }

void postprocess_cuda(const PostprocessRawView& raw, const PostprocessRequest& request,
                      const PostprocessProducerStatus* producer,
                      DetectionCandidates* output, std::size_t output_bytes,
                      void* workspace, std::size_t workspace_bytes,
                      cudaStream_t stream) {
    validate_postprocess_arguments(raw, request);
    if (output_bytes < sizeof(DetectionCandidates) || workspace_bytes < sizeof(Workspace))
        throw std::invalid_argument("insufficient postprocess output/workspace capacity");
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        throw std::invalid_argument("postprocess requires an explicit non-default stream");
    int device = -1;
    ck(cudaGetDevice(&device), "cudaGetDevice(postprocess)");
    Region regions[9]{};
    for (unsigned level = 0; level < 3; ++level) {
        regions[2 * level] = {raw.cls[level], raw.cls_elements[level] * sizeof(float), alignof(float)};
        regions[2 * level + 1] = {raw.reg[level], raw.reg_elements[level] * sizeof(float), alignof(float)};
    }
    regions[6] = {producer, sizeof(PostprocessProducerStatus), alignof(PostprocessProducerStatus)};
    regions[7] = {output, sizeof(DetectionCandidates), alignof(DetectionCandidates)};
    regions[8] = {workspace, sizeof(Workspace), alignof(Workspace)};
    for (const auto& region : regions) require_pointer(region, device);
    for (unsigned i = 0; i < 9; ++i) for (unsigned j = i + 1; j < 9; ++j)
        if (overlaps(regions[i], regions[j]))
            throw std::invalid_argument("overlapping postprocess buffers");
    unsigned flags = 0;
    ck(cudaStreamGetFlags(stream, &flags), "cudaStreamGetFlags(postprocess)");
    int stream_device = -1;
    ck(cudaStreamGetDevice(stream, &stream_device), "cudaStreamGetDevice(postprocess)");
    if (stream_device != device)
        throw std::invalid_argument("postprocess stream belongs to another CUDA device");
    KernelInputs inputs{};
    for (unsigned level = 0; level < 3; ++level) {
        inputs.cls[level] = raw.cls[level]; inputs.reg[level] = raw.reg[level];
    }
    auto* work = static_cast<Workspace*>(workspace);
    initialize<<<1, 1, 0, stream>>>(producer, output, work, request);
    ck(cudaGetLastError(), "postprocess initialize launch");
    constexpr unsigned blocks = (kPostprocessPositions + kThreads - 1) / kThreads;
    decode<<<blocks, kThreads, 0, stream>>>(inputs, work);
    ck(cudaGetLastError(), "postprocess decode launch");
    rank_top<<<blocks, kThreads, 0, stream>>>(work);
    ck(cudaGetLastError(), "postprocess rank launch");
    finalize<<<1, 1, 0, stream>>>(output, work, request);
    ck(cudaGetLastError(), "postprocess finalize launch");
}

std::string postprocess_cuda_load_kernels() {
    std::ostringstream text;
    const void* kernels[]{reinterpret_cast<const void*>(initialize), reinterpret_cast<const void*>(decode),
                          reinterpret_cast<const void*>(rank_top), reinterpret_cast<const void*>(finalize)};
    for (const auto kernel : kernels) {
        cudaFuncAttributes attributes{};
        ck(cudaFuncGetAttributes(&attributes, kernel), "cudaFuncGetAttributes(postprocess)");
        text << "registers=" << attributes.numRegs << " local_bytes=" << attributes.localSizeBytes
             << " shared_bytes=" << attributes.sharedSizeBytes << '\n';
    }
    return text.str();
}

std::string postprocess_cuda_identity() {
    int device = -1, runtime = 0, driver = 0;
    cudaDeviceProp properties{};
    ck(cudaGetDevice(&device), "cudaGetDevice(postprocess identity)");
    ck(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties(postprocess identity)");
    ck(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion(postprocess identity)");
    ck(cudaDriverGetVersion(&driver), "cudaDriverGetVersion(postprocess identity)");
    std::ostringstream text;
    text << "policy=" << kPostprocessPolicy << "\nsource_commit=" << SACCADE_BUILD_COMMIT
         << "\nsource_dirty=" << SACCADE_BUILD_DIRTY << "\nnvcc=" << __CUDACC_VER_MAJOR__
         << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
         << "\ncuda_flags=" << SACCADE_POSTPROCESS_CUDA_FLAGS
#ifdef __CUDA_ARCH_LIST__
         << "\ncuda_arch_list=" << SACCADE_PP_STRINGIFY(__CUDA_ARCH_LIST__)
#endif
         << "\nhost_compiler=" << __VERSION__
         << "\ncuda_runtime=" << runtime << "\ncuda_driver_api=" << driver
         << "\ngpu=" << properties.name
         << "\ncompute_capability=" << properties.major << '.' << properties.minor
         << "\ndevice_ordinal=" << device
         << "\nworkspace_bytes=" << sizeof(Workspace)
         << "\nordering=score-descending-model-id-ascending\n";
    return text.str();
}
} // namespace saccade

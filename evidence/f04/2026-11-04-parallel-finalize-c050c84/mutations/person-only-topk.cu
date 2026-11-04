#include "saccade/postprocess_cuda.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/version.cuh>
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
constexpr unsigned kSortThreads = 256;
constexpr unsigned kRankThreshold = 100;
constexpr unsigned kRadixThreshold = 512;
constexpr unsigned kDenseTileItems = 1024;
constexpr unsigned kMaxPositions = kHrModelProfile.positions();
constexpr unsigned kDenseTiles = (kMaxPositions + kDenseTileItems - 1) / kDenseTileItems;
constexpr unsigned kCandidateIdBits = 15;
constexpr unsigned kSortKeyBits = 32 + kCandidateIdBits;
constexpr std::uint64_t kInvalidKey = ~std::uint64_t{0};
constexpr std::uint64_t kCandidateIdMask = (std::uint64_t{1} << kCandidateIdBits) - 1;
static_assert(kMaxPositions <= (1u << kCandidateIdBits));

// Positions are indexed by the original model ID; top_ids is the stable,
// class-independent score order. No atomically assigned output row exists.
struct Decoded {
    float x1, y1, x2, y2, score;
    std::uint32_t class_id;
};
struct Workspace {
    std::uint32_t run_stage, numeric, qualified;
    Decoded positions[kMaxPositions];
    std::uint32_t top_ids[kPostprocessCapacity];
    std::uint64_t full_keys[kMaxPositions];
    std::uint64_t compact_keys[kRadixThreshold];
    std::uint64_t partial_keys[kDenseTiles * kPostprocessCapacity];
};
struct KernelInputs {
    const float* cls[3]; const float* reg[3];
    unsigned widths[3], planes[3], positions, tiles;
};

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
    if (id >= input.positions || work->run_stage == 0) return;
    unsigned level = 0, local = id;
    while (local >= input.planes[level]) { local -= input.planes[level]; ++level; }
    const unsigned width = input.widths[level], plane = input.planes[level], stride = 8u << level;
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
    const float ax = __fadd_rn(static_cast<float>(local % width), 0.5f);
    const float ay = __fadd_rn(static_cast<float>(local / width), 0.5f);
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
    const bool qualified = best_score >= kPostprocessScoreFloor && best_class == 0;
    // Probabilities are finite and nonnegative after sigmoid. Their FP32 bit
    // order is numeric order, so this ascending, unique 47-bit key preserves
    // score descending / original model ID ascending without quantization.
    const std::uint64_t key = qualified
        ? (std::uint64_t{~__float_as_uint(best_score)} << kCandidateIdBits) | id
        : kInvalidKey;
    work->full_keys[id] = key;
    if (qualified) {
        // Arrival order only places keys in scratch; the complete unique key
        // determines the published order. Dense input uses full_keys instead
        // of an incomplete compact prefix, so no candidate is lost.
        const unsigned compact_index = atomicAdd(&work->qualified, 1u);
        if (compact_index < kRadixThreshold) work->compact_keys[compact_index] = key;
    }
}

using MediumSort = cub::BlockRadixSort<std::uint64_t, kSortThreads, kRadixThreshold / kSortThreads>;
using DenseSort = cub::BlockRadixSort<std::uint64_t, kSortThreads, kDenseTileItems / kSortThreads>;
union AdaptiveSortShared {
    typename MediumSort::TempStorage medium;
    typename DenseSort::TempStorage dense;
    std::uint64_t small[kRankThreshold];
};

// Every gate/branch containing a collective is uniform within a CTA. M and
// numeric are complete before this kernel starts; only decode writes them.
// The launch shape stays fixed, including graph replay, without host counts.
__global__ void adaptive_prefix(Workspace* work, KernelInputs input) {
    const unsigned count = work->qualified;
    if (work->run_stage == 0 || work->numeric != 0 || count == 0) return;
    __shared__ AdaptiveSortShared shared;
    if (count <= kRankThreshold) {
        if (blockIdx.x != 0) return;
        if (threadIdx.x < count) shared.small[threadIdx.x] = work->compact_keys[threadIdx.x];
        __syncthreads();
        if (threadIdx.x < count) {
            const std::uint64_t key = shared.small[threadIdx.x];
            unsigned rank = 0;
            for (unsigned j = 0; j < count; ++j) rank += shared.small[j] < key;
            work->top_ids[rank] = static_cast<std::uint32_t>(key & kCandidateIdMask);
        }
        return;
    }
    if (count <= kRadixThreshold) {
        if (blockIdx.x != 0) return;
        std::uint64_t keys[kRadixThreshold / kSortThreads];
#pragma unroll
        for (unsigned i = 0; i < kRadixThreshold / kSortThreads; ++i) {
            const unsigned index = threadIdx.x + i * kSortThreads;
            keys[i] = index < count ? work->compact_keys[index] : kInvalidKey;
        }
        MediumSort(shared.medium).SortBlockedToStriped(keys, 0, kSortKeyBits);
#pragma unroll
        for (unsigned i = 0; i < kRadixThreshold / kSortThreads; ++i) {
            const unsigned rank = threadIdx.x + i * kSortThreads;
            if (rank < count && rank < kPostprocessCapacity)
                work->top_ids[rank] = static_cast<std::uint32_t>(keys[i] & kCandidateIdMask);
        }
        return;
    }
    std::uint64_t keys[kDenseTileItems / kSortThreads];
#pragma unroll
    for (unsigned i = 0; i < kDenseTileItems / kSortThreads; ++i) {
        const unsigned index = blockIdx.x * kDenseTileItems + threadIdx.x + i * kSortThreads;
        keys[i] = index < input.positions ? work->full_keys[index] : kInvalidKey;
    }
    DenseSort(shared.dense).SortBlockedToStriped(keys, 0, kSortKeyBits);
#pragma unroll
    for (unsigned i = 0; i < kDenseTileItems / kSortThreads; ++i) {
        const unsigned rank = threadIdx.x + i * kSortThreads;
        if (rank < kPostprocessCapacity)
            work->partial_keys[blockIdx.x * kPostprocessCapacity + rank] = keys[i];
    }
}

// A member of the global top-300 must occur in its tile's local top-300.
// For such a member, every other tile contributes its exact count of smaller
// keys; a truncated count >=300 can only reject a nonmember. Unique valid keys
// make output ranks distinct even when scores tie. Invalid sentinels are never
// assigned a rank, avoiding duplicate writes in sparse/partly filled tiles.
__global__ void dense_merge(Workspace* work, unsigned tiles) {
    if (work->run_stage == 0 || work->numeric != 0 || work->qualified <= kRadixThreshold) return;
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tiles * kPostprocessCapacity) return;
    const std::uint64_t key = work->partial_keys[index];
    if (key == kInvalidKey) return;
    const unsigned own_tile = index / kPostprocessCapacity;
    unsigned rank = index % kPostprocessCapacity;
    for (unsigned tile = 0; tile < tiles; ++tile) {
        if (tile == own_tile) continue;
        unsigned begin = 0, end = kPostprocessCapacity;
        const auto* list = work->partial_keys + tile * kPostprocessCapacity;
        while (begin != end) {
            const unsigned middle = (begin + end) / 2;
            if (list[middle] < key) begin = middle + 1;
            else end = middle;
        }
        rank += begin;
        if (rank >= kPostprocessCapacity) return;
    }
    work->top_ids[rank] = static_cast<std::uint32_t>(key & kCandidateIdMask);
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

constexpr unsigned kNmsWords = (kPostprocessCapacity + 63) / 64;
using FinalizeScan = cub::BlockScan<unsigned, kSortThreads>;
struct FinalizeShared {
    typename FinalizeScan::TempStorage scan;
    DetectionCandidate pool[kPostprocessCapacity];
    float areas[kPostprocessCapacity];
    std::uint64_t fail[kPostprocessCapacity][kNmsWords];
    std::uint64_t over[kPostprocessCapacity][kNmsWords];
    std::uint64_t kept[kNmsWords];
    unsigned count, bad;
};
static_assert(sizeof(FinalizeShared) <= 48 * 1024);

__global__ void finalize(DetectionCandidates* output, Workspace* work,
                         PostprocessRequest request, unsigned positions) {
    if (work->run_stage == 0) {
        __syncthreads();
        if (threadIdx.x == 0) output->header.done = kPostprocessDone;
        return;
    }
    if (work->numeric != 0) {
        if (threadIdx.x == 0) reset(output, request, kPostprocessNumeric);
        __syncthreads();
        if (threadIdx.x == 0) output->header.done = kPostprocessDone;
        return;
    }
    __shared__ FinalizeShared shared;
    auto& header = output->header;
    if (threadIdx.x == 0) {
        shared.bad = 0;
        header.low_score = positions - work->qualified;
        header.topk_dropped = work->qualified > kPostprocessCapacity
                           ? work->qualified - kPostprocessCapacity : 0u;
    }
    const unsigned selected = work->qualified < kPostprocessCapacity
                            ? work->qualified : kPostprocessCapacity;
    constexpr unsigned items = (kPostprocessCapacity + kSortThreads - 1) / kSortThreads;
    unsigned accepted[items]{}, offsets[items], total;
    unsigned class_dropped = 0, degenerate = 0;
#pragma unroll
    for (unsigned item = 0; item < items; ++item) {
        const unsigned i = threadIdx.x * items + item;
        if (i >= selected) continue;
        const auto& candidate = work->positions[work->top_ids[i]];
        if (candidate.class_id != 0) ++class_dropped;
        else if (candidate.x2 <= candidate.x1 || candidate.y2 <= candidate.y1) ++degenerate;
        else accepted[item] = 1;
    }
    // Blocked item order is the original score/ID order, including the tail.
    FinalizeScan(shared.scan).ExclusiveSum(accepted, offsets, total);
    if (threadIdx.x == 0) shared.count = total;
    if (class_dropped != 0) atomicAdd(&header.class_dropped, class_dropped);
    if (degenerate != 0) atomicAdd(&header.degenerate, degenerate);
#pragma unroll
    for (unsigned item = 0; item < items; ++item) {
        if (accepted[item] == 0) continue;
        const unsigned id = work->top_ids[threadIdx.x * items + item];
        const auto& candidate = work->positions[id];
        shared.pool[offsets[item]] = {candidate.x1, candidate.y1, candidate.x2, candidate.y2,
                                      candidate.score, candidate.class_id, id, 0};
    }
    __syncthreads();
    if (shared.count > request.capacity) {
        if (threadIdx.x == 0) reset(output, request, kPostprocessOverflow);
        __syncthreads();
        if (threadIdx.x == 0) output->header.done = kPostprocessDone;
        return;
    }
    for (unsigned i = threadIdx.x; i < shared.count; i += blockDim.x)
        if (!box_area(shared.pool[i], shared.areas[i])) atomicExch(&shared.bad, 1u);
    __syncthreads();
    if (shared.bad == 0) {
        // One thread owns each word, including its zero bits beyond j<i.
        // A speculative failure is only data; the ordered greedy pass decides
        // whether that pair would actually have been visited by scalar NMS.
        for (unsigned index = threadIdx.x; index < shared.count * kNmsWords; index += blockDim.x) {
            const unsigned i = index / kNmsWords, word = index % kNmsWords;
            std::uint64_t failures = 0, overlaps = 0;
            for (unsigned j = word * 64; j < i && j < (word + 1) * 64; ++j) {
                float overlap = 0.0f;
                const auto bit = std::uint64_t{1} << (j % 64);
                if (!iou(shared.pool[i], shared.pool[j], shared.areas[i], shared.areas[j], overlap))
                    failures |= bit;
                else if (overlap > kPostprocessNmsIou) overlaps |= bit;
            }
            shared.fail[i][word] = failures;
            shared.over[i][word] = overlaps;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (unsigned word = 0; word < kNmsWords; ++word) shared.kept[word] = 0;
            for (unsigned i = 0; i < shared.count; ++i) {
                shared.pool[i].flags = kDetectionKept;
                for (unsigned word = 0; word < kNmsWords; ++word) {
                    const std::uint64_t failures = shared.fail[i][word];
                    const auto decisive = (failures | shared.over[i][word]) & shared.kept[word];
                    if (decisive == 0) continue;
                    const auto first = decisive & (~decisive + 1);
                    if ((failures & first) != 0) shared.bad = 1;
                    else shared.pool[i].flags = kDetectionSuppressed;
                    break;
                }
                if (shared.bad != 0) break;
                if (shared.pool[i].flags == kDetectionKept)
                    shared.kept[i / 64] |= std::uint64_t{1} << (i % 64);
            }
        }
    }
    __syncthreads();
    const float sx = static_cast<float>(request.mapping.scale_x);
    const float sy = static_cast<float>(request.mapping.scale_y);
    unsigned mapping_bad = 0;
    if (shared.bad == 0) {
        for (unsigned i = threadIdx.x; i < shared.count; i += blockDim.x) {
            auto& candidate = shared.pool[i];
            candidate.x1 = __fdiv_rn(candidate.x1, sx);
            candidate.y1 = __fdiv_rn(candidate.y1, sy);
            candidate.x2 = __fdiv_rn(candidate.x2, sx);
            candidate.y2 = __fdiv_rn(candidate.y2, sy);
            if (!finite32(candidate.x1) || !finite32(candidate.y1) ||
                !finite32(candidate.x2) || !finite32(candidate.y2)) mapping_bad = 1;
        }
    }
    __syncthreads();
    if (mapping_bad != 0) atomicExch(&shared.bad, 1u);
    __syncthreads();
    if (shared.bad != 0) {
        if (threadIdx.x == 0) reset(output, request, kPostprocessNumeric);
    } else {
        auto* result = rows(output);
        for (unsigned i = threadIdx.x; i < shared.count; i += blockDim.x) {
            const auto candidate = shared.pool[i];
            result[i] = candidate;
            if (candidate.flags == kDetectionKept) atomicAdd(&header.kept, 1u);
            else atomicAdd(&header.suppressed, 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (shared.bad == 0) header.count = shared.count;
        output->header.done = kPostprocessDone;
    }
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
    const auto& spec = model_profile(request.profile);
    inputs.positions = spec.positions();
    inputs.tiles = (inputs.positions + kDenseTileItems - 1) / kDenseTileItems;
    for (unsigned level = 0; level < 3; ++level) {
        inputs.cls[level] = raw.cls[level]; inputs.reg[level] = raw.reg[level];
        inputs.widths[level] = static_cast<unsigned>(spec.level_width(level));
        inputs.planes[level] = spec.level_positions(level);
    }
    auto* work = static_cast<Workspace*>(workspace);
    initialize<<<1, 1, 0, stream>>>(producer, output, work, request);
    ck(cudaGetLastError(), "postprocess initialize launch");
    const unsigned blocks = (inputs.positions + kThreads - 1) / kThreads;
    decode<<<blocks, kThreads, 0, stream>>>(inputs, work);
    ck(cudaGetLastError(), "postprocess decode launch");
    adaptive_prefix<<<inputs.tiles, kSortThreads, 0, stream>>>(work, inputs);
    ck(cudaGetLastError(), "postprocess adaptive prefix launch");
    const unsigned merge_blocks = (inputs.tiles * kPostprocessCapacity + kThreads - 1) / kThreads;
    dense_merge<<<merge_blocks, kThreads, 0, stream>>>(work, inputs.tiles);
    ck(cudaGetLastError(), "postprocess dense merge launch");
    finalize<<<1, kSortThreads, 0, stream>>>(output, work, request, inputs.positions);
    ck(cudaGetLastError(), "postprocess finalize launch");
}

std::string postprocess_cuda_load_kernels() {
    std::ostringstream text;
    const void* kernels[]{reinterpret_cast<const void*>(initialize), reinterpret_cast<const void*>(decode),
                          reinterpret_cast<const void*>(adaptive_prefix), reinterpret_cast<const void*>(dense_merge),
                          reinterpret_cast<const void*>(finalize)};
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
         << "\ncub_version=" << CUB_VERSION
         << "\ncccl_version=" << CCCL_VERSION
         << "\nsort_strategy=decode-compaction/adaptive-top300-v1"
         << "\nfinalize_strategy=cta-bitmask-greedy-v1"
         << "\nsort_rank_threshold=" << kRankThreshold
         << "\nsort_radix_threshold=" << kRadixThreshold
         << "\nsort_dense_tile_items=" << kDenseTileItems
         << "\nsort_max_dense_tiles=" << kDenseTiles
         << "\nmodel_profiles=t01-640x640,hr-1280x768"
         << "\nsort_key_bits=" << kSortKeyBits
         << "\nsort_candidate_id_bits=" << kCandidateIdBits
         << "\nsort_stage_kernels=2\nkernels_per_enqueue=5"
         << "\nordering=score-descending-model-id-ascending\n";
    return text.str();
}
} // namespace saccade

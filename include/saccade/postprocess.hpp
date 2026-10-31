#pragma once

#include "saccade/image_mapping.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace saccade {

// F04's fixed FP32, batch-one profile (docs/POSTPROCESS_POLICY.md).
inline constexpr const char* kPostprocessPolicy = "f04-postprocess-fp32/1";
inline constexpr std::uint32_t kPostprocessSchema = 1;
inline constexpr std::uint32_t kPostprocessCapacity = 300;
inline constexpr std::uint32_t kPostprocessPositions = 8400;
inline constexpr std::uint32_t kPostprocessClasses = 80;
inline constexpr std::array<std::uint32_t, 3> kPostprocessSizes{80, 40, 20};
inline constexpr std::array<std::uint32_t, 3> kPostprocessStrides{8, 16, 32};
inline constexpr float kPostprocessScoreFloor = 0.001f;
inline constexpr float kPostprocessNmsIou = 0.5f;
inline constexpr std::uint32_t kPostprocessProducerDone = 0x34525046u; // "FPR4"
inline constexpr std::uint32_t kPostprocessDone = 0x34504F46u; // "FOP4"

enum PostprocessFailure : std::uint32_t {
    kPostprocessNumeric = 1u << 0,
    kPostprocessUpstream = 1u << 1,
    kPostprocessProtocol = 1u << 2,
    kPostprocessOverflow = 1u << 3,
};
enum DetectionCandidateFlag : std::uint32_t {
    kDetectionKept = 1u,
    kDetectionSuppressed = 2u,
};

// Caller-owned aggregate producer status. The future F03 adapter must gate all
// plugin and finite statuses, write flags and only then done on the producing
// stream. F04 borrows it read-only; lifetime/freshness belong to the caller.
struct PostprocessProducerStatus {
    std::uint32_t flags = 0, done = 0;
};

// NCHW FP32, one pointer/count per level. Counts must match exactly:
// cls: 80*size*size, reg: 4*size*size. The reference reads host memory; the
// CUDA entrypoint reads device memory with this same pointer-only descriptor.
struct PostprocessRawView {
    std::array<const float*, 3> cls{};
    std::array<const float*, 3> reg{};
    std::array<std::size_t, 3> cls_elements{};
    std::array<std::size_t, 3> reg_elements{};
};
struct PostprocessRequest {
    std::uint64_t epoch = 0, frame = 0;
    ImageMapping mapping{640, 640, 640, 640, 1.0, 1.0};
    std::uint32_t capacity = kPostprocessCapacity;
};

// Internal device ABI, not ResultPacket wire encoding. Explicit fields and
// static assertions leave no padding. Every call initializes every byte.
struct DetectionCandidate {
    float x1, y1, x2, y2, score;
    std::uint32_t class_id, candidate_id, flags;
};
struct DetectionCandidatesHeader {
    std::uint64_t epoch, frame;
    std::uint32_t schema, count, capacity, flags, done, kept, suppressed;
    std::uint32_t low_score, topk_dropped, class_dropped, degenerate, reserved;
};
struct DetectionCandidates {
    DetectionCandidatesHeader header;
    std::array<DetectionCandidate, kPostprocessCapacity> rows;
};
static_assert(sizeof(DetectionCandidate) == 32);
static_assert(sizeof(DetectionCandidatesHeader) == 64);
static_assert(sizeof(DetectionCandidates) == 64 + 32 * kPostprocessCapacity);
static_assert(std::is_trivially_copyable_v<DetectionCandidates>);

// Validates the fixed profile and mappings (dimensions [1,65535], model 640,
// exact binary64 scales 640/original_dimension). Does not dereference data.
// Host argument violations throw invalid_argument before work starts.
void validate_postprocess_arguments(const PostprocessRawView& raw, const PostprocessRequest& request);

// Independent scalar FP32 reference: std::exp(float), separate FP32 arithmetic
// with contraction disabled. A valid empty frame has count=0 and flags=0;
// numeric/upstream/protocol/overflow frames have count/statistics=0, zero rows
// and the corresponding flags. Identity/capacity/schema and done are retained.
DetectionCandidates postprocess_reference(const PostprocessRawView& raw,
                                           const PostprocessRequest& request,
                                           const PostprocessProducerStatus& producer);

} // namespace saccade

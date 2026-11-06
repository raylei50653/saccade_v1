#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

// F05/F06 contract, profile v1-motion/1 (docs/TRACKING.md §4-§5). Line A owns
// the implementation and may extend this header within its own scope.
namespace saccade {

inline constexpr const char* kMotionProfile = "v1-motion/1";

struct MotionConfig {
    int downscale = 4;               // integer original -> MotionImage factor
    float pcr_threshold = 5.0f;      // PCR at which confidence reaches 1
    float max_shift_fraction = 0.25f; // |peak| above floor(fraction * axis) is rejected
};
// invalid_argument unless downscale >= 1, pcr_threshold finite > 0 and
// max_shift_fraction finite in (0, 0.5].
void validate_motion_config(const MotionConfig& config);

// Representation key; fixed for a sequence (a change fails the sequence).
struct MotionImageKey {
    int original_width = 0, original_height = 0, downscale = 0, width = 0, height = 0;
    bool operator==(const MotionImageKey&) const = default;
};
// width = floor(W/d), height = floor(H/d); invalid_argument if either is < 8.
MotionImageKey motion_image_key(int original_width, int original_height, const MotionConfig& config);

enum class MotionStatus : std::uint32_t { disabled = 0, uninitialized = 1, valid = 2, rejected = 3 };

// Reference frame -> current frame translation in original pixels. tx/ty are
// zero unless status is valid. reference_frame is -1 when uninitialized.
struct MotionContext {
    std::uint64_t epoch = 0;
    std::int64_t frame = -1, reference_frame = -1;
    MotionStatus status = MotionStatus::uninitialized;
    std::int32_t peak_x = 0, peak_y = 0; // wrapped integer peak on the MotionImage grid
    float pcr = 0, confidence = 0;
    float tx = 0, ty = 0;
};

// Hanning table of length n: float(0.5 * (1 - cos(2*pi*i/(n-1)))) in binary64.
// CUDA uploads this same table, so F05 matches bit for bit. n must be >= 2.
std::vector<float> hanning_window(int n);

// F05 CPU reference. planar_rgb holds R, G and B planes of W*H bytes each
// (NativePreprocessor's PreprocessedFrame::rgb layout). Returns the windowed
// grayscale MotionImage, row-major width*height (docs/TRACKING.md §4).
std::vector<float> motion_image_reference(std::span<const std::uint8_t> planar_rgb, const MotionImageKey& key);

// F06 CPU reference with a binary64 separable direct DFT. An empty previous
// image yields uninitialized; otherwise valid or rejected (docs/TRACKING.md §5).
// Non-finite intermediate values throw std::domain_error (NUMERIC).
MotionContext phase_correlation_reference(std::span<const float> previous, std::span<const float> current,
                                          const MotionImageKey& key, const MotionConfig& config,
                                          std::uint64_t epoch, std::int64_t frame);

// Per-sequence MotionContext stream written by tools/motion_sequence. Explicit
// little-endian encoding: "SCMO", u32 version=1, five i32 key fields, u64 epoch, u64 count,
// then per frame i64 frame, i64 reference_frame, u32 status, i32 peak_x,
// i32 peak_y, f32 pcr, f32 confidence, f32 tx, f32 ty. The reader rejects any
// other magic/version, truncation, trailing bytes or unknown status. The record
// epoch is written once and assigned to every frame on read. Frames are zero-based.
struct MotionRecords {
    MotionImageKey key;
    std::uint64_t epoch = 0;
    std::vector<MotionContext> frames;
};
void write_motion_records(const std::filesystem::path& path, const MotionRecords& records);
MotionRecords read_motion_records(const std::filesystem::path& path);
// Decodes the exact immutable bytes that callers can hash in the run receipt.
MotionRecords decode_motion_records(std::string_view bytes);

} // namespace saccade

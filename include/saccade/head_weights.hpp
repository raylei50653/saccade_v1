#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace saccade {

// Parameters of the T=1 YOLO26m Mamba-FPN head (docs/HEAD_TRT.md §2,
// docs/SCAN_CONTRACT.md §2) and their native bundle. Three levels P3/P4/P5,
// d_model 128, d_inner 256, d_state 16, one spatial block per level, spatial
// reduction 4, Cross-Scan, PixelShuffle 4, cls 80, reg 4 (reg_max 1).
inline constexpr int kHeadLevels = 3;
inline constexpr std::array<std::int64_t, 3> kHeadInChannels{256, 512, 512};
inline constexpr std::array<std::int64_t, 3> kHeadSize{80, 40, 20}; // H = W at 640 input
inline constexpr std::array<const char*, 3> kHeadLevelNames{"p3", "p4", "p5"};
inline constexpr std::int64_t kHeadDModel = 128, kHeadDInner = 256, kHeadDState = 16, kHeadCRank = 1, kHeadReduction = 4,
                              kHeadConvKernel = 4, kHeadClasses = 80, kHeadRegs = 4;
inline constexpr const char* kHeadSchema = "yolo26m-mamba-fpn-head-t1/1";
inline constexpr const char* kHeadBundleMagic = "SCWB 1";

// One active parameter: bundle name, shape and layout, and the checkpoint key
// it comes from (relative to the checkpoint's `student` state dict).
struct HeadParam {
    std::string name, layout, source_key;
    std::vector<std::int64_t> shape;
};
// The 69 parameters of the T=1 forward, in bundle order (level-major).
const std::vector<HeadParam>& head_parameter_schema();
// The 39 checkpoint keys the T=1 forward never reads, with the reason.
const std::vector<std::pair<std::string, std::string>>& head_excluded_keys();

struct HeadTensor {
    std::vector<std::int64_t> shape;
    std::vector<float> values;
};

struct HeadWeights {
    std::map<std::string, HeadTensor, std::less<>> tensors; // by bundle name
    std::string schema = kHeadSchema, source_sha256, source_archive;
    std::int64_t source_epoch = -1;
    const HeadTensor& at(std::string_view name) const; // throws when absent
};

// Exactly the schema's names and shapes, every value finite. Throws
// std::invalid_argument naming the first violation.
void validate_head_weights(const HeadWeights& weights);

struct HeadConversion {
    HeadWeights weights;
    std::string bundle;  // encode_head_bundle(weights)
    std::string mapping; // TSV: every checkpoint key, its storage view and its mapping or exclusion
};

// Converts a torch.save checkpoint of the head without running its code: checks
// the file SHA-256, the archive and pickle structure, the head configuration
// (mamba_args and the shapes it implies), every active tensor's dtype, shape,
// contiguity and storage range, and that every value is finite. Rejects missing,
// unknown or duplicate keys; never fills, guesses or substitutes a parameter.
HeadConversion convert_head_checkpoint(std::string checkpoint, std::string_view expected_sha256);

// Bundle: canonical ASCII header (magic, schema, provenance, one line per tensor
// with dtype, shape, data offset, bytes and SHA-256, "end"), zero padding to a
// 64-byte boundary, then the little-endian FP32 data of each tensor at 64-byte
// aligned offsets with zero padding. Decoding re-encodes and requires identical
// bytes, so non-canonical, truncated or padded bundles are rejected.
std::string encode_head_bundle(const HeadWeights& weights);
HeadWeights decode_head_bundle(std::string_view bytes);

} // namespace saccade

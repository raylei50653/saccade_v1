#pragma once

#include "saccade/selective_scan_cuda.hpp"

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace saccade {

// S03 minimal engines: strongly typed FP32 networks whose only layers are the
// scan plugins (docs/SCAN_TRT.md §3). Every engine is built with TF32 off,
// builder optimization level 0, no auxiliary streams, a 16 MiB builder
// workspace and no external timing cache. Status outputs are ordinary INT32[2]
// execution outputs named after their plugin.
inline constexpr std::size_t kScanTrtBuilderWorkspace = 16u << 20;

// One optimization profile for SelectiveScan with dynamic batch, length,
// channels, state, A rows and C width: min, opt and max per dimension.
struct ScanProfile {
    std::int64_t batch[3] = {1, 4, 8};
    std::int64_t length[3] = {0, 100, 2048};
    std::int64_t channels[3] = {1, 256, 512};
    std::int64_t state[3] = {1, 16, 32};
};

// Inputs u, delta, A, B, C[, D] -> outputs "y", "scan_status".
std::vector<std::uint8_t> build_selective_scan_engine(nvinfer1::ILogger& logger, bool has_d, const ScanProfile& profile = {});
// Input x (B,C,H,W), dynamic up to 8 x 512 x 64 x 64 -> "tokens", "expand_status".
std::vector<std::uint8_t> build_cross_scan_expand_engine(nvinfer1::ILogger& logger);
// Input y (4B, height*width, C), dynamic B <= 8, C <= 512 -> "out", "merge_status".
std::vector<std::uint8_t> build_cross_scan_merge_engine(nvinfer1::ILogger& logger, std::int32_t height, std::int32_t width);
// Static Cross-Scan around one scan, as in the head with C = Dc:
// x (B,C,H,W), delta (4B,HW,C), A (1,N), B (4B,HW,N), C (4B,HW,1)[, D (C)]
// -> "out" (B,C,H,W), "expand_status", "scan_status", "merge_status".
std::vector<std::uint8_t> build_cross_scan_pipeline_engine(nvinfer1::ILogger& logger, const CrossScanShape& shape, std::int32_t state,
                                                           bool has_d = true);

// Engine I/O summary, one tensor per line: name, mode, dtype, format, location,
// shape and whether it is shape-inference I/O.
std::string describe_scan_engine(const nvinfer1::ICudaEngine& engine);

} // namespace saccade

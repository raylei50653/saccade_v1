#pragma once
#include "saccade/motion.hpp"
#include <algorithm>
#include <stdexcept>

namespace saccade {
inline constexpr const char* kFlowMotionProfile = "v1-motion/2";
// TRACKING §20.2: fixed profile, with no alternative session settings.
struct FlowConfig {
    static constexpr int reference_width = 1920, reference_height = 1080;
    static constexpr int grid = 4, cell = 8, min_cells = 8;
    static constexpr float mask_score = 0.45f;
    static constexpr const char* perf = "SLOW";
};
inline int flow_scale(int width, int height) {
    return std::max({1, (width + FlowConfig::reference_width - 1) / FlowConfig::reference_width,
                       (height + FlowConfig::reference_height - 1) / FlowConfig::reference_height});
}
inline MotionImageKey flow_image_key(int width, int height) {
    const int s = flow_scale(width, height);
    if (width / s < 32 || height / s < 32) throw std::invalid_argument("OFA image dimensions must be at least 32");
    return {width, height, s, width / s, height / s};
}
}

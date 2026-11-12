#pragma once

namespace saccade {

// Continuous image-edge coordinates: model_x = original_x * scale_x.
// The resize filter itself uses half-pixel sampling (align_corners=false).
struct ImageMapping {
    int original_width, original_height, model_width, model_height;
    double scale_x, scale_y;
};

} // namespace saccade

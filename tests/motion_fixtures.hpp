#pragma once

#include "saccade/motion.hpp"
#include <cstdint>
#include <vector>

namespace motion_test {
inline std::vector<float> texture(const saccade::MotionImageKey& key) {
    std::vector<float> image(static_cast<std::size_t>(key.width) * key.height);
    std::uint32_t state = 42;
    for (auto& v : image) {
        state = state * 1664525u + 1013904223u;
        v = float(state >> 8) / 16777216.0f;
    }
    return image;
}
inline std::vector<float> shifted(const std::vector<float>& image, const saccade::MotionImageKey& key, int dx, int dy) {
    std::vector<float> output(image.size());
    for (int y = 0; y < key.height; ++y)
        for (int x = 0; x < key.width; ++x)
            output[static_cast<std::size_t>((y + dy + key.height) % key.height) * key.width + (x + dx + key.width) % key.width] =
                image[static_cast<std::size_t>(y) * key.width + x];
    return output;
}
inline std::vector<std::uint8_t> rgb(const saccade::MotionImageKey& key, std::uint32_t seed = 123) {
    std::vector<std::uint8_t> image(static_cast<std::size_t>(key.original_width) * key.original_height * 3);
    for (auto& v : image) { seed = seed * 1664525u + 1013904223u; v = seed >> 24; }
    return image;
}
} // namespace motion_test

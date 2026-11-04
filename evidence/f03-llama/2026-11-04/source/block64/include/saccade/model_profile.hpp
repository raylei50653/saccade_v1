#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace saccade {

enum class ModelProfile { T01_640x640, Hr1280x768 };
inline constexpr ModelProfile kDefaultModelProfile = ModelProfile::T01_640x640;

struct ModelProfileSpec {
    std::string_view name;
    std::int64_t width, height;
    static constexpr std::array<std::uint32_t, 3> strides{8, 16, 32};
    static constexpr std::int64_t reduction = 4;
    constexpr std::int64_t level_height(std::size_t level) const { return height / strides[level]; }
    constexpr std::int64_t level_width(std::size_t level) const { return width / strides[level]; }
    constexpr std::uint32_t level_positions(std::size_t level) const {
        return static_cast<std::uint32_t>(level_height(level) * level_width(level));
    }
    constexpr std::uint32_t level_offset(std::size_t level) const {
        std::uint32_t offset = 0;
        for (std::size_t i = 0; i < level; ++i) offset += level_positions(i);
        return offset;
    }
    constexpr std::uint32_t positions() const { return level_offset(3); }
    constexpr std::int64_t scan_length(std::size_t level) const {
        return (level_height(level) / reduction) * (level_width(level) / reduction);
    }
    constexpr std::size_t input_elements() const { return static_cast<std::size_t>(3 * height * width); }
    constexpr std::size_t raw_elements() const { return std::size_t{84} * positions(); }
};

inline constexpr ModelProfileSpec kT01ModelProfile{"t01-640x640", 640, 640};
inline constexpr ModelProfileSpec kHrModelProfile{"hr-1280x768", 1280, 768};
constexpr const ModelProfileSpec& model_profile(ModelProfile profile = kDefaultModelProfile) {
    return profile == ModelProfile::Hr1280x768 ? kHrModelProfile : kT01ModelProfile;
}
inline ModelProfile parse_model_profile(std::string_view name) {
    if (name == kT01ModelProfile.name) return ModelProfile::T01_640x640;
    if (name == kHrModelProfile.name) return ModelProfile::Hr1280x768;
    throw std::invalid_argument("unknown model profile (expected t01-640x640 or hr-1280x768)");
}

} // namespace saccade

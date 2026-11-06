#pragma once

#include "saccade/resolved_config.hpp"
#include "saccade/tracking_session.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <stdexcept>

namespace saccade {

inline void mark_native_input_origins(ResolvedTrackingConfig& resolved, std::string_view source_name) {
    for (auto& origin : resolved.origins) {
        if (origin.layer != "default") continue;
        if (origin.key == "source.width" || origin.key == "source.height") {
            origin.layer = "derived"; origin.name = source_name;
        } else if (origin.key == "preprocess.max_width" || origin.key == "preprocess.max_height") {
            origin.layer = "derived"; origin.name = "session-effective-max-dimensions";
        }
    }
}

// Resolve only source-dimension defaults for pre-open validation. The complete
// resolver still checks every layer, duplicate, type and dependency afterward.
inline ResolvedTrackingConfig native_preview_defaults(ResolvedTrackingConfig defaults,
    std::string_view file, const ConfigAssignments& environment, const ConfigAssignments& cli) {
    const auto apply = [&](const ConfigAssignments& assignments, bool is_environment) {
        for (const auto& [name, value] : assignments) {
            std::string key = name;
            if (is_environment) {
                key.clear();
                for (const auto& field : tracking_config_schema())
                    if (field.environment == name) { key = field.key; break; }
            }
            if (key != "source.width" && key != "source.height") continue;
            std::uint32_t dimension = 0;
            const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), dimension);
            if (error != std::errc{} || end != value.data() + value.size())
                throw std::invalid_argument("invalid source dimension override");
            (key == "source.width" ? defaults.source.width : defaults.source.height) = dimension;
        }
    };
    apply(parse_tracking_config_file(file), false);
    apply(environment, true); apply(cli, false);
    defaults.preprocess.max_width = std::max(1920u, defaults.source.width);
    defaults.preprocess.max_height = std::max(1080u, defaults.source.height);
    return defaults;
}

inline TrackingSessionConfig session_config_from_resolved(const ResolvedTrackingConfig& resolved) {
    validate_resolved_config(resolved, {{"cuda"}, {"jpeg", "host-rgb", "device-video"}});
    if (resolved.observer.event_dump)
        throw std::invalid_argument("CUDA full event dump is not provided; use the CPU reference event tool");
    if (resolved.implementation.motion_profile != kFlowMotionProfile)
        throw std::invalid_argument("CUDA session requires the provided v1-motion/2 implementation");
    if (resolved.source.width > static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
        resolved.source.height > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("source dimensions exceed the CUDA session representation");
    const auto width = static_cast<int>(resolved.source.width), height = static_cast<int>(resolved.source.height);
    if (resolved.preprocess.max_width != static_cast<std::uint32_t>(std::max(1920, width)) ||
        resolved.preprocess.max_height != static_cast<std::uint32_t>(std::max(1080, height)))
        throw std::invalid_argument("preprocess maxima must equal the session effective max(1920,width)/max(1080,height)");
    if (resolved.preprocess.max_jpeg_bytes > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("JPEG capacity exceeds the CUDA session representation");
    TrackingSessionConfig config;
    config.epoch = resolved.source.epoch;
    config.device = static_cast<int>(resolved.execution.device);
    config.original_width = width; config.original_height = height;
    config.candidate_slots = resolved.execution.candidate_slots;
    config.motion_slots = resolved.execution.motion_slots;
    config.result_slots = resolved.execution.result_slots;
    config.preprocess.max_width = static_cast<int>(resolved.preprocess.max_width);
    config.preprocess.max_height = static_cast<int>(resolved.preprocess.max_height);
    config.preprocess.max_jpeg_bytes = static_cast<std::size_t>(resolved.preprocess.max_jpeg_bytes);
    config.preprocess.input = resolved.implementation.input == "jpeg" ? PreprocessInput::HardwareJpeg :
        resolved.implementation.input == "host-rgb" ? PreprocessInput::HostRgb : PreprocessInput::DeviceVideo;
    config.profile = resolved.implementation.model_profile;
    config.preprocess.output_width = static_cast<int>(model_profile(config.profile).width);
    config.preprocess.output_height = static_cast<int>(model_profile(config.profile).height);
    config.tracker = resolved.tracker;
    config.mode = resolved.execution.mode == "serial" ? TrackingSessionMode::Serial : TrackingSessionMode::Pipelined;
    config.cuda_graph = resolved.execution.cuda_graph;
    config.perturb_seed = resolved.execution.perturb_seed;
    return config;
}

} // namespace saccade

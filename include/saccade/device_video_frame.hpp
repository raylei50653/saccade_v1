#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace saccade {
enum class VideoSurfaceFormat { Nv12, P010 };
enum class VideoColorMatrix { Bt601, Bt709, Bt2020 };

// Device-only view. Owner retains the AVFrame, its hardware frames context and
// readiness event. No host pixel buffer is part of this interface. Context,
// stream and ready_event are CUDA driver handles, kept opaque for CPU builds.
struct DeviceVideoFrame {
    std::array<const std::uint8_t*, 2> planes{};
    std::array<std::size_t, 2> pitches{};
    int width = 0, height = 0, device = 0;
    VideoSurfaceFormat format = VideoSurfaceFormat::Nv12;
    VideoColorMatrix matrix = VideoColorMatrix::Bt601;
    bool full_range = false;
    void* context = nullptr;
    void* stream = nullptr;
    void* ready_event = nullptr;
    std::shared_ptr<void> owner;
};
} // namespace saccade

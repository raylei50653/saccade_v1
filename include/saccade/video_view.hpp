#pragma once

#include "saccade/device_video_frame.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace saccade {
struct VideoViewConfig {
    std::string url;
    int device = 0;
    int timeout_ms = 5000;
};
struct ViewBox {
    std::uint32_t id;
    std::array<float, 4> xyxy; // original pixels, inclusive rounded edges
};

// Single host-thread, NV12-only GPU copy/overlay and h264_nvenc output. The
// decoder surface is read only. Missing or non-increasing PTS is fatal. close()
// checks encoder drain, muxer trailer and I/O; destructor cleanup is best effort.
// Diagnostics never contain the target. Callers control FFmpeg's global logger.
class VideoView {
public:
    VideoView(const VideoViewConfig&, int width, int height);
    ~VideoView();
    VideoView(const VideoView&) = delete;
    VideoView& operator=(const VideoView&) = delete;
    void write(const DeviceVideoFrame&, std::optional<std::int64_t> pts_us, std::span<const ViewBox>);
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace saccade

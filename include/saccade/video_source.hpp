#pragma once

#include "saccade/device_video_frame.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace saccade {

enum class VideoDecodeBackend { Cpu, Cuda };

struct VideoSourceConfig {
    std::string input_format; // empty: autodetect; e.g. v4l2, dshow, avfoundation
    std::string rtsp_transport = "tcp";
    int timeout_ms = 5000; // deadline for open/probe and each read
    int max_width = 8192, max_height = 8192;
    VideoDecodeBackend backend = VideoDecodeBackend::Cpu;
    int device = 0;
};

struct VideoFrame {
    std::vector<std::uint8_t> rgb; // contiguous R, G, B planes, original resolution
    int width = 0, height = 0;
    std::uint64_t index = 0; // decoded presentation order, contiguous and zero-based
    std::optional<std::int64_t> pts_us; // source time; may start away from zero
    DeviceVideoFrame device_frame; // CUDA backend: leased NV12/P010; rgb is empty
};

// FFmpeg CPU decode/color conversion or strict native NVDEC device surfaces.
// CUDA requires SACCADE_ENABLE_NVDEC and has no CPU fallback. Each device frame
// retains its surface, primary CUDA context, producer stream and readiness event
// until its last owner releases it, including after VideoSource::close().
// No JPEG encoding, background frame
// queue, implicit dropping, reconnect, scaling or rotation. Resolution is fixed
// by the first decoded frame. A fresh source/session is required after failure
// or a discontinuity. All operations except request_stop() use one host thread.
// HDR transfer functions and BT.2020 constant luminance are rejected; there is
// no tone mapping. Unspecified YUV matrix metadata uses FFmpeg's default BT.601.
// Open/probe is bounded by timeout_ms; request_stop is available after construction.
class VideoSource {
public:
    explicit VideoSource(const std::string& uri, const VideoSourceConfig& config = {});
    ~VideoSource();
    VideoSource(const VideoSource&) = delete;
    VideoSource& operator=(const VideoSource&) = delete;

    // false at clean EOF (after decoder drain), or after request_stop(). Read
    // errors/timeouts throw and latch failure. No URI appears in our diagnostics.
    bool read(VideoFrame& frame);
    void request_stop() noexcept; // thread safe, interrupts a blocked read
    // Repeatable checked producer drain; retained frames remain valid. Their
    // eventual destructor cleanup is best effort. No concurrent read/close.
    void close();
    std::string identity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace saccade

#include "saccade/video_source.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
#ifdef SACCADE_VIDEO_NVDEC
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#endif
}

#include <atomic>
#include <chrono>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace saccade {
namespace {
using Clock = std::chrono::steady_clock;
void initialize_ffmpeg() {
    static std::once_flag initialized;
    std::call_once(initialized, [] {
        if (avformat_network_init() < 0) throw std::runtime_error("FFmpeg network initialization failed");
        avdevice_register_all();
    });
}
std::int64_t now_ms() noexcept {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
std::string ffmpeg_error(int code) {
    char text[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(code, text, sizeof(text));
    return text;
}
int color_matrix(AVColorSpace colorspace) {
    switch (colorspace) {
    case AVCOL_SPC_BT709: return SWS_CS_ITU709;
    case AVCOL_SPC_FCC: return SWS_CS_FCC;
    case AVCOL_SPC_SMPTE240M: return SWS_CS_SMPTE240M;
    case AVCOL_SPC_BT2020_NCL: return SWS_CS_BT2020;
    default: return SWS_CS_DEFAULT;
    }
}
#ifdef SACCADE_VIDEO_NVDEC
void cu(CUresult code, const char* operation) {
    if (code == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorName(code, &message);
    throw std::runtime_error(std::string(operation) + ": " + (message ? message : "CUDA driver error"));
}
struct CurrentContext {
    explicit CurrentContext(CUcontext context) { cu(cuCtxPushCurrent(context), "activate video CUDA context"); }
    ~CurrentContext() { CUcontext ignored = nullptr; (void)cuCtxPopCurrent(&ignored); }
};
// Own the primary context independently of FFmpeg's CURRENT_CONTEXT wrapper.
// This interoperates with an already initialized CUDA runtime without changing
// its scheduling flags. Every exported frame retains this state, so the stream
// and primary context outlive both decoder close and delayed surface consumers.
struct VideoCudaContext {
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    AVBufferRef* device_ref = nullptr;
    explicit VideoCudaContext(int ordinal) {
        try {
            cu(cuInit(0), "initialize video CUDA driver");
            cu(cuDeviceGet(&device, ordinal), "select video CUDA device");
            cu(cuDevicePrimaryCtxRetain(&context, device), "retain video primary CUDA context");
            CurrentContext current(context);
            cu(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create video decode stream");
            const auto device_text = std::to_string(ordinal);
            const auto result = av_hwdevice_ctx_create(&device_ref, AV_HWDEVICE_TYPE_CUDA,
                device_text.c_str(), nullptr, AV_CUDA_USE_CURRENT_CONTEXT);
            if (result < 0) throw std::runtime_error("create FFmpeg CUDA device: " + ffmpeg_error(result));
            auto* hardware = static_cast<AVHWDeviceContext*>(static_cast<void*>(device_ref->data));
            auto* cuda = static_cast<AVCUDADeviceContext*>(hardware->hwctx);
            require(cuda->cuda_ctx == context, "FFmpeg CUDA context differs from retained primary context");
            cuda->stream = stream;
        } catch (...) { release(); throw; }
    }
    void release() noexcept {
        if (!context) return;
        CUcontext ignored = nullptr;
        const bool pushed = cuCtxPushCurrent(context) == CUDA_SUCCESS;
        if (pushed && stream) (void)cuStreamSynchronize(stream);
        av_buffer_unref(&device_ref);
        if (pushed && stream) (void)cuStreamDestroy(stream);
        if (pushed) (void)cuCtxPopCurrent(&ignored);
        (void)cuDevicePrimaryCtxRelease(device);
        context = nullptr; stream = nullptr;
    }
    ~VideoCudaContext() { release(); }
};
struct VideoSurfaceLease {
    std::shared_ptr<VideoCudaContext> cuda;
    AVFrame* frame = nullptr;
    CUevent ready = nullptr;
    explicit VideoSurfaceLease(std::shared_ptr<VideoCudaContext> context, const AVFrame* source)
        : cuda(std::move(context)) {
        try {
            frame = av_frame_clone(source);
            if (!frame) throw std::bad_alloc();
            CurrentContext current(cuda->context);
            cu(cuEventCreate(&ready, CU_EVENT_DISABLE_TIMING), "create video surface ready event");
            cu(cuEventRecord(ready, cuda->stream), "record video surface ready event");
        } catch (...) { release(); throw; }
    }
    void release() noexcept {
        CUcontext ignored = nullptr;
        const bool pushed = cuCtxPushCurrent(cuda->context) == CUDA_SUCCESS;
        if (pushed && ready) (void)cuEventSynchronize(ready);
        if (pushed && ready) (void)cuEventDestroy(ready);
        ready = nullptr;
        av_frame_free(&frame);
        if (pushed) (void)cuCtxPopCurrent(&ignored);
    }
    ~VideoSurfaceLease() { release(); }
};
AVPixelFormat cuda_pixel_format(AVCodecContext*, const AVPixelFormat* formats) {
    for (auto format = formats; *format != AV_PIX_FMT_NONE; ++format)
        if (*format == AV_PIX_FMT_CUDA) return *format;
    return AV_PIX_FMT_NONE; // never let get_format select a software fallback
}
VideoColorMatrix device_color_matrix(AVColorSpace colorspace) {
    switch (colorspace) {
    case AVCOL_SPC_UNSPECIFIED: case AVCOL_SPC_BT470BG: case AVCOL_SPC_SMPTE170M:
        return VideoColorMatrix::Bt601;
    case AVCOL_SPC_BT709: return VideoColorMatrix::Bt709;
    case AVCOL_SPC_BT2020_NCL: return VideoColorMatrix::Bt2020;
    default: throw std::invalid_argument("NVDEC video color matrix is unsupported");
    }
}
#endif
} // namespace

struct VideoSource::Impl {
    VideoSourceConfig config;
    const std::thread::id owner = std::this_thread::get_id();
    std::atomic<bool> stopped{false};
    std::int64_t deadline = 0; // read only by interrupt callback on owner thread
    AVFormatContext* format = nullptr;
    AVCodecContext* decoder = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* decoded = nullptr;
    SwsContext* conversion = nullptr;
    int stream = -1, width = 0, height = 0;
    std::uint64_t next_frame = 0;
    bool demux_eof = false, flushed = false, eof = false, failed = false, closed = false;
    std::string codec_name, format_name;
#ifdef SACCADE_VIDEO_NVDEC
    std::shared_ptr<VideoCudaContext> cuda_context;
#endif

    static int interrupt(void* opaque) noexcept {
        const auto& self = *static_cast<Impl*>(opaque);
        return self.stopped.load(std::memory_order_relaxed) ||
               (self.deadline && now_ms() >= self.deadline);
    }
    void check_owner() const {
        require(owner == std::this_thread::get_id(), "VideoSource requires its original host thread");
    }
    void check(int code, const char* operation) const {
        if (code >= 0) return;
        if (stopped.load(std::memory_order_relaxed)) throw std::runtime_error("video source stopped");
        if (deadline && now_ms() >= deadline) throw std::runtime_error(std::string(operation) + ": video source timeout");
        throw std::runtime_error(std::string(operation) + ": " + ffmpeg_error(code));
    }
    void dimensions(int w, int h) const {
        require(w > 0 && h > 0 && w <= config.max_width && h <= config.max_height,
                "video dimensions exceed configured capacity");
        require(width == 0 || (w == width && h == height), "video resolution changed; start a fresh tracking session");
    }
    explicit Impl(const std::string& uri, const VideoSourceConfig& cfg) : config(cfg) {
        require(!uri.empty(), "video source must not be empty");
        require(cfg.timeout_ms > 0 && cfg.timeout_ms <= 3600000, "video timeout_ms must be in 1..3600000");
        require(cfg.max_width > 0 && cfg.max_width <= 8192 && cfg.max_height > 0 && cfg.max_height <= 8192,
                "video maximum dimensions must be in 1..8192");
        require(cfg.rtsp_transport == "tcp" || cfg.rtsp_transport == "udp", "RTSP transport must be tcp or udp");
        require(cfg.device >= 0, "video CUDA device must be nonnegative");
        require(cfg.backend == VideoDecodeBackend::Cpu || cfg.backend == VideoDecodeBackend::Cuda,
                "unknown video decode backend");
#ifndef SACCADE_VIDEO_NVDEC
        require(cfg.backend == VideoDecodeBackend::Cpu, "CUDA video decoding requires SACCADE_ENABLE_NVDEC");
#endif
        initialize_ffmpeg();
        AVDictionary* options = nullptr;
        try {
            deadline = now_ms() + cfg.timeout_ms;
            const AVInputFormat* input = nullptr;
            if (!cfg.input_format.empty()) {
                input = av_find_input_format(cfg.input_format.c_str());
                require(input != nullptr, "requested video input format is unavailable");
            }
            format = avformat_alloc_context();
            if (!format) throw std::bad_alloc();
            format->interrupt_callback = {interrupt, this};
            format->probesize = 2 * 1024 * 1024;
            format->max_analyze_duration = static_cast<std::int64_t>(cfg.timeout_ms) * 1000;
            if (uri.starts_with("rtsp://") || uri.starts_with("rtsps://"))
                check(av_dict_set(&options, "rtsp_transport", cfg.rtsp_transport.c_str(), 0), "set RTSP transport");
            check(avformat_open_input(&format, uri.c_str(), input, &options), "open video input");
            av_dict_free(&options);
            check(avformat_find_stream_info(format, nullptr), "probe video input");
            const AVCodec* codec = nullptr;
            stream = av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
            check(stream, "find video stream");
            const auto* parameters = format->streams[stream]->codecpar;
            if (parameters->width || parameters->height) dimensions(parameters->width, parameters->height);
            decoder = avcodec_alloc_context3(codec);
            if (!decoder) throw std::bad_alloc();
            check(avcodec_parameters_to_context(decoder, parameters), "set video decoder parameters");
            decoder->max_pixels = static_cast<std::int64_t>(cfg.max_width) * cfg.max_height;
            decoder->thread_count = 2;
#ifdef SACCADE_VIDEO_NVDEC
            if (cfg.backend == VideoDecodeBackend::Cuda) {
                bool supported = false;
                for (int index = 0; const auto* hardware = avcodec_get_hw_config(codec, index); ++index) {
                    if (hardware->device_type == AV_HWDEVICE_TYPE_CUDA && hardware->pix_fmt == AV_PIX_FMT_CUDA &&
                        (hardware->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) { supported = true; break; }
                }
                require(supported, "video codec has no native CUDA/NVDEC device decoder; CPU fallback is disabled");
                cuda_context = std::make_shared<VideoCudaContext>(cfg.device);
                decoder->hw_device_ctx = av_buffer_ref(cuda_context->device_ref);
                if (!decoder->hw_device_ctx) throw std::bad_alloc();
                decoder->get_format = cuda_pixel_format;
                decoder->thread_count = 1;
            }
#endif
            check(avcodec_open2(decoder, codec, nullptr), "open video decoder");
            codec_name = codec->name;
            format_name = format->iformat->name;
            packet = av_packet_alloc(); decoded = av_frame_alloc();
            if (!packet || !decoded) throw std::bad_alloc();
            deadline = 0;
        } catch (...) {
            av_dict_free(&options);
            release();
            throw;
        }
    }
    void release() noexcept {
        if (closed) return;
        closed = true;
        stopped.store(true, std::memory_order_relaxed);
        av_packet_free(&packet);
        av_frame_free(&decoded);
        sws_freeContext(conversion); conversion = nullptr;
        avcodec_free_context(&decoder);
#ifdef SACCADE_VIDEO_NVDEC
        cuda_context.reset();
#endif
        avformat_close_input(&format);
    }
    ~Impl() { release(); }

    void convert(VideoFrame& output) {
        dimensions(decoded->width, decoded->height);
        require(!(decoded->flags & AV_FRAME_FLAG_CORRUPT), "video decoder produced a corrupt frame");
        require(decoded->color_trc != AVCOL_TRC_SMPTE2084 && decoded->color_trc != AVCOL_TRC_ARIB_STD_B67,
                "HDR video needs an explicit tone-mapping policy");
        require(decoded->colorspace != AVCOL_SPC_BT2020_CL,
                "BT.2020 constant-luminance video needs an explicit color conversion policy");
#ifdef SACCADE_VIDEO_NVDEC
        if (config.backend == VideoDecodeBackend::Cuda) {
            require(decoded->format == AV_PIX_FMT_CUDA && decoded->hw_frames_ctx,
                    "NVDEC returned a non-device frame; CPU fallback is disabled");
            const auto* frames = static_cast<const AVHWFramesContext*>(static_cast<const void*>(decoded->hw_frames_ctx->data));
            require(frames->sw_format == AV_PIX_FMT_NV12 || frames->sw_format == AV_PIX_FMT_P010LE,
                    "NVDEC output must be NV12 or P010");
            require(decoded->width % 2 == 0 && decoded->height % 2 == 0,
                    "NVDEC NV12/P010 dimensions must be even");
            const auto matrix = device_color_matrix(decoded->colorspace);
            require(decoded->color_range == AVCOL_RANGE_UNSPECIFIED || decoded->color_range == AVCOL_RANGE_MPEG ||
                    decoded->color_range == AVCOL_RANGE_JPEG, "NVDEC color range is unsupported");
            const auto row_bytes = decoded->width * (frames->sw_format == AV_PIX_FMT_NV12 ? 1 : 2);
            require(decoded->data[0] && decoded->data[1] && decoded->linesize[0] >= row_bytes &&
                    decoded->linesize[1] >= row_bytes, "NVDEC output planes/pitches are invalid");
            auto lease = std::make_shared<VideoSurfaceLease>(cuda_context, decoded);
            DeviceVideoFrame device_frame;
            device_frame.planes = {decoded->data[0], decoded->data[1]};
            device_frame.pitches = {static_cast<std::size_t>(decoded->linesize[0]), static_cast<std::size_t>(decoded->linesize[1])};
            device_frame.width = decoded->width; device_frame.height = decoded->height;
            device_frame.device = config.device;
            device_frame.format = frames->sw_format == AV_PIX_FMT_NV12 ? VideoSurfaceFormat::Nv12 : VideoSurfaceFormat::P010;
            device_frame.matrix = matrix; device_frame.full_range = decoded->color_range == AVCOL_RANGE_JPEG;
            device_frame.context = cuda_context->context; device_frame.stream = cuda_context->stream;
            device_frame.ready_event = lease->ready; device_frame.owner = std::move(lease);
            output.device_frame = std::move(device_frame);
            output.rgb.clear();
        } else
#endif
        {
        const auto pixel_format = static_cast<AVPixelFormat>(decoded->format);
        require(sws_isSupportedInput(pixel_format), "video decoder pixel format cannot be converted to RGB");
        conversion = sws_getCachedContext(conversion, decoded->width, decoded->height, pixel_format,
                                         decoded->width, decoded->height, AV_PIX_FMT_GBRP,
                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!conversion) throw std::runtime_error("create video RGB conversion");
        const auto* description = av_pix_fmt_desc_get(pixel_format);
        const bool rgb = description && (description->flags & AV_PIX_FMT_FLAG_RGB);
        const auto* coefficients = sws_getCoefficients(color_matrix(decoded->colorspace));
        check(sws_setColorspaceDetails(conversion, coefficients,
                                      rgb || decoded->color_range == AVCOL_RANGE_JPEG,
                                      coefficients, 1, 0, 1 << 16, 1 << 16), "configure video color conversion");
        const auto plane = static_cast<std::size_t>(decoded->width) * decoded->height;
        output.rgb.resize(plane * 3);
        // swscale's planar format is GBR; direct the planes into our RGB layout.
        std::uint8_t* destination[4]{output.rgb.data() + plane, output.rgb.data() + 2 * plane, output.rgb.data(), nullptr};
        int stride[4]{decoded->width, decoded->width, decoded->width, 0};
        const auto rows = sws_scale(conversion, decoded->data, decoded->linesize, 0, decoded->height, destination, stride);
        require(rows == decoded->height, "video RGB conversion returned an incomplete frame");
        output.device_frame = {};
        }
        width = output.width = decoded->width;
        height = output.height = decoded->height;
        output.index = next_frame++;
        const auto timestamp = decoded->best_effort_timestamp;
        output.pts_us = timestamp == AV_NOPTS_VALUE ? std::nullopt :
            std::optional<std::int64_t>(av_rescale_q(timestamp, format->streams[stream]->time_base, AV_TIME_BASE_Q));
    }
    bool read(VideoFrame& output) {
        check_owner();
        require(!closed, "video source is closed");
        require(!failed, "video source has failed");
        if (eof || stopped.load(std::memory_order_relaxed)) return false;
        deadline = now_ms() + config.timeout_ms;
        try {
            for (;;) {
                if (stopped.load(std::memory_order_relaxed)) { deadline = 0; return false; }
                if (now_ms() >= deadline) throw std::runtime_error("read video input: video source timeout");
                const auto received = avcodec_receive_frame(decoder, decoded);
                if (received >= 0) {
                    convert(output);
                    av_frame_unref(decoded);
                    deadline = 0;
                    return true;
                }
                if (received == AVERROR_EOF) { eof = true; deadline = 0; return false; }
                if (received != AVERROR(EAGAIN)) check(received, "decode video frame");
                if (demux_eof) {
                    require(!flushed, "video decoder requested input after EOF flush");
                    check(avcodec_send_packet(decoder, nullptr), "flush video decoder");
                    flushed = true;
                    continue;
                }
                const auto read_result = av_read_frame(format, packet);
                if (read_result == AVERROR_EOF) {
                    // Some I/O protocols expose a read error alongside EOF.
                    if (format->pb && format->pb->error < 0) check(format->pb->error, "read video input");
                    demux_eof = true;
                    continue;
                }
                if (read_result < 0 && stopped.load(std::memory_order_relaxed)) { deadline = 0; return false; }
                check(read_result, "read video input");
                if (packet->stream_index == stream) {
                    const auto sent = avcodec_send_packet(decoder, packet);
                    av_packet_unref(packet);
                    check(sent, "submit video packet");
                } else av_packet_unref(packet);
            }
        } catch (...) {
            failed = true;
            deadline = 0;
            av_packet_unref(packet);
            av_frame_unref(decoded);
            throw;
        }
    }
};

VideoSource::VideoSource(const std::string& uri, const VideoSourceConfig& config)
    : impl_(std::make_unique<Impl>(uri, config)) {}
VideoSource::~VideoSource() = default;
bool VideoSource::read(VideoFrame& frame) { return impl_->read(frame); }
void VideoSource::request_stop() noexcept { impl_->stopped.store(true, std::memory_order_relaxed); }
void VideoSource::close() {
    impl_->check_owner();
    if (impl_->closed) return;
    std::exception_ptr failure;
#ifdef SACCADE_VIDEO_NVDEC
    if (impl_->cuda_context) {
        try {
            CurrentContext current(impl_->cuda_context->context);
            cu(cuStreamSynchronize(impl_->cuda_context->stream), "drain video decode stream");
        } catch (...) { failure = std::current_exception(); }
    }
#endif
    impl_->release();
    if (failure) std::rethrow_exception(failure);
}
std::string VideoSource::identity() const {
    const auto& s = *impl_;
    s.check_owner();
    std::ostringstream out;
    const bool device = s.config.backend == VideoDecodeBackend::Cuda;
    out << "video.backend=" << (device ? "ffmpeg-nvdec" : "ffmpeg-cpu") << "\nvideo.libavformat=" << avformat_version()
        << "\nvideo.libavcodec=" << avcodec_version() << "\nvideo.libswscale=" << swscale_version()
        << "\nvideo.format=" << s.format_name << "\nvideo.decoder=" << s.codec_name
        << "\nvideo.output=" << (device ? "cuda-nv12-or-p010" : "rgb-planar-u8")
        << "\nvideo.device=" << s.config.device << "\nvideo.width=" << s.width << "\nvideo.height=" << s.height
        << "\nvideo.timeout_ms=" << s.config.timeout_ms << "\nvideo.rtsp_transport=" << s.config.rtsp_transport << '\n';
    return out.str();
}
} // namespace saccade

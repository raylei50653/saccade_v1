#include "saccade/video_view.hpp"
#include "saccade/video_overlay.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
}

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace saccade {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void check(int result, const char* operation) {
    if (result >= 0) return;
    char error[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(result, error, sizeof(error));
    throw std::runtime_error(std::string(operation)+": "+error);
}
void cu(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* error = nullptr;
    cuGetErrorName(result, &error);
    throw std::runtime_error(std::string(operation)+": "+(error ? error : "CUDA driver error"));
}
struct CurrentContext {
    explicit CurrentContext(CUcontext ctx) { cu(cuCtxPushCurrent(ctx), "activate view CUDA context"); }
    ~CurrentContext() { CUcontext ignored = nullptr; (void)cuCtxPopCurrent(&ignored); }
};
std::int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

std::vector<video_overlay::Box> video_overlay::prepare(std::span<const ViewBox> input, int height) {
    require(input.size() <= max_boxes, "view supports at most 256 boxes");
    require(height > 0 && height <= 8192, "invalid overlay height");
    std::vector<ViewBox> sorted(input.begin(), input.end());
    std::stable_sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    std::vector<Box> boxes;
    boxes.reserve(sorted.size());
    const int scale = std::max(2, height/360);
    for (const auto& source : sorted) {
        std::int64_t edges[4];
        for (int i = 0; i < 4; ++i) {
            const double v = source.xyxy[i];
            require(std::isfinite(v) && v >= std::numeric_limits<int>::min() && v <= std::numeric_limits<int>::max(),
                    "view box coordinate is not a finite pixel coordinate");
            edges[i] = std::lround(v);
        }
        if (edges[2] < edges[0] || edges[3] < edges[1]) continue;
        Box box{};
        box.x1 = edges[0]; box.y1 = edges[1]; box.x2 = edges[2]; box.y2 = edges[3];
        box.color = palette[source.id%8];
        const auto text = std::to_string(source.id);
        box.digits = static_cast<int>(text.size());
        for (int i = 0; i < box.digits; ++i) box.text[i] = text[i]-'0';
        box.label_width = (6*box.digits+1)*scale;
        box.label_height = 9*scale;
        box.label_y = box.y1-box.label_height >= 0 ? box.y1-box.label_height : box.y1;
        boxes.push_back(box);
    }
    return boxes;
}

struct VideoView::Impl {
    const std::thread::id owner = std::this_thread::get_id();
    VideoViewConfig config;
    int width, height;
    bool rtsp, header = false, closed = false, failed = false;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    CUdeviceptr device_boxes = 0;
    video_overlay::Box* host_boxes = nullptr;
    AVBufferRef* device_ref = nullptr;
    AVBufferRef* frames_ref = nullptr;
    AVCodecContext* encoder = nullptr;
    AVFormatContext* muxer = nullptr;
    AVStream* video = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;
    AVIOContext* file_io = nullptr;
    int fd = -1;
    std::int64_t deadline = 0;
    std::optional<std::int64_t> first_pts, last_pts;

    static int interrupt(void* opaque) {
        return now_ms() >= static_cast<Impl*>(opaque)->deadline;
    }
    static int write_file(void* opaque, const std::uint8_t* data, int size) {
        const auto& self = *static_cast<Impl*>(opaque);
        int total = 0;
        while (total < size) {
            const auto n = ::write(self.fd, data+total, static_cast<std::size_t>(size-total));
            if (n < 0) { if (errno == EINTR) continue; return AVERROR(errno); }
            if (n == 0) return AVERROR(EIO);
            total += static_cast<int>(n);
        }
        return total;
    }
    static std::int64_t seek_file(void* opaque, std::int64_t offset, int whence) {
        const auto& self = *static_cast<Impl*>(opaque);
        if (whence == AVSEEK_SIZE) {
            struct stat status{};
            return ::fstat(self.fd, &status) == 0 ? status.st_size : AVERROR(errno);
        }
        const auto result = ::lseek(self.fd, offset, whence & ~AVSEEK_FORCE);
        return result < 0 ? AVERROR(errno) : result;
    }
    void arm() { deadline = now_ms()+config.timeout_ms; }
    void check_owner() const {
        require(owner == std::this_thread::get_id(), "VideoView requires its original host thread");
    }
    Impl(const VideoViewConfig& cfg, int w, int h)
        : config(cfg), width(w), height(h), rtsp(cfg.url.starts_with("rtsp://")) {
        require(!cfg.url.empty(), "view target must not be empty");
        require(cfg.device >= 0 && cfg.timeout_ms > 0 && cfg.timeout_ms <= 3600000, "invalid view device or timeout");
        require(w > 0 && h > 0 && w <= 8192 && h <= 8192 && w%2 == 0 && h%2 == 0,
                "view requires even NV12 dimensions in 2..8192");
        AVDictionary* options = nullptr;
        try {
            cu(cuInit(0), "initialize view CUDA driver");
            cu(cuDeviceGet(&device, cfg.device), "select view CUDA device");
            cu(cuDevicePrimaryCtxRetain(&context, device), "retain view primary CUDA context");
            CurrentContext current(context);
            cu(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING), "create view CUDA stream");
            check(av_hwdevice_ctx_create(&device_ref, AV_HWDEVICE_TYPE_CUDA, std::to_string(cfg.device).c_str(),
                                         nullptr, AV_CUDA_USE_CURRENT_CONTEXT), "create view FFmpeg CUDA device");
            auto* hardware = reinterpret_cast<AVHWDeviceContext*>(device_ref->data);
            auto* cuda = static_cast<AVCUDADeviceContext*>(hardware->hwctx);
            require(cuda->cuda_ctx == context, "view FFmpeg CUDA context differs from retained primary context");
            cuda->stream = stream;
            frames_ref = av_hwframe_ctx_alloc(device_ref);
            if (!frames_ref) throw std::bad_alloc();
            auto* frames = reinterpret_cast<AVHWFramesContext*>(frames_ref->data);
            frames->format = AV_PIX_FMT_CUDA; frames->sw_format = AV_PIX_FMT_NV12;
            frames->width = w; frames->height = h; frames->initial_pool_size = 4;
            check(av_hwframe_ctx_init(frames_ref), "initialize view CUDA frame pool");
            cu(cuMemAlloc(&device_boxes, sizeof(video_overlay::Box)*video_overlay::max_boxes), "allocate view boxes");
            cu(cuMemAllocHost(reinterpret_cast<void**>(&host_boxes), sizeof(video_overlay::Box)*video_overlay::max_boxes),
               "allocate view box staging");
            check(avformat_network_init(), "initialize view networking");
            check(avformat_alloc_output_context2(&muxer, nullptr, rtsp ? "rtsp" : nullptr, cfg.url.c_str()),
                  "select view muxer");
            if (!muxer) throw std::bad_alloc();
            muxer->interrupt_callback = {interrupt, this};
            const AVCodec* codec = avcodec_find_encoder_by_name("h264_nvenc");
            if (!codec) throw std::runtime_error("view requires h264_nvenc");
            encoder = avcodec_alloc_context3(codec);
            if (!encoder) throw std::bad_alloc();
            encoder->width = w; encoder->height = h;
            encoder->pix_fmt = AV_PIX_FMT_CUDA;
            encoder->time_base = {1,1000000}; encoder->framerate = {30,1};
            encoder->gop_size = 30; encoder->max_b_frames = 0;
            encoder->flags |= AV_CODEC_FLAG_FRAME_DURATION;
            encoder->bit_rate = 8000000; encoder->rc_max_rate = 12000000;
            encoder->hw_frames_ctx = av_buffer_ref(frames_ref);
            if (!encoder->hw_frames_ctx) throw std::bad_alloc();
            if (muxer->oformat->flags & AVFMT_GLOBALHEADER) encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            check(av_opt_set(encoder->priv_data, "preset", "p4", 0), "set view preset");
            check(av_opt_set(encoder->priv_data, "tune", "ll", 0), "set view tune");
            check(av_opt_set(encoder->priv_data, "zerolatency", "1", 0), "set view zero latency");
            check(av_opt_set(encoder->priv_data, "rc", "vbr", 0), "set view rate control");
            check(avcodec_open2(encoder, codec, nullptr), "open view h264_nvenc encoder");
            video = avformat_new_stream(muxer, nullptr);
            if (!video) throw std::bad_alloc();
            video->time_base = encoder->time_base;
            check(avcodec_parameters_from_context(video->codecpar, encoder), "set view stream parameters");
            frame = av_frame_alloc(); packet = av_packet_alloc();
            if (!frame || !packet) throw std::bad_alloc();
            if (!rtsp) {
                // Own the exclusive descriptor through custom AVIO; do not
                // check then reopen by pathname (which would allow races).
                fd = ::open(cfg.url.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
                if (fd < 0) throw std::runtime_error("view exclusive file open failed");
                auto* buffer = static_cast<unsigned char*>(av_malloc(65536));
                if (!buffer) throw std::bad_alloc();
                file_io = avio_alloc_context(buffer, 65536, 1, this, nullptr, write_file, seek_file);
                if (!file_io) { av_free(buffer); throw std::bad_alloc(); }
                muxer->pb = file_io;
                muxer->flags |= AVFMT_FLAG_CUSTOM_IO;
            } else {
                check(av_dict_set(&options, "rtsp_transport", "tcp", 0), "set view RTSP transport");
                // The shared RTSP option is in microseconds (including for
                // output, although ffmpeg -h muxer hides decoding-tagged opts).
                check(av_opt_set_int(muxer->priv_data, "timeout", static_cast<std::int64_t>(cfg.timeout_ms)*1000, 0),
                      "set view RTSP socket timeout");
            }
            arm();
            check(avformat_write_header(muxer, &options), "open view muxer");
            av_dict_free(&options);
            header = true;
        } catch (...) {
            av_dict_free(&options);
            release();
            throw;
        }
    }
    void drain() {
        for (;;) {
            const int result = avcodec_receive_packet(encoder, packet);
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) return;
            check(result, "receive view encoded packet");
            av_packet_rescale_ts(packet, encoder->time_base, video->time_base);
            packet->stream_index = video->index;
            arm();
            check(av_interleaved_write_frame(muxer, packet), "write view packet");
            if (file_io) check(file_io->error, "write view file");
        }
    }
    void write(const DeviceVideoFrame& source, std::optional<std::int64_t> pts, std::span<const ViewBox> input) {
        check_owner();
        if (closed || failed) throw std::runtime_error("view is closed or failed");
        try {
            require(source.format == VideoSurfaceFormat::Nv12, "view requires NV12; P010 is unsupported");
            require(source.width == width && source.height == height && source.device == config.device &&
                    source.context == context && source.ready_event && source.owner &&
                    source.planes[0] && source.planes[1] && source.pitches[0] >= static_cast<unsigned>(width) &&
                    source.pitches[1] >= static_cast<unsigned>(width), "view source surface contract mismatch");
            if (!pts || (last_pts && *pts <= *last_pts))
                throw std::runtime_error("view requires strictly increasing source pts_us");
            if (!first_pts) first_pts = pts;
            if (*first_pts < 0 && *pts > std::numeric_limits<std::int64_t>::max()+*first_pts)
                throw std::runtime_error("view timestamp range overflow");
            const auto boxes = video_overlay::prepare(input, height);
            CurrentContext current(context);
            cu(cuStreamWaitEvent(stream, static_cast<CUevent>(source.ready_event), 0), "wait for view source surface");
            av_frame_unref(frame);
            check(av_hwframe_get_buffer(frames_ref, frame, 0), "acquire view encoder frame");
            for (int plane = 0; plane < 2; ++plane) {
                CUDA_MEMCPY2D copy{};
                copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                copy.srcDevice = reinterpret_cast<CUdeviceptr>(source.planes[plane]);
                copy.srcPitch = source.pitches[plane];
                copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                copy.dstDevice = reinterpret_cast<CUdeviceptr>(frame->data[plane]);
                copy.dstPitch = frame->linesize[plane];
                copy.WidthInBytes = width; copy.Height = plane == 0 ? height : height/2;
                cu(cuMemcpy2DAsync(&copy, stream), "copy view NV12 plane");
            }
            if (!boxes.empty()) {
                std::copy(boxes.begin(), boxes.end(), host_boxes);
                cu(cuMemcpyHtoDAsync(device_boxes, host_boxes, sizeof(video_overlay::Box)*boxes.size(), stream),
                   "upload view boxes");
                const auto result = video_overlay::launch({frame->data[0], frame->data[1],
                    static_cast<std::size_t>(frame->linesize[0]), static_cast<std::size_t>(frame->linesize[1]), width, height},
                    reinterpret_cast<const video_overlay::Box*>(device_boxes), static_cast<unsigned>(boxes.size()), stream);
                if (result != cudaSuccess) throw std::runtime_error("launch view overlay failed");
            }
            cu(cuStreamSynchronize(stream), "complete view copy and overlay");
            frame->pts = *pts-*first_pts;
            // A zero final duration makes MP4's edit list discard the last
            // sample. Use the observed preceding interval, or one time-base
            // tick before an interval exists; never derive PTS from framerate.
            frame->duration = last_pts ? *pts-*last_pts : 1;
            check(avcodec_send_frame(encoder, frame), "submit view encoder frame");
            last_pts = pts;
            drain();
        } catch (...) {
            failed = true;
            // A failed upload/launch can leave a decoder-surface read queued.
            // Finish it before the caller unwinds and releases that surface.
            if (cuCtxPushCurrent(context) == CUDA_SUCCESS) {
                (void)cuStreamSynchronize(stream);
                CUcontext ignored = nullptr;
                (void)cuCtxPopCurrent(&ignored);
            }
            throw;
        }
    }
    void release() noexcept {
        CUcontext ignored = nullptr;
        const bool pushed = context && cuCtxPushCurrent(context) == CUDA_SUCCESS;
        if (pushed && stream) (void)cuStreamSynchronize(stream);
        av_packet_free(&packet); av_frame_free(&frame); avcodec_free_context(&encoder);
        av_buffer_unref(&frames_ref); av_buffer_unref(&device_ref);
        if (muxer) { muxer->pb = nullptr; avformat_free_context(muxer); muxer = nullptr; }
        if (file_io) { av_freep(&file_io->buffer); avio_context_free(&file_io); }
        if (fd >= 0) { (void)::close(fd); fd = -1; }
        if (pushed && device_boxes) (void)cuMemFree(device_boxes);
        if (host_boxes) (void)cuMemFreeHost(host_boxes);
        host_boxes = nullptr; device_boxes = 0;
        if (pushed && stream) (void)cuStreamDestroy(stream);
        if (pushed) (void)cuCtxPopCurrent(&ignored);
        if (context) (void)cuDevicePrimaryCtxRelease(device);
        context = nullptr; stream = nullptr;
    }
    void close() {
        check_owner();
        if (closed) return;
        closed = true;
        std::exception_ptr error;
        const auto attempt = [&](auto operation) {
            try { operation(); } catch (...) { if (!error) error = std::current_exception(); }
        };
        attempt([&] {
            CurrentContext current(context);
            cu(cuStreamSynchronize(stream), "drain view CUDA stream");
            check(avcodec_send_frame(encoder, nullptr), "flush view encoder");
            drain();
        });
        if (header) attempt([&] { arm(); check(av_write_trailer(muxer), "close view muxer"); });
        if (file_io) attempt([&] { avio_flush(file_io); check(file_io->error, "flush view file"); });
        if (fd >= 0) attempt([&] {
            const int descriptor = fd; fd = -1;
            if (::close(descriptor) != 0) throw std::runtime_error("close view file failed");
        });
        release();
        if (error) std::rethrow_exception(error);
    }
    ~Impl() { release(); }
};
VideoView::VideoView(const VideoViewConfig& cfg, int width, int height) : impl_(std::make_unique<Impl>(cfg,width,height)) {}
VideoView::~VideoView() = default;
void VideoView::write(const DeviceVideoFrame& f, std::optional<std::int64_t> pts, std::span<const ViewBox> boxes) {
    impl_->write(f, pts, boxes);
}
void VideoView::close() { impl_->close(); }
} // namespace saccade

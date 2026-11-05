#include "saccade/preprocess.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/resize_cuda.hpp"
#include "saccade/sha256.hpp"
#include "saccade/video_color_cuda.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <cstring>
#include <sstream>
#include <stdexcept>

namespace saccade {
namespace {
void cuda(cudaError_t rc, const char* operation) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(rc));
}
void jpeg_check(nvjpegStatus_t rc, const char* operation) {
    if (rc != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": nvJPEG status " + std::to_string(rc));
}
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
void driver(CUresult rc, const char* operation) {
    if (rc != CUDA_SUCCESS) {
        const char* message = nullptr;
        (void)cuGetErrorString(rc, &message);
        throw std::runtime_error(std::string(operation) + ": " + (message ? message : "CUDA driver failure"));
    }
}
}

struct NativePreprocessor::Impl {
    PreprocessConfig config;
    cudaStream_t stream;
    int device = -1;
    nvjpegHandle_t handle = nullptr;
    nvjpegJpegState_t state = nullptr;
    nvjpegJpegStream_t parsed = nullptr;
    cudaEvent_t done = nullptr;
    std::uint8_t *staging = nullptr, *rgb_staging = nullptr, *raw = nullptr;
    float *images = nullptr, *workspace = nullptr;
    std::size_t raw_capacity = 0, output_capacity = 0, workspace_capacity = 0;
    std::uint64_t next_frame = 0;
    nvjpegImage_t destination{};
    const unsigned char* input = nullptr;
    std::size_t length = 0;
    bool pending = false, failed = false, closed = false;
    std::shared_ptr<void> surface_owner;

    Impl(cudaStream_t d, const PreprocessConfig& cfg) : config(cfg), stream(d) {
        require(d != nullptr && d != cudaStreamLegacy && d != cudaStreamPerThread, "detect stream must be explicit and non-default");
        require(cfg.max_width > 0 && cfg.max_width <= 8192 && cfg.max_height > 0 && cfg.max_height <= 8192,
                "max dimensions must be in 1..8192");
        require(cfg.output_width > 0 && cfg.output_width <= 4096 && cfg.output_height > 0 && cfg.output_height <= 4096,
                "output dimensions must be in 1..4096");
        require(cfg.max_jpeg_bytes > 0 && cfg.max_jpeg_bytes <= 256 * 1024 * 1024, "max_jpeg_bytes must be in 1..256MiB");
        require(cfg.input == PreprocessInput::HardwareJpeg || cfg.input == PreprocessInput::HostRgb || cfg.input == PreprocessInput::DeviceVideo,
                "invalid preprocessing input mode");
        cuda(cudaGetDevice(&device), "cudaGetDevice");
        int stream_device = -1;
        cuda(cudaStreamGetDevice(stream, &stream_device), "cudaStreamGetDevice");
        require(stream_device == device, "detect stream belongs to another device");
        cudaStreamCaptureStatus capture{};
        cuda(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing");
        require(capture == cudaStreamCaptureStatusNone, "JPEG preprocessing does not support graph capture");
        try {
            (void)resize_cuda_load_kernels();
            if (cfg.input == PreprocessInput::DeviceVideo) {
                CUcontext current = nullptr, primary = nullptr;
                driver(cuCtxGetCurrent(&current), "preprocess current context");
                driver(cuDevicePrimaryCtxRetain(&primary, device), "preprocess primary context");
                const bool matches = current == primary;
                driver(cuDevicePrimaryCtxRelease(device), "preprocess release primary reference");
                require(matches, "device video preprocessing requires the primary CUDA context");
                video_color_cuda_load_kernels();
            }
            if (cfg.input == PreprocessInput::HardwareJpeg) {
                jpeg_check(nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, NVJPEG_FLAGS_DEFAULT, &handle), "nvjpegCreateEx HARDWARE");
                jpeg_check(nvjpegJpegStateCreate(handle, &state), "nvjpegJpegStateCreate");
                jpeg_check(nvjpegJpegStreamCreate(handle, &parsed), "nvjpegJpegStreamCreate");
                jpeg_check(nvjpegDecodeBatchedInitialize(handle, state, 1, 1, NVJPEG_OUTPUT_RGB), "nvjpegDecodeBatchedInitialize");
                jpeg_check(nvjpegDecodeBatchedPreAllocate(handle, state, 1, cfg.max_width, cfg.max_height,
                                                         NVJPEG_CSS_420, NVJPEG_OUTPUT_RGB), "nvjpegDecodeBatchedPreAllocate");
            }
            const ResizeShape shape{cfg.max_width, cfg.max_height, cfg.output_width, cfg.output_height};
            raw_capacity = resize_input_bytes(shape);
            output_capacity = resize_output_elements(shape);
            workspace_capacity = resize_workspace_elements(shape);
            if (cfg.input == PreprocessInput::HardwareJpeg)
                cuda(cudaMallocHost(&staging, cfg.max_jpeg_bytes), "cudaMallocHost JPEG");
            else if (cfg.input == PreprocessInput::HostRgb)
                cuda(cudaMallocHost(&rgb_staging, raw_capacity), "cudaMallocHost RGB");
            cuda(cudaMalloc(&raw, raw_capacity), "cudaMalloc RGB");
            cuda(cudaMalloc(&images, output_capacity * sizeof(float)), "cudaMalloc images");
            cuda(cudaMalloc(&workspace, workspace_capacity * sizeof(float)), "cudaMalloc resize workspace");
            cuda(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "cudaEventCreate");
        } catch (...) { release(); throw; }
    }

    void check_open() const {
        require(!closed, "preprocessor is closed");
        require(!failed, "preprocessor sequence has failed");
        int current = -1;
        cuda(cudaGetDevice(&current), "cudaGetDevice");
        require(current == device, "preprocessor requires its original CUDA device");
        cudaStreamCaptureStatus capture{};
        cuda(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing");
        require(capture == cudaStreamCaptureStatusNone, "JPEG preprocessing does not support graph capture");
    }
    void wait_pending() {
        if (pending) {
            cuda(cudaEventSynchronize(done), "preprocess completion event");
            pending = false;
            surface_owner.reset();
        }
    }

    CleanupReport release() noexcept {
        CleanupReport r;
        if (closed) return r;
        closed = true;
        // Resources belong to their original device, even if the caller has
        // changed the current device before checked close/destruction.
        int previous = -1;
        try { cuda(cudaGetDevice(&previous), "cleanup cudaGetDevice"); cuda(cudaSetDevice(device), "cleanup cudaSetDevice"); }
        catch (...) { // Retain the error, still attempt all independent releases.
            int token = 1;
            r.release(token, [](int) { throw; });
        }
        int drain = 1;
        r.release(drain, [&](int) { cuda(cudaStreamSynchronize(stream), "preprocess drain"); });
        surface_owner.reset(); // Also protects D-only tails when event recording failed.
        r.release(parsed, [](auto h) { jpeg_check(nvjpegJpegStreamDestroy(h), "nvjpegJpegStreamDestroy"); });
        r.release(state, [](auto h) { jpeg_check(nvjpegJpegStateDestroy(h), "nvjpegJpegStateDestroy"); });
        r.release(handle, [](auto h) { jpeg_check(nvjpegDestroy(h), "nvjpegDestroy"); });
        r.release(done, [](auto h) { cuda(cudaEventDestroy(h), "cudaEventDestroy"); });
        r.release(workspace, [](auto p) { cuda(cudaFree(p), "cudaFree workspace"); });
        r.release(images, [](auto p) { cuda(cudaFree(p), "cudaFree images"); });
        r.release(raw, [](auto p) { cuda(cudaFree(p), "cudaFree RGB"); });
        r.release(rgb_staging, [](auto p) { cuda(cudaFreeHost(p), "cudaFreeHost RGB"); });
        r.release(staging, [](auto p) { cuda(cudaFreeHost(p), "cudaFreeHost JPEG"); });
        if (previous >= 0 && previous != device) {
            int restore = 1;
            r.release(restore, [&](int) { cuda(cudaSetDevice(previous), "cleanup restore device"); });
        }
        return r;
    }
    ~Impl() { release(); }
};

NativePreprocessor::NativePreprocessor(cudaStream_t d, const PreprocessConfig& cfg) : impl_(std::make_unique<Impl>(d, cfg)) {}
NativePreprocessor::~NativePreprocessor() = default;

PreprocessedFrame NativePreprocessor::enqueue(std::span<const std::uint8_t> data, std::uint64_t frame, int ew, int eh) {
    auto& s = *impl_;
    try {
        s.check_open();
        require(s.config.input == PreprocessInput::HardwareJpeg, "JPEG input requires HardwareJpeg mode");
        require(frame == s.next_frame, "frame identity must be contiguous and zero-based");
        require((ew == 0 && eh == 0) || (ew > 0 && eh > 0), "expected dimensions must both be positive or both zero");
        require(data.size() >= 4 && data.size() <= s.config.max_jpeg_bytes, "JPEG size is empty, truncated or over capacity");
        require(data.front() == 0xff && data[1] == 0xd8 && data[data.size()-2] == 0xff && data.back() == 0xd9,
                "JPEG must have complete SOI/EOI markers");
        s.wait_pending(); // Own compressed bytes/state cannot be reused earlier.
        std::memcpy(s.staging, data.data(), data.size());
        jpeg_check(nvjpegJpegStreamParseHeader(s.handle, s.staging, data.size(), s.parsed), "nvjpegJpegStreamParseHeader");
        nvjpegJpegEncoding_t encoding{};
        jpeg_check(nvjpegJpegStreamGetJpegEncoding(s.parsed, &encoding), "nvjpegJpegStreamGetJpegEncoding");
        require(encoding == NVJPEG_ENCODING_BASELINE_DCT, "only baseline JPEG is supported");
        unsigned precision = 0;
        jpeg_check(nvjpegJpegStreamGetSamplePrecision(s.parsed, &precision), "nvjpegJpegStreamGetSamplePrecision");
        require(precision == 8, "only 8-bit JPEG is supported");
        int components = 0, widths[NVJPEG_MAX_COMPONENT]{}, heights[NVJPEG_MAX_COMPONENT]{};
        nvjpegChromaSubsampling_t subsampling{};
        jpeg_check(nvjpegGetImageInfo(s.handle, s.staging, data.size(), &components, &subsampling, widths, heights), "nvjpegGetImageInfo");
        require(components == 3 && subsampling == NVJPEG_CSS_420, "only three-component YCbCr 4:2:0 JPEG is supported");
        require(widths[0] > 0 && widths[0] <= s.config.max_width && heights[0] > 0 && heights[0] <= s.config.max_height,
                "JPEG dimensions exceed configured capacity");
        require(ew == 0 || (widths[0] == ew && heights[0] == eh), "JPEG dimensions differ from sequence metadata");
        int unsupported = 1;
        jpeg_check(nvjpegDecodeBatchedSupported(s.handle, s.parsed, &unsupported), "nvjpegDecodeBatchedSupported");
        require(unsupported == 0, "JPEG is unsupported by the hardware backend");
        const std::size_t plane = static_cast<std::size_t>(widths[0]) * heights[0];
        for (int c = 0; c < 3; ++c) {
            s.destination.channel[c] = s.raw + c * plane;
            s.destination.pitch[c] = static_cast<std::size_t>(widths[0]);
        }
        s.input = s.staging;
        s.length = data.size();
        jpeg_check(nvjpegDecodeBatched(s.handle, s.state, &s.input, &s.length, &s.destination, s.stream), "nvjpegDecodeBatched HARDWARE");
        const ResizeShape shape{widths[0], heights[0], s.config.output_width, s.config.output_height};
        resize_rgb_chw_cuda(shape, s.raw, s.raw_capacity, s.images, s.output_capacity,
                            s.workspace, s.workspace_capacity, s.stream);
        cuda(cudaEventRecord(s.done, s.stream), "cudaEventRecord preprocessing");
        s.pending = true;
        ++s.next_frame;
        return {frame, {widths[0], heights[0], shape.output_width, shape.output_height,
                       static_cast<double>(shape.output_width) / widths[0], static_cast<double>(shape.output_height) / heights[0]}, s.raw, s.images};
    } catch (...) { s.failed = true; throw; }
}

PreprocessedFrame NativePreprocessor::enqueue_rgb(std::span<const std::uint8_t> data,
                                                  std::uint64_t frame, int width, int height) {
    auto& s = *impl_;
    try {
        s.check_open();
        require(s.config.input == PreprocessInput::HostRgb, "RGB input requires HostRgb mode");
        require(frame == s.next_frame, "frame identity must be contiguous and zero-based");
        require(width > 0 && width <= s.config.max_width && height > 0 && height <= s.config.max_height,
                "RGB dimensions exceed configured capacity");
        const ResizeShape shape{width, height, s.config.output_width, s.config.output_height};
        const auto bytes = resize_input_bytes(shape);
        require(data.size() == bytes, "RGB input must contain exactly three contiguous planes");
        s.wait_pending(); // The previous H2D must retire before overwriting pinned staging.
        std::memcpy(s.rgb_staging, data.data(), bytes);
        cuda(cudaMemcpyAsync(s.raw, s.rgb_staging, bytes, cudaMemcpyHostToDevice, s.stream), "upload RGB");
        resize_rgb_chw_cuda(shape, s.raw, s.raw_capacity, s.images, s.output_capacity,
                            s.workspace, s.workspace_capacity, s.stream);
        cuda(cudaEventRecord(s.done, s.stream), "cudaEventRecord preprocessing");
        s.pending = true;
        ++s.next_frame;
        return {frame, {width, height, shape.output_width, shape.output_height,
                       static_cast<double>(shape.output_width) / width, static_cast<double>(shape.output_height) / height},
                s.raw, s.images};
    } catch (...) { s.failed = true; throw; }
}

PreprocessedFrame NativePreprocessor::enqueue_device(const DeviceVideoFrame& input, std::uint64_t frame) {
    auto& s = *impl_;
    try {
        s.check_open();
        require(s.config.input == PreprocessInput::DeviceVideo, "device video input requires DeviceVideo mode");
        require(frame == s.next_frame, "frame identity must be contiguous and zero-based");
        require(input.width > 0 && input.width <= s.config.max_width && input.height > 0 && input.height <= s.config.max_height,
                "device video dimensions exceed configured capacity");
        s.wait_pending();
        // Retain before the first D operation. On any later error, close() must
        // drain the D-only tail before releasing this surface.
        s.surface_owner = input.owner;
        video_color_cuda(input, s.raw, s.raw_capacity, s.stream);
        const ResizeShape shape{input.width, input.height, s.config.output_width, s.config.output_height};
        resize_rgb_chw_cuda(shape, s.raw, s.raw_capacity, s.images, s.output_capacity,
                            s.workspace, s.workspace_capacity, s.stream);
        cuda(cudaEventRecord(s.done, s.stream), "cudaEventRecord preprocessing");
        s.pending = true;
        ++s.next_frame;
        return {frame, {input.width, input.height, shape.output_width, shape.output_height,
                       static_cast<double>(shape.output_width) / input.width,
                       static_cast<double>(shape.output_height) / input.height}, s.raw, s.images};
    } catch (...) { s.failed = true; throw; }
}

void NativePreprocessor::wait() {
    try { impl_->check_open(); impl_->wait_pending(); }
    catch (...) { impl_->failed = true; throw; }
}
void NativePreprocessor::close() { impl_->release().throw_if_failed(); }
std::string NativePreprocessor::identity() const {
    const auto& s = *impl_;
    int major = 0, minor = 0, patch = 0, runtime = 0, driver = 0;
    if (s.config.input == PreprocessInput::HardwareJpeg) {
        jpeg_check(nvjpegGetProperty(MAJOR_VERSION, &major), "nvjpegGetProperty major");
        jpeg_check(nvjpegGetProperty(MINOR_VERSION, &minor), "nvjpegGetProperty minor");
        jpeg_check(nvjpegGetProperty(PATCH_LEVEL, &patch), "nvjpegGetProperty patch");
    }
    cuda(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    cuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    cudaDeviceProp gpu{};
    cuda(cudaGetDeviceProperties(&gpu, s.device), "cudaGetDeviceProperties");
    std::ostringstream out;
    std::ostringstream config;
    config << "max_width=" << s.config.max_width << "\nmax_height=" << s.config.max_height
           << "\noutput_width=" << s.config.output_width << "\noutput_height=" << s.config.output_height
           << "\nmax_jpeg_bytes=" << s.config.max_jpeg_bytes << '\n';
    if (s.config.input == PreprocessInput::HardwareJpeg) {
        out << "contract=t01-input/1\nbackend=NVJPEG_BACKEND_HARDWARE\noutput=NVJPEG_OUTPUT_RGB\nbatch=1\n"
            << "nvjpeg=" << major << '.' << minor << '.' << patch << '\n';
    } else if (s.config.input == PreprocessInput::HostRgb) {
        config << "input=host-planar-rgb8\n";
        out << "contract=host-rgb-input/1\nresize_contract=t01-input/1\nbackend=HOST_RGB_UPLOAD\noutput=PLANAR_RGB8\nbatch=1\n";
    } else {
        config << "input=device-nv12-p010\ncolor_contract=device-video-color/1\n";
        out << "contract=device-video-input/1\nresize_contract=t01-input/1\nbackend=CUDA_DEVICE_VIDEO\n"
            << "output=PLANAR_RGB8\nbatch=1\ncolor_contract=device-video-color/1\n"
            << "pixel_transfer=none\nchroma=nearest-x2-y2\nrgb8_rounding=nearest-even\n";
    }
    out << "cuda_runtime=" << runtime << "\ncuda_driver_api=" << driver
        << "\ngpu=" << gpu.name << "\nsm=" << gpu.major << '.' << gpu.minor << "\nsource_commit=" << SACCADE_BUILD_COMMIT
        << "\nsource_dirty=" << SACCADE_BUILD_DIRTY << '\n' << config.str() << "config_sha256=" << sha256_hex(config.str()) << '\n';
    return out.str();
}
} // namespace saccade

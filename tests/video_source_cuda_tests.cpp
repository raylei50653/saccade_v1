#include "saccade/video_source.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>
extern "C" {
#include <libavutil/log.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
std::size_t checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void cu(CUresult result, const char* message) {
    if (result != CUDA_SUCCESS) throw std::runtime_error(message);
}
template<class F> void reject(F&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
// D2H exists only in this test oracle. Production VideoSource exports device
// planes directly and never calls av_hwframe_transfer_data or CUDA host copies.
std::vector<std::uint8_t> downloaded_rgb(const saccade::DeviceVideoFrame& frame) {
    cu(cuCtxPushCurrent(static_cast<CUcontext>(frame.context)), "activate download context");
    struct Pop { ~Pop() { CUcontext ignored = nullptr; (void)cuCtxPopCurrent(&ignored); } } pop;
    cu(cuEventSynchronize(static_cast<CUevent>(frame.ready_event)), "surface ready event");
    const auto row = static_cast<std::size_t>(frame.width) * (frame.format == saccade::VideoSurfaceFormat::Nv12 ? 1 : 2);
    std::array<std::vector<std::uint8_t>, 2> host;
    for (unsigned plane = 0; plane < 2; ++plane) {
        const auto rows = static_cast<std::size_t>(frame.height) / (plane ? 2 : 1);
        host[plane].resize(row * rows);
        CUDA_MEMCPY2D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = reinterpret_cast<CUdeviceptr>(frame.planes[plane]); copy.srcPitch = frame.pitches[plane];
        copy.dstMemoryType = CU_MEMORYTYPE_HOST; copy.dstHost = host[plane].data(); copy.dstPitch = row;
        copy.WidthInBytes = row; copy.Height = rows;
        cu(cuMemcpy2D(&copy), "test-only surface download");
    }
    // Compare reconstruction using the CPU decoder's planar layout. swscale's
    // packed NV12 fast path has different chroma interpolation from its planar
    // YUV420 path; losslessly deinterleave first so that is not counted as a
    // hardware decoding error. P010 stores ten bits in the high bits of words.
    const bool ten_bit = frame.format == saccade::VideoSurfaceFormat::P010;
    const auto pixels = static_cast<std::size_t>(frame.width) * frame.height;
    const auto bytes_per_sample = ten_bit ? 2U : 1U;
    std::array<std::vector<std::uint8_t>, 3> planar;
    planar[0].resize(pixels * bytes_per_sample);
    planar[1].resize(pixels / 4 * bytes_per_sample);
    planar[2].resize(pixels / 4 * bytes_per_sample);
    const auto read_sample = [&](const std::vector<std::uint8_t>& data, std::size_t index) {
        return ten_bit ? static_cast<unsigned>((data[index * 2] | (data[index * 2 + 1] << 8)) >> 6) :
            static_cast<unsigned>(data[index]);
    };
    const auto write_sample = [&](std::vector<std::uint8_t>& data, std::size_t index, unsigned value) {
        data[index * bytes_per_sample] = static_cast<std::uint8_t>(value);
        if (ten_bit) data[index * 2 + 1] = static_cast<std::uint8_t>(value >> 8);
    };
    for (std::size_t i = 0; i < pixels; ++i) write_sample(planar[0], i, read_sample(host[0], i));
    for (std::size_t i = 0; i < pixels / 4; ++i) {
        write_sample(planar[1], i, read_sample(host[1], 2 * i));
        write_sample(planar[2], i, read_sample(host[1], 2 * i + 1));
    }
    const auto format = ten_bit ? AV_PIX_FMT_YUV420P10LE : AV_PIX_FMT_YUV420P;
    auto* conversion = sws_getContext(frame.width, frame.height, format, frame.width, frame.height,
        AV_PIX_FMT_GBRP, SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!conversion) throw std::runtime_error("create test RGB converter");
    struct Free { SwsContext* context; ~Free() { sws_freeContext(context); } } free{conversion};
    const int matrix = frame.matrix == saccade::VideoColorMatrix::Bt709 ? SWS_CS_ITU709 :
        frame.matrix == saccade::VideoColorMatrix::Bt2020 ? SWS_CS_BT2020 : SWS_CS_ITU601;
    const auto* coefficients = sws_getCoefficients(matrix);
    if (sws_setColorspaceDetails(conversion, coefficients, frame.full_range, coefficients, 1, 0, 1 << 16, 1 << 16) < 0)
        throw std::runtime_error("configure test RGB converter");
    const std::uint8_t* inputs[4]{planar[0].data(), planar[1].data(), planar[2].data(), nullptr};
    const int input_stride[4]{static_cast<int>(row), static_cast<int>(row / 2), static_cast<int>(row / 2), 0};
    std::vector<std::uint8_t> rgb(pixels * 3);
    std::uint8_t* outputs[4]{rgb.data() + pixels, rgb.data() + 2 * pixels, rgb.data(), nullptr};
    const int output_stride[4]{frame.width, frame.width, frame.width, 0};
    check(sws_scale(conversion, inputs, input_stride, 0, frame.height, outputs, output_stride) == frame.height,
          "complete downloaded surface conversion");
    return rgb;
}
void source_test(const std::string& path, saccade::VideoSurfaceFormat expected_format) {
    using namespace saccade;
    VideoSourceConfig gpu_config; gpu_config.backend = VideoDecodeBackend::Cuda;
    VideoSource gpu(path, gpu_config), cpu(path);
    VideoFrame frame, reference;
    std::vector<VideoFrame> retained;
    unsigned max_error = 0;
    std::uint64_t sum_error = 0, elements = 0;
    std::vector<std::uint8_t> first_snapshot;
    while (gpu.read(frame)) {
        check(cpu.read(reference), "CPU decoder has corresponding presentation frame");
        check(frame.index == retained.size() && frame.index == reference.index, "CUDA presentation indices match CPU");
        check(frame.pts_us == reference.pts_us, "CUDA presentation timestamps match CPU");
        check(frame.width == 320 && frame.height == 192, "NVDEC dimensions preserved");
        const auto& surface = frame.device_frame;
        check(frame.rgb.empty(), "CUDA path has no host RGB image");
        check(surface.owner && surface.planes[0] && surface.planes[1] && surface.ready_event && surface.context,
              "complete leased CUDA surface");
        check(surface.stream != nullptr, "NVDEC producer stream is explicit and non-default");
        check(surface.format == expected_format && surface.matrix == VideoColorMatrix::Bt709 && !surface.full_range,
              "NVDEC format/color metadata preserved");
        CUcontext current = nullptr; cu(cuCtxGetCurrent(&current), "get current CUDA context");
        check(current == surface.context, "source shares existing runtime primary context");
        const auto actual = downloaded_rgb(surface);
        check(actual.size() == reference.rgb.size(), "CPU/GPU RGB extent matches");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto error = static_cast<unsigned>(std::abs(static_cast<int>(actual[i]) - reference.rgb[i]));
            max_error = std::max(max_error, error); sum_error += error;
        }
        elements += actual.size();
        if (retained.empty()) first_snapshot = actual;
        retained.push_back(frame);
    }
    check(retained.size() == 9 && !cpu.read(reference), "NVDEC EOF drains all nine B frames");
    check(!gpu.read(frame), "NVDEC EOF remains terminal");
    check(gpu.identity().find("video.backend=ffmpeg-nvdec") != std::string::npos, "identity records native NVDEC");
    gpu.close(); cpu.close();
    check(downloaded_rgb(retained.front().device_frame) == first_snapshot,
          "frame remains valid after later reads, EOF and source close");
    // Owner cleanup must activate its original CUDA context on a different
    // thread, while other retained frames still keep the device alive.
    frame = {}; // read()'s output otherwise also owns the last retained surface
    const std::weak_ptr<void> released_owner = retained.back().device_frame.owner;
    std::thread release([held = std::move(retained.back())]() mutable { held = {}; }); release.join();
    check(released_owner.expired(), "last surface owner was destroyed on the worker thread");
    retained.pop_back();
    check(downloaded_rgb(retained.front().device_frame) == first_snapshot, "cross-thread lease release preserves other frames");
    const double mean = static_cast<double>(sum_error) / static_cast<double>(elements);
    std::cout << "video_source_cuda comparison format=" << (expected_format == VideoSurfaceFormat::Nv12 ? "nv12" : "p010")
              << " frames=9 max_rgb_error=" << max_error << " mean_rgb_error=" << mean << '\n';
    // Hardware/CPU reconstruction can differ by a least-significant sample.
    check(max_error <= 3 && mean <= 0.15, "NVDEC reconstruction matches CPU within fixed RGB tolerance");
}
void tests(int argc, char** argv) {
    if (argc != 5) throw std::invalid_argument("expected H264 HEVC8 HEVC10 unsupported fixtures");
    av_log_set_level(AV_LOG_QUIET);
    check(cudaSetDevice(0) == cudaSuccess, "initialize existing CUDA runtime context");
    check(cudaFree(nullptr) == cudaSuccess, "activate existing CUDA runtime context");
    source_test(argv[1], saccade::VideoSurfaceFormat::Nv12);
    source_test(argv[2], saccade::VideoSurfaceFormat::Nv12);
    source_test(argv[3], saccade::VideoSurfaceFormat::P010);
    saccade::VideoSourceConfig config; config.backend = saccade::VideoDecodeBackend::Cuda;
    reject([&] { saccade::VideoSource unsupported(argv[4], config); }, "unsupported codec fails instead of CPU fallback");
    config.device = 99999;
    reject([&] { saccade::VideoSource invalid(argv[1], config); }, "unavailable CUDA device fails explicitly");
    config.device = -1;
    reject([&] { saccade::VideoSource invalid(argv[1], config); }, "negative CUDA device rejected");
}
} // namespace
int main(int argc, char** argv) {
    try {
        tests(argc, argv);
        std::cout << "video_source_cuda_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "video_source_cuda_tests: " << error.what() << '\n';
        return 1;
    }
}

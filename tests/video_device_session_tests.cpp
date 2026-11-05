#include "saccade/tracking_session.hpp"
#include "saccade/video_source.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace saccade;
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void cu(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
using Packets = std::vector<PacketBytes>;
Packets run(const char* engine, TrackingSessionConfig cfg,
            const std::vector<DeviceVideoFrame>& surfaces,
            const std::vector<std::vector<std::uint8_t>>& rgb) {
    TrackingSession session(engine, cfg);
    Packets output;
    auto receive = [&] {
        const auto result = session.receive();
        check(result.packet.status == ResultStatus::ok, "non-OK tracking packet");
        check(result.packet.frame == static_cast<FrameId>(output.size()), "frame identity");
        output.push_back(result.bytes);
    };
    for (std::size_t k = 0; k < surfaces.size(); ++k) {
        if (session.pending() == cfg.result_slots) receive();
        if (cfg.preprocess.input == PreprocessInput::DeviceVideo) session.enqueue_device(surfaces[k], k);
        else session.enqueue_rgb(rgb[k], k);
    }
    session.finish();
    while (session.pending()) receive();
    check(session.state() == TrackingSessionState::Finished, "EOF state");
    session.close();
    return output;
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: video_device_session_tests HR_ENGINE H264_VIDEO");
        VideoSourceConfig source_cfg;
        source_cfg.backend = VideoDecodeBackend::Cuda;
        VideoSource source(argv[2], source_cfg);
        VideoFrame frame;
        std::vector<DeviceVideoFrame> surfaces;
        while (source.read(frame)) {
            check(frame.rgb.empty(), "CUDA decoding returned host pixels");
            check(frame.device_frame.owner != nullptr, "missing device surface lease");
            surfaces.push_back(frame.device_frame);
        }
        check(surfaces.size() == 9, "expected nine decoded frames including B-frame tail");
        source.close(); // retained AVFrame/context leases must survive source close
        frame = {};

        std::vector<std::vector<std::uint8_t>> rgb;
        cudaStream_t stream = nullptr;
        cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            PreprocessConfig cfg;
            cfg.input = PreprocessInput::DeviceVideo;
            cfg.max_width = surfaces[0].width; cfg.max_height = surfaces[0].height;
            NativePreprocessor preprocessor(stream, cfg);
            for (std::size_t k = 0; k < surfaces.size(); ++k) {
                const auto input = preprocessor.enqueue_device(surfaces[k], k);
                preprocessor.wait();
                rgb.emplace_back(static_cast<std::size_t>(surfaces[k].width) * surfaces[k].height * 3);
                // Test-only readback creates an exactly identical host baseline.
                cu(cudaMemcpy(rgb.back().data(), input.rgb, rgb.back().size(), cudaMemcpyDeviceToHost));
            }
            preprocessor.close();
        } catch (...) { (void)cudaStreamDestroy(stream); throw; }
        cu(cudaStreamDestroy(stream));

        TrackingSessionConfig cfg;
        cfg.original_width = surfaces[0].width; cfg.original_height = surfaces[0].height;
        cfg.profile = ModelProfile::Hr1280x768;
        cfg.preprocess.input = PreprocessInput::HostRgb;
        cfg.mode = TrackingSessionMode::Serial; cfg.cuda_graph = false; cfg.result_slots = 1;
        const auto expected = run(argv[1], cfg, surfaces, rgb);
        cfg.preprocess.input = PreprocessInput::DeviceVideo;
        check(run(argv[1], cfg, surfaces, rgb) == expected, "device/identical host RGB serial packet bytes differ");
        cfg.mode = TrackingSessionMode::Pipelined; cfg.cuda_graph = true;
        cfg.candidate_slots = 3; cfg.motion_slots = 2; cfg.result_slots = 4;
        check(run(argv[1], cfg, surfaces, rgb) == expected, "device graph pipelined packet bytes differ");
        cfg.cuda_graph = false; cfg.perturb_seed = 29;
        check(run(argv[1], cfg, surfaces, rgb) == expected, "device perturbed packet bytes differ");
        TrackingSession cancelled(argv[1], cfg);
        cancelled.enqueue_device(surfaces[0], 0);
        cancelled.cancel();
        check(cancelled.state() == TrackingSessionState::Cancelled, "cancel state");
        check(cancelled.receive().bytes == expected[0], "cancel retained packet");
        cancelled.close();

        TrackingSession failed(argv[1], cfg);
        auto mismatch = surfaces[0]; --mismatch.width;
        bool rejected = false;
        try { failed.enqueue_device(mismatch, 0); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected && failed.state() == TrackingSessionState::Failed && failed.pending() == 0,
              "dimension mismatch must latch failure before submission");
        failed.close();
        std::cout << "NVDEC device session: 9 frames identical host/device serial/graph/perturbed complete packets; source-close leases, cancellation and dimension failure passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "video_device_session_tests: " << error.what() << '\n';
        return 1;
    }
}

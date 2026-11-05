#include "saccade/tracking_session.hpp"
#include "saccade/file_io.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using namespace saccade;
void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void cu(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
std::span<const std::uint8_t> bytes(const std::string& data) {
    return {reinterpret_cast<const std::uint8_t*>(data.data()), data.size()};
}
using Packets = std::vector<PacketBytes>;
Packets run(const char* engine, TrackingSessionConfig cfg,
            const std::vector<std::string>& jpeg,
            const std::vector<std::vector<std::uint8_t>>& rgb) {
    TrackingSession session(engine, cfg);
    Packets packets;
    auto receive = [&] {
        const auto r = session.receive();
        check(r.packet.status == ResultStatus::ok, "non-OK packet");
        check(r.packet.frame == static_cast<FrameId>(packets.size()), "packet frame");
        packets.push_back(r.bytes);
    };
    const bool host = cfg.preprocess.input == PreprocessInput::HostRgb;
    check(session.identity().find(host ? "backend=HOST_RGB_UPLOAD" : "backend=NVJPEG_BACKEND_HARDWARE") != std::string::npos,
          "input identity");
    for (std::size_t k = 0; k < jpeg.size(); ++k) {
        if (session.pending() == cfg.result_slots) receive();
        if (host) {
            auto owned = rgb[k];
            session.enqueue_rgb(owned, static_cast<FrameId>(k));
            std::fill(owned.begin(), owned.end(), 0); // caller may immediately reuse its bytes
        } else session.enqueue(bytes(jpeg[k]), static_cast<FrameId>(k));
        if (k == 0 && cfg.result_slots == 1) {
            bool full = false;
            try {
                if (host) session.enqueue_rgb(rgb[1], 1);
                else session.enqueue(bytes(jpeg[1]), 1);
            } catch (const std::length_error&) { full = true; }
            check(full && session.state() == TrackingSessionState::Running, "recoverable backpressure");
        }
    }
    session.finish();
    while (session.pending()) receive();
    check(session.state() == TrackingSessionState::Finished, "EOF finish");
    session.close(); session.close();
    return packets;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: tracking_rgb_session_tests HR_ENGINE MOT17-10_DIR");
        constexpr int width = 1920, height = 1080;
        std::vector<std::string> jpeg;
        std::vector<std::vector<std::uint8_t>> rgb;
        cudaStream_t stream = nullptr;
        cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        try {
            NativePreprocessor decoder(stream);
            for (unsigned k = 0; k < 9; ++k) {
                std::ostringstream name;
                name << std::setfill('0') << std::setw(6) << k + 1 << ".jpg";
                jpeg.push_back(read_text_file(std::filesystem::path(argv[2]) / "img1" / name.str()));
                const auto frame = decoder.enqueue(bytes(jpeg.back()), k, width, height);
                decoder.wait();
                rgb.emplace_back(static_cast<std::size_t>(width) * height * 3);
                cu(cudaMemcpy(rgb.back().data(), frame.rgb, rgb.back().size(), cudaMemcpyDeviceToHost));
            }
            decoder.close();
        } catch (...) { (void)cudaStreamDestroy(stream); throw; }
        cu(cudaStreamDestroy(stream));

        TrackingSessionConfig cfg;
        cfg.profile = ModelProfile::Hr1280x768;
        cfg.mode = TrackingSessionMode::Serial; cfg.cuda_graph = false;
        cfg.result_slots = 1;
        const auto expected = run(argv[1], cfg, jpeg, rgb);
        cfg.preprocess.input = PreprocessInput::HostRgb;
        check(run(argv[1], cfg, jpeg, rgb) == expected, "JPEG/RGB ordinary serial packet bytes differ");
        cfg.mode = TrackingSessionMode::Pipelined; cfg.cuda_graph = true;
        cfg.candidate_slots = 3; cfg.motion_slots = 2; cfg.result_slots = 4;
        check(run(argv[1], cfg, jpeg, rgb) == expected, "RGB graph pipelined packet bytes differ");
        cfg.cuda_graph = false;
        cfg.perturb_seed = 23;
        check(run(argv[1], cfg, jpeg, rgb) == expected, "RGB perturbed packet bytes differ");

        TrackingSession cancelled(argv[1], cfg);
        cancelled.enqueue_rgb(rgb[0], 0);
        cancelled.cancel();
        check(cancelled.state() == TrackingSessionState::Cancelled && cancelled.pending() == 1, "cancel retains result");
        check(cancelled.receive().bytes == expected[0], "cancel packet bytes");
        cancelled.close();
        TrackingSession failed(argv[1], cfg);
        bool rejected = false;
        try { failed.enqueue_rgb(std::span<const std::uint8_t>(rgb[0]).first(7), 0); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected && failed.state() == TrackingSessionState::Failed && failed.pending() == 0, "invalid RGB latches failure");
        failed.close();
        std::cout << "RGB session: 9 frames JPEG/host-RGB serial/graph/perturbed packet bytes identical; backpressure, cancel and failure passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tracking_rgb_session_tests: " << e.what() << '\n';
        return 1;
    }
}

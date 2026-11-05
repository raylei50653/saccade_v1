#include "saccade/video_source.hpp"

extern "C" {
#include <libavutil/log.h>
}

#include <charconv>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
template<class T> T number(std::string_view text) {
    T result{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("invalid integer argument");
    return result;
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::invalid_argument("usage: video_probe SOURCE [--format NAME] [--frames N] [--timeout-ms N] [--rtsp-transport tcp|udp] [--decode cpu|cuda] [--device N]");
        saccade::VideoSourceConfig config;
        std::uint64_t limit = 0;
        for (int i = 2; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--format" && i + 1 < argc) config.input_format = argv[++i];
            else if (arg == "--frames" && i + 1 < argc) limit = number<std::uint64_t>(argv[++i]);
            else if (arg == "--timeout-ms" && i + 1 < argc) config.timeout_ms = number<int>(argv[++i]);
            else if (arg == "--rtsp-transport" && i + 1 < argc) config.rtsp_transport = argv[++i];
            else if (arg == "--device" && i + 1 < argc) config.device = number<int>(argv[++i]);
            else if (arg == "--decode" && i + 1 < argc) {
                const std::string_view backend = argv[++i];
                if (backend == "cpu") config.backend = saccade::VideoDecodeBackend::Cpu;
                else if (backend == "cuda") config.backend = saccade::VideoDecodeBackend::Cuda;
                else throw std::invalid_argument("decode backend must be cpu or cuda");
            }
            else throw std::invalid_argument("unknown or incomplete option");
        }
        // The CLI owns process logging policy. Keep URI credentials out of
        // FFmpeg's global logger; reusable VideoSource leaves it unchanged.
        av_log_set_level(AV_LOG_QUIET);
        saccade::VideoSource source(argv[1], config);
        saccade::VideoFrame frame;
        std::uint64_t frames = 0;
        const auto start = std::chrono::steady_clock::now();
        while ((!limit || frames < limit) && source.read(frame)) ++frames;
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << source.identity() << "video.frames=" << frames << "\nvideo.decode_wall_ms=" << ms << '\n';
        source.close();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "video_probe: " << error.what() << '\n';
        return 1;
    }
}

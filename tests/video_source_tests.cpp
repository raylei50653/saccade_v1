#include "saccade/video_source.hpp"

extern "C" {
#include <libavutil/log.h>
}

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {
std::size_t checks = 0;
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
template<class F> void reject(F&& function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        path = std::filesystem::temp_directory_path() /
            ("saccade-video-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("create video test directory");
    }
    ~TemporaryDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
void ppm(std::ostream& output, int width, int height, std::uint8_t offset = 0) {
    output << "P6\n" << width << ' ' << height << "\n255\n";
    for (int pixel = 0; pixel < width * height; ++pixel) {
        const std::array<std::uint8_t,3> rgb{static_cast<std::uint8_t>(10 + pixel + offset),
            static_cast<std::uint8_t>(60 + pixel + offset), static_cast<std::uint8_t>(150 + pixel + offset)};
        output.write(reinterpret_cast<const char*>(rgb.data()), rgb.size());
    }
}
#ifdef __linux__
// A loopback HTTP peer deliberately holds its connection open after a finite
// payload, allowing real FFmpeg I/O timeout/cancellation without an external
// camera, RTSP server, Internet access or timing a decoder mock.
class StalledHttpServer {
    int listener = -1;
    std::jthread worker;
    unsigned port = 0;
public:
    std::atomic<bool> worker_failed{false};
    explicit StalledHttpServer(std::string payload = {}) {
        listener = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener < 0) throw std::runtime_error("create loopback HTTP socket");
        sockaddr_in address{};
        address.sin_family = AF_INET; address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) < 0 || ::listen(listener, 1) < 0) {
            ::close(listener); throw std::runtime_error("bind loopback HTTP socket");
        }
        socklen_t length = sizeof(address);
        if (::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &length) < 0) {
            ::close(listener); throw std::runtime_error("query loopback HTTP socket");
        }
        port = ntohs(address.sin_port);
        worker = std::jthread([this, data = std::move(payload)](std::stop_token stop) {
            while (!stop.stop_requested()) {
                pollfd descriptor{listener, POLLIN, 0};
                const auto result = ::poll(&descriptor, 1, 10);
                if (result < 0 && errno == EINTR) continue;
                if (result < 0) { worker_failed = true; return; }
                if (!result) continue;
                const int peer = ::accept(listener, nullptr, nullptr);
                if (peer < 0) { worker_failed = true; return; }
                if (!data.empty()) {
                    const std::string response = "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n" + data;
                    std::size_t offset = 0;
                    while (offset < response.size()) {
                        const auto bytes = ::send(peer, response.data() + offset, response.size() - offset, MSG_NOSIGNAL);
                        if (bytes < 0 && errno == EINTR) continue;
                        if (bytes <= 0) { worker_failed = true; break; }
                        offset += static_cast<std::size_t>(bytes);
                    }
                }
                while (!stop.stop_requested()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
                ::close(peer);
                return;
            }
        });
    }
    std::string uri() const { return "http://private-user:private-password@127.0.0.1:" + std::to_string(port) + "/"; }
    ~StalledHttpServer() { worker.request_stop(); worker.join(); ::close(listener); }
};
void network_tests() {
    using namespace saccade;
    {
        StalledHttpServer server;
        VideoSourceConfig config; config.timeout_ms = 150;
        const auto start = std::chrono::steady_clock::now();
        bool timed_out = false;
        try { VideoSource source(server.uri(), config); }
        catch (const std::runtime_error& error) {
            const std::string diagnostic = error.what();
            timed_out = diagnostic.find("timeout") != std::string::npos;
            check(diagnostic.find("private-") == std::string::npos, "timeout diagnostic omits URI credentials");
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        check(timed_out, "stalled network open reports timeout");
        check(elapsed < std::chrono::seconds(2), "network open respects deadline");
        check(!server.worker_failed, "loopback timeout peer succeeded");
    }
    std::ostringstream payload;
    for (unsigned index = 0; index < 500; ++index) ppm(payload, 4, 3);
    {
        StalledHttpServer server(payload.str());
        VideoSourceConfig config; config.input_format = "image2pipe"; config.timeout_ms = 200;
        VideoSource source(server.uri(), config);
        VideoFrame frame; std::size_t count = 0;
        bool timed_out = false;
        const auto start = std::chrono::steady_clock::now();
        try { while (source.read(frame)) ++count; }
        catch (const std::runtime_error& error) { timed_out = std::string(error.what()).find("timeout") != std::string::npos; }
        check(count > 0, "network read produced frames before stall");
        check(timed_out, "network read stall is an error rather than clean EOF");
        check(std::chrono::steady_clock::now() - start < std::chrono::seconds(2), "network read respects deadline");
        reject([&] { source.read(frame); }, "network read failure remains latched");
        source.close();
        check(!server.worker_failed, "loopback read peer succeeded");
    }
    {
        StalledHttpServer server(payload.str());
        VideoSourceConfig config; config.input_format = "image2pipe"; config.timeout_ms = 5000;
        VideoSource source(server.uri(), config);
        VideoFrame frame; std::size_t count = 0;
        const auto start = std::chrono::steady_clock::now();
        std::jthread stopper([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            source.request_stop();
        });
        while (source.read(frame)) ++count;
        stopper.join();
        check(count > 0, "network cancellation follows decoded frames");
        check(std::chrono::steady_clock::now() - start < std::chrono::seconds(2), "request_stop interrupts stalled network read");
        check(!source.read(frame), "network cancellation remains terminal");
        source.close();
        check(!server.worker_failed, "loopback cancellation peer succeeded");
    }
}
#endif
void tests(const std::filesystem::path& reordered_video) {
    using namespace saccade;
    TemporaryDirectory temp;
    const auto images = temp.path / "images.ppm";
    {
        std::ofstream output(images, std::ios::binary);
        ppm(output, 4, 3); ppm(output, 4, 3, 7);
        output.close(); check(!output.fail(), "write lossless PPM fixture");
    }
    VideoSourceConfig image_config;
    image_config.input_format = "image2pipe";
    VideoSource source(images.string(), image_config);
    VideoFrame frame;
    for (unsigned index = 0; index < 2; ++index) {
        check(source.read(frame), "read complete frame");
        check(frame.width == 4 && frame.height == 3 && frame.rgb.size() == 36, "original frame geometry");
        check(frame.index == index, "contiguous zero-based presentation index");
        for (unsigned channel = 0; channel < 3; ++channel) {
            const unsigned base = channel == 0 ? 10 : channel == 1 ? 60 : 150;
            for (unsigned pixel = 0; pixel < 12; ++pixel)
                check(frame.rgb[channel * 12 + pixel] == base + pixel + 7 * index, "planar RGB color/channel order exact");
        }
    }
    check(!source.read(frame) && !source.read(frame), "clean EOF is repeatable");
    check(source.identity().find(images.string()) == std::string::npos, "identity omits source URI");
    source.close(); source.close();
    reject([&] { source.read(frame); }, "read after close rejected");

    VideoSource stopped(images.string(), image_config);
    std::thread stop_thread([&] { stopped.request_stop(); }); stop_thread.join();
    check(!stopped.read(frame) && !stopped.read(frame), "cross-thread request_stop prevents submission");
    stopped.close();

    VideoSource limited(images.string(), image_config);
    bool wrong_thread = false;
    std::thread wrong([&] { try { limited.read(frame); } catch (const std::invalid_argument&) { wrong_thread = true; } });
    wrong.join(); check(wrong_thread, "read requires original thread");
    check(limited.read(frame), "wrong-thread use did not latch owner failure");
    limited.close();

    const auto changing = temp.path / "changing.ppm";
    {
        std::ofstream output(changing, std::ios::binary);
        ppm(output, 4, 3); ppm(output, 5, 3);
        output.close(); check(!output.fail(), "write changing-resolution fixture");
    }
    VideoSource changes(changing.string(), image_config);
    check(changes.read(frame), "initial dimensions accepted");
    reject([&] { changes.read(frame); }, "midstream dimension change rejected");
    reject([&] { changes.read(frame); }, "source failure latched");
    changes.close();
    auto small_config = image_config;
    small_config.max_width = 3;
    reject([&] { VideoSource small(images.string(), small_config); small.read(frame); }, "oversized source rejected");
    reject([&] { VideoSource missing((temp.path / "missing").string()); }, "missing source rejected");
    reject([&] { VideoSource empty(""); }, "empty source rejected");
    auto config = image_config; config.timeout_ms = 0;
    reject([&] { VideoSource invalid(images.string(), config); }, "invalid timeout rejected");
    config = image_config; config.input_format = "saccade-unknown-demuxer";
    reject([&] { VideoSource invalid(images.string(), config); }, "unavailable input format rejected");
    config = image_config; config.rtsp_transport = "unknown";
    reject([&] { VideoSource invalid(images.string(), config); }, "invalid RTSP transport rejected");

    VideoSource delayed(reordered_video.string());
    std::size_t count = 0;
    std::optional<std::int64_t> previous;
    while (delayed.read(frame)) {
        check(frame.index == count++, "B-frame presentation index is contiguous");
        check(frame.width == 32 && frame.height == 24, "B-frame geometry preserved");
        check(!frame.pts_us || !previous || *frame.pts_us >= *previous, "presentation timestamps are ordered");
        previous = frame.pts_us;
    }
    check(count == 9, "EOF drains all delayed B frames");
    check(!delayed.read(frame), "B-frame EOF repeatable");
    delayed.close();
#ifdef __linux__
    network_tests();
#endif
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("expected decoder-drain fixture path");
        av_log_set_level(AV_LOG_QUIET);
        tests(argv[1]);
        std::cout << "video_source_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "video_source_tests: " << error.what() << '\n';
        return 1;
    }
}

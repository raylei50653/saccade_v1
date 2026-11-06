#include "saccade/tracking_session.hpp"
#include "saccade/video_source.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/config_cli.hpp"
#include "saccade/resolved_session.hpp"

extern "C" {
#include <libavutil/log.h>
}

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {
std::atomic<std::sig_atomic_t> stop_signal{0};
static_assert(decltype(stop_signal)::is_always_lock_free);
extern "C" void handle_stop(int signal) { stop_signal.store(signal, std::memory_order_relaxed); }
int interrupted() { return stop_signal.load(std::memory_order_relaxed); }

constexpr std::string_view usage = "usage: track_stream ENGINE SOURCE OUT.txt [--decode cuda|cpu] [--device N] [--format NAME] [--profile NAME] [--frames N] [--timeout-ms N] [--rtsp-transport tcp|udp] [--packets OUT.bin] [--no-cuda-graph] [--serial] [--epoch N] [--config FILE] [--set KEY=VALUE] [--resolved-config FILE]";

class Signals {
public:
    Signals() {
        interrupt_ = std::signal(SIGINT, handle_stop);
        if (interrupt_ == SIG_ERR) throw std::runtime_error("cannot install SIGINT handler");
        terminate_ = std::signal(SIGTERM, handle_stop);
        if (terminate_ == SIG_ERR) {
            std::signal(SIGINT, interrupt_);
            throw std::runtime_error("cannot install SIGTERM handler");
        }
    }
    ~Signals() {
        std::signal(SIGINT, interrupt_);
        std::signal(SIGTERM, terminate_);
    }
    Signals(const Signals&) = delete;
    Signals& operator=(const Signals&) = delete;
private:
    using Handler = void (*)(int);
    Handler interrupt_ = SIG_DFL, terminate_ = SIG_DFL;
};

template<class T> T number(std::string_view text, std::string_view flag) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("invalid integer for " + std::string(flag));
    return value;
}

struct Options {
    std::string engine, source;
    std::filesystem::path output, packets, config_path, resolved_path;
    std::string config_text;
    saccade::ConfigAssignments assignments, environment;
    std::set<std::string> explicit_options;
    saccade::VideoSourceConfig video;
    saccade::TrackingSessionConfig tracking;
    std::uint64_t frames = static_cast<std::uint64_t>(std::numeric_limits<saccade::FrameId>::max());
};

saccade::ResolvedTrackingConfig stream_defaults(std::uint32_t width, std::uint32_t height) {
    saccade::ResolvedTrackingConfig defaults;
    defaults.source.width = width; defaults.source.height = height;
    defaults.preprocess.max_width = std::max(1920u, width);
    defaults.preprocess.max_height = std::max(1080u, height);
    defaults.implementation.backend = "cuda"; defaults.implementation.input = "device-video";
    defaults.implementation.motion_profile = saccade::kFlowMotionProfile;
    defaults.implementation.model_profile = saccade::ModelProfile::Hr1280x768;
    defaults.execution.result_slots = 1;
    defaults.execution.mode = "pipelined"; defaults.execution.cuda_graph = true;
    defaults.output_profile = "stream-observations/1";
    defaults.mot.interpolate_max_gap = 0;
    return defaults;
}

saccade::ResolvedTrackingConfig resolve_stream(const Options& options, std::uint32_t width, std::uint32_t height) {
    const auto resolved = saccade::resolve_tracking_config(options.config_text, options.environment,
        options.assignments, {{"cuda"}, {"host-rgb", "device-video"}}, stream_defaults(width, height), "track_stream");
    if (resolved.output_profile != "stream-observations/1")
        throw std::invalid_argument("track_stream requires stream-observations/1 output");
    return resolved;
}

Options parse(int argc, char** argv) {
    if (argc < 4)
        throw std::invalid_argument(std::string(usage));
    Options options;
    options.engine = argv[1]; options.source = argv[2]; options.output = argv[3];
    for (int i = 4; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--no-cuda-graph") options.assignments.emplace_back("execution.cuda_graph", "false");
        else if (argument == "--serial") options.assignments.emplace_back("execution.mode", "serial");
        else {
            if (i + 1 == argc) throw std::invalid_argument("incomplete stream option");
            const std::string_view value = argv[++i];
            const bool alias = argument == "--decode" || argument == "--device" || argument == "--profile" ||
                               argument == "--epoch" || argument == "--set";
            if (!alias && !options.explicit_options.insert(std::string(argument)).second)
                throw std::invalid_argument("duplicate stream option");
            if (argument == "--config") options.config_path = value;
            else if (argument == "--resolved-config") options.resolved_path = value;
            else if (argument == "--set") options.assignments.push_back(saccade::tracking_cli_assignment(value));
            else if (argument == "--format") options.video.input_format = value;
            else if (argument == "--decode") {
                if (value == "cuda") options.assignments.emplace_back("implementation.input", "device-video");
                else if (value == "cpu") options.assignments.emplace_back("implementation.input", "host-rgb");
                else throw std::invalid_argument("--decode must be cuda or cpu");
            }
            else if (argument == "--device") options.assignments.emplace_back("execution.device", value);
            else if (argument == "--profile") options.assignments.emplace_back("implementation.model_profile", value);
            else if (argument == "--frames") options.frames = number<std::uint64_t>(value, argument);
            else if (argument == "--timeout-ms") options.video.timeout_ms = number<int>(value, argument);
            else if (argument == "--rtsp-transport") options.video.rtsp_transport = value;
            else if (argument == "--packets") options.packets = value;
            else if (argument == "--epoch") options.assignments.emplace_back("source.epoch", value);
            else throw std::invalid_argument("unknown stream option");
        }
    }
    if (options.frames == 0 || options.frames > static_cast<std::uint64_t>(std::numeric_limits<saccade::FrameId>::max()))
        throw std::invalid_argument("--frames must be in 1..INT64_MAX");
    if (options.video.timeout_ms <= 0) throw std::invalid_argument("--timeout-ms must be positive");
    if (options.video.rtsp_transport != "tcp" && options.video.rtsp_transport != "udp")
        throw std::invalid_argument("--rtsp-transport must be tcp or udp");
    for (const unsigned char c : options.video.input_format)
        if (!(c >= 'a' && c <= 'z') && !(c >= 'A' && c <= 'Z') && !(c >= '0' && c <= '9') &&
            c != '_' && c != '-') throw std::invalid_argument("--format requires a format name");
    if (!options.config_path.empty()) options.config_text = saccade::read_text_file(options.config_path);
    options.environment = saccade::declared_tracking_environment();
    // Validate typed options before opening a potentially blocking source. Only
    // dimensions are provisional here; the first decoded frame locks them below.
    const auto preview = saccade::native_preview_defaults(stream_defaults(1920, 1080), options.config_text,
                                                         options.environment, options.assignments);
    const auto resolved = resolve_stream(options, preview.source.width, preview.source.height);
    options.tracking = saccade::session_config_from_resolved(resolved);
    options.video.device = options.tracking.device;
    options.video.backend = resolved.implementation.input == "device-video" ?
        saccade::VideoDecodeBackend::Cuda : saccade::VideoDecodeBackend::Cpu;
    return options;
}

std::filesystem::path normalized(const std::filesystem::path& path) {
    return std::filesystem::weakly_canonical(std::filesystem::absolute(path));
}

void validate_outputs(const Options& options) {
    const auto output = normalized(options.output);
    const auto engine = normalized(options.engine);
    const bool local_source = options.source.find("://") == std::string::npos;
    const auto source = local_source ? normalized(options.source) : std::filesystem::path{};
    const auto check = [&](const std::filesystem::path& path) {
        const auto resolved = normalized(path);
        if (resolved == engine || (local_source && resolved == source))
            throw std::invalid_argument("stream output must differ from engine and source");
        if (std::filesystem::symlink_status(path).type() != std::filesystem::file_type::not_found)
            throw std::invalid_argument("stream output already exists; choose a new path");
    };
    check(options.output);
    if (!options.packets.empty()) {
        if (normalized(options.packets) == output)
            throw std::invalid_argument("MOT and packet outputs must differ");
        check(options.packets);
    }
    if (!options.resolved_path.empty()) {
        const auto resolved = normalized(options.resolved_path);
        if (resolved == output || (!options.packets.empty() && resolved == normalized(options.packets)) ||
            (!options.config_path.empty() && resolved == normalized(options.config_path)))
            throw std::invalid_argument("resolved config output must differ from inputs and other outputs");
        check(options.resolved_path);
    }
}

class ExclusiveOutput {
public:
    ExclusiveOutput() = default;
    ~ExclusiveOutput() { if (file_) (void)std::fclose(file_); }
    ExclusiveOutput(const ExclusiveOutput&) = delete;
    ExclusiveOutput& operator=(const ExclusiveOutput&) = delete;
    void open(const std::filesystem::path& path, const char* label) {
        label_ = label;
        const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
        if (descriptor < 0) throw std::runtime_error(label_ + " stream open failed: " + std::strerror(errno));
        file_ = ::fdopen(descriptor, "wb");
        if (!file_) {
            const int primary = errno;
            if (::close(descriptor) != 0) std::cerr << "track_stream cleanup: output descriptor close failed\n";
            throw std::runtime_error(label_ + " stream fdopen failed: " + std::strerror(primary));
        }
    }
    bool is_open() const { return file_ != nullptr; }
    void write(const void* data, std::size_t size) {
        if (size && std::fwrite(data, 1, size, file_) != size)
            throw std::runtime_error(label_ + " stream write failed");
        if (std::fflush(file_) != 0) throw std::runtime_error(label_ + " stream flush failed");
    }
    void close() {
        if (!file_) return;
        auto* file = file_;
        file_ = nullptr;
        if (std::fclose(file) != 0) throw std::runtime_error(label_ + " stream close failed");
    }
private:
    std::FILE* file_ = nullptr;
    std::string label_;
};

// Online observations use the stable packet output_id directly. EOF-dependent
// interpolation and all-time ID remapping belong to the offline MOT writer.
class StreamOutput {
public:
    explicit StreamOutput(const Options& options) {
        try {
            // The earlier validation is advisory; O_EXCL enforces no overwrite
            // after input probing and engine setup, including concurrent runs.
            mot_.open(options.output, "MOT");
            if (!options.packets.empty()) {
                packets_.open(options.packets, "packet");
            }
        } catch (...) {
            const auto primary = std::current_exception();
            try { close(); }
            catch (const std::exception& error) { std::cerr << "track_stream cleanup: " << error.what() << '\n'; }
            std::rethrow_exception(primary);
        }
    }
    void add(const saccade::TrackingSessionResult& result) {
        if (packets_.is_open()) {
            packets_.write(result.bytes.data(), result.bytes.size());
        }
        if (result.packet.status != saccade::ResultStatus::ok) {
            failed_ = true;
            return;
        }
        // No rows after a failed packet may enter the valid observation prefix.
        if (failed_) return;
        std::vector<saccade::MotRow> rows;
        rows.reserve(result.packet.tracks.size());
        for (const auto& row : result.packet.tracks)
            rows.push_back({result.packet.frame + 1, row.output_id, row.xyxy[0], row.xyxy[1],
                            row.xyxy[2], row.xyxy[3], row.observed_score, false});
        std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) { return left.id < right.id; });
        const auto text = saccade::format_mot_rows(rows);
        mot_.write(text.data(), text.size());
    }
    bool failed() const { return failed_; }
    void close() {
        std::exception_ptr error;
        const auto checked_close = [&](ExclusiveOutput& file) {
            try { file.close(); }
            catch (...) { if (!error) error = std::current_exception(); }
        };
        checked_close(mot_);
        checked_close(packets_);
        if (error) std::rethrow_exception(error);
    }
private:
    ExclusiveOutput mot_, packets_;
    bool failed_ = false;
};

template<class Operation> void cleanup(Operation operation, std::exception_ptr& primary) {
    try { operation(); }
    catch (...) {
        if (!primary) primary = std::current_exception();
        else {
            // Keep the original failure authoritative while reporting cleanup.
            try { throw; }
            catch (const std::exception& error) { std::cerr << "track_stream cleanup: " << error.what() << '\n'; }
            catch (...) { std::cerr << "track_stream cleanup: unknown error\n"; }
        }
    }
}

int run(Options options) {
    validate_outputs(options);
    Signals signals;
    const auto start = std::chrono::steady_clock::now();
    std::unique_ptr<saccade::VideoSource> owned_source;
    try { owned_source = std::make_unique<saccade::VideoSource>(options.source, options.video); }
    catch (...) {
        // The source does not exist yet, so request_stop() cannot be called.
        // Open remains bounded by its configured I/O timeout; preserve a user
        // cancellation once construction returns or throws at that boundary.
        if (const int signal = interrupted()) {
            std::cerr << "track_stream: cancelled while opening video source\n";
            std::cout << "track_stream submitted=0 received=0 end=cancelled stage=source-open\n";
            return 128 + signal;
        }
        throw;
    }
    auto& source = *owned_source;
    std::jthread watcher([&](std::stop_token token) {
        while (!token.stop_requested()) {
            if (interrupted()) { source.request_stop(); return; }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    std::unique_ptr<saccade::TrackingSession> session;
    std::unique_ptr<StreamOutput> output;
    std::exception_ptr error;
    std::uint64_t submitted = 0, received = 0;
    bool stopped = false, frame_limit = false;
    try {
        saccade::VideoFrame frame;
        const bool first = source.read(frame);
        stopped = interrupted() != 0;
        if (!first && !stopped) throw std::runtime_error("video source contains no frames");
        if (first) {
            auto resolved = resolve_stream(options, static_cast<std::uint32_t>(frame.width),
                                                  static_cast<std::uint32_t>(frame.height));
            if (resolved.source.width != static_cast<std::uint32_t>(frame.width) ||
                resolved.source.height != static_cast<std::uint32_t>(frame.height))
                throw std::invalid_argument("resolved source dimensions differ from decoded metadata");
            saccade::mark_native_input_origins(resolved, "decoded-frame-metadata");
            options.tracking = saccade::session_config_from_resolved(resolved);
            session = std::make_unique<saccade::TrackingSession>(options.engine, options.tracking);
            const auto build_identity = saccade::tracking_build_identity(options.engine) +
                "video.open.input_format=" + options.video.input_format + '\n' +
                "video.open.max_width=" + std::to_string(options.video.max_width) + '\n' +
                "video.open.max_height=" + std::to_string(options.video.max_height) + '\n' +
                "video.open.frame_limit=" + std::to_string(options.frames) + '\n' +
                (!options.config_path.empty() ? "input.config_file_sha256=" + saccade::sha256_hex(options.config_text) + '\n' : std::string{}) +
                source.identity();
            const auto record = saccade::resolved_config_record(resolved, build_identity);
            if (!options.resolved_path.empty()) saccade::write_resolved_config_file(options.resolved_path, record);
            output = std::make_unique<StreamOutput>(options);
            std::cout << "output_profile=stream-observations/1\noutput_ids=native\ninterpolation=disabled\n"
                      << "tracker_time_model=fixed-step\n" << source.identity() << '\n' << session->identity() << record;
            const auto receive = [&] {
                const auto result = session->receive();
                ++received;
                output->add(result);
            };
            while (!interrupted()) {
                if (frame.width != options.tracking.original_width || frame.height != options.tracking.original_height)
                    throw std::runtime_error("video dimensions changed; begin a new tracking session");
                if (frame.index != submitted) throw std::runtime_error("video frame index is not contiguous");
                if (session->pending()) receive();
                if (output->failed()) break;
                if (options.video.backend == saccade::VideoDecodeBackend::Cuda)
                    session->enqueue_device(frame.device_frame, static_cast<saccade::FrameId>(submitted));
                else session->enqueue_rgb(frame.rgb, static_cast<saccade::FrameId>(submitted));
                ++submitted;
                if (session->pending()) receive();
                if (output->failed()) break;
                if (submitted == options.frames) { frame_limit = true; break; }
                if (!source.read(frame)) break;
            }
            stopped = interrupted() != 0;
            if (stopped || output->failed()) session->cancel();
            else session->finish();
            while (session->pending()) receive();
            if (output->failed()) throw std::runtime_error("tracking returned a failed packet; outputs contain a partial prefix");
        }
    } catch (...) {
        error = std::current_exception();
        if (session) {
            cleanup([&] { session->cancel(); }, error);
            while (session->pending()) {
                bool obtained = false;
                cleanup([&] {
                    const auto result = session->receive();
                    ++received;
                    obtained = true;
                    if (output) output->add(result);
                }, error);
                // A failed receive may leave the same slot pending. Checked
                // close still drains stream tails; never retry indefinitely.
                if (!obtained) break;
            }
        }
    }
    watcher.request_stop();
    watcher.join();
    if (session) cleanup([&] { session->close(); }, error);
    cleanup([&] { source.close(); }, error);
    if (output) cleanup([&] { output->close(); }, error);
    if (error) std::rethrow_exception(error);
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "track_stream submitted=" << submitted << " received=" << received
              << " end=" << (stopped ? "cancelled" : frame_limit ? "frame-limit" : "eof")
              << " wall_ms=" << elapsed << " timing=source-open-through-output-close\n";
    return stopped ? 128 + interrupted() : 0;
}
} // namespace

int main(int argc, char** argv) {
    // FFmpeg's default logger can include credential-bearing source URLs.
    av_log_set_level(AV_LOG_QUIET);
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << usage << '\n';
        return 0;
    }
    try { return run(parse(argc, argv)); }
    catch (const std::exception& error) {
        std::cerr << "track_stream: " << error.what() << '\n';
        return 1;
    }
}

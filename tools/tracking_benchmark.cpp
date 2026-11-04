#include "saccade/tracking_session.hpp"
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Bytes = std::vector<saccade::PacketBytes>;
constexpr const char* usage = "usage: tracking_benchmark ENGINE SEQ_DIR NEW_OUT_DIR [--frames 120] [--warmup 20] [--rounds 2] [--profile hr-1280x768] [--only ordinary|graph]";

template<class T> T number(std::string_view text) {
    T result{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (ec != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("invalid integer");
    return result;
}
std::string trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    return first == std::string_view::npos ? "" : std::string(text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1));
}
std::string quote(std::string_view text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}
double microseconds(Clock::duration value) { return std::chrono::duration<double, std::micro>(value).count(); }
double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}
// Nearest-rank percentiles of measured frames, never of batch means.
double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const auto i = static_cast<std::size_t>(std::ceil(q * static_cast<double>(values.size())));
    return values[std::max<std::size_t>(1, i) - 1];
}
std::string packet_hash(const Bytes& bytes) {
    return saccade::sha256_hex({reinterpret_cast<const char*>(bytes.data()), bytes.size() * sizeof(bytes[0])});
}
void same_packets(const Bytes& expected, const Bytes& actual) {
    if (expected.size() != actual.size()) throw std::runtime_error("packet frame count changed");
    for (std::size_t frame = 0; frame < actual.size(); ++frame) {
        const auto mismatch = std::mismatch(expected[frame].begin(), expected[frame].end(), actual[frame].begin());
        if (mismatch.first != expected[frame].end())
            throw std::runtime_error("packet mismatch at frame=" + std::to_string(frame) + " byte=" +
                                     std::to_string(mismatch.first - expected[frame].begin()));
    }
}
struct Run {
    unsigned round = 0, ordinal = 0;
    bool graph = false, queued = false;
    double init_us = 0, first_frame_us = 0, warmup_us = 0, wall_us = 0;
    std::vector<double> latency_us;
    std::string hash;
};
Run run(const std::string& engine, saccade::TrackingSessionConfig cfg,
        const std::vector<std::string>& images, std::size_t warmup, bool queued,
        unsigned round, unsigned ordinal, const std::filesystem::path& out, Bytes& reference) {
    Run result;
    result.round = round; result.ordinal = ordinal; result.graph = cfg.cuda_graph; result.queued = queued;
    const std::string id = "r" + std::to_string(round) + "-" + std::to_string(ordinal) + "-" +
                           (cfg.cuda_graph ? "graph" : "ordinary") + "-" + (queued ? "queued" : "single");
    const auto before = Clock::now();
    saccade::TrackingSession session(engine, cfg);
    result.init_us = microseconds(Clock::now() - before);
    Bytes packets(images.size());
    std::size_t received = 0;
    Clock::time_point first_frame_start;
    const auto receive = [&] {
        const auto packet = session.receive();
        if (received == 0) result.first_frame_us = microseconds(Clock::now() - first_frame_start);
        if (packet.packet.status != saccade::ResultStatus::ok || packet.packet.frame != static_cast<saccade::FrameId>(received))
            throw std::runtime_error("non-OK or wrong frame in " + id + " frame=" + std::to_string(received));
        packets[received++] = packet.bytes;
    };
    const auto enqueue = [&](std::size_t frame) {
        const auto& image = images[frame];
        if (frame == 0) first_frame_start = Clock::now();
        session.enqueue({reinterpret_cast<const std::uint8_t*>(image.data()), image.size()}, frame);
    };
    // Same prefix and tracker history in every fresh session; warmup packets
    // are part of the byte comparison but excluded from all timing statistics.
    const auto warmup_start = Clock::now();
    for (std::size_t frame = 0; frame < warmup; ++frame) {
        enqueue(frame); receive();
    }
    result.warmup_us = microseconds(Clock::now() - warmup_start);
    result.latency_us.reserve(images.size() - warmup);
    const auto start = Clock::now();
    for (std::size_t frame = warmup; frame < images.size(); ++frame) {
        if (queued) {
            if (session.pending() == cfg.result_slots) receive();
            enqueue(frame);
        } else {
            const auto submitted = Clock::now();
            enqueue(frame); receive();
            const double elapsed = microseconds(Clock::now() - submitted);
            result.latency_us.push_back(elapsed);
        }
    }
    while (session.pending()) receive();
    result.wall_us = microseconds(Clock::now() - start);
    session.finish();
    if (session.state() != saccade::TrackingSessionState::Finished) throw std::runtime_error("session did not finish normally");
    saccade::write_text_file(out / (id + ".identity.txt"), session.identity());
    session.close();
    if (reference.empty()) reference = packets;
    else same_packets(reference, packets);
    result.hash = packet_hash(packets);
    saccade::write_binary_file(out / (id + ".bin"),
        {reinterpret_cast<const std::uint8_t*>(packets.data()), packets.size() * sizeof(packets[0])});
    std::ostringstream samples;
    samples << "frame,enqueue_to_receive_us\n" << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < result.latency_us.size(); ++i) samples << i + warmup << ',' << result.latency_us[i] << '\n';
    if (!queued) saccade::write_text_file(out / (id + ".csv"), samples.str());
    std::cout << std::fixed << std::setprecision(3) << id << " init_ms=" << result.init_us / 1000
              << " first_frame_ms=" << result.first_frame_us / 1000 << " warmup_ms=" << result.warmup_us / 1000
              << " wall_ms=" << result.wall_us / 1000 << " wall_us_per_frame=" << result.wall_us / (images.size() - warmup);
    if (!queued) std::cout << " latency_mean_us=" << mean(result.latency_us)
                          << " p50_us=" << percentile(result.latency_us, .50)
                          << " p95_us=" << percentile(result.latency_us, .95)
                          << " p99_us=" << percentile(result.latency_us, .99);
    std::cout << " packet_sha256=" << result.hash << " packets_equal=PASS\n" << std::flush;
    return result;
}
} // namespace

int main(int argc, char** argv) {
    std::filesystem::path output_directory;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") { std::cout << usage << '\n'; return 0; }
        if (argc < 4) throw std::invalid_argument(usage);
        std::size_t frames = 120, warmup = 20;
        unsigned rounds = 2;
        std::string only;
        saccade::TrackingSessionConfig cfg;
        cfg.profile = saccade::parse_model_profile("hr-1280x768");
        for (int i = 4; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument(usage);
            const std::string_view value = argv[++i];
            if (arg == "--frames") frames = number<std::size_t>(value);
            else if (arg == "--warmup") warmup = number<std::size_t>(value);
            else if (arg == "--rounds") rounds = number<unsigned>(value);
            else if (arg == "--profile") cfg.profile = saccade::parse_model_profile(value);
            else if (arg == "--only" && (value == "ordinary" || value == "graph")) only = value;
            else throw std::invalid_argument("unknown or invalid option: " + std::string(arg));
        }
        if (frames == 0 || warmup == 0 || rounds == 0 || rounds > 100 || frames > 1000000 || warmup > 1000000)
            throw std::invalid_argument("positive frames/warmup/rounds required; rounds <= 100, frames/warmup <= 1000000");
        const std::filesystem::path seq = argv[2], out = argv[3];
        std::map<std::string, std::string> fields;
        const auto sequence_info = saccade::read_text_file(seq / "seqinfo.ini");
        std::istringstream info(sequence_info);
        std::string line;
        while (std::getline(info, line)) {
            const auto equals = line.find('=');
            if (equals != std::string::npos) fields[trim(std::string_view(line).substr(0, equals))] = trim(std::string_view(line).substr(equals + 1));
        }
        if (frames + warmup > number<std::size_t>(fields.at("seqLength"))) throw std::invalid_argument("sequence shorter than warmup + frames");
        cfg.original_width = number<int>(fields.at("imWidth")); cfg.original_height = number<int>(fields.at("imHeight"));
        if (!std::filesystem::create_directory(out)) throw std::invalid_argument("output directory must be new");
        output_directory = out;
        saccade::write_text_file(out / "status.json", "{\"status\":\"RUNNING\"}\n");
        std::ostringstream manifest;
        manifest << saccade::sha256_hex(saccade::read_text_file(argv[1])) << "  " << argv[1] << '\n'
                 << saccade::sha256_hex(sequence_info) << "  " << (seq / "seqinfo.ini").string() << '\n';
        std::vector<std::string> images;
        images.reserve(frames + warmup);
        for (std::size_t frame = 0; frame < frames + warmup; ++frame) {
            std::ostringstream filename;
            filename << std::setfill('0') << std::setw(6) << frame + 1 << fields.at("imExt");
            const auto path = seq / fields.at("imDir") / filename.str();
            images.push_back(saccade::read_text_file(path));
            manifest << saccade::sha256_hex(images.back()) << "  " << path.string() << '\n';
        }
        saccade::write_text_file(out / "inputs.sha256", manifest.str());
        std::ostringstream command;
        for (int i = 0; i < argc; ++i) command << (i ? " " : "") << quote(argv[i]);
        saccade::write_text_file(out / "command.txt", command.str() + '\n');
        std::vector<Run> runs;
        Bytes reference;
        // ABBA controls slow clock/thermal drift. Every entry uses fresh
        // sessions for single-frame latency and queued throughput separately.
        for (unsigned round = 0; round < rounds; ++round) {
            unsigned ordinal = 0;
            for (const bool graph : {false, true, true, false}) {
                ++ordinal;
                if ((!only.empty()) && graph != (only == "graph")) continue;
                cfg.cuda_graph = graph;
                for (const bool queued : {false, true}) runs.push_back(run(argv[1], cfg, images, warmup, queued, round, ordinal, out, reference));
            }
        }
        std::ostringstream csv, json;
        csv << "round,ordinal,graph,style,init_us,first_frame_us,warmup_us,wall_us,wall_us_per_frame,mean_latency_us,p50_us,p95_us,p99_us,packet_sha256\n" << std::fixed << std::setprecision(6);
        for (const auto& r : runs) {
            csv << r.round << ',' << r.ordinal << ',' << r.graph << ',' << (r.queued ? "queued" : "single") << ','
                << r.init_us << ',' << r.first_frame_us << ',' << r.warmup_us << ',' << r.wall_us << ',' << r.wall_us / frames << ',';
            if (!r.queued) csv << mean(r.latency_us) << ',' << percentile(r.latency_us, .50) << ',' << percentile(r.latency_us, .95) << ',' << percentile(r.latency_us, .99);
            else csv << ",,,";
            csv << ',' << r.hash << '\n';
        }
        saccade::write_text_file(out / "runs.csv", csv.str());
        json << std::fixed << std::setprecision(6) << "{\n  \"schema\":\"tracking-latency/1\",\n  \"status\":\"PASS\",\n"
             << "  \"sequence\":" << quote(std::filesystem::absolute(seq).string()) << ",\n  \"frames_per_session\":" << frames
             << ",\n  \"warmup_frames\":" << warmup << ",\n  \"abba_rounds\":" << rounds
             << ",\n  \"input_io\":\"JPEGs preloaded; excludes disk I/O and file output\",\n"
             << "  \"warmup_scope\":\"constructor, first frame, and warmup times separately reported; lazy capture during warmup is excluded from latency samples\",\n"
             << "  \"latency_scope\":\"one enqueue through blocking receive, no frames in flight at start\",\n"
             << "  \"queued_scope\":\"wall time per frame is throughput, not individual frame latency\",\n"
             << "  \"packet_comparison\":" << quote(only.empty() ? "ordinary/graph and single/queued, all runs, including warmup" : "selected mode repeats and single/queued, including warmup")
             << ",\n  \"packet_sha256\":" << quote(packet_hash(reference)) << ",\n  \"groups\":[\n";
        bool first = true;
        for (const bool graph : {false, true}) {
            std::vector<double> latencies, throughputs, initialization, first_frame_times, warmup_times;
            for (const auto& r : runs) if (r.graph == graph) {
                initialization.push_back(r.init_us);
                first_frame_times.push_back(r.first_frame_us);
                warmup_times.push_back(r.warmup_us);
                if (r.queued) throughputs.push_back(r.wall_us / frames);
                else latencies.insert(latencies.end(), r.latency_us.begin(), r.latency_us.end());
            }
            if (latencies.empty()) continue;
            if (!first) json << ",\n";
            first = false;
            json << "    {\"mode\":" << quote(graph ? "graph" : "ordinary") << ",\"latency_samples\":" << latencies.size()
                 << ",\"latency_mean_us\":" << mean(latencies) << ",\"latency_p50_us\":" << percentile(latencies,.50)
                 << ",\"latency_p95_us\":" << percentile(latencies,.95) << ",\"latency_p99_us\":" << percentile(latencies,.99)
                 << ",\"queued_us_per_frame_mean\":" << mean(throughputs)
                 << ",\"queued_us_per_frame_min\":" << *std::min_element(throughputs.begin(), throughputs.end())
                 << ",\"queued_us_per_frame_max\":" << *std::max_element(throughputs.begin(), throughputs.end())
                 << ",\"init_us_mean\":" << mean(initialization)
                 << ",\"first_frame_us_mean\":" << mean(first_frame_times)
                 << ",\"warmup_us_mean\":" << mean(warmup_times) << '}';
        }
        json << "\n  ]";
        if (only.empty()) {
            std::vector<double> ordinary, graph, ordinary_queued, graph_queued;
            for (const auto& r : runs) {
                if (r.queued) (r.graph ? graph_queued : ordinary_queued).push_back(r.wall_us / frames);
                else {
                    auto& samples = r.graph ? graph : ordinary;
                    samples.insert(samples.end(), r.latency_us.begin(), r.latency_us.end());
                }
            }
            json << ",\n  \"comparison\":{\"mean_latency_speedup\":" << mean(ordinary) / mean(graph)
                 << ",\"mean_latency_reduction_percent\":" << 100.0 * (1.0 - mean(graph) / mean(ordinary))
                 << ",\"p95_latency_speedup\":" << percentile(ordinary,.95) / percentile(graph,.95)
                 << ",\"queued_throughput_speedup\":" << mean(ordinary_queued) / mean(graph_queued) << '}';
        }
        json << "\n}\n";
        saccade::write_text_file(out / "summary.json", json.str());
        saccade::write_text_file(out / "status.json", "{\"status\":\"PASS\"}\n");
        std::cout << json.str();
        return 0;
    } catch (const std::exception& e) {
        if (!output_directory.empty()) {
            try { saccade::write_text_file(output_directory / "status.json", "{\"status\":\"FAIL\",\"error\":" + quote(e.what()) + "}\n"); }
            catch (const std::exception& write_error) { std::cerr << "record failure: " << write_error.what() << '\n'; }
        }
        std::cerr << "tracking_benchmark: " << e.what() << '\n';
        return 1;
    }
}

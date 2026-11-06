#include "saccade/tracking_session.hpp"
#include "saccade/file_io.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/config_cli.hpp"
#include "saccade/resolved_session.hpp"

#include <charconv>
#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
std::string trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos) return {};
    return std::string(text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1));
}

template<class T> T number(std::string_view text) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("invalid integer: " + std::string(text));
    return value;
}

struct SequenceInfo {
    std::string directory, extension;
    std::uint64_t frames;
    int width, height;
};

SequenceInfo read_sequence_info(const std::filesystem::path& path) {
    std::istringstream input(saccade::read_text_file(path));
    std::map<std::string, std::string> fields;
    std::string line;
    while (std::getline(input, line)) {
        const auto equals = line.find('=');
        if (equals != std::string::npos)
            fields[trim(std::string_view(line).substr(0, equals))] = trim(std::string_view(line).substr(equals + 1));
    }
    return {fields.at("imDir"), fields.at("imExt"), number<std::uint64_t>(fields.at("seqLength")),
            number<int>(fields.at("imWidth")), number<int>(fields.at("imHeight"))};
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::invalid_argument("usage: track_sequence ENGINE SEQ_DIR OUT.txt [--serial] [--no-cuda-graph] [--packets OUT.bin] [--epoch N] [--profile NAME] [--config FILE] [--set KEY=VALUE] [--resolved-config FILE]");
        std::filesystem::path packets, motion_path, config_path, resolved_path;
        saccade::ConfigAssignments assignments;
        bool have_config = false, have_resolved = false;
        for (int i = 4; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--serial") assignments.emplace_back("execution.mode", "serial");
            else if (arg == "--no-cuda-graph") assignments.emplace_back("execution.cuda_graph", "false");
            else if (arg == "--packets" && i + 1 < argc) packets = argv[++i];
            else if (arg == "--motion" && i + 1 < argc) motion_path = argv[++i];
            else if (arg == "--epoch" && i + 1 < argc) assignments.emplace_back("source.epoch", argv[++i]);
            else if (arg == "--profile" && i + 1 < argc) assignments.emplace_back("implementation.model_profile", argv[++i]);
            else if (arg == "--set" && i + 1 < argc) assignments.push_back(saccade::tracking_cli_assignment(argv[++i]));
            else if (arg == "--config" && i + 1 < argc) {
                if (have_config) throw std::invalid_argument("duplicate --config");
                have_config = true; config_path = argv[++i];
            } else if (arg == "--resolved-config" && i + 1 < argc) {
                if (have_resolved) throw std::invalid_argument("duplicate --resolved-config");
                have_resolved = true; resolved_path = argv[++i];
            }
            else throw std::invalid_argument("unknown or incomplete argument: " + std::string(arg));
        }
        const std::filesystem::path sequence = argv[2];
        const auto info = read_sequence_info(sequence / "seqinfo.ini");
        if (info.width <= 0 || info.height <= 0) throw std::invalid_argument("invalid sequence dimensions");
        saccade::ResolvedTrackingConfig defaults;
        defaults.source.width = static_cast<std::uint32_t>(info.width);
        defaults.source.height = static_cast<std::uint32_t>(info.height);
        defaults.preprocess.max_width = static_cast<std::uint32_t>(std::max(1920, info.width));
        defaults.preprocess.max_height = static_cast<std::uint32_t>(std::max(1080, info.height));
        defaults.implementation.backend = "cuda"; defaults.implementation.input = "jpeg";
        defaults.implementation.motion_profile = saccade::kFlowMotionProfile;
        defaults.execution.mode = "pipelined"; defaults.execution.cuda_graph = true;
        const auto config_text = have_config ? saccade::read_text_file(config_path) : std::string{};
        auto resolved = saccade::resolve_tracking_config(
            config_text,
            saccade::declared_tracking_environment(), assignments, {{"cuda"}, {"jpeg"}}, defaults, "track_sequence");
        if (resolved.source.width != static_cast<std::uint32_t>(info.width) ||
            resolved.source.height != static_cast<std::uint32_t>(info.height))
            throw std::invalid_argument("resolved source dimensions differ from sequence metadata");
        if (resolved.output_profile != "v1-mot/1")
            throw std::invalid_argument("track_sequence requires v1-mot/1 output");
        saccade::mark_native_input_origins(resolved, "seqinfo.ini");
        const auto config = saccade::session_config_from_resolved(resolved);
        if (!resolved_path.empty()) {
            const auto normalized = [](const std::filesystem::path& path) {
                return std::filesystem::weakly_canonical(std::filesystem::absolute(path));
            };
            const auto destination = normalized(resolved_path);
            for (const auto& path : {std::filesystem::path(argv[1]), std::filesystem::path(argv[3]), packets,
                                     motion_path, config_path, sequence / "seqinfo.ini"})
                if (!path.empty() && destination == normalized(path))
                    throw std::invalid_argument("resolved config output must differ from inputs and other outputs");
            if (std::filesystem::symlink_status(resolved_path).type() != std::filesystem::file_type::not_found)
                throw std::invalid_argument("resolved config output already exists; choose a new path");
        }
        saccade::TrackingSession session(argv[1], config);
        std::cout << session.identity();
        const auto build_identity = saccade::tracking_build_identity(argv[1]) +
            "input.seqinfo_sha256=" + saccade::sha256_hex(saccade::read_text_file(sequence / "seqinfo.ini")) + '\n' +
            (have_config ? "input.config_file_sha256=" + saccade::sha256_hex(config_text) + '\n' : std::string{});
        const auto record = saccade::resolved_config_record(resolved, build_identity);
        std::cout << record;
        if (!resolved_path.empty()) saccade::write_resolved_config_file(resolved_path, record);
        saccade::SequenceOutput output(resolved.mot);
        saccade::MotionRecords motion{saccade::flow_image_key(info.width,info.height),config.epoch,{}};
        std::vector<std::uint8_t> raw;
        if (!packets.empty()) raw.reserve(info.frames * saccade::packet_bytes);
        const auto start = std::chrono::steady_clock::now();
        const auto receive = [&] {
            const auto result = session.receive();
            output.add(result.packet);
            if (output.failed()) throw std::runtime_error("tracking failed at frame " + std::to_string(result.packet.frame));
            if (!packets.empty()) raw.insert(raw.end(), result.bytes.begin(), result.bytes.end());
            if (!motion_path.empty()) motion.frames.push_back(result.motion);
        };
        try {
            for (std::uint64_t frame = 0; frame < info.frames; ++frame) {
                std::ostringstream filename;
                filename << std::setfill('0') << std::setw(6) << frame + 1 << info.extension;
                const auto jpeg = saccade::read_text_file(sequence / info.directory / filename.str());
                if (session.pending() == config.result_slots) receive();
                session.enqueue({reinterpret_cast<const std::uint8_t*>(jpeg.data()), jpeg.size()}, frame);
            }
            session.finish();
            while (session.pending()) receive();
        } catch (...) { session.close(); throw; }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        session.close();
        saccade::write_mot_file(argv[3], output.finish());
        if (!packets.empty()) saccade::write_binary_file(packets, raw);
        if (!motion_path.empty()) saccade::write_motion_records(motion_path, motion);
        std::cout << "track_sequence frames=" << info.frames << " mode="
                  << (config.mode == saccade::TrackingSessionMode::Serial ? "serial" : "pipelined")
                  << " wall_ms=" << ms << " ms_per_frame=" << ms / info.frames << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "track_sequence: " << e.what() << '\n';
        return 1;
    }
}

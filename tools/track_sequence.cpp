#include "saccade/tracking_session.hpp"
#include "saccade/file_io.hpp"
#include "saccade/mot_output.hpp"

#include <charconv>
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
        if (argc < 4) throw std::invalid_argument("usage: track_sequence ENGINE SEQ_DIR OUT.txt [--serial] [--packets OUT.bin] [--epoch N]");
        saccade::TrackingSessionConfig config;
        std::filesystem::path packets;
        for (int i = 4; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--serial") config.mode = saccade::TrackingSessionMode::Serial;
            else if (arg == "--packets" && i + 1 < argc) packets = argv[++i];
            else if (arg == "--epoch" && i + 1 < argc) config.epoch = number<std::uint64_t>(argv[++i]);
            else throw std::invalid_argument("unknown or incomplete argument: " + std::string(arg));
        }
        const std::filesystem::path sequence = argv[2];
        const auto info = read_sequence_info(sequence / "seqinfo.ini");
        config.original_width = info.width; config.original_height = info.height;
        saccade::TrackingSession session(argv[1], config);
        std::cout << session.identity();
        saccade::SequenceOutput output;
        std::vector<std::uint8_t> raw;
        if (!packets.empty()) raw.reserve(info.frames * saccade::packet_bytes);
        const auto start = std::chrono::steady_clock::now();
        const auto receive = [&] {
            const auto result = session.receive();
            output.add(result.packet);
            if (output.failed()) throw std::runtime_error("tracking failed at frame " + std::to_string(result.packet.frame));
            if (!packets.empty()) raw.insert(raw.end(), result.bytes.begin(), result.bytes.end());
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
        std::cout << "track_sequence frames=" << info.frames << " mode="
                  << (config.mode == saccade::TrackingSessionMode::Serial ? "serial" : "pipelined")
                  << " wall_ms=" << ms << " ms_per_frame=" << ms / info.frames << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "track_sequence: " << e.what() << '\n';
        return 1;
    }
}

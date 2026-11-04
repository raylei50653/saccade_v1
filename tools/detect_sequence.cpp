#include "saccade/detection_session.hpp"
#include "saccade/file_io.hpp"
#include "saccade/mot_output.hpp"

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
    std::filesystem::path output, temporary;
    try {
        const auto start = std::chrono::steady_clock::now();
        if (argc < 4 || argc % 2 != 0)
            throw std::invalid_argument("usage: detect_sequence ENGINE SEQ_DIR OUT.scdr [--epoch N] [--profile NAME]");
        output = argv[3];
        temporary = output.string() + ".tmp";
        saccade::DetectionSessionConfig config;
        for (int i = 4; i < argc; i += 2) {
            const std::string_view flag = argv[i];
            if (flag == "--epoch") config.epoch = number<std::uint64_t>(argv[i + 1]);
            else if (flag == "--profile") config.profile = saccade::parse_model_profile(argv[i + 1]);
            else throw std::invalid_argument("unexpected option: " + std::string(flag));
        }
        const std::filesystem::path sequence = argv[2];
        const auto info = read_sequence_info(sequence / "seqinfo.ini");
        config.preprocess.max_width = std::max(1920, info.width);
        config.preprocess.max_height = std::max(1080, info.height);
        saccade::DetectionRecords records{config.epoch, {}};
        std::uint64_t kept = 0, suppressed = 0;
        saccade::DetectionSession session(argv[1], config);
        const auto receive = [&] {
            const auto result = session.receive();
            if (result.status != saccade::DetectionSessionStatus::Ok || result.frame != records.frames.size())
                throw std::runtime_error("failed or out-of-order detection result");
            records.frames.push_back(result.candidates);
            kept += result.candidates.header.kept;
            suppressed += result.candidates.header.suppressed;
        };
        try {
            for (std::uint64_t frame = 0; frame < info.frames; ++frame) {
                std::ostringstream filename;
                filename << std::setfill('0') << std::setw(6) << frame + 1 << info.extension;
                const auto jpeg = saccade::read_text_file(sequence / info.directory / filename.str());
                for (;;) {
                    try {
                        session.enqueue({reinterpret_cast<const std::uint8_t*>(jpeg.data()), jpeg.size()},
                                        frame, info.width, info.height);
                        break;
                    } catch (const std::length_error&) {
                        receive();
                    }
                }
            }
            session.finish();
            while (session.pending() != 0) receive();
        } catch (...) {
            session.close();
            throw;
        }
        session.close();
        saccade::write_detection_records(temporary, records);
        std::filesystem::rename(temporary, output);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::cout << "detect_sequence: frames=" << records.frames.size() << " kept=" << kept
                  << " suppressed=" << suppressed << " wall_seconds=" << std::fixed << std::setprecision(3)
                  << seconds << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::error_code ignored;
        if (!temporary.empty()) std::filesystem::remove(temporary, ignored);
        if (!output.empty()) std::filesystem::remove(output, ignored);
        std::cerr << "detect_sequence: " << error.what() << '\n';
        return 1;
    }
}

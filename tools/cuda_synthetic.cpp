#include "saccade/synthetic_runtime.hpp"
#include "saccade/file_io.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
std::string snapshot_text(const saccade::SyntheticSnapshot& state) {
    std::ostringstream text;
    text << "current_frame=" << state.current_frame << '\n'
         << "committed_frame=" << state.committed_frame << '\n'
         << "fail_frame=" << state.fail_frame << '\n'
         << "fail_code=" << state.fail_code << '\n'
         << "committed_slot=" << state.committed_slot << '\n'
         << "tracks_frame=" << state.tracks_frame << '\n'
         << "spectrum_frame=" << state.spectrum_frame << '\n'
         << "scratch_frame=" << state.scratch_frame << '\n'
         << "predict_count=" << state.predict_count << '\n'
         << "update_count=" << state.update_count << '\n'
         << "input_sum=" << state.input_sum << '\n'
         << "fft_completed=" << state.fft_completed << '\n'
         << "spectrum_bits=";
    for (std::size_t i = 0; i < state.spectrum_bits.size(); ++i) {
        if (i != 0) text << ',';
        text << std::hex << std::setw(8) << std::setfill('0') << state.spectrum_bits[i];
    }
    text << '\n' << "tracks_bytes=";
    for (const auto byte : state.tracks_bytes)
        text << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    text << '\n';
    return text.str();
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: cuda_synthetic CONFIG OUTPUT_DIRECTORY\n";
        return 2;
    }
    try {
        const auto config = saccade::parse_config(saccade::read_text_file(argv[1]));
        const std::filesystem::path directory(argv[2]);
        std::filesystem::create_directories(directory);
        saccade::write_text_file(directory / "resolved.cfg", saccade::canonical_config(config));
        saccade::write_text_file(directory / "behavior.cfg", saccade::behavior_config(config));
        saccade::write_text_file(directory / "config.sha256", saccade::config_hash(config) + "\n");
        saccade::write_text_file(directory / "behavior.sha256", saccade::behavior_hash(config) + "\n");
        const auto run = saccade::run_synthetic(config);
        // Preserve exactly the single-copy GPU output before any decoding. The
        // logical CPU encoder is used only by independent test expectations.
        saccade::write_binary_file(directory / "packets.bin", run.packets);
        saccade::write_text_file(directory / "identity.txt", run.identity);
        std::ostringstream audit;
        audit << "d_frames=" << run.d_frames << '\n'
              << "m_frames=" << run.m_frames << '\n'
              << "packet_copies=" << run.packet_copies << '\n'
              << "d_drained=" << std::boolalpha << run.d_drained << '\n'
              << "m_drained=" << run.m_drained << '\n'
              << "stopped_on_failure=" << run.stopped_on_failure << '\n'
              << "cancelled=" << run.cancelled << '\n';
        for (const auto& line : run.ledger) audit << line << '\n';
        saccade::write_text_file(directory / "audit.txt", audit.str());
        if (config.terminal_snapshot)
            saccade::write_text_file(directory / "terminal-snapshot.txt", snapshot_text(run.snapshot));
        if (run.packets.size() % saccade::packet_bytes != 0)
            throw std::runtime_error("runtime returned truncated GPU packet bytes");
        std::size_t failures = 0, skipped = 0;
        for (std::size_t offset = 0; offset < run.packets.size(); offset += saccade::packet_bytes) {
            const auto packet = saccade::decode_packet(
                std::span<const std::uint8_t>(run.packets).subspan(offset, saccade::packet_bytes));
            failures += packet.status == saccade::ResultStatus::failed;
            skipped += packet.status == saccade::ResultStatus::skipped_after_failure;
        }
        std::cout << "packets=" << run.packets.size() / saccade::packet_bytes
                  << " bytes=" << run.packets.size() << " failed=" << failures
                  << " skipped=" << skipped << " d_drained=" << run.d_drained
                  << " m_drained=" << run.m_drained << '\n'
                  << "config_sha256=" << saccade::config_hash(config) << '\n'
                  << "behavior_sha256=" << saccade::behavior_hash(config) << '\n';
        return failures != 0 || run.cancelled ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_synthetic: " << error.what() << '\n';
        return 2;
    }
}

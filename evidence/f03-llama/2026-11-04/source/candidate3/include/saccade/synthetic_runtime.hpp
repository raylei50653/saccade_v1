#pragma once
#include "saccade/synthetic_config.hpp"
#include "saccade/protocol.hpp"
#include <string>
#include <vector>

namespace saccade {
struct SyntheticSnapshot {
    FrameId current_frame = -1, committed_frame = -1, fail_frame = -1;
    std::uint32_t fail_code = 0, committed_slot = 1;
    FrameId tracks_frame = -1, spectrum_frame = -1, scratch_frame = -1;
    std::uint64_t predict_count = 0, update_count = 0, input_sum = 0;
    // Raw float bits of the entire committed spectrum, separate terminal-only readback.
    std::vector<std::uint32_t> spectrum_bits;
    // Complete internal TS payload for same-build terminal isolation checks.
    // This is diagnostic evidence only, never a wire format.
    std::vector<std::uint8_t> tracks_bytes;
    std::uint64_t fft_completed = 0;
};
struct SyntheticRun {
    // Original full GPU D2H bytes, never reconstructed on host.
    std::vector<std::uint8_t> packets;
    SyntheticSnapshot snapshot;
    int d_frames = 0, m_frames = 0;
    bool stopped_on_failure = false, cancelled = false;
    bool d_drained = false, m_drained = false;
    std::size_t packet_copies = 0;
    std::vector<std::string> ledger;
    std::string identity;
};
SyntheticRun run_synthetic(const SyntheticConfig& config);
} // namespace saccade

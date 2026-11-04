#pragma once

#include "saccade/sha256.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace saccade {

// Schema 1: synthetic fixtures alternate 0 candidates (even frame) and 2
// candidates (odd frame). This profile is deliberately separate from P08.
struct SyntheticConfig {
    std::string mode = "pipelined";
    int n_mi = 2, n_cs = 2, n_h = 2, n_rp = 2;
    int width = 8, height = 8, frames = 16;
    std::uint64_t epoch = 1, seed = 7, delay_seed = 11;
    std::uint32_t delay_cycles = 20000;
    int fail_frame = -1;
    std::string fail_stage = "none";
    int cancel_frame = -1;
    std::string cancel_cut = "none";
    bool fixed_submit = false, terminal_snapshot = false;
    bool operator==(const SyntheticConfig&) const = default;
};

// ASCII key=value records, LF terminated (final LF optional), no whitespace,
// comments, quoting, or implicit conversions. Empty records are ignored.
// Missing fields receive the typed defaults above. Duplicate/unknown fields
// and noncanonical integer spellings are rejected. Optional schema must be 1.
SyntheticConfig parse_config(std::string_view input);
void validate_config(const SyntheticConfig& config);

// Fixed key order, all defaults explicit, schema=1 first, decimal integers,
// lowercase bool/tokens and one LF per record including the final record.
std::string canonical_config(const SyntheticConfig& config);
// Comparison allowlist removes only mode, n_mi/n_cs/n_h/n_rp, delay_seed,
// delay_cycles and terminal_snapshot. All other fields must match.
std::string behavior_config(const SyntheticConfig& config);
std::string config_hash(const SyntheticConfig& config);
std::string behavior_hash(const SyntheticConfig& config);

} // namespace saccade

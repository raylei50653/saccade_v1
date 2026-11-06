#pragma once

#include "saccade/model_profile.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/tracking.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace saccade {

inline constexpr std::string_view kResolvedConfigSchema = "saccade-tracking-config/1";
enum class ConfigCategory { Behavior, Implementation, Execution, Observer };
enum class ConfigType { Boolean, Unsigned, Float32, Choice };
struct ConfigField {
    std::string key;
    ConfigCategory category;
    ConfigType type;
    std::string owner, behavior, purpose, default_value, constraint, environment;
};
struct ConfigOrigin {
    std::string key, layer, name;
};
// Supplied by the calling build, never inferred from the current environment.
struct ConfigCapabilities {
    std::vector<std::string> backends{"reference"};
    std::vector<std::string> inputs{"records"};
};
struct ResolvedTrackingConfig {
    TrackerConfig tracker{};
    struct Source {
        std::uint64_t id = 1;
        std::uint64_t epoch = 1;
        std::uint32_t width = 1920, height = 1080;
    } source;
    struct Preprocess {
        std::uint32_t max_width = 1920, max_height = 1080;
        std::uint64_t max_jpeg_bytes = 16 * 1024 * 1024;
    } preprocess;
    struct Implementation {
        std::string backend = "reference", input = "records";
        ModelProfile model_profile = kDefaultModelProfile;
        std::string motion_profile = "recorded";
    } implementation;
    struct Execution {
        std::string mode = "serial";
        std::uint32_t device = 0, candidate_slots = 2, motion_slots = 2, result_slots = 4;
        bool cuda_graph = false;
        std::uint64_t perturb_seed = 0;
    } execution;
    struct Observer { bool event_dump = false; } observer;
    std::string output_profile = "v1-mot/1";
    MotOutputConfig mot{};
    std::vector<ConfigOrigin> origins;
};

using ConfigAssignments = std::vector<std::pair<std::string, std::string>>;
// Lexical parsing only; resolution below checks schema, duplicates and values.
ConfigAssignments parse_tracking_config_file(std::string_view text);
// File: ASCII key=value records, optional surrounding spaces/tabs and blank or
// whole-line # comments. Any nonempty file requires schema=<version>.
// Overrides use the same typed values. Environment pairs MUST contain only the
// declared SACCADE_CFG_* names; callers pass a filtered namespace, not environ.
// Precedence: defaults < file < environment < CLI. Duplicates within each layer
// are errors, including repeated aliases; higher layers may override lower ones.
// No function reads the process environment. All floats must be finite FP32.
ResolvedTrackingConfig resolve_tracking_config(
    std::string_view file = {}, const ConfigAssignments& environment = {},
    const ConfigAssignments& cli = {}, const ConfigCapabilities& capabilities = {},
    const ResolvedTrackingConfig& defaults = {}, std::string_view defaults_name = "defaults");
std::vector<ConfigField> tracking_config_schema();
void validate_resolved_config(const ResolvedTrackingConfig& config,
                              const ConfigCapabilities& capabilities = {});
// Fixed field order, LF endings, uint decimal, bool true/false, enum tokens and
// float32 bits f32:0xXXXXXXXX. Sources/build identity are excluded from the hash.
std::string canonical_resolved_config(const ResolvedTrackingConfig& config);
std::string resolved_config_hash(const ResolvedTrackingConfig& config);
// Adds provenance after the canonical content. A build identity is required;
// config equality never substitutes for build/source/binary equality.
std::string resolved_config_record(const ResolvedTrackingConfig& config, std::string_view build_identity);
struct ParsedResolvedConfigRecord {
    ResolvedTrackingConfig config;
    std::string build_identity;
};
// Requires the exact canonical config prefix, matching SHA256, record schema,
// build identity, and exactly one provenance record for every schema field.
ParsedResolvedConfigRecord parse_resolved_config_record(
    std::string_view record, const ConfigCapabilities& capabilities = {});
struct ConfigDifference { std::string key, expected, actual; };
// Only explicitly listed execution/probe fields may differ. Behavior and
// implementation differences always remain visible; invalid allowlists throw.
std::vector<ConfigDifference> compare_resolved_config(
    const ResolvedTrackingConfig& expected, const ResolvedTrackingConfig& actual,
    const std::vector<std::string>& allowed_differences = {});

} // namespace saccade

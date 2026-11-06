#include "saccade/resolved_config.hpp"

#include "saccade/flow_config.hpp"
#include "saccade/sha256.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace saccade {
namespace {
[[noreturn]] void invalid(std::string_view key, std::string_view why) {
    throw std::invalid_argument("resolved config " + std::string(key) + ": " + std::string(why));
}
std::string_view trim(std::string_view s) {
    const auto first = s.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r") - first + 1);
}
std::string hex32(std::uint32_t bits) {
    constexpr char digits[] = "0123456789abcdef";
    std::string s(8, '0');
    for (unsigned i = 0; i < 8; ++i) s[7 - i] = digits[(bits >> (4 * i)) & 15u];
    return s;
}
template<class T> std::string value_text(const T& value) {
    if constexpr (std::is_same_v<T, bool>) return value ? "true" : "false";
    else if constexpr (std::is_same_v<T, float>) return "f32:0x" + hex32(std::bit_cast<std::uint32_t>(value));
    else if constexpr (std::is_integral_v<T>) return std::to_string(value);
    else return value;
}
template<class T> T parse_value(std::string_view key, std::string_view text) {
    if constexpr (std::is_same_v<T, bool>) {
        if (text == "true") return true;
        if (text == "false") return false;
        invalid(key, "expected true or false");
    } else if constexpr (std::is_same_v<T, float>) {
        float result{};
        if (text.starts_with("f32:0x")) {
            if (text.size() != 14) invalid(key, "expected eight float32 hex digits");
            std::uint32_t bits{};
            const auto [end, ec] = std::from_chars(text.data() + 6, text.data() + text.size(), bits, 16);
            if (ec != std::errc{} || end != text.data() + text.size()) invalid(key, "invalid float32 bits");
            result = std::bit_cast<float>(bits);
        } else {
            const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), result,
                                                   std::chars_format::general);
            if (ec != std::errc{} || end != text.data() + text.size() || text.empty())
                invalid(key, "expected finite float32 number");
        }
        if (!std::isfinite(result)) invalid(key, "non-finite float32");
        return result;
    } else if constexpr (std::is_integral_v<T>) {
        if (text.empty() || !std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }))
            invalid(key, "expected unsigned decimal integer");
        T result{};
        const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), result);
        if (ec != std::errc{} || end != text.data() + text.size()) invalid(key, "integer out of range");
        return result;
    } else return std::string(text);
}
bool contains(const std::vector<std::string>& choices, std::string_view v) {
    return std::find(choices.begin(), choices.end(), v) != choices.end();
}
std::string environment_name(std::string_view key) {
    std::string s = "SACCADE_CFG_";
    for (char c : key) s += c == '.' ? '_' : c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : c;
    return s;
}
struct Binding {
    ConfigField field;
    std::function<std::string(const ResolvedTrackingConfig&)> get;
    std::function<void(ResolvedTrackingConfig&, std::string_view)> set;
};
const std::vector<Binding>& bindings() {
    static const auto fields = [] {
        std::vector<Binding> result;
        const ResolvedTrackingConfig defaults;
        const auto add = [&](std::string key, ConfigCategory category, std::string owner, std::string behavior,
                             std::string purpose, std::string constraint, auto access) {
            using T = std::remove_cvref_t<decltype(access(std::declval<ResolvedTrackingConfig&>()))>;
            constexpr auto type = std::is_same_v<T, bool> ? ConfigType::Boolean :
                std::is_same_v<T, float> ? ConfigType::Float32 :
                std::is_integral_v<T> ? ConfigType::Unsigned : ConfigType::Choice;
            const auto get = [access](const ResolvedTrackingConfig& c) { return value_text(access(c)); };
            const auto set = [access, key](ResolvedTrackingConfig& c, std::string_view text) {
                access(c) = parse_value<T>(key, text);
            };
            ConfigField meta{key, category, type, std::move(owner), std::move(behavior), std::move(purpose),
                             get(defaults), std::move(constraint), environment_name(key)};
            result.push_back({std::move(meta), get, set});
        };
#define ADD(key, category, owner, behavior, purpose, constraint, member) \
        add(key, ConfigCategory::category, owner, behavior, purpose, constraint, \
            [](auto& c) -> auto& { return c.member; })
#define TRACK(member, owner, behavior, constraint) \
        ADD("tracker." #member, Behavior, owner, behavior, #member, constraint, tracker.member)
        TRACK(kalman.r_scale, "F08a", "B07", "finite >0");
        TRACK(kalman.std_position, "F08a", "B07", "finite >0");
        TRACK(kalman.std_velocity, "F08a", "B07", "finite >0");
        TRACK(batch.track_thresh, "F04b", "B06", "[0,1]; <=assoc.mid_thresh<=assoc.high_thresh");
        TRACK(continuation.enabled, "F04b", "B06", "boolean");
        TRACK(continuation.min_score, "F04b", "B06", "[0,1]");
        TRACK(continuation.prior_iou, "F04b", "B06", "[0,1]");
        TRACK(continuation.dedup_iou, "F04b", "B06", "[0,1]");
        TRACK(continuation.max_candidates, "F04b", "B06", "[0,300]");
        TRACK(assoc.high_thresh, "F07", "B07", "[0,1]; >=assoc.mid_thresh");
        TRACK(assoc.mid_thresh, "F07", "B07", "[0,1]; >=batch.track_thresh");
        TRACK(assoc.match_thresh, "F07", "B07", "[0,1]");
        TRACK(assoc.stage2_match_thresh, "F07", "B07", "[0,1]");
        TRACK(assoc.lambda, "F07", "B07", "finite >0");
        TRACK(assoc.stability_cost_w, "F07", "B07", "finite >=0");
        TRACK(assoc.top_k, "F07", "B07", "[1,8]");
        TRACK(assoc.auction_eps, "F07", "B07", "finite >0");
        TRACK(assoc.stability_bid_w, "F07", "B07", "finite >=0");
        TRACK(life.new_track_thresh, "F08b", "B08", "[0,1]");
        TRACK(life.confirm_streak, "F08b", "B08", "uint32 >=1");
        TRACK(life.confirm_score, "F08b", "B08", "[0,1]");
        TRACK(life.track_buffer, "F08a", "B08", "uint32 >=1");
        TRACK(bridge.enabled, "F08b", "B08", "boolean");
        TRACK(bridge.trigger_streak, "F08b", "B08", "uint32 >=4");
        TRACK(bridge.min_lost, "F08b", "B08", "uint32 >=1");
        TRACK(bridge.max_distance, "F08b", "B08", "finite >=0");
        TRACK(bridge.height_ratio_lo, "F08b", "B08", "0<lo<hi");
        TRACK(bridge.height_ratio_hi, "F08b", "B08", "hi>lo>0");
        TRACK(bridge.margin, "F08b", "B08", "finite >=0");
        TRACK(bridge.speed_norm, "F08b", "B08", "finite >0");
#undef TRACK
        ADD("source.id", Behavior, "caller", "B01", "stable source identity", "uint64", source.id);
        ADD("source.epoch", Behavior, "caller", "B01", "state reset epoch", "uint64", source.epoch);
        ADD("source.width", Behavior, "F01", "B02", "input width", "[1,8192]", source.width);
        ADD("source.height", Behavior, "F01", "B02", "input height", "[1,8192]", source.height);
        ADD("preprocess.max_width", Behavior, "F01", "B02", "image allocation and size rejection; inactive for records", "[1,8192]; >=source.width", preprocess.max_width);
        ADD("preprocess.max_height", Behavior, "F01", "B02", "image allocation and size rejection; inactive for records", "[1,8192]; >=source.height", preprocess.max_height);
        ADD("preprocess.max_jpeg_bytes", Behavior, "F01", "B02", "compressed input capacity; inactive for records", "[1,268435456]", preprocess.max_jpeg_bytes);
        ADD("implementation.backend", Implementation, "build", "B03", "tracker implementation", "reference|cuda; build availability", implementation.backend);
        ADD("implementation.input", Implementation, "F01", "B02", "input and decode implementation", "records|jpeg|host-rgb|device-video; build availability", implementation.input);
        // ModelProfile is an enum rather than a configurable free-form string.
        result.push_back({{"implementation.model_profile", ConfigCategory::Implementation, ConfigType::Choice,
            "F03", "B03", "model shape; upstream declaration only for records", std::string(kT01ModelProfile.name),
            "t01-640x640|hr-1280x768", environment_name("implementation.model_profile")},
            [](const ResolvedTrackingConfig& c) { return std::string(model_profile(c.implementation.model_profile).name); },
            [](ResolvedTrackingConfig& c, std::string_view v) { c.implementation.model_profile = parse_model_profile(v); }});
        ADD("implementation.motion_profile", Implementation, "F06", "B05", "recorded, disabled or fixed G1 motion", "recorded|disabled|v1-motion/2", implementation.motion_profile);
        ADD("execution.mode", Execution, "session", "B10", "host scheduling", "serial|pipelined", execution.mode);
        ADD("execution.device", Execution, "session", "B10", "CUDA device ordinal; inactive for reference", "[0,INT_MAX]", execution.device);
        ADD("execution.candidate_slots", Execution, "session", "B10", "candidate ring size; inactive for reference", "uint32 >=1", execution.candidate_slots);
        ADD("execution.motion_slots", Execution, "session", "B10", "motion ring size; inactive for reference", "uint32 >=1", execution.motion_slots);
        ADD("execution.result_slots", Execution, "session", "B10", "result ring size; inactive for reference", "uint32 >=1", execution.result_slots);
        ADD("execution.cuda_graph", Execution, "session", "B10", "CUDA Graph launch implementation", "boolean", execution.cuda_graph);
        ADD("execution.perturb_seed", Observer, "session", "V04", "diagnostic schedule perturbation only", "uint64", execution.perturb_seed);
        ADD("observer.event_dump", Observer, "observer", "E09", "diagnostic event observation only", "boolean; complete tracker dump only for reference backend", observer.event_dump);
        ADD("output.profile", Implementation, "F09", "B09", "MOT formatting and identity policy", "v1-mot/1|stream-observations/1", output_profile);
        ADD("output.interpolate_max_gap", Behavior, "F09", "B09", "offline interpolation maximum missing frames", "uint32; stream profile requires 0", mot.interpolate_max_gap);
        ADD("output.interpolate_min_rows", Behavior, "F09", "B09", "offline interpolation minimum observed rows", "uint32 >=1", mot.interpolate_min_rows);
#undef ADD
        const auto fixed = [&](std::string key, std::string owner, std::string behavior, std::string purpose,
                               std::function<std::string(const ResolvedTrackingConfig&)> get) {
            result.push_back({{key, ConfigCategory::Implementation, ConfigType::Choice, std::move(owner),
                std::move(behavior), std::move(purpose), get(defaults), "fixed or derived; cannot override", ""},
                std::move(get), {}});
        };
        fixed("policy.tracker", "F08", "B07", "tracker semantics", [](const auto&) { return kTrackerProfile; });
        fixed("policy.postprocess", "F04", "B06", "candidate policy", [](const auto&) { return kPostprocessPolicy; });
        fixed("policy.packet", "F09", "B09", "packet wire format", [](const auto&) { return "ResultPacket/1"; });
        fixed("policy.precision", "F03-F08", "B03", "runtime arithmetic; engine internal precision requires separate engine provenance", [](const auto& c) {
            return c.implementation.backend == "reference" ? "reference-fp32-tracker;recorded-model-precision-external-receipt" :
                "fp32-input-output-scan-tracker;engine-internal-external-receipt";
        });
        fixed("policy.input", "F01", "B02", "input and color contract", [](const auto& c) {
            if (c.implementation.input == "device-video") return "device-video-input/1+device-video-color/1";
            if (c.implementation.input == "host-rgb") return "host-rgb-input/1";
            if (c.implementation.input == "jpeg") return "nvjpeg-hardware-ycbcr420/1";
            return c.implementation.motion_profile == "disabled" ? "SCDR/1+disabled-motion" : "SCDR/1+SCMO/1";
        });
        fixed("policy.resize", "F02", "B04", "resize semantics; records bypass F01-F03", [](const auto& c) {
            return c.implementation.input == "records" ? "not-applicable-recorded-detections" : "t01-input/1";
        });
        fixed("capacity.tracks", "F08", "P04", "fixed track capacity", [](const auto&) { return std::to_string(kTrackCapacity); });
        fixed("capacity.candidates", "F04", "P04", "fixed candidate and batch capacity", [](const auto&) { return std::to_string(kPostprocessCapacity); });
        fixed("capacity.packet_rows", "F09", "P04", "fixed packet row capacity", [](const auto&) { return std::to_string(packet_capacity); });
        fixed("flow.policy", "F06", "B05", "fixed G1 policy when native motion is selected", [](const auto& c) {
            if (c.implementation.motion_profile != "v1-motion/2") return std::string("not-applicable");
            return std::string(kFlowMotionProfile) + ":reference-" + std::to_string(FlowConfig::reference_width) + 'x' +
                std::to_string(FlowConfig::reference_height) + ":grid-" + std::to_string(FlowConfig::grid) +
                ":cell-" + std::to_string(FlowConfig::cell) + ":min_cells-" + std::to_string(FlowConfig::min_cells) +
                ":mask_score-" + value_text(FlowConfig::mask_score) + ":perf-" + FlowConfig::perf;
        });
        return result;
    }();
    return fields;
}
const Binding& find_binding(std::string_view key) {
    for (const auto& b : bindings()) if (b.field.key == key) return b;
    invalid(key, "unknown field");
}
void validate_common(const ResolvedTrackingConfig& c) {
    validate_tracker_config(c.tracker);
    if (c.implementation.model_profile != ModelProfile::T01_640x640 && c.implementation.model_profile != ModelProfile::Hr1280x768)
        invalid("implementation.model_profile", "invalid enum");
    if (!contains({"reference", "cuda"}, c.implementation.backend)) invalid("implementation.backend", "unsupported backend");
    if (!contains({"records", "jpeg", "host-rgb", "device-video"}, c.implementation.input)) invalid("implementation.input", "unsupported input");
    if (!contains({"recorded", "disabled", "v1-motion/2"}, c.implementation.motion_profile)) invalid("implementation.motion_profile", "unsupported motion profile");
    if (c.implementation.backend == "reference") {
        if (c.implementation.input != "records") invalid("implementation.input", "reference backend requires records");
        if (c.implementation.motion_profile == "v1-motion/2") invalid("implementation.motion_profile", "reference consumes recorded or disabled motion");
        if (c.execution.mode != "serial" || c.execution.cuda_graph || c.execution.perturb_seed != 0)
            invalid("execution", "reference backend only provides serial execution without CUDA Graph or perturbation");
    } else {
        if (c.implementation.input == "records") invalid("implementation.input", "native session requires an image input");
        if (c.implementation.motion_profile != "v1-motion/2") invalid("implementation.motion_profile", "native session only provides fixed G1");
        if (c.observer.event_dump) invalid("observer.event_dump", "complete tracker event dump unavailable for CUDA backend");
        if (c.source.width < 32 || c.source.height < 32) invalid("source", "native G1 image dimensions must be at least 32");
        if (c.preprocess.max_width != std::max(1920u, c.source.width) ||
            c.preprocess.max_height != std::max(1080u, c.source.height))
            invalid("preprocess", "native session fixes max dimensions to max(default, source)");
    }
    if (!contains({"serial", "pipelined"}, c.execution.mode)) invalid("execution.mode", "expected serial or pipelined");
    if (c.execution.device > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) invalid("execution.device", "exceeds signed device ordinal");
    if (c.execution.candidate_slots == 0 || c.execution.motion_slots == 0 || c.execution.result_slots == 0)
        invalid("execution", "ring slots must be positive");
    if (c.source.width == 0 || c.source.width > 8192 || c.source.height == 0 || c.source.height > 8192)
        invalid("source", "dimensions must be in 1..8192");
    if (c.preprocess.max_width < c.source.width || c.preprocess.max_width > 8192 ||
        c.preprocess.max_height < c.source.height || c.preprocess.max_height > 8192)
        invalid("preprocess", "max dimensions must contain source and be <=8192");
    if (c.preprocess.max_jpeg_bytes == 0 || c.preprocess.max_jpeg_bytes > 256 * 1024 * 1024)
        invalid("preprocess.max_jpeg_bytes", "expected 1..256MiB");
    if (!contains({"v1-mot/1", "stream-observations/1"}, c.output_profile)) invalid("output.profile", "unsupported output profile");
    if (c.mot.interpolate_min_rows == 0) invalid("output.interpolate_min_rows", "must be positive");
    if (c.output_profile == "stream-observations/1" && c.mot.interpolate_max_gap != 0)
        invalid("output.interpolate_max_gap", "stream output has no interpolation; must be zero");
}
ConfigAssignments parse_file(std::string_view text) {
    ConfigAssignments result;
    for (unsigned char c : text)
        if (c > 126 || (c < 32 && c != '\n' && c != '\r' && c != '\t')) invalid("syntax", "expected printable ASCII records");
    while (!text.empty()) {
        const auto nl = text.find('\n');
        const auto line = trim(text.substr(0, nl));
        if (nl == std::string_view::npos) text = {}; else text.remove_prefix(nl + 1);
        if (line.empty() || line.front() == '#') continue;
        const auto eq = line.find('=');
        if (eq == std::string_view::npos || line.find('=', eq + 1) != std::string_view::npos)
            invalid("syntax", "expected one key=value record");
        const auto key = trim(line.substr(0, eq)), value = trim(line.substr(eq + 1));
        if (key.empty() || value.empty()) invalid("syntax", "empty key or value");
        result.emplace_back(key, value);
    }
    return result;
}
std::string escaped(std::string_view value) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (unsigned char c : value) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
            c == '.' || c == '-' || c == '_' || c == '/') result += static_cast<char>(c);
        else { result += '%'; result += digits[c >> 4]; result += digits[c & 15]; }
    }
    return result;
}
std::string unescaped(std::string_view value) {
    std::string result;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '%') { result += value[i]; continue; }
        if (i + 2 >= value.size()) invalid("record", "truncated percent escape");
        unsigned byte{};
        const auto [end, ec] = std::from_chars(value.data() + i + 1, value.data() + i + 3, byte, 16);
        if (ec != std::errc{} || end != value.data() + i + 3) invalid("record", "invalid percent escape");
        result += static_cast<char>(byte);
        i += 2;
    }
    if (escaped(result) != value) invalid("record", "noncanonical metadata escape");
    return result;
}
void validate_origin(const ConfigOrigin& origin) {
    const auto& binding = find_binding(origin.key);
    if (!contains({"default", "derived", "file", "environment", "cli", "typed"}, origin.layer) || origin.name.empty())
        invalid(origin.key, "invalid provenance layer or name");
    if (!binding.set && origin.layer != "derived" && origin.layer != "typed")
        invalid(origin.key, "fixed policy provenance must be derived or typed");
}
} // namespace

ConfigAssignments parse_tracking_config_file(std::string_view text) { return parse_file(text); }
std::vector<ConfigField> tracking_config_schema() {
    std::vector<ConfigField> result;
    for (const auto& b : bindings()) result.push_back(b.field);
    return result;
}
void validate_resolved_config(const ResolvedTrackingConfig& c, const ConfigCapabilities& capabilities) {
    validate_common(c);
    if (!contains(capabilities.backends, c.implementation.backend)) invalid("implementation.backend", "not provided by this build");
    if (!contains(capabilities.inputs, c.implementation.input)) invalid("implementation.input", "not provided by this build");
}
ResolvedTrackingConfig resolve_tracking_config(std::string_view file, const ConfigAssignments& environment,
    const ConfigAssignments& cli, const ConfigCapabilities& capabilities, const ResolvedTrackingConfig& defaults,
    std::string_view defaults_name) {
    auto c = defaults;
    c.origins.clear();
    for (const auto& b : bindings()) c.origins.push_back({b.field.key, b.set ? "default" : "derived", std::string(defaults_name)});
    std::map<std::string, std::string> derived_assertions;
    const auto apply = [&](const ConfigAssignments& pairs, std::string_view layer, bool env) {
        std::set<std::string> seen;
        bool schema = false;
        for (const auto& [name, raw] : pairs) {
            std::string key = name;
            if (env) {
                key.clear();
                for (const auto& b : bindings()) if (!b.field.environment.empty() && b.field.environment == name) {
                    key = b.field.key; break;
                }
                if (key.empty()) invalid(name, "undeclared environment override");
            }
            if (!seen.insert(key).second) invalid(key, "duplicate field in " + std::string(layer));
            if (key == "schema") {
                if (env || layer != "file") invalid(key, "schema may only appear in a file");
                if (raw != kResolvedConfigSchema) invalid(key, "unsupported schema");
                schema = true;
                continue;
            }
            const auto& b = find_binding(key);
            if (b.set) {
                b.set(c, raw);
                auto origin = std::find_if(c.origins.begin(), c.origins.end(), [&](const auto& o) { return o.key == key; });
                *origin = {key, std::string(layer), name};
            } else derived_assertions[key] = raw;
        }
        if (layer == "file" && !pairs.empty() && !schema) invalid("schema", "required in a nonempty config file");
    };
    apply(parse_file(file), "file", false);
    apply(environment, "environment", true);
    apply(cli, "cli", false);
    validate_resolved_config(c, capabilities);
    for (const auto& [key, expected] : derived_assertions)
        if (find_binding(key).get(c) != expected) invalid(key, "fixed/derived value does not match resolved configuration");
    return c;
}
std::string canonical_resolved_config(const ResolvedTrackingConfig& c) {
    validate_common(c);
    std::string result = "schema=" + std::string(kResolvedConfigSchema) + '\n';
    for (const auto& b : bindings()) result += b.field.key + '=' + b.get(c) + '\n';
    return result;
}
std::string resolved_config_hash(const ResolvedTrackingConfig& c) { return sha256_hex(canonical_resolved_config(c)); }
std::string resolved_config_record(const ResolvedTrackingConfig& c, std::string_view build_identity) {
    if (build_identity.empty()) invalid("build_identity", "required separately from config hash");
    std::string result = canonical_resolved_config(c);
    result += "record.schema=saccade-resolved-record/1\nconfig.sha256=" + resolved_config_hash(c) + '\n';
    result += "build.identity=" + escaped(build_identity) + '\n';
    std::map<std::string, ConfigOrigin> origins;
    for (const auto& origin : c.origins) {
        validate_origin(origin);
        if (!origins.emplace(origin.key, origin).second) invalid(origin.key, "duplicate provenance field");
    }
    for (const auto& b : bindings()) {
        const auto found = origins.find(b.field.key);
        result += "origin." + b.field.key + '=';
        if (found == origins.end()) result += "typed:caller";
        else result += escaped(found->second.layer) + ':' + escaped(found->second.name);
        result += '\n';
    }
    return result;
}
ParsedResolvedConfigRecord parse_resolved_config_record(std::string_view record, const ConfigCapabilities& capabilities) {
    constexpr std::string_view marker = "record.schema=saccade-resolved-record/1\n";
    const auto position = record.find("record.schema=");
    if (position == std::string_view::npos || position == 0 || record[position - 1] != '\n' ||
        !record.substr(position).starts_with(marker) || record.empty() || record.back() != '\n')
        invalid("record", "missing or unsupported record schema or truncated record");
    const auto canonical = record.substr(0, position);
    ParsedResolvedConfigRecord result{resolve_tracking_config(canonical, {}, {}, capabilities), {}};
    if (canonical_resolved_config(result.config) != canonical) invalid("record", "configuration prefix is not canonical and complete");
    result.config.origins.clear();
    const auto metadata = parse_file(record.substr(position + marker.size()));
    std::set<std::string> seen;
    std::string hash;
    for (const auto& [key, value] : metadata) {
        if (!seen.insert(key).second) invalid(key, "duplicate record metadata");
        if (key == "config.sha256") hash = value;
        else if (key == "build.identity") result.build_identity = unescaped(value);
        else if (key.starts_with("origin.")) {
            const auto colon = value.find(':');
            if (colon == std::string::npos) invalid(key, "missing provenance layer/name separator");
            ConfigOrigin origin{key.substr(7), unescaped(std::string_view(value).substr(0, colon)),
                unescaped(std::string_view(value).substr(colon + 1))};
            validate_origin(origin);
            result.config.origins.push_back(std::move(origin));
        } else invalid(key, "unknown record metadata");
    }
    if (hash != resolved_config_hash(result.config)) invalid("config.sha256", "missing or incorrect config hash");
    if (result.build_identity.empty()) invalid("build.identity", "required");
    if (result.config.origins.size() != bindings().size()) invalid("origin", "incomplete provenance records");
    // There is exactly one origin per known key (duplicates rejected above).
    // Re-encode to reject reordered, whitespace-normalized or truncated metadata.
    if (resolved_config_record(result.config, result.build_identity) != record)
        invalid("record", "metadata is not canonical and complete");
    return result;
}
std::vector<ConfigDifference> compare_resolved_config(const ResolvedTrackingConfig& a,
    const ResolvedTrackingConfig& b, const std::vector<std::string>& allowed_differences) {
    validate_common(a); validate_common(b);
    const std::set<std::string> allowed_fields{"execution.mode", "execution.candidate_slots", "execution.motion_slots",
        "execution.result_slots", "execution.cuda_graph", "execution.perturb_seed", "observer.event_dump"};
    std::set<std::string> allow;
    for (const auto& field : allowed_differences) {
        if (!allowed_fields.contains(field)) invalid(field, "not a permitted scheduling/observation comparison difference");
        if (!allow.insert(field).second) invalid(field, "duplicate comparison allowlist field");
    }
    std::vector<ConfigDifference> result;
    for (const auto& binding : bindings()) {
        const auto av = binding.get(a), bv = binding.get(b);
        if (av != bv && !allow.contains(binding.field.key)) result.push_back({binding.field.key, av, bv});
    }
    return result;
}
} // namespace saccade

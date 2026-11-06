#include "saccade/resolved_config.hpp"
#include "saccade/sha256.hpp"

#include <algorithm>
#include <bit>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
std::size_t checks = 0;
void check(bool condition, std::string_view message) {
    ++checks;
    if (!condition) throw std::runtime_error(std::string(message));
}
template<class F> void reject(F function, std::string_view message) {
    bool failed = false;
    try { function(); } catch (const std::invalid_argument&) { failed = true; }
    check(failed, message);
}
std::string file(std::string_view fields) {
    return "schema=" + std::string(saccade::kResolvedConfigSchema) + '\n' + std::string(fields);
}
std::string replace_once(std::string input, std::string_view from, std::string_view to) {
    const auto found = input.find(from);
    if (found == std::string::npos) throw std::runtime_error("test fixture substring missing");
    input.replace(found, from.size(), to);
    return input;
}
void tests() {
    using namespace saccade;
    const auto defaults = resolve_tracking_config();
    const auto canonical = canonical_resolved_config(defaults);
    check(canonical.starts_with("schema=saccade-tracking-config/1\n"), "versioned config encoding");
    check(canonical.ends_with("flow.policy=not-applicable\n"), "fixed policy and final LF");
    check(canonical.find("tracker.kalman.r_scale=f32:0x40600000\n") != std::string::npos,
          "float32 encoding pinned to bits and independent of formatting/locale");
    check(canonical.find("capacity.tracks=512\ncapacity.candidates=300\ncapacity.packet_rows=256\n") != std::string::npos,
          "behavior-changing fixed capacities recorded");
    check(canonical_resolved_config(resolve_tracking_config(canonical)) == canonical, "canonical parse roundtrip");
    check(resolved_config_hash(defaults) == sha256_hex(canonical), "hash covers canonical content");
    check(canonical_tracker_config(defaults.tracker) == canonical_tracker_config(TrackerConfig{}), "existing tracker defaults unchanged");
    const auto schema = tracking_config_schema();
    std::set<std::string> keys, env_names;
    for (const auto& f : schema) {
        check(keys.insert(f.key).second, "schema field unique");
        check(!f.owner.empty() && !f.behavior.empty() && !f.purpose.empty() && !f.default_value.empty() && !f.constraint.empty(),
              "schema publishes ownership, behavior, purpose, default and constraints");
        if (!f.environment.empty()) check(env_names.insert(f.environment).second, "environment names unambiguous");
    }
    const auto source = resolve_tracking_config(
        file(" tracker.bridge.enabled = false \n tracker.life.confirm_streak=5\n"),
        {{"SACCADE_CFG_TRACKER_BRIDGE_ENABLED", "true"}, {"SACCADE_CFG_TRACKER_LIFE_CONFIRM_STREAK", "6"}},
        {{"tracker.life.confirm_streak", "7"}});
    check(source.tracker.bridge.enabled && source.tracker.life.confirm_streak == 7, "file < environment < CLI");
    const auto origin = [&](std::string_view key) -> const ConfigOrigin& {
        return *std::find_if(source.origins.begin(), source.origins.end(), [&](const auto& o) { return o.key == key; });
    };
    check(origin("tracker.bridge.enabled").layer == "environment" &&
          origin("tracker.bridge.enabled").name == "SACCADE_CFG_TRACKER_BRIDGE_ENABLED", "environment provenance retained");
    check(origin("tracker.life.confirm_streak").layer == "cli", "CLI provenance retained");
    check(origin("tracker.assoc.top_k").layer == "default", "default provenance retained");
    const auto same_a = resolve_tracking_config(file("tracker.assoc.high_thresh=0.5\ntracker.bridge.enabled=false\n"));
    const auto same_b = resolve_tracking_config(file(" # arbitrary key order and whitespace\n tracker.bridge.enabled = false\r\ntracker.assoc.high_thresh=5e-1\n"));
    const auto same_c = resolve_tracking_config({}, {}, {{"tracker.bridge.enabled", "false"}, {"tracker.assoc.high_thresh", "0.5000000"}});
    check(resolved_config_hash(same_a) == resolved_config_hash(same_b) &&
          resolved_config_hash(same_b) == resolved_config_hash(same_c), "whitespace, order, numeric spelling and source do not change hash");
    auto custom_defaults = ResolvedTrackingConfig{};
    custom_defaults.tracker.assoc.top_k = 8;
    const auto preset = resolve_tracking_config({}, {}, {}, {}, custom_defaults, "offline preset/1");
    check(preset.tracker.assoc.top_k == 8 && preset.origins.front().name == "offline preset/1", "explicit entrypoint defaults and label preserved");
    check(resolve_tracking_config(file("tracker.assoc.top_k=1"), {}, {}, {}, custom_defaults).tracker.assoc.top_k == 1,
          "file overrides entrypoint preset");
    auto negative_zero = defaults;
    negative_zero.tracker.bridge.margin = -0.0f;
    check(canonical_resolved_config(negative_zero).find("tracker.bridge.margin=f32:0x80000000\n") != std::string::npos,
          "signed zero preserved exactly");
    const auto roundtrip_zero = resolve_tracking_config(canonical_resolved_config(negative_zero));
    check(std::bit_cast<std::uint32_t>(roundtrip_zero.tracker.bridge.margin) == 0x80000000u, "signed zero roundtrip");
    for (const std::string_view bad : {
        "schema=other\n", "schema=saccade-tracking-config/1\nschema=saccade-tracking-config/1\n",
        "tracker.bridge.enabled=true\n", "schema=saccade-tracking-config/1\nunknown.key=1\n",
        "schema=saccade-tracking-config/1\nsource.width=2=3\n", "schema=saccade-tracking-config/1\n=3\n",
        "schema=saccade-tracking-config/1\nsource.width=\n", "schema=saccade-tracking-config/1\nsource.width 3\n"
    }) reject([&] { resolve_tracking_config(bad); }, "schema and syntax rejection");
    reject([&] { resolve_tracking_config(file("source.width=1\nsource.width=2\n")); }, "duplicate file key");
    reject([&] { resolve_tracking_config({}, {{"SACCADE_CFG_UNKNOWN", "1"}}); }, "unknown environment override");
    reject([&] { resolve_tracking_config({}, {{"SACCADE_CFG_SOURCE_WIDTH", "1"}, {"SACCADE_CFG_SOURCE_WIDTH", "2"}}); }, "duplicate environment key");
    reject([&] { resolve_tracking_config({}, {}, {{"tracker.bridge.enabled", "true"}, {"tracker.bridge.enabled", "false"}}); }, "duplicate CLI key");
    reject([&] { resolve_tracking_config(std::string("schema=saccade-tracking-config/1\n\0", 34)); }, "embedded NUL rejected");
    reject([&] { resolve_tracking_config(file("implementation.backend=référence\n")); }, "non-ASCII input rejected");
    for (const ConfigAssignments& bad : {
        ConfigAssignments{{"tracker.bridge.enabled", "1"}}, {{"tracker.bridge.enabled", "True"}},
        {{"tracker.life.confirm_streak", "3.0"}}, {{"source.epoch", "-1"}}, {{"source.epoch", "+1"}},
        {{"source.epoch", "18446744073709551616"}}, {{"tracker.life.confirm_streak", "4294967296"}},
        {{"tracker.kalman.r_scale", "nan"}}, {{"tracker.kalman.r_scale", "inf"}},
        {{"tracker.kalman.r_scale", "1e999"}}, {{"tracker.kalman.r_scale", "f32:0x7f800000"}},
        {{"tracker.kalman.r_scale", "f32:0x7fc00000"}}, {{"tracker.kalman.r_scale", "f32:0x00000"}},
        {{"tracker.assoc.top_k", "0"}}, {{"tracker.assoc.top_k", "9"}},
        {{"tracker.batch.track_thresh", "0.2"}}, {{"tracker.assoc.mid_thresh", "0.5"}},
        {{"tracker.kalman.r_scale", "0"}}, {{"tracker.kalman.std_position", "-1"}},
        {{"tracker.kalman.std_velocity", "0"}}, {{"tracker.assoc.lambda", "0"}},
        {{"tracker.assoc.auction_eps", "0"}}, {{"tracker.bridge.speed_norm", "0"}},
        {{"tracker.assoc.stability_cost_w", "-1"}}, {{"tracker.assoc.stability_bid_w", "-1"}},
        {{"tracker.bridge.margin", "-1"}}, {{"tracker.bridge.max_distance", "-1"}},
        {{"tracker.bridge.height_ratio_lo", "0"}}, {{"tracker.bridge.height_ratio_hi", "0.5"}},
        {{"tracker.life.confirm_streak", "0"}}, {{"tracker.life.track_buffer", "0"}},
        {{"tracker.bridge.trigger_streak", "3"}}, {{"tracker.bridge.min_lost", "0"}},
        {{"tracker.continuation.max_candidates", "301"}}, {{"tracker.continuation.min_score", "1.1"}},
        {{"source.width", "0"}}, {{"source.height", "8193"}}, {{"preprocess.max_width", "100"}},
        {{"preprocess.max_jpeg_bytes", "0"}}, {{"preprocess.max_jpeg_bytes", "268435457"}},
        {{"execution.device", "2147483648"}}, {{"execution.result_slots", "0"}},
        {{"execution.candidate_slots", "0"}}, {{"execution.motion_slots", "0"}},
        {{"execution.mode", "parallel"}}, {{"execution.mode", "pipelined"}}, {{"execution.cuda_graph", "true"}},
        {{"execution.perturb_seed", "1"}}, {{"implementation.backend", "unknown"}},
        {{"implementation.input", "jpeg"}}, {{"implementation.input", "unknown"}},
        {{"implementation.motion_profile", "v1-motion/1"}}, {{"implementation.motion_profile", "v1-motion/2"}},
        {{"implementation.model_profile", "unknown"}}, {{"output.profile", "unknown"}},
        {{"output.interpolate_min_rows", "0"}}, {{"output.profile", "stream-observations/1"}},
        {{"policy.tracker", "other"}}, {{"capacity.tracks", "1024"}}, {{"schema", "saccade-tracking-config/1"}}
    }) reject([&] { resolve_tracking_config({}, {}, bad); }, "typed range, dependency, enum and fixed-policy rejection");
    check(resolve_tracking_config({}, {}, {{"source.epoch", "18446744073709551615"}}).source.epoch == UINT64_MAX,
          "uint64 upper boundary accepted");
    const ConfigCapabilities gpu{{"reference", "cuda"}, {"records", "jpeg", "host-rgb", "device-video"}};
    auto native_defaults = ResolvedTrackingConfig{};
    native_defaults.implementation.backend = "cuda";
    native_defaults.implementation.input = "jpeg";
    native_defaults.implementation.motion_profile = "v1-motion/2";
    native_defaults.execution.mode = "pipelined";
    native_defaults.execution.cuda_graph = true;
    const auto native = resolve_tracking_config({}, {}, {}, gpu, native_defaults, "track_sequence");
    reject([&] { resolve_tracking_config({}, {}, {}, {}, native_defaults); }, "build capability rejects CUDA unavailable");
    reject([&] { resolve_tracking_config({}, {}, {{"implementation.input", "device-video"}}, {{"cuda"}, {"jpeg"}}, native_defaults); },
           "build capability rejects NVDEC path unavailable");
    reject([&] { resolve_tracking_config({}, {}, {{"observer.event_dump", "true"}}, gpu, native_defaults); }, "CUDA cannot claim complete event dump");
    reject([&] { resolve_tracking_config({}, {}, {{"implementation.motion_profile", "disabled"}}, gpu, native_defaults); }, "native motion fallback forbidden");
    reject([&] { resolve_tracking_config({}, {}, {{"source.width", "16"}}, gpu, native_defaults); }, "G1 image minimum");
    reject([&] { resolve_tracking_config({}, {}, {{"preprocess.max_width", "4096"}}, gpu, native_defaults); }, "ignored native dimension overrides rejected");
    auto hr = native_defaults;
    hr.implementation.model_profile = ModelProfile::Hr1280x768;
    const auto hr_native = resolve_tracking_config({}, {}, {}, gpu, hr);
    check(canonical_resolved_config(hr_native).find("policy.precision=fp32-input-output-scan-tracker;engine-internal-external-receipt\n") != std::string::npos,
          "runtime precision recorded without inferring internal engine precision from shape");
    check(canonical_resolved_config(resolve_tracking_config(canonical_resolved_config(hr_native), {}, {}, gpu)) == canonical_resolved_config(hr_native),
          "native canonical roundtrip independent of assignment ordering");
    auto stream = native_defaults;
    stream.implementation.input = "device-video";
    stream.output_profile = "stream-observations/1";
    stream.mot.interpolate_max_gap = 0;
    const auto stream_resolved = resolve_tracking_config({}, {}, {}, gpu, stream);
    check(canonical_resolved_config(stream_resolved).find("policy.input=device-video-input/1+device-video-color/1\n") != std::string::npos,
          "device color policy recorded");
    auto serial = native;
    serial.execution.mode = "serial";
    serial.execution.cuda_graph = false;
    serial.execution.perturb_seed = 7;
    serial.execution.result_slots = 1;
    check(compare_resolved_config(native, serial).size() == 4, "default comparison includes execution/probe changes");
    const std::vector<std::string> schedule{"execution.mode", "execution.cuda_graph", "execution.perturb_seed", "execution.result_slots"};
    check(compare_resolved_config(native, serial, schedule).empty(), "explicit scheduling allowlist accepted");
    serial.tracker.bridge.enabled = false;
    const auto behavior_change = compare_resolved_config(native, serial, schedule);
    check(behavior_change.size() == 1 && behavior_change.front().key == "tracker.bridge.enabled", "behavior checked despite allowed full-hash differences");
    reject([&] { compare_resolved_config(native, serial, {"tracker.bridge.enabled"}); }, "behavior cannot be allowlisted");
    reject([&] { compare_resolved_config(native, serial, {"execution.device"}); }, "device cannot be allowlisted");
    reject([&] { compare_resolved_config(native, serial, {"execution.mode", "execution.mode"}); }, "duplicate allowlist rejected");
    auto observed = defaults;
    observed.observer.event_dump = true;
    check(compare_resolved_config(defaults, observed).size() == 1 && compare_resolved_config(defaults, observed, {"observer.event_dump"}).empty(),
          "observer equality requires explicit permission");
    const std::string build = "source=abc\nbinary=/tmp/with space\\bin";
    const auto record = resolved_config_record(source, build);
    const auto parsed = parse_resolved_config_record(record);
    check(parsed.build_identity == build && resolved_config_hash(parsed.config) == resolved_config_hash(source), "record config/hash/build roundtrip with escaped bytes");
    check(resolved_config_record(parsed.config, parsed.build_identity) == record, "provenance record exact roundtrip");
    const auto other_build = parse_resolved_config_record(resolved_config_record(source, "different-build"));
    check(other_build.build_identity != parsed.build_identity && compare_resolved_config(other_build.config, parsed.config).empty(),
          "same config hash does not collapse build identity");
    reject([&] { resolved_config_record(source, ""); }, "build identity required");
    reject([&] { parse_resolved_config_record(canonical); }, "bare config not misread as receipt");
    reject([&] { parse_resolved_config_record(record.substr(0, record.size() - 1)); }, "truncated record final LF");
    reject([&] { parse_resolved_config_record(record.substr(0, record.find("origin.flow.policy="))); }, "truncated provenance");
    reject([&] { parse_resolved_config_record(replace_once(record, "config.sha256=", "config.sha256=0")); }, "tampered hash rejected");
    reject([&] { parse_resolved_config_record(replace_once(record, "tracker.life.confirm_streak=7\n", "tracker.life.confirm_streak=8\n")); }, "tampered canonical content rejected");
    reject([&] { parse_resolved_config_record(record + "unknown.metadata=x\n"); }, "unknown record metadata");
    reject([&] { parse_resolved_config_record(record + "build.identity=other\n"); }, "duplicate metadata");
    reject([&] { parse_resolved_config_record(replace_once(record, "record.schema=saccade-resolved-record/1", "record.schema=saccade-resolved-record/2")); }, "unsupported receipt version");
    reject([&] { parse_resolved_config_record(replace_once(record, "origin.source.id=default:defaults", "origin.source.id=bogus:defaults")); }, "invalid provenance layer");
    reject([&] { parse_resolved_config_record(replace_once(record, "origin.source.id=default:defaults", "origin.unknown=default:defaults")); }, "unknown provenance field");
    reject([&] { parse_resolved_config_record(replace_once(record, "build.identity=", "build.identity=%zz")); }, "malformed percent escape");
    auto bad_typed = defaults;
    bad_typed.implementation.model_profile = static_cast<ModelProfile>(99);
    reject([&] { canonical_resolved_config(bad_typed); }, "invalid typed enum rejected before hashing");
    bad_typed = defaults;
    bad_typed.tracker.assoc.match_thresh = 2;
    reject([&] { resolved_config_hash(bad_typed); }, "invalid typed numeric config rejected before hashing");
}
} // namespace
int main() {
    try {
        tests();
        std::cout << "resolved_config_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "resolved_config_tests: " << error.what() << '\n';
        return 1;
    }
}

#include "saccade/synthetic_config.hpp"

#include <charconv>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace saccade {
namespace {
[[noreturn]] void invalid(std::string_view field, std::string_view why) {
    throw std::invalid_argument("config " + std::string(field) + ": " + std::string(why));
}

template <typename T> T integer(std::string_view key, std::string_view value) {
    static_assert(std::is_integral_v<T>);
    if (value.empty()) invalid(key, "empty integer");
    auto digits = value;
    if (digits.front() == '-') {
        if constexpr (std::is_unsigned_v<T>) invalid(key, "negative unsigned integer");
        digits.remove_prefix(1);
        if (digits.empty() || digits == "0") invalid(key, "noncanonical integer");
    }
    if (digits.size() > 1 && digits.front() == '0') invalid(key, "leading zero");
    for (const char ch : digits) if (ch < '0' || ch > '9') invalid(key, "expected decimal integer");
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) invalid(key, "integer out of range");
    return result;
}
bool boolean(std::string_view key, std::string_view value) {
    if (value == "true") return true;
    if (value == "false") return false;
    invalid(key, "expected true or false");
}
bool one_of(std::string_view value, std::initializer_list<std::string_view> choices) {
    for (const auto choice : choices) if (value == choice) return true;
    return false;
}
std::string serialize(const SyntheticConfig& c, bool behavior) {
    validate_config(c);
    std::string out = "schema=1\n";
    const auto add = [&](std::string_view key, const auto& value) {
        out += key;
        out += '=';
        using T = std::remove_cvref_t<decltype(value)>;
        if constexpr (std::is_same_v<T, bool>) out += value ? "true" : "false";
        else if constexpr (std::is_integral_v<T>) out += std::to_string(value);
        else out += value;
        out += '\n';
    };
    if (!behavior) {
        add("mode", c.mode);
        add("n_mi", c.n_mi); add("n_cs", c.n_cs);
        add("n_h", c.n_h); add("n_rp", c.n_rp);
    }
    add("width", c.width); add("height", c.height); add("frames", c.frames);
    add("epoch", c.epoch); add("seed", c.seed);
    if (!behavior) {
        add("delay_seed", c.delay_seed); add("delay_cycles", c.delay_cycles);
    }
    add("fail_frame", c.fail_frame); add("fail_stage", c.fail_stage);
    add("cancel_frame", c.cancel_frame); add("cancel_cut", c.cancel_cut);
    add("fixed_submit", c.fixed_submit);
    if (!behavior) add("terminal_snapshot", c.terminal_snapshot);
    return out;
}
} // namespace

SyntheticConfig parse_config(std::string_view input) {
    SyntheticConfig c;
    std::set<std::string> seen;
    for (const unsigned char ch : input)
        if (ch > 126 || (ch < 33 && ch != '\n')) invalid("syntax", "ASCII records without whitespace required");
    while (!input.empty()) {
        const auto newline = input.find('\n');
        const auto record = input.substr(0, newline);
        if (newline == std::string_view::npos) input = {};
        else input.remove_prefix(newline + 1);
        if (record.empty()) continue;
        const auto equals = record.find('=');
        if (equals == std::string_view::npos || equals == 0 || equals + 1 == record.size() ||
            record.find('=', equals + 1) != std::string_view::npos)
            invalid("syntax", "expected one key=value per record");
        const auto key = record.substr(0, equals);
        const auto value = record.substr(equals + 1);
        if (!seen.insert(std::string(key)).second) invalid(key, "duplicate field");
        if (key == "schema") { if (value != "1") invalid(key, "unsupported schema"); }
        else if (key == "mode") c.mode = value;
        else if (key == "n_mi") c.n_mi = integer<int>(key, value);
        else if (key == "n_cs") c.n_cs = integer<int>(key, value);
        else if (key == "n_h") c.n_h = integer<int>(key, value);
        else if (key == "n_rp") c.n_rp = integer<int>(key, value);
        else if (key == "width") c.width = integer<int>(key, value);
        else if (key == "height") c.height = integer<int>(key, value);
        else if (key == "frames") c.frames = integer<int>(key, value);
        else if (key == "epoch") c.epoch = integer<std::uint64_t>(key, value);
        else if (key == "seed") c.seed = integer<std::uint64_t>(key, value);
        else if (key == "delay_seed") c.delay_seed = integer<std::uint64_t>(key, value);
        else if (key == "delay_cycles") c.delay_cycles = integer<std::uint32_t>(key, value);
        else if (key == "fail_frame") c.fail_frame = integer<int>(key, value);
        else if (key == "fail_stage") c.fail_stage = value;
        else if (key == "cancel_frame") c.cancel_frame = integer<int>(key, value);
        else if (key == "cancel_cut") c.cancel_cut = value;
        else if (key == "fixed_submit") c.fixed_submit = boolean(key, value);
        else if (key == "terminal_snapshot") c.terminal_snapshot = boolean(key, value);
        else invalid(key, "unknown field");
    }
    validate_config(c);
    return c;
}

void validate_config(const SyntheticConfig& c) {
    if (!one_of(c.mode, {"serial", "pipelined"})) invalid("mode", "expected serial or pipelined");
    const auto ring = [](int n, std::string_view key) {
        if (n < 1 || n > 16) invalid(key, "expected 1..16");
    };
    ring(c.n_mi, "n_mi"); ring(c.n_cs, "n_cs"); ring(c.n_h, "n_h"); ring(c.n_rp, "n_rp");
    const auto shape = [](int n, std::string_view key) {
        if (n < 8 || n > 32 || (n & (n - 1)) != 0) invalid(key, "expected power of two in 8..32");
    };
    shape(c.width, "width"); shape(c.height, "height");
    if (c.frames < 1 || c.frames > 100000) invalid("frames", "expected 1..100000");
    if (c.epoch == 0) invalid("epoch", "must be nonzero");
    if (c.seed > std::numeric_limits<std::uint32_t>::max()) invalid("seed", "expected uint32 range");
    if (c.delay_seed > std::numeric_limits<std::uint32_t>::max()) invalid("delay_seed", "expected uint32 range");
    if (c.delay_cycles > 1000000) invalid("delay_cycles", "expected 0..1000000");
    if (c.fail_frame < -1 || c.fail_frame >= c.frames) invalid("fail_frame", "outside submitted frame range");
    if (!one_of(c.fail_stage, {"none", "before_spectrum", "after_spectrum", "after_predict", "after_update"}))
        invalid("fail_stage", "unsupported stage");
    if ((c.fail_frame == -1) != (c.fail_stage == "none")) invalid("fail_stage", "must pair with fail_frame");
    if (c.cancel_frame < -1 || c.cancel_frame >= c.frames) invalid("cancel_frame", "outside submitted frame range");
    if (!one_of(c.cancel_cut, {"none", "after_h2d", "after_mi", "after_cs", "after_begin", "after_fft",
                              "after_predict", "after_update", "after_packet"}))
        invalid("cancel_cut", "unsupported cut");
    if ((c.cancel_frame == -1) != (c.cancel_cut == "none")) invalid("cancel_cut", "must pair with cancel_frame");
    if (c.fail_frame >= 0 && c.cancel_frame >= 0) invalid("cancel_frame", "failure plus cancellation unsupported");
}

std::string canonical_config(const SyntheticConfig& c) { return serialize(c, false); }
std::string behavior_config(const SyntheticConfig& c) { return serialize(c, true); }

std::string config_hash(const SyntheticConfig& c) { return sha256_hex(canonical_config(c)); }
std::string behavior_hash(const SyntheticConfig& c) { return sha256_hex(behavior_config(c)); }

} // namespace saccade

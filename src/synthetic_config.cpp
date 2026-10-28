#include "saccade/synthetic_config.hpp"

#include <array>
#include <bit>
#include <charconv>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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

std::string sha256_hex(std::string_view input) {
    // FIPS 180-4 SHA-256, byte-oriented. Length is encoded modulo 2^64.
    constexpr std::array<std::uint32_t, 64> k{
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t, 8> h{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                   0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::vector<std::uint8_t> bytes(input.begin(), input.end());
    const std::uint64_t bits = static_cast<std::uint64_t>(input.size()) * 8;
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56) bytes.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8) bytes.push_back(static_cast<std::uint8_t>(bits >> shift));
    for (std::size_t block = 0; block < bytes.size(); block += 64) {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16; ++i)
            for (std::size_t j = 0; j < 4; ++j) w[i] = (w[i] << 8) | bytes[block + 4 * i + j];
        for (std::size_t i = 16; i < 64; ++i) {
            const auto s0 = std::rotr(w[i-15], 7) ^ std::rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const auto s1 = std::rotr(w[i-2], 17) ^ std::rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (std::size_t i = 0; i < 64; ++i) {
            const auto s1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const auto ch = (e & f) ^ (~e & g);
            const auto t1 = hh + s1 + ch + k[i] + w[i];
            const auto s0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const auto maj = (a & b) ^ (a & c) ^ (b & c);
            const auto t2 = s0 + maj;
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string output;
    output.reserve(64);
    for (const auto word : h) for (int shift = 28; shift >= 0; shift -= 4) output += hex[(word >> shift) & 15];
    return output;
}
std::string config_hash(const SyntheticConfig& c) { return sha256_hex(canonical_config(c)); }
std::string behavior_hash(const SyntheticConfig& c) { return sha256_hex(behavior_config(c)); }

} // namespace saccade

#include "saccade/synthetic_config.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
std::size_t checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template <typename F> void reject(F&& f, const char* message) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, message);
}
void tests() {
    using namespace saccade;
    check(sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA256 empty known answer");
    check(sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA256 abc known answer");
    check(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "SHA256 multiblock known answer");
    check(sha256_hex(std::string(1000000, 'a')) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "SHA256 million-byte known answer");
    const SyntheticConfig c;
    const std::string expected =
        "schema=1\nmode=pipelined\nn_mi=2\nn_cs=2\nn_h=2\nn_rp=2\nwidth=8\nheight=8\nframes=16\n"
        "epoch=1\nseed=7\ndelay_seed=11\ndelay_cycles=20000\nfail_frame=-1\nfail_stage=none\n"
        "cancel_frame=-1\ncancel_cut=none\nfixed_submit=false\nterminal_snapshot=false\n";
    check(canonical_config(c) == expected, "pinned schema/order/defaults/final LF");
    check(parse_config(expected) == c && parse_config("") == c, "canonical and default parse");
    check(parse_config("frames=16\nmode=pipelined\n") == c, "field order irrelevant");
    check(parse_config("frames=16") == c, "optional final LF");
    check(parse_config("\nframes=16\n\n") == c, "empty records harmless");
    check(config_hash(c).size() == 64 && config_hash(c) == sha256_hex(expected), "config hashes canonical bytes");
    auto changed = c;
    changed.mode = "serial"; changed.n_mi = 1; changed.n_cs = 16; changed.n_h = 3; changed.n_rp = 4;
    changed.delay_seed = 99; changed.delay_cycles = 0; changed.terminal_snapshot = true;
    check(config_hash(changed) != config_hash(c), "full hash sees allowed differences");
    check(behavior_config(changed) == behavior_config(c) && behavior_hash(changed) == behavior_hash(c),
          "explicit scheduling/observation comparison allowlist");
    for (int field = 0; field < 7; ++field) {
        changed = c;
        if (field == 0) changed.seed++;
        if (field == 1) changed.epoch++;
        if (field == 2) changed.width = 16;
        if (field == 3) changed.height = 32;
        if (field == 4) changed.frames++;
        if (field == 5) { changed.fail_frame = 2; changed.fail_stage = "after_update"; }
        if (field == 6) changed.fixed_submit = true;
        check(behavior_hash(changed) != behavior_hash(c), "behavioral change cannot use allowlist");
    }
    changed = c; changed.cancel_frame = 4; changed.cancel_cut = "after_cs";
    check(behavior_hash(changed) != behavior_hash(c), "cancellation changes comparison identity");
    for (const std::string_view input : {
        "unknown=1", "frames=3\nframes=4", "schema=1\nschema=1", "schema=2", "schema=true",
        "frames=", "=1", "frames", "frames=3=4", "frames= 3", "frames=3 ", "frames=3\r\n",
        "frames=3\t", "frames=3\n#comment", "frames=3.0", "frames=true", "frames=+3", "frames=03",
        "fail_frame=-0", "frames=2147483648", "seed=-1", "seed=18446744073709551616", "epoch=0",
        "seed=4294967296", "delay_seed=4294967296", "delay_cycles=-1", "delay_cycles=1000001",
        "delay_cycles=4294967296", "fixed_submit=1", "terminal_snapshot=True", "mode=parallel",
        "n_mi=0", "n_cs=-1", "n_h=17", "n_rp=0", "width=7", "height=64", "width=12",
        "frames=0", "frames=100001", "fail_frame=-2", "fail_frame=16\nfail_stage=after_predict",
        "fail_frame=2", "fail_stage=after_predict", "fail_stage=garbage", "cancel_frame=2",
        "cancel_cut=after_cs", "cancel_frame=-2", "cancel_frame=16\ncancel_cut=after_cs",
        "cancel_cut=garbage", "fail_frame=2\nfail_stage=after_update\ncancel_frame=3\ncancel_cut=after_cs"
    }) reject([&] { parse_config(input); }, "malformed or unsupported config rejected");
    reject([] { parse_config(std::string("mode=serial\n\0", 13)); }, "embedded NUL rejected");
    reject([] { parse_config("mode=sériál"); }, "non-ASCII rejected");
    for (const std::string stage : {"before_spectrum", "after_spectrum", "after_predict", "after_update"}) {
        const auto injected = parse_config("fail_frame=0\nfail_stage=" + stage + "\nfixed_submit=true");
        check(injected.fail_frame == 0 && injected.fail_stage == stage && injected.fixed_submit,
              "failure stages parse");
    }
    for (const std::string cut : {"after_h2d", "after_mi", "after_cs", "after_begin", "after_fft",
                                  "after_predict", "after_update", "after_packet"}) {
        const auto cancelled = parse_config("cancel_frame=15\ncancel_cut=" + cut);
        check(cancelled.cancel_frame == 15 && cancelled.cancel_cut == cut, "cancellation cuts parse");
    }
    const auto limits = parse_config("epoch=18446744073709551615\nseed=4294967295\ndelay_seed=4294967295\n"
                                     "width=32\nheight=16\nframes=100000\nn_mi=16\ndelay_cycles=1000000");
    check(limits.seed == 4294967295ULL && limits.frames == 100000 && limits.epoch == UINT64_MAX,
          "upper legal bounds accepted");
    changed = c; changed.n_rp = 0;
    reject([&] { canonical_config(changed); }, "typed config validated before serialization");
    reject([&] { behavior_hash(changed); }, "typed config validated before hashing");
}
} // namespace

int main() {
    try {
        tests();
        std::cout << "synthetic_config_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "synthetic_config_tests: " << error.what() << '\n';
        return 1;
    }
}

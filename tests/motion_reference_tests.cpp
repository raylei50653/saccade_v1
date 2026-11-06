#include "saccade/motion.hpp"
#include "motion_fixtures.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace {
using namespace saccade;
std::size_t checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class E, class F> void rejects(F&& fn) {
    bool caught = false;
    try { fn(); } catch (const E&) { caught = true; }
    check(caught, "expected rejection");
}
bool bits(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }
void config_and_window() {
    validate_motion_config({}); check(true, "default config");
    for (const int d : {0, -1}) rejects<std::invalid_argument>([&] { validate_motion_config({d, 5, .25f}); });
    for (const float p : {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        rejects<std::invalid_argument>([&] { validate_motion_config({4, p, .25f}); });
    for (const float f : {0.f, -1.f, .50001f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        rejects<std::invalid_argument>([&] { validate_motion_config({4, 5, f}); });
    validate_motion_config({1, 1, .5f}); check(true, "inclusive shift config boundary");
    for (const auto wh : {std::array{31, 32}, std::array{32, 31}, std::array{0, 32}, std::array{-32, 32}})
        rejects<std::invalid_argument>([&] { (void)motion_image_key(wh[0], wh[1], {}); });
    check(motion_image_key(35, 39, {}) == MotionImageKey{35, 39, 4, 8, 9}, "floor dimensions/key");
    check(!(motion_image_key(35, 39, {}) == motion_image_key(34, 39, {})), "original dimensions part of key");
    check(hanning_window(2) == std::vector<float>{0, 0}, "two sample window");
    check(hanning_window(3) == std::vector<float>{0, 1, 0}, "three sample window");
    const auto window = hanning_window(8);
    for (int i = 0; i < 8; ++i)
        check(bits(window[i], float(.5 * (1 - std::cos(2 * std::numbers::pi * i / 7)))), "binary64 Hanning definition");
    rejects<std::invalid_argument>([] { (void)hanning_window(1); });
}
void f05() {
    const MotionConfig config{2, 5, .25f};
    const auto key = motion_image_key(17, 19, config);
    const auto plane = static_cast<std::size_t>(17 * 19);
    auto rgb = std::vector<std::uint8_t>(3 * plane, 255);
    const auto white = motion_image_reference(rgb, key);
    const auto wx = hanning_window(8), wy = hanning_window(9);
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 8; ++x) check(bits(white[y * 8 + x], wx[x] * wy[y]), "white hand F05");
    std::fill(rgb.begin(), rgb.end(), 0);
    for (int y = 0; y < 18; ++y) {
        for (int x = 0; x < 16; ++x) {
            const auto i = static_cast<std::size_t>(y * 17 + x);
            if (y % 2 == 0 && x % 2 == 1) rgb[i] = 255;
            if (y % 2 == 1 && x % 2 == 0) rgb[plane + i] = 255;
            if (y % 2 == 1 && x % 2 == 1) rgb[2 * plane + i] = 255;
        }
    }
    const auto tiles = motion_image_reference(rgb, key);
    for (int y = 0; y < 9; ++y)
        for (int x = 0; x < 8; ++x) check(bits(tiles[y * 8 + x], (.25f * wx[x]) * wy[y]), "RGB tile hand F05");
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 19; ++y) rgb[c * plane + y * 17 + 16] = 255;
        for (int x = 0; x < 17; ++x) rgb[c * plane + 18 * 17 + x] = 255;
    }
    check(tiles == motion_image_reference(rgb, key), "incomplete border blocks ignored");
    const auto one = motion_image_key(8, 8, {1, 5, .25f});
    check(motion_image_reference(std::vector<std::uint8_t>(3 * 64), one) == std::vector<float>(64), "black d=1");
    rejects<std::invalid_argument>([&] { (void)motion_image_reference({}, key); });
    auto bad = key; ++bad.width;
    rejects<std::invalid_argument>([&] { (void)motion_image_reference(rgb, bad); });
}
void f06() {
    MotionConfig config{4, 5, .25f};
    const auto key = motion_image_key(64, 64, config);
    const auto previous = motion_test::texture(key);
    auto first = phase_correlation_reference({}, previous, key, config, 123, 0);
    check(first.status == MotionStatus::uninitialized && first.reference_frame == -1 && first.frame == 0 && first.epoch == 123, "first identity");
    check(first.tx == 0 && first.ty == 0 && first.pcr == 0 && first.confidence == 0 && first.peak_x == 0 && first.peak_y == 0, "first zeros");
    for (const auto shift : {std::array{0, 0}, std::array{3, 0}, std::array{-3, 0}, std::array{0, 2},
                             std::array{0, -2}, std::array{2, -3}, std::array{-2, 3}, std::array{4, -4}}) {
        const auto current = motion_test::shifted(previous, key, shift[0], shift[1]);
        const auto f = phase_correlation_reference(previous, current, key, config, 123, 7);
        check(f.peak_x == shift[0] && f.peak_y == shift[1], "integer shift peak/direction");
        check(f.status == MotionStatus::valid && f.confidence == 1, "integer shift valid/full confidence");
        check(bits(f.tx, float(shift[0] * 4)) && bits(f.ty, float(shift[1] * 4)), "integer translation");
        check(std::abs(f.pcr - 16) < 1e-4f && f.epoch == 123 && f.frame == 7 && f.reference_frame == 6, "PCR and identity");
    }
    auto current = motion_test::shifted(previous, key, 5, 0);
    auto f = phase_correlation_reference(previous, current, key, config, 1, 1);
    check(f.status == MotionStatus::rejected && f.peak_x == 5 && f.tx == 0 && f.ty == 0 && f.pcr > 0, "beyond max fraction rejected");
    current = motion_test::shifted(previous, key, 0, -5);
    check(phase_correlation_reference(previous, current, key, config, 1, 1).status == MotionStatus::rejected, "y max fraction");
    config.max_shift_fraction = .5f;
    for (const auto shift : {std::array{8, 0}, std::array{-8, 0}, std::array{0, 8}, std::array{0, -8}, std::array{9, 9}}) {
        current = motion_test::shifted(previous, key, shift[0], shift[1]);
        f = phase_correlation_reference(previous, current, key, config, 1, 1);
        check(f.status == MotionStatus::valid && f.peak_x == (shift[0] == 9 ? -7 : std::abs(shift[0])) &&
              f.peak_y == (shift[1] == 9 ? -7 : std::abs(shift[1])), "wrap strict > half boundary");
    }
    config.pcr_threshold = 100;
    current = motion_test::shifted(previous, key, 2, -3);
    f = phase_correlation_reference(previous, current, key, config, 1, 1);
    check(f.confidence < 1 && bits(f.confidence, f.pcr / 100.f) && bits(f.tx, 8.f * f.confidence) &&
          bits(f.ty, -12.f * f.confidence), "confidence attenuates translation");
    const MotionConfig tiny{4, std::numeric_limits<float>::denorm_min(), .25f};
    rejects<std::domain_error>([&] { (void)phase_correlation_reference(previous, previous, key, tiny, 1, 1); });
    const std::vector<float> zero(previous.size());
    f = phase_correlation_reference(zero, zero, key, config, 1, 1);
    check(f.status == MotionStatus::rejected && f.peak_x == 0 && f.peak_y == 0 && f.pcr == 0 &&
          f.confidence == 0 && f.tx == 0 && f.ty == 0, "zero RMS and row-major tied peak");
    rejects<std::invalid_argument>([&] { (void)phase_correlation_reference({}, {}, key, config, 1, 1); });
    rejects<std::invalid_argument>([&] { (void)phase_correlation_reference(std::vector<float>(1), current, key, config, 1, 1); });
    for (const float value : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        current[0] = value;
        rejects<std::domain_error>([&] { (void)phase_correlation_reference({}, current, key, config, 1, 0); });
        rejects<std::domain_error>([&] { (void)phase_correlation_reference(current, previous, key, config, 1, 1); });
    }
}
struct File {
    std::filesystem::path path = "motion-reference-tests.scmo";
    ~File() { std::error_code ec; std::filesystem::remove(path, ec); }
};
void records() {
    File file;
    MotionRecords records{{35, 39, 4, 8, 9}, 999, {}};
    for (unsigned i = 0; i < 4; ++i) {
        MotionContext f;
        f.epoch = 999; f.frame = i; f.reference_frame = std::int64_t(i) - 1; f.status = static_cast<MotionStatus>(i);
        f.peak_x = -3; f.peak_y = 2; f.pcr = 5.25f; f.confidence = .5f; f.tx = -6; f.ty = 4;
        records.frames.push_back(f);
    }
    write_motion_records(file.path, records);
    auto decoded = read_motion_records(file.path);
    check(decoded.key == records.key && decoded.epoch == 999 && decoded.frames.size() == 4, "SCMO header roundtrip");
    for (std::size_t i = 0; i < 4; ++i) {
        const auto& a = decoded.frames[i]; const auto& b = records.frames[i];
        check(a.epoch == 999 && a.frame == b.frame && a.reference_frame == b.reference_frame && a.status == b.status &&
              a.peak_x == b.peak_x && a.peak_y == b.peak_y && bits(a.pcr, b.pcr) && bits(a.confidence, b.confidence) &&
              bits(a.tx, b.tx) && bits(a.ty, b.ty), "SCMO frame roundtrip with record epoch");
    }
    std::ifstream in(file.path, std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(in), {}};
    in.close();
    const auto memory_decoded = decode_motion_records(std::string_view(bytes.data(), bytes.size()));
    check(memory_decoded.key == decoded.key && memory_decoded.epoch == decoded.epoch &&
          memory_decoded.frames.size() == decoded.frames.size(), "SCMO immutable receipt bytes header");
    for (std::size_t i = 0; i < decoded.frames.size(); ++i) {
        const auto& a = memory_decoded.frames[i]; const auto& b = decoded.frames[i];
        check(a.epoch == b.epoch && a.frame == b.frame && a.reference_frame == b.reference_frame &&
              a.status == b.status && a.peak_x == b.peak_x && a.peak_y == b.peak_y && bits(a.pcr, b.pcr) &&
              bits(a.confidence, b.confidence) && bits(a.tx, b.tx) && bits(a.ty, b.ty),
              "SCMO immutable receipt bytes preserve every frame field");
    }
    check(bytes.size() == 44 + 4 * 44 && bytes[4] == 1 && bytes[8] == 35 &&
          static_cast<unsigned char>(bytes[28]) == 0xe7 && bytes[29] == 3 && bytes[36] == 4 &&
          static_cast<unsigned char>(bytes[52]) == 255 && static_cast<unsigned char>(bytes[64]) == 253,
          "SCMO canonical offsets/little endian");
    auto write_bytes = [&](const std::vector<char>& data) {
        std::ofstream out(file.path, std::ios::binary | std::ios::trunc);
        out.write(data.data(), data.size()); out.close();
    };
    for (std::size_t length = 0; length < bytes.size(); ++length) {
        write_bytes({bytes.begin(), bytes.begin() + length});
        rejects<std::runtime_error>([&] { (void)read_motion_records(file.path); });
        rejects<std::runtime_error>([&] { (void)decode_motion_records(std::string_view(bytes.data(), length)); });
    }
    for (const auto offset : {0u, 4u, 60u}) {
        auto bad = bytes; bad[offset] = 99; write_bytes(bad);
        rejects<std::runtime_error>([&] { (void)read_motion_records(file.path); });
        rejects<std::runtime_error>([&] { (void)decode_motion_records(std::string_view(bad.data(), bad.size())); });
    }
    auto extra = bytes; extra.push_back(0); write_bytes(extra);
    rejects<std::runtime_error>([&] { (void)read_motion_records(file.path); });
    rejects<std::runtime_error>([&] { (void)decode_motion_records(std::string_view(extra.data(), extra.size())); });
    write_motion_records(file.path, {records.key, 0, {}});
    check(read_motion_records(file.path).frames.empty(), "SCMO empty stream");
    std::filesystem::remove(file.path);
    rejects<std::runtime_error>([&] { (void)read_motion_records(file.path); });
    rejects<std::runtime_error>([&] { write_motion_records(file.path / "absent" / "out", records); });
    if (std::filesystem::exists("/dev/full")) rejects<std::runtime_error>([&] { write_motion_records("/dev/full", records); });
}
}
int main() {
    try {
        config_and_window(); f05(); f06(); records();
        std::cout << "motion_reference_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "motion_reference_tests: " << e.what() << '\n'; return 1; }
}

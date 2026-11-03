#include "saccade/motion.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <fstream>
#include <numbers>
#include <stdexcept>
#include <type_traits>

namespace saccade {
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
void check_key(const MotionImageKey& key) {
    MotionConfig config;
    config.downscale = key.downscale;
    require(key == motion_image_key(key.original_width, key.original_height, config), "inconsistent motion key");
}
void finite(double value) {
    if (!std::isfinite(value)) throw std::domain_error("motion NUMERIC");
}
using Complex = std::complex<double>;
void finite(Complex value) { finite(value.real()); finite(value.imag()); }

std::vector<Complex> roots(int n, bool inverse) {
    std::vector<Complex> table(static_cast<std::size_t>(n) * n);
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            const double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi * (double(k) * j) / n;
            table[static_cast<std::size_t>(k) * n + j] = {std::cos(angle), std::sin(angle)};
        }
    }
    return table;
}
std::vector<Complex> dft(std::span<const Complex> image, int w, int h, bool inverse) {
    const auto wx = roots(w, inverse), wy = roots(h, inverse);
    std::vector<Complex> rows(image.size()), output(image.size());
    for (int y = 0; y < h; ++y) {
        for (int k = 0; k < w; ++k) {
            Complex sum{};
            for (int x = 0; x < w; ++x)
                sum += image[static_cast<std::size_t>(y) * w + x] * wx[static_cast<std::size_t>(k) * w + x];
            finite(sum);
            rows[static_cast<std::size_t>(y) * w + k] = inverse ? sum / double(w) : sum;
        }
    }
    for (int x = 0; x < w; ++x) {
        for (int k = 0; k < h; ++k) {
            Complex sum{};
            for (int y = 0; y < h; ++y)
                sum += rows[static_cast<std::size_t>(y) * w + x] * wy[static_cast<std::size_t>(k) * h + y];
            finite(sum);
            output[static_cast<std::size_t>(k) * w + x] = inverse ? sum / double(h) : sum;
        }
    }
    return output;
}
std::vector<Complex> spectrum(std::span<const float> image, int w, int h) {
    std::vector<Complex> input(image.size());
    for (std::size_t i = 0; i < image.size(); ++i) { finite(image[i]); input[i] = image[i]; }
    return dft(input, w, h, false);
}

template<class T> void put(std::ostream& out, T value) {
    using U = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
    const auto bits = std::bit_cast<U>(value);
    std::array<char, sizeof(T)> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<char>((bits >> (8 * i)) & 255);
    out.write(bytes.data(), bytes.size());
}
template<class T> T get(std::istream& in) {
    using U = std::conditional_t<sizeof(T) == 8, std::uint64_t, std::uint32_t>;
    std::array<unsigned char, sizeof(T)> bytes{};
    if (!in.read(reinterpret_cast<char*>(bytes.data()), bytes.size())) throw std::runtime_error("truncated SCMO");
    U bits = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) bits |= U(bytes[i]) << (8 * i);
    return std::bit_cast<T>(bits);
}
}

void validate_motion_config(const MotionConfig& config) {
    require(config.downscale >= 1, "motion downscale must be >= 1");
    require(std::isfinite(config.pcr_threshold) && config.pcr_threshold > 0, "invalid motion PCR threshold");
    require(std::isfinite(config.max_shift_fraction) && config.max_shift_fraction > 0 &&
            config.max_shift_fraction <= 0.5f, "invalid motion max shift fraction");
}
MotionImageKey motion_image_key(int original_width, int original_height, const MotionConfig& config) {
    validate_motion_config(config);
    const int w = original_width / config.downscale, h = original_height / config.downscale;
    require(w >= 8 && h >= 8, "MotionImage dimensions must be >= 8");
    return {original_width, original_height, config.downscale, w, h};
}
std::vector<float> hanning_window(int n) {
    require(n >= 2, "Hanning length must be >= 2");
    std::vector<float> window(n);
    for (int i = 0; i < n; ++i)
        window[i] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * i / (n - 1))));
    return window;
}
std::vector<float> motion_image_reference(std::span<const std::uint8_t> rgb, const MotionImageKey& key) {
    check_key(key);
    const auto plane = static_cast<std::size_t>(key.original_width) * key.original_height;
    require(rgb.size() == 3 * plane, "wrong motion RGB size");
    const auto wx = hanning_window(key.width), wy = hanning_window(key.height);
    const int d = key.downscale;
    std::vector<float> output(static_cast<std::size_t>(key.width) * key.height);
    for (int y = 0; y < key.height; ++y) {
        for (int x = 0; x < key.width; ++x) {
            float sum = 0;
            for (int dy = 0; dy < d; ++dy) {
                for (int dx = 0; dx < d; ++dx) {
                    const auto i = static_cast<std::size_t>(y * d + dy) * key.original_width + x * d + dx;
                    const float g = (0.299f * float(rgb[i]) + 0.587f * float(rgb[plane + i])) + 0.114f * float(rgb[2 * plane + i]);
                    sum += g;
                }
            }
            const float v = (sum / float(d * d)) / 255.0f;
            output[static_cast<std::size_t>(y) * key.width + x] = (v * wx[x]) * wy[y];
        }
    }
    return output;
}
MotionContext phase_correlation_reference(std::span<const float> previous, std::span<const float> current,
                                          const MotionImageKey& key, const MotionConfig& config,
                                          std::uint64_t epoch, std::int64_t frame) {
    validate_motion_config(config);
    require(key == motion_image_key(key.original_width, key.original_height, config), "inconsistent motion key/config");
    const auto n = static_cast<std::size_t>(key.width) * key.height;
    require(current.size() == n && (previous.empty() || previous.size() == n), "wrong MotionImage size");
    MotionContext output;
    output.epoch = epoch; output.frame = frame;
    const auto cur = spectrum(current, key.width, key.height);
    if (previous.empty()) return output;
    output.reference_frame = frame - 1;
    const auto prev = spectrum(previous, key.width, key.height);
    std::vector<Complex> cross(n);
    for (std::size_t i = 0; i < n; ++i) {
        const Complex c = std::conj(prev[i]) * cur[i];
        finite(c);
        const double magnitude = std::abs(c);
        finite(magnitude);
        cross[i] = c / (magnitude + 1e-6);
        finite(cross[i]);
    }
    const auto correlation = dft(cross, key.width, key.height, true);
    std::size_t peak = 0;
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        finite(correlation[i]);
        if (correlation[i].real() > correlation[peak].real()) peak = i;
        sum += correlation[i].real() * correlation[i].real();
        finite(sum);
    }
    output.peak_x = static_cast<int>(peak % key.width);
    output.peak_y = static_cast<int>(peak / key.width);
    if (output.peak_x > key.width / 2) output.peak_x -= key.width;
    if (output.peak_y > key.height / 2) output.peak_y -= key.height;
    output.status = MotionStatus::rejected;
    const double rms = std::sqrt(sum / double(n));
    finite(rms);
    if (rms <= 0) return output;
    output.pcr = static_cast<float>(correlation[peak].real() / rms);
    finite(output.pcr);
    const float scaled = output.pcr / config.pcr_threshold;
    finite(scaled);
    output.confidence = std::min(1.0f, scaled);
    if (std::abs(output.peak_x) > std::floor(config.max_shift_fraction * float(key.width)) ||
        std::abs(output.peak_y) > std::floor(config.max_shift_fraction * float(key.height))) return output;
    output.status = MotionStatus::valid;
    output.tx = float(output.peak_x * key.downscale) * output.confidence;
    output.ty = float(output.peak_y * key.downscale) * output.confidence;
    finite(output.tx); finite(output.ty);
    return output;
}

void write_motion_records(const std::filesystem::path& path, const MotionRecords& records) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open SCMO output");
    out.write("SCMO", 4);
    put(out, std::uint32_t{1});
    for (const int v : {records.key.original_width, records.key.original_height, records.key.downscale,
                        records.key.width, records.key.height}) put(out, static_cast<std::int32_t>(v));
    put(out, records.epoch);
    put(out, static_cast<std::uint64_t>(records.frames.size()));
    for (const auto& f : records.frames) {
        put(out, f.frame); put(out, f.reference_frame); put(out, static_cast<std::uint32_t>(f.status));
        put(out, f.peak_x); put(out, f.peak_y); put(out, f.pcr); put(out, f.confidence); put(out, f.tx); put(out, f.ty);
    }
    out.close();
    if (!out) throw std::runtime_error("cannot write/close SCMO output");
}
MotionRecords read_motion_records(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open SCMO input");
    std::array<char, 4> magic{};
    if (!in.read(magic.data(), magic.size())) throw std::runtime_error("truncated SCMO");
    if (magic != std::array<char, 4>{'S', 'C', 'M', 'O'}) throw std::runtime_error("unknown SCMO magic");
    if (get<std::uint32_t>(in) != 1) throw std::runtime_error("unknown SCMO version");
    MotionRecords records;
    records.key.original_width = get<std::int32_t>(in); records.key.original_height = get<std::int32_t>(in);
    records.key.downscale = get<std::int32_t>(in); records.key.width = get<std::int32_t>(in); records.key.height = get<std::int32_t>(in);
    records.epoch = get<std::uint64_t>(in);
    const auto count = get<std::uint64_t>(in);
    for (std::uint64_t i = 0; i < count; ++i) {
        MotionContext f;
        f.epoch = records.epoch;
        f.frame = get<std::int64_t>(in); f.reference_frame = get<std::int64_t>(in);
        const auto status = get<std::uint32_t>(in);
        if (status > static_cast<std::uint32_t>(MotionStatus::rejected)) throw std::runtime_error("unknown SCMO status");
        f.status = static_cast<MotionStatus>(status);
        f.peak_x = get<std::int32_t>(in); f.peak_y = get<std::int32_t>(in);
        f.pcr = get<float>(in); f.confidence = get<float>(in); f.tx = get<float>(in); f.ty = get<float>(in);
        records.frames.push_back(f);
    }
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("trailing SCMO bytes");
    if (in.bad()) throw std::runtime_error("cannot read SCMO input");
    return records;
}
} // namespace saccade

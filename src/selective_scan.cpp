#include "saccade/selective_scan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace saccade {
namespace {

// The deployed kernel and plugin index every tensor with int32 arithmetic.
constexpr std::size_t kMaxElements = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
constexpr double kFp32Max = std::numeric_limits<float>::max();

[[noreturn]] void reject(const std::string& what) { throw std::invalid_argument("selective_scan: " + what); }

std::size_t elements(std::initializer_list<std::size_t> dims, const char* tensor) {
    std::size_t product = 1;
    for (const auto dim : dims) {
        if (dim != 0 && product > kMaxElements / dim)
            reject(std::string(tensor) + " element count exceeds int32 capacity");
        product *= dim;
    }
    return product;
}

void expect_size(std::span<const float> values, std::size_t count, const char* tensor) {
    if (values.size() != count)
        reject(std::string(tensor) + " has " + std::to_string(values.size()) + " values, shape needs " + std::to_string(count));
}

void expect_finite(std::span<const float> values, const char* tensor) {
    for (std::size_t i = 0; i < values.size(); ++i)
        if (!std::isfinite(values[i])) reject(std::string(tensor) + "[" + std::to_string(i) + "] is not finite");
}

double in_range(double value, const char* quantity, std::size_t b, std::size_t t, std::size_t d) {
    if (!(std::fabs(value) <= kFp32Max))
        throw std::range_error("selective_scan: " + std::string(quantity) + " leaves FP32 range at batch " + std::to_string(b) +
                               ", token " + std::to_string(t) + ", channel " + std::to_string(d));
    return value;
}

std::size_t cross_scan_tokens(std::size_t batch, std::size_t channels, std::size_t height, std::size_t width) {
    if (batch == 0 || channels == 0 || height == 0 || width == 0) throw std::invalid_argument("cross_scan: dimensions must be positive");
    std::size_t product = 4;
    for (const auto dim : {batch, height, width, channels}) {
        if (product > kMaxElements / dim) throw std::invalid_argument("cross_scan: element count exceeds int32 capacity");
        product *= dim;
    }
    return product;
}

} // namespace

double selective_scan_softplus(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }

std::vector<double> selective_scan_reference(const SelectiveScanShape& s, const SelectiveScanInputs& in) {
    if (s.batch == 0 || s.channels == 0) reject("batch and channels must be positive");
    if (s.state == 0 || s.state > 32 || (s.state & (s.state - 1)) != 0)
        reject("state " + std::to_string(s.state) + " must be a power of two in [1, 32]");
    if (s.a_rows != 1 && s.a_rows != s.channels) reject("A rows must be 1 or channels");
    if (s.c_width != 1 && s.c_width != s.state) reject("C width must be 1 or state");
    // Capacity first, so impossible shapes are rejected without touching data.
    const auto ud = elements({s.batch, s.length, s.channels}, "u/delta/y");
    const auto bn = elements({s.batch, s.length, s.state}, "B");
    const auto cn = elements({s.batch, s.length, s.c_width}, "C");
    const auto an = elements({s.a_rows, s.state}, "A");
    expect_size(in.u, ud, "u");
    expect_size(in.delta, ud, "delta");
    expect_size(in.a, an, "A");
    expect_size(in.b, bn, "B");
    expect_size(in.c, cn, "C");
    expect_size(in.d, s.has_d ? s.channels : 0, "D");
    expect_finite(in.u, "u");
    expect_finite(in.delta, "delta");
    expect_finite(in.a, "A");
    expect_finite(in.b, "B");
    expect_finite(in.c, "C");
    expect_finite(in.d, "D");

    const std::size_t L = s.length, Dc = s.channels, N = s.state;
    std::vector<double> y(ud);
    std::vector<double> h(N);
    for (std::size_t b = 0; b < s.batch; ++b) {
        for (std::size_t d = 0; d < Dc; ++d) {
            const float* a_row = in.a.data() + (s.a_rows == 1 ? 0 : d * N);
            std::fill(h.begin(), h.end(), 0.0);
            for (std::size_t t = 0; t < L; ++t) {
                const std::size_t token = b * L + t;
                const double u = in.u[token * Dc + d];
                const double delta = selective_scan_softplus(in.delta[token * Dc + d]);
                double acc = 0.0;
                for (std::size_t n = 0; n < N; ++n) {
                    const double decay = in_range(std::exp(delta * a_row[n]), "decay", b, t, d);
                    const double drive = in_range(delta * in.b[token * N + n] * u, "drive", b, t, d);
                    h[n] = in_range(decay * h[n] + drive, "state", b, t, d);
                    acc += in_range(in.c[token * s.c_width + (s.c_width == 1 ? 0 : n)] * h[n], "C*state", b, t, d);
                }
                if (s.has_d) acc += in_range(static_cast<double>(in.d[d]) * u, "D*u", b, t, d);
                y[token * Dc + d] = in_range(acc, "output", b, t, d);
            }
        }
    }
    return y;
}

std::size_t cross_scan_source(std::size_t direction, std::size_t token, std::size_t height, std::size_t width) {
    if (direction > 3 || height == 0 || width == 0 || token / width >= height) throw std::invalid_argument("cross_scan: token or direction out of range");
    const std::size_t row = token / width, col = token % width;
    const bool flip_h = direction == 1 || direction == 3, flip_w = direction == 1 || direction == 2;
    return (flip_h ? height - 1 - row : row) * width + (flip_w ? width - 1 - col : col);
}

std::vector<float> cross_scan_expand(std::span<const float> x, std::size_t batch, std::size_t channels,
                                     std::size_t height, std::size_t width) {
    const auto total = cross_scan_tokens(batch, channels, height, width);
    if (x.size() != total / 4) throw std::invalid_argument("cross_scan: input size does not match shape");
    for (const auto v : x) if (!std::isfinite(v)) throw std::invalid_argument("cross_scan: input is not finite");
    const std::size_t L = height * width;
    std::vector<float> out(total);
    for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t i = 0; i < batch; ++i)
            for (std::size_t t = 0; t < L; ++t) {
                const auto source = cross_scan_source(k, t, height, width);
                for (std::size_t c = 0; c < channels; ++c)
                    out[((k * batch + i) * L + t) * channels + c] = x[(i * channels + c) * L + source];
            }
    return out;
}

std::vector<double> cross_scan_merge(std::span<const double> y, std::size_t batch, std::size_t channels,
                                     std::size_t height, std::size_t width) {
    const auto total = cross_scan_tokens(batch, channels, height, width);
    if (y.size() != total) throw std::invalid_argument("cross_scan: merge input size does not match shape");
    for (const auto v : y) if (!std::isfinite(v)) throw std::invalid_argument("cross_scan: merge input is not finite");
    const std::size_t L = height * width;
    std::vector<double> out(total / 4);
    // Each flip is an involution, so source(k, p) is also the token of p in direction k.
    for (std::size_t i = 0; i < batch; ++i)
        for (std::size_t c = 0; c < channels; ++c)
            for (std::size_t p = 0; p < L; ++p) {
                double sum = 0.0;
                for (std::size_t k = 0; k < 4; ++k)
                    sum += y[((k * batch + i) * L + cross_scan_source(k, p, height, width)) * channels + c];
                out[(i * channels + c) * L + p] = sum / 4.0;
            }
    return out;
}

} // namespace saccade

#include "saccade/head_reference.hpp"
#include "saccade/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace saccade {
namespace {
using namespace head_policy;
using head_ref::T;

[[noreturn]] void bad(const std::string& why) { throw std::invalid_argument("head reference: " + why); }

std::size_t count(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (const auto d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

T make(std::vector<std::int64_t> shape) {
    const auto n = count(shape);
    return {std::move(shape), std::vector<double>(n), std::vector<double>(n), std::vector<double>(n)};
}

// Each index runs on exactly one thread in a fixed order, so results do not
// depend on the thread count.
template <typename F> void parallel_for(std::int64_t n, F&& f) {
    const auto hw = static_cast<std::int64_t>(std::max(1u, std::thread::hardware_concurrency()));
    const auto workers = std::max<std::int64_t>(1, std::min(n, hw));
    std::atomic<std::int64_t> next{0};
    std::exception_ptr error;
    std::mutex mutex;
    const auto work = [&] {
        try {
            for (std::int64_t i; (i = next++) < n;) f(i);
        } catch (...) {
            std::lock_guard lock(mutex);
            if (!error) error = std::current_exception();
            next = n;
        }
    };
    std::vector<std::thread> pool;
    for (std::int64_t t = 1; t < workers; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

double sigmoid(double x) {
    if (x >= 0) return 1 / (1 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1 + e);
}

// Token t of direction k reads source position t of the k-flipped map:
// 0 identity, 1 flip H and W, 2 flip W, 3 flip H. Flips are involutions, so
// the same map returns the token of a source position.
std::int64_t source(int k, std::int64_t t, std::int64_t height, std::int64_t width, bool direction_identity) {
    std::int64_t h = t / width, w = t % width;
    if (k == 1 && !direction_identity) { h = height - 1 - h; w = width - 1 - w; }
    if (k == 2) w = width - 1 - w;
    if (k == 3) h = height - 1 - h;
    return h * width + w;
}

// One FP32 dot product y = sum_k w_k x_k (+ b) of `terms` terms (bias
// included) in any summation order: 2 terms roundings (products, additions),
// each on a partial value bounded by M = sum |w x| + |b|.
//   bound: (1 + g) sum |w| E_x + g M + 2 terms eta,  g = gamma(terms)
//   sigma: sqrt(sum w^2 s_x^2 + 2 terms u^2 M^2)
struct Dot {
    double terms, g;
    explicit Dot(double t) : terms(t), g(gamma(t)) {}
    double bound(double err, double mag) const { return (1 + g) * err + g * mag + 2 * terms * kUnderflow; }
    double sigma(double var, double mag) const { return std::sqrt(var + 2 * terms * kUnit * kUnit * mag * mag); }
};

} // namespace

double head_policy::gamma(double n) { return n * kUnit / (1 - n * kUnit); }

std::vector<std::int64_t> head_input_shape(int level) {
    const auto i = static_cast<std::size_t>(level);
    return {1, kHeadInChannels.at(i), kHeadSize.at(i), kHeadSize.at(i)};
}

std::vector<std::int64_t> head_output_shape(int output) {
    if (output < 0 || output >= 6) bad("output index out of range");
    const auto s = kHeadSize[static_cast<std::size_t>(output % 3)];
    return {1, output < 3 ? kHeadClasses : kHeadRegs, s, s};
}

namespace head_ref {

double softplus(double x) { return x > 20 ? x : std::log1p(std::exp(x)); }

T from_float(std::span<const float> values, std::vector<std::int64_t> shape) {
    auto t = make(std::move(shape));
    if (values.size() != t.value.size()) bad("input size differs from its shape");
    for (std::size_t i = 0; i < values.size(); ++i) t.value[i] = values[i];
    return t;
}

T conv2d(const T& x, const HeadTensor& w, const HeadTensor* bias, int stride, int pad, bool flipped) {
    if (x.shape.size() != 4 || x.shape[0] != 1 || w.shape.size() != 4 || w.shape[1] != x.shape[1]) bad("conv2d shapes");
    const auto C = x.shape[1], H = x.shape[2], W = x.shape[3], O = w.shape[0], KH = w.shape[2], KW = w.shape[3];
    if (bias && bias->values.size() != static_cast<std::size_t>(O)) bad("conv2d bias");
    const auto Ho = (H + 2 * pad - KH) / stride + 1, Wo = (W + 2 * pad - KW) / stride + 1;
    auto y = make({1, O, Ho, Wo});
    const Dot dot(static_cast<double>(C * KH * KW) + (bias ? 1 : 0));
    std::vector<double> xa(x.value.size()), xs2(x.value.size());
    for (std::size_t i = 0; i < xa.size(); ++i) {
        xa[i] = std::fabs(x.value[i]);
        xs2[i] = x.sigma[i] * x.sigma[i];
    }
    parallel_for(O, [&](std::int64_t o) {
        const auto plane = static_cast<std::size_t>(Ho * Wo);
        std::vector<double> mag(plane), err(plane), var(plane);
        double* acc = y.value.data() + static_cast<std::size_t>(o) * plane;
        for (std::int64_t c = 0; c < C; ++c)
            for (std::int64_t i = 0; i < KH; ++i)
                for (std::int64_t j = 0; j < KW; ++j) {
                    const auto ki = flipped ? KH - 1 - i : i, kj = flipped ? KW - 1 - j : j;
                    const double wv = w.values[static_cast<std::size_t>(((o * C + c) * KH + ki) * KW + kj)], aw = std::fabs(wv), w2 = wv * wv;
                    const auto lo = pad > j ? (pad - j + stride - 1) / stride : 0;
                    const auto hi = std::min(Wo, (W - 1 + pad - j) / stride + 1);
                    for (std::int64_t ho = 0; ho < Ho; ++ho) {
                        const auto hin = ho * stride - pad + i;
                        if (hin < 0 || hin >= H) continue;
                        const auto row = static_cast<std::size_t>((c * H + hin) * W);
                        const auto out = static_cast<std::size_t>(ho * Wo);
                        for (std::int64_t wo = lo; wo < hi; ++wo) {
                            const auto at = row + static_cast<std::size_t>(wo * stride - pad + j), to = out + static_cast<std::size_t>(wo);
                            acc[to] += wv * x.value[at];
                            mag[to] += aw * xa[at];
                            err[to] += aw * x.bound[at];
                            var[to] += w2 * xs2[at];
                        }
                    }
                }
        const double b = bias ? bias->values[static_cast<std::size_t>(o)] : 0.0;
        const auto first = static_cast<std::size_t>(o) * plane;
        for (std::size_t p = 0; p < plane; ++p) {
            acc[p] += b;
            y.bound[first + p] = dot.bound(err[p], mag[p] + std::fabs(b));
            y.sigma[first + p] = dot.sigma(var[p], mag[p] + std::fabs(b));
        }
    });
    return y;
}

T linear(const T& x, const HeadTensor& w, const HeadTensor* bias) {
    if (x.shape.empty() || w.shape.size() != 2 || w.shape[1] != x.shape.back()) bad("linear shapes");
    const auto I = w.shape[1], O = w.shape[0];
    if (bias && bias->values.size() != static_cast<std::size_t>(O)) bad("linear bias");
    auto shape = x.shape;
    shape[shape.size() - 1] = O;
    auto y = make(shape);
    const auto rows = static_cast<std::int64_t>(x.value.size()) / I;
    const Dot dot(static_cast<double>(I) + (bias ? 1 : 0));
    parallel_for(rows, [&](std::int64_t r) {
        const double* xv = x.value.data() + r * I;
        const double* xe = x.bound.data() + r * I;
        const double* xs = x.sigma.data() + r * I;
        for (std::int64_t o = 0; o < O; ++o) {
            const float* wr = w.values.data() + o * I;
            double acc = 0, mag = 0, err = 0, var = 0;
            for (std::int64_t i = 0; i < I; ++i) {
                acc += wr[i] * xv[i];
                mag += std::fabs(wr[i]) * std::fabs(xv[i]);
                err += std::fabs(wr[i]) * xe[i];
                var += static_cast<double>(wr[i]) * wr[i] * xs[i] * xs[i];
            }
            const double b = bias ? bias->values[static_cast<std::size_t>(o)] : 0.0;
            const auto at = static_cast<std::size_t>(r * O + o);
            y.value[at] = acc + b;
            y.bound[at] = dot.bound(err, mag + std::fabs(b));
            y.sigma[at] = dot.sigma(var, mag + std::fabs(b));
        }
    });
    return y;
}

T causal_conv1d(const T& x, const HeadTensor& w, const HeadTensor& bias, bool anticausal) {
    if (x.shape.size() != 3 || w.shape.size() != 3 || w.shape[0] != x.shape[2] || w.shape[1] != 1 || bias.values.size() != static_cast<std::size_t>(w.shape[0]))
        bad("causal_conv1d shapes");
    const auto B = x.shape[0], L = x.shape[1], C = x.shape[2], K = w.shape[2];
    auto y = make(x.shape);
    const Dot dot(static_cast<double>(K + 1));
    for (std::int64_t b = 0; b < B; ++b)
        for (std::int64_t t = 0; t < L; ++t)
            for (std::int64_t c = 0; c < C; ++c) {
                double acc = 0, mag = 0, err = 0, var = 0;
                for (std::int64_t k = 0; k < K; ++k) {
                    const auto s = anticausal ? t + k : t + k - (K - 1);
                    if (s < 0 || s >= L) continue;
                    const auto at = static_cast<std::size_t>((b * L + s) * C + c);
                    const double wv = w.values[static_cast<std::size_t>(c * K + k)];
                    acc += wv * x.value[at];
                    mag += std::fabs(wv * x.value[at]);
                    err += std::fabs(wv) * x.bound[at];
                    var += wv * wv * x.sigma[at] * x.sigma[at];
                }
                const double bv = bias.values[static_cast<std::size_t>(c)];
                const auto at = static_cast<std::size_t>((b * L + t) * C + c);
                y.value[at] = acc + bv;
                y.bound[at] = dot.bound(err, mag + std::fabs(bv));
                y.sigma[at] = dot.sigma(var, mag + std::fabs(bv));
            }
    return y;
}

// bound: Lip E_x + eps_t (|silu(x)| + Lip E_x) + floor (|x| + E_x)
// sigma: sqrt((silu'(x) s_x)^2 + (eps_t |silu(x)|)^2) + floor |x|
T silu(const T& x) {
    auto y = make(x.shape);
    for (std::size_t i = 0; i < x.value.size(); ++i) {
        const double v = x.value[i], sg = sigmoid(v), out = v * sg, slope = sg * (1 + v * (1 - sg));
        const double e = kSiluLipschitz * x.bound[i];
        y.value[i] = out;
        y.bound[i] = e + kTranscendental * (std::fabs(out) + e) + kSiluFloor * (std::fabs(v) + x.bound[i]);
        y.sigma[i] = std::hypot(slope * x.sigma[i], kTranscendental * out) + kSiluFloor * std::fabs(v);
    }
    return y;
}

// Elementwise products and sums combine operands that may share upstream
// sources, so their scales add linearly (Minkowski), never in quadrature.
T multiply(const T& a, const T& b) {
    if (a.shape != b.shape) bad("multiply shapes");
    auto y = make(a.shape);
    for (std::size_t i = 0; i < a.value.size(); ++i) {
        const double v = a.value[i] * b.value[i];
        const double e = std::fabs(b.value[i]) * a.bound[i] + std::fabs(a.value[i]) * b.bound[i] + a.bound[i] * b.bound[i];
        y.value[i] = v;
        y.bound[i] = e + kUnit * (std::fabs(v) + e) + kUnderflow;
        y.sigma[i] = std::fabs(b.value[i]) * a.sigma[i] + std::fabs(a.value[i]) * b.sigma[i] + kUnit * std::fabs(v);
    }
    return y;
}

T add(const T& a, const T& b) {
    if (a.shape != b.shape) bad("add shapes");
    auto y = make(a.shape);
    for (std::size_t i = 0; i < a.value.size(); ++i) {
        const double v = a.value[i] + b.value[i], e = a.bound[i] + b.bound[i];
        y.value[i] = v;
        y.bound[i] = (1 + kUnit) * e + kUnit * std::fabs(v); // a sum's subnormal result is exact
        y.sigma[i] = a.sigma[i] + b.sigma[i] + kUnit * std::fabs(v);
    }
    return y;
}

namespace {
void copy_element(const T& from, std::size_t i, T& to, std::size_t j) {
    to.value[j] = from.value[i];
    to.bound[j] = from.bound[i];
    to.sigma[j] = from.sigma[i];
}
} // namespace

T slice_last(const T& x, std::int64_t begin, std::int64_t end) {
    if (x.shape.empty()) bad("slice of a scalar");
    const auto last = x.shape.back();
    if (begin < 0 || end > last || begin >= end) bad("slice range");
    auto shape = x.shape;
    shape[shape.size() - 1] = end - begin;
    auto y = make(shape);
    const auto rows = static_cast<std::int64_t>(x.value.size()) / last, n = end - begin;
    for (std::int64_t r = 0; r < rows; ++r)
        for (std::int64_t k = 0; k < n; ++k) copy_element(x, static_cast<std::size_t>(r * last + begin + k), y, static_cast<std::size_t>(r * n + k));
    return y;
}

T pixel_shuffle(const T& x, int r, bool dcr) {
    if (x.shape.size() != 4 || x.shape[0] != 1 || x.shape[1] % (r * r) != 0) bad("pixel_shuffle shape");
    const auto C = x.shape[1] / (r * r), H = x.shape[2], W = x.shape[3];
    auto y = make({1, C, H * r, W * r});
    for (std::int64_t c = 0; c < C; ++c)
        for (std::int64_t i = 0; i < r; ++i)
            for (std::int64_t j = 0; j < r; ++j) {
                const auto in_c = dcr ? (i * r + j) * C + c : c * r * r + i * r + j;
                for (std::int64_t h = 0; h < H; ++h)
                    for (std::int64_t w = 0; w < W; ++w)
                        copy_element(x, static_cast<std::size_t>((in_c * H + h) * W + w), y,
                                     static_cast<std::size_t>((c * H * r + h * r + i) * W * r + w * r + j));
            }
    return y;
}

T concat_channels(const T& a, const T& b) {
    if (a.shape.size() != 4 || b.shape.size() != 4 || a.shape[0] != 1 || b.shape[0] != 1 || a.shape[2] != b.shape[2] || a.shape[3] != b.shape[3])
        bad("concat shapes");
    auto y = make({1, a.shape[1] + b.shape[1], a.shape[2], a.shape[3]});
    for (std::size_t i = 0; i < a.value.size(); ++i) copy_element(a, i, y, i);
    for (std::size_t i = 0; i < b.value.size(); ++i) copy_element(b, i, y, a.value.size() + i);
    return y;
}

T cross_expand(const T& x, bool direction_identity) {
    if (x.shape.size() != 4 || x.shape[0] != 1) bad("cross_expand shape");
    const auto C = x.shape[1], H = x.shape[2], W = x.shape[3], L = H * W;
    auto y = make({4, L, C});
    for (int k = 0; k < 4; ++k)
        for (std::int64_t t = 0; t < L; ++t) {
            const auto p = source(k, t, H, W, direction_identity);
            for (std::int64_t c = 0; c < C; ++c) copy_element(x, static_cast<std::size_t>(c * L + p), y, static_cast<std::size_t>((k * L + t) * C + c));
        }
    return y;
}

// SCAN_CUDA §5 merge, (((y0 + y1) + y2) + y3) * 0.25:
//   bound: (1 + g4) sum E_k / 4 + g4 sum |y_k| / 4 + eta
//   sigma: sum s_k / 4 + sqrt(3) u sum |y_k| / 4 (the four directions carry the
//          same token, so their scales add linearly)
T cross_merge(const T& y, std::int64_t height, std::int64_t width, bool direction_identity, bool mismatch) {
    const auto L = height * width;
    if (y.shape.size() != 3 || y.shape[0] != 4 || y.shape[1] != L) bad("cross_merge shape");
    const auto C = y.shape[2];
    auto out = make({1, C, height, width});
    const double g4 = gamma(4);
    for (std::int64_t p = 0; p < L; ++p)
        for (std::int64_t c = 0; c < C; ++c) {
            double sum = 0, err = 0, mag = 0, scale = 0;
            for (int k = 0; k < 4; ++k) {
                const int restore = mismatch && k >= 2 ? 5 - k : k;
                const auto at = static_cast<std::size_t>((k * L + source(restore, p, height, width, direction_identity)) * C + c);
                sum += y.value[at];
                err += y.bound[at];
                mag += std::fabs(y.value[at]);
                scale += y.sigma[at];
            }
            const auto to = static_cast<std::size_t>(c * L + p);
            out.value[to] = sum / 4;
            out.bound[to] = (1 + g4) * err / 4 + g4 * mag / 4 + kUnderflow;
            out.sigma[to] = scale / 4 + std::sqrt(3.0) * kUnit * mag / 4;
        }
    return out;
}

T neg_exp(const HeadTensor& a_log) {
    auto a = make(a_log.shape);
    for (std::size_t i = 0; i < a.value.size(); ++i) {
        a.value[i] = -std::exp(static_cast<double>(a_log.values[i]));
        a.bound[i] = a.sigma[i] = kTranscendental * std::fabs(a.value[i]);
    }
    return a;
}

// Per (sequence, channel) lane and state n, with q = exp(dt A_n), r = dt B_n u:
// own rounding: S02's FP32 bound (SCAN_CUDA §5) in both measures;
// inputs, first order, dh_t = q dh_{t-1} + (h_{t-1} q A_n + B_n u) d(dt) + dt u dB_n + dt B_n du + h_{t-1} q dt dA_n:
//   bound: every term aligned;
//   sigma: the step-t terms (one token) add linearly, distinct steps in
//          quadrature, the shared A_n coherently over steps; states add
//          linearly into y (they share dt, u and C).
T selective_scan(const T& u, const T& delta, const T& a, const T& b, const T& c, const HeadTensor* d, bool c_first_state_only,
                 const HeadTensor* delta_bias_after) {
    if (u.shape.size() != 3 || delta.shape != u.shape || a.shape.size() != 2 || a.shape[0] != 1) bad("scan shapes");
    const auto Bt = u.shape[0], L = u.shape[1], Dc = u.shape[2], N = a.shape[1];
    if (b.shape != std::vector<std::int64_t>{Bt, L, N} || c.shape != std::vector<std::int64_t>{Bt, L, 1}) bad("scan B/C shapes");
    if (d && d->values.size() != static_cast<std::size_t>(Dc)) bad("scan D shape");
    // SCAN_CUDA §5 constants of the S02 FP32 algorithm.
    constexpr double eps_delta = 6 * kUnit, eta_decay = 0x1p-148;
    int levels = 0;
    while ((std::int64_t{1} << levels) < N) ++levels;
    const double g = gamma(levels + 2);
    auto y = make(u.shape);
    parallel_for(Bt * Dc, [&](std::int64_t lane) {
        const auto bb = lane / Dc, ch = lane % Dc;
        const auto states = static_cast<std::size_t>(N);
        std::vector<double> h(states), e(states), worst(states), spread(states), shared_a(states);
        for (std::int64_t t = 0; t < L; ++t) {
            const auto token = bb * L + t;
            const auto at = static_cast<std::size_t>(token * Dc + ch);
            const double uu = u.value[at], raw = delta.value[at], slope = sigmoid(raw); // softplus' = sigmoid
            double dt = softplus(raw);
            if (delta_bias_after) dt += delta_bias_after->values[static_cast<std::size_t>(ch)];
            const auto ct = static_cast<std::size_t>(token);
            double out = 0, sum_c_e = 0, scale = 0, in_bound = 0, in_sigma = 0, sum_h = 0, a_var = 0;
            for (std::int64_t n = 0; n < N; ++n) {
                const auto k = static_cast<std::size_t>(n);
                const double av = a.value[k], arg = dt * av, q = std::exp(arg);
                const auto bat = static_cast<std::size_t>(token * N + n);
                const double bv = b.value[bat], r = dt * bv * uu, prev = h[k], next = q * prev + r;
                // S02 FP32 rounding of decay, drive and state (SCAN_CUDA §5).
                const double rho = std::expm1(std::fabs(arg) * (eps_delta + kUnit)) * (1 + 4 * kUnit) + 4 * kUnit;
                const double keep = q == 0 ? 0 : q * (1 + rho), lost = q == 0 ? 0 : rho * q;
                e[k] = ((keep + eta_decay) * e[k] + (lost + eta_decay) * std::fabs(prev) + (eps_delta + 2 * kUnit) * std::fabs(r) +
                        kUnit * std::fabs(next) + 2 * kUnderflow) / (1 - kUnit);
                // First-order effect of the inputs on the exact recurrence.
                const double ddt = std::fabs(prev * q * av + bv * uu), ddb = std::fabs(dt * uu), ddu = std::fabs(dt * bv);
                worst[k] = q * worst[k] + ddt * slope * delta.bound[at] + ddb * b.bound[bat] + ddu * u.bound[at] + std::fabs(prev) * q * dt * a.bound[k];
                const double step = ddt * slope * delta.sigma[at] + ddb * b.sigma[bat] + ddu * u.sigma[at];
                spread[k] = std::sqrt(q * q * spread[k] * spread[k] + step * step);
                shared_a[k] = q * shared_a[k] + prev * q * dt; // signed d h / d A_n
                h[k] = next;
                const double cn = c_first_state_only && n > 0 ? 0.0 : c.value[ct];
                const bool reads_c = !(c_first_state_only && n > 0);
                out += cn * next;
                sum_c_e += std::fabs(cn) * e[k];
                scale += std::fabs(cn * next);
                in_bound += std::fabs(cn) * worst[k]; // worst[] already carries A_n's bound per step
                in_sigma += std::fabs(cn) * spread[k];
                a_var += std::pow(cn * shared_a[k] * a.sigma[k], 2);
                if (reads_c) sum_h += next;
            }
            in_bound += std::fabs(sum_h) * c.bound[ct];
            in_sigma += std::sqrt(a_var) + std::fabs(sum_h) * c.sigma[ct];
            if (d) {
                const double dv = d->values[static_cast<std::size_t>(ch)];
                out += dv * uu;
                scale += std::fabs(dv * uu);
                in_bound += std::fabs(dv) * u.bound[at];
                in_sigma += std::fabs(dv) * u.sigma[at];
            }
            const double own = (1 + g) * sum_c_e + g * scale + static_cast<double>(N + 1) * kUnderflow;
            y.value[at] = out;
            y.bound[at] = own + in_bound;
            y.sigma[at] = own + in_sigma;
        }
    });
    return y;
}

} // namespace head_ref

HeadReferenceResult head_reference(const HeadWeights& w, const std::array<std::span<const float>, 3>& features, const HeadSemantics& s,
                                   bool keep_intermediates) {
    validate_head_weights(w);
    std::array<T, 3> inputs;
    for (int level = 0; level < kHeadLevels; ++level)
        inputs[static_cast<std::size_t>(level)] = head_ref::from_float(features[static_cast<std::size_t>(level)], head_input_shape(level));
    return head_reference_from_refs(w, inputs, s, keep_intermediates);
}

HeadReferenceResult head_reference_from_refs(const HeadWeights& w, const std::array<HeadTensorRef, 3>& inputs, const HeadSemantics& s,
                                             bool keep_intermediates) {
    validate_head_weights(w);
    for (int level = 0; level < kHeadLevels; ++level) {
        const auto& x = inputs[static_cast<std::size_t>(level)];
        if (x.shape != head_input_shape(level) || x.value.size() != count(x.shape) || x.bound.size() != x.value.size() ||
            x.sigma.size() != x.value.size())
            bad("input " + std::to_string(level) + " differs from its shape");
    }
    using namespace head_ref;
    HeadReferenceResult result;
    for (int level = 0; level < kHeadLevels; ++level) {
        const std::string name = kHeadLevelNames[static_cast<std::size_t>(level)];
        const int weights_level = s.swap_level_weights && level > 0 ? 3 - level : level;
        const std::string p = std::string(kHeadLevelNames[static_cast<std::size_t>(weights_level)]) + ".";
        const auto W = [&](const char* key) -> const HeadTensor& { return w.at(p + key); };
        const auto keep = [&](const char* step, const T& t) {
            if (keep_intermediates) result.intermediates.emplace_back(name + "." + step, t);
        };
        const auto size = kHeadSize[static_cast<std::size_t>(level)], small = size / kHeadReduction;
        const auto& x = inputs[static_cast<std::size_t>(level)];
        const auto a = conv2d(x, W("input_proj.weight"), &W("input_proj.bias"), 1, 0);
        keep("input_proj", a);
        const auto down = conv2d(a, W("downsample.weight"), &W("downsample.bias"), static_cast<int>(kHeadReduction), 0);
        keep("downsample", down);
        const auto tokens = cross_expand(down, s.direction_identity);
        keep("tokens", tokens);
        const auto xz = linear(tokens, W("mamba.in_proj.weight"), nullptr);
        keep("in_proj", xz);
        const auto xs = slice_last(xz, s.swap_xz ? kHeadDInner : 0, s.swap_xz ? 2 * kHeadDInner : kHeadDInner);
        const auto z = slice_last(xz, s.swap_xz ? 0 : kHeadDInner, s.swap_xz ? kHeadDInner : 2 * kHeadDInner);
        const auto conv = causal_conv1d(xs, W("mamba.conv1d.weight"), W("mamba.conv1d.bias"), s.anticausal_conv);
        keep("conv1d", conv);
        const auto u = silu(conv);
        keep("u", u);
        const auto dbc = linear(u, W("mamba.x_proj.weight"), nullptr);
        keep("x_proj", dbc);
        const auto dt = slice_last(dbc, s.swap_dt_b ? kHeadDState : 0, s.swap_dt_b ? 2 * kHeadDState : kHeadDState);
        const auto bm = slice_last(dbc, s.swap_dt_b ? 0 : kHeadDState, s.swap_dt_b ? kHeadDState : 2 * kHeadDState);
        const auto cm = slice_last(dbc, 2 * kHeadDState, 2 * kHeadDState + kHeadCRank);
        const auto delta = linear(dt, W("mamba.dt_proj.weight"), s.delta_bias_after_softplus ? nullptr : &W("mamba.dt_proj.bias"));
        keep("delta", delta);
        const auto am = neg_exp(W("mamba.A_log"));
        const auto y = selective_scan(u, delta, am, bm, cm, s.no_d_skip ? nullptr : &W("mamba.D"), s.c_first_state_only,
                                      s.delta_bias_after_softplus ? &W("mamba.dt_proj.bias") : nullptr);
        keep("scan", y);
        const auto gated = multiply(y, silu(z));
        keep("gate", gated);
        const auto o = linear(gated, W("mamba.out_proj.weight"), nullptr);
        keep("out_proj", o);
        const auto r = s.no_residual ? o : add(o, tokens);
        keep("residual", r);
        const auto merged = cross_merge(r, small, small, s.direction_identity, s.merge_mismatch);
        keep("merge", merged);
        const auto up = conv2d(merged, W("upsample.weight"), &W("upsample.bias"), 1, 1, s.flipped_kernels);
        keep("upsample", up);
        const auto shuffled = pixel_shuffle(up, static_cast<int>(kHeadReduction), s.pixel_shuffle_dcr);
        keep("pixel_shuffle", shuffled);
        const auto cat = s.concat_reversed ? concat_channels(shuffled, a) : concat_channels(a, shuffled);
        const auto cls_hidden = silu(conv2d(cat, W("cls.conv0.weight"), &W("cls.conv0.bias"), 1, 1, s.flipped_kernels));
        keep("cls_hidden", cls_hidden);
        result.outputs[static_cast<std::size_t>(level)] = conv2d(cls_hidden, W("cls.conv1.weight"), &W("cls.conv1.bias"), 1, 0);
        const auto reg_hidden = silu(conv2d(cat, W("reg.conv0.weight"), &W("reg.conv0.bias"), 1, 1, s.flipped_kernels));
        keep("reg_hidden", reg_hidden);
        result.outputs[static_cast<std::size_t>(3 + level)] = conv2d(reg_hidden, W("reg.conv1.weight"), &W("reg.conv1.bias"), 1, 0);
    }
    return result;
}

namespace {
struct XorShift {
    std::uint64_t state;
    double uniform() { // [0, 1) with 53 random bits, xorshift64*
        state ^= state >> 12; state ^= state << 25; state ^= state >> 27;
        return static_cast<double>((state * 0x2545F4914F6CDD1DULL) >> 11) * 0x1p-53;
    }
};
} // namespace

std::array<std::vector<float>, 3> head_test_features(const std::string& kind, std::uint64_t seed) {
    std::array<std::vector<float>, 3> out;
    for (int level = 0; level < 3; ++level) {
        const auto shape = head_input_shape(level);
        const auto C = shape[1], H = shape[2], W = shape[3];
        auto& v = out[static_cast<std::size_t>(level)];
        v.resize(count(shape));
        if (kind == "asymmetric") {
            const double phase = 0.013 * static_cast<double>(seed % 1000);
            for (std::int64_t c = 0; c < C; ++c)
                for (std::int64_t h = 0; h < H; ++h)
                    for (std::int64_t w = 0; w < W; ++w)
                        v[static_cast<std::size_t>((c * H + h) * W + w)] = static_cast<float>(
                            0.5 * std::sin(phase + 0.37 * static_cast<double>(c) + 0.71 * static_cast<double>(h * (1 + c % 3)) - 0.23 * static_cast<double>(w)) +
                            0.002 * (static_cast<double>(h) - 0.3 * static_cast<double>(w)) + 0.01 * static_cast<double>(c % 7));
        } else if (kind == "random") {
            XorShift g{0x9E3779B97F4A7C15ULL ^ (seed * 0x100000001B3ULL + static_cast<std::uint64_t>(level) + 1)};
            for (auto& x : v) {
                double sum = -6;
                for (int k = 0; k < 12; ++k) sum += g.uniform();
                const double z = 1.5 * sum;
                x = static_cast<float>(z * sigmoid(z));
            }
        } else {
            bad("unknown feature kind " + kind);
        }
    }
    return out;
}

HeadWeights synthetic_head_weights(std::uint64_t seed) {
    HeadWeights w;
    XorShift g{0xD1B54A32D192ED03ULL ^ seed};
    for (const auto& p : head_parameter_schema()) {
        HeadTensor t{p.shape, std::vector<float>(count(p.shape))};
        if (p.name.ends_with("A_log")) {
            for (std::size_t n = 0; n < t.values.size(); ++n) t.values[n] = static_cast<float>(std::log(static_cast<double>(n + 1)));
        } else if (p.name.ends_with("mamba.D")) {
            std::fill(t.values.begin(), t.values.end(), 1.0f);
        } else {
            // Biases take the fan-in of their layer's weight, which precedes them in the schema.
            const auto& shape = p.shape.size() > 1 ? p.shape : w.tensors.at(p.name.substr(0, p.name.size() - 4) + "weight").shape;
            double fan_in = 1;
            for (std::size_t k = 1; k < shape.size(); ++k) fan_in *= static_cast<double>(shape[k]);
            const double scale = 1 / std::sqrt(fan_in);
            for (auto& x : t.values) x = static_cast<float>(scale * (2 * g.uniform() - 1));
        }
        w.tensors.emplace(p.name, std::move(t));
    }
    w.source_sha256 = sha256_hex("synthetic-head-weights/" + std::to_string(seed));
    w.source_archive = "synthetic";
    w.source_epoch = -1;
    validate_head_weights(w);
    return w;
}

} // namespace saccade

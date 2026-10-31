#include "saccade/file_io.hpp"
#include "saccade/head_reference.hpp"
#include "saccade/selective_scan.hpp"
#include "saccade/sha256.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// S04 reference checks (docs/HEAD_TRT.md §4.2). Every primitive the head uses
// is checked on small cases whose expected values are written out by hand (or
// by bc for SiLU); the S01 binary64 scan is the comparator for the scan; an
// FP32 host evaluation of each primitive must stay inside its bound; and every
// semantic variant of HeadSemantics must leave the policy's 2E band of the real
// head on the deployment shapes. No GPU, TensorRT or S02 code is involved.
namespace {
using saccade::HeadTensor;
using saccade::HeadTensorRef;
namespace ref = saccade::head_ref;

std::size_t checks = 0;
std::ostringstream report;
void check(bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void expect_values(const HeadTensorRef& t, const std::vector<double>& want, const std::string& name, double tol = 0) {
    check(t.value.size() == want.size(), name + ": element count");
    for (std::size_t i = 0; i < want.size(); ++i) {
        std::ostringstream m;
        m << name << ": element " << i << " = " << std::setprecision(17) << t.value[i] << ", expected " << want[i];
        check(std::fabs(t.value[i] - want[i]) <= tol, m.str());
    }
    report << "primitive\t" << name << "\tPASS\n";
}
HeadTensorRef tensor(std::vector<std::int64_t> shape, std::vector<float> v) { return ref::from_float(v, std::move(shape)); }
HeadTensor weight(std::vector<std::int64_t> shape, std::vector<float> v) { return {std::move(shape), std::move(v)}; }

void primitives() {
    // 1x1 conv: x0 = 1..4, x1 = 5..8; outputs x0 + 0.5, x1, x0 - x1 + 1.
    const auto x = tensor({1, 2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    const auto w1 = weight({3, 2, 1, 1}, {1, 0, 0, 1, 1, -1}), b1 = weight({3}, {0.5f, 0, 1});
    expect_values(ref::conv2d(x, w1, &b1, 1, 0), {1.5, 2.5, 3.5, 4.5, 5, 6, 7, 8, -3, -3, -3, -3}, "conv1x1");
    // 4x4 stride 4 on x[h][w] = 8h + w (4 x 8): taps (0,1) weight 1 and (2,3) weight 10 pick
    // x[0][1] + 10 x[2][3] = 1 + 190 and x[0][5] + 10 x[2][7] = 5 + 230.
    std::vector<float> ramp(32);
    for (int i = 0; i < 32; ++i) ramp[static_cast<std::size_t>(i)] = static_cast<float>(i);
    std::vector<float> k4(16, 0);
    k4[1] = 1; k4[11] = 10;
    expect_values(ref::conv2d(tensor({1, 1, 4, 8}, ramp), weight({1, 1, 4, 4}, k4), nullptr, 4, 0), {191, 235}, "conv4x4_stride4");
    // 3x3 pad 1 with only the top-left tap: out[h][w] = x[h-1][w-1]; flipped reads x[h+1][w+1].
    const auto x9 = tensor({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    std::vector<float> tl(9, 0);
    tl[0] = 1;
    expect_values(ref::conv2d(x9, weight({1, 1, 3, 3}, tl), nullptr, 1, 1), {0, 0, 0, 0, 1, 2, 0, 4, 5}, "conv3x3_pad1_correlation");
    expect_values(ref::conv2d(x9, weight({1, 1, 3, 3}, tl), nullptr, 1, 1, true), {5, 6, 0, 8, 9, 0, 0, 0, 0}, "conv3x3_flipped_variant");
    // Linear over the last axis with bias.
    const auto lw = weight({2, 3}, {1, 2, 3, -1, 0, 1}), lb = weight({2}, {10, 20});
    expect_values(ref::linear(tensor({2, 3}, {1, 1, 1, 2, 0, -1}), lw, &lb), {16, 20, 9, 17}, "linear");
    // Causal depthwise conv1d, kernel [1, 10, 100, 1000] over 1..5: y[t] = sum_k w[k] x[t + k - 3].
    const auto cw = weight({1, 1, 4}, {1, 10, 100, 1000}), cb = weight({1}, {0});
    const auto seq = tensor({1, 5, 1}, {1, 2, 3, 4, 5});
    expect_values(ref::causal_conv1d(seq, cw, cb), {1000, 2100, 3210, 4321, 5432}, "causal_conv1d");
    expect_values(ref::causal_conv1d(seq, cw, cb, true), {4321, 5432, 543, 54, 5}, "anticausal_conv1d_variant");
    // SiLU: values from bc -l, x / (1 + e(-x)).
    expect_values(ref::silu(tensor({4}, {0, 1, -1, 20})), {0, 0.73105857863000487925, -0.26894142136999512075, 19.999999958776927636195928}, "silu", 1e-14); // a few binary64 ulps at 20
    // PixelShuffle r = 2, C = 2, in channel k = k: CRD out[c][i][j] = in[4c + 2i + j]; DCR = in[(2i + j) 2 + c].
    const auto ps = tensor({1, 8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
    expect_values(ref::pixel_shuffle(ps, 2), {0, 1, 2, 3, 4, 5, 6, 7}, "pixel_shuffle_crd");
    expect_values(ref::pixel_shuffle(ps, 2, true), {0, 2, 4, 6, 1, 3, 5, 7}, "pixel_shuffle_dcr_variant");
    // PixelShuffle spatial placement: one channel group of a 2x1 map.
    expect_values(ref::pixel_shuffle(tensor({1, 4, 1, 2}, {1, 2, 3, 4, 5, 6, 7, 8}), 2), {1, 3, 2, 4, 5, 7, 6, 8}, "pixel_shuffle_spatial");
    expect_values(ref::concat_channels(tensor({1, 1, 1, 2}, {1, 2}), tensor({1, 2, 1, 2}, {3, 4, 5, 6})), {1, 2, 3, 4, 5, 6}, "concat_order");
    expect_values(ref::slice_last(tensor({2, 4}, {0, 1, 2, 3, 4, 5, 6, 7}), 1, 3), {1, 2, 5, 6}, "slice_last");
    expect_values(ref::multiply(tensor({2}, {3, -2}), tensor({2}, {0.5f, 4})), {1.5, -8}, "multiply");
    expect_values(ref::add(tensor({2}, {3, -2}), tensor({2}, {0.5f, 4})), {3.5, 2}, "add");
    // Cross-Scan on a 2x3 map holding its own position p = 3h + w (one channel):
    // direction 0 identity, 1 flip H and W, 2 flip W, 3 flip H.
    const auto map = tensor({1, 1, 2, 3}, {0, 1, 2, 3, 4, 5});
    const auto tokens = ref::cross_expand(map);
    expect_values(tokens, {0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0, 2, 1, 0, 5, 4, 3, 3, 4, 5, 0, 1, 2}, "cross_expand_order");
    for (std::size_t k = 0; k < 4; ++k)
        for (std::size_t t = 0; t < 6; ++t)
            check(tokens.value[k * 6 + t] == static_cast<double>(saccade::cross_scan_source(k, t, 2, 3)), "expand agrees with S01 cross_scan_source");
    expect_values(ref::cross_merge(tokens, 2, 3), {0, 1, 2, 3, 4, 5}, "cross_merge_restores");
    // Merge averages the four directions: a constant offset per direction averages out.
    auto offset = tokens;
    for (std::size_t i = 0; i < 24; ++i) offset.value[i] += static_cast<double>(i / 6) * 4 - 6;
    expect_values(ref::cross_merge(offset, 2, 3), {0, 1, 2, 3, 4, 5}, "cross_merge_mean");
    const auto a = ref::neg_exp(weight({1, 3}, {0, std::log(2.0f), std::log(16.0f)}));
    expect_values(a, {-1, -2, -16}, "neg_exp", 2e-6);
    check(ref::softplus(20) == std::log1p(std::exp(20.0)) && ref::softplus(20.5) == 20.5, "softplus threshold 20");
}

// The head's scan on binary64 inputs equals the S01 reference on the same FP32
// values (both binary64 with the same order), for the m shapes and shared A,
// rank-1 C and D.
void scan_against_s01() {
    for (const std::size_t L : {25u, 100u}) {
        const saccade::SelectiveScanShape s{4, L, 256, 16, 1, 1, true};
        std::vector<float> u(4 * L * 256), delta(u.size()), a(16), b(4 * L * 16), c(4 * L), d(256);
        std::uint64_t st = 0x5eed;
        const auto next = [&](float lo, float hi) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            return lo + (hi - lo) * static_cast<float>(static_cast<double>(st >> 40) * 0x1p-24);
        };
        for (auto& v : u) v = next(-2, 2);
        for (auto& v : delta) v = next(-6, 3);
        for (std::size_t n = 0; n < 16; ++n) a[n] = -static_cast<float>(n + 1);
        for (auto& v : b) v = next(-1.5f, 1.5f);
        for (auto& v : c) v = next(-1.5f, 1.5f);
        for (auto& v : d) v = next(-1, 1);
        const auto want = saccade::selective_scan_reference(s, {u, delta, a, b, c, d});
        const auto at = [&](std::vector<float>& v, std::vector<std::int64_t> shape) { return ref::from_float(v, std::move(shape)); };
        const auto L64 = static_cast<std::int64_t>(L);
        const HeadTensor dd{{256}, d};
        const auto got = ref::selective_scan(at(u, {4, L64, 256}), at(delta, {4, L64, 256}), at(a, {1, 16}), at(b, {4, L64, 16}), at(c, {4, L64, 1}), &dd);
        double worst = 0;
        for (std::size_t i = 0; i < want.size(); ++i) worst = std::max(worst, std::fabs(got.value[i] - want[i]));
        check(worst == 0, "head scan equals the S01 reference bit for bit (L=" + std::to_string(L) + ", max diff " + std::to_string(worst) + ")");
        report << "scan_vs_s01\tL=" << L << "\tmax_abs_diff=" << worst << "\tPASS\n";
        // First-order sensitivity: moving each input by up to its bound moves the
        // exact output by at most the bound attributed to it.
        auto pu = at(u, {4, L64, 256}), pd = at(delta, {4, L64, 256}), pb = at(b, {4, L64, 16}), pc = at(c, {4, L64, 1});
        auto pa = at(a, {1, 16});
        std::uint64_t sign = 0x1234567;
        const auto perturb = [&](HeadTensorRef& t, double rel) {
            auto moved = t;
            for (std::size_t i = 0; i < t.value.size(); ++i) {
                t.bound[i] = rel * std::fabs(t.value[i]) + 1e-9;
                sign ^= sign << 13; sign ^= sign >> 7; sign ^= sign << 17;
                moved.value[i] += (sign & 1 ? 1 : -1) * t.bound[i];
            }
            return moved;
        };
        const auto mu = perturb(pu, 1e-5), md = perturb(pd, 1e-5), mb = perturb(pb, 1e-5), mc = perturb(pc, 1e-5), ma = perturb(pa, 1e-5);
        const auto bounded = ref::selective_scan(pu, pd, pa, pb, pc, &dd);
        const auto moved = ref::selective_scan(mu, md, ma, mb, mc, &dd);
        double ratio = 0;
        for (std::size_t i = 0; i < want.size(); ++i) ratio = std::max(ratio, std::fabs(moved.value[i] - got.value[i]) / bounded.bound[i]);
        check(ratio <= 1, "perturbed inputs stay inside the scan bound (ratio " + std::to_string(ratio) + ")");
        report << "scan_input_sensitivity\tL=" << L << "\tmax_change_over_bound=" << ratio << "\tPASS\n";
    }
}

// FP32 host evaluations of each primitive in a different summation order must
// lie inside the bound (factor 1): the bound covers real FP32 rounding.
void fp32_inside_bounds() {
    std::uint64_t st = 0xabcdef;
    const auto next = [&](float lo, float hi) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        return lo + (hi - lo) * static_cast<float>(static_cast<double>(st >> 40) * 0x1p-24);
    };
    const std::int64_t C = 96, H = 7, W = 9, O = 5;
    std::vector<float> xv(static_cast<std::size_t>(C * H * W)), wv(static_cast<std::size_t>(O * C * 9)), bv(static_cast<std::size_t>(O));
    for (auto& v : xv) v = next(-3, 3);
    for (auto& v : wv) v = next(-0.2f, 0.2f);
    for (auto& v : bv) v = next(-1, 1);
    const auto x = ref::from_float(xv, {1, C, H, W});
    const HeadTensor w{{O, C, 3, 3}, wv}, b{{O}, bv};
    const auto r = ref::conv2d(x, w, &b, 1, 1);
    double worst = 0;
    for (std::int64_t o = 0; o < O; ++o)
        for (std::int64_t h = 0; h < H; ++h)
            for (std::int64_t q = 0; q < W; ++q) {
                float acc = 0; // reverse channel order, single-precision accumulation
                for (std::int64_t c = C; c-- > 0;)
                    for (std::int64_t i = 0; i < 3; ++i)
                        for (std::int64_t j = 0; j < 3; ++j) {
                            const auto hi = h + i - 1, wi = q + j - 1;
                            if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;
                            acc += wv[static_cast<std::size_t>(((o * C + c) * 3 + i) * 3 + j)] * xv[static_cast<std::size_t>((c * H + hi) * W + wi)];
                        }
                acc += bv[static_cast<std::size_t>(o)];
                const auto at = static_cast<std::size_t>((o * H + h) * W + q);
                worst = std::max(worst, std::fabs(static_cast<double>(acc) - r.value[at]) / r.bound[at]);
            }
    check(worst <= 1, "FP32 conv3x3 inside its bound (ratio " + std::to_string(worst) + ")");
    report << "fp32_inside_bound\tconv3x3\tmax_err_over_bound=" << worst << "\tPASS\n";
    // SiLU through expf, on a sweep that includes the underflow region.
    std::vector<float> sx;
    for (float v = -110; v <= 40; v += 0.37f) sx.push_back(v);
    const auto s = ref::silu(ref::from_float(sx, {static_cast<std::int64_t>(sx.size())}));
    worst = 0;
    for (std::size_t i = 0; i < sx.size(); ++i) {
        const float f = sx[i] / (1.0f + std::exp(-sx[i]));
        worst = std::max(worst, std::fabs(static_cast<double>(f) - s.value[i]) / s.bound[i]);
    }
    check(worst <= 1, "FP32 SiLU inside its bound (ratio " + std::to_string(worst) + ")");
    report << "fp32_inside_bound\tsilu\tmax_err_over_bound=" << worst << "\tPASS\n";
    // Linear K = 256 and the merge.
    std::vector<float> lx(256), lw(256 * 3);
    for (auto& v : lx) v = next(-2, 2);
    for (auto& v : lw) v = next(-0.1f, 0.1f);
    const auto l = ref::linear(ref::from_float(lx, {1, 256}), HeadTensor{{3, 256}, lw}, nullptr);
    worst = 0;
    for (std::size_t o = 0; o < 3; ++o) {
        float acc = 0;
        for (std::size_t i = 256; i-- > 0;) acc += lw[o * 256 + i] * lx[i];
        worst = std::max(worst, std::fabs(static_cast<double>(acc) - l.value[o]) / l.bound[o]);
    }
    check(worst <= 1, "FP32 linear inside its bound (ratio " + std::to_string(worst) + ")");
    report << "fp32_inside_bound\tlinear\tmax_err_over_bound=" << worst << "\tPASS\n";
}


double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::array<std::span<const float>, 3> spans(const std::array<std::vector<float>, 3>& f) { return {f[0], f[1], f[2]}; }

double tolerance(const HeadTensorRef& t, std::size_t i) {
    return saccade::head_policy::kLambda * t.sigma[i] + saccade::head_policy::kFloor;
}

// ------------------------------------------- whole-head FP32 CPU evaluation
// A plain single-precision implementation of the head (each dot product
// accumulated in float with input channels in reverse order, SiLU through
// expf, the scan as h = q h + (dt B) u with softplus = log1pf(expf)). It is a
// model check of HEAD_TRT §4.3 before any GPU run: its error must stay inside
// lambda sigma (and 2 bound). It is not the GPU path and proves nothing about it.
using V = std::vector<float>;

template <typename F> void threads_for(std::int64_t n, F&& f) {
    const auto workers = std::max<std::int64_t>(1, std::min<std::int64_t>(n, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (std::int64_t t = 0; t < workers; ++t)
        pool.emplace_back([&, t] { for (std::int64_t i = t; i < n; i += workers) f(i); });
    for (auto& th : pool) th.join();
}

V conv_f(const V& x, std::int64_t C, std::int64_t H, std::int64_t W, const HeadTensor& w, const HeadTensor& b, int stride, int pad) {
    const auto O = w.shape[0], KH = w.shape[2], KW = w.shape[3], Ho = (H + 2 * pad - KH) / stride + 1, Wo = (W + 2 * pad - KW) / stride + 1;
    V y(static_cast<std::size_t>(O * Ho * Wo));
    threads_for(O, [&](std::int64_t o) {
        for (std::int64_t ho = 0; ho < Ho; ++ho)
            for (std::int64_t wo = 0; wo < Wo; ++wo) {
                float acc = 0;
                for (std::int64_t c = C; c-- > 0;)
                    for (std::int64_t i = 0; i < KH; ++i) {
                        const auto hi = ho * stride - pad + i;
                        if (hi < 0 || hi >= H) continue;
                        for (std::int64_t j = 0; j < KW; ++j) {
                            const auto wi = wo * stride - pad + j;
                            if (wi < 0 || wi >= W) continue;
                            acc += w.values[static_cast<std::size_t>(((o * C + c) * KH + i) * KW + j)] * x[static_cast<std::size_t>((c * H + hi) * W + wi)];
                        }
                    }
                y[static_cast<std::size_t>((o * Ho + ho) * Wo + wo)] = acc + b.values[static_cast<std::size_t>(o)];
            }
    });
    return y;
}
V linear_f(const V& x, const HeadTensor& w, const HeadTensor* b) {
    const auto O = w.shape[0], I = w.shape[1], rows = static_cast<std::int64_t>(x.size()) / I;
    V y(static_cast<std::size_t>(rows * O));
    threads_for(rows, [&](std::int64_t r) {
        for (std::int64_t o = 0; o < O; ++o) {
            float acc = 0;
            for (std::int64_t i = I; i-- > 0;) acc += w.values[static_cast<std::size_t>(o * I + i)] * x[static_cast<std::size_t>(r * I + i)];
            y[static_cast<std::size_t>(r * O + o)] = b ? acc + b->values[static_cast<std::size_t>(o)] : acc;
        }
    });
    return y;
}
float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

std::array<V, 6> fp32_head(const saccade::HeadWeights& wt, const std::array<std::vector<float>, 3>& features) {
    std::array<V, 6> out;
    for (int level = 0; level < 3; ++level) {
        const std::string p = std::string(saccade::kHeadLevelNames[static_cast<std::size_t>(level)]) + ".";
        const auto W = [&](const char* k) -> const HeadTensor& { return wt.at(p + k); };
        const auto C = saccade::kHeadInChannels[static_cast<std::size_t>(level)], S = saccade::kHeadSize[static_cast<std::size_t>(level)], s = S / 4, L = s * s;
        const auto a = conv_f(features[static_cast<std::size_t>(level)], C, S, S, W("input_proj.weight"), W("input_proj.bias"), 1, 0);
        const auto down = conv_f(a, 128, S, S, W("downsample.weight"), W("downsample.bias"), 4, 0);
        V tokens(static_cast<std::size_t>(4 * L * 128));
        for (std::size_t k = 0; k < 4; ++k)
            for (std::int64_t t = 0; t < L; ++t)
                for (std::int64_t c = 0; c < 128; ++c)
                    tokens[(k * static_cast<std::size_t>(L) + static_cast<std::size_t>(t)) * 128 + static_cast<std::size_t>(c)] =
                        down[static_cast<std::size_t>(c * L) + saccade::cross_scan_source(k, static_cast<std::size_t>(t), static_cast<std::size_t>(s), static_cast<std::size_t>(s))];
        const auto xz = linear_f(tokens, W("mamba.in_proj.weight"), nullptr);
        const auto& cw = W("mamba.conv1d.weight");
        const auto& cb = W("mamba.conv1d.bias");
        V u(static_cast<std::size_t>(4 * L * 256)), z(u.size());
        for (std::int64_t b = 0; b < 4; ++b)
            for (std::int64_t t = 0; t < L; ++t)
                for (std::int64_t d = 0; d < 256; ++d) {
                    float acc = 0;
                    for (std::int64_t k = 3; k >= 0; --k) {
                        const auto src = t + k - 3;
                        if (src >= 0) acc += cw.values[static_cast<std::size_t>(d * 4 + k)] * xz[static_cast<std::size_t>((b * L + src) * 512 + d)];
                    }
                    const auto at = static_cast<std::size_t>((b * L + t) * 256 + d);
                    u[at] = silu_f(acc + cb.values[static_cast<std::size_t>(d)]);
                    z[at] = xz[static_cast<std::size_t>((b * L + t) * 512 + 256 + d)];
                }
        const auto dbc = linear_f(u, W("mamba.x_proj.weight"), nullptr);
        V dt(static_cast<std::size_t>(4 * L * 16));
        for (std::int64_t r = 0; r < 4 * L; ++r)
            for (std::int64_t n = 0; n < 16; ++n) dt[static_cast<std::size_t>(r * 16 + n)] = dbc[static_cast<std::size_t>(r * 33 + n)];
        const auto delta = linear_f(dt, W("mamba.dt_proj.weight"), &W("mamba.dt_proj.bias"));
        float A[16];
        for (int n = 0; n < 16; ++n) A[n] = -std::exp(W("mamba.A_log").values[static_cast<std::size_t>(n)]);
        const auto& D = W("mamba.D");
        V gated(u.size());
        threads_for(4 * 256, [&](std::int64_t lane) {
            const auto b = lane / 256, d = lane % 256;
            float h[16] = {};
            for (std::int64_t t = 0; t < L; ++t) {
                const auto row = b * L + t;
                const auto at = static_cast<std::size_t>(row * 256 + d);
                const float x = delta[at], step = x > 20 ? x : std::log1p(std::exp(x)), uu = u[at], c = dbc[static_cast<std::size_t>(row * 33 + 32)];
                float y = 0;
                for (int n = 0; n < 16; ++n) {
                    const float q = std::exp(step * A[n]);
                    h[n] = q * h[n] + (step * dbc[static_cast<std::size_t>(row * 33 + 16 + n)]) * uu;
                    y += c * h[n];
                }
                y += D.values[static_cast<std::size_t>(d)] * uu;
                gated[at] = y * silu_f(z[at]);
            }
        });
        auto r = linear_f(gated, W("mamba.out_proj.weight"), nullptr);
        for (std::size_t i = 0; i < r.size(); ++i) r[i] += tokens[i];
        V merged(static_cast<std::size_t>(128 * L));
        for (std::int64_t c = 0; c < 128; ++c)
            for (std::int64_t q = 0; q < L; ++q) {
                float sum = 0;
                for (std::size_t k = 0; k < 4; ++k)
                    sum += r[(k * static_cast<std::size_t>(L) + saccade::cross_scan_source(k, static_cast<std::size_t>(q), static_cast<std::size_t>(s), static_cast<std::size_t>(s))) * 128 + static_cast<std::size_t>(c)];
                merged[static_cast<std::size_t>(c * L + q)] = sum * 0.25f;
            }
        const auto up = conv_f(merged, 128, s, s, W("upsample.weight"), W("upsample.bias"), 1, 1);
        V cat(static_cast<std::size_t>(256 * S * S));
        std::copy(a.begin(), a.end(), cat.begin());
        for (std::int64_t c = 0; c < 128; ++c)
            for (std::int64_t i = 0; i < 4; ++i)
                for (std::int64_t j = 0; j < 4; ++j)
                    for (std::int64_t h = 0; h < s; ++h)
                        for (std::int64_t w = 0; w < s; ++w)
                            cat[static_cast<std::size_t>(((128 + c) * S + h * 4 + i) * S + w * 4 + j)] = up[static_cast<std::size_t>(((c * 16 + i * 4 + j) * s + h) * s + w)];
        for (const bool cls : {true, false}) {
            auto hidden = conv_f(cat, 256, S, S, W(cls ? "cls.conv0.weight" : "reg.conv0.weight"), W(cls ? "cls.conv0.bias" : "reg.conv0.bias"), 1, 1);
            for (auto& v : hidden) v = silu_f(v);
            out[static_cast<std::size_t>(cls ? level : 3 + level)] =
                conv_f(hidden, 128, S, S, W(cls ? "cls.conv1.weight" : "reg.conv1.weight"), W(cls ? "cls.conv1.bias" : "reg.conv1.bias"), 1, 0);
        }
    }
    return out;
}

void fp32_head_inside_policy(const saccade::HeadWeights& w, const std::string& label, const std::string& kind) {
    const auto features = saccade::head_test_features(kind, 1);
    const auto exact = saccade::head_reference(w, spans(features));
    const auto f = fp32_head(w, features);
    for (std::size_t k = 0; k < 6; ++k) {
        double ratio = 0, bound_ratio = 0, max_abs = 0;
        std::size_t worst = 0, outside = 0;
        for (std::size_t i = 0; i < f[k].size(); ++i) {
            const double err = std::fabs(static_cast<double>(f[k][i]) - exact.outputs[k].value[i]);
            const double r = err / tolerance(exact.outputs[k], i);
            outside += r > 1;
            if (r > ratio) { ratio = r; worst = i; }
            bound_ratio = std::max(bound_ratio, err / (2 * exact.outputs[k].bound[i]));
            max_abs = std::max(max_abs, err);
        }
        report << "fp32_head_model_check\t" << label << '\t' << kind << '\t' << saccade::kHeadOutputNames[k] << "\tmax_abs=" << std::setprecision(3) << max_abs
               << "\tmax_err_over_lambda_sigma=" << ratio << "\tworst_index=" << worst << "\tmax_err_over_2bound=" << bound_ratio << "\toutside=" << outside
               << '\t' << (outside ? "FAIL" : "PASS") << '\n';
        check(outside == 0, std::string("FP32 CPU head inside the policy: ") + saccade::kHeadOutputNames[k]);
    }
}

// Each semantic variant must put elements of the six outputs outside the
// policy band lambda sigma around the real head on the deployment shapes.
void discrimination(const saccade::HeadWeights& w, const std::string& label, bool required) {
    const auto features = saccade::head_test_features("asymmetric", 1);
    auto t0 = now();
    const auto exact = saccade::head_reference(w, spans(features));
    report << "reference_seconds\t" << label << '\t' << std::setprecision(3) << now() - t0 << '\n';
    const auto again = saccade::head_reference(w, spans(features));
    for (std::size_t k = 0; k < 6; ++k) {
        check(again.outputs[k].value == exact.outputs[k].value && again.outputs[k].bound == exact.outputs[k].bound &&
                  again.outputs[k].sigma == exact.outputs[k].sigma,
              "reference is deterministic");
        bool ordered = true;
        for (std::size_t i = 0; i < exact.outputs[k].value.size(); ++i) ordered = ordered && exact.outputs[k].sigma[i] <= exact.outputs[k].bound[i];
        check(ordered, "probabilistic scale never exceeds the worst-case bound");
    }
    using S = saccade::HeadSemantics;
    const std::vector<std::pair<const char*, S>> variants{
        {"pixel_shuffle_dcr", [] { S s; s.pixel_shuffle_dcr = true; return s; }()},
        {"anticausal_conv", [] { S s; s.anticausal_conv = true; return s; }()},
        {"direction_identity", [] { S s; s.direction_identity = true; return s; }()},
        {"merge_mismatch", [] { S s; s.merge_mismatch = true; return s; }()},
        {"concat_reversed", [] { S s; s.concat_reversed = true; return s; }()},
        {"flipped_kernels", [] { S s; s.flipped_kernels = true; return s; }()},
        {"swap_xz", [] { S s; s.swap_xz = true; return s; }()},
        {"swap_dt_b", [] { S s; s.swap_dt_b = true; return s; }()},
        {"c_first_state_only", [] { S s; s.c_first_state_only = true; return s; }()},
        {"delta_bias_after_softplus", [] { S s; s.delta_bias_after_softplus = true; return s; }()},
        {"no_d_skip", [] { S s; s.no_d_skip = true; return s; }()},
        {"no_residual", [] { S s; s.no_residual = true; return s; }()},
        {"swap_level_weights", [] { S s; s.swap_level_weights = true; return s; }()},
    };
    std::string missed;
    for (const auto& [name, semantics] : variants) {
        const auto v = saccade::head_reference(w, spans(features), semantics);
        std::size_t outside = 0, total = 0;
        double ratio = 0;
        for (std::size_t k = 0; k < 6; ++k)
            for (std::size_t i = 0; i < v.outputs[k].value.size(); ++i) {
                const double err = std::fabs(v.outputs[k].value[i] - exact.outputs[k].value[i]), tol = tolerance(exact.outputs[k], i);
                outside += err > tol;
                ratio = std::max(ratio, err / tol);
                ++total;
            }
        report << "semantic_variant\t" << label << '\t' << name << "\toutside_lambda_sigma=" << outside << '/' << total << "\tmax_err_over_tol="
               << std::setprecision(3) << ratio << '\t' << (outside ? "DETECTED" : "MISSED") << '\n';
        if (!outside) missed += std::string(missed.empty() ? "" : ", ") + name;
    }
    report << "semantic_variants\t" << label << "\t" << (required ? "required" : "informational") << "\tmissed=" << (missed.empty() ? "none" : missed) << '\n';
    if (required) check(missed.empty(), "every semantic variant leaves the policy band; missed: " + missed);
    for (std::size_t k = 0; k < 6; ++k) {
        double peak = 0, sigma_peak = 0, bound_peak = 0;
        for (std::size_t i = 0; i < exact.outputs[k].value.size(); ++i) {
            peak = std::max(peak, std::fabs(exact.outputs[k].value[i]));
            sigma_peak = std::max(sigma_peak, tolerance(exact.outputs[k], i));
            bound_peak = std::max(bound_peak, 2 * exact.outputs[k].bound[i]);
        }
        report << "tolerance_scale\t" << label << '\t' << saccade::kHeadOutputNames[k] << "\tmax_abs_ref=" << peak << "\tmax_lambda_sigma=" << sigma_peak
               << "\tmax_2bound=" << bound_peak << '\n';
    }
}

// head_reference_from_refs: on exact (from_float) inputs it is head_reference
// bit for bit; input scales only widen every output scale; shapes are checked.
void from_refs(const saccade::HeadWeights& w) {
    const auto features = saccade::head_test_features("random", 1);
    const auto exact = saccade::head_reference(w, spans(features));
    std::array<HeadTensorRef, 3> inputs;
    for (int level = 0; level < 3; ++level)
        inputs[static_cast<std::size_t>(level)] = ref::from_float(features[static_cast<std::size_t>(level)], saccade::head_input_shape(level));
    const auto same = saccade::head_reference_from_refs(w, inputs);
    for (std::size_t k = 0; k < 6; ++k)
        check(same.outputs[k].value == exact.outputs[k].value && same.outputs[k].bound == exact.outputs[k].bound &&
                  same.outputs[k].sigma == exact.outputs[k].sigma,
              "from_refs on exact inputs equals head_reference bit for bit");
    for (auto& x : inputs)
        for (std::size_t i = 0; i < x.value.size(); ++i) {
            x.sigma[i] = 0x1p-12 * std::fabs(x.value[i]);
            x.bound[i] = 0x1p-10 * std::fabs(x.value[i]);
        }
    const auto widened = saccade::head_reference_from_refs(w, inputs);
    for (std::size_t k = 0; k < 6; ++k) {
        bool values = widened.outputs[k].value == exact.outputs[k].value, wider = true, ordered = true, grew = false;
        for (std::size_t i = 0; i < exact.outputs[k].value.size(); ++i) {
            wider = wider && widened.outputs[k].sigma[i] >= exact.outputs[k].sigma[i] && widened.outputs[k].bound[i] >= exact.outputs[k].bound[i];
            ordered = ordered && widened.outputs[k].sigma[i] <= widened.outputs[k].bound[i];
            grew = grew || widened.outputs[k].sigma[i] > exact.outputs[k].sigma[i];
        }
        check(values && wider && ordered && grew, std::string("input scales widen the scales of ") + saccade::kHeadOutputNames[k]);
    }
    auto wrong = inputs;
    wrong[1].sigma.pop_back();
    bool rejected = false;
    try {
        (void)saccade::head_reference_from_refs(w, wrong);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "from_refs rejects an input whose scale has the wrong size");
    report << "from_refs\tPASS\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 1 && argc != 3) throw std::invalid_argument("usage: head_reference_tests [BUNDLE.swb EXPECTED_SHA256]");
        primitives();
        scan_against_s01();
        fp32_inside_bounds();
        const auto synthetic = saccade::synthetic_head_weights(3);
        for (const char* kind : {"asymmetric", "random"}) fp32_head_inside_policy(synthetic, "synthetic", kind);
        discrimination(synthetic, "synthetic", false); // arbitrary weights: informational
        from_refs(synthetic);
        if (argc == 3) {
            const auto bytes = saccade::read_text_file(argv[1]);
            check(saccade::sha256_hex(bytes) == argv[2], "bundle SHA-256 matches");
            const auto t1 = saccade::decode_head_bundle(bytes);
            for (const char* kind : {"asymmetric", "random"}) fp32_head_inside_policy(t1, "t1_bundle", kind);
            discrimination(t1, "t1_bundle", true);
        } else {
            report << "skipped\tt1_bundle\tno bundle given\n";
        }
        std::cout << report.str() << "head_reference_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << report.str() << "head_reference_tests: " << error.what() << '\n';
        return 1;
    }
}

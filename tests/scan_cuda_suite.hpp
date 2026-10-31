#pragma once

#include "saccade/selective_scan_cuda.hpp"
#include "saccade/selective_scan.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Shared GPU acceptance of the S02 scan contract (docs/SCAN_CUDA.md): the case
// fixtures, the §5 running error bound and its comparison, and the numeric,
// failure and Cross-Scan groups. The GPU and the S01 binary64 reference read
// the same FP32 input bytes; every output element must lie within twice the
// running error bound, computed here in binary64 from the documented FP32
// algorithm. Failures must match the reference's rejection, poison the whole
// output and still mark the status done. The groups execute through a
// ScanBackend: S02 calls the CUDA API directly (selective_scan_cuda_tests),
// S03 runs the same bytes through TensorRT engines (scan_trt_tests).
namespace saccade::scan_test {
using saccade::ScanStatus;
using saccade::SelectiveScanShape;
using saccade::kScanDone;

inline std::size_t checks = 0;
inline std::ostringstream report;

inline void check(bool value, std::string_view message) {
    ++checks;
    if (!value) throw std::runtime_error(std::string(message));
}
template <typename Exception, typename F> void expect_throw(F&& action, std::string_view diagnostic) {
    try { action(); }
    catch (const Exception& error) {
        check(std::string(error.what()).find(diagnostic) != std::string::npos,
              "unexpected diagnostic: " + std::string(error.what()) + "; wanted " + std::string(diagnostic));
        return;
    } catch (const std::exception& error) {
        check(false, "wrong exception type: " + std::string(error.what()) + "; wanted " + std::string(diagnostic));
    }
    check(false, "expected rejection: " + std::string(diagnostic));
}

inline void ck(cudaError_t result, const char* call) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(result));
}
#define CUDA(call) ck((call), #call)

inline std::string hex(std::uint32_t v) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(8) << std::setfill('0') << v;
    return out.str();
}

// ------------------------------------------------------------ device helpers

template <typename T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count) CUDA(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t bytes() const { return count_ * sizeof(T); }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
public:
    Stream() { CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_{};
};

// Pageable sources: cudaMemcpyAsync stages them before returning, and the copy
// stays ordered on the stream. Reads synchronize the stream before returning.
template <typename T> void put(const DeviceBuffer<T>& device, const std::vector<T>& host, cudaStream_t s) {
    if (host.size() != device.size()) throw std::logic_error("upload size mismatch");
    if (!host.empty()) CUDA(cudaMemcpyAsync(device.get(), host.data(), device.bytes(), cudaMemcpyHostToDevice, s));
}
template <typename T> std::vector<T> fetch(const DeviceBuffer<T>& device, cudaStream_t s) {
    std::vector<T> host(device.size());
    if (!host.empty()) CUDA(cudaMemcpyAsync(host.data(), device.get(), device.bytes(), cudaMemcpyDeviceToHost, s));
    CUDA(cudaStreamSynchronize(s));
    return host;
}

constexpr ScanStatus kStatusSentinel{0xFFFFFFFFu, 0x12345678u}; // cleared and completed by every call
constexpr std::uint32_t kPoison = 0x7fc00000u;

// --------------------------------------------------------------- scan cases

struct Case {
    SelectiveScanShape shape;
    std::vector<float> u, delta, a, b, c, d;
    saccade::SelectiveScanInputs inputs() const { return {u, delta, a, b, c, d}; }
    std::vector<double> run() const { return saccade::selective_scan_reference(shape, inputs()); }
};

// Shapes are filled in (batch, length, channels, state, a_rows, c_width, has_d) order.
inline Case filled(SelectiveScanShape s, float u, float delta, float b, float c) {
    const auto tokens = s.batch * s.length;
    return {s, std::vector<float>(tokens * s.channels, u), std::vector<float>(tokens * s.channels, delta),
            std::vector<float>(s.a_rows * s.state, -1.0f), std::vector<float>(tokens * s.state, b),
            std::vector<float>(tokens * s.c_width, c), std::vector<float>(s.has_d ? s.channels : 0, 0.0f)};
}

inline float gen(int a, int m, int offset, float scale) { return static_cast<float>(a % m - offset) / scale; }

// Same generator as tests/selective_scan_tests.cpp.
inline Case varied(SelectiveScanShape s) {
    Case k = filled(s, 0, 0, 0, 0);
    for (std::size_t i = 0; i < k.u.size(); ++i) {
        k.u[i] = gen(static_cast<int>(5 * i + 1), 11, 5, 4);
        k.delta[i] = gen(static_cast<int>(3 * i + 2), 13, 6, 2);
    }
    for (std::size_t i = 0; i < k.b.size(); ++i) k.b[i] = gen(static_cast<int>(7 * i + 3), 9, 4, 4);
    for (std::size_t i = 0; i < k.c.size(); ++i) k.c[i] = gen(static_cast<int>(3 * i + 1), 7, 3, 2);
    for (std::size_t i = 0; i < k.a.size(); ++i) k.a[i] = -static_cast<float>(i % s.state + 1) / 4;
    for (std::size_t i = 0; i < k.d.size(); ++i) k.d[i] = gen(static_cast<int>(i), 3, 1, 2);
    return k;
}

// Deterministic xorshift64; the values only need to be the same bytes for the
// GPU and the reference.
struct Random {
    std::uint64_t state;
    float uniform(float lo, float hi) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        return lo + (hi - lo) * static_cast<float>(static_cast<double>(state >> 40) * 0x1p-24);
    }
};
struct Ranges { float u = 2, delta_lo = -6, delta_hi = 3, a_lo = -4, a_hi = -0.05f, bc = 1.5f, d = 1; };
inline Case random_case(SelectiveScanShape s, std::uint64_t seed, Ranges r = {}) {
    Random g{seed};
    Case k = filled(s, 0, 0, 0, 0);
    for (auto& v : k.u) v = g.uniform(-r.u, r.u);
    for (auto& v : k.delta) v = g.uniform(r.delta_lo, r.delta_hi);
    for (auto& v : k.a) v = g.uniform(r.a_lo, r.a_hi);
    for (auto& v : k.b) v = g.uniform(-r.bc, r.bc);
    for (auto& v : k.c) v = g.uniform(-r.bc, r.bc);
    for (auto& v : k.d) v = g.uniform(-r.d, r.d);
    return k;
}

// ------------------------------------------------- §5 running error bound

constexpr double kU = 0x1p-24, kEpsDelta = 6 * kU, kEta = 0x1p-150, kEtaDecay = 0x1p-148;
inline double gamma_n(int j) { return j * kU / (1 - j * kU); }
inline int log2_state(std::size_t n) { int k = 0; while ((std::size_t{1} << k) < n) ++k; return k; }

inline std::vector<double> scan_bound(const Case& k) {
    const auto& s = k.shape;
    const std::size_t L = s.length, Dc = s.channels, N = s.state;
    const double g = gamma_n(log2_state(N) + 2);
    std::vector<double> bound(s.batch * L * Dc), h(N), e(N);
    for (std::size_t b = 0; b < s.batch; ++b)
        for (std::size_t d = 0; d < Dc; ++d) {
            const float* a_row = k.a.data() + (s.a_rows == 1 ? 0 : d * N);
            std::fill(h.begin(), h.end(), 0.0);
            std::fill(e.begin(), e.end(), 0.0);
            for (std::size_t t = 0; t < L; ++t) {
                const std::size_t token = b * L + t;
                const double u = k.u[token * Dc + d], delta = saccade::selective_scan_softplus(k.delta[token * Dc + d]);
                double sum_c_e = 0, scale = 0;
                for (std::size_t n = 0; n < N; ++n) {
                    const double arg = delta * a_row[n], q = std::exp(arg);
                    const double rho = std::expm1(std::fabs(arg) * (kEpsDelta + kU)) * (1 + 4 * kU) + 4 * kU;
                    const double keep = q == 0 ? 0 : q * (1 + rho), lost = q == 0 ? 0 : rho * q; // q = 0 in both precisions
                    const double r = delta * k.b[token * N + n] * u, next = q * h[n] + r;
                    e[n] = ((keep + kEtaDecay) * e[n] + (lost + kEtaDecay) * std::fabs(h[n]) + (kEpsDelta + 2 * kU) * std::fabs(r) +
                            kU * std::fabs(next) + 2 * kEta) / (1 - kU);
                    h[n] = next;
                    const double c = k.c[token * s.c_width + (s.c_width == 1 ? 0 : n)];
                    sum_c_e += std::fabs(c) * e[n];
                    scale += std::fabs(c * next);
                }
                if (s.has_d) scale += std::fabs(static_cast<double>(k.d[d]) * u);
                bound[token * Dc + d] = (1 + g) * sum_c_e + g * scale + static_cast<double>(N + 1) * kEta;
            }
        }
    return bound;
}

struct Stats { double max_abs = 0, max_rel = 0, max_ratio = 0; std::size_t failures = 0, worst = 0; };

// |gpu - ref| <= 2 * bound for every element; the worst element is the one
// closest to (or furthest past) its threshold.
inline Stats compare(const std::string& name, const std::vector<float>& gpu, const std::vector<double>& ref,
              const std::vector<double>& bound, bool record = true) {
    check(gpu.size() == ref.size() && ref.size() == bound.size(), name + ": output count matches reference");
    Stats st;
    for (std::size_t i = 0; i < gpu.size(); ++i) {
        const double err = std::fabs(static_cast<double>(gpu[i]) - ref[i]), tol = 2 * bound[i];
        const double rel = ref[i] == 0 ? err : err / std::fabs(ref[i]);
        const double ratio = tol > 0 ? err / tol : (err == 0 ? 0 : std::numeric_limits<double>::infinity());
        if (!(err <= tol)) ++st.failures;
        if (!(ratio <= st.max_ratio)) { st.max_ratio = ratio; st.worst = i; }
        st.max_abs = std::max(st.max_abs, err);
        st.max_rel = std::max(st.max_rel, rel);
        if (record && gpu.size() <= 32)
            report << "element\t" << name << '\t' << i << '\t' << std::setprecision(17) << ref[i] << '\t'
                   << std::setprecision(9) << gpu[i] << '\t' << hex(std::bit_cast<std::uint32_t>(gpu[i])) << '\t'
                   << std::setprecision(3) << err << '\t' << tol << '\n';
    }
    if (record) {
        report << "case\t" << name << "\tcount=" << gpu.size() << std::setprecision(3) << "\tmax_abs=" << st.max_abs
               << "\tmax_rel=" << st.max_rel << "\tmax_err_over_tol=" << st.max_ratio << "\tfailures=" << st.failures;
        if (!gpu.empty())
            report << "\tworst_index=" << st.worst << std::setprecision(9) << "\tworst_gpu=" << gpu[st.worst] << '('
                   << hex(std::bit_cast<std::uint32_t>(gpu[st.worst])) << ')' << std::setprecision(17)
                   << "\tworst_ref=" << ref[st.worst] << std::setprecision(3) << "\tworst_tol=" << 2 * bound[st.worst];
        report << '\t' << (st.failures ? "FAIL" : "PASS") << '\n';
    }
    check(st.failures == 0, name + ": every element within 2x the running error bound");
    return st;
}

struct GpuResult { std::vector<float> y; ScanStatus status; };

// Device copies of one shape's six inputs, the output and the status.
struct ScanBuffers {
    SelectiveScanShape shape;
    DeviceBuffer<float> u, delta, a, b, c, d, y;
    DeviceBuffer<ScanStatus> status{1};
    explicit ScanBuffers(const SelectiveScanShape& s)
        : shape(s), u(s.batch * s.length * s.channels), delta(u.size()), a(s.a_rows * s.state),
          b(s.batch * s.length * s.state), c(s.batch * s.length * s.c_width), d(s.has_d ? s.channels : 0), y(u.size()) {}
    void load(const Case& k, cudaStream_t s, bool reset_outputs = true) const {
        put(u, k.u, s); put(delta, k.delta, s); put(a, k.a, s); put(b, k.b, s); put(c, k.c, s); put(d, k.d, s);
        if (reset_outputs) {
            if (y.size()) CUDA(cudaMemsetAsync(y.get(), 0xA5, y.bytes(), s));
            put(status, {kStatusSentinel}, s);
        }
    }
    saccade::SelectiveScanDeviceInputs inputs() const { return {u.get(), delta.get(), a.get(), b.get(), c.get(), d.get()}; }
    void enqueue(cudaStream_t s) const { saccade::selective_scan_cuda(shape, inputs(), y.get(), status.get(), s); }
    GpuResult result(cudaStream_t s) const { return {fetch(y, s), fetch(status, s)[0]}; }
};

struct PipelineResult { std::vector<float> out; ScanStatus expand, scan, merge; };

// Runs one case of a shared group and returns the output and status after the
// stream has finished it.
class ScanBackend {
public:
    virtual ~ScanBackend() = default;
    virtual GpuResult scan(const Case& k, cudaStream_t s) = 0;
    virtual GpuResult expand(const saccade::CrossScanShape& shape, const std::vector<float>& x, cudaStream_t s) = 0;
    virtual GpuResult merge(const saccade::CrossScanShape& shape, const std::vector<float>& y, cudaStream_t s) = 0;
    // x -> tokens, which replace u of k -> scan -> merge, on one stream.
    virtual PipelineResult pipeline(const saccade::CrossScanShape& shape, const std::vector<float>& x, const Case& k,
                                    cudaStream_t s) = 0;
};

// The S02 CUDA API itself, on fresh buffers for every case.
class DirectBackend final : public ScanBackend {
public:
    GpuResult scan(const Case& k, cudaStream_t s) override {
        ScanBuffers buffers(k.shape);
        buffers.load(k, s);
        buffers.enqueue(s);
        return buffers.result(s);
    }
    GpuResult expand(const saccade::CrossScanShape& shape, const std::vector<float>& x, cudaStream_t s) override {
        DeviceBuffer<float> dx(x.size()), dt(x.size() * 4);
        put(dx, x, s);
        put(status_, {kStatusSentinel}, s);
        saccade::cross_scan_expand_cuda(shape, dx.get(), dt.get(), status_.get(), s);
        return GpuResult{fetch(dt, s), fetch(status_, s)[0]};
    }
    GpuResult merge(const saccade::CrossScanShape& shape, const std::vector<float>& y, cudaStream_t s) override {
        DeviceBuffer<float> dy(y.size()), dout(y.size() / 4);
        put(dy, y, s);
        put(status_, {kStatusSentinel}, s);
        saccade::cross_scan_merge_cuda(shape, dy.get(), dout.get(), status_.get(), s);
        return GpuResult{fetch(dout, s), fetch(status_, s)[0]};
    }
    PipelineResult pipeline(const saccade::CrossScanShape& shape, const std::vector<float>& x, const Case& k,
                            cudaStream_t s) override {
        ScanBuffers buffers(k.shape);
        buffers.load(k, s);
        DeviceBuffer<float> dx(x.size()), out(x.size());
        DeviceBuffer<ScanStatus> statuses(2);
        put(dx, x, s);
        CUDA(cudaMemsetAsync(buffers.u.get(), 0xFF, buffers.u.bytes(), s)); // tokens come from expand only
        saccade::cross_scan_expand_cuda(shape, dx.get(), buffers.u.get(), statuses.get(), s);
        buffers.enqueue(s);
        saccade::cross_scan_merge_cuda(shape, buffers.y.get(), out.get(), statuses.get() + 1, s);
        const auto st = fetch(statuses, s);
        return {fetch(out, s), st[0], fetch(buffers.status, s)[0], st[1]};
    }
private:
    DeviceBuffer<ScanStatus> status_{1};
};

inline ScanBackend* backend = nullptr; // set by main before any group runs

inline GpuResult run_gpu(const Case& k, cudaStream_t s) { return backend->scan(k, s); }

inline void expect_valid(const ScanStatus& st, const std::string& name) {
    check(st.flags == 0 && st.done == kScanDone, name + ": status valid (flags " + hex(st.flags) + ", done " + hex(st.done) + ")");
}

inline Stats scan_case(const std::string& name, const Case& k, cudaStream_t s) {
    const auto ref = k.run();
    const auto bound = scan_bound(k);
    const auto r = run_gpu(k, s);
    expect_valid(r.status, name);
    return compare(name, r.y, ref, bound);
}

inline void expect_poisoned(const std::vector<float>& y, const std::string& name) {
    bool all = true;
    for (const float v : y) all = all && std::bit_cast<std::uint32_t>(v) == kPoison;
    check(all, name + ": every output element is the quiet-NaN poison");
}

// The reference rejects with Exception and its S01 diagnostic; the GPU must
// report `expected` (exactly, or as a subset of range bits when several lanes
// or states can fail), poison the output and still complete.
template <typename Exception>
inline void failure_case(const std::string& name, const Case& k, std::uint32_t expected, bool exact, std::string_view diagnostic,
                  cudaStream_t s) {
    expect_throw<Exception>([&] { (void)k.run(); }, diagnostic);
    const auto r = run_gpu(k, s);
    check(r.status.done == kScanDone, name + ": status done after failure");
    const std::uint32_t f = r.status.flags;
    if (expected & saccade::kScanInputFailures)
        check((f & saccade::kScanInputFailures) == expected, name + ": input bits " + hex(f) + " want " + hex(expected));
    else if (exact)
        check(f == expected, name + ": range bits " + hex(f) + " want " + hex(expected));
    else
        check((f & expected) == expected && (f & ~saccade::kScanRangeFailures) == 0,
              name + ": range bits " + hex(f) + " include " + hex(expected));
    expect_poisoned(r.y, name);
    report << "failure\t" << name << "\tflags=" << hex(f) << "\tPASS\n";
}

// ------------------------------------------------------------- test groups

// The S01 numeric fixtures (tests/selective_scan_tests.cpp contract_values).
inline void s01_values(cudaStream_t s) {
    scan_case("s01_single_token", {{1, 1, 2, 2, 1, 2, true}, {2, -4}, {0, 21}, {-1, -2}, {0.5f, 0.75f}, {1, 2}, {0.5f, -1}}, s);
    {
        Case k = filled({1, 3, 1, 1, 1, 1, false}, 0, 0, 1, 1);
        k.u = {1, 2, 3};
        scan_case("s01_three_token_decay", k, s);
    }
    {
        Case k = filled({1, 1, 6, 1, 1, 1, false}, 1, 0, 1, 1);
        k.delta = {-30, -20, -1, 0, 20, 20.5f};
        scan_case("s01_softplus_branches", k, s);
    }
    {
        Case k = filled({1, 400, 4, 16, 1, 1, true}, 0, 0, 1, 1);
        for (std::size_t t = 0; t < 400; ++t)
            for (std::size_t d = 0; d < 4; ++d) k.u[t * 4 + d] = static_cast<float>(d + 1) / 4;
        for (std::size_t n = 0; n < 16; ++n) k.a[n] = -static_cast<float>(n + 1);
        k.d.assign(4, 0.5f);
        scan_case("s01_geometric_p3_length", k, s);
    }
    {
        Case k = filled({1, 25, 3, 2, 3, 2, true}, 0, 0, 0, 0);
        k.a = {-1, -2, -0.5f, -4, 0, -1};
        k.d = {0, 0.25f, -1};
        const float u[3] = {1, -2, 0.5f}, b[2] = {1, -0.5f}, c[2] = {2, 1};
        for (std::size_t t = 0; t < 25; ++t) {
            for (std::size_t d = 0; d < 3; ++d) k.u[t * 3 + d] = u[d];
            for (std::size_t n = 0; n < 2; ++n) { k.b[t * 2 + n] = b[n]; k.c[t * 2 + n] = c[n]; }
        }
        scan_case("s01_per_channel_a", k, s);
    }
    {
        Case k = filled({1, 100, 2, 32, 1, 1, false}, 1, 0, 1, 0.5f);
        for (std::size_t n = 0; n < 32; ++n) k.a[n] = -static_cast<float>(n + 1) / 8;
        for (std::size_t t = 0; t < 100; ++t) k.u[t * 2 + 1] = -1;
        scan_case("s01_state_32_p4_length", k, s);
    }
    {
        Case k = filled({2, 7, 3, 4, 1, 4, true}, 0, 0, 0, 0);
        k.a = {-0.5f, -1, -2, -3};
        k.d = {-0.5f, 0, 0.5f};
        for (int b = 0; b < 2; ++b)
            for (int t = 0; t < 7; ++t) {
                const auto token = static_cast<std::size_t>(b * 7 + t);
                for (int d = 0; d < 3; ++d) {
                    k.delta[token * 3 + d] = gen(7 * b + 3 * t + 5 * d, 13, 6, 2);
                    k.u[token * 3 + d] = gen(5 * b + 7 * t + 3 * d, 11, 5, 4);
                }
                for (int n = 0; n < 4; ++n) {
                    k.b[token * 4 + n] = gen(3 * b + 5 * t + 7 * n, 9, 4, 4);
                    k.c[token * 4 + n] = gen(7 * b + 3 * t + 5 * n, 7, 3, 2);
                }
            }
        scan_case("s01_varying_direct_sum", k, s);
    }
    for (const std::size_t length : {25u, 100u, 400u}) {
        Case k = filled({4, length, 256, 16, 1, 1, true}, 0, 0, 0, 0);
        for (std::size_t n = 0; n < 16; ++n) k.a[n] = -static_cast<float>(n + 1);
        k.d.resize(256);
        for (int d = 0; d < 256; ++d) k.d[d] = gen(7 * d, 5, 2, 4);
        for (int b = 0; b < 4; ++b)
            for (std::size_t t = 0; t < length; ++t) {
                const auto token = static_cast<std::size_t>(b) * length + t;
                k.c[token] = static_cast<float>(b + 1) / 2;
                for (std::size_t n = 0; n < 16; ++n) k.b[token * 16 + n] = static_cast<float>(n % 4 + 1) / 4;
                for (int d = 0; d < 256; ++d) k.u[token * 256 + d] = gen(37 * b + 11 * d, 17, 8, 8);
            }
        scan_case("s01_m_head_abi_L" + std::to_string(length), k, s);
    }
    const float big = std::ldexp(1.0f, 100), small = std::ldexp(1.0f, -100), top = std::ldexp(1.0f, 127);
    {
        Case k = filled({1, 3, 1, 1, 1, 1, false}, 0, big, small, 1);
        k.u = {3, -5, 7};
        scan_case("s01_extreme_large_delta_memoryless", k, s);
    }
    {
        Case k = filled({1, 2, 1, 1, 1, 1, true}, 0, -big, 1, 1);
        k.u = {big, -big};
        k.d = {small};
        scan_case("s01_extreme_negative_delta", k, s);
    }
    scan_case("s01_extreme_fp32_max_input", filled({1, 1, 1, 1, 1, 1, false}, top, -100, 1, 1), s);
    scan_case("s01_extreme_near_fp32_limit_output", filled({1, 1, 1, 1, 1, 1, false}, top, 21, std::ldexp(1.0f, -5), 1), s);
    {
        const Case k = filled({1, 2, 1, 1, 1, 1, false}, std::numeric_limits<float>::denorm_min(), 0, 1, 1);
        scan_case("s01_extreme_subnormal_input", k, s);
        // --ftz=false in effect: ln2 * 2^-149 rounds to the smallest subnormal, not to zero.
        check(run_gpu(k, s).y[0] == std::numeric_limits<float>::denorm_min(), "subnormal results are not flushed to zero");
    }
    {
        Case k = filled({1, 3, 1, 1, 1, 1, false}, 0, 0, 1, 1);
        k.u = {1, 2, 3};
        k.a = {-std::numeric_limits<float>::max()};
        scan_case("s01_extreme_most_negative_a", k, s);
    }
}

inline void coverage(cudaStream_t s) {
    std::uint64_t seed = 0x5eed0001;
    for (const std::size_t n : {1u, 2u, 4u, 8u, 16u, 32u}) {
        const auto tag = "_N" + std::to_string(n);
        scan_case("shared_a_rank1_c_d" + tag, random_case({2, 37, 5, n, 1, 1, true}, ++seed), s);
        scan_case("per_channel_a_full_c_no_d" + tag, random_case({3, 11, 7, n, 7, n, false}, ++seed), s);
        scan_case("single_token_per_channel_a" + tag, random_case({1, 1, 3, n, 3, 1, true}, ++seed), s);
    }
    // Lane counts that leave partial warps and blocks.
    scan_case("tail_1x1x1_N1", random_case({1, 1, 1, 1, 1, 1, false}, ++seed), s);
    scan_case("tail_Dc13_N2", random_case({5, 3, 13, 2, 13, 2, true}, ++seed), s);
    scan_case("tail_Dc257_N4", random_case({2, 4, 257, 4, 1, 4, true}, ++seed), s);
    scan_case("tail_Dc33_N32", random_case({1, 3, 33, 32, 33, 32, true}, ++seed), s);
    {
        const Case k = random_case({2, 0, 3, 4, 1, 1, true}, ++seed);
        const auto r = run_gpu(k, s);
        expect_valid(r.status, "empty_sequence");
        check(r.y.empty() && k.run().empty(), "zero-length sequence yields an empty output");
        report << "case\tempty_sequence\tcount=0\tPASS\n";
    }
    // Slow decay over a long sequence: error accumulates over ~1/(Delta |A|) steps.
    scan_case("long_slow_decay_L2000", random_case({1, 2000, 4, 16, 1, 1, true}, ++seed, {1, -4, -2, -0.01f, -0.001f, 1, 1}), s);
    // The m head's three scan nodes with frozen A = -1..-16 and data-scale inputs.
    for (const std::size_t length : {25u, 100u, 400u}) {
        Case k = random_case({4, length, 256, 16, 1, 1, true}, ++seed, {3, -5, 2, -1, -1, 2, 1});
        for (std::size_t n = 0; n < 16; ++n) k.a[n] = -static_cast<float>(n + 1);
        scan_case("m_head_random_L" + std::to_string(length), k, s);
    }
    // Near-cancellation: two almost identical states read with C = +1 / -1.
    {
        Case k = random_case({1, 50, 3, 2, 1, 2, false}, ++seed);
        k.a = {-1.0f, -1.0001f};
        for (std::size_t t = 0; t < 50; ++t) {
            k.b[t * 2 + 1] = k.b[t * 2];
            k.c[t * 2] = 1; k.c[t * 2 + 1] = -1;
        }
        const auto st = scan_case("cancellation_two_states", k, s);
        report << "note\tcancellation_two_states\tmax_rel_to_|y|=" << std::setprecision(3) << st.max_rel
               << "\t(threshold scales with the term magnitude, not |y|)\n";
    }
}

// Legacy kernel softplus (SCAN_CONTRACT §7), emulated with host FP32 libm.
inline float legacy_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log(1.0f + std::exp(x));
}

inline void softplus(cudaStream_t s) {
    // Review case: the legacy form rounds 1 + e^-18 to 1 and outputs 0.
    {
        Case k = filled({1, 1, 1, 1, 1, 1, true}, 1, -18, 1e8f, 1);
        const auto ref = k.run();
        const auto bound = scan_bound(k);
        const auto r = run_gpu(k, s);
        expect_valid(r.status, "softplus_minus18");
        compare("softplus_minus18", r.y, ref, bound);
        const double legacy = static_cast<double>(legacy_softplus(-18.0f)) * 1e8;
        check(std::fabs(legacy - ref[0]) > 2 * bound[0], "the threshold rejects the legacy softplus at x = -18");
        report << "discrimination\tsoftplus_minus18\tref=" << std::setprecision(10) << ref[0] << "\tlegacy=" << legacy
               << std::setprecision(3) << "\ttol=" << 2 * bound[0] << "\tlegacy_err_over_tol=" << std::fabs(legacy - ref[0]) / (2 * bound[0])
               << '\n';
    }
    // Sweep over every branch and its edges; u scales each channel so that
    // Delta * B * u lies in [1, 2) and relative errors of Delta show at O(1).
    std::vector<float> xs{-1100, -1024, -1000, -300, -160, -150, -120, -110, -103.5f, -100, -95, -90, -87.5f, -85, -80.5f,
                          std::nextafter(-80.0f, -1000.0f), -80, std::nextafter(-80.0f, 0.0f), -79.5f, -60, -50, -30, -25,
                          -20.5f, -20, -19.5f, -18, -17, -15, -12, -10, -9, -8, -6, -4, -2, -1, -0.5f, 0, 0.5f, 1, 2, 5, 10,
                          15, 19.5f, 20, std::nextafter(20.0f, 100.0f), 20.5f, 25, 100, 1000};
    Case k = filled({1, 1, xs.size(), 1, 1, 1, false}, 1, 0, std::ldexp(1.0f, 100), 1);
    k.delta = xs;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double delta = saccade::selective_scan_softplus(xs[i]) * 0x1p100;
        const int e = delta > 0 ? static_cast<int>(std::floor(std::log2(delta))) : -127;
        k.u[i] = std::ldexp(1.0f, std::clamp(-e, -126, 127));
    }
    const auto ref = k.run();
    const auto bound = scan_bound(k);
    const auto r = run_gpu(k, s);
    expect_valid(r.status, "softplus_sweep");
    const auto st = compare("softplus_sweep", r.y, ref, bound, false);
    std::size_t legacy_rejected = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double legacy = static_cast<double>(legacy_softplus(xs[i])) * 0x1p100 * k.u[i];
        const double err = std::fabs(static_cast<double>(r.y[i]) - ref[i]), tol = 2 * bound[i];
        const bool rejected = std::fabs(legacy - ref[i]) > tol;
        legacy_rejected += rejected;
        report << "sweep\t" << std::setprecision(9) << xs[i] << '\t' << std::setprecision(17) << ref[i] << '\t'
               << std::setprecision(9) << r.y[i] << '\t' << std::setprecision(3) << err << '\t' << tol << "\tlegacy_err="
               << std::fabs(legacy - ref[i]) << (rejected ? "\tlegacy_rejected" : "") << '\n';
    }
    report << "case\tsoftplus_sweep\tcount=" << xs.size() << std::setprecision(3) << "\tmax_abs=" << st.max_abs
           << "\tmax_rel=" << st.max_rel << "\tmax_err_over_tol=" << st.max_ratio << "\tfailures=" << st.failures
           << "\tlegacy_rejected=" << legacy_rejected << "\tPASS\n";
    check(legacy_rejected >= 10, "the threshold rejects the legacy softplus across (-20, -8) and below -87");
}

// Reference-finite cases whose plain FP32 intermediates over- or underflow.
inline void intermediate_range(cudaStream_t s) {
    const float p = 1;
    auto pow2 = [](int e) { return std::ldexp(1.0f, e); };
    scan_case("drive_intermediate_overflow", filled({1, 2, 1, 1, 1, 1, false}, pow2(-100), pow2(100), pow2(100), p), s);
    scan_case("drive_delta_underflow_minus100", filled({1, 1, 1, 1, 1, 1, false}, pow2(27), -100, pow2(100), p), s);
    scan_case("drive_delta_underflow_minus150", filled({1, 1, 1, 1, 1, 1, false}, pow2(127), -150, pow2(127), p), s);
    scan_case("drive_subnormal_b", filled({1, 1, 1, 1, 1, 1, false}, pow2(127), 0.5413248f, pow2(-140), p), s);
    {
        Case k = filled({1, 1, 1, 4, 1, 1, true}, 1, 21, 0, 1.75f);
        k.b = {pow2(122), -pow2(122), pow2(122), -pow2(122)};
        k.d = {1};
        scan_case("butterfly_partial_sum_overflow", k, s);
    }
    {
        Case k = filled({1, 1, 1, 2, 1, 1, true}, 1, 21, pow2(122), 1.75f);
        k.d = {-pow2(127)};
        scan_case("partial_sum_overflow_restored_by_skip", k, s);
    }
}

inline void failures(cudaStream_t s) {
    const Case ok = varied({1, 4, 4, 16, 1, 1, true});
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    { Case k = ok; k.u[5] = nan; failure_case<std::invalid_argument>("nonfinite_u", k, saccade::kScanNonFiniteU, true, "u[5] is not finite", s); }
    { Case k = ok; k.delta[0] = inf; failure_case<std::invalid_argument>("nonfinite_delta", k, saccade::kScanNonFiniteDelta, true, "delta[0] is not finite", s); }
    { Case k = ok; k.a[3] = -inf; failure_case<std::invalid_argument>("nonfinite_a", k, saccade::kScanNonFiniteA, true, "A[3] is not finite", s); }
    { Case k = ok; k.b[63] = nan; failure_case<std::invalid_argument>("nonfinite_b", k, saccade::kScanNonFiniteB, true, "B[63] is not finite", s); }
    { Case k = ok; k.c[2] = inf; failure_case<std::invalid_argument>("nonfinite_c", k, saccade::kScanNonFiniteC, true, "C[2] is not finite", s); }
    { Case k = ok; k.d[1] = nan; failure_case<std::invalid_argument>("nonfinite_d", k, saccade::kScanNonFiniteD, true, "D[1] is not finite", s); }
    // L = 0 still checks A and D, as the reference does.
    { Case k = varied({2, 0, 3, 4, 1, 1, true}); k.a[2] = nan; failure_case<std::invalid_argument>("empty_nonfinite_a", k, saccade::kScanNonFiniteA, true, "A[2] is not finite", s); }
    { Case k = varied({2, 0, 3, 4, 3, 1, true}); k.d[2] = -inf; failure_case<std::invalid_argument>("empty_nonfinite_d", k, saccade::kScanNonFiniteD, true, "D[2] is not finite", s); }

    const float top = std::ldexp(1.0f, 127);
    failure_case<std::range_error>("range_drive", filled({1, 1, 1, 1, 1, 1, false}, top, 21, 4, 1), saccade::kScanDriveRange, true,
                                   "drive leaves FP32 range at batch 0, token 0", s);
    { Case k = filled({1, 1, 1, 1, 1, 1, false}, 1, 21, 1, 1); k.a = {100};
      failure_case<std::range_error>("range_decay", k, saccade::kScanDecayRange, true, "decay leaves FP32 range", s); }
    { Case k = filled({1, 200, 1, 1, 1, 1, false}, 1, 0, 1, 1); k.a = {1};
      failure_case<std::range_error>("range_state", k, saccade::kScanStateRange, true, "state leaves FP32 range", s); }
    failure_case<std::range_error>("range_c_state", filled({1, 2, 1, 1, 1, 1, false}, 1, 21, std::ldexp(1.0f, 120), top),
                                   saccade::kScanTermRange, true, "C*state leaves FP32 range", s);
    { Case k = filled({1, 1, 1, 1, 1, 1, true}, top, -100, 1, 1); k.d = {4};
      failure_case<std::range_error>("range_d_u", k, saccade::kScanSkipRange, true, "D*u leaves FP32 range", s); }
    failure_case<std::range_error>("range_output", filled({1, 1, 1, 4, 1, 1, false}, top, 21, std::ldexp(1.0f, -5), 1),
                                   saccade::kScanOutputRange, true, "output leaves FP32 range", s);
    // One failing lane among valid ones invalidates the whole output.
    {
        Case k = random_case({2, 5, 6, 4, 1, 4, true}, 0x5eed0100);
        const std::size_t at = (1 * 5 + 3) * 6 + 2; // batch 1, token 3, channel 2
        k.u[at] = top;
        k.delta[at] = 21;
        for (std::size_t n = 0; n < 4; ++n) k.b[(1 * 5 + 3) * 4 + n] = 1;
        failure_case<std::range_error>("range_one_lane_of_twelve", k, saccade::kScanDriveRange, false,
                                       "drive leaves FP32 range at batch 1, token 3, channel 2", s);
    }
}

// ------------------------------------------------------------- Cross-Scan

inline std::vector<float> random_floats(std::size_t n, std::uint64_t seed, float range) {
    Random g{seed};
    std::vector<float> v(n);
    for (auto& x : v) x = g.uniform(-range, range);
    return v;
}

// Cross-Scan numerics: expand bit-identical to the reference permutation,
// merge and expand -> scan -> merge within the §5 bounds, non-finite inputs.
inline void cross_scan_numeric(cudaStream_t s) {
    using saccade::CrossScanShape;
    const auto expand = [&](const CrossScanShape& shape, const std::vector<float>& x) { return backend->expand(shape, x, s); };
    const auto merge = [&](const CrossScanShape& shape, const std::vector<float>& y) { return backend->merge(shape, y, s); };
    const auto merge_bound = [](const CrossScanShape& shape, const std::vector<float>& y) {
        const std::size_t L = shape.height * shape.width, C = shape.channels;
        std::vector<double> bound(shape.batch * C * L);
        for (std::size_t i = 0; i < shape.batch; ++i)
            for (std::size_t c = 0; c < C; ++c)
                for (std::size_t p = 0; p < L; ++p) {
                    double sum = 0;
                    for (std::size_t k = 0; k < 4; ++k)
                        sum += std::fabs(y[((k * shape.batch + i) * L + saccade::cross_scan_source(k, p, shape.height, shape.width)) * C + c]);
                    bound[(i * C + c) * L + p] = gamma_n(4) * sum / 4 + kEta;
                }
        return bound;
    };
    std::uint64_t seed = 0x5eed0400;
    const CrossScanShape shapes[] = {{1, 2, 2, 3}, {2, 3, 3, 5}, {1, 4, 1, 4}, {1, 4, 4, 1}, {1, 1, 1, 1},
                                     {2, 5, 7, 3}, {1, 128, 20, 20}, {1, 128, 10, 10}, {1, 128, 5, 5}};
    for (const auto& shape : shapes) {
        const auto tag = std::to_string(shape.batch) + "x" + std::to_string(shape.channels) + "x" +
                         std::to_string(shape.height) + "x" + std::to_string(shape.width);
        const std::size_t n = shape.batch * shape.channels * shape.height * shape.width;
        std::vector<float> x = random_floats(n, ++seed, 4);
        if (tag == "1x2x2x3") // S01 hand-listed order: x[c, h, w] = 100 c + 10 h + w
            for (int c = 0; c < 2; ++c)
                for (int h = 0; h < 2; ++h)
                    for (int w = 0; w < 3; ++w) x[(c * 2 + h) * 3 + w] = static_cast<float>(100 * c + 10 * h + w);
        const auto tokens = expand(shape, x);
        expect_valid(tokens.status, "expand_" + tag);
        const auto want = saccade::cross_scan_expand(x, shape.batch, shape.channels, shape.height, shape.width);
        check(tokens.y.size() == want.size() && std::memcmp(tokens.y.data(), want.data(), want.size() * sizeof(float)) == 0,
              "expand_" + tag + ": identical to the reference permutation");
        const auto y = random_floats(4 * n, ++seed, 8);
        const auto merged = merge(shape, y);
        expect_valid(merged.status, "merge_" + tag);
        compare("merge_" + tag, merged.y, saccade::cross_scan_merge(std::vector<double>(y.begin(), y.end()), shape.batch,
                                                                     shape.channels, shape.height, shape.width),
                merge_bound(shape, y));
    }
    report << "expand\tall_shapes\tbitwise\tPASS\n";
    {
        const CrossScanShape shape{1, 2, 1, 2};
        std::vector<float> y(16, 0.75f * std::numeric_limits<float>::max());
        for (std::size_t i = 8; i < 16; ++i) y[i] = -y[i] / 3;
        const auto merged = merge(shape, y);
        expect_valid(merged.status, "merge_partial_sum_overflow");
        compare("merge_partial_sum_overflow", merged.y,
                saccade::cross_scan_merge(std::vector<double>(y.begin(), y.end()), 1, 2, 1, 2), merge_bound(shape, y));
    }
    {
        std::vector<float> x(12, 1.0f);
        x[7] = std::numeric_limits<float>::infinity();
        const auto r = expand({1, 2, 2, 3}, x);
        check(r.status.done == kScanDone && r.status.flags == saccade::kCrossScanNonFiniteInput, "expand flags a non-finite input");
        expect_poisoned(r.y, "expand_nonfinite");
        std::vector<float> y(48, 1.0f);
        y[30] = std::numeric_limits<float>::quiet_NaN();
        const auto m = merge({1, 2, 2, 3}, y);
        check(m.status.done == kScanDone && m.status.flags == saccade::kCrossScanNonFiniteInput, "merge flags a non-finite input");
        expect_poisoned(m.y, "merge_nonfinite");
    }
    // Expand -> scan -> merge on one stream against the reference pipeline.
    for (const auto& shape : {CrossScanShape{1, 1, 2, 3}, CrossScanShape{1, 8, 5, 5}}) {
        const auto tag = std::to_string(shape.channels) + "x" + std::to_string(shape.height) + "x" + std::to_string(shape.width);
        const std::size_t L = shape.height * shape.width, C = shape.channels;
        const bool hand = C == 1;
        const std::vector<float> x = hand ? std::vector<float>{1, 2, 3, 4, 5, 6} : random_floats(C * L, ++seed, 2);
        Case k = hand ? filled({4, L, 1, 1, 1, 1, false}, 0, 0, 1, 1) : random_case({4, L, C, 4, 1, 1, true}, ++seed);
        if (hand) k.a = {0};
        k.u = saccade::cross_scan_expand(x, 1, C, shape.height, shape.width);
        const auto ref_y = k.run();
        const auto scan_e = scan_bound(k);
        const auto ref = saccade::cross_scan_merge(ref_y, 1, C, shape.height, shape.width);
        std::vector<double> bound(ref.size());
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t p = 0; p < L; ++p) {
                double e = 0, mag = 0;
                for (std::size_t dir = 0; dir < 4; ++dir) {
                    const std::size_t at = (dir * L + saccade::cross_scan_source(dir, p, shape.height, shape.width)) * C + c;
                    e += scan_e[at];
                    mag += std::fabs(ref_y[at]);
                }
                bound[c * L + p] = (1 + gamma_n(4)) * e / 4 + gamma_n(4) * mag / 4 + kEta;
            }
        const auto p = backend->pipeline(shape, x, k, s);
        expect_valid(p.expand, "pipeline_expand_" + tag);
        expect_valid(p.scan, "pipeline_scan_" + tag);
        expect_valid(p.merge, "pipeline_merge_" + tag);
        compare("pipeline_expand_scan_merge_" + tag, p.out, ref, bound);
    }
}

} // namespace saccade::scan_test

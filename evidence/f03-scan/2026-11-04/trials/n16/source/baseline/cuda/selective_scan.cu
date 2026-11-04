#include "saccade/selective_scan_cuda.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef SACCADE_BUILD_COMMIT
#define SACCADE_BUILD_COMMIT "unknown"
#endif
#ifndef SACCADE_BUILD_DIRTY
#define SACCADE_BUILD_DIRTY "unknown"
#endif
#ifndef SACCADE_SCAN_CUDA_FLAGS
#define SACCADE_SCAN_CUDA_FLAGS "unknown"
#endif
#define SACCADE_STRINGIFY_(x) #x
#define SACCADE_STRINGIFY(x) SACCADE_STRINGIFY_(x)

namespace saccade {
namespace {

// Contract capacity (docs/SCAN_CONTRACT.md §3); kernels index in 64 bits.
constexpr std::size_t kMaxElements = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
constexpr unsigned kBlock = 256;
constexpr unsigned kMaxGridStrideBlocks = 4096;

void ck(cudaError_t result, const char* call) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(result));
}
#define CUDA(call) ck((call), #call)

// ---------------------------------------------------------------- device side

__device__ bool finite_fp32(float x) { return fabsf(x) <= FLT_MAX; } // false for NaN and infinities

// value = m * 2^e with m in [0.5, 1) or m == 0; frexpf and ldexpf are exact.
struct Scaled { float m; int e; };
__device__ Scaled split(float x) { int e = 0; const float m = frexpf(x, &e); return {m, e}; }

// Delta = softplus(x) with the S01 threshold, kept as (m, e) so a value below
// the FP32 normal range keeps full precision (docs/SCAN_CUDA.md §3).
__device__ Scaled softplus_scaled(float x) {
    if (x > 20.0f) return split(x);
    if (x >= -80.0f) return split(log1pf(expf(x)));
    // log1p(e^x) equals e^x within 1e-35 relative here; form e^r * 2^k without underflow.
    if (x < -1024.0f) return {0.0f, 0}; // e^x < 2^-1477: no effect at FP32 resolution
    const float k = rintf(__fmul_rn(x, 1.44269504f));
    const float r = __fmaf_rn(-k, 1.42860677e-6f, __fmaf_rn(-k, 0.693145752f, x)); // ln2 = hi + lo
    Scaled s = split(expf(r));
    s.e += static_cast<int>(k);
    return s;
}

// Butterfly over one (b, d) group: IEEE addition commutes, so every thread of
// the group ends with the same bits and the order is fixed by the state count.
__device__ float group_sum(float v, unsigned mask, unsigned width) {
    for (unsigned offset = width / 2; offset > 0; offset /= 2) v = __fadd_rn(v, __shfl_xor_sync(mask, v, offset, width));
    return v;
}

struct ScanParams {
    const float *u, *delta, *a, *b, *c, *d;
    float* y;
    unsigned long long length, channels, lanes;
    unsigned state, log2_state;
    bool a_shared, c_full, has_d;
};

// One thread per (batch, channel, state); the N threads of a lane are
// consecutive within a warp, so every early exit below is group-uniform.
__global__ void scan_kernel(ScanParams p, ScanStatus* status) {
    const unsigned long long thread = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long lane = thread >> p.log2_state;
    if (lane >= p.lanes) return;
    const unsigned N = p.state, n = static_cast<unsigned>(thread & (N - 1));
    const unsigned mask = N == 32 ? 0xffffffffu : ((1u << N) - 1) << ((threadIdx.x & 31u) & ~(N - 1));
    const unsigned long long Dc = p.channels, L = p.length, b = lane / Dc, d = lane % Dc;
    const float a = p.a[(p.a_shared ? 0 : d * N) + n];
    const float skip_d = p.has_d ? p.d[d] : 0.0f;
    float h = 0.0f;
    for (unsigned long long t = 0; t < L; ++t) {
        const unsigned long long token = b * L + t, at = token * Dc + d;
        const float u = p.u[at], bv = p.b[token * N + n];
        const float cv = p.c[p.c_full ? token * N + n : token];
        const Scaled delta = softplus_scaled(p.delta[at]), sb = split(bv), su = split(u);
        const float decay = expf(ldexpf(__fmul_rn(delta.m, a), delta.e));
        const float drive = ldexpf(__fmul_rn(__fmul_rn(delta.m, sb.m), su.m), delta.e + sb.e + su.e);
        h = __fmaf_rn(decay, h, drive);
        const float term = __fmul_rn(cv, h);
        unsigned fail = !finite_fp32(decay) ? kScanDecayRange : !finite_fp32(drive) ? kScanDriveRange
                      : !finite_fp32(h) ? kScanStateRange : !finite_fp32(term) ? kScanTermRange : 0u;
        for (unsigned offset = N / 2; offset > 0; offset /= 2) fail |= __shfl_xor_sync(mask, fail, offset, N);
        float y = 0.0f;
        if (!fail) {
            const float sum = group_sum(term, mask, N);
            const float skip = __fmul_rn(skip_d, u);
            if (p.has_d && !finite_fp32(skip)) {
                fail = kScanSkipRange;
            } else {
                y = p.has_d ? __fadd_rn(sum, skip) : sum;
                if (!finite_fp32(y)) {
                    // Every term is finite but a partial sum left the FP32 range:
                    // repeat with exact power-of-two scaling.
                    const float scaled = group_sum(__fmul_rn(term, 0x1p-8f), mask, N);
                    y = p.has_d ? __fadd_rn(scaled, __fmul_rn(skip, 0x1p-8f)) : scaled;
                    if (fabsf(y) <= FLT_MAX * 0x1p-8f) y = __fmul_rn(y, 0x1p8f);
                    else fail = kScanOutputRange;
                }
            }
        }
        if (fail) {
            if (n == 0) atomicOr(&status->flags, fail);
            return;
        }
        if (n == 0) p.y[at] = y;
    }
}

struct CheckedTensor { const float* data; unsigned long long count; std::uint32_t bit; };
struct CheckList { CheckedTensor tensor[6]; unsigned size; };

__global__ void check_finite(CheckList list, ScanStatus* status) {
    const unsigned long long first = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    std::uint32_t found = 0;
    for (unsigned k = 0; k < list.size; ++k)
        for (unsigned long long i = first; i < list.tensor[k].count; i += stride)
            if (!finite_fp32(list.tensor[k].data[i])) found |= list.tensor[k].bit;
    if (found) atomicOr(&status->flags, found);
}

// Last kernel of every call: no other kernel writes flags now, and done is a
// separate word written once.
__global__ void finalize(float* out, unsigned long long count, ScanStatus* status) {
    const unsigned long long first = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    if (status->flags)
        for (unsigned long long i = first; i < count; i += stride) out[i] = __int_as_float(0x7fc00000);
    if (first == 0) status->done = kScanDone;
}

struct CrossParams { unsigned long long batch, channels, height, width; };

// Direction k reads token t at the row-major position of the k-flipped map.
__device__ unsigned long long cross_source(unsigned direction, unsigned long long token, const CrossParams& p) {
    const unsigned long long row = token / p.width, col = token % p.width;
    const bool flip_h = direction == 1 || direction == 3, flip_w = direction == 1 || direction == 2;
    return (flip_h ? p.height - 1 - row : row) * p.width + (flip_w ? p.width - 1 - col : col);
}

__global__ void expand_kernel(CrossParams p, const float* x, float* tokens) {
    const unsigned long long L = p.height * p.width, C = p.channels, total = 4 * p.batch * L * C;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        const unsigned long long c = i % C, row = i / C / L, t = i / C % L;
        const unsigned k = static_cast<unsigned>(row / p.batch);
        const unsigned long long image = row % p.batch;
        tokens[i] = x[(image * C + c) * L + cross_source(k, t, p)];
    }
}

__global__ void merge_kernel(CrossParams p, const float* y, float* out) {
    const unsigned long long L = p.height * p.width, C = p.channels, total = p.batch * C * L;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        const unsigned long long position = i % L, c = i / L % C, image = i / L / C;
        float v[4];
        // Each flip is an involution: source(k, position) is also position's token in direction k.
        for (unsigned k = 0; k < 4; ++k) v[k] = y[((k * p.batch + image) * L + cross_source(k, position, p)) * C + c];
        const float sum = __fadd_rn(__fadd_rn(__fadd_rn(v[0], v[1]), v[2]), v[3]);
        float mean = __fmul_rn(sum, 0.25f);
        if (!finite_fp32(sum))
            mean = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(v[0], 0.25f), __fmul_rn(v[1], 0.25f)), __fmul_rn(v[2], 0.25f)),
                             __fmul_rn(v[3], 0.25f));
        out[i] = mean;
    }
}

// ------------------------------------------------------------------ host side

[[noreturn]] void reject(const char* api, const std::string& what) { throw std::invalid_argument(std::string(api) + ": " + what); }

std::size_t elements(const char* api, std::initializer_list<std::size_t> dims, const char* tensor) {
    std::size_t product = 1;
    for (const auto dim : dims) {
        if (dim != 0 && product > kMaxElements / dim) reject(api, std::string(tensor) + " element count exceeds int32 capacity");
        product *= dim;
    }
    return product;
}

void require_stream(const char* api, cudaStream_t stream) {
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        reject(api, "an explicit non-default stream is required");
}

struct Region { const void* data; std::size_t bytes; const char* name; };

void require_pointer(const char* api, const Region& r) {
    if (r.bytes == 0) return;
    if (r.data == nullptr) reject(api, std::string(r.name) + " is null");
    if (reinterpret_cast<std::uintptr_t>(r.data) % alignof(float) != 0) reject(api, std::string(r.name) + " is not 4-byte aligned");
}

bool overlaps(const Region& a, const Region& b) {
    if (a.bytes == 0 || b.bytes == 0) return false;
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes && b0 < a0 + a.bytes;
}

// Outputs (output tensor, status) must not overlap inputs or each other.
void require_regions(const char* api, std::initializer_list<Region> inputs, const Region& out, const Region& status) {
    for (const auto& r : inputs) require_pointer(api, r);
    require_pointer(api, out);
    require_pointer(api, status);
    for (const auto& r : inputs) {
        if (overlaps(out, r)) reject(api, std::string(out.name) + " overlaps " + r.name);
        if (overlaps(status, r)) reject(api, std::string("status overlaps ") + r.name);
    }
    if (overlaps(status, out)) reject(api, std::string("status overlaps ") + out.name);
}

unsigned grid_stride_blocks(unsigned long long count) {
    const unsigned long long blocks = (count + kBlock - 1) / kBlock;
    return static_cast<unsigned>(blocks == 0 ? 1 : blocks < kMaxGridStrideBlocks ? blocks : kMaxGridStrideBlocks);
}

void launched(const char* kernel) { ck(cudaGetLastError(), kernel); }

std::size_t cross_scan_total(const char* api, const CrossScanShape& s) {
    if (s.batch == 0 || s.channels == 0 || s.height == 0 || s.width == 0) reject(api, "dimensions must be positive");
    return elements(api, {4, s.batch, s.height, s.width, s.channels}, "tokens");
}

} // namespace

std::string selective_scan_cuda_load_kernels() {
    const std::pair<const char*, const void*> kernels[] = {
        {"check_finite", reinterpret_cast<const void*>(check_finite)}, {"scan_kernel", reinterpret_cast<const void*>(scan_kernel)},
        {"finalize", reinterpret_cast<const void*>(finalize)}, {"expand_kernel", reinterpret_cast<const void*>(expand_kernel)},
        {"merge_kernel", reinterpret_cast<const void*>(merge_kernel)}};
    std::ostringstream report;
    for (const auto& [name, kernel] : kernels) {
        cudaFuncAttributes attributes{};
        CUDA(cudaFuncGetAttributes(&attributes, kernel)); // loads the kernel under lazy loading
        report << "kernel=" << name << " registers=" << attributes.numRegs << " local_bytes=" << attributes.localSizeBytes
               << " shared_bytes=" << attributes.sharedSizeBytes << " max_threads=" << attributes.maxThreadsPerBlock << '\n';
    }
    return report.str();
}

void selective_scan_cuda(const SelectiveScanShape& s, const SelectiveScanDeviceInputs& in, float* y,
                         ScanStatus* status, cudaStream_t stream) {
    constexpr const char* api = "selective_scan_cuda";
    require_stream(api, stream);
    if (s.batch == 0 || s.channels == 0) reject(api, "batch and channels must be positive");
    if (s.state == 0 || s.state > 32 || (s.state & (s.state - 1)) != 0)
        reject(api, "state " + std::to_string(s.state) + " must be a power of two in [1, 32]");
    if (s.a_rows != 1 && s.a_rows != s.channels) reject(api, "A rows must be 1 or channels");
    if (s.c_width != 1 && s.c_width != s.state) reject(api, "C width must be 1 or state");
    const auto ud = elements(api, {s.batch, s.length, s.channels}, "u/delta/y");
    const auto bn = elements(api, {s.batch, s.length, s.state}, "B");
    const auto cn = elements(api, {s.batch, s.length, s.c_width}, "C");
    const auto an = elements(api, {s.a_rows, s.state}, "A");
    const auto dn = s.has_d ? s.channels : 0;
    constexpr auto f = sizeof(float);
    require_regions(api,
                    {{in.u, ud * f, "u"}, {in.delta, ud * f, "delta"}, {in.a, an * f, "A"},
                     {in.b, bn * f, "B"}, {in.c, cn * f, "C"}, {in.d, dn * f, "D"}},
                    {y, ud * f, "y"}, {status, sizeof(ScanStatus), "status"});

    CUDA(cudaMemsetAsync(status, 0, sizeof(ScanStatus), stream));
    const CheckList list{{{in.u, ud, kScanNonFiniteU}, {in.delta, ud, kScanNonFiniteDelta}, {in.a, an, kScanNonFiniteA},
                          {in.b, bn, kScanNonFiniteB}, {in.c, cn, kScanNonFiniteC}, {in.d, dn, kScanNonFiniteD}}, 6};
    check_finite<<<grid_stride_blocks(std::max({ud, an, dn})), kBlock, 0, stream>>>(list, status);
    launched("check_finite");
    if (s.length > 0) {
        unsigned log2_state = 0;
        while ((std::size_t{1} << log2_state) < s.state) ++log2_state;
        const ScanParams params{in.u, in.delta, in.a, in.b, in.c, in.d, y, s.length, s.channels, s.batch * s.channels,
                                static_cast<unsigned>(s.state), log2_state, s.a_rows == 1, s.c_width != 1, s.has_d};
        // lanes <= INT32_MAX and N <= 32, so the block count fits gridDim.x.
        const unsigned blocks = static_cast<unsigned>(((s.batch * s.channels << log2_state) + kBlock - 1) / kBlock);
        scan_kernel<<<blocks, kBlock, 0, stream>>>(params, status);
        launched("scan_kernel");
    }
    finalize<<<grid_stride_blocks(ud), kBlock, 0, stream>>>(y, ud, status);
    launched("finalize");
}

void cross_scan_expand_cuda(const CrossScanShape& s, const float* x, float* tokens, ScanStatus* status, cudaStream_t stream) {
    constexpr const char* api = "cross_scan_expand_cuda";
    require_stream(api, stream);
    const auto total = cross_scan_total(api, s);
    require_regions(api, {{x, total / 4 * sizeof(float), "x"}}, {tokens, total * sizeof(float), "tokens"},
                    {status, sizeof(ScanStatus), "status"});
    CUDA(cudaMemsetAsync(status, 0, sizeof(ScanStatus), stream));
    const CheckList list{{{x, total / 4, kCrossScanNonFiniteInput}}, 1};
    check_finite<<<grid_stride_blocks(total / 4), kBlock, 0, stream>>>(list, status);
    launched("check_finite");
    const CrossParams params{s.batch, s.channels, s.height, s.width};
    expand_kernel<<<grid_stride_blocks(total), kBlock, 0, stream>>>(params, x, tokens);
    launched("expand_kernel");
    finalize<<<grid_stride_blocks(total), kBlock, 0, stream>>>(tokens, total, status);
    launched("finalize");
}

void cross_scan_merge_cuda(const CrossScanShape& s, const float* y, float* out, ScanStatus* status, cudaStream_t stream) {
    constexpr const char* api = "cross_scan_merge_cuda";
    require_stream(api, stream);
    const auto total = cross_scan_total(api, s);
    require_regions(api, {{y, total * sizeof(float), "y"}}, {out, total / 4 * sizeof(float), "out"},
                    {status, sizeof(ScanStatus), "status"});
    CUDA(cudaMemsetAsync(status, 0, sizeof(ScanStatus), stream));
    const CheckList list{{{y, total, kCrossScanNonFiniteInput}}, 1};
    check_finite<<<grid_stride_blocks(total), kBlock, 0, stream>>>(list, status);
    launched("check_finite");
    const CrossParams params{s.batch, s.channels, s.height, s.width};
    merge_kernel<<<grid_stride_blocks(total / 4), kBlock, 0, stream>>>(params, y, out);
    launched("merge_kernel");
    finalize<<<grid_stride_blocks(total / 4), kBlock, 0, stream>>>(out, total / 4, status);
    launched("finalize");
}

std::string selective_scan_cuda_identity() {
    std::ostringstream id;
    id << "build_commit=" << SACCADE_BUILD_COMMIT << "\nbuild_dirty=" << SACCADE_BUILD_DIRTY << "\nnvcc="
       << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
       << "\nscan_cuda_flags=" << SACCADE_SCAN_CUDA_FLAGS
#ifdef __CUDA_ARCH_LIST__
       << "\ncuda_arch_list=" << SACCADE_STRINGIFY(__CUDA_ARCH_LIST__)
#endif
       << "\nhost_compiler=" << __VERSION__ << '\n';
    return id.str();
}

} // namespace saccade

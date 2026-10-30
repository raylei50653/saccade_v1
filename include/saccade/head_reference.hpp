#pragma once

#include "saccade/head_weights.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace saccade {

// S04 binary64 reference of the complete T=1 head (docs/HEAD_TRT.md §4) with
// two per-element error measures of the FP32 TensorRT implementation. Inputs
// and weights are FP32 bytes widened exactly; every intermediate is binary64;
// exp, log1p and expm1 are glibc's. Under the operation model of HEAD_TRT §4.3:
//   sigma: probabilistic scale of |FP32 implementation - exact math| (roundings
//          independent, zero mean, bounded; sqrt(K) growth in sums; sources that
//          may be shared are added linearly). Acceptance: |gpu - reference| <=
//          kLambda * sigma + kFloor for every element of all six outputs.
//   bound: first-order worst-case bound (every rounding aligned); always valid,
//          reported, but too loose through the head to separate semantic errors.
// Nothing in this file calls TensorRT, CUDA or the S02 kernels.

inline constexpr std::array<const char*, 3> kHeadInputNames{"p3", "p4", "p5"};
// Fixed output order: cls_p3, cls_p4, cls_p5, reg_p3, reg_p4, reg_p5.
inline constexpr std::array<const char*, 6> kHeadOutputNames{"cls_p3", "cls_p4", "cls_p5", "reg_p3", "reg_p4", "reg_p5"};
std::vector<std::int64_t> head_input_shape(int level);   // {1, C, H, W}
std::vector<std::int64_t> head_output_shape(int output); // {1, 80 or 4, H, W}

// Operation model of the FP32 implementation (HEAD_TRT §4.3); fixed with the
// policy before the first complete-head GPU comparison.
namespace head_policy {
inline constexpr double kUnit = 0x1p-24;           // FP32 unit roundoff u
inline constexpr double kUnderflow = 0x1p-150;     // absolute error of one FP32 rounding in the subnormal range
inline constexpr double kTranscendental = 0x1p-17; // relative error of a TensorRT FP32 exp, sigmoid or SiLU result
inline constexpr double kSiluFloor = 0x1p-125;     // SiLU absolute error per unit |x| where sigmoid underflows
inline constexpr double kSiluLipschitz = 1.1;      // >= max |SiLU'(x)| = 1.0998...
inline constexpr double kLambda = 16;              // acceptance: |gpu - ref| <= kLambda * sigma + kFloor
inline constexpr double kFloor = 0x1p-110;         // absolute floor for exactly representable zero scales
double gamma(double n);                            // n u / (1 - n u)
} // namespace head_policy

// Test-only semantic variants: each changes one convention of the head so the
// tests can show that the comparison separates it from the real head. The
// default value is the head of SCAN_CONTRACT §2 and is the only one used for
// acceptance.
struct HeadSemantics {
    bool pixel_shuffle_dcr = false;     // DCR instead of CRD channel order
    bool anticausal_conv = false;       // keep the last L conv1d outputs instead of the first L
    bool direction_identity = false;    // direction 1 neither flipped on expand nor on merge
    bool merge_mismatch = false;        // merge restores direction 2 with flip H and 3 with flip W
    bool concat_reversed = false;       // [upsampled, input_proj] instead of [input_proj, upsampled]
    bool flipped_kernels = false;       // 3x3 kernels read spatially flipped
    bool swap_xz = false;               // in_proj halves exchanged
    bool swap_dt_b = false;             // x_proj dt and B slices exchanged
    bool c_first_state_only = false;    // rank-1 C applied to state 0 only instead of broadcast
    bool delta_bias_after_softplus = false;
    bool no_d_skip = false;
    bool no_residual = false;
    bool swap_level_weights = false;    // P4 and P5 parameters exchanged (same shapes)
};

// Binary64 values, worst-case bounds and probabilistic scales of one tensor (row-major).
struct HeadTensorRef {
    std::vector<std::int64_t> shape;
    std::vector<double> value, bound, sigma;
};

struct HeadReferenceResult {
    std::array<HeadTensorRef, 6> outputs; // kHeadOutputNames order
    std::vector<std::pair<std::string, HeadTensorRef>> intermediates; // forward order, when requested
};

HeadReferenceResult head_reference(const HeadWeights& weights, const std::array<std::span<const float>, 3>& features,
                                   const HeadSemantics& semantics = {}, bool keep_intermediates = false);

// Primitives of the reference, each with its bound and scale propagation;
// exposed for the independently derivable checks of tests/head_reference_tests.cpp.
namespace head_ref {
using T = HeadTensorRef;
T from_float(std::span<const float> values, std::vector<std::int64_t> shape); // exact widening, bound and sigma 0
// NCHW batch 1, OIHW weight, zero padding; `flipped` reads w[o][c][kh-1-i][kw-1-j].
T conv2d(const T& x, const HeadTensor& w, const HeadTensor* bias, int stride, int pad, bool flipped = false);
// Last axis: y[..., o] = sum_i w[o][i] x[..., i] (+ bias[o]).
T linear(const T& x, const HeadTensor& w, const HeadTensor* bias);
// x (B, L, C) along L, weight (C, 1, K): y[t] = b + sum_k w[k] x[t + k - (K-1)], zero for t < 0.
T causal_conv1d(const T& x, const HeadTensor& w, const HeadTensor& bias, bool anticausal = false);
T silu(const T& x);
T multiply(const T& a, const T& b);
T add(const T& a, const T& b);
// Columns [begin, end) of the last axis.
T slice_last(const T& x, std::int64_t begin, std::int64_t end);
// (1, C r^2, H, W) -> (1, C, H r, W r); CRD: out[c][h r + i][w r + j] = in[c r^2 + i r + j][h][w].
T pixel_shuffle(const T& x, int r, bool dcr = false);
T concat_channels(const T& a, const T& b);
// (1, C, H, W) -> (4, H W, C), direction k reads the k-flipped map in row-major order.
T cross_expand(const T& x, bool direction_identity = false);
// (4, H W, C) -> (1, C, H, W): each direction mapped back, (((y0 + y1) + y2) + y3) / 4.
T cross_merge(const T& y, std::int64_t height, std::int64_t width, bool direction_identity = false, bool mismatch = false);
// A = -exp(A_log), (rows, N).
T neg_exp(const HeadTensor& a_log);
// SCAN_CONTRACT §3 on binary64 inputs: u, delta (B, L, Dc), A (1, N), B (B, L, N), C (B, L, 1), D (Dc) or null.
// The scan's own FP32 rounding is S02's bound (SCAN_CUDA §5), in both measures;
// the first-order effect of the inputs' bounds / scales is added to it.
T selective_scan(const T& u, const T& delta, const T& a, const T& b, const T& c, const HeadTensor* d, bool c_first_state_only = false,
                 const HeadTensor* delta_bias_after = nullptr);
double softplus(double x); // x for x > 20, else log1p(exp(x))
} // namespace head_ref

// Deterministic FP32 head inputs (HEAD_TRT §4.5), fixed before any GPU run:
//   "asymmetric": 0.5 sin(0.37 c + 0.71 h (1 + c mod 3) - 0.23 w) + 0.002 (h - 0.3 w) + 0.01 (c mod 7)
//   "random": SiLU of a zero-mean deviate with standard deviation 1.5 (xorshift64*, Irwin-Hall 12), per seed and level
std::array<std::vector<float>, 3> head_test_features(const std::string& kind, std::uint64_t seed);
// Deterministic synthetic parameters with the schema's shapes (structure tests
// without the checkpoint): uniform +-1/sqrt(fan_in), A_log = ln(1..16), D = 1.
HeadWeights synthetic_head_weights(std::uint64_t seed);

} // namespace saccade

#pragma once

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define SACCADE_HD __host__ __device__
#else
#define SACCADE_HD
#endif

namespace saccade::tracking_math {

SACCADE_HD inline uint32_t float_bits(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    return __builtin_bit_cast(uint32_t, x);
#endif
}
SACCADE_HD inline float bits_float(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(x);
#else
    return __builtin_bit_cast(float, x);
#endif
}
SACCADE_HD inline bool finite(float x) { return (float_bits(x) & 0x7f800000u) != 0x7f800000u; }
SACCADE_HD inline float min(float a, float b) { return a < b ? a : b; }
SACCADE_HD inline float max(float a, float b) { return a > b ? a : b; }
SACCADE_HD inline float clamp(float x, float lo, float hi) { return min(max(x, lo), hi); }

SACCADE_HD inline float det_expf(float x) {
    x = clamp(x, -87.0f, 88.0f);
    const float nf = floorf(x * 1.4426950408889634f + 0.5f);
    const int n = static_cast<int>(nf);
    const float r = (x - nf * 0.693145751953125f) - nf * 1.428606765330187e-6f;
    float p = 0x1.a01a02p-16f;
    p = p * r + 0x1.a01a02p-13f;
    p = p * r + 0x1.6c16c2p-10f;
    p = p * r + 0x1.111112p-7f;
    p = p * r + 0x1.555556p-5f;
    p = p * r + 0x1.555556p-3f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    return p * bits_float(static_cast<uint32_t>(n + 127) << 23);
}

SACCADE_HD inline void box_to_state(const float* b, float* x) {
    const float w = b[2] - b[0], h = b[3] - b[1];
    x[0] = (b[0] + b[2]) / 2.0f;
    x[1] = (b[1] + b[3]) / 2.0f;
    x[2] = w / h;
    x[3] = h;
}
SACCADE_HD inline void state_to_box(const float* x, float* b) {
    const float half_w = (x[2] * x[3]) / 2.0f, half_h = x[3] / 2.0f;
    b[0] = x[0] - half_w;
    b[1] = x[1] - half_h;
    b[2] = x[0] + half_w;
    b[3] = x[1] + half_h;
}
SACCADE_HD inline float iou(const float* a, const float* b) {
    const float inter = max(0.0f, min(a[2], b[2]) - max(a[0], b[0])) *
                        max(0.0f, min(a[3], b[3]) - max(a[1], b[1]));
    const float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    const float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    const float area_union = (area_a + area_b) - inter;
    if (!finite(inter) || !finite(area_a) || !finite(area_b) || !finite(area_union))
        return bits_float(0x7fc00000u);
    return area_union > 0.0f ? inter / area_union : 0.0f;
}
SACCADE_HD inline float height_denominator(float hi, float hj) {
    return 1.0f + fabsf(hi - hj) / max(hj, 1e-3f);
}
SACCADE_HD inline float height_similarity(float hi, float hj) {
    return 1.0f / height_denominator(hi, hj);
}
SACCADE_HD inline float aspect_gain(float r) {
    if (r > 0.8f) return max(0.5f, 1.0f - (r - 0.8f));
    if (r < 0.15f) return max(0.5f, 1.0f - 5.0f * (0.15f - r));
    return 1.0f;
}
SACCADE_HD inline float association_cost(float overlap, float hi, float hj, float lambda, float weight) {
    const float denominator = height_denominator(hi, hj);
    if (!finite(denominator)) return bits_float(0x7fc00000u);
    const float reward = (weight / max(lambda, 1.0f)) / denominator;
    return clamp(1.0f - overlap * det_expf(reward), 0.0f, 1.0f);
}
SACCADE_HD inline float association_value(float cost, float wj, float hj, float lambda) {
    return det_expf(-lambda * cost) * aspect_gain(wj / hj);
}
SACCADE_HD inline float auction_bid(float p1, float p2, bool has_second, float eps,
                                    float weight, float hi, float hj) {
    const float delta = has_second ? (p1 - p2) + eps : eps;
    const float denominator = height_denominator(hi, hj);
    if (!finite(denominator)) return bits_float(0x7fc00000u);
    return delta + weight / denominator;
}

SACCADE_HD inline void kalman_init(const float* box, float* x, float* P, float std_p, float std_v) {
    box_to_state(box, x);
    for (int i = 4; i < 8; ++i) x[i] = 0.0f;
    for (int i = 0; i < 64; ++i) P[i] = 0.0f;
    const float sp = 2.0f * (std_p * x[3]), sv = 10.0f * (std_v * x[3]);
    const float dp[8] = {sp * sp, sp * sp, 1e-4f, sp * sp, sv * sv, sv * sv, 1e-10f, sv * sv};
    for (int i = 0; i < 8; ++i) P[i * 8 + i] = dp[i];
}
SACCADE_HD inline void kalman_predict(float* x, float* P, float std_p, float std_v) {
    for (int i = 0; i < 4; ++i) x[i] = x[i] + x[i + 4];
    float fp[64], next[64];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            fp[i * 8 + j] = i < 4 ? P[i * 8 + j] + P[(i + 4) * 8 + j] : P[i * 8 + j];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            next[i * 8 + j] = j < 4 ? fp[i * 8 + j] + fp[i * 8 + j + 4] : fp[i * 8 + j];
    const float sp = std_p * x[3], sv = std_v * x[3];
    const float q[8] = {sp * sp, sp * sp, 1e-4f, sp * sp, sv * sv, sv * sv, 1e-10f, sv * sv};
    for (int i = 0; i < 8; ++i) next[i * 8 + i] = next[i * 8 + i] + q[i];
    for (int i = 0; i < 64; ++i) P[i] = next[i];
}
SACCADE_HD inline void kalman_update(float* x, float* P, const float* box, float std_p, float r_scale) {
    float z[4];
    box_to_state(box, z);
    const float sp = std_p * x[3];
    const float noise[4] = {(sp * sp) * r_scale, (sp * sp) * r_scale,
                            1e-2f * r_scale, (sp * sp) * r_scale};
    float aug[32]; // Fixed-order Gauss-Jordan inversion of the 4x4 innovation covariance.
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 8; ++j)
            aug[i * 8 + j] = j < 4 ? P[i * 8 + j] + (i == j ? noise[i] : 0.0f)
                                               : (j == i + 4 ? 1.0f : 0.0f);
    for (int k = 0; k < 4; ++k) {
        const float pivot = aug[k * 8 + k];
        for (int j = 0; j < 8; ++j) aug[k * 8 + j] = aug[k * 8 + j] / pivot;
        for (int i = 0; i < 4; ++i) {
            if (i == k) continue;
            const float factor = aug[i * 8 + k];
            for (int j = 0; j < 8; ++j)
                aug[i * 8 + j] = aug[i * 8 + j] - factor * aug[k * 8 + j];
        }
    }
    float K[32], innovation[4], next[64];
    for (int j = 0; j < 4; ++j) innovation[j] = z[j] - x[j];
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum = sum + P[i * 8 + k] * aug[k * 8 + j + 4];
            K[i * 4 + j] = sum;
        }
        float delta = 0.0f;
        for (int j = 0; j < 4; ++j) delta = delta + K[i * 4 + j] * innovation[j];
        x[i] = x[i] + delta;
    }
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum = sum + K[i * 4 + k] * P[k * 8 + j];
            next[i * 8 + j] = P[i * 8 + j] - sum;
        }
    for (int i = 0; i < 64; ++i) P[i] = next[i];
}

SACCADE_HD inline float regression_velocity(float p0, float p1, float p2, float p3) {
    return ((3.0f * p3 + p2) - p1 - 3.0f * p0) / 10.0f;
}
SACCADE_HD inline float norm(float x, float y) { return sqrtf(x * x + y * y); }
SACCADE_HD inline float height_ema(float previous, float h) { return 0.95f * previous + 0.05f * h; }
SACCADE_HD inline float bridge_distance(float lx, float ly, float cx, float cy,
                                       float vlx, float vly, float vcx, float vcy, float gap,
                                       float hl, float hc, float speed_norm) {
    const float href = max((hl + hc) / 2.0f, 1.0f);
    const float rf = norm((lx + vlx * gap) - cx, (ly + vly * gap) - cy) / href;
    const float rb = norm((cx - vcx * gap) - lx, (cy - vcy * gap) - ly) / href;
    const float dh = norm(lx - cx, ly - cy) / href;
    const float speed = (norm(vlx, vly) / href) / speed_norm;
    if (!finite(href) || !finite(rf) || !finite(rb) || !finite(dh) || !finite(speed))
        return bits_float(0x7fc00000u);
    const float w = sqrtf(clamp(speed, 0.0f, 1.0f));
    return (w * (rf + rb)) / 2.0f + (1.0f - w) * dh;
}

} // namespace saccade::tracking_math

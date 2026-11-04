#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace saccade::flow::detail {
// Shared by the SCFL extractor and session F05. Compile with --fmad=false.
__device__ inline float rgb_gray(const std::uint8_t* rgb, std::size_t n, std::size_t i) {
    return (0.299f * float(rgb[i]) + 0.587f * float(rgb[n+i])) + 0.114f * float(rgb[2*n+i]);
}
__device__ inline std::uint8_t gray_pixel(const std::uint8_t* rgb, int width, int height, int s, int x, int y) {
    const auto n = static_cast<std::size_t>(width) * height;
    float v;
    if (s == 1) v = rgb_gray(rgb, n, static_cast<std::size_t>(y) * width + x);
    else {
        float sum = 0;
        for (int dy = 0; dy < s; ++dy) for (int dx = 0; dx < s; ++dx)
            sum += rgb_gray(rgb, n, static_cast<std::size_t>(y*s+dy) * width + x*s+dx);
        v = sum / float(s*s);
    }
    return static_cast<std::uint8_t>(fminf(255.f, floorf(v + 0.5f)));
}
}

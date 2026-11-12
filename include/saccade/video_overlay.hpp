#pragma once

#include "saccade/video_view.hpp"

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace saccade::video_overlay {
constexpr unsigned max_boxes = 256;
// 5x7 digits: high five bits of each row, with a one-pixel glyph gap.
inline constexpr std::uint8_t font[10][7] = {
    {14,17,19,21,25,17,14}, {4,12,4,4,4,4,14}, {14,17,1,2,4,8,31},
    {30,1,1,14,1,1,30}, {2,6,10,18,31,2,2}, {31,16,16,30,1,1,30},
    {14,16,16,30,17,17,14}, {31,1,2,4,8,8,8}, {14,17,17,14,17,17,14},
    {14,17,17,15,1,1,14}
};
struct Color { std::uint8_t y, u, v; };
constexpr Color bt709(int r, int g, int b) {
    const double luma = 0.2126*r + 0.7152*g + 0.0722*b;
    return {static_cast<std::uint8_t>(16 + 219*luma/255 + 0.5),
            static_cast<std::uint8_t>(128 + 224*(b-luma)/(1.8556*255) + 0.5),
            static_cast<std::uint8_t>(128 + 224*(r-luma)/(1.5748*255) + 0.5)};
}
inline constexpr Color palette[8] = {
    bt709(31,119,180), bt709(255,127,14), bt709(44,160,44), bt709(214,39,40),
    bt709(148,103,189), bt709(140,86,75), bt709(227,119,194), bt709(23,190,207)
};
struct Box {
    std::int64_t x1, y1, x2, y2, label_y;
    int label_width, label_height;
    Color color;
    int digits;
    std::uint8_t text[10];
};
struct Nv12 {
    std::uint8_t* y;
    std::uint8_t* uv;
    std::size_t y_pitch, uv_pitch;
    int width, height;
};
// Shared host preparation locks rounding, palette and ascending ID order.
std::vector<Box> prepare(std::span<const ViewBox>, int height);
// boxes is device storage and must live until stream completion. No allocation,
// atomics or readback. Each thread owns exactly one luma or interleaved UV sample.
cudaError_t launch(Nv12, const Box* boxes, unsigned count, cudaStream_t);
} // namespace saccade::video_overlay

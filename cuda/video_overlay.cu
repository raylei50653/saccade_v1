#include "saccade/video_overlay.hpp"

#include <cuda_runtime.h>

namespace saccade::video_overlay {
namespace {
__device__ __constant__ std::uint8_t digits[10][7] = {
    {14,17,19,21,25,17,14}, {4,12,4,4,4,4,14}, {14,17,1,2,4,8,31},
    {30,1,1,14,1,1,30}, {2,6,10,18,31,2,2}, {31,16,16,30,1,1,30},
    {14,16,16,30,17,17,14}, {31,1,2,4,8,8,8}, {14,17,17,14,17,17,14},
    {14,17,17,15,1,1,14}
};
__device__ bool outline(const Box& b, std::int64_t x, std::int64_t y, int t) {
    return x >= b.x1 && x <= b.x2 && y >= b.y1 && y <= b.y2 &&
        (x-b.x1 < t || b.x2-x < t || y-b.y1 < t || b.y2-y < t);
}
__device__ bool label(const Box& b, std::int64_t x, std::int64_t y) {
    return x >= b.x1 && x < b.x1+b.label_width &&
           y >= b.label_y && y < b.label_y+b.label_height;
}
__device__ bool digit(const Box& b, std::int64_t x, std::int64_t y, int s) {
    const auto dx = x-b.x1-s, dy = y-b.label_y-s;
    if (dx < 0 || dy < 0 || dy >= 7*s) return false;
    const auto column = dx/s, glyph = column/6;
    return glyph < b.digits && column%6 < 5 &&
        (digits[b.text[glyph]][dy/s] & (1 << (4-column%6)));
}
__global__ void overlay(Nv12 f, const Box* boxes, unsigned count) {
    const unsigned index = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned luma_count = f.width*f.height;
    const unsigned chroma_count = (f.width/2)*(f.height/2);
    const int t = max(2, f.height/360);
    if (index < luma_count) {
        const int x = index%f.width, y = index/f.width;
        auto& pixel = f.y[y*f.y_pitch+x];
        auto value = pixel;
        for (unsigned i = 0; i < count; ++i)
            if (outline(boxes[i], x, y, t)) value = boxes[i].color.y;
        for (unsigned i = 0; i < count; ++i)
            if (label(boxes[i], x, y))
                value = digit(boxes[i], x, y, t) ? 16 : boxes[i].color.y;
        pixel = value;
    } else if (index-luma_count < chroma_count) {
        const unsigned c = index-luma_count;
        const int x = 2*(c%(f.width/2)), y = 2*(c/(f.width/2));
        auto* uv = f.uv+(y/2)*f.uv_pitch+x;
        auto u = uv[0], v = uv[1];
        for (int layer = 0; layer < 2; ++layer) {
            for (unsigned i = 0; i < count; ++i) {
                bool covered = false;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx)
                        covered |= layer ? label(boxes[i], x+dx, y+dy) : outline(boxes[i], x+dx, y+dy, t);
                if (covered) { u = boxes[i].color.u; v = boxes[i].color.v; }
            }
        }
        uv[0] = u; uv[1] = v;
    }
}
} // namespace
cudaError_t launch(Nv12 frame, const Box* boxes, unsigned count, cudaStream_t stream) {
    const unsigned samples = frame.width*frame.height + (frame.width/2)*(frame.height/2);
    overlay<<<(samples+255)/256,256,0,stream>>>(frame, boxes, count);
    return cudaGetLastError();
}
} // namespace saccade::video_overlay

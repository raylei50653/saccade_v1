#include "saccade/video_overlay.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using saccade::ViewBox;
namespace overlay = saccade::video_overlay;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void cuda(cudaError_t result) { if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); }

// Independent scalar painter: visit elements, paint clipped luma pixels and
// their UV cells, then paint glyphs in Y only. The GPU instead visits samples
// and searches every element. Whole allocations (including padding) are checked.
void reference(overlay::Nv12 f, std::vector<ViewBox> boxes) {
    std::stable_sort(boxes.begin(), boxes.end(), [](auto a, auto b) { return a.id < b.id; });
    const int t = std::max(2, f.height/360);
    for (int layer = 0; layer < 2; ++layer) {
        for (const auto& b : boxes) {
            const long x1 = std::lround(b.xyxy[0]), y1 = std::lround(b.xyxy[1]);
            const long x2 = std::lround(b.xyxy[2]), y2 = std::lround(b.xyxy[3]);
            if (x2 < x1 || y2 < y1) continue;
            const auto color = overlay::palette[b.id%8];
            const auto text = std::to_string(b.id);
            const long lw = (6*static_cast<long>(text.size())+1)*t, lh = 9*t;
            const long top = y1-lh >= 0 ? y1-lh : y1;
            const long left = std::max(0L,x1), right = std::min<long>(f.width-1, layer ? x1+lw-1 : x2);
            const long begin = std::max(0L,layer ? top : y1);
            const long end = std::min<long>(f.height-1,layer ? top+lh-1 : y2);
            for (long y = begin; y <= end; ++y) {
                for (long x = left; x <= right; ++x) {
                    if (!layer && x-x1 >= t && x2-x >= t && y-y1 >= t && y2-y >= t) continue;
                    f.y[y*f.y_pitch+x] = color.y;
                    auto* uv = f.uv+(y/2)*f.uv_pitch+(x/2)*2;
                    uv[0] = color.u; uv[1] = color.v;
                }
            }
            if (!layer) continue;
            for (std::size_t glyph = 0; glyph < text.size(); ++glyph)
                for (int row = 0; row < 7; ++row)
                    for (int col = 0; col < 5; ++col) {
                        if (!(overlay::font[text[glyph]-'0'][row] & (1 << (4-col)))) continue;
                        for (int dy = 0; dy < t; ++dy)
                            for (int dx = 0; dx < t; ++dx) {
                                const long x = x1+t+(6*static_cast<long>(glyph)+col)*t+dx;
                                const long y = top+t+row*t+dy;
                                if (x >= 0 && x < f.width && y >= 0 && y < f.height)
                                    f.y[y*f.y_pitch+x] = 16;
                            }
                    }
        }
    }
}

void run_case(int width, int height, const std::vector<ViewBox>& boxes, unsigned& cases) {
    const std::size_t yp = width+13, uvp = width+29;
    std::vector<std::uint8_t> expected_y(yp*height), expected_uv(uvp*(height/2));
    for (std::size_t i = 0; i < expected_y.size(); ++i) expected_y[i] = (i*17+3)%256;
    for (std::size_t i = 0; i < expected_uv.size(); ++i) expected_uv[i] = (i*31+5)%256;
    auto actual_y = expected_y, actual_uv = expected_uv;
    std::uint8_t *y = nullptr, *uv = nullptr;
    overlay::Box* device_boxes = nullptr;
    cudaStream_t stream = nullptr;
    try {
        cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cuda(cudaMalloc(reinterpret_cast<void**>(&y),expected_y.size()));
        cuda(cudaMalloc(reinterpret_cast<void**>(&uv),expected_uv.size()));
        cuda(cudaMalloc(reinterpret_cast<void**>(&device_boxes),sizeof(overlay::Box)*overlay::max_boxes));
        cuda(cudaMemcpyAsync(y,actual_y.data(),actual_y.size(),cudaMemcpyHostToDevice,stream));
        cuda(cudaMemcpyAsync(uv,actual_uv.data(),actual_uv.size(),cudaMemcpyHostToDevice,stream));
        const auto prepared = overlay::prepare(boxes,height);
        if (!prepared.empty()) cuda(cudaMemcpyAsync(device_boxes,prepared.data(),prepared.size()*sizeof(overlay::Box),
                                                   cudaMemcpyHostToDevice,stream));
        cuda(overlay::launch({y,uv,yp,uvp,width,height},device_boxes,static_cast<unsigned>(prepared.size()),stream));
        cuda(cudaMemcpyAsync(actual_y.data(),y,actual_y.size(),cudaMemcpyDeviceToHost,stream));
        cuda(cudaMemcpyAsync(actual_uv.data(),uv,actual_uv.size(),cudaMemcpyDeviceToHost,stream));
        cuda(cudaStreamSynchronize(stream));
        reference({expected_y.data(),expected_uv.data(),yp,uvp,width,height},boxes);
        check(actual_y == expected_y,"luma byte parity or padding failed");
        check(actual_uv == expected_uv,"chroma byte parity or padding failed");
        cuda(cudaFree(y)); y = nullptr;
        cuda(cudaFree(uv)); uv = nullptr;
        cuda(cudaFree(device_boxes)); device_boxes = nullptr;
        cuda(cudaStreamDestroy(stream)); stream = nullptr;
    } catch (...) {
        if (stream) (void)cudaStreamSynchronize(stream);
        (void)cudaFree(y); (void)cudaFree(uv); (void)cudaFree(device_boxes);
        if (stream) (void)cudaStreamDestroy(stream);
        throw;
    }
    ++cases;
    std::cout << "PASS " << width << 'x' << height << " odd pitches=" << yp << ',' << uvp
              << " boxes=" << boxes.size() << '\n';
}
} // namespace
int main() {
    try {
        unsigned cases = 0;
        for (const auto [w,h] : {std::pair{1920,1080},std::pair{322,194}}) {
            run_case(w,h,{},cases);
            run_case(w,h,{{4294967295u,{12.5f,2.5f,72.5f,100.5f}}, {7,{-17.5f,-1.5f,34.5f,35.5f}},
                {0,{60,80,60,80}}, {12345,{61,81,61,130}}, {4,{100,100,99,120}}, {5,{2,8,20,7}},
                {6,{-100,-200,-90,-190}}, {2,{static_cast<float>(w-10),static_cast<float>(h-10),
                                           static_cast<float>(w+50),static_cast<float>(h+50)}},
                {3,{static_cast<float>(w+100),30,static_cast<float>(w+300),90}}},cases);
            run_case(w,h,{{12345,{15,55,180,140}}, {7,{30,70,190,160}}, {0,{10,50,170,130}},
                         {4294967295u,{16,4,185,120}}, {1,{-0.5f,30.5f,70.5f,90.5f}}},cases);
            std::vector<ViewBox> full;
            for (unsigned i = 0; i < 256; ++i) {
                const float x = static_cast<float>((i*43)%(w+90))-45.5f;
                const float y = static_cast<float>((i*29)%(h+60))-30.5f;
                full.push_back({255-i,{x,y,x+30+(i%20),y+40+(i%70)}});
            }
            run_case(w,h,full,cases);
        }
        bool rejected = false;
        try { overlay::prepare(std::vector<ViewBox>(257),194); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected,"box capacity must reject 257");
        std::cout << "PASS: " << cases << " CPU/GPU byte parity cases; capacity rejection; source padding preserved\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

#include "saccade/flow_cuda.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
std::vector<std::uint8_t> texture(int w, int h) {
    // A fixed integer hash; no random_device, external image or quality tuning.
    std::vector<std::uint8_t> image(static_cast<std::size_t>(w)*h);
    for (int y=0;y<h;++y) for (int x=0;x<w;++x) {
        std::uint32_t v = static_cast<std::uint32_t>(y)*static_cast<std::uint32_t>(w)+x+0x9e3779b9u;
        v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
        image[static_cast<std::size_t>(y)*w+x] = static_cast<std::uint8_t>(v >> 24);
    }
    return image;
}
std::vector<std::uint8_t> shift(const std::vector<std::uint8_t>& a, int w, int h, float dx, float dy) {
    std::vector<std::uint8_t> b(a.size());
    auto at=[&](int x,int y){return float(a[static_cast<std::size_t>(std::clamp(y,0,h-1))*w+std::clamp(x,0,w-1)]);};
    for (int y=0;y<h;++y) for(int x=0;x<w;++x) {
        const float sx=float(x)-dx, sy=float(y)-dy;
        const int ix=static_cast<int>(std::floor(sx)), iy=static_cast<int>(std::floor(sy));
        const float fx=sx-float(ix), fy=sy-float(iy);
        const float lo=at(ix,iy)*(1-fx)+at(ix+1,iy)*fx, hi=at(ix,iy+1)*(1-fx)+at(ix+1,iy+1)*fx;
        b[static_cast<std::size_t>(y)*w+x]=static_cast<std::uint8_t>(std::floor(lo*(1-fy)+hi*fy+0.5f));
    }
    return b;
}
float median(std::vector<float> v) {
    std::sort(v.begin(),v.end()); const auto n=v.size();
    return n%2 ? v[n/2] : (v[n/2-1]+v[n/2])*0.5f;
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2 || (std::string(argv[1])!="--f1" && std::string(argv[1])!="--f2"))
            throw std::runtime_error("usage: flow_cuda_tests --f1|--f2");
        const bool f1=std::string(argv[1])=="--f1";
        for (auto size: {std::array{1920,1080},std::array{640,480}}) {
            const int w=size[0], h=size[1]; const auto a=texture(w,h);
            const std::array<saccade::flow::Vector,3> shifts{{{8,4},{-5,3},{2.5f,-1.5f}}};
            for (int j=0;j<(f1?1:3);++j) {
                const auto truth=shifts[j];
                const auto b=shift(a,w,h,truth.dx,truth.dy);
                saccade::flow::Engine engine(w,h,saccade::flow::Perf::slow);
                saccade::flow::Timing timing;
                (void)engine.push_gray(a,0,timing);
                const auto flow=engine.push_gray(b,1,timing);
                std::cout << "NvOFExecute NV_OF_STATUS=0 " << w << 'x' << h << '\n';
                if(!f1) {
                    std::vector<float> x,y;
                    for(auto v:flow.cells) { x.push_back(v.dx); y.push_back(v.dy); }
                    const float mx=median(x), my=median(y);
                    const bool pass=std::abs(mx-truth.dx)<=0.25f && std::abs(my-truth.dy)<=0.25f;
                    std::cout << std::fixed << std::setprecision(6) << "F2 " << w << 'x' << h
                              << " truth=(" << truth.dx << ',' << truth.dy << ") all_cells=" << x.size()
                              << " median=(" << mx << ',' << my << ") " << (pass?"PASS":"FAIL") << std::endl;
                    if(!pass) throw std::runtime_error("F2 component median outside truth +/-0.25 px");
                }
                engine.close();
            }
        }
        std::cout << (f1?"F1":"F2") << " PASS\n"; return 0;
    } catch(const std::exception& e) { std::cerr << "flow_cuda_tests: " << e.what() << '\n'; return 1; }
}

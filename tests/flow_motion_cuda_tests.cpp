#include "saccade/flow_motion_cuda.hpp"
#include "saccade/gmc_flow.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using namespace saccade;
void ck(cudaError_t s) { if(s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void check(bool b,const char* label) { if(!b) throw std::runtime_error(label); }
template<class T> struct Device {
    T* p=nullptr;
    explicit Device(std::size_t n) { ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); ck(cudaMemset(p,0,n*sizeof(T))); }
    ~Device() { cudaFree(p); }
    void put(const T* src,std::size_t n) { ck(cudaMemcpy(p,src,n*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> get(std::size_t n) { std::vector<T> v(n); ck(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
};
void test(int W,int H,cudaStream_t stream) {
    const auto key=flow_image_key(W,H);
    const int s=key.downscale,w=key.width,h=key.height,cols=w/8,rows=h/8,n=cols*rows,pixels=w*h;
    const std::size_t plane=std::size_t(W)*H,pitch=w+128;
    std::mt19937 rng(0x92347);
    std::vector<std::uint8_t> rgb(3*plane);
    for(auto& x:rgb) x=static_cast<std::uint8_t>(rng());
    Device<std::uint8_t> d_rgb(rgb.size()),gray(pitch*h),state(2*pixels),masks(2*n);
    d_rgb.put(rgb.data(),rgb.size());
    flow_gray_cuda(d_rgb.p,gray.p,pitch,key,stream);
    ck(cudaStreamSynchronize(stream));
    const auto actual=gray.get(pitch*h);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) {
        float sum=0;
        for(int dy=0;dy<s;++dy) for(int dx=0;dx<s;++dx) {
            const auto i=std::size_t(y*s+dy)*W+x*s+dx;
            const float g=(0.299f*float(rgb[i])+0.587f*float(rgb[plane+i]))+0.114f*float(rgb[2*plane+i]);
            sum+=g;
        }
        const auto expected=static_cast<std::uint8_t>(std::min(255.f,std::floor(sum/float(s*s)+0.5f)));
        check(actual[y*pitch+x]==expected,"F05 grayscale bytes");
    }
    Device<Control> ctl(1);
    Control control; control.epoch=7; control.current_frame=1; control.committed_frame=0;
    control.committed_slot=1; control.working_slot=0; control.run_stage=true;
    ctl.put(&control,1);
    flow_output_cuda(gray.p,pitch,state.p,key,ctl.p,stream);
    ck(cudaStreamSynchronize(stream));
    auto copied=state.get(2*pixels);
    for(int y=0;y<h;++y) for(int x=0;x<w;++x) check(copied[y*w+x]==actual[y*pitch+x],"working gray copy");
    // Copy back from either committed slot, preserving the other slot.
    control.committed_slot=0; ctl.put(&control,1);
    ck(cudaMemsetAsync(gray.p,0,pitch*h,stream));
    flow_input_cuda(state.p,gray.p,pitch,key,ctl.p,stream); ck(cudaStreamSynchronize(stream));
    check(gray.get(pitch*h)==actual,"committed gray copy including pitch padding");
    control.run_stage=false; ctl.put(&control,1);
    ck(cudaMemsetAsync(gray.p,0x5a,pitch*h,stream));
    flow_input_cuda(state.p,gray.p,pitch,key,ctl.p,stream);
    flow_output_cuda(gray.p,pitch,state.p,key,ctl.p,stream); ck(cudaStreamSynchronize(stream));
    check(state.get(2*pixels)==copied,"gray copies obey run_stage");
    control.run_stage=true; ctl.put(&control,1);
    const int gw=(w+3)/4,gh=(h+3)/4;
    const std::size_t raw_pitch=gw*4+128;
    std::vector<std::uint8_t> raw(raw_pitch*gh,0);
    for(int y=0;y<gh;++y) for(int x=0;x<gw;++x) for(int axis=0;axis<2;++axis) {
        const auto v=static_cast<std::int16_t>(rng());
        std::memcpy(raw.data()+y*raw_pitch+4*x+2*axis,&v,2);
    }
    Device<std::uint8_t> d_raw(raw.size()); d_raw.put(raw.data(),raw.size());
    Device<flow::Vector> cells(n);
    flow_cells_cuda(d_raw.p,raw_pitch,cells.p,cols,rows,ctl.p,stream); ck(cudaStreamSynchronize(stream));
    flow::Frame frame{1,flow::Status::valid,cells.get(n)};
    for(int y=0;y<rows;++y) for(int x=0;x<cols;++x) {
        std::array<std::array<std::int16_t,2>,4> values{};
        for(int dy=0;dy<2;++dy) for(int dx=0;dx<2;++dx)
            std::memcpy(values[dy*2+dx].data(),raw.data()+(2*y+dy)*raw_pitch+(2*x+dx)*4,4);
        const auto expected=flow::reduce_cell(values), actual_cell=frame.cells[y*cols+x];
        if(actual_cell!=expected) {
            std::cerr << "first differing cells input W=" << W << " H=" << H << " s=" << s
                      << " cell_x=" << x << " cell_y=" << y << " grid=";
            for(const auto& v:values) std::cerr << '(' << v[0] << ',' << v[1] << ')';
            std::cerr << " GPU=(" << std::hexfloat << actual_cell.dx << ',' << actual_cell.dy
                      << ") CPU=(" << expected.dx << ',' << expected.dy << ")\n" << std::defaultfloat;
        }
        check(actual_cell==expected,"raw grid cell bytes");
    }
    DetectionCandidates detections{}; detections.header.count=6;
    detections.rows[0]={float(s*4),float(s*4),float(s*20),float(s*20),.45f,0,1,kDetectionKept};
    detections.rows[1]={0,0,float(W/2),float(H/2),.449f,0,2,kDetectionKept};
    detections.rows[2]={0,0,float(W),float(H),.9f,1,3,kDetectionKept};
    detections.rows[3]={0,0,float(W),float(H),.9f,0,4,kDetectionSuppressed};
    detections.rows[4]={float(W/2),float(H/2),float(W),float(H),.9f,0,5,kDetectionKept};
    detections.rows[5]={float(s*28)-.001f,0,float(s*28),float(H),.9f,0,6,kDetectionKept};
    Device<DetectionCandidates> candidates(1); candidates.put(&detections,1);
    control.committed_slot=1; control.working_slot=0; ctl.put(&control,1);
    flow_mask_cuda(candidates.p,masks.p,key,ctl.p,stream); ck(cudaStreamSynchronize(stream));
    auto mask=masks.get(2*n);
    const gmc_flow::FrameControl reference(frame,cols,rows,w,&detections,{},s);
    for(int i=0;i<n;++i) {
        check(mask[i]==reference.masked(i%cols,i/cols),"mask exact score/class/half-open boundaries");
        check(mask[n+i]==0,"mask preserves committed slot");
    }
    Device<std::uint8_t> scratch(flow_median_workspace_bytes(n));
    Device<MotionContext> context(1);
    // Full grid and odd/even/rejected subsets, signed extremes and ties.
    for(int kept:{n,9,8,7}) {
        std::fill(mask.begin(),mask.end(),1);
        for(int i=0;i<kept;++i) mask[n+i]=0;
        masks.put(mask.data(),mask.size());
        flow_median_cuda(cells.p,masks.p,n,s,ctl.p,context.p,scratch.p,flow_median_workspace_bytes(n),stream);
        ck(cudaStreamSynchronize(stream));
        const auto m=context.get(1)[0];
        check(m.epoch==7 && m.frame==1 && m.reference_frame==0 && m.peak_x==0 && m.peak_y==0 && m.pcr==0,"motion identity");
        if(kept<8) check(m.status==MotionStatus::rejected && m.tx==0 && m.ty==0 && m.confidence==0,"min_cells rejected");
        else {
            const auto med=gmc_flow::median({frame.cells.begin(),frame.cells.begin()+kept});
            check(m.status==MotionStatus::valid && m.confidence==1,"valid context");
            check(std::bit_cast<std::uint32_t>(m.tx)==std::bit_cast<std::uint32_t>(float(s)*med.dx) &&
                  std::bit_cast<std::uint32_t>(m.ty)==std::bit_cast<std::uint32_t>(float(s)*med.dy),"median exact bytes");
        }
    }
    control.committed_frame=-1; ctl.put(&control,1);
    flow_median_cuda(cells.p,masks.p,n,s,ctl.p,context.p,scratch.p,flow_median_workspace_bytes(n),stream);
    ck(cudaStreamSynchronize(stream));
    auto first=context.get(1)[0]; check(first.status==MotionStatus::uninitialized && first.confidence==0 && first.tx==0 && first.ty==0,"first frame");
    control.run_stage=false; ctl.put(&control,1);
    flow_mask_cuda(candidates.p,masks.p,key,ctl.p,stream);
    flow_cells_cuda(d_raw.p,raw_pitch,cells.p,cols,rows,ctl.p,stream);
    flow_median_cuda(cells.p,masks.p,n,s,ctl.p,context.p,scratch.p,flow_median_workspace_bytes(n),stream);
    ck(cudaStreamSynchronize(stream));
    const auto frozen=context.get(1)[0];
    check(std::memcmp(&first,&frozen,sizeof(first))==0 && masks.get(2*n)==mask && cells.get(n)==frame.cells,"guard freezes state/context");
    std::cout << "flow kernels " << W << 'x' << H << " s=" << s << " gray/cells/mask/median/copies byte parity PASS\n";
}
}
int main() {
    try {
        cudaStream_t stream; ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        test(1920,1080,stream); test(3840,2160,stream);
        ck(cudaStreamDestroy(stream)); return 0;
    } catch(const std::exception& e) { std::cerr << "flow_motion_cuda_tests: " << e.what() << '\n'; return 1; }
}

#include "saccade/gmc_flow.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

namespace {
void check(bool b,const char* text) { if(!b) throw std::runtime_error(text); }
}
int main() {
    try {
        using namespace saccade;
        using namespace saccade::gmc_flow;
        check(contains({4,4,12,12},4,4) && !contains({4,4,12,12},12,4) &&
              !contains({4,4,12,12},4,12),"left-closed right-open box");
        check(median({{9,-5},{1,7},{4,3}})==flow::Vector{4,3},"odd component median");
        check(median({{9,-5},{1,7},{4,3},{2,1}})==flow::Vector{3,2},"even component median");
        flow::Frame f{1,flow::Status::valid,{}};
        for(int r=0;r<8;++r) for(int c=0;c<8;++c) f.cells.push_back({float(8*c+4),float(8*r+4)});
        MotionContext pcr; pcr.status=MotionStatus::valid; pcr.tx=3; pcr.ty=-4;
        DetectionCandidates d{}; d.header.count=4;
        d.rows[0]={4,4,12,12,0.45f,0,0,kDetectionKept};
        d.rows[1]={12,4,20,12,0.449f,0,1,kDetectionKept};
        d.rows[2]={20,4,28,12,0.9f,1,2,kDetectionKept};
        d.rows[3]={28,4,36,12,0.9f,0,3,kDetectionSuppressed};
        FrameControl masked(f,8,8,64,&d,pcr);
        check(masked.masked(0,0) && !masked.masked(1,0) && !masked.masked(2,0) &&
              !masked.masked(3,0) && !masked.masked(0,1) && masked.available()==63,"kept/class/score/box mask");
        FrameControl plain(f,8,8,64,nullptr,pcr);
        const auto ring=plain.evaluate(Arm::l1,{24,16,40,48});
        check(!ring.fallback && ring.local_cells==28 && ring.value==flow::Vector{32,32},"ring geometry and own-box exclusion");
        const auto narrow=plain.evaluate(Arm::l2,{24,16,40,48});
        check(narrow.fallback && narrow.local_cells==6 && narrow.value==plain.global(),"foot band six cells falls back");
        const auto band=plain.evaluate(Arm::l2,{16,8,48,40});
        check(!band.fallback && band.local_cells==8 && band.value==flow::Vector{32,44},"foot band eight cells accepted");
        check(plain.evaluate(Arm::off,{}).value==flow::Vector{} &&
              plain.evaluate(Arm::g0,{}).value==flow::Vector{3,-4},"OFF and G0 controls");
        d.header.count=1; d.rows[0]={8,8,56,56,0.9f,0,0,kDetectionKept};
        FrameControl occluded(f,8,8,64,&d,pcr);
        const auto fallback=occluded.evaluate(Arm::l1,{24,16,40,48});
        check(fallback.fallback && fallback.local_cells==0 && fallback.value==occluded.global(),"local foreground mask forces G1");
        flow::Frame tiny{1,flow::Status::valid,std::vector<flow::Vector>(8,{2,-2})};
        FrameControl eight(tiny,8,1,64,nullptr,pcr);
        check(eight.valid() && eight.global()==flow::Vector{2,-2},"global eight-cell cutoff inclusive");
        check(eight.evaluate(Arm::g1q,{}).value==flow::Vector{4,-4},"positive and negative half-away quantization ties");
        DetectionCandidates one{}; one.header.count=1; one.rows[0]={0,0,8,8,0.45f,0,0,kDetectionKept};
        FrameControl seven(tiny,8,1,64,&one,pcr);
        check(!seven.valid() && seven.global()==flow::Vector{},"seven global cells rejected and zeroed");
        const auto rejected=seven.evaluate(Arm::l2,{0,0,8,8});
        check(rejected.fallback && rejected.value==flow::Vector{},"local fallback to rejected G1 is zero");
        std::vector<double> u,values;
        for(int i=0;i<80;++i) {
            u.push_back(i-40);
            values.push_back(2+0.125*(i-40)+(i%5==0?40:0));
        }
        const auto fit=tukey_line(u,values);
        check(std::abs(fit.a-2)<1e-8 && std::abs(fit.b-0.125)<1e-8,"Tukey recovers line with 20 percent outliers");
        check(concentration_alpha(0)==0 && concentration_alpha(0.2)==0 &&
              concentration_alpha(0.7)==1 && concentration_alpha(1)==1 &&
              std::abs(concentration_alpha(0.4)-0.5)<1e-12,"alpha endpoints and ramp");
        const std::vector<WeightedCell> modes{{-5,0,1},{5,0,1}};
        check(density_start(modes)==0,"LD first cell wins density tie");
        const std::vector<WeightedCell> cluster{{-5,0,1},{3,2,1},{3,2,1},{3,2,1}};
        check(density_start(cluster)==1,"LD starts at highest density, first in tied mode");
        const auto mode=mean_shift(cluster,{3.2,2.1});
        check(std::hypot(mode[0]-3,mode[1]-2)<0.01,"mean-shift converges to known mode");
        const auto remote=mean_shift(cluster,{100,100});
        check(remote[0]==100 && remote[1]==100,"mean-shift stops for kernel sum below threshold");
        flow::Frame rows{1,flow::Status::valid,{}};
        for(int r=0;r<10;++r) for(int c=0;c<16;++c) rows.cells.push_back({float(r),float(2*r)});
        DetectionCandidates rowmask{}; rowmask.header.count=2;
        rowmask.rows[0]={0,8,128,48,0.9f,0,0,kDetectionKept};
        rowmask.rows[1]={0,56,128,80,0.9f,0,1,kDetectionKept};
        FrameControl neighbors(rows,16,10,128,&rowmask,pcr);
        const auto neighbor=neighbors.evaluate(Arm::r,{32,0,64,24});
        check(!neighbor.r_fallback && neighbor.row_offset==-2 && neighbor.value==flow::Vector{0,0},
              "row needs 16 cells and searches negative before positive at same distance");
        rowmask.header.count=1; rowmask.rows[0]={0,8,128,80,0.9f,0,0,kDetectionKept};
        FrameControl distant(rows,16,10,128,&rowmask,pcr);
        const auto far=distant.evaluate(Arm::r,{32,0,64,56});
        check(far.r_fallback && far.value==distant.global(),"R search stops at four rows and falls back to G1");
        rowmask.header.count=2; rowmask.rows[1]={0,0,8,8,0.9f,0,1,kDetectionKept};
        FrameControl insufficient(rows,16,10,128,&rowmask,pcr);
        check(insufficient.evaluate(Arm::r,{32,0,64,0}).r_fallback,"15 support cells make row unavailable");
        flow::Frame plane{1,flow::Status::valid,std::vector<flow::Vector>(160,{2,-3})};
        FrameControl constant(plane,16,10,128,nullptr,pcr);
        const Box wide{32,0,96,40};
        check(constant.evaluate(Arm::r,wide).value==flow::Vector{2,-3},"R evaluates available foot row");
        check(constant.evaluate(Arm::ld,wide).value==flow::Vector{2,-3},"LD uses registered foot band and known mode");
        const auto rc=constant.evaluate(Arm::rc,wide);
        check(rc.alpha==1 && rc.density==1 && rc.value==flow::Vector{2,-3},"RC zero residual has unit concentration and alpha");
        const auto rc_fallback=constant.evaluate(Arm::rc,{32,0,96,79});
        check(rc_fallback.fallback && rc_fallback.alpha==0 && rc_fallback.value==flow::Vector{2,-3},"RC below eight band cells falls back to R");
        check(constant.evaluate(Arm::ld,{32,0,96,79}).fallback,"LD below eight band cells falls back to G1");
        std::cout << "row prior, neighbor search, LD, mean-shift and RC PASS\n";
        std::cout << "mask, ring, foot band, median, quantization and fallback PASS\n"; return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

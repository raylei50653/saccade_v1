#include "saccade/gmc_flow.hpp"
#include <iostream>
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
        FrameControl masked(f,8,8,&d,pcr);
        check(masked.masked(0,0) && !masked.masked(1,0) && !masked.masked(2,0) &&
              !masked.masked(3,0) && !masked.masked(0,1) && masked.available()==63,"kept/class/score/box mask");
        FrameControl plain(f,8,8,nullptr,pcr);
        const auto ring=plain.evaluate(Arm::l1,{24,16,40,48});
        check(!ring.fallback && ring.local_cells==28 && ring.value==flow::Vector{32,32},"ring geometry and own-box exclusion");
        const auto narrow=plain.evaluate(Arm::l2,{24,16,40,48});
        check(narrow.fallback && narrow.local_cells==6 && narrow.value==plain.global(),"foot band six cells falls back");
        const auto band=plain.evaluate(Arm::l2,{16,8,48,40});
        check(!band.fallback && band.local_cells==8 && band.value==flow::Vector{32,44},"foot band eight cells accepted");
        check(plain.evaluate(Arm::off,{}).value==flow::Vector{} &&
              plain.evaluate(Arm::g0,{}).value==flow::Vector{3,-4},"OFF and G0 controls");
        d.header.count=1; d.rows[0]={8,8,56,56,0.9f,0,0,kDetectionKept};
        FrameControl occluded(f,8,8,&d,pcr);
        const auto fallback=occluded.evaluate(Arm::l1,{24,16,40,48});
        check(fallback.fallback && fallback.local_cells==0 && fallback.value==occluded.global(),"local foreground mask forces G1");
        flow::Frame tiny{1,flow::Status::valid,std::vector<flow::Vector>(8,{2,-2})};
        FrameControl eight(tiny,8,1,nullptr,pcr);
        check(eight.valid() && eight.global()==flow::Vector{2,-2},"global eight-cell cutoff inclusive");
        check(eight.evaluate(Arm::g1q,{}).value==flow::Vector{4,-4},"positive and negative half-away quantization ties");
        DetectionCandidates one{}; one.header.count=1; one.rows[0]={0,0,8,8,0.45f,0,0,kDetectionKept};
        FrameControl seven(tiny,8,1,&one,pcr);
        check(!seven.valid() && seven.global()==flow::Vector{},"seven global cells rejected and zeroed");
        const auto rejected=seven.evaluate(Arm::l2,{0,0,8,8});
        check(rejected.fallback && rejected.value==flow::Vector{},"local fallback to rejected G1 is zero");
        std::cout << "mask, ring, foot band, median, quantization and fallback PASS\n"; return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

#include "saccade/gmc_g1.hpp"
#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* label) {
    if (!condition) throw std::runtime_error(label);
}
}
int main() {
    try {
        using namespace saccade;
        using namespace saccade::gmc_g1;
        check(contains({4,4,12,12},4,4) && !contains({4,4,12,12},12,4) &&
              !contains({4,4,12,12},4,12), "half-open box");
        check(median({{9,-5},{1,7},{4,3}})==flow::Vector{4,3}, "odd component median");
        check(median({{9,-5},{1,7},{4,3},{2,1}})==flow::Vector{3,2}, "even component median");
        flow::Frame frame{1,flow::Status::valid,{}};
        for (int row=0;row<8;++row) for (int col=0;col<8;++col)
            frame.cells.push_back({float(8*col+4),float(8*row+4)});
        DetectionCandidates detections{}; detections.header.count=4;
        detections.rows[0]={4,4,12,12,.45f,0,0,kDetectionKept};
        detections.rows[1]={12,4,20,12,.449f,0,1,kDetectionKept};
        detections.rows[2]={20,4,28,12,.9f,1,2,kDetectionKept};
        detections.rows[3]={28,4,36,12,.9f,0,3,kDetectionSuppressed};
        FrameControl masked(frame,8,8,&detections);
        check(masked.masked(0,0) && !masked.masked(1,0) && !masked.masked(2,0) &&
              !masked.masked(3,0) && !masked.masked(0,1) && masked.available()==63,
              "kept pedestrian score boundary mask");
        FrameControl plain(frame,8,8,nullptr);
        check(plain.valid() && plain.global()==flow::Vector{32,32}, "unmasked G1 median");
        flow::Frame tiny{1,flow::Status::valid,std::vector<flow::Vector>(8,{2,-2})};
        FrameControl eight(tiny,8,1,nullptr);
        check(eight.valid() && eight.global()==flow::Vector{2,-2}, "eight-cell cutoff");
        DetectionCandidates one{}; one.header.count=1;
        one.rows[0]={0,0,8,8,.45f,0,0,kDetectionKept};
        FrameControl seven(tiny,8,1,&one);
        check(!seven.valid() && seven.global()==flow::Vector{}, "seven cells rejected");
        tiny.status=flow::Status::uninitialized;
        FrameControl failed(tiny,8,1,nullptr);
        check(!failed.valid() && failed.global()==flow::Vector{}, "uninitialized flow rejected");
        // The scaled mask must use original-image cell centers and the same
        // half-open threshold arithmetic as the CUDA G1 implementation.
        detections.header.count=1;
        detections.rows[0]={8,8,24,24,.45f,0,0,kDetectionKept};
        FrameControl scaled(frame,8,8,&detections,2);
        check(scaled.masked(0,0) && !scaled.masked(1,0) && !scaled.masked(0,1), "scaled centers");
        std::cout << "G1 mask, median, status and minimum-cell controls PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}

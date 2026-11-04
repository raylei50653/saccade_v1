#pragma once

#include "saccade/flow_records.hpp"
#include "saccade/motion.hpp"
#include "saccade/postprocess.hpp"
#include <array>
#include <string_view>

namespace saccade::gmc_flow {
enum class Arm { off, g0, g1, g1q, l1, l2 };
inline constexpr std::array arms{Arm::off,Arm::g0,Arm::g1,Arm::g1q,Arm::l1,Arm::l2};
Arm parse_arm(std::string_view text);
const char* name(Arm arm);
using Box = std::array<float,4>;
bool contains(const Box& box, float x, float y);
flow::Vector median(std::vector<flow::Vector> values);
struct Control {
    flow::Vector value;
    bool fallback = false;
    std::size_t local_cells = 0;
};
// One frame's fixed mask and G1. Both the tracker hook and GT residual call
// this exact implementation. Boxes are in the previous image coordinates.
class FrameControl {
public:
    FrameControl(const flow::Frame& frame, int cols, int rows,
                 const DetectionCandidates* previous, const MotionContext& pcr);
    Control evaluate(Arm arm, const Box& own_box) const;
    flow::Vector global() const { return global_; }
    bool valid() const { return available_ >= 8 && frame_->status == flow::Status::valid; }
    std::size_t available() const { return available_; }
    bool masked(int col, int row) const { return mask_[static_cast<std::size_t>(row)*cols_+col] != 0; }
private:
    const flow::Frame* frame_;
    int cols_, rows_;
    flow::Vector pcr_, global_;
    std::vector<std::uint8_t> mask_;
    std::size_t available_ = 0;
};
} // namespace saccade::gmc_flow

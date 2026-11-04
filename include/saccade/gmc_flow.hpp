#pragma once

#include "saccade/flow_records.hpp"
#include "saccade/motion.hpp"
#include "saccade/postprocess.hpp"
#include <array>
#include <string_view>

namespace saccade::gmc_flow {
enum class Arm { off, g0, g1, g1q, l1, l2, ld, r, rc };
inline constexpr std::array arms{Arm::off,Arm::g0,Arm::g1,Arm::g1q,Arm::l1,Arm::l2,Arm::ld,Arm::r,Arm::rc};
Arm parse_arm(std::string_view text);
const char* name(Arm arm);
using Box = std::array<float,4>;
bool contains(const Box& box, float x, float y);
flow::Vector median(std::vector<flow::Vector> values);
struct Line { double a = 0, b = 0; };
Line tukey_line(const std::vector<double>& u, const std::vector<double>& d);
double concentration_alpha(double density);
struct WeightedCell { double dx, dy, weight; };
std::size_t density_start(const std::vector<WeightedCell>& cells);
std::array<double,2> mean_shift(const std::vector<WeightedCell>& cells, std::array<double,2> start);
struct Control {
    flow::Vector value;
    bool fallback = false;
    std::size_t local_cells = 0;
    bool r_fallback = false;
    int row_offset = 0;
    double alpha = 0;
    double density = 0;
};
// One frame's fixed mask and G1. Both the tracker hook and GT residual call
// this exact implementation. Boxes are in the previous image coordinates.
class FrameControl {
public:
    FrameControl(const flow::Frame& frame, int cols, int rows, int width,
                 const DetectionCandidates* previous, const MotionContext& pcr);
    Control evaluate(Arm arm, const Box& own_box) const;
    flow::Vector global() const { return global_; }
    bool valid() const { return available_ >= 8 && frame_->status == flow::Status::valid; }
    std::size_t available() const { return available_; }
    bool masked(int col, int row) const { return mask_[static_cast<std::size_t>(row)*cols_+col] != 0; }
private:
    struct RowModel { bool available = false; Line dx, dy; };
    void prepare_rows() const;
    flow::Vector row_value(int row, double x) const;
    Control ground(const Box& box) const;
    const flow::Frame* frame_;
    int cols_, rows_, width_;
    flow::Vector pcr_, global_;
    std::vector<std::uint8_t> mask_;
    std::size_t available_ = 0;
    mutable std::vector<RowModel> ground_rows_;
};
} // namespace saccade::gmc_flow

#pragma once

#include "saccade/flow_records.hpp"
#include "saccade/postprocess.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace saccade::gmc_g1 {
using Box = std::array<float, 4>;
bool contains(const Box& box, float x, float y);
flow::Vector median(std::vector<flow::Vector> values);

// Production G1 reference: previous-frame pedestrian mask and component median.
// The caller retains the SCFL frame for the lifetime of this view. Arithmetic
// and half-open mask boundaries match the original G1 experiment implementation.
class FrameControl {
public:
    FrameControl(const flow::Frame& frame, int cols, int rows,
                 const DetectionCandidates* previous, int scale = 1);
    flow::Vector global() const { return global_; }
    bool valid() const { return available_ >= 8 && frame_->status == flow::Status::valid; }
    std::size_t available() const { return available_; }
    bool masked(int col, int row) const { return mask_[static_cast<std::size_t>(row)*cols_+col] != 0; }

private:
    const flow::Frame* frame_;
    int cols_;
    flow::Vector global_{};
    std::vector<std::uint8_t> mask_;
    std::size_t available_ = 0;
};
} // namespace saccade::gmc_g1

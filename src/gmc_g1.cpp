#include "saccade/gmc_g1.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace saccade::gmc_g1 {
bool contains(const Box& b, float x, float y) {
    return x >= b[0] && x < b[2] && y >= b[1] && y < b[3];
}

flow::Vector median(std::vector<flow::Vector> values) {
    if (values.empty()) throw std::invalid_argument("median of empty cells");
    const auto n = values.size();
    auto component = [&](auto member) {
        std::sort(values.begin(), values.end(), [&](const auto& a, const auto& b) {
            return a.*member < b.*member;
        });
        return n%2 ? values[n/2].*member : (values[n/2-1].*member + values[n/2].*member)*0.5f;
    };
    return {component(&flow::Vector::dx), component(&flow::Vector::dy)};
}

FrameControl::FrameControl(const flow::Frame& frame, int cols, int rows,
                           const DetectionCandidates* previous, int scale)
    : frame_(&frame), cols_(cols), mask_(static_cast<std::size_t>(cols)*rows, 0) {
    if (previous) {
        for (std::uint32_t i = 0; i < previous->header.count; ++i) {
            const auto& d = previous->rows[i];
            if (!(d.flags & kDetectionKept) || d.class_id != 0 || d.score < 0.45f) continue;
            const Box box{d.x1, d.y1, d.x2, d.y2};
            const auto bound = [&](float edge, int size) {
                int lo = 0, hi = size;
                while (lo < hi) {
                    const int mid = lo + (hi-lo)/2;
                    if (float(scale*(8*mid+4)) < edge) lo = mid+1;
                    else hi = mid;
                }
                return lo;
            };
            const int c0 = bound(box[0], cols), c1 = bound(box[2], cols);
            const int r0 = bound(box[1], rows), r1 = bound(box[3], rows);
            for (int r = r0; r < r1; ++r)
                for (int c = c0; c < c1; ++c)
                    mask_[static_cast<std::size_t>(r)*cols+c] = 1;
        }
    }
    std::vector<flow::Vector> background;
    for (std::size_t i = 0; i < mask_.size(); ++i)
        if (!mask_[i]) background.push_back(frame.cells[i]);
    available_ = background.size();
    if (valid()) global_ = median(std::move(background));
}
} // namespace saccade::gmc_g1

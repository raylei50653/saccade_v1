#include "saccade/tracking.hpp"

#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace saccade {
void validate_tracker_config(const TrackerConfig& c) {
    const float values[] = {
        c.kalman.r_scale, c.kalman.std_position, c.kalman.std_velocity, c.batch.track_thresh,
        c.continuation.min_score, c.continuation.prior_iou, c.continuation.dedup_iou,
        c.assoc.high_thresh, c.assoc.mid_thresh, c.assoc.match_thresh, c.assoc.stage2_match_thresh,
        c.assoc.lambda, c.assoc.stability_cost_w, c.assoc.auction_eps, c.assoc.stability_bid_w,
        c.life.new_track_thresh, c.life.confirm_score, c.bridge.max_distance, c.bridge.height_ratio_lo,
        c.bridge.height_ratio_hi, c.bridge.margin, c.bridge.speed_norm};
    for (float v : values)
        if (!std::isfinite(v)) throw std::invalid_argument("non-finite tracker config");
    const float thresholds[] = {
        c.batch.track_thresh, c.continuation.min_score, c.continuation.prior_iou, c.continuation.dedup_iou,
        c.assoc.high_thresh, c.assoc.mid_thresh, c.assoc.match_thresh, c.assoc.stage2_match_thresh,
        c.life.new_track_thresh, c.life.confirm_score};
    for (float v : thresholds)
        if (v < 0 || v > 1) throw std::invalid_argument("tracker threshold outside [0,1]");
    if (!(c.batch.track_thresh <= c.assoc.mid_thresh && c.assoc.mid_thresh <= c.assoc.high_thresh))
        throw std::invalid_argument("tracker score thresholds out of order");
    if (c.kalman.r_scale <= 0 || c.kalman.std_position <= 0 || c.kalman.std_velocity <= 0 ||
        c.assoc.lambda <= 0 || c.assoc.auction_eps <= 0 || c.bridge.speed_norm <= 0)
        throw std::invalid_argument("tracker scale must be positive");
    if (c.assoc.stability_cost_w < 0 || c.assoc.stability_bid_w < 0 ||
        c.bridge.margin < 0 || c.bridge.max_distance < 0)
        throw std::invalid_argument("negative tracker weight or distance");
    if (c.assoc.top_k < 1 || c.assoc.top_k > 8) throw std::invalid_argument("tracker top_k outside [1,8]");
    if (c.life.confirm_streak < 1 || c.life.track_buffer < 1 || c.bridge.trigger_streak < kBridgeHistory ||
        c.bridge.min_lost < 1) throw std::invalid_argument("invalid tracker lifecycle count");
    if (!(0 < c.bridge.height_ratio_lo && c.bridge.height_ratio_lo < c.bridge.height_ratio_hi))
        throw std::invalid_argument("invalid bridge height ratios");
    if (c.continuation.max_candidates > kBatchCapacity)
        throw std::invalid_argument("continuation max_candidates exceeds capacity");
}

std::string canonical_tracker_config(const TrackerConfig& c) {
    validate_tracker_config(c);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hexfloat;
#define FIELD(name) out << #name "=" << c.name << '\n'
    FIELD(kalman.r_scale);
    FIELD(kalman.std_position);
    FIELD(kalman.std_velocity);
    FIELD(batch.track_thresh);
    FIELD(continuation.enabled);
    FIELD(continuation.min_score);
    FIELD(continuation.prior_iou);
    FIELD(continuation.dedup_iou);
    FIELD(continuation.max_candidates);
    FIELD(assoc.high_thresh);
    FIELD(assoc.mid_thresh);
    FIELD(assoc.match_thresh);
    FIELD(assoc.stage2_match_thresh);
    FIELD(assoc.lambda);
    FIELD(assoc.stability_cost_w);
    FIELD(assoc.top_k);
    FIELD(assoc.auction_eps);
    FIELD(assoc.stability_bid_w);
    FIELD(life.new_track_thresh);
    FIELD(life.confirm_streak);
    FIELD(life.confirm_score);
    FIELD(life.track_buffer);
    FIELD(bridge.enabled);
    FIELD(bridge.trigger_streak);
    FIELD(bridge.min_lost);
    FIELD(bridge.max_distance);
    FIELD(bridge.height_ratio_lo);
    FIELD(bridge.height_ratio_hi);
    FIELD(bridge.margin);
    FIELD(bridge.speed_norm);
#undef FIELD
    return out.str();
}
} // namespace saccade

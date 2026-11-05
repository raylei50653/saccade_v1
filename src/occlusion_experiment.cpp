#include "saccade/occlusion_experiment.hpp"
#include "saccade/tracking_math.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace saccade {
namespace {
namespace math = tracking_math;
float numeric_failure() { return math::bits_float(0x7fc00000u); }
} // namespace

void validate_occlusion_experiment_config(const OcclusionExperimentConfig& c) {
    if (!math::finite(c.oao_tau) || c.oao_tau < 0 || !c.oao_ramp_frames ||
        !math::finite(c.front_iou_thresh) || c.front_iou_thresh < 0 || c.front_iou_thresh > 1 ||
        !math::finite(c.front_foot_gap) || c.front_foot_gap < 0 || !c.front_ttl ||
        !math::finite(c.front_cost_weight) || c.front_cost_weight < 0)
        throw std::invalid_argument("invalid CPU occlusion experiment config");
}

OcclusionExperiment::OcclusionExperiment(const TrackerConfig& tracker,
                                         const OcclusionExperimentConfig& experiment)
    : tracker_(tracker), config_(experiment) {
    validate_tracker_config(tracker_);
    validate_occlusion_experiment_config(config_);
}

const OcclusionTrackState* OcclusionExperiment::state(std::uint64_t uid) const {
    const auto found = states_.find(uid);
    return found == states_.end() ? nullptr : &found->second;
}

Error OcclusionExperiment::prepare(const PredictedState& predicted) {
    if (predicted.error != Error::none) return predicted.error;
    if (predicted.frame < 0) return Error::protocol;
    const bool same_epoch = prepared_ && epoch_ == predicted.epoch;
    if (same_epoch && frame_ == predicted.frame) return Error::none;
    if (same_epoch && predicted.frame - 1 != frame_) return Error::protocol;

    std::vector<std::array<float, 4>> boxes(predicted.tracks.size());
    std::uint64_t previous_uid = 0;
    for (std::size_t i = 0; i < predicted.tracks.size(); ++i) {
        const auto& t = predicted.tracks[i];
        if (!t.uid || t.uid <= previous_uid) return Error::protocol;
        previous_uid = t.uid;
        for (float x : t.x) if (!math::finite(x)) return Error::numeric;
        if (t.x[2] <= 0 || t.x[3] <= 0) return Error::numeric;
        math::state_to_box(t.x.data(), boxes[i].data());
        for (float value : boxes[i]) if (!math::finite(value)) return Error::numeric;
    }

    std::map<std::uint64_t, OcclusionTrackState> next;
    auto diagnostics = diagnostics_;
    for (std::size_t i = 0; i < predicted.tracks.size(); ++i) {
        const auto& t = predicted.tracks[i];
        OcclusionTrackState current;
        if (same_epoch) {
            const auto* previous = state(t.uid);
            if (previous) current = *previous;
        }
        float maximum = 0;
        std::size_t partner = predicted.tracks.size();
        for (std::size_t j = 0; j < predicted.tracks.size(); ++j) {
            if (i == j) continue;
            const float overlap = math::iou(boxes[i].data(), boxes[j].data());
            if (!math::finite(overlap)) return Error::numeric;
            if (overlap > maximum) { maximum = overlap; partner = j; }
        }
        current.max_overlap = maximum;
        current.partner_uid = partner < predicted.tracks.size() ? predicted.tracks[partner].uid : 0;
        if (maximum > 0) {
            if (current.overlap_duration < std::numeric_limits<std::uint32_t>::max())
                ++current.overlap_duration;
            ++diagnostics.overlapped_track_frames;
        } else current.overlap_duration = 0;
        const float ramp = math::min(1.0f, static_cast<float>(current.overlap_duration) /
                                                   static_cast<float>(config_.oao_ramp_frames));
        current.oao_coefficient = maximum * ramp;

        bool front = false;
        if (config_.front_enabled && partner < predicted.tracks.size() && maximum >= config_.front_iou_thresh) {
            const auto& other = predicted.tracks[partner];
            const float foot = t.x[1] + t.x[3] * 0.5f;
            const float other_foot = other.x[1] + other.x[3] * 0.5f;
            const float reference_height = 0.5f * (t.x[3] + other.x[3]);
            const float gap = (foot - other_foot) / math::max(reference_height, 1e-3f);
            if (!math::finite(foot) || !math::finite(other_foot) ||
                !math::finite(reference_height) || !math::finite(gap)) return Error::numeric;
            front = std::fabs(gap) <= config_.front_foot_gap && foot > other_foot;
        }
        current.front_ttl = front ? config_.front_ttl : current.front_ttl ? current.front_ttl - 1 : 0;
        if (current.front_ttl) ++diagnostics.front_latched_track_frames;
        next.emplace(t.uid, current);
    }
    ++diagnostics.frames;
    states_ = std::move(next);
    diagnostics_ = diagnostics;
    prepared_ = true;
    epoch_ = predicted.epoch;
    frame_ = predicted.frame;
    return Error::none;
}

float OcclusionExperiment::cost(const Track& t, const BatchDetection& d, float overlap) {
    ++diagnostics_.cost_calls;
    const float height = d.xyxy[3] - d.xyxy[1];
    if (!math::finite(overlap) || !math::finite(height) ||
        !math::finite(t.x[1]) || !math::finite(t.x[3]) || !math::finite(d.xyxy[3]))
        return numeric_failure();
    if (!config_.oao_enabled && !config_.front_enabled)
        return math::association_cost(overlap, t.x[3], height, tracker_.assoc.lambda, tracker_.assoc.stability_cost_w);
    const auto* current = state(t.uid);
    if (!current) return numeric_failure();
    float penalty = 0;
    if (config_.oao_enabled) {
        const float oao = config_.oao_tau * current->oao_coefficient;
        if (!math::finite(oao)) return numeric_failure();
        if (oao > 0) { penalty += oao; ++diagnostics_.oao_positive_calls; }
    }
    if (config_.front_enabled && current->front_ttl && config_.front_cost_weight > 0) {
        const float foot = t.x[1] + t.x[3] * 0.5f;
        const float under = (foot - d.xyxy[3]) / math::max(t.x[3], 1e-3f);
        if (!math::finite(foot) || !math::finite(under)) return numeric_failure();
        if (under > 0) {
            const float front = config_.front_cost_weight * under;
            if (!math::finite(front)) return numeric_failure();
            penalty += front;
            ++diagnostics_.front_positive_calls;
        }
    }
    if (!math::finite(penalty)) return numeric_failure();
    if (penalty == 0)
        return math::association_cost(overlap, t.x[3], height, tracker_.assoc.lambda, tracker_.assoc.stability_cost_w);
    const float denominator = math::height_denominator(t.x[3], height);
    if (!math::finite(denominator)) return numeric_failure();
    const float reward = (tracker_.assoc.stability_cost_w / math::max(tracker_.assoc.lambda, 1.0f)) / denominator;
    const float exponent = -(penalty - reward);
    if (!math::finite(reward) || !math::finite(exponent)) return numeric_failure();
    const float result = 1.0f - overlap * math::det_expf(exponent);
    return math::finite(result) ? math::clamp(result, 0.0f, 1.0f) : numeric_failure();
}

} // namespace saccade

#pragma once

#include "saccade/tracking.hpp"

#include <cstdint>
#include <map>

namespace saccade {

// Local CPU replay experiment; these switches do not change v1-tracker/1 or
// the native CUDA implementation. Legacy spatial ablation gates are omitted.
struct OcclusionExperimentConfig {
    bool oao_enabled = false;
    bool front_enabled = false;
    float oao_tau = 0.50f;
    std::uint32_t oao_ramp_frames = 25;
    float front_iou_thresh = 0.45f;
    float front_foot_gap = 0.15f;
    std::uint32_t front_ttl = 4;
    float front_cost_weight = 0.50f;
};

// invalid_argument on non-finite/negative weights or gap, IoU outside [0,1],
// zero duration-ramp frames, or zero front TTL.
void validate_occlusion_experiment_config(const OcclusionExperimentConfig& config);

struct OcclusionTrackState {
    float max_overlap = 0;
    float oao_coefficient = 0;
    std::uint32_t overlap_duration = 0;
    std::uint32_t front_ttl = 0;
    std::uint64_t partner_uid = 0; // zero means no positive-overlap partner
};

struct OcclusionDiagnostics {
    std::uint64_t frames = 0;
    std::uint64_t overlapped_track_frames = 0;
    std::uint64_t front_latched_track_frames = 0;
    std::uint64_t cost_calls = 0;
    std::uint64_t oao_positive_calls = 0;
    std::uint64_t front_positive_calls = 0;
};

class OcclusionExperiment {
public:
    explicit OcclusionExperiment(const TrackerConfig& tracker,
                                 const OcclusionExperimentConfig& experiment = {});

    // Advance once after predict/GMC, before association. Every surviving
    // track participates, including lost and tentative tracks. Equal maximum
    // overlaps select the lowest UID. Missing UIDs lose all state; output_id
    // changes caused by bridges do not transfer state to another UID.
    // Repeating the current epoch/frame is a no-op. Later frames in one epoch
    // must be consecutive. A new epoch starts with empty per-track state.
    // Numeric/protocol failures leave state and frame counters unchanged.
    Error prepare(const PredictedState& predicted);

    // Suitable for AssociationCostHook. NUMERIC is signalled by a NaN return,
    // which associate_and_update rejects. No-penalty pairs use the exact stock
    // association_cost expression.
    float cost(const Track& track, const BatchDetection& detection, float overlap);

    const OcclusionDiagnostics& diagnostics() const { return diagnostics_; }
    const OcclusionTrackState* state(std::uint64_t uid) const;

private:
    TrackerConfig tracker_;
    OcclusionExperimentConfig config_;
    std::map<std::uint64_t, OcclusionTrackState> states_;
    OcclusionDiagnostics diagnostics_;
    bool prepared_ = false;
    Epoch epoch_ = 0;
    FrameId frame_ = no_frame;
};

} // namespace saccade

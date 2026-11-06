#pragma once

#include "saccade/motion.hpp"
#include "saccade/postprocess.hpp"
#include "saccade/protocol.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// F08a/F04b/F07/F08b contract, profile v1-tracker/1 (docs/TRACKING.md §6-§9).
// Line B owns the CPU reference and may extend this header within its scope.
// These are logical host types; the M2 CUDA layout is defined separately and
// compared through these values and the encoded ResultPacket.
namespace saccade {

class TrackingEventObserver;

inline constexpr const char* kTrackerProfile = "v1-tracker/1";
inline constexpr std::uint32_t kTrackCapacity = 512;
inline constexpr std::uint32_t kBatchCapacity = kPostprocessCapacity;
inline constexpr std::uint32_t kBridgeHistory = 4;

struct TrackerConfig {
    struct Kalman {
        float r_scale = 3.5f;
        float std_position = 1.0f / 20.0f;
        float std_velocity = 1.0f / 160.0f;
    } kalman;
    struct Batch {
        float track_thresh = 0.05f;
    } batch;
    struct Continuation {
        bool enabled = true;
        float min_score = 0.10f, prior_iou = 0.30f, dedup_iou = 0.70f;
        std::uint32_t max_candidates = 50;
    } continuation;
    struct Association {
        float high_thresh = 0.45f, mid_thresh = 0.10f;
        float match_thresh = 0.50f, stage2_match_thresh = 0.50f;
        float lambda = 10.0f, stability_cost_w = 0.20f;
        std::uint32_t top_k = 3;
        float auction_eps = 0.01f, stability_bid_w = 0.10f;
    } assoc;
    struct Lifecycle {
        float new_track_thresh = 0.28f;
        std::uint32_t confirm_streak = 3;
        float confirm_score = 0.50f;
        std::uint32_t track_buffer = 30;
    } life;
    struct Bridge {
        bool enabled = true;
        std::uint32_t trigger_streak = 4, min_lost = 2;
        float max_distance = 0.4f, height_ratio_lo = 0.6f, height_ratio_hi = 1.7f;
        float margin = 0.05f, speed_norm = 0.12f;
    } bridge;
};
// invalid_argument on: any non-finite float; score/IoU/cost thresholds outside
// [0,1]; not track_thresh <= mid_thresh <= high_thresh; lambda, r_scale,
// std_*, eps or speed_norm <= 0; negative weights/margin/max_distance;
// top_k outside [1,8];
// confirm_streak or track_buffer < 1; trigger_streak < kBridgeHistory;
// min_lost < 1; height ratios not 0 < lo < hi; max_candidates > kBatchCapacity.
void validate_tracker_config(const TrackerConfig& config);
// One "key=value\n" line per field in declaration order; floats as C99 %a,
// booleans as 0/1. sha256 of this text is the tracker config hash.
std::string canonical_tracker_config(const TrackerConfig& config);

enum class TrackLifecycle : std::uint32_t { tentative = 1, confirmed = 2 };
struct BridgeSample {
    float cx = 0, cy = 0, h = 0;
};
struct Track {
    std::uint64_t uid = 0;       // never reused within an epoch
    std::uint64_t output_id = 0; // uid at birth; bridge may replace it
    TrackLifecycle lifecycle = TrackLifecycle::tentative;
    std::array<float, 8> x{};    // cx, cy, a=w/h, h, vx, vy, va, vh
    std::array<float, 64> P{};   // row-major 8x8 covariance
    std::uint32_t age = 0;       // frames since the last match
    std::uint32_t hit_streak = 0;
    float score_sum = 0;
    std::uint32_t class_id = 0;
    float observed_score = 0, association_score = 0; // of the last match
    FrameId birth_frame = no_frame, last_observed_frame = no_frame;
    std::array<BridgeSample, kBridgeHistory> history{}; // oldest first
    std::uint32_t history_count = 0;
    float h_ema = 0;
};
struct TrackState {
    Epoch epoch = 0;
    FrameId committed_frame = no_frame;
    std::uint64_t next_uid = 1;
    std::vector<Track> tracks; // ascending uid, size <= kTrackCapacity
};
TrackState initial_track_state(Epoch epoch);

// F08a output: this frame's working version, never committed by itself.
struct PredictedPrior {
    std::uint64_t uid = 0;
    std::array<float, 4> xyxy{};
    std::uint32_t class_id = 0;
};
struct PredictedState {
    Epoch epoch = 0;
    FrameId source_committed_frame = no_frame, frame = no_frame;
    MotionContext motion{};
    std::uint64_t next_uid = 1;
    std::vector<Track> tracks;          // survivors, ascending uid
    std::vector<PredictedPrior> priors; // confirmed survivors, ascending uid
    Error error = Error::none;          // NUMERIC only
};
// GMC control, one predict, age and removal (docs/TRACKING.md §6). frame is the
// target frame k; identity consistency is F07's job, not checked here.
PredictedState predict_tracks(const TrackState& committed, const MotionContext& motion, FrameId frame,
                              const TrackerConfig& config, TrackingEventObserver* observer = nullptr,
                              std::uint64_t source = 1);

// F04b output, owned by M after the candidate slot is consumed.
struct BatchDetection {
    std::array<float, 4> xyxy{};
    float observed_score = 0, association_score = 0;
    std::uint32_t class_id = 0, candidate_id = 0;
    bool can_continue = false, can_birth = false;
};
struct DetectionBatch {
    Epoch epoch = 0;
    FrameId frame = no_frame;
    std::vector<BatchDetection> rows; // ascending candidate_id
    Error error = Error::none;        // Header errors or NUMERIC in candidate/IoU values.
};
// docs/TRACKING.md §7. On a header error, rows is empty.
DetectionBatch assemble_detection_batch(const DetectionCandidates& candidates, const PredictedState& predicted,
                                        const TrackerConfig& config);

// F07 + F08b (docs/TRACKING.md §8-§9). On any error, next is unspecified and
// must not be committed; rows is empty.
struct UpdateResult {
    TrackState next;
    std::vector<TrackRow> rows; // ResultPacket rows, ascending uid
    Error error = Error::none;
};
UpdateResult associate_and_update(const PredictedState& predicted, const DetectionBatch& batch,
                                  const MotionContext& motion, const TrackerConfig& config,
                                  TrackingEventObserver* observer = nullptr, std::uint64_t source = 1);

// Serial CPU session: F08a -> F04b -> F07/F08b -> commit/packet per frame with
// the RESULT_PACKET status model (first failure FAILED, then
// SKIPPED_AFTER_FAILURE; committed state never changes after a failure).
// Frames are implicit and consecutive from 0.
class TrackerReference {
public:
    explicit TrackerReference(Epoch epoch, const TrackerConfig& config = {},
                              TrackingEventObserver* observer = nullptr, std::uint64_t source = 1);
    ResultPacket step(const DetectionCandidates& candidates, const MotionContext& motion);
    const TrackState& committed() const { return committed_; }
    bool failed() const { return fail_frame_ != no_frame; }

private:
    TrackerConfig config_;
    TrackState committed_;
    FrameId next_frame_ = 0;
    FrameId fail_frame_ = no_frame;
    Error fail_code_ = Error::none;
    TrackingEventObserver* observer_ = nullptr; // optional read-only decision snapshots; caller owns lifetime
    std::uint64_t source_ = 1;
};

} // namespace saccade

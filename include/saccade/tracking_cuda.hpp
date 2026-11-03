#pragma once

#include "saccade/tracking.hpp"
#include <cuda_runtime_api.h>
#include <type_traits>

namespace saccade {

struct DeviceBridgeSample { float cx, cy, h; };
struct DeviceMotionContext {
    Epoch epoch;
    FrameId frame, reference_frame;
    MotionStatus status;
    std::int32_t peak_x, peak_y;
    float pcr, confidence, tx, ty;
};
struct DeviceTrack {
    std::uint64_t uid, output_id;
    TrackLifecycle lifecycle;
    float x[8], P[64];
    std::uint32_t age, hit_streak;
    float score_sum;
    std::uint32_t class_id;
    float observed_score, association_score;
    FrameId birth_frame, last_observed_frame;
    DeviceBridgeSample history[kBridgeHistory];
    std::uint32_t history_count;
    float h_ema;
};
struct DeviceTrackState {
    Epoch epoch;
    FrameId committed_frame;
    std::uint64_t next_uid;
    std::uint32_t count;
    DeviceTrack tracks[kTrackCapacity];
};
struct DevicePredictedPrior {
    std::uint64_t uid;
    float xyxy[4];
    std::uint32_t class_id;
};
struct DevicePredicted {
    Epoch epoch;
    FrameId source_committed_frame, frame;
    DeviceMotionContext motion;
    std::uint64_t next_uid;
    std::uint32_t count, prior_count;
    Error error;
    DeviceTrack tracks[kTrackCapacity];
    DevicePredictedPrior priors[kTrackCapacity];
};
struct DeviceBatchDetection {
    float xyxy[4], observed_score, association_score;
    std::uint32_t class_id, candidate_id;
    bool can_continue, can_birth;
};
struct DeviceBatch {
    Epoch epoch;
    FrameId frame;
    std::uint32_t count;
    Error error;
    DeviceBatchDetection rows[kBatchCapacity];
};
struct DeviceTrackRow {
    std::uint64_t uid, output_id;
    std::uint32_t class_id, flags;
    float xyxy[4], observed_score, association_score;
};
struct DeviceTrackingRows {
    std::uint32_t count;
    Error error;
    DeviceTrackRow rows[packet_capacity];
};
struct TrackingCudaProposal { int detection; float p, bid; };
// Borrowed working storage: exclusively owned by one in-flight stage.
struct TrackingCudaScratch {
    DeviceTrack tracks[kTrackCapacity];
    int status[kTrackCapacity], matched[kTrackCapacity], claims[kTrackCapacity];
    bool removed[kTrackCapacity], used[kBatchCapacity];
    TrackingCudaProposal proposals[kTrackCapacity];
    int order[kBatchCapacity];
    DeviceBatchDetection batch_rows[kBatchCapacity];
    Error error;
};
inline constexpr std::size_t kTrackingCudaScratchBytes = sizeof(TrackingCudaScratch);
static_assert(std::is_trivial_v<DeviceTrack> && std::is_standard_layout_v<DeviceTrack>);
static_assert(std::is_trivial_v<DeviceTrackState> && std::is_trivial_v<DevicePredicted>);
static_assert(std::is_trivial_v<DeviceBatch> && std::is_trivial_v<DeviceTrackingRows>);
static_assert(std::is_trivial_v<TrackingCudaScratch>);

// Host-only value conversions. Logical inputs must fit their fixed capacities.
DeviceTrack to_device(const Track& value);
Track to_host(const DeviceTrack& value);
DeviceTrackState to_device(const TrackState& value);
TrackState to_host(const DeviceTrackState& value);
DeviceMotionContext to_device(const MotionContext& value);
MotionContext to_host(const DeviceMotionContext& value);
DevicePredicted to_device(const PredictedState& value);
PredictedState to_host(const DevicePredicted& value);
DeviceBatch to_device(const DetectionBatch& value);
DetectionBatch to_host(const DeviceBatch& value);
UpdateResult to_host(const DeviceTrackState& next, const DeviceTrackingRows& rows);

// All pointers refer to device memory, including frame and config. Inputs,
// outputs and scratch do not alias and remain alive until stream completion.
// Config is validated by the caller once. Counts obey the fixed capacities.
// A null run_stage always executes; a false device flag leaves storage untouched.
// Enqueue only: no allocation, synchronization or readback. An explicit
// non-default stream is required (invalid_argument); launch failures throw
// runtime_error. Outputs follow the CPU reference, including partial predict
// on NUMERIC and F04b's independent processing of a failed prediction.
// An update with rows.error != none must never be committed; next is unspecified.
void enqueue_predict_tracks(const DeviceTrackState* committed, const DeviceMotionContext* motion,
                            const FrameId* frame, const TrackerConfig* config, DevicePredicted* output,
                            TrackingCudaScratch* scratch, cudaStream_t stream, const bool* run_stage = nullptr);
void enqueue_assemble_detection_batch(const DetectionCandidates* candidates, const DevicePredicted* predicted,
                                     const TrackerConfig* config, DeviceBatch* output,
                                     TrackingCudaScratch* scratch, cudaStream_t stream, const bool* run_stage = nullptr);
void enqueue_associate_and_update(const DevicePredicted* predicted, const DeviceBatch* batch,
                                 const DeviceMotionContext* motion, const TrackerConfig* config,
                                 DeviceTrackState* next, DeviceTrackingRows* rows,
                                 TrackingCudaScratch* scratch, cudaStream_t stream, const bool* run_stage = nullptr);

} // namespace saccade

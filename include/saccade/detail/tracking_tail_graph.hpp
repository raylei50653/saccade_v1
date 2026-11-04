#pragma once

#include "saccade/tracking_cuda.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace saccade::detail {
struct TrackingSessionControl;

// Every address is stable for the life of the graph. Frame identity, the
// working state slot and failure gates are read from control on the device.
struct TrackingTailPointers {
    TrackingSessionControl* control;
    DeviceTrackState* tracks;
    DevicePredicted* predicted;
    DeviceBatch* batch;
    DeviceTrackingRows* rows;
    TrackingCudaScratch* scratch;
    TrackerConfig* tracker_config;
    DeviceMotionContext* tracker_motion;
    std::uint8_t* packet;
};

// The caller first waits for the selected candidate's ready event. Release
// consumed only after this work, then copy packet to the leased host slot.
void enqueue_tracking_tail(const TrackingTailPointers&, const DetectionCandidates*, cudaStream_t);

class TrackingTailGraph {
public:
    // Capture only: no frame is executed or committed by construction.
    TrackingTailGraph(const TrackingTailPointers&, std::span<DetectionCandidates* const>, cudaStream_t);
    ~TrackingTailGraph();
    TrackingTailGraph(const TrackingTailGraph&) = delete;
    TrackingTailGraph& operator=(const TrackingTailGraph&) = delete;
    void launch(std::size_t candidate_slot, cudaStream_t);
    void close(); // Caller drains every launch before checked destruction.
    std::size_t slots() const noexcept { return executables_.size(); }
    std::size_t kernel_nodes() const noexcept { return kernel_nodes_; }
private:
    std::vector<cudaGraphExec_t> executables_;
    int device_ = -1;
    std::size_t kernel_nodes_ = 0;
};
} // namespace saccade::detail

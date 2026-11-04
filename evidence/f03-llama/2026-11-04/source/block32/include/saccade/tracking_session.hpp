#pragma once

#include "saccade/model_profile.hpp"

#include "saccade/motion_cuda.hpp"
#include "saccade/preprocess.hpp"
#include "saccade/tracking_cuda.hpp"

#include <memory>
#include <span>
#include <string>

namespace saccade {
enum class TrackingSessionMode { Serial, Pipelined };
enum class TrackingSessionState { Running, Finished, Cancelled, Failed, Closed };
struct TrackingSessionConfig {
    Epoch epoch = 1;
    int device = 0;
    int original_width = 1920, original_height = 1080;
    unsigned candidate_slots = 2, motion_slots = 2, result_slots = 4;
    PreprocessConfig preprocess{}; // output size is selected by the model profile
    ModelProfile profile = kDefaultModelProfile;
    MotionConfig motion{};
    TrackerConfig tracker{};
    TrackingSessionMode mode = TrackingSessionMode::Pipelined;
    bool cuda_graph = true; // F03/producer/F04 and tracking tail; nvJPEG stays outside
    std::uint64_t perturb_seed = 0;
};
struct TrackingSessionResult {
    PacketBytes bytes{};
    ResultPacket packet;
};
namespace detail {
// Slot guards select TS from device CTL without changing the M2 pointer API.
struct TrackingSessionControl {
    Control value;
    bool slot_run[2];
};
struct TrackingSessionDiagnostic {
    Control control;
    DeviceTrackState tracks[2];
    DetectionCandidates candidates;
    MotionContext motion;
};
void tracking_session_load_kernels();
void tracking_session_initialize(TrackingSessionControl*, DeviceTrackState*, Epoch, cudaStream_t);
void tracking_session_begin(TrackingSessionControl*, Epoch, FrameId, cudaStream_t);
void tracking_session_stage(TrackingSessionControl*, const Error*, const MotionContext*, DeviceMotionContext*, cudaStream_t);
void tracking_session_candidates(TrackingSessionControl*, const DetectionCandidates*, cudaStream_t);
void tracking_session_commit(TrackingSessionControl*, cudaStream_t);
void tracking_session_packet(const Control*, const DeviceTrackingRows*, std::uint8_t*, cudaStream_t);
void tracking_session_perturb(std::uint64_t seed, FrameId frame, unsigned boundary, cudaStream_t);
void tracking_session_spin(unsigned long long cycles, cudaStream_t);
void tracking_session_producer_flag(PostprocessProducerStatus*, std::uint32_t, cudaStream_t);
}

// One host submitter and one epoch. All storage and library warmups precede
// frame 0. MI/CS use ready/consumed leases; HP is leased until receive().
class TrackingSession {
public:
    explicit TrackingSession(const std::string& engine_path, const TrackingSessionConfig& config = {});
    ~TrackingSession();
    TrackingSession(const TrackingSession&) = delete;
    TrackingSession& operator=(const TrackingSession&) = delete;
    // Contiguous zero-based frame. A full result ring throws length_error
    // before submission and is recoverable by receive(). Other errors latch Failed.
    void enqueue(std::span<const std::uint8_t> jpeg, FrameId frame);
    TrackingSessionResult receive();
    std::size_t pending() const;
    // Drain both streams, retaining outstanding results for ordered receive().
    void finish();
    void cancel();
    void close(); // checked release; repeatable; destructor is best effort
    TrackingSessionState state() const;
    std::string identity() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend struct TrackingSessionTestAccess;
    void enqueue_for_test(std::span<const std::uint8_t>, FrameId, std::uint32_t, bool = false);
    // Test-only snapshot after draining both current stream tails.
    detail::TrackingSessionDiagnostic diagnostic_for_test() const;
};
} // namespace saccade

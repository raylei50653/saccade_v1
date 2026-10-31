#pragma once

#include "saccade/postprocess.hpp"
#include "saccade/preprocess.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>

namespace saccade {
inline constexpr const char* kDetectionSessionPolicy = "f01-f04-session/1";
inline constexpr std::uint64_t kDetectionSessionNoFrame = std::numeric_limits<std::uint64_t>::max();
enum class DetectionSessionStatus : std::uint32_t { Ok, Failed, SkippedAfterFailure };
enum class DetectionSessionState { Running, Finished, Cancelled, Failed, Closed };
struct DetectionSessionConfig {
    std::uint64_t epoch = 1;
    int device = 0;
    unsigned candidate_slots = 2; // [1,16], device ready/consumed leases
    unsigned result_slots = 4;    // [1,64], pinned host receive leases
    std::uint32_t output_capacity = kPostprocessCapacity;
    PreprocessConfig preprocess{}; // output must be 640x640
};
// Bounded detection delivery ABI, not the tracking ResultPacket. Every byte,
// including reserved and unused candidate rows, is initialized each frame.
struct DetectionSessionResult {
    std::uint64_t epoch, frame, fail_frame, committed_frame;
    DetectionSessionStatus status;
    std::uint32_t fail_code;
    std::uint64_t reserved;
    DetectionCandidates candidates;
};
static_assert(sizeof(DetectionSessionResult) == 48 + sizeof(DetectionCandidates));

// One host submitter, one epoch, one sequence. Owns explicit D/M streams and
// all resources; JPEG staging remains the conservative NativePreprocessor's
// single staging path. D performs F01/F02 -> F03 -> aggregate -> F04; M reads
// candidates after ready, latches first failure, copies full results and records
// consumed before D may reuse a candidate slot. At most result_slots results
// may be outstanding. No intermediate per-frame host count/status readback.
class DetectionSession {
public:
    explicit DetectionSession(const std::string& engine_path, const DetectionSessionConfig& config = {});
    ~DetectionSession();
    DetectionSession(const DetectionSession&) = delete;
    DetectionSession& operator=(const DetectionSession&) = delete;

    // Frame is zero-based, contiguous. JPEG bytes are copied before return.
    // Full host result ring throws length_error BEFORE submission, recoverably:
    // receive() frees a slot. Other submission errors latch sequence Failed;
    // finish/cancel/close still drain both streams, including a D-only tail.
    void enqueue(std::span<const std::uint8_t> jpeg, std::uint64_t frame,
                 int expected_width = 0, int expected_height = 0);
    // Ordered, blocking receipt; returns a value so the pinned ring can be
    // reused immediately. Failed/skipped results retain first failure identity
    // and last successful frame, with zero candidate rows. Empty throws.
    DetectionSessionResult receive();
    std::size_t pending() const;
    // Stop submitting, drain BOTH streams, retain outstanding results for
    // receive. Finished only on normal finish with no observed device/host
    // failure; cancellation is distinct from EOF. Repeated calls are safe.
    void finish();
    void cancel();
    // Checked drain and all independent releases; consumes ownership even
    // when a release reports failure. Repeated close is safe. Destructor is
    // best effort; use close for a checked lifecycle boundary.
    void close();
    DetectionSessionState state() const;
    std::string identity() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend struct DetectionSessionTestAccess;
    void enqueue_for_test(std::span<const std::uint8_t>, std::uint64_t, int, int, std::uint32_t, bool, bool = false);
    cudaStream_t consumer_stream_for_test() const;
    cudaStream_t detect_stream_for_test() const;
};
} // namespace saccade

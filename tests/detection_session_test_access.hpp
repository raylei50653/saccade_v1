#pragma once
#include "saccade/detection_session.hpp"
namespace saccade {
// Directed test seam. Normal callers use only the public session API.
struct DetectionSessionTestAccess {
    static void enqueue(DetectionSession& session, std::span<const std::uint8_t> jpeg,
                        std::uint64_t frame, std::uint32_t producer_flags, bool corrupt_identity = false) {
        session.enqueue_for_test(jpeg, frame, 0, 0, producer_flags, corrupt_identity);
    }
    static void enqueue_d_only(DetectionSession& session, std::span<const std::uint8_t> jpeg, std::uint64_t frame) {
        session.enqueue_for_test(jpeg, frame, 0, 0, 0, false, true);
    }
    static cudaStream_t detect_stream(const DetectionSession& session) {
        return session.detect_stream_for_test();
    }
    static cudaStream_t consumer_stream(const DetectionSession& session) {
        return session.consumer_stream_for_test();
    }
};
}

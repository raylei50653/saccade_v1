#pragma once
#include "saccade/detection_session.hpp"
namespace saccade::detail {
struct DetectionSessionControl {
    std::uint64_t fail_frame, committed_frame;
    std::uint32_t fail_code;
    std::uint32_t reserved;
};
void detection_session_load_kernels();
void detection_session_initialize(DetectionSessionControl*, cudaStream_t);
void detection_session_finalize(const DetectionCandidates*, DetectionSessionResult*, DetectionSessionControl*,
                                std::uint64_t epoch, std::uint64_t frame, std::uint32_t capacity, cudaStream_t);
void detection_session_test_override(PostprocessProducerStatus*, std::uint32_t, cudaStream_t);
void detection_session_test_corrupt(DetectionCandidates*, cudaStream_t);
}

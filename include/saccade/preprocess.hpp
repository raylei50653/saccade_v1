#pragma once

#include "saccade/image_mapping.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace saccade {

struct PreprocessConfig {
    int max_width = 1920, max_height = 1080;
    int output_width = 640, output_height = 640;
    std::size_t max_jpeg_bytes = 16 * 1024 * 1024;
};

struct PreprocessedFrame {
    std::uint64_t frame;
    ImageMapping mapping;
    const std::uint8_t* rgb; // contiguous planar RGB, original resolution
    float* images;          // contiguous [1,3,H,W] FP32, t01-input/1
};

// F01/F02, single host submitter and one borrowed, non-default detect stream.
// HARDWARE batch-size-1 decode only: baseline RGB YCbCr 4:2:0 JPEGs; unsupported
// sources fail explicitly. All device buffers are preallocated. No Python,
// CPU decode, fallback decoder, graph capture or multiframe host overlap.
//
// enqueue owns a copy of the compressed bytes. Before reusing host staging it
// waits for the previous F02 event. RAW/images are borrowed until the next
// enqueue; downstream F05/F03 must be queued on the same stream before that
// call. Stream ordering protects downstream reads, including those queued
// after the event. The stream must outlive this object. Any enqueue/wait error
// latches failure; create a fresh session to start another sequence.
class NativePreprocessor {
public:
    NativePreprocessor(cudaStream_t detect_stream, const PreprocessConfig& config = {});
    ~NativePreprocessor();
    NativePreprocessor(const NativePreprocessor&) = delete;
    NativePreprocessor& operator=(const NativePreprocessor&) = delete;

    // frame is zero-based and contiguous. Optional expected dimensions must
    // both be positive or both zero; a metadata mismatch is a sequence error.
    PreprocessedFrame enqueue(std::span<const std::uint8_t> jpeg, std::uint64_t frame,
                              int expected_width = 0, int expected_height = 0);
    void wait(); // frame boundary for consumers that need host observation
    std::string identity() const;
    void close(); // checked drain and release; destructor is best effort

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace saccade

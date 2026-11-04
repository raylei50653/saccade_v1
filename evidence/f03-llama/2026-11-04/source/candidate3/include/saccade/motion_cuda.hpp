#pragma once

#include "saccade/motion.hpp"
#include "saccade/protocol.hpp"

#include <cuda_runtime_api.h>
#include <memory>

namespace saccade {

// F05 primitive. All pointers are device memory; wx/wy are uploaded host-built
// hanning_window tables. The caller orders input/table/output lifetimes on D.
void motion_image_cuda(const std::uint8_t* planar_rgb, float* image, const MotionImageKey& key,
                       const float* wx, const float* wy, cudaStream_t detect, const bool* run_stage = nullptr);

// One host submitter, sequence and epoch. Submit D(k), M(k), commit(k), D(k+1).
// Borrowed non-default D/M streams must remain fixed and outlive this object.
// All storage/plans and kernel loading are prepared in the constructor.
// Non-finite F06 arithmetic is reported by a non-finite pcr (NUMERIC); consumers
// must check it before commit. Without CTL, commit selection is host-owned.
class MotionCuda {
public:
    MotionCuda(const MotionImageKey& key, const MotionConfig& config, std::uint64_t epoch, unsigned image_slots = 2);
    ~MotionCuda();
    MotionCuda(const MotionCuda&) = delete;
    MotionCuda& operator=(const MotionCuda&) = delete;

    void enqueue_image(cudaStream_t detect, const std::uint8_t* device_planar_rgb, std::int64_t frame);
    // With CTL, spectrum selection and reference identity are device-owned.
    void enqueue_estimate(cudaStream_t main, std::int64_t frame, MotionContext* device_out,
                          const bool* run_stage = nullptr, Control* control = nullptr);
    void enqueue_commit(cudaStream_t main, std::int64_t frame);
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace saccade

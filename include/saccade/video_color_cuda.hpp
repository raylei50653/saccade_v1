#pragma once

#include "saccade/device_video_frame.hpp"

#include <cuda_runtime_api.h>
#include <cstddef>

namespace saccade {
// NV12/P010 -> dense planar RGB8, no host pixel transfer or allocation.
// Chroma is sampled at (x/2,y/2). P010 uses the high ten bits of each word.
// BT.601/709/2020-NCL luma coefficients, full/limited code ranges, and nearest
// even RGB8 rounding after clamping define device-video-color/1.
// Validates device/context/geometry/pitches/disjoint pointers before waiting
// on input.ready_event on the explicit nondefault stream and launching.
// Caller retains input.owner through completion of all work on stream.
void video_color_cuda(const DeviceVideoFrame& input, std::uint8_t* output,
                      std::size_t output_capacity, cudaStream_t stream);
void video_color_cuda_load_kernels();
} // namespace saccade

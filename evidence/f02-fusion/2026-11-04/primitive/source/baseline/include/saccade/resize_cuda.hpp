#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace saccade {

// T01 input contract: batch one, dense planar RGB, float normalization before
// half-pixel antialiased bilinear stretch. No clamp, quantization or letterbox.
struct ResizeShape {
    int input_width = 0;
    int input_height = 0;
    int output_width = 640;
    int output_height = 640;
};

std::size_t resize_input_bytes(const ResizeShape& shape);
std::size_t resize_output_elements(const ResizeShape& shape);
// Horizontal intermediate in CHW order: 3 * input_height * output_width.
std::size_t resize_workspace_elements(const ResizeShape& shape);

// Caller owns all allocations and keeps them alive through completion on stream.
// Requires disjoint device allocations on the current device and an explicit
// non-default stream. Capacities count input bytes and FP32 elements respectively.
// Validates all arguments before enqueueing; throws invalid_argument for an
// invalid argument and runtime_error for a CUDA error. No allocation, host
// transfer, synchronization or default-stream work in the enqueue path.
void resize_rgb_chw_cuda(const ResizeShape& shape,
                         const std::uint8_t* input, std::size_t input_bytes,
                         float* output, std::size_t output_elements,
                         float* workspace, std::size_t workspace_elements,
                         cudaStream_t stream);

// Call once before frame work; kernel loading may synchronize under lazy loading.
std::string resize_cuda_load_kernels();
std::string resize_cuda_identity();

} // namespace saccade

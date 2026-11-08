#include "saccade/tracking_math.hpp"
#include <cuda_runtime.h>

namespace {
__global__ void multipliers(const float* scores, float* output, unsigned count, float s0) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = saccade::tracking_math::nsa_v2_multiplier(scores[i], s0);
}
}
cudaError_t launch_nsa_v2_test(const float* scores, float* output, unsigned count, float s0, cudaStream_t stream) {
    multipliers<<<(count + 255) / 256, 256, 0, stream>>>(scores, output, count, s0);
    return cudaGetLastError();
}

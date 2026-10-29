// Positive controls: each mode plants one defect that its compute-sanitizer
// tool must report, so a clean tool run on the scan tests is not mistaken for
// a tool that was never active. Usage: sanitizer-control memcheck|initcheck|racecheck|synccheck|shuffle-mask
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void out_of_bounds(float* p) { p[threadIdx.x + 32] = 1.0f; }              // 32 floats allocated
__global__ void uninitialized_read(const float* in, float* out) { out[threadIdx.x] = in[threadIdx.x] + 1.0f; }
__global__ void shared_race(float* out) {
    volatile __shared__ float cells[64];
    cells[threadIdx.x] = static_cast<float>(threadIdx.x);
    out[threadIdx.x] = cells[(threadIdx.x + 32) % 64]; // the other warp's write, no barrier
}
// Not a control: records that synccheck does not check __shfl_sync masks
// (thread 1 names a mask that excludes itself; expected to pass unreported).
__global__ void bad_shuffle_mask(float* out) {
    if (threadIdx.x < 2) out[threadIdx.x] = __shfl_sync(0x1u, static_cast<float>(threadIdx.x), 0);
}
__global__ void bad_warp_barrier(float* out) {
    // Thread 1 names a __syncwarp mask that excludes itself.
    if (threadIdx.x < 2) __syncwarp(0x1u);
    out[threadIdx.x] = 1.0f;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: sanitizer-control memcheck|initcheck|racecheck|synccheck|shuffle-mask\n"); return 2; }
    float *a = nullptr, *b = nullptr;
    if (cudaMalloc(&a, 32 * sizeof(float)) != cudaSuccess || cudaMalloc(&b, 32 * sizeof(float)) != cudaSuccess) return 2;
    if (cudaMemset(b, 0, 32 * sizeof(float)) != cudaSuccess) return 2;
    if (!std::strcmp(argv[1], "memcheck")) out_of_bounds<<<1, 32>>>(a);
    else if (!std::strcmp(argv[1], "initcheck")) uninitialized_read<<<1, 32>>>(a, b);
    else if (!std::strcmp(argv[1], "racecheck")) shared_race<<<1, 64>>>(b);
    else if (!std::strcmp(argv[1], "synccheck")) bad_warp_barrier<<<1, 32>>>(b);
    else if (!std::strcmp(argv[1], "shuffle-mask")) bad_shuffle_mask<<<1, 32>>>(b);
    else return 2;
    const cudaError_t result = cudaDeviceSynchronize();
    std::printf("control %s: cudaDeviceSynchronize -> %s\n", argv[1], cudaGetErrorString(result));
    cudaFree(a);
    cudaFree(b);
    return 0;
}

// Test-only planted failures. Each sanitizer must find its own control; no
// fault can enter production F04 or the default CTest suite.
#include <cuda_runtime.h>
#include <iostream>
#include <string>

__global__ void invalid_write(float* output) { output[1] = 1.0f; }
__global__ void uninitialized_read(const float* input, float* output) { output[0] = input[0]; }
__global__ void shared_race(float* output) {
    __shared__ volatile float shared[64];
    shared[threadIdx.x] = static_cast<float>(threadIdx.x);
    output[threadIdx.x] = shared[(threadIdx.x + 32) % 64];
}
__global__ void divergent_barrier(float* output) {
    // Thread 1 participates with a mask that excludes itself.
    if (threadIdx.x < 2) __syncwarp(0x1u);
    output[threadIdx.x] = 1.0f;
}
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    float* input = nullptr; float* output = nullptr; cudaStream_t stream = nullptr;
    if (cudaSetDevice(0) != cudaSuccess || cudaMalloc(&input, 4) != cudaSuccess ||
        cudaMalloc(&output, 64 * sizeof(float)) != cudaSuccess ||
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) return 2;
    const std::string mode = argv[1];
    if (mode == "memcheck") invalid_write<<<1, 1, 0, stream>>>(input);
    else if (mode == "initcheck") uninitialized_read<<<1, 1, 0, stream>>>(input, output);
    else if (mode == "racecheck") shared_race<<<1, 64, 0, stream>>>(output);
    else if (mode == "synccheck") divergent_barrier<<<1, 32, 0, stream>>>(output);
    else return 2;
    const auto launch = cudaGetLastError(), completion = cudaStreamSynchronize(stream);
    const auto released_output = cudaFree(output), released_input = cudaFree(input), destroyed_stream = cudaStreamDestroy(stream);
    std::cout << "control=" << mode << " launch=" << static_cast<int>(launch)
              << " completion=" << static_cast<int>(completion) << '\n';
    return launch == cudaSuccess && completion == cudaSuccess && released_output == cudaSuccess &&
           released_input == cudaSuccess && destroyed_stream == cudaSuccess ? 0 : 1;
}

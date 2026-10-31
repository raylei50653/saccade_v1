#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// Test-only asynchronous device fault for the S03 failure channel 3: an
// illegal instruction enqueued after the engine on the detect stream. The
// CUDA context is unusable afterwards, so it runs in a process of its own.
namespace {
__global__ void trap_kernel() { __trap(); }
} // namespace

void scan_trt_test_load_trap() {
    cudaFuncAttributes attributes{};
    const cudaError_t e = cudaFuncGetAttributes(&attributes, trap_kernel);
    if (e != cudaSuccess) throw std::runtime_error(std::string("load trap kernel: ") + cudaGetErrorString(e));
}

void scan_trt_test_trap(cudaStream_t s) {
    trap_kernel<<<1, 1, 0, s>>>();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("trap launch: ") + cudaGetErrorString(e));
}

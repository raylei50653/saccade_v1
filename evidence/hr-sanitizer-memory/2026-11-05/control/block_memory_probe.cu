// Diagnostic-only CUDA block-count reproducer, no project CUDA libraries.
#include <cuda_runtime.h>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr unsigned threads = 256;
__global__ void global_store(unsigned* out) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = idx;
}
__global__ void shared_neighbor(unsigned* out) {
    __shared__ unsigned tmp[threads];
    const unsigned tid = threadIdx.x;
    const unsigned idx = blockIdx.x * blockDim.x + tid;
    tmp[tid] = idx;
    __syncthreads();
    out[idx] = tmp[(tid + 1) % threads];
}
void cu(cudaError_t e, const char* call) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(e));
}
unsigned number(std::string_view text) {
    unsigned n = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), n);
    if (error != std::errc{} || end != text.data() + text.size() || !n)
        throw std::invalid_argument("expected positive unsigned integer");
    return n;
}
}
int main(int argc, char** argv) {
    unsigned* out = nullptr;
    try {
        unsigned blocks = 1, repeats = 1;
        std::string kind = "global";
        for (int i = 1; i < argc; ++i) {
            const std::string_view key = argv[i];
            if (i + 1 >= argc)
                throw std::invalid_argument("usage: block_memory_probe --kind global|shared --blocks N [--threads 256] [--repeats N]");
            const std::string_view value = argv[++i];
            if (key == "--kind") kind = value;
            else if (key == "--blocks") blocks = number(value);
            else if (key == "--repeats") repeats = number(value);
            else if (key == "--threads") {
                if (number(value) != threads) throw std::invalid_argument("threads must equal 256");
            } else throw std::invalid_argument("unknown flag");
        }
        if ((kind != "global" && kind != "shared") || blocks > 1000000 || repeats > 100000)
            throw std::invalid_argument("invalid kind or bounds");
        const std::size_t elements = std::size_t{blocks} * threads;
        const std::size_t bytes = elements * sizeof(unsigned);
        std::cout << "probe=block-count/1 kind=" << kind << " blocks=" << blocks
                  << " threads=" << threads << " repeats=" << repeats
                  << " explicit_device_bytes=" << bytes << '\n' << std::flush;
        cu(cudaSetDevice(0), "cudaSetDevice");
        std::cout << "BEFORE_ALLOC\n" << std::flush;
        cu(cudaMalloc(&out, bytes), "cudaMalloc");
        std::cout << "AFTER_ALLOC\n" << std::flush;
        std::vector<unsigned> values(elements);
        for (unsigned repeat = 0; repeat < repeats; ++repeat) {
            std::cout << "BEFORE_LAUNCH repeat=" << repeat << '\n' << std::flush;
            if (kind == "global") global_store<<<blocks, threads>>>(out);
            else shared_neighbor<<<blocks, threads>>>(out);
            cu(cudaGetLastError(), "kernel launch");
            std::cout << "AFTER_LAUNCH repeat=" << repeat << '\n' << std::flush;
            cu(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            std::cout << "AFTER_SYNC repeat=" << repeat << '\n' << std::flush;
            cu(cudaMemcpy(values.data(), out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
            std::cout << "AFTER_D2H repeat=" << repeat << '\n' << std::flush;
            for (std::size_t i = 0; i < elements; ++i) {
                const unsigned expected = kind == "global" ? static_cast<unsigned>(i)
                    : static_cast<unsigned>((i / threads) * threads + (i % threads + 1) % threads);
                if (values[i] != expected)
                    throw std::runtime_error("output mismatch at index " + std::to_string(i));
            }
            std::cout << "VERIFY_PASS repeat=" << repeat << " elements=" << elements << '\n' << std::flush;
        }
        cu(cudaFree(out), "cudaFree"); out = nullptr;
        std::cout << "AFTER_FREE\nblock_memory_probe PASS\n" << std::flush;
        return 0;
    } catch (const std::exception& e) {
        if (out) cudaFree(out);
        std::cerr << "block_memory_probe: " << e.what() << '\n';
        return 1;
    }
}

#include "saccade/model_producer_cuda.hpp"
#include "saccade/tracking_session.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
// This is only a hardware-counter access probe, not a workload benchmark.
__global__ void counter_access_probe(float* out) { out[threadIdx.x] = float(threadIdx.x); }

int main(int argc, char** argv) {
    try {
        check(cudaSetDevice(0));
        cudaDeviceProp p{};
        check(cudaGetDeviceProperties(&p, 0));
        int driver = 0, runtime = 0;
        check(cudaDriverGetVersion(&driver));
        check(cudaRuntimeGetVersion(&runtime));
        size_t set_aside = 0;
        check(cudaDeviceGetLimit(&set_aside, cudaLimitPersistingL2CacheSize));
        std::cout << "{\n  \"name\":\"" << p.name << "\",\n";
#define FIELD(name, value) std::cout << "  \"" name "\":" << (value) << ",\n"
        FIELD("compute_major", p.major);
        FIELD("compute_minor", p.minor);
        FIELD("sm_count", p.multiProcessorCount);
        FIELD("l2_bytes", p.l2CacheSize);
        FIELD("persisting_l2_max_bytes", p.persistingL2CacheMaxSize);
        FIELD("persisting_l2_current_bytes", set_aside);
        FIELD("access_policy_max_window_bytes", p.accessPolicyMaxWindowSize);
        FIELD("shared_per_sm_bytes", p.sharedMemPerMultiprocessor);
        FIELD("shared_per_block_bytes", p.sharedMemPerBlock);
        FIELD("shared_per_block_optin_bytes", p.sharedMemPerBlockOptin);
        FIELD("reserved_shared_per_block_bytes", p.reservedSharedMemPerBlock);
        FIELD("registers_per_sm_32bit", p.regsPerMultiprocessor);
        FIELD("max_threads_per_sm", p.maxThreadsPerMultiProcessor);
        FIELD("max_blocks_per_sm", p.maxBlocksPerMultiProcessor);
        FIELD("warp_size", p.warpSize);
        FIELD("global_memory_bytes", p.totalGlobalMem);
        FIELD("driver_version", driver);
        FIELD("runtime_version", runtime);
        FIELD("DeviceTrack_bytes", sizeof(saccade::DeviceTrack));
        FIELD("DeviceTrackState_bytes", sizeof(saccade::DeviceTrackState));
        FIELD("DevicePredicted_bytes", sizeof(saccade::DevicePredicted));
        FIELD("DeviceBatch_bytes", sizeof(saccade::DeviceBatch));
        FIELD("DeviceTrackingRows_bytes", sizeof(saccade::DeviceTrackingRows));
        FIELD("TrackingCudaScratch_bytes", sizeof(saccade::TrackingCudaScratch));
        FIELD("DetectionCandidates_bytes", sizeof(saccade::DetectionCandidates));
        FIELD("DeviceTrack_P_offset", offsetof(saccade::DeviceTrack, P));
        FIELD("DeviceTrack_x_offset", offsetof(saccade::DeviceTrack, x));
        FIELD("DeviceTrack_history_offset", offsetof(saccade::DeviceTrack, history));
        FIELD("DeviceTrack_history_count_offset", offsetof(saccade::DeviceTrack, history_count));
        FIELD("DeviceTrack_h_ema_offset", offsetof(saccade::DeviceTrack, h_ema));
        std::cout << "  \"scope\":\"device query and ABI only; no cache hit-rate measurement\"\n}\n";
        if (argc == 2 && std::string_view(argv[1]) == "--launch") {
            float* out = nullptr;
            check(cudaMalloc(&out, 256 * sizeof(float)));
            counter_access_probe<<<1, 256>>>(out);
            check(cudaGetLastError());
            check(cudaDeviceSynchronize());
            check(cudaFree(out));
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

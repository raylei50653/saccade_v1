#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

// CUPTI callback trace of every CUDA runtime and driver API entry made while
// armed (S03 execution-constraint evidence, docs/SCAN_TRT.md §6). Stream
// arguments are decoded for launches, memsets, copies, event records, stream
// waits and graph launches; any other stream-bearing call stays undecoded.
struct TracedCall {
    std::string name;
    bool driver = false;
    bool stream_known = false;  // the call takes a stream and it was decoded
    bool stream_bearing = false; // the call takes a stream
    cudaStream_t stream{};
    bool device_to_host = false; // a copy whose destination is host memory
};

class CudaApiTrace {
public:
    CudaApiTrace();  // subscribes; throws std::runtime_error when CUPTI refuses
    ~CudaApiTrace(); // unsubscribes
    CudaApiTrace(const CudaApiTrace&) = delete;
    CudaApiTrace& operator=(const CudaApiTrace&) = delete;
    void arm();
    std::vector<TracedCall> disarm();
};

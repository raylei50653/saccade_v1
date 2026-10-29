#pragma once

#include "saccade/selective_scan_cuda.hpp"

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace saccade {

// Frame-end control of the S03 harness on the M stream: the ASYNC_CONTROL §5
// begin / gate / guarded publish / commit / packet sequence, with the scan
// plugins' status outputs as the producer status (docs/SCAN_TRT.md §4).
inline constexpr unsigned kScanFrameMaxStatuses = 4;

// M-only control block; not a wire ABI.
struct ScanFrameControl {
    std::uint64_t epoch;
    std::int64_t current, committed, fail_frame;
    std::uint32_t committed_slot, working_slot, frame_status, fail_code, run_stage, published;
    std::uint32_t scan_flags; // OR of every plugin flags word the gate read (diagnostic)
};

// What the gate read and the publish consumed for one frame; diagnostic,
// read on the host only after both terminal drains.
struct ScanFrameRecord {
    std::uint32_t flags[kScanFrameMaxStatuses];
    std::uint32_t done[kScanFrameMaxStatuses];
    std::uint32_t frame_status; // after the gate
    std::uint32_t published;    // 1 when the guarded publish copied the payload
    std::uint64_t digest;       // order-independent digest of the payload bits it copied
};

// Device pointers to one engine slot's status outputs.
struct ScanStatusList {
    ScanStatus* status[kScanFrameMaxStatuses];
    unsigned count;
};

// Loads the frame kernels in the current CUDA context (initialization only).
std::string scan_frame_load_kernels();

// All launches are single-stream and asynchronous; they throw std::runtime_error
// when a launch fails.
void scan_frame_initialize(ScanFrameControl* control, std::uint64_t epoch, std::int64_t* result_frames, cudaStream_t m);
void scan_frame_begin(ScanFrameControl* control, std::uint64_t epoch, std::int64_t frame, cudaStream_t m);
// Always runs. Reads every status, maps it to the frame's error bits
// (flags != 0 -> NUMERIC; done != kScanDone or unknown flag bits -> PROTOCOL),
// then retires each status to {0, 0} so a later read without a new plugin
// write fails, and fixes run_stage for the publish.
void scan_frame_gate(ScanFrameControl* control, ScanStatusList statuses, ScanFrameRecord* records, std::int64_t capacity, cudaStream_t m);
// Guarded by run_stage: copies the payload into result version working_slot.
void scan_frame_publish(ScanFrameControl* control, const float* payload, std::size_t count, float* results,
                        std::int64_t* result_frames, ScanFrameRecord* records, std::int64_t capacity, cudaStream_t m);
// Always runs: a successful frame with a published payload advances the
// committed version; the first failing frame locks fail_frame / fail_code.
void scan_frame_commit(ScanFrameControl* control, cudaStream_t m);
// Always runs: the complete ResultPacket v1 bytes (header; count = 0, rows zero).
void scan_frame_packet(const ScanFrameControl* control, unsigned char* packet, cudaStream_t m);
// Clock-cycle busy wait used to stress slot ownership.
void scan_frame_delay(unsigned cycles, cudaStream_t s);

// Host twin of the publish digest for expected payloads.
std::uint64_t scan_frame_digest(const float* payload, std::size_t count);

} // namespace saccade

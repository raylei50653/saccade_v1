#pragma once

#include "saccade/selective_scan.hpp"

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <string>

namespace saccade {

// S02 CUDA FP32 forward of the S01 SelectiveScan and Cross-Scan contract
// (docs/SCAN_CONTRACT.md §3-§4). Algorithm, stream/status rules, support limits
// and comparison policy: docs/SCAN_CUDA.md. Tensors are dense row-major FP32
// device memory with the layouts of selective_scan.hpp.

// Device-side outcome of one call in caller-owned device memory. Every call
// clears it on its stream before any other work and the last kernel writes
// `done`, so an output is valid only when done == kScanDone and flags == 0.
// On failure every output element is a quiet NaN; no partial result remains.
struct ScanStatus {
    std::uint32_t flags;
    std::uint32_t done;
};
inline constexpr std::uint32_t kScanDone = 0x4E414353u; // "SCAN"

// Input bits are exact: set iff that tensor holds a NaN or an infinity. Range
// bits name, per state thread, the first quantity in the reference's order whose
// FP32 value left the finite range; they carry no meaning when an input bit is set.
enum ScanFailure : std::uint32_t {
    kScanNonFiniteU = 1u << 0,
    kScanNonFiniteDelta = 1u << 1,
    kScanNonFiniteA = 1u << 2,
    kScanNonFiniteB = 1u << 3,
    kScanNonFiniteC = 1u << 4,
    kScanNonFiniteD = 1u << 5,
    kScanDecayRange = 1u << 8,
    kScanDriveRange = 1u << 9,
    kScanStateRange = 1u << 10,
    kScanTermRange = 1u << 11,
    kScanSkipRange = 1u << 12,
    kScanOutputRange = 1u << 13,
    kCrossScanNonFiniteInput = 1u << 16,
};
inline constexpr std::uint32_t kScanInputFailures = 0x3Fu;
inline constexpr std::uint32_t kScanRangeFailures = 0x3F00u;

struct SelectiveScanDeviceInputs {
    const float* u = nullptr;
    const float* delta = nullptr;
    const float* a = nullptr;
    const float* b = nullptr;
    const float* c = nullptr;
    const float* d = nullptr;
};

// Call once per process during session initialization, before frame work is in
// flight. Under CUDA lazy loading (the default), the first launch of a kernel
// loads it and can wait for work already running on the device; loading every
// kernel here keeps that implicit synchronization out of the frame path. Returns
// each kernel's registers, local, shared and thread limits, one per line.
std::string selective_scan_cuda_load_kernels();

// Validates on the host, then enqueues on `stream` only: status clear, input
// check, scan, finalize. Once the kernels are loaded: no allocation,
// synchronization, host readback, event or default-stream work, and no
// workspace, since the recurrence state lives in registers. Throws std::invalid_argument for shape, capacity, pointer, overlap
// or stream violations before enqueueing anything, and std::runtime_error when a
// CUDA API call fails (the status then never reaches done).
void selective_scan_cuda(const SelectiveScanShape& shape, const SelectiveScanDeviceInputs& inputs,
                         float* y, ScanStatus* status, cudaStream_t stream);

// x: (batch, channels, height, width); tokens: (4 * batch, height * width,
// channels), direction-major rows, each in its direction's token order.
struct CrossScanShape {
    std::size_t batch = 0, channels = 0, height = 0, width = 0;
};
void cross_scan_expand_cuda(const CrossScanShape& shape, const float* x, float* tokens,
                            ScanStatus* status, cudaStream_t stream);
// y: tokens layout -> out: (batch, channels, height, width), the mean of the
// four directions mapped back to (h, w).
void cross_scan_merge_cuda(const CrossScanShape& shape, const float* y, float* out,
                           ScanStatus* status, cudaStream_t stream);

// Build identity of the compiled kernels: commit, dirty flag, nvcc version,
// CUDA flags and target architectures, one key=value per line.
std::string selective_scan_cuda_identity();

} // namespace saccade

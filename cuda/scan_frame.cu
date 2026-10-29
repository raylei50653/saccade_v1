#include "saccade/scan_frame_kernels.hpp"
#include "saccade/protocol.hpp"

#include <cuda_runtime.h>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace saccade {
namespace {

void ck(cudaError_t result, const char* call) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(result));
}
#define CUDA(call) ck((call), #call)
void launched(const char* kernel) { ck(cudaGetLastError(), kernel); }

constexpr unsigned kProtocol = static_cast<unsigned>(Error::protocol), kNumeric = static_cast<unsigned>(Error::numeric);
constexpr unsigned kKnownScanFlags = kScanInputFailures | kScanRangeFailures | kCrossScanNonFiniteInput;

__host__ __device__ std::uint64_t mix(std::uint64_t x) { // splitmix64 finalizer
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__global__ void initialize(ScanFrameControl* ctl, std::uint64_t epoch, std::int64_t* result_frames) {
    if (threadIdx.x || blockIdx.x) return;
    *ctl = ScanFrameControl{};
    ctl->epoch = epoch;
    ctl->current = ctl->committed = ctl->fail_frame = -1;
    ctl->committed_slot = 1;
    result_frames[0] = result_frames[1] = -1;
}

__global__ void begin(ScanFrameControl* ctl, std::uint64_t epoch, std::int64_t frame) {
    if (threadIdx.x || blockIdx.x) return;
    ctl->current = frame;
    ctl->working_slot = 1 - ctl->committed_slot;
    ctl->frame_status = 0;
    ctl->run_stage = 0;
    ctl->published = 0;
    if (epoch != ctl->epoch || (!ctl->fail_code && frame != ctl->committed + 1)) ctl->frame_status |= kProtocol;
}

__global__ void gate(ScanFrameControl* ctl, ScanStatusList list, ScanFrameRecord* records, std::int64_t capacity) {
    if (threadIdx.x || blockIdx.x) return;
    ScanFrameRecord* record = ctl->current >= 0 && ctl->current < capacity ? records + ctl->current : nullptr;
    if (record) *record = ScanFrameRecord{};
    for (unsigned i = 0; i < list.count; ++i) {
        ScanStatus* s = list.status[i];
        const std::uint32_t flags = s->flags, done = s->done;
        if (done != kScanDone) ctl->frame_status |= kProtocol;
        if (flags) ctl->frame_status |= kNumeric;
        if (flags & ~kKnownScanFlags) ctl->frame_status |= kProtocol;
        ctl->scan_flags |= flags;
        s->flags = 0;
        s->done = 0; // retired: only the next plugin call can make it valid again
        if (record) { record->flags[i] = flags; record->done[i] = done; }
    }
    if (list.count == 0) ctl->frame_status |= kProtocol; // a frame without a producer status cannot pass
    if (record) record->frame_status = ctl->frame_status;
    ctl->run_stage = !ctl->fail_code && !ctl->frame_status;
}

__global__ void publish(ScanFrameControl* ctl, const float* payload, unsigned long long count, float* results,
                        std::int64_t* result_frames, ScanFrameRecord* records, std::int64_t capacity) {
    if (!ctl->run_stage) return; // fixed by the gate; this grid never writes it
    const unsigned long long first = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    float* out = results + ctl->working_slot * count;
    std::uint64_t digest = 0;
    for (unsigned long long i = first; i < count; i += stride) {
        const float v = payload[i];
        out[i] = v;
        digest += mix((i << 32) | __float_as_uint(v));
    }
    const bool logged = ctl->current >= 0 && ctl->current < capacity;
    if (logged && digest) atomicAdd(reinterpret_cast<unsigned long long*>(&records[ctl->current].digest), digest);
    if (first == 0) {
        result_frames[ctl->working_slot] = ctl->current;
        ctl->published = 1;
        if (logged) records[ctl->current].published = 1;
    }
}

__global__ void commit(ScanFrameControl* ctl) {
    if (threadIdx.x || blockIdx.x) return;
    if (!ctl->fail_code) {
        if (!ctl->frame_status && !ctl->published) ctl->frame_status |= kProtocol;
        if (ctl->frame_status) {
            ctl->fail_frame = ctl->current;
            ctl->fail_code = ctl->frame_status;
        } else {
            ctl->committed = ctl->current;
            ctl->committed_slot = ctl->working_slot;
        }
    }
    ctl->run_stage = 0;
}

__device__ void put(unsigned char* p, unsigned long long v, int n) {
    for (int i = 0; i < n; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}

// ResultPacket v1 (docs/RESULT_PACKET.md), field by field; this harness has
// no tracks, so every packet has count 0 and all rows stay zero.
__global__ void packet(const ScanFrameControl* ctl, unsigned char* out) {
    for (int i = threadIdx.x; i < static_cast<int>(packet_bytes); i += blockDim.x) out[i] = 0;
    __syncthreads();
    if (threadIdx.x) return;
    out[0] = 'S'; out[1] = 'C'; out[2] = 'R'; out[3] = 'P';
    put(out + 4, 1, 2); put(out + 6, packet_header_bytes, 2); put(out + 8, packet_bytes, 4); put(out + 12, packet_capacity, 4);
    put(out + 16, ctl->epoch, 8); put(out + 24, ctl->current, 8); put(out + 32, ctl->committed, 8); put(out + 40, ctl->fail_frame, 8);
    const unsigned status = !ctl->fail_code ? 0u : (ctl->current == ctl->fail_frame ? 1u : 2u);
    put(out + 48, status, 4);
    put(out + 56, status == 1 ? ctl->fail_code : 0u, 4);
    put(out + 60, ctl->fail_code, 4);
}

__global__ void delay(unsigned cycles) {
    const long long start = clock64();
    while (clock64() - start < cycles) {
    }
}

constexpr unsigned kBlock = 256;

} // namespace

std::string scan_frame_load_kernels() {
    const std::pair<const char*, const void*> kernels[] = {
        {"initialize", reinterpret_cast<const void*>(initialize)}, {"begin", reinterpret_cast<const void*>(begin)},
        {"gate", reinterpret_cast<const void*>(gate)}, {"publish", reinterpret_cast<const void*>(publish)},
        {"commit", reinterpret_cast<const void*>(commit)}, {"packet", reinterpret_cast<const void*>(packet)},
        {"delay", reinterpret_cast<const void*>(delay)}};
    std::ostringstream report;
    for (const auto& [name, kernel] : kernels) {
        cudaFuncAttributes attributes{};
        CUDA(cudaFuncGetAttributes(&attributes, kernel)); // loads the kernel under lazy loading
        report << "frame_kernel=" << name << " registers=" << attributes.numRegs << " local_bytes=" << attributes.localSizeBytes << '\n';
    }
    return report.str();
}

void scan_frame_initialize(ScanFrameControl* control, std::uint64_t epoch, std::int64_t* result_frames, cudaStream_t m) {
    initialize<<<1, 1, 0, m>>>(control, epoch, result_frames);
    launched("scan_frame initialize");
}
void scan_frame_begin(ScanFrameControl* control, std::uint64_t epoch, std::int64_t frame, cudaStream_t m) {
    begin<<<1, 1, 0, m>>>(control, epoch, frame);
    launched("scan_frame begin");
}
void scan_frame_gate(ScanFrameControl* control, ScanStatusList statuses, ScanFrameRecord* records, std::int64_t capacity, cudaStream_t m) {
    if (statuses.count > kScanFrameMaxStatuses) throw std::invalid_argument("scan_frame_gate: too many statuses");
    gate<<<1, 1, 0, m>>>(control, statuses, records, capacity);
    launched("scan_frame gate");
}
void scan_frame_publish(ScanFrameControl* control, const float* payload, std::size_t count, float* results, std::int64_t* result_frames,
                        ScanFrameRecord* records, std::int64_t capacity, cudaStream_t m) {
    const unsigned long long blocks = (count + kBlock - 1) / kBlock;
    publish<<<static_cast<unsigned>(blocks == 0 ? 1 : blocks < 1024 ? blocks : 1024), kBlock, 0, m>>>(control, payload, count, results,
                                                                                                      result_frames, records, capacity);
    launched("scan_frame publish");
}
void scan_frame_commit(ScanFrameControl* control, cudaStream_t m) {
    commit<<<1, 1, 0, m>>>(control);
    launched("scan_frame commit");
}
void scan_frame_packet(const ScanFrameControl* control, unsigned char* out, cudaStream_t m) {
    packet<<<1, kBlock, 0, m>>>(control, out);
    launched("scan_frame packet");
}
void scan_frame_delay(unsigned cycles, cudaStream_t s) {
    delay<<<1, 1, 0, s>>>(cycles);
    launched("scan_frame delay");
}

std::uint64_t scan_frame_digest(const float* payload, std::size_t count) {
    std::uint64_t digest = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, payload + i, sizeof bits);
        digest += mix((static_cast<std::uint64_t>(i) << 32) | bits);
    }
    return digest;
}

} // namespace saccade

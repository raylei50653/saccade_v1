#include "saccade/tracking_session.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace saccade::detail {
namespace {
void ck(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(std::string("tracking session kernel: ") + cudaGetErrorString(code));
}
__device__ Error merge(Error a, Error b) { return static_cast<Error>(static_cast<unsigned>(a) | static_cast<unsigned>(b)); }
__device__ void prepare(TrackingSessionControl* c) {
    c->value.run_stage = c->value.fail_code == Error::none && c->value.frame_status == Error::none;
    for (unsigned i = 0; i < 2; ++i) c->slot_run[i] = c->value.run_stage && c->value.working_slot == i;
}
__global__ void initialize(TrackingSessionControl* c, DeviceTrackState* states, Epoch epoch) {
    if (threadIdx.x || blockIdx.x) return;
    c->value = Control{};
    c->value.epoch = epoch;
    c->slot_run[0] = c->slot_run[1] = false;
    for (unsigned i = 0; i < 2; ++i) {
        states[i] = {};
        states[i].epoch = epoch;
        states[i].committed_frame = no_frame;
        states[i].next_uid = 1;
    }
}
__global__ void begin(TrackingSessionControl* c, Epoch epoch, FrameId frame) {
    auto& v = c->value;
    v.current_frame = frame;
    v.working_slot = 1 - v.committed_slot;
    v.frame_status = Error::none;
    v.run_stage = c->slot_run[0] = c->slot_run[1] = false;
    if (v.fail_code == Error::none && (v.epoch != epoch || frame != v.committed_frame + 1))
        v.frame_status = Error::protocol;
}
__global__ void stage(TrackingSessionControl* c, const Error* error,
                      const MotionContext* motion, DeviceMotionContext* converted) {
    if (error) c->value.frame_status = merge(c->value.frame_status, *error);
    if (motion && c->value.fail_code == Error::none && c->value.frame_status == Error::none) {
        if (!isfinite(motion->pcr) || !isfinite(motion->confidence) || !isfinite(motion->tx) || !isfinite(motion->ty))
            c->value.frame_status = merge(c->value.frame_status, Error::numeric);
        if (motion->epoch != c->value.epoch || motion->frame != c->value.current_frame ||
            motion->reference_frame != c->value.committed_frame)
            c->value.frame_status = merge(c->value.frame_status, Error::protocol);
        *converted = {motion->epoch, motion->frame, motion->reference_frame, motion->status,
                      motion->peak_x, motion->peak_y, motion->pcr, motion->confidence, motion->tx, motion->ty};
    }
    prepare(c);
}
__global__ void candidates(TrackingSessionControl* c, const DetectionCandidates* source) {
    const auto& h = source->header;
    if (h.epoch != c->value.epoch || h.frame != static_cast<std::uint64_t>(c->value.current_frame) || h.done != kPostprocessDone)
        c->value.frame_status = merge(c->value.frame_status, Error::protocol);
    if (h.flags & kPostprocessOverflow) c->value.frame_status = merge(c->value.frame_status, Error::overflow);
    if (h.flags & ~kPostprocessOverflow) c->value.frame_status = merge(c->value.frame_status, Error::producer);
    prepare(c);
}
__global__ void commit(TrackingSessionControl* c) {
    auto& v = c->value;
    v.run_stage = c->slot_run[0] = c->slot_run[1] = false;
    if (v.fail_code != Error::none) return;
    if (v.frame_status != Error::none) {
        v.fail_frame = v.current_frame;
        v.fail_code = v.frame_status;
    } else {
        v.committed_frame = v.current_frame;
        v.committed_slot = v.working_slot;
    }
}
__device__ void put(std::uint8_t* out, unsigned offset, std::uint64_t value, unsigned size) {
    for (unsigned i = 0; i < size; ++i) out[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
}
__device__ void put_float(std::uint8_t* out, unsigned offset, float value) {
    put(out, offset, value == 0 ? 0 : __float_as_uint(value), 4);
}
__global__ void packet(const Control* c, const DeviceTrackingRows* rows, std::uint8_t* out) {
    for (unsigned i = threadIdx.x; i < packet_bytes; i += blockDim.x) out[i] = 0;
    __syncthreads();
    const bool ok = c->fail_code == Error::none;
    const bool failed = !ok && c->current_frame == c->fail_frame;
    const unsigned count = ok ? rows->count : 0;
    if (threadIdx.x == 0) {
        out[0] = 'S'; out[1] = 'C'; out[2] = 'R'; out[3] = 'P';
        put(out, 4, 1, 2); put(out, 6, packet_header_bytes, 2);
        put(out, 8, packet_bytes, 4); put(out, 12, packet_capacity, 4);
        put(out, 16, c->epoch, 8); put(out, 24, c->current_frame, 8);
        put(out, 32, c->committed_frame, 8); put(out, 40, c->fail_frame, 8);
        put(out, 48, ok ? 0 : failed ? 1 : 2, 4); put(out, 52, count, 4);
        put(out, 56, failed ? static_cast<unsigned>(c->fail_code) : 0, 4);
        put(out, 60, static_cast<unsigned>(c->fail_code), 4);
    }
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) {
        const auto& row = rows->rows[i];
        const unsigned offset = packet_header_bytes + packet_row_bytes * i;
        put(out, offset, row.uid, 8); put(out, offset + 8, row.output_id, 8);
        put(out, offset + 16, row.class_id, 4); put(out, offset + 20, row.flags, 4);
        for (unsigned j = 0; j < 4; ++j) put_float(out, offset + 24 + 4 * j, row.xyxy[j]);
        put_float(out, offset + 40, row.observed_score); put_float(out, offset + 44, row.association_score);
    }
}
__global__ void perturb(unsigned cycles) {
    const auto start = clock64();
    while (clock64() - start < cycles) {}
}
__global__ void producer_flag(PostprocessProducerStatus* p, std::uint32_t flags) { p->flags = flags; }
}
void tracking_session_load_kernels() {
    cudaFuncAttributes a{};
    ck(cudaFuncGetAttributes(&a, initialize)); ck(cudaFuncGetAttributes(&a, begin));
    ck(cudaFuncGetAttributes(&a, stage)); ck(cudaFuncGetAttributes(&a, candidates));
    ck(cudaFuncGetAttributes(&a, commit)); ck(cudaFuncGetAttributes(&a, packet));
    ck(cudaFuncGetAttributes(&a, perturb)); ck(cudaFuncGetAttributes(&a, producer_flag));
}
void tracking_session_initialize(TrackingSessionControl* c, DeviceTrackState* s, Epoch e, cudaStream_t stream) {
    initialize<<<1, 1, 0, stream>>>(c, s, e); ck(cudaGetLastError());
}
void tracking_session_begin(TrackingSessionControl* c, Epoch e, FrameId k, cudaStream_t stream) {
    begin<<<1, 1, 0, stream>>>(c, e, k); ck(cudaGetLastError());
}
void tracking_session_stage(TrackingSessionControl* c, const Error* e, const MotionContext* m, DeviceMotionContext* d, cudaStream_t stream) {
    stage<<<1, 1, 0, stream>>>(c, e, m, d); ck(cudaGetLastError());
}
void tracking_session_candidates(TrackingSessionControl* c, const DetectionCandidates* d, cudaStream_t stream) {
    candidates<<<1, 1, 0, stream>>>(c, d); ck(cudaGetLastError());
}
void tracking_session_commit(TrackingSessionControl* c, cudaStream_t stream) {
    commit<<<1, 1, 0, stream>>>(c); ck(cudaGetLastError());
}
void tracking_session_packet(const Control* c, const DeviceTrackingRows* r, std::uint8_t* out, cudaStream_t stream) {
    packet<<<1, 256, 0, stream>>>(c, r, out); ck(cudaGetLastError());
}
void tracking_session_perturb(std::uint64_t seed, FrameId frame, unsigned boundary, cudaStream_t stream) {
    if (!seed) return;
    auto x = seed ^ (static_cast<std::uint64_t>(frame) * 0x9e3779b97f4a7c15ULL) ^ boundary;
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27;
    perturb<<<1, 1, 0, stream>>>(1000 + x % 100000); ck(cudaGetLastError());
}
void tracking_session_producer_flag(PostprocessProducerStatus* p, std::uint32_t flags, cudaStream_t stream) {
    producer_flag<<<1, 1, 0, stream>>>(p, flags); ck(cudaGetLastError());
}
} // namespace saccade::detail

#include "saccade/detail/detection_session_kernels.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace saccade::detail {
namespace {
void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string("detection session kernel: ") + cudaGetErrorString(rc));
}
__global__ void initialize(DetectionSessionControl* c) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        c->fail_frame = kDetectionSessionNoFrame;
        c->committed_frame = kDetectionSessionNoFrame;
        c->fail_code = 0;
        c->reserved = 0;
    }
}
__global__ void finalize(const DetectionCandidates* source, DetectionSessionResult* result,
                         DetectionSessionControl* control, std::uint64_t epoch, std::uint64_t frame,
                         std::uint32_t capacity) {
    // Single thread deliberately fixes observation/commit order and initializes
    // the entire bounded ABI. No grid-wide or mutable per-block guard.
    if (threadIdx.x || blockIdx.x) return;
    *result = {};
    result->epoch = epoch;
    result->frame = frame;
    const auto& h = source->header;
    std::uint32_t error = h.flags;
    if (h.epoch != epoch || h.frame != frame || h.schema != kPostprocessSchema ||
        h.capacity != capacity || h.done != kPostprocessDone || h.count > capacity ||
        h.kept > capacity || h.suppressed > capacity || h.kept + h.suppressed != h.count || h.reserved != 0 ||
        (h.flags & ~(kPostprocessNumeric | kPostprocessUpstream | kPostprocessProtocol | kPostprocessOverflow)))
        error |= kPostprocessProtocol;
    if (control->fail_code == 0 && frame != (control->committed_frame == kDetectionSessionNoFrame ? 0 : control->committed_frame + 1))
        error |= kPostprocessProtocol;
    if (control->fail_code != 0) {
        result->status = DetectionSessionStatus::SkippedAfterFailure;
    } else if (error != 0) {
        control->fail_frame = frame;
        control->fail_code = error;
        result->status = DetectionSessionStatus::Failed;
    } else {
        control->committed_frame = frame;
        result->status = DetectionSessionStatus::Ok;
        result->candidates = *source;
    }
    result->fail_frame = control->fail_frame;
    result->committed_frame = control->committed_frame;
    result->fail_code = control->fail_code;
    if (result->status != DetectionSessionStatus::Ok) {
        auto& out = result->candidates.header;
        out.epoch = epoch;
        out.frame = frame;
        out.schema = kPostprocessSchema;
        out.capacity = capacity;
        out.flags = control->fail_code;
        out.done = kPostprocessDone;
    }
}
__global__ void test_override(PostprocessProducerStatus* p, std::uint32_t flags) {
    if (threadIdx.x == 0 && blockIdx.x == 0) p->flags = flags;
}
__global__ void test_corrupt(DetectionCandidates* p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) ++p->header.epoch;
}
}
void detection_session_load_kernels() {
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, initialize));
    check(cudaFuncGetAttributes(&attributes, finalize));
    check(cudaFuncGetAttributes(&attributes, test_override));
    check(cudaFuncGetAttributes(&attributes, test_corrupt));
}
void detection_session_initialize(DetectionSessionControl* c, cudaStream_t stream) {
    initialize<<<1,1,0,stream>>>(c); check(cudaPeekAtLastError());
}
void detection_session_finalize(const DetectionCandidates* source, DetectionSessionResult* result,
                                DetectionSessionControl* control, std::uint64_t epoch, std::uint64_t frame,
                                std::uint32_t capacity, cudaStream_t stream) {
    finalize<<<1,1,0,stream>>>(source,result,control,epoch,frame,capacity); check(cudaPeekAtLastError());
}
void detection_session_test_override(PostprocessProducerStatus* p, std::uint32_t flags, cudaStream_t stream) {
    test_override<<<1,1,0,stream>>>(p,flags); check(cudaPeekAtLastError());
}
void detection_session_test_corrupt(DetectionCandidates* p, cudaStream_t stream) {
    test_corrupt<<<1,1,0,stream>>>(p); check(cudaPeekAtLastError());
}
}

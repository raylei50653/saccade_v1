#pragma once

#include "saccade/postprocess.hpp"

#include <cuda_runtime_api.h>
#include <cstddef>
#include <string>

namespace saccade {

// Opaque fixed workspace. It must be device memory aligned to at least
// alignof(float); cudaMalloc satisfies this. No per-frame allocation is needed.
std::size_t postprocess_workspace_bytes();

// F04, f04-postprocess-fp32/1. All raw tensors, producer, output and workspace
// are disjoint allocations/ranges on the current CUDA device. Exact tensor
// counts and mapping are checked before any work is enqueued. output_bytes and
// workspace_bytes must cover the sizes reported here. The caller owns all
// buffers and keeps them alive through completion on the given explicit,
// non-default stream; output cannot be overwritten until its consumer finishes.
// Each accessed range is also checked against the actual allocation reported
// by cuMemGetAddressRange. cudaMalloc allocations and contained arena subranges
// are supported; allocation types whose extent cannot be established by that
// driver API are rejected before enqueueing (including unsupported pool/virtual
// memory allocations).
//
// The producer is borrowed read-only and must have been freshly published on
// this stream, or ordered by a caller-owned producer-ready event. F04 does not
// establish the upstream dependency, retire the producer, latch a session-wide
// failure, or manage a downstream consumer lease. Those are session duties.
//
// Each host submitter thread must initialize/activate its device context with
// cudaSetDevice(device) before frame work; another thread's initialization does
// not establish the current driver context needed by the allocation queries.
// Call load_kernels once per context before frame work. F04 does not initialize
// or switch the caller's thread context in the enqueue path.
// After initialization/load_kernels: no allocation, transfer, synchronization,
// event, or default-stream work in this enqueue path. Capture/replay is allowed
// with fixed pointers/request. Host argument errors throw invalid_argument;
// CUDA API/launch failures throw runtime_error and do not promise a done header.
void postprocess_cuda(const PostprocessRawView& raw, const PostprocessRequest& request,
                      const PostprocessProducerStatus* producer,
                      DetectionCandidates* output, std::size_t output_bytes,
                      void* workspace, std::size_t workspace_bytes,
                      cudaStream_t stream);

// Initialization only: eager kernel loading prevents CUDA lazy loading from
// introducing a first-use synchronization in the frame path.
std::string postprocess_cuda_load_kernels();
std::string postprocess_cuda_identity();

} // namespace saccade

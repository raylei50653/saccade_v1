#pragma once

#include "saccade/postprocess.hpp"
#include "saccade/selective_scan_cuda.hpp"

#include <array>
#include <cstddef>
#include <string>

namespace saccade {

inline constexpr const char* kModelProducerPolicy = "f03-producer/1";
inline constexpr std::size_t kModelProducerInputElements = 3 * 640 * 640;
inline constexpr unsigned kModelProducerPlugins = 9;

// Exactly model_io_spec()'s nine status outputs: P3, P4, P5, each in
// expand / scan / merge order. They are consumed and retired to {0,0}.
using ModelProducerPlugins = std::array<ScanStatus*, kModelProducerPlugins>;

// Two internal finite-check statuses. Query this build's size; device memory
// must be aligned to alignof(ScanStatus). No per-frame allocation is needed.
std::size_t model_producer_workspace_bytes();

// After the F03 engine on the caller's explicit non-default detect stream:
// check EVERY images/raw element for NaN/Inf, gate all nine plugin statuses
// and both finite checks, retire them, then publish producer.flags followed by
// producer.done = kPostprocessProducerDone (including failures).
//
// Aggregate flags use PostprocessFailure: any source flags -> Numeric;
// missing source done or unknown/incorrect-role bits -> Protocol. These can
// coexist. F04 maps any aggregate flags to Upstream, as before.
//
// Fixed batch-one FP32 profile, images [1,3,640,640], six raw tensors in F04
// layout. All accessed ranges must be disjoint device memory on the current
// device with queryable allocation extents (cudaMalloc or arena subranges).
// Host errors throw invalid_argument BEFORE any enqueue; API/launch errors
// throw runtime_error and do not promise a completion marker.
//
// Caller initializes the thread context with cudaSetDevice and preloads these
// kernels once per context. Thereafter no allocation, transfer, synchronization,
// event or default-stream work occurs here; fixed-pointer graph capture is
// supported. Engine outputs/statuses must be fresh and ready on this stream
// (or ordered by a caller-owned ready event). Only this adapter may retire
// these statuses; do not also give them to scan_frame_gate.
// Initialize plugin statuses to {0,0} once when creating a fresh slot, before
// any engine launch, so even an omitted first write has a defined failure.
//
// images/raw/plugins/workspace/producer live through completion. Queue F04
// before reusing raw/producer; all consumers must finish before overwrite.
// The eight-byte aggregate has no frame identity and cannot detect a stale
// successful engine write. Ready/consumed edges and session identity/failure
// latching remain caller responsibilities.
void model_producer_cuda(const float* images, std::size_t image_elements,
                         const PostprocessRawView& raw, const ModelProducerPlugins& plugins,
                         PostprocessProducerStatus* producer, void* workspace,
                         std::size_t workspace_bytes, cudaStream_t stream);

std::string model_producer_cuda_load_kernels();
std::string model_producer_cuda_identity();

} // namespace saccade

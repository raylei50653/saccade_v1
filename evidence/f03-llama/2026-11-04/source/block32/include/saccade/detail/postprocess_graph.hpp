#pragma once

#include "saccade/postprocess_cuda.hpp"

namespace saccade::detail {
// The two F04 nodes which embed frame metadata by value. All other captured
// inputs, allocation extents, shapes and pointers remain fixed for the graph.
struct PostprocessGraphNodes {
    cudaGraphNode_t initialize = nullptr, finalize = nullptr;
};
// Capture must end with the five F04 kernels on one stream, with no work
// appended after finalize. Reject a different topology instead of guessing.
PostprocessGraphNodes postprocess_graph_nodes(cudaGraph_t);
void postprocess_graph_set_request(cudaGraphExec_t, const PostprocessGraphNodes&,
                                  const PostprocessProducerStatus*, DetectionCandidates*,
                                  void* workspace, const PostprocessRequest&);
} // namespace saccade::detail

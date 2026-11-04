#include "saccade/detail/tracking_tail_graph.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/tracking_session.hpp"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace saccade::detail {
namespace {
void ck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
void explicit_stream(cudaStream_t stream) {
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        throw std::invalid_argument("tracking tail graph requires an explicit non-default stream");
}
}

void enqueue_tracking_tail(const TrackingTailPointers& p, const DetectionCandidates* candidate, cudaStream_t stream) {
    tracking_session_candidates(p.control, candidate, stream);
    enqueue_assemble_detection_batch(candidate, p.predicted, p.tracker_config, p.batch,
                                     p.scratch, stream, &p.control->value.run_stage);
    tracking_session_candidates(p.control, candidate, stream);
    tracking_session_stage(p.control, &p.batch->error, nullptr, nullptr, stream);
    for (unsigned i = 0; i < 2; ++i)
        enqueue_associate_and_update(p.predicted, p.batch, p.tracker_motion, p.tracker_config, p.tracks + i,
                                     p.rows, p.scratch, stream, &p.control->slot_run[i]);
    tracking_session_stage(p.control, &p.rows->error, nullptr, nullptr, stream);
    tracking_session_commit(p.control, stream);
    tracking_session_packet(&p.control->value, p.rows, p.packet, stream);
}

TrackingTailGraph::TrackingTailGraph(const TrackingTailPointers& pointers,
                                   std::span<DetectionCandidates* const> candidates, cudaStream_t stream) {
    explicit_stream(stream);
    if (candidates.empty()) throw std::invalid_argument("tracking tail graph requires candidate slots");
    if (!pointers.control || !pointers.tracks || !pointers.predicted || !pointers.batch || !pointers.rows ||
        !pointers.scratch || !pointers.tracker_config || !pointers.tracker_motion || !pointers.packet)
        throw std::invalid_argument("tracking tail graph requires all device pointers");
    ck(cudaGetDevice(&device_), "tracking tail graph device");
    executables_.resize(candidates.size(), nullptr);
    try {
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            if (!candidates[i]) throw std::invalid_argument("tracking tail graph candidate is null");
            cudaGraph_t graph = nullptr;
            bool capturing = false;
            try {
                ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture tracking tail");
                capturing = true;
                enqueue_tracking_tail(pointers, candidates[i], stream);
                const auto ended = cudaStreamEndCapture(stream, &graph);
                capturing = false;
                ck(ended, "end tracking tail capture");
                std::size_t count = 0;
                ck(cudaGraphGetNodes(graph, nullptr, &count), "count tracking tail nodes");
                std::vector<cudaGraphNode_t> nodes(count);
                ck(cudaGraphGetNodes(graph, nodes.data(), &count), "get tracking tail nodes");
                std::size_t kernels = 0;
                for (auto node : nodes) {
                    cudaGraphNodeType type{};
                    ck(cudaGraphNodeGetType(node, &type), "get tracking tail node type");
                    if (type == cudaGraphNodeTypeKernel) ++kernels;
                }
                if (i == 0) kernel_nodes_ = kernels;
                else if (kernels != kernel_nodes_)
                    throw std::runtime_error("tracking tail graph node count differs between slots");
                ck(cudaGraphInstantiate(&executables_[i], graph, 0), "instantiate tracking tail");
                auto* retired = graph;
                graph = nullptr;
                ck(cudaGraphDestroy(retired), "destroy tracking tail capture");
            } catch (...) {
                if (capturing) {
                    cudaGraph_t abandoned = nullptr;
                    (void)cudaStreamEndCapture(stream, &abandoned);
                    if (abandoned) (void)cudaGraphDestroy(abandoned);
                }
                if (graph) (void)cudaGraphDestroy(graph);
                throw;
            }
        }
    } catch (...) {
        for (auto executable : executables_)
            if (executable) (void)cudaGraphExecDestroy(executable);
        executables_.clear();
        throw;
    }
}

TrackingTailGraph::~TrackingTailGraph() {
    try { close(); } catch (...) {}
}

void TrackingTailGraph::launch(std::size_t candidate_slot, cudaStream_t stream) {
    explicit_stream(stream);
    if (candidate_slot >= executables_.size()) throw std::out_of_range("tracking tail graph slot");
    ck(cudaGraphLaunch(executables_[candidate_slot], stream), "launch tracking tail graph");
}

void TrackingTailGraph::close() {
    CleanupReport report;
    if (executables_.empty()) return;
    int previous = -1, token = 1;
    report.release(token, [&](int) {
        ck(cudaGetDevice(&previous), "tracking tail graph cleanup device");
        ck(cudaSetDevice(device_), "tracking tail graph cleanup set device");
    });
    for (auto& executable : executables_)
        report.release(executable, [](auto handle) { ck(cudaGraphExecDestroy(handle), "destroy tracking tail graph"); });
    executables_.clear();
    if (previous >= 0 && previous != device_) {
        token = 1;
        report.release(token, [&](int) { ck(cudaSetDevice(previous), "tracking tail graph cleanup restore device"); });
    }
    report.throw_if_failed();
}
} // namespace saccade::detail

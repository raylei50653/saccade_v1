#include "saccade/scan_trt_frame.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/scan_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"

#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace saccade {
namespace {
using namespace nvinfer1;

void ck(cudaError_t result, const char* call) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(result));
}
#define CUDA(call) ck((call), #call)
void require(bool condition, const std::string& why) {
    if (!condition) throw std::runtime_error("scan frame session: " + why);
}

// The generation is the frame whose record the event currently represents.
struct Event {
    cudaEvent_t handle{};
    std::int64_t generation = -1;
};

struct Slot {
    std::unique_ptr<IExecutionContext> context;
    void* context_memory = nullptr;
    std::vector<float*> inputs;
    std::vector<float*> staging; // pinned host, one per input
    float* payload = nullptr;
    std::vector<ScanStatus*> statuses;
    Event staged, ready, consumed;
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    std::string nodes;
};

std::string node_type(cudaGraphNodeType type) {
    switch (type) {
    case cudaGraphNodeTypeKernel: return "kernel";
    case cudaGraphNodeTypeMemcpy: return "memcpy";
    case cudaGraphNodeTypeMemset: return "memset";
    case cudaGraphNodeTypeHost: return "host";
    case cudaGraphNodeTypeGraph: return "child_graph";
    case cudaGraphNodeTypeEmpty: return "empty";
    case cudaGraphNodeTypeWaitEvent: return "wait_event";
    case cudaGraphNodeTypeEventRecord: return "event_record";
    case cudaGraphNodeTypeMemAlloc: return "mem_alloc";
    case cudaGraphNodeTypeMemFree: return "mem_free";
    default: return "other";
    }
}

// One line per node in topological order: index, type, kernel name and the
// indices of the nodes it depends on.
std::string inspect(cudaGraph_t graph) {
    std::size_t count = 0, edges = 0;
    CUDA(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    CUDA(cudaGraphGetNodes(graph, nodes.data(), &count));
    CUDA(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges));
    std::vector<cudaGraphNode_t> from(edges), to(edges);
    if (edges) CUDA(cudaGraphGetEdges(graph, from.data(), to.data(), nullptr, &edges));
    std::map<cudaGraphNode_t, std::size_t> index;
    for (std::size_t i = 0; i < count; ++i) index[nodes[i]] = i;
    std::vector<std::vector<std::size_t>> deps(count);
    std::vector<std::size_t> indegree(count);
    for (std::size_t e = 0; e < edges; ++e) {
        deps[index.at(to[e])].push_back(index.at(from[e]));
        ++indegree[index.at(to[e])];
    }
    std::vector<std::size_t> order, ready, position(count);
    for (std::size_t i = 0; i < count; ++i)
        if (!indegree[i]) ready.push_back(i);
    while (!ready.empty()) {
        const auto i = ready.front();
        ready.erase(ready.begin());
        position[i] = order.size();
        order.push_back(i);
        for (std::size_t e = 0; e < edges; ++e)
            if (index.at(from[e]) == i && --indegree[index.at(to[e])] == 0) ready.push_back(index.at(to[e]));
    }
    require(order.size() == count, "captured graph has a cycle");
    std::ostringstream out;
    for (const auto i : order) {
        cudaGraphNodeType type{};
        CUDA(cudaGraphNodeGetType(nodes[i], &type));
        out << "node=" << position[i] << " type=" << node_type(type);
        if (type == cudaGraphNodeTypeKernel) {
            CUDA_KERNEL_NODE_PARAMS params{};
            const char* name = nullptr;
            if (cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(nodes[i]), &params) == CUDA_SUCCESS) {
                if (params.func) (void)cuFuncGetName(&name, params.func);
                else if (params.kern) (void)cuKernelGetName(&name, params.kern);
            }
            out << " name=" << (name ? name : "unknown");
        }
        if (type == cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms params{};
            CUDA(cudaGraphMemcpyNodeGetParams(nodes[i], &params));
            out << " kind=" << static_cast<int>(params.kind);
        }
        std::vector<std::size_t> d;
        for (const auto j : deps[i]) d.push_back(position[j]);
        std::sort(d.begin(), d.end());
        out << " deps=";
        for (std::size_t k = 0; k < d.size(); ++k) out << (k ? "," : "") << d[k];
        out << '\n';
    }
    return out.str();
}

std::size_t product(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("scan frame session: allocation size overflow");
    return a * b;
}

std::size_t volume(const Dims& d) {
    std::size_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) n = product(n, static_cast<std::size_t>(d.d[i]));
    return n;
}
} // namespace

struct ScanFrameSession::Impl {
    ScanFrameConfig cfg;
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::vector<std::string> input_names, status_names;
    std::vector<std::size_t> input_count;
    std::string payload_name;
    std::size_t payload_count = 0;
    std::vector<Slot> slots;
    cudaStream_t d{}, m{};
    ScanFrameControl* ctl{};
    float* results{};
    std::int64_t* result_frames{};
    ScanFrameRecord* records{};
    unsigned char *rp{}, *hp{};
    ScanFrameControl* snap_ctl{};
    float *snap_results{}, *snap_payloads{};
    std::int64_t* snap_frames{};
    ScanFrameRecord* snap_records{};
    ScanStatus* snap_statuses{};
    std::vector<Event> pkt;
    Event d_end, m_end;
    std::vector<std::int64_t> pending;
    std::vector<void*> device_allocations, host_allocations;
    std::vector<cudaEvent_t> events;
    CleanupReport cleanup;
    bool drained = true, closed = false;
    std::uint64_t runs = 0;
    std::string identity;
    ScanFrameRun current; // the run in progress, or the last one (kept after an exception)
    ScanFrameRun* run = nullptr;

    ~Impl() {
        if (closed) return;
        if (!drained) {
            if (d) cudaStreamSynchronize(d);
            if (m) cudaStreamSynchronize(m);
        }
        release_resources();
    }

    template <typename T> T* allocate(std::size_t count) {
        void* p{};
        CUDA(cudaMalloc(&p, std::max<std::size_t>(product(count, sizeof(T)), 1)));
        try { device_allocations.push_back(p); }
        catch (...) { cudaFree(p); throw; }
        return static_cast<T*>(p);
    }
    template <typename T> T* pinned(std::size_t count) {
        void* p{};
        const auto bytes = std::max<std::size_t>(product(count, sizeof(T)), 1);
        CUDA(cudaMallocHost(&p, bytes));
        try { host_allocations.push_back(p); }
        catch (...) { cudaFreeHost(p); throw; }
        std::memset(p, 0, bytes);
        return static_cast<T*>(p);
    }
    Event event() {
        Event e;
        CUDA(cudaEventCreateWithFlags(&e.handle, cudaEventDisableTiming));
        try { events.push_back(e.handle); }
        catch (...) { cudaEventDestroy(e.handle); throw; }
        return e;
    }
    void log(const char* op, const std::string& resource, int slot, std::int64_t gen) {
        if (run) run->ledger.push_back(std::string(op) + ' ' + resource + " slot=" + std::to_string(slot) + " gen=" + std::to_string(gen));
    }
    void record(Event& e, cudaStream_t s, std::int64_t gen, const std::string& name, int slot) {
        CUDA(cudaEventRecord(e.handle, s));
        e.generation = gen;
        log("record", name, slot, gen);
    }
    void wait(Event& e, cudaStream_t s, std::int64_t expected, const std::string& name, int slot) {
        require(e.generation == expected, "event generation mismatch: " + name);
        CUDA(cudaStreamWaitEvent(s, e.handle, 0));
        log("wait", name, slot, expected);
    }
    void host_wait(Event& e, std::int64_t expected, const std::string& name, int slot) {
        require(e.generation == expected && expected >= 0, "host event generation mismatch: " + name);
        CUDA(cudaEventSynchronize(e.handle));
        log("host_wait", name, slot, expected);
    }

    void release_resources() noexcept {
        for (auto& s : slots) {
            cleanup.release(s.exec, [](cudaGraphExec_t h) { CUDA(cudaGraphExecDestroy(h)); });
            cleanup.release(s.graph, [](cudaGraph_t h) { CUDA(cudaGraphDestroy(h)); });
            s.context.reset(); // before its device memory is freed
        }
        engine.reset();
        runtime.reset();
        for (auto& s : slots) cleanup.release(s.context_memory, [](void* p) { CUDA(cudaFree(p)); });
        for (auto& p : device_allocations) cleanup.release(p, [](void* h) { CUDA(cudaFree(h)); });
        for (auto& p : host_allocations) cleanup.release(p, [](void* h) { CUDA(cudaFreeHost(h)); });
        for (auto& e : events) cleanup.release(e, [](cudaEvent_t h) { CUDA(cudaEventDestroy(h)); });
        cleanup.release(d, [](cudaStream_t h) { CUDA(cudaStreamDestroy(h)); });
        cleanup.release(m, [](cudaStream_t h) { CUDA(cudaStreamDestroy(h)); });
    }

    void init(std::span<const std::uint8_t> bytes, ILogger& logger, IErrorRecorder* recorder) {
        require(cfg.slots >= 1 && cfg.packet_slots >= 1 && cfg.max_frames >= 1, "slot counts and frame capacity must be positive");
        (void)product(static_cast<std::size_t>(cfg.max_frames), sizeof(ScanFrameRecord));
        drained = false; // construction can fail with initialization work queued
        CUDA(cudaSetDevice(0));
        CUDA(cudaStreamCreateWithFlags(&d, cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&m, cudaStreamNonBlocking));
        register_scan_trt_plugins(*getPluginRegistry());
        runtime.reset(createInferRuntime(logger));
        require(runtime != nullptr, "createInferRuntime failed");
        if (recorder) runtime->setErrorRecorder(recorder); // inherited by the engine and its contexts
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        require(engine != nullptr, "deserializeCudaEngine failed (are the scan plugins registered?)");
        require(engine->getNbAuxStreams() == 0, "the engine must not use auxiliary streams");
        for (std::int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            const auto dims = engine->getTensorShape(name);
            for (int k = 0; k < dims.nbDims; ++k) require(dims.d[k] >= 0, std::string("dynamic dimension in ") + name);
            require(engine->getTensorLocation(name) == TensorLocation::kDEVICE && engine->getTensorFormat(name) == TensorFormat::kLINEAR &&
                        !engine->isShapeInferenceIO(name),
                    std::string("not a linear device execution tensor: ") + name);
            const auto type = engine->getTensorDataType(name);
            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                require(type == DataType::kFLOAT, std::string("input is not FP32: ") + name);
                input_names.emplace_back(name);
                input_count.push_back(volume(dims));
            } else if (type == DataType::kFLOAT) {
                require(payload_name.empty(), "more than one FP32 output");
                payload_name = name;
                payload_count = volume(dims);
            } else {
                require(type == DataType::kINT32 && dims.nbDims == 1 && dims.d[0] == kScanStatusWords,
                        std::string("output is neither the FP32 payload nor an INT32[2] status: ") + name);
                status_names.emplace_back(name);
            }
        }
        require(!payload_name.empty() && !status_names.empty() && status_names.size() <= kScanFrameMaxStatuses,
                "the engine needs one FP32 payload and 1..4 status outputs");
        identity = scan_trt_plugin_identity() + describe_scan_engine(*engine) + scan_frame_load_kernels();

        const auto memory = engine->getDeviceMemorySizeV2();
        require(memory >= 0, "invalid execution context memory size");
        slots.resize(static_cast<std::size_t>(cfg.slots));
        for (std::size_t s = 0; s < slots.size(); ++s) {
            auto& slot = slots[s];
            // attachToContext runs here: each slot's plugins load the kernels in this CUDA context.
            slot.context.reset(engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
            require(slot.context != nullptr, "createExecutionContext failed for slot " + std::to_string(s));
            if (memory > 0) CUDA(cudaMalloc(&slot.context_memory, static_cast<std::size_t>(memory)));
            slot.context->setDeviceMemoryV2(slot.context_memory, memory);
            for (std::size_t i = 0; i < input_names.size(); ++i) {
                slot.inputs.push_back(allocate<float>(input_count[i]));
                slot.staging.push_back(pinned<float>(input_count[i]));
                require(slot.context->setTensorAddress(input_names[i].c_str(), slot.inputs.back()), "setTensorAddress " + input_names[i]);
            }
            slot.payload = allocate<float>(payload_count);
            require(slot.context->setTensorAddress(payload_name.c_str(), slot.payload), "setTensorAddress " + payload_name);
            for (const auto& name : status_names) {
                slot.statuses.push_back(allocate<ScanStatus>(1));
                require(slot.context->setTensorAddress(name.c_str(), slot.statuses.back()), "setTensorAddress " + name);
            }
            slot.staged = event();
            slot.ready = event();
            slot.consumed = event();
        }
        ctl = allocate<ScanFrameControl>(1);
        results = allocate<float>(product(2, payload_count));
        result_frames = allocate<std::int64_t>(2);
        records = allocate<ScanFrameRecord>(static_cast<std::size_t>(cfg.max_frames));
        rp = allocate<unsigned char>(packet_bytes);
        hp = pinned<unsigned char>(product(static_cast<std::size_t>(cfg.packet_slots), packet_bytes));
        snap_ctl = pinned<ScanFrameControl>(1);
        snap_results = pinned<float>(product(2, payload_count));
        snap_frames = pinned<std::int64_t>(2);
        snap_records = pinned<ScanFrameRecord>(static_cast<std::size_t>(cfg.max_frames));
        snap_payloads = pinned<float>(product(slots.size(), payload_count));
        snap_statuses = pinned<ScanStatus>(product(slots.size(), status_names.size()));
        for (int r = 0; r < cfg.packet_slots; ++r) pkt.push_back(event());
        d_end = event();
        m_end = event();

        // Warm every slot once with zero inputs: TensorRT's deferred setup and
        // the first launches happen here, not on the frame path. Then retire
        // the warmup statuses so no slot starts with a valid status.
        for (auto& slot : slots) {
            for (std::size_t i = 0; i < slot.inputs.size(); ++i)
                CUDA(cudaMemsetAsync(slot.inputs[i], 0, input_count[i] * sizeof(float), d));
            require(slot.context->enqueueV3(d), "warmup enqueueV3 failed");
        }
        for (auto& slot : slots)
            for (auto* status : slot.statuses) CUDA(cudaMemsetAsync(status, 0, sizeof(ScanStatus), d));
        CUDA(cudaStreamSynchronize(d));
        if (cfg.graphs) {
            for (std::size_t s = 0; s < slots.size(); ++s) {
                auto& slot = slots[s];
                // Global mode: an allocation, synchronization or legacy-stream
                // call inside enqueueV3 invalidates the capture.
                CUDA(cudaStreamBeginCapture(d, cudaStreamCaptureModeGlobal));
                const bool enqueued = slot.context->enqueueV3(d);
                const cudaError_t ended = cudaStreamEndCapture(d, &slot.graph);
                require(enqueued && ended == cudaSuccess && slot.graph != nullptr,
                        "graph capture of enqueueV3 failed for slot " + std::to_string(s) + ": " + cudaGetErrorString(ended));
                CUDA(cudaGraphInstantiate(&slot.exec, slot.graph, 0));
                CUDA(cudaGraphUpload(slot.exec, d));
                slot.nodes = inspect(slot.graph);
            }
        }
        CUDA(cudaDeviceSynchronize()); // A09: initialization complete before frame 0
        drained = true;
        std::ostringstream id;
        id << "slots=" << cfg.slots << "\npacket_slots=" << cfg.packet_slots << "\ngraphs=" << (cfg.graphs ? "true" : "false")
           << "\ncontext_memory_bytes=" << memory << "\npayload=" << payload_name << "\nstatuses=";
        for (std::size_t i = 0; i < status_names.size(); ++i) id << (i ? "," : "") << status_names[i];
        id << '\n';
        identity += id.str();
    }

    bool consume(int r, std::uint64_t epoch) {
        if (pending[static_cast<std::size_t>(r)] < 0) return false;
        const auto frame = pending[static_cast<std::size_t>(r)];
        host_wait(pkt[static_cast<std::size_t>(r)], frame, "pkt", r);
        const auto* bytes = hp + static_cast<std::size_t>(r) * packet_bytes;
        run->packets.insert(run->packets.end(), bytes, bytes + packet_bytes);
        ScanFrameOutcome outcome;
        outcome.frame = frame;
        try {
            const auto packet = decode_packet({bytes, packet_bytes});
            require(packet.epoch == epoch && packet.frame == frame, "packet identity differs from the submitted frame");
            outcome.status = packet.status;
            outcome.frame_error = packet.frame_error;
            outcome.fail_code = packet.fail_code;
            // Publication rule: only an OK packet of this exact frame is a result.
            outcome.accepted = packet.status == ResultStatus::ok;
        } catch (const std::exception& error) {
            run->ledger.push_back("reject frame=" + std::to_string(frame) + " malformed packet: " + error.what());
        }
        if (outcome.accepted) run->published.push_back(frame);
        log(outcome.accepted ? "accept" : "reject", "pkt", r, frame);
        run->outcomes.push_back(outcome);
        pending[static_cast<std::size_t>(r)] = -1;
        return outcome.status == ResultStatus::failed;
    }

    ScanStatusList status_list(const Slot& slot) const {
        ScanStatusList list{};
        for (std::size_t i = 0; i < slot.statuses.size(); ++i) list.status[i] = slot.statuses[i];
        list.count = static_cast<unsigned>(slot.statuses.size());
        return list;
    }

    ScanFrameSlotView view(const Slot& slot) const { return {slot.inputs, slot.payload, slot.statuses}; }

    ScanFrameRun execute(const std::vector<std::vector<std::vector<float>>>& frames, const ScanFrameHook& hook) {
        require(!closed, "session is closed");
        require(!frames.empty() && static_cast<std::int64_t>(frames.size()) <= cfg.max_frames, "frame count outside 1..max_frames");
        for (const auto& f : frames) {
            require(f.size() == input_names.size(), "frame input count differs from the engine");
            for (std::size_t i = 0; i < f.size(); ++i) require(f[i].size() == input_count[i], "frame input size differs: " + input_names[i]);
        }
        current = ScanFrameRun{};
        run = &current;
        auto& result = current;
        const std::uint64_t epoch = ++runs;
        result.epoch = epoch;
        // Quiescent reset (ASYNC_CONTROL §6): the previous run is drained.
        drained = false;
        scan_frame_initialize(ctl, epoch, result_frames, m);
        CUDA(cudaMemsetAsync(records, 0, static_cast<std::size_t>(cfg.max_frames) * sizeof(ScanFrameRecord), m));
        CUDA(cudaMemsetAsync(results, 0, 2 * payload_count * sizeof(float), m));
        CUDA(cudaDeviceSynchronize());
        for (auto& slot : slots) slot.staged.generation = slot.ready.generation = slot.consumed.generation = -1;
        for (auto& e : pkt) e.generation = -1;
        pending.assign(static_cast<std::size_t>(cfg.packet_slots), -1);
        const auto S = static_cast<std::int64_t>(slots.size());
        for (std::int64_t k = 0; k < static_cast<std::int64_t>(frames.size()); ++k) {
            const int s = static_cast<int>(k % S), r = static_cast<int>(k % cfg.packet_slots);
            auto& slot = slots[static_cast<std::size_t>(s)];
            if (consume(r, epoch) && !cfg.fixed_submit) {
                result.stopped_on_failure = true;
                break;
            }
            if (k >= S) host_wait(slot.staged, k - S, "staged", s);
            for (std::size_t i = 0; i < slot.staging.size(); ++i)
                std::memcpy(slot.staging[i], frames[static_cast<std::size_t>(k)][i].data(), input_count[i] * sizeof(float));
            // ---- D: the slot's previous frame must be consumed by M before it is overwritten.
            if (k >= S) wait(slot.consumed, d, k - S, "consumed", s);
            for (std::size_t i = 0; i < slot.inputs.size(); ++i)
                CUDA(cudaMemcpyAsync(slot.inputs[i], slot.staging[i], input_count[i] * sizeof(float), cudaMemcpyHostToDevice, d));
            record(slot.staged, d, k, "staged", s);
            if (cfg.graphs) CUDA(cudaGraphLaunch(slot.exec, d));
            else if (!slot.context->enqueueV3(d)) {
                // Host-detected contract or launch error: no ready record, no M
                // work and no packet for this frame; the run stops (§6 host error).
                result.ledger.push_back("host_failure frame=" + std::to_string(k) + " enqueueV3 returned false");
                result.stopped_on_failure = true;
                break;
            }
            log("engine", cfg.graphs ? "graph" : "enqueueV3", s, k);
            if (hook) hook(k, view(slot), d);
            record(slot.ready, d, k, "ready", s);
            ++result.submitted;
            // ---- M: frame end.
            scan_frame_begin(ctl, epoch, k, m);
            wait(slot.ready, m, k, "ready", s);
            if (cfg.gate_delay_cycles) scan_frame_delay(cfg.gate_delay_cycles, m);
            scan_frame_gate(ctl, status_list(slot), records, cfg.max_frames, m);
            scan_frame_publish(ctl, slot.payload, payload_count, results, result_frames, records, cfg.max_frames, m);
            record(slot.consumed, m, k, "consumed", s); // last M read of the slot
            scan_frame_commit(ctl, m);
            scan_frame_packet(ctl, rp, m);
            CUDA(cudaMemcpyAsync(hp + static_cast<std::size_t>(r) * packet_bytes, rp, packet_bytes, cudaMemcpyDeviceToHost, m));
            record(pkt[static_cast<std::size_t>(r)], m, k, "pkt", r);
            pending[static_cast<std::size_t>(r)] = k;
        }
        // Consume every queued packet in frame order, then drain both streams.
        while (true) {
            const auto next = std::min_element(pending.begin(), pending.end(), [](auto a, auto b) { return a >= 0 && (b < 0 || a < b); });
            if (next == pending.end() || *next < 0) break;
            if (consume(static_cast<int>(next - pending.begin()), epoch) && !cfg.fixed_submit) result.stopped_on_failure = true;
        }
        record(d_end, d, result.submitted, "d_terminal", 0);
        record(m_end, m, result.submitted, "m_terminal", 0);
        host_wait(d_end, result.submitted, "d_terminal", 0);
        host_wait(m_end, result.submitted, "m_terminal", 0);
        snapshot(result);
        drained = true;
        run = nullptr;
        return result;
    }

    // Test-only observation after both terminal drains.
    void snapshot(ScanFrameRun& result) {
        const auto n = static_cast<std::size_t>(std::max<std::int64_t>(result.submitted, 0));
        CUDA(cudaMemcpyAsync(snap_ctl, ctl, sizeof(ScanFrameControl), cudaMemcpyDeviceToHost, m));
        CUDA(cudaMemcpyAsync(snap_results, results, 2 * payload_count * sizeof(float), cudaMemcpyDeviceToHost, m));
        CUDA(cudaMemcpyAsync(snap_frames, result_frames, 2 * sizeof(std::int64_t), cudaMemcpyDeviceToHost, m));
        if (n) CUDA(cudaMemcpyAsync(snap_records, records, n * sizeof(ScanFrameRecord), cudaMemcpyDeviceToHost, m));
        for (std::size_t s = 0; s < slots.size(); ++s) {
            CUDA(cudaMemcpyAsync(snap_payloads + s * payload_count, slots[s].payload, payload_count * sizeof(float), cudaMemcpyDeviceToHost, m));
            for (std::size_t i = 0; i < status_names.size(); ++i)
                CUDA(cudaMemcpyAsync(snap_statuses + s * status_names.size() + i, slots[s].statuses[i], sizeof(ScanStatus),
                                     cudaMemcpyDeviceToHost, m));
        }
        CUDA(cudaStreamSynchronize(m));
        result.control = *snap_ctl;
        const auto c = snap_ctl->committed_slot;
        result.committed_payload.assign(snap_results + c * payload_count, snap_results + (c + 1) * payload_count);
        result.committed_payload_frame = snap_frames[c];
        result.records.assign(snap_records, snap_records + n);
        for (std::size_t s = 0; s < slots.size(); ++s) {
            result.slot_payloads.emplace_back(snap_payloads + s * payload_count, snap_payloads + (s + 1) * payload_count);
            result.slot_statuses.emplace_back(snap_statuses + s * status_names.size(), snap_statuses + (s + 1) * status_names.size());
        }
        result.ledger.push_back("terminal_snapshot after_both_drains");
    }
};

ScanFrameSession::ScanFrameSession(std::span<const std::uint8_t> engine, const ScanFrameConfig& config, ILogger& logger,
                                   IErrorRecorder* recorder)
    : impl_(std::make_unique<Impl>()) {
    impl_->cfg = config;
    impl_->init(engine, logger, recorder);
}

ScanFrameSession::~ScanFrameSession() = default;

const std::vector<std::string>& ScanFrameSession::input_names() const { return impl_->input_names; }
std::size_t ScanFrameSession::input_elements(std::size_t input) const { return impl_->input_count.at(input); }
std::size_t ScanFrameSession::payload_elements() const { return impl_->payload_count; }
std::string ScanFrameSession::identity() const { return impl_->identity; }
std::string ScanFrameSession::graph_nodes(int slot) const { return impl_->slots.at(static_cast<std::size_t>(slot)).nodes; }
IExecutionContext& ScanFrameSession::context(int slot) { return *impl_->slots.at(static_cast<std::size_t>(slot)).context; }
ScanFrameSlotView ScanFrameSession::slot(int slot) const { return impl_->view(impl_->slots.at(static_cast<std::size_t>(slot))); }
cudaStream_t ScanFrameSession::detect_stream() const { return impl_->d; }

ScanFrameRun ScanFrameSession::run(const std::vector<std::vector<std::vector<float>>>& frames, const ScanFrameHook& hook) {
    return impl_->execute(frames, hook);
}
const ScanFrameRun& ScanFrameSession::last_run() const { return impl_->current; }

void ScanFrameSession::close() {
    if (impl_->closed) return;
    require(impl_->drained, "close requires both terminal drains");
    impl_->release_resources();
    impl_->closed = true;
    impl_->cleanup.throw_if_failed();
}

} // namespace saccade

#include "saccade/scan_trt_frame.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/scan_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"

#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <set>
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
    float* input_arena = nullptr; // every input at its aligned offset
    std::vector<float*> inputs;
    std::vector<float*> staging; // pinned host, one per input
    float* payload = nullptr;    // output arena: every output at its aligned offset
    std::vector<float*> outputs;
    std::vector<ScanStatus*> statuses; // engine statuses, then harness check statuses
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
// indices of the nodes it depends on; dependencies that are not ordinary
// completion edges (e.g. programmatic launch edges) are listed separately.
std::string inspect(cudaGraph_t graph) {
    std::size_t count = 0, edges = 0;
    CUDA(cudaGraphGetNodes(graph, nullptr, &count));
    std::vector<cudaGraphNode_t> nodes(count);
    CUDA(cudaGraphGetNodes(graph, nodes.data(), &count));
    CUDA(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges));
    std::vector<cudaGraphNode_t> from(edges), to(edges);
    std::vector<cudaGraphEdgeData> data(edges);
    // Edge data is required: TensorRT graphs can carry typed edges, and
    // reading them without it is refused as lossy.
    if (edges) CUDA(cudaGraphGetEdges(graph, from.data(), to.data(), data.data(), &edges));
    std::map<cudaGraphNode_t, std::size_t> index;
    for (std::size_t i = 0; i < count; ++i) index[nodes[i]] = i;
    std::vector<std::vector<std::size_t>> deps(count);
    std::vector<std::vector<std::pair<std::size_t, std::string>>> typed(count); // upstream node, type and ports
    std::vector<std::size_t> indegree(count);
    for (std::size_t e = 0; e < edges; ++e) {
        deps[index.at(to[e])].push_back(index.at(from[e]));
        ++indegree[index.at(to[e])];
        const auto& d = data[e];
        if (d.type != cudaGraphDependencyTypeDefault || d.from_port != 0 || d.to_port != 0)
            typed[index.at(to[e])].emplace_back(index.at(from[e]), ":type" + std::to_string(d.type) + ":port" + std::to_string(d.from_port) + ">" +
                                                                      std::to_string(d.to_port));
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
        if (!typed[i].empty()) {
            out << " typed_edges=";
            for (std::size_t k = 0; k < typed[i].size(); ++k) out << (k ? "," : "") << position[typed[i][k].first] << typed[i][k].second;
        }
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

bool same_shape(const Dims& d, const std::vector<std::int64_t>& shape) {
    if (d.nbDims != static_cast<int>(shape.size())) return false;
    for (int i = 0; i < d.nbDims; ++i)
        if (d.d[i] != shape[static_cast<std::size_t>(i)]) return false;
    return true;
}

// Tensors share an arena at 256-byte (64-float) aligned element offsets.
constexpr std::size_t kArenaAlign = 64;
std::size_t align_up(std::size_t n) { return (n + kArenaAlign - 1) / kArenaAlign * kArenaAlign; }
} // namespace

struct ScanFrameSession::Impl {
    ScanFrameConfig cfg;
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::vector<std::string> input_names;
    std::vector<std::size_t> input_count, input_offset;
    std::size_t input_arena_count = 0;
    std::vector<ScanFrameSession::OutputSlot> outputs;
    std::size_t payload_count = 0; // output arena elements
    std::vector<FrameStatusSpec> statuses; // gate order: engine statuses, then harness checks
    std::size_t engine_statuses = 0;
    bool check_inputs = false, check_outputs = false;
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

    // S03 engines: one FP32 output, 1..4 INT32[2] statuses, inferred.
    void infer_io() {
        std::string payload_name;
        for (std::int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            const auto dims = engine->getTensorShape(name);
            const auto type = engine->getTensorDataType(name);
            if (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) {
                require(type == DataType::kFLOAT, std::string("input is not FP32: ") + name);
                input_names.emplace_back(name);
                input_count.push_back(volume(dims));
            } else if (type == DataType::kFLOAT) {
                require(payload_name.empty(), "more than one FP32 output");
                payload_name = name;
                outputs.push_back({name, 0, volume(dims)});
            } else {
                require(type == DataType::kINT32 && dims.nbDims == 1 && dims.d[0] == kScanStatusWords,
                        std::string("output is neither the FP32 payload nor an INT32[2] status: ") + name);
                statuses.push_back({name, "engine"});
            }
        }
        require(!payload_name.empty() && !statuses.empty() && statuses.size() <= 4, "the engine needs one FP32 payload and 1..4 status outputs");
    }

    // S04: the engine must have exactly the spec's tensors.
    void spec_io(const FrameIoSpec& spec) {
        std::map<std::string, std::string> wanted; // name -> role
        const auto add = [&](const std::string& name, const char* role) {
            require(!name.empty() && wanted.emplace(name, role).second, "I/O spec names a tensor twice or with an empty name: " + name);
        };
        for (const auto& t : spec.inputs) add(t.name, "input");
        for (const auto& t : spec.outputs) add(t.name, "output");
        for (const auto& s : spec.statuses) add(s.name, "status");
        require(!spec.inputs.empty() && !spec.outputs.empty() && !spec.statuses.empty(), "I/O spec needs inputs, outputs and statuses");
        std::set<std::string> seen;
        for (std::int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
            const std::string name = engine->getIOTensorName(i);
            const auto it = wanted.find(name);
            require(it != wanted.end(), "engine tensor not in the I/O spec: " + name);
            require(seen.insert(name).second, "engine names a tensor twice: " + name);
            const bool input = engine->getTensorIOMode(name.c_str()) == TensorIOMode::kINPUT;
            const auto type = engine->getTensorDataType(name.c_str());
            require(input == (it->second == "input"), "engine tensor has the wrong I/O mode: " + name);
            require(type == (it->second == "status" ? DataType::kINT32 : DataType::kFLOAT), "engine tensor has the wrong data type: " + name);
        }
        for (const auto& [name, role] : wanted) require(seen.contains(name), "engine lacks the " + role + " " + name);
        for (const auto& t : spec.inputs) {
            require(same_shape(engine->getTensorShape(t.name.c_str()), t.shape), "engine input shape differs from the spec: " + t.name);
            input_names.push_back(t.name);
            input_count.push_back(volume(engine->getTensorShape(t.name.c_str())));
        }
        std::size_t offset = 0;
        for (const auto& t : spec.outputs) {
            require(same_shape(engine->getTensorShape(t.name.c_str()), t.shape), "engine output shape differs from the spec: " + t.name);
            const auto count = volume(engine->getTensorShape(t.name.c_str()));
            outputs.push_back({t.name, offset, count});
            offset = align_up(offset + count);
        }
        for (const auto& s : spec.statuses) {
            const auto dims = engine->getTensorShape(s.name.c_str());
            require(dims.nbDims == 1 && dims.d[0] == kScanStatusWords, "engine status is not INT32[2]: " + s.name);
            statuses.push_back(s);
        }
        check_inputs = spec.check_inputs;
        check_outputs = spec.check_outputs;
    }

    void init(std::span<const std::uint8_t> bytes, ILogger& logger, IErrorRecorder* recorder, const FrameIoSpec* spec) {
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
        }
        if (spec) spec_io(*spec);
        else infer_io();
        engine_statuses = statuses.size();
        if (check_inputs) statuses.push_back({kFrameInputCheck, "harness: finite check of every input element"});
        if (check_outputs) statuses.push_back({kFrameOutputCheck, "harness: finite check of every output element"});
        require(statuses.size() <= kScanFrameMaxStatuses, "more statuses than the gate reads");
        for (const auto& o : outputs) payload_count = std::max(payload_count, o.offset + o.count);
        for (const auto count : input_count) {
            input_offset.push_back(input_arena_count);
            input_arena_count = align_up(input_arena_count + count);
        }
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
            slot.input_arena = allocate<float>(input_arena_count);
            CUDA(cudaMemset(slot.input_arena, 0, std::max<std::size_t>(input_arena_count, 1) * sizeof(float))); // padding stays finite
            for (std::size_t i = 0; i < input_names.size(); ++i) {
                slot.inputs.push_back(slot.input_arena + input_offset[i]);
                slot.staging.push_back(pinned<float>(input_count[i]));
                require(slot.context->setTensorAddress(input_names[i].c_str(), slot.inputs.back()), "setTensorAddress " + input_names[i]);
            }
            slot.payload = allocate<float>(payload_count);
            CUDA(cudaMemset(slot.payload, 0, std::max<std::size_t>(payload_count, 1) * sizeof(float)));
            for (const auto& o : outputs) {
                slot.outputs.push_back(slot.payload + o.offset);
                require(slot.context->setTensorAddress(o.name.c_str(), slot.outputs.back()), "setTensorAddress " + o.name);
            }
            for (std::size_t i = 0; i < statuses.size(); ++i) {
                slot.statuses.push_back(allocate<ScanStatus>(1));
                if (i < engine_statuses)
                    require(slot.context->setTensorAddress(statuses[i].name.c_str(), slot.statuses.back()), "setTensorAddress " + statuses[i].name);
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
        snap_statuses = pinned<ScanStatus>(product(slots.size(), statuses.size()));
        for (int r = 0; r < cfg.packet_slots; ++r) pkt.push_back(event());
        d_end = event();
        m_end = event();

        // Warm every slot once with zero inputs: TensorRT's deferred setup and
        // the first launches happen here, not on the frame path. Then retire
        // the warmup statuses so no slot starts with a valid status.
        for (auto& slot : slots) {
            CUDA(cudaMemsetAsync(slot.input_arena, 0, input_arena_count * sizeof(float), d));
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
           << "\ncontext_memory_bytes=" << memory << "\npayload=";
        for (std::size_t i = 0; i < outputs.size(); ++i) id << (i ? "," : "") << outputs[i].name;
        id << "\nstatuses=";
        for (std::size_t i = 0; i < statuses.size(); ++i) id << (i ? "," : "") << statuses[i].name;
        id << '\n';
        if (spec) {
            for (const auto& o : outputs) id << "output=" << o.name << " arena_offset=" << o.offset << " elements=" << o.count << '\n';
            for (const auto& st : statuses) id << "status=" << st.name << " producer=" << st.producer << '\n';
        }
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

    ScanFrameSlotView view(const Slot& slot) const { return {slot.inputs, slot.payload, slot.outputs, slot.statuses}; }

    ScanFrameRun execute(const std::vector<std::vector<std::vector<float>>>& frames, const ScanFrameHook& hook, const ScanFrameHook& after_checks) {
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
            if (check_inputs) scan_frame_check_finite(slot.input_arena, input_arena_count, slot.statuses[engine_statuses], kFrameNonFiniteInput, d);
            if (check_outputs)
                scan_frame_check_finite(slot.payload, payload_count, slot.statuses[engine_statuses + (check_inputs ? 1 : 0)], kFrameNonFiniteOutput, d);
            if (after_checks) after_checks(k, view(slot), d);
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
            for (std::size_t i = 0; i < statuses.size(); ++i)
                CUDA(cudaMemcpyAsync(snap_statuses + s * statuses.size() + i, slots[s].statuses[i], sizeof(ScanStatus),
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
            result.slot_statuses.emplace_back(snap_statuses + s * statuses.size(), snap_statuses + (s + 1) * statuses.size());
        }
        result.ledger.push_back("terminal_snapshot after_both_drains");
    }
};

ScanFrameSession::ScanFrameSession(std::span<const std::uint8_t> engine, const ScanFrameConfig& config, ILogger& logger,
                                   IErrorRecorder* recorder, const FrameIoSpec* spec)
    : impl_(std::make_unique<Impl>()) {
    impl_->cfg = config;
    impl_->init(engine, logger, recorder, spec);
}

ScanFrameSession::~ScanFrameSession() = default;

const std::vector<std::string>& ScanFrameSession::input_names() const { return impl_->input_names; }
std::size_t ScanFrameSession::input_elements(std::size_t input) const { return impl_->input_count.at(input); }
std::size_t ScanFrameSession::payload_elements() const { return impl_->payload_count; }
const std::vector<ScanFrameSession::OutputSlot>& ScanFrameSession::outputs() const { return impl_->outputs; }
const std::vector<FrameStatusSpec>& ScanFrameSession::statuses() const { return impl_->statuses; }
std::string ScanFrameSession::identity() const { return impl_->identity; }
std::string ScanFrameSession::graph_nodes(int slot) const { return impl_->slots.at(static_cast<std::size_t>(slot)).nodes; }
IExecutionContext& ScanFrameSession::context(int slot) { return *impl_->slots.at(static_cast<std::size_t>(slot)).context; }
ScanFrameSlotView ScanFrameSession::slot(int slot) const { return impl_->view(impl_->slots.at(static_cast<std::size_t>(slot))); }
cudaStream_t ScanFrameSession::detect_stream() const { return impl_->d; }

ScanFrameRun ScanFrameSession::run(const std::vector<std::vector<std::vector<float>>>& frames, const ScanFrameHook& hook,
                                   const ScanFrameHook& after_checks) {
    return impl_->execute(frames, hook, after_checks);
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

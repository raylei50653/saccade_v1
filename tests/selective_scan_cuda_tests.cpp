#include "scan_cuda_suite.hpp"
#include "saccade/file_io.hpp"

#include <condition_variable>
#include <mutex>

// S02 acceptance for the CUDA FP32 forward (docs/SCAN_CUDA.md): the shared
// groups of scan_cuda_suite.hpp through the CUDA API itself, plus the checks
// that are specific to calling it directly: reset on reused buffers, the
// caller's stream, graph capture and host rejection.
namespace {
using namespace saccade::scan_test;

// initcheck makes the enqueue wait on a stream held by a host callback, so its
// run skips that one test (--no-held-stream); every other tool runs it.
bool held_stream_test = true;

void reset_and_determinism(cudaStream_t s) {
    const SelectiveScanShape shape{2, 17, 9, 8, 9, 8, true};
    const Case first = random_case(shape, 0x5eed0200);
    Case broken = first;
    broken.delta[40] = std::numeric_limits<float>::quiet_NaN();
    const Case other = random_case(shape, 0x5eed0201, {50, -2, 6, -0.5f, -0.01f, 3, 2});
    ScanBuffers buffers(shape);
    // Outputs and status are reset only once; each call must clear its own.
    buffers.load(first, s);
    buffers.enqueue(s);
    const auto a1 = buffers.result(s);
    expect_valid(a1.status, "reset_first");
    compare("reset_first", a1.y, first.run(), scan_bound(first));
    buffers.load(broken, s, false);
    buffers.enqueue(s);
    const auto failed = buffers.result(s);
    check(failed.status.done == kScanDone && (failed.status.flags & saccade::kScanInputFailures) == saccade::kScanNonFiniteDelta,
          "reset: middle call fails");
    expect_poisoned(failed.y, "reset_middle_failure");
    buffers.load(first, s, false);
    buffers.enqueue(s);
    const auto a2 = buffers.result(s);
    expect_valid(a2.status, "reset_after_failure");
    check(std::memcmp(a1.y.data(), a2.y.data(), a1.y.size() * sizeof(float)) == 0, "A -> failing B -> A: identical output bits");
    buffers.load(other, s, false);
    buffers.enqueue(s);
    const auto b = buffers.result(s);
    expect_valid(b.status, "reset_other");
    compare("reset_other_valid", b.y, other.run(), scan_bound(other));
    buffers.load(first, s, false);
    buffers.enqueue(s);
    const auto a3 = buffers.result(s);
    expect_valid(a3.status, "reset_after_other");
    check(std::memcmp(a1.y.data(), a3.y.data(), a1.y.size() * sizeof(float)) == 0, "A -> valid B -> A: identical output bits");
    report << "reset\tA_B(fail)_A_B(valid)_A\tPASS\n";

    Case m = random_case({4, 400, 256, 16, 1, 1, true}, 0x5eed0202, {3, -5, 2, -1, -1, 2, 1});
    for (std::size_t n = 0; n < 16; ++n) m.a[n] = -static_cast<float>(n + 1);
    const auto r1 = run_gpu(m, s), r2 = run_gpu(m, s);
    check(std::memcmp(r1.y.data(), r2.y.data(), r1.y.size() * sizeof(float)) == 0, "repeat on fresh buffers: identical output bits");
    report << "determinism\tm_head_L400_repeat\tPASS\n";
}

// Stalls a stream inside a host callback until released; the callback makes no CUDA calls.
struct Gate {
    std::mutex mutex;
    std::condition_variable opened;
    bool open = false;
    static void CUDART_CB wait(void* self) {
        auto* gate = static_cast<Gate*>(self);
        std::unique_lock lock(gate->mutex);
        gate->opened.wait(lock, [&] { return gate->open; });
    }
    void release() {
        { std::lock_guard lock(mutex); open = true; }
        opened.notify_all();
    }
};
struct GateGuard { Gate& gate; ~GateGuard() { gate.release(); } };

template <typename T> class PinnedCopy {
public:
    explicit PinnedCopy(const std::vector<T>& v) : count_(v.size()) {
        if (count_) { CUDA(cudaMallocHost(reinterpret_cast<void**>(&data_), count_ * sizeof(T))); std::memcpy(data_, v.data(), count_ * sizeof(T)); }
    }
    ~PinnedCopy() { if (data_) cudaFreeHost(data_); }
    PinnedCopy(const PinnedCopy&) = delete;
    PinnedCopy& operator=(const PinnedCopy&) = delete;
    void copy_to(const DeviceBuffer<T>& device, cudaStream_t s) const {
        if (count_) CUDA(cudaMemcpyAsync(device.get(), data_, count_ * sizeof(T), cudaMemcpyHostToDevice, s));
    }
private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

void streams(cudaStream_t s) {
    // 1. Work goes to the caller's stream: while the stream is held, the scan has
    //    not completed; after release, the output reflects inputs written only
    //    after release. Device inputs start as NaN so a scan placed on any other
    //    stream reads them and cannot produce the expected result.
    if (!held_stream_test) report << "skipped\tstream_gate\t--no-held-stream\n";
    else {
        const Case k = random_case({2, 9, 33, 8, 1, 8, true}, 0x5eed0300);
        ScanBuffers buffers(k.shape);
        for (const auto* t : {&buffers.u, &buffers.delta, &buffers.a, &buffers.b, &buffers.c, &buffers.d})
            if (t->size()) CUDA(cudaMemsetAsync(t->get(), 0xFF, t->bytes(), s));
        CUDA(cudaMemsetAsync(buffers.y.get(), 0xA5, buffers.y.bytes(), s));
        put(buffers.status, {kStatusSentinel}, s);
        CUDA(cudaStreamSynchronize(s));
        const PinnedCopy u(k.u), delta(k.delta), a(k.a), b(k.b), c(k.c), d(k.d);
        cudaEvent_t done{};
        CUDA(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        Gate gate;
        cudaError_t held = cudaSuccess;
        {
            GateGuard guard{gate};
            CUDA(cudaLaunchHostFunc(s, Gate::wait, &gate));
            u.copy_to(buffers.u, s); delta.copy_to(buffers.delta, s); a.copy_to(buffers.a, s);
            b.copy_to(buffers.b, s); c.copy_to(buffers.c, s); d.copy_to(buffers.d, s);
            buffers.enqueue(s);
            CUDA(cudaEventRecord(done, s));
            // Anything wrongly put on the legacy stream finishes here; s is non-blocking.
            CUDA(cudaStreamSynchronize(cudaStreamLegacy));
            held = cudaEventQuery(done);
            if (held == cudaErrorNotReady && cudaPeekAtLastError() == cudaErrorNotReady) (void)cudaGetLastError();
        }
        check(held == cudaErrorNotReady, "scan completion waits for the held caller stream");
        CUDA(cudaEventSynchronize(done));
        CUDA(cudaEventDestroy(done));
        const auto r = buffers.result(s);
        expect_valid(r.status, "stream_gate");
        compare("stream_gate", r.y, k.run(), scan_bound(k));
    }
    // 2. Global-mode capture rejects implicit synchronization, allocation and
    //    legacy-stream work; the graph must hold exactly the call's four nodes.
    {
        Case k1 = random_case({4, 100, 256, 16, 1, 1, true}, 0x5eed0301, {3, -5, 2, -1, -1, 2, 1});
        for (std::size_t n = 0; n < 16; ++n) k1.a[n] = -static_cast<float>(n + 1);
        Case k2 = random_case(k1.shape, 0x5eed0302, {3, -5, 2, -1, -1, 2, 1});
        k2.a = k1.a;
        k2.a[3] = -0.25f; // A is read from its input, not assumed frozen
        ScanBuffers buffers(k1.shape);
        buffers.load(k1, s);
        buffers.enqueue(s);
        const auto direct = buffers.result(s);
        expect_valid(direct.status, "graph_direct");
        cudaGraph_t graph{};
        CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        try { buffers.enqueue(s); }
        catch (...) {
            cudaGraph_t partial{};
            cudaStreamEndCapture(s, &partial);
            if (partial) cudaGraphDestroy(partial);
            throw;
        }
        CUDA(cudaStreamEndCapture(s, &graph));
        std::size_t count = 0;
        CUDA(cudaGraphGetNodes(graph, nullptr, &count));
        std::vector<cudaGraphNode_t> nodes(count);
        CUDA(cudaGraphGetNodes(graph, nodes.data(), &count));
        std::size_t kernels = 0, memsets = 0, other = 0;
        for (const auto node : nodes) {
            cudaGraphNodeType type{};
            CUDA(cudaGraphNodeGetType(node, &type));
            kernels += type == cudaGraphNodeTypeKernel;
            memsets += type == cudaGraphNodeTypeMemset;
            other += type != cudaGraphNodeTypeKernel && type != cudaGraphNodeTypeMemset;
        }
        check(memsets == 1 && kernels == 3 && other == 0, "captured call is one memset and three kernels, no copies or host nodes");
        cudaGraphExec_t exec{};
        CUDA(cudaGraphInstantiate(&exec, graph, 0));
        buffers.load(k1, s);
        CUDA(cudaGraphLaunch(exec, s));
        const auto replay1 = buffers.result(s);
        expect_valid(replay1.status, "graph_replay_1");
        check(std::memcmp(direct.y.data(), replay1.y.data(), direct.y.size() * sizeof(float)) == 0, "graph replay equals the direct call bit for bit");
        buffers.load(k2, s);
        CUDA(cudaGraphLaunch(exec, s));
        const auto replay2 = buffers.result(s);
        expect_valid(replay2.status, "graph_replay_2");
        compare("graph_replay_new_inputs", replay2.y, k2.run(), scan_bound(k2));
        CUDA(cudaGraphExecDestroy(exec));
        CUDA(cudaGraphDestroy(graph));
        report << "graph\tnodes=memset:" << memsets << ",kernel:" << kernels << ",other:" << other << "\tPASS\n";
    }
    // 3. Two calls in flight on two streams share no hidden state.
    {
        const Stream other;
        const Case k1 = random_case({4, 100, 64, 16, 1, 1, true}, 0x5eed0303);
        const Case k2 = random_case({3, 50, 40, 8, 40, 8, false}, 0x5eed0304);
        ScanBuffers b1(k1.shape), b2(k2.shape);
        b1.load(k1, s);
        b2.load(k2, other.get());
        CUDA(cudaStreamSynchronize(s));
        CUDA(cudaStreamSynchronize(other.get()));
        for (int i = 0; i < 3; ++i) { b1.enqueue(s); b2.enqueue(other.get()); }
        const auto r1 = b1.result(s), r2 = b2.result(other.get());
        expect_valid(r1.status, "concurrent_1");
        expect_valid(r2.status, "concurrent_2");
        compare("concurrent_stream_1", r1.y, k1.run(), scan_bound(k1));
        compare("concurrent_stream_2", r2.y, k2.run(), scan_bound(k2));
    }
}

// The call must reject on the host and enqueue nothing: a status sentinel stays intact.
template <typename F> void host_rejects(const DeviceBuffer<ScanStatus>& status, cudaStream_t s, F&& call, std::string_view diagnostic) {
    const ScanStatus sentinel{0xDEADBEEFu, 0x0BADF00Du};
    put(status, {sentinel}, s);
    CUDA(cudaStreamSynchronize(s));
    expect_throw<std::invalid_argument>(call, diagnostic);
    const auto after = fetch(status, s)[0];
    check(after.flags == sentinel.flags && after.done == sentinel.done, "rejected call enqueued nothing: " + std::string(diagnostic));
}

void host_rejections(cudaStream_t s) {
    const Case k = varied({1, 4, 4, 16, 1, 1, true});
    ScanBuffers buf(k.shape);
    buf.load(k, s);
    const auto in = buf.inputs();
    auto call = [&](SelectiveScanShape shape, saccade::SelectiveScanDeviceInputs inputs, float* y, ScanStatus* st, cudaStream_t stream) {
        return [=] { saccade::selective_scan_cuda(shape, inputs, y, st, stream); };
    };
    const auto& st = buf.status;
    for (const std::size_t n : {0u, 3u, 64u}) {
        auto shape = k.shape;
        shape.state = n;
        host_rejects(st, s, call(shape, in, buf.y.get(), st.get(), s), "must be a power of two in [1, 32]");
    }
    { auto shape = k.shape; shape.a_rows = 2; host_rejects(st, s, call(shape, in, buf.y.get(), st.get(), s), "A rows must be 1 or channels"); }
    { auto shape = k.shape; shape.c_width = 4; host_rejects(st, s, call(shape, in, buf.y.get(), st.get(), s), "C width must be 1 or state"); }
    { auto shape = k.shape; shape.batch = 0; host_rejects(st, s, call(shape, in, buf.y.get(), st.get(), s), "batch and channels must be positive"); }
    { auto shape = k.shape; shape.channels = 0; host_rejects(st, s, call(shape, in, buf.y.get(), st.get(), s), "batch and channels must be positive"); }
    host_rejects(st, s, call({1, 65536, 65536, 1, 1, 1, false}, {}, nullptr, st.get(), s), "u/delta/y element count exceeds int32 capacity");
    host_rejects(st, s, call({1, std::size_t{1} << 27, 1, 32, 1, 1, false}, {}, nullptr, st.get(), s), "B element count exceeds int32 capacity");
    host_rejects(st, s, call({2, std::numeric_limits<std::size_t>::max() / 2, 1, 1, 1, 1, false}, {}, nullptr, st.get(), s), "exceeds int32 capacity");
    { auto bad = in; bad.u = nullptr; host_rejects(st, s, call(k.shape, bad, buf.y.get(), st.get(), s), "u is null"); }
    { auto bad = in; bad.d = nullptr; host_rejects(st, s, call(k.shape, bad, buf.y.get(), st.get(), s), "D is null"); }
    {
        auto bad = in;
        bad.delta = reinterpret_cast<const float*>(reinterpret_cast<const char*>(in.delta) + 2);
        host_rejects(st, s, call(k.shape, bad, buf.y.get(), st.get(), s), "delta is not 4-byte aligned");
    }
    host_rejects(st, s, call(k.shape, in, nullptr, st.get(), s), "y is null");
    host_rejects(st, s, call(k.shape, in, buf.u.get(), st.get(), s), "y overlaps u");
    host_rejects(st, s, call(k.shape, in, buf.d.get(), st.get(), s), "y overlaps D");
    host_rejects(st, s, call(k.shape, in, buf.y.get(), reinterpret_cast<ScanStatus*>(buf.y.get() + 3), s), "status overlaps y");
    host_rejects(st, s, call(k.shape, in, buf.y.get(), reinterpret_cast<ScanStatus*>(buf.c.get()), s), "status overlaps C");
    expect_throw<std::invalid_argument>(call(k.shape, in, buf.y.get(), nullptr, s), "status is null");
    for (const cudaStream_t implicit : {cudaStream_t{}, cudaStreamLegacy, cudaStreamPerThread})
        host_rejects(st, s, call(k.shape, in, buf.y.get(), st.get(), implicit), "an explicit non-default stream is required");
    // L = 0 needs no u/delta/B/C/y storage.
    {
        const Case empty = varied({3, 0, 5, 8, 5, 8, true});
        ScanBuffers e(empty.shape);
        e.load(empty, s);
        saccade::selective_scan_cuda(empty.shape, {nullptr, nullptr, e.a.get(), nullptr, nullptr, e.d.get()}, nullptr, e.status.get(), s);
        expect_valid(fetch(e.status, s)[0], "empty_null_storage");
    }
    report << "host_rejection\tall\tPASS\n";
}

void cross_scan(cudaStream_t s) {
    cross_scan_numeric(s);
    DeviceBuffer<ScanStatus> status(1);
    DeviceBuffer<float> x(12), t(48);
    host_rejects(status, s, [&] { saccade::cross_scan_expand_cuda({1, 2, 0, 3}, x.get(), t.get(), status.get(), s); }, "dimensions must be positive");
    host_rejects(status, s, [&] { saccade::cross_scan_expand_cuda({std::size_t{1} << 20, 1, 1 << 10, 1 << 10}, x.get(), t.get(), status.get(), s); },
                 "exceeds int32 capacity");
    host_rejects(status, s, [&] { saccade::cross_scan_expand_cuda({1, 2, 2, 3}, t.get() + 4, t.get(), status.get(), s); }, "tokens overlaps x");
    host_rejects(status, s, [&] { saccade::cross_scan_merge_cuda({1, 2, 2, 3}, t.get(), x.get(), status.get(), cudaStreamLegacy); },
                 "an explicit non-default stream is required");
    host_rejects(status, s, [&] { saccade::cross_scan_merge_cuda({1, 2, 2, 3}, nullptr, x.get(), status.get(), s); }, "y is null");
    report << "cross_scan\tall\tPASS\n";
}

std::string identity() {
    int device = 0, runtime = 0, driver = 0;
    CUDA(cudaGetDevice(&device));
    CUDA(cudaRuntimeGetVersion(&runtime));
    CUDA(cudaDriverGetVersion(&driver));
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, device));
    std::ostringstream id;
    id << saccade::selective_scan_cuda_identity() << "cuda_runtime=" << runtime << "\ncuda_driver_api=" << driver
       << "\ngpu=" << prop.name << "\ncompute_capability=" << prop.major << '.' << prop.minor << "\nuuid=";
    for (const char byte : prop.uuid.bytes) id << std::hex << std::setw(2) << std::setfill('0') << (static_cast<unsigned>(byte) & 0xFFu);
    id << std::dec << '\n';
    return id.str();
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args(argv + 1, argv + argc);
        if (!args.empty() && args.front() == "--no-held-stream") { held_stream_test = false; args.erase(args.begin()); }
        std::string only; // --group NAME runs one group, for focused mutation and debugging runs
        if (args.size() >= 2 && args.front() == "--group") { only = args[1]; args.erase(args.begin(), args.begin() + 2); }
        if (args.size() > 1) throw std::invalid_argument("usage: selective_scan_cuda_tests [--no-held-stream] [--group NAME] [REPORT.tsv]");
        // Session initialization: load every kernel before any stream is held (lazy loading).
        const auto id = identity() + saccade::selective_scan_cuda_load_kernels();
        std::istringstream lines(id);
        for (std::string line; std::getline(lines, line);) report << "identity\t" << line << '\n';
        const Stream stream;
        DirectBackend direct;
        backend = &direct;
        const std::pair<const char*, void (*)(cudaStream_t)> groups[] = {
            {"s01_values", s01_values}, {"coverage", coverage}, {"softplus", softplus},
            {"intermediate_range", intermediate_range}, {"failures", failures},
            {"reset_and_determinism", reset_and_determinism}, {"streams", streams},
            {"host_rejections", host_rejections}, {"cross_scan", cross_scan}};
        bool matched = only.empty();
        for (const auto& [name, group] : groups) {
            if (!only.empty() && only != name) continue;
            matched = true;
            std::cerr << "group " << name << std::endl; // progress under slow sanitizers
            group(stream.get());
        }
        if (!matched) throw std::invalid_argument("unknown group: " + only);
        if (!only.empty()) report << "group_only\t" << only << '\n';
        report << "checks\t" << checks << '\n';
        if (!args.empty()) saccade::write_text_file(args.front(), report.str());
        std::cout << report.str() << "selective_scan_cuda_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << report.str() << "selective_scan_cuda_tests: " << error.what() << '\n';
        return 1;
    }
}

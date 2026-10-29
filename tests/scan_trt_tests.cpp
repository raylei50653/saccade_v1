#include "scan_cuda_suite.hpp"
#include "scan_trt_api_trace.hpp"
#include "saccade/file_io.hpp"
#include "saccade/protocol.hpp"
#include "saccade/scan_frame_kernels.hpp"
#include "saccade/scan_trt_frame.hpp"
#include "saccade/scan_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"

#include <NvInfer.h>
#include <cuda.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <tuple>

// S03 acceptance for the TensorRT scan plugins (docs/SCAN_TRT.md §6). The
// shared S02 groups run through TensorRT engines with the same fixtures and the
// same SCAN_CUDA §5 rule; every engine call is also compared bit for bit with
// the S02 API on the same bytes. The remaining groups check what the TensorRT
// wrapping adds: contract rejection, reset and ownership, serialization-free
// isolation, the frame-end status flow, CUDA graphs and execution constraints.
void scan_trt_test_load_trap();       // scan_trt_fault.cu
void scan_trt_test_trap(cudaStream_t); // scan_trt_fault.cu

namespace {
using namespace saccade::scan_test;
using namespace nvinfer1;
using saccade::CrossScanShape;
using saccade::Error;
using saccade::ResultStatus;

// initcheck makes stream operations wait on a stream held by a host callback,
// so its run skips the held-stream tests (--no-held-stream).
bool held_stream_test = true;

std::uint32_t bits(Error e) { return static_cast<std::uint32_t>(e); }

// ---------------------------------------------------------- TensorRT plumbing

// Collects error messages; groups that expect an error take it out with
// expect_error, and any error left over fails the group.
class Logger final : public ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity > Severity::kWARNING) return;
        try {
            std::lock_guard lock(mutex_);
            if (severity <= Severity::kERROR) errors_.emplace_back(message);
            else warnings_.emplace_back(message);
        } catch (...) {
        }
    }
    bool take(std::string_view needle) {
        std::lock_guard lock(mutex_);
        for (auto it = errors_.begin(); it != errors_.end(); ++it)
            if (it->find(needle) != std::string::npos) { errors_.erase(it); return true; }
        return false;
    }
    std::vector<std::string> drain_errors() {
        std::lock_guard lock(mutex_);
        return std::exchange(errors_, {});
    }
    std::vector<std::string> drain_warnings() {
        std::lock_guard lock(mutex_);
        return std::exchange(warnings_, {});
    }
private:
    std::mutex mutex_;
    std::vector<std::string> errors_, warnings_;
};
Logger logger;

// Receives what the plugins report through the execution context.
class Recorder final : public IErrorRecorder {
public:
    std::int32_t getNbErrors() const noexcept override { std::lock_guard lock(mutex_); return static_cast<std::int32_t>(errors_.size()); }
    ErrorCode getErrorCode(std::int32_t i) const noexcept override {
        std::lock_guard lock(mutex_);
        return i >= 0 && static_cast<std::size_t>(i) < errors_.size() ? errors_[static_cast<std::size_t>(i)].first : ErrorCode::kUNSPECIFIED_ERROR;
    }
    ErrorDesc getErrorDesc(std::int32_t i) const noexcept override {
        std::lock_guard lock(mutex_);
        return i >= 0 && static_cast<std::size_t>(i) < errors_.size() ? errors_[static_cast<std::size_t>(i)].second.c_str() : "";
    }
    bool hasOverflowed() const noexcept override { return false; }
    void clear() noexcept override { std::lock_guard lock(mutex_); errors_.clear(); }
    bool reportError(ErrorCode code, ErrorDesc desc) noexcept override {
        try { std::lock_guard lock(mutex_); errors_.emplace_back(code, desc ? desc : ""); }
        catch (...) {
        }
        return true;
    }
    RefCount incRefCount() noexcept override { return ++refs_; }
    RefCount decRefCount() noexcept override { return --refs_; }
    bool take(std::string_view needle) {
        std::lock_guard lock(mutex_);
        for (auto it = errors_.begin(); it != errors_.end(); ++it)
            if (it->second.find(needle) != std::string::npos) { errors_.erase(it); return true; }
        return false;
    }
    std::vector<std::string> drain() {
        std::lock_guard lock(mutex_);
        std::vector<std::string> out;
        for (auto& e : errors_) out.push_back(std::move(e.second));
        errors_.clear();
        return out;
    }
private:
    mutable std::mutex mutex_;
    std::vector<std::pair<ErrorCode, std::string>> errors_;
    std::atomic<RefCount> refs_{0};
};
Recorder recorder;

void expect_error(std::string_view needle) {
    const bool recorded = recorder.take(needle);
    const bool found = recorded || logger.take(needle);
    check(found, "TensorRT or the plugin reported: " + std::string(needle));
    report << "diagnostic_channel\t" << needle << '\t' << (recorded ? "recorder" : "logger") << '\n';
}

void no_unexpected_errors(const std::string& group) {
    auto errors = logger.drain_errors();
    for (auto& e : recorder.drain()) errors.push_back(std::move(e));
    for (const auto& e : errors) report << "unexpected_error\t" << group << '\t' << e << '\n';
    check(errors.empty(), group + ": no unexpected TensorRT or plugin error" + (errors.empty() ? "" : ": " + errors.front()));
}

struct EngineDeleter { void operator()(ICudaEngine* e) const { delete e; } };
using Engine = std::shared_ptr<ICudaEngine>;

// Declared before the global engine cache so its destruction follows all
// cached engines. TensorRT requires runtimes to outlive their engines.
std::unique_ptr<IRuntime> runtime_owner;
IRuntime& runtime() {
    if (!runtime_owner) {
        saccade::register_scan_trt_plugins(*getPluginRegistry());
        std::unique_ptr<IRuntime> r(createInferRuntime(logger));
        if (!r) throw std::runtime_error("createInferRuntime failed");
        r->setErrorRecorder(&recorder);
        runtime_owner = std::move(r);
    }
    return *runtime_owner;
}

Engine deserialize(const std::vector<std::uint8_t>& plan) {
    Engine e(runtime().deserializeCudaEngine(plan.data(), plan.size()), EngineDeleter{});
    if (!e) throw std::runtime_error("deserializeCudaEngine failed");
    return e;
}

// Engines and their plans are built once per process and kept by key.
struct Engines {
    std::map<std::string, std::vector<std::uint8_t>> plans;
    std::map<std::string, Engine> engines;
    template <typename Build> Engine get(const std::string& key, Build&& build) {
        if (auto it = engines.find(key); it != engines.end()) return it->second;
        const auto start = std::chrono::steady_clock::now();
        plans[key] = build();
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        report << "engine\t" << key << "\tplan_bytes=" << plans[key].size() << std::setprecision(3) << "\tbuild_seconds=" << seconds << '\n';
        return engines[key] = deserialize(plans[key]);
    }
    Engine scan(bool has_d) {
        return get(has_d ? "scan_d" : "scan_no_d", [&] { return saccade::build_selective_scan_engine(logger, has_d); });
    }
    Engine expand() { return get("expand", [] { return saccade::build_cross_scan_expand_engine(logger); }); }
    Engine merge(std::size_t h, std::size_t w) {
        return get("merge_" + std::to_string(h) + "x" + std::to_string(w), [&] {
            return saccade::build_cross_scan_merge_engine(logger, static_cast<std::int32_t>(h), static_cast<std::int32_t>(w));
        });
    }
    std::string pipeline_key(const CrossScanShape& s, std::size_t n, bool has_d) {
        return "pipeline_" + std::to_string(s.batch) + "x" + std::to_string(s.channels) + "x" + std::to_string(s.height) + "x" +
               std::to_string(s.width) + "_N" + std::to_string(n) + (has_d ? "_d" : "_no_d");
    }
    Engine pipeline(const CrossScanShape& s, std::size_t n, bool has_d = true) {
        return get(pipeline_key(s, n, has_d), [&] {
            return saccade::build_cross_scan_pipeline_engine(logger, s, static_cast<std::int32_t>(n), has_d);
        });
    }
    const std::vector<std::uint8_t>& plan(const CrossScanShape& s, std::size_t n, bool has_d = true) {
        pipeline(s, n, has_d);
        return plans.at(pipeline_key(s, n, has_d));
    }
};
Engines engines;

// Creators are function statics initialized during the first engine build.
// Retire every TensorRT object while those registered creators still exist,
// including when a test throws and unwinds out of main's try block.
struct CacheScope {
    void close() noexcept {
        engines.engines.clear();
        engines.plans.clear();
        runtime_owner.reset();
    }
    ~CacheScope() { close(); }
};

// An execution context with its own, pre-allocated device memory (largest
// profile), so contexts never share TensorRT scratch.
class Context {
public:
    explicit Context(Engine engine) : engine_(std::move(engine)) {
        context_.reset(engine_->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
        if (!context_) throw std::runtime_error("createExecutionContext failed");
        const auto bytes = engine_->getNbOptimizationProfiles() > 0 ? engine_->getDeviceMemorySizeForProfileV2(0) : engine_->getDeviceMemorySizeV2();
        if (bytes > 0) CUDA(cudaMalloc(&memory_, static_cast<std::size_t>(bytes)));
        context_->setDeviceMemoryV2(memory_, bytes);
    }
    ~Context() {
        context_.reset();
        if (memory_) cudaFree(memory_);
    }
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    IExecutionContext& get() { return *context_; }
    void shape(const char* name, std::initializer_list<std::int64_t> dims) {
        Dims d{};
        d.nbDims = static_cast<std::int32_t>(dims.size());
        int i = 0;
        for (const auto v : dims) d.d[i++] = v;
        if (!context_->setInputShape(name, d)) throw std::runtime_error(std::string("setInputShape failed: ") + name);
    }
    // Zero-size tensors still need a valid address.
    void address(const char* name, const void* p) {
        if (!context_->setTensorAddress(name, const_cast<void*>(p ? p : placeholder()))) throw std::runtime_error(std::string("setTensorAddress failed: ") + name);
    }
    bool enqueue(cudaStream_t s) { return context_->enqueueV3(s); }
private:
    static const void* placeholder() {
        static DeviceBuffer<float> storage(64);
        return storage.get();
    }
    Engine engine_;
    std::unique_ptr<IExecutionContext> context_;
    void* memory_ = nullptr;
};

void bind_scan(Context& c, const ScanBuffers& b) {
    const auto& s = b.shape;
    const auto B = static_cast<std::int64_t>(s.batch), L = static_cast<std::int64_t>(s.length), D = static_cast<std::int64_t>(s.channels),
               N = static_cast<std::int64_t>(s.state);
    c.shape("u", {B, L, D});
    c.shape("delta", {B, L, D});
    c.shape("A", {static_cast<std::int64_t>(s.a_rows), N});
    c.shape("B", {B, L, N});
    c.shape("C", {B, L, static_cast<std::int64_t>(s.c_width)});
    if (s.has_d) c.shape("D", {D});
    c.address("u", b.u.get()); c.address("delta", b.delta.get()); c.address("A", b.a.get());
    c.address("B", b.b.get()); c.address("C", b.c.get());
    if (s.has_d) c.address("D", b.d.get());
    c.address("y", b.y.get());
    c.address("scan_status", b.status.get());
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
bool same_status(const ScanStatus& a, const ScanStatus& b) { return a.flags == b.flags && a.done == b.done; }

// ------------------------------------------------- TensorRT backend for the shared groups

class TrtBackend final : public ScanBackend {
public:
    std::size_t calls = 0;
    GpuResult scan(const Case& k, cudaStream_t s) override {
        auto& c = scan_context(k.shape.has_d);
        ScanBuffers trt(k.shape);
        trt.load(k, s);
        bind_scan(c, trt);
        const bool enqueued = c.enqueue(s);
        check(enqueued, "enqueueV3 accepted a valid scan case");
        GpuResult r = trt.result(s);
        const GpuResult want = direct_.scan(k, s);
        same(r.y, r.status, want.y, want.status, "scan");
        return r;
    }
    GpuResult expand(const CrossScanShape& shape, const std::vector<float>& x, cudaStream_t s) override {
        if (!expand_) expand_ = std::make_unique<Context>(engines.expand());
        DeviceBuffer<float> dx(x.size()), dt(x.size() * 4);
        DeviceBuffer<ScanStatus> st(1);
        put(dx, x, s);
        put(st, {kStatusSentinel}, s);
        expand_->shape("x", {static_cast<std::int64_t>(shape.batch), static_cast<std::int64_t>(shape.channels),
                             static_cast<std::int64_t>(shape.height), static_cast<std::int64_t>(shape.width)});
        expand_->address("x", dx.get());
        expand_->address("tokens", dt.get());
        expand_->address("expand_status", st.get());
        check(expand_->enqueue(s), "enqueueV3 accepted a valid expand case");
        GpuResult r{fetch(dt, s), fetch(st, s)[0]};
        const GpuResult want = direct_.expand(shape, x, s);
        same(r.y, r.status, want.y, want.status, "expand");
        return r;
    }
    GpuResult merge(const CrossScanShape& shape, const std::vector<float>& y, cudaStream_t s) override {
        Context c(engines.merge(shape.height, shape.width));
        DeviceBuffer<float> dy(y.size()), dout(y.size() / 4);
        DeviceBuffer<ScanStatus> st(1);
        put(dy, y, s);
        put(st, {kStatusSentinel}, s);
        c.shape("y", {4 * static_cast<std::int64_t>(shape.batch), static_cast<std::int64_t>(shape.height * shape.width),
                      static_cast<std::int64_t>(shape.channels)});
        c.address("y", dy.get());
        c.address("out", dout.get());
        c.address("merge_status", st.get());
        check(c.enqueue(s), "enqueueV3 accepted a valid merge case");
        GpuResult r{fetch(dout, s), fetch(st, s)[0]};
        const GpuResult want = direct_.merge(shape, y, s);
        same(r.y, r.status, want.y, want.status, "merge");
        return r;
    }
    PipelineResult pipeline(const CrossScanShape& shape, const std::vector<float>& x, const Case& k, cudaStream_t s) override {
        Context c(engines.pipeline(shape, k.shape.state, k.shape.has_d));
        ScanBuffers buffers(k.shape);
        buffers.load(k, s);
        DeviceBuffer<float> dx(x.size()), out(x.size());
        DeviceBuffer<ScanStatus> statuses(3);
        put(dx, x, s);
        put(statuses, {kStatusSentinel, kStatusSentinel, kStatusSentinel}, s);
        c.address("x", dx.get());
        c.address("delta", buffers.delta.get()); c.address("A", buffers.a.get()); c.address("B", buffers.b.get());
        c.address("C", buffers.c.get());
        if (k.shape.has_d) c.address("D", buffers.d.get());
        c.address("out", out.get());
        c.address("expand_status", statuses.get());
        c.address("scan_status", statuses.get() + 1);
        c.address("merge_status", statuses.get() + 2);
        check(c.enqueue(s), "enqueueV3 accepted a valid pipeline case");
        const auto st = fetch(statuses, s);
        PipelineResult r{fetch(out, s), st[0], st[1], st[2]};
        const PipelineResult want = direct_.pipeline(shape, x, k, s);
        same(r.out, r.expand, want.out, want.expand, "pipeline");
        check(same_status(r.scan, want.scan) && same_status(r.merge, want.merge), "pipeline: plugin statuses equal the S02 API statuses");
        return r;
    }
private:
    Context& scan_context(bool has_d) {
        auto& slot = has_d ? scan_d_ : scan_no_d_;
        if (!slot) slot = std::make_unique<Context>(engines.scan(has_d));
        return *slot;
    }
    void same(const std::vector<float>& y, const ScanStatus& st, const std::vector<float>& want, const ScanStatus& want_st, const char* op) {
        ++calls;
        check(same_bits(y, want), std::string(op) + ": plugin output is bit-identical to the S02 API");
        check(same_status(st, want_st), std::string(op) + ": plugin status equals the S02 API status (flags " + hex(st.flags) + " vs " +
                                            hex(want_st.flags) + ")");
    }
    DirectBackend direct_;
    std::unique_ptr<Context> scan_d_, scan_no_d_, expand_;
};

// ------------------------------------------------------ pipeline frame inputs

// One frame of the Cross-Scan pipeline engine: x, the scan inputs of k (whose
// u are x's tokens), the reference merge result and its §5 bound.
struct PipelineFrame {
    CrossScanShape shape;
    Case k;
    std::vector<float> x;
    std::vector<std::vector<float>> inputs(const saccade::ScanFrameSession& session) const {
        std::vector<std::vector<float>> in;
        for (const auto& name : session.input_names()) {
            if (name == "x") in.push_back(x);
            else if (name == "delta") in.push_back(k.delta);
            else if (name == "A") in.push_back(k.a);
            else if (name == "B") in.push_back(k.b);
            else if (name == "C") in.push_back(k.c);
            else if (name == "D") in.push_back(k.d);
            else throw std::logic_error("unexpected engine input " + name);
        }
        return in;
    }
};

PipelineFrame pipeline_frame(const CrossScanShape& shape, std::size_t state, std::uint64_t seed) {
    PipelineFrame f{shape, {}, random_floats(shape.batch * shape.channels * shape.height * shape.width, seed, 2)};
    f.k = random_case({4 * shape.batch, shape.height * shape.width, shape.channels, state, 1, 1, true}, seed + 1, {3, -5, 2, -1, -1, 2, 1});
    for (std::size_t n = 0; n < state; ++n) f.k.a[n] = -static_cast<float>(n + 1);
    f.k.u = saccade::cross_scan_expand(f.x, shape.batch, shape.channels, shape.height, shape.width);
    return f;
}

// Reference merge output and the SCAN_CUDA §5 expand -> scan -> merge bound.
std::pair<std::vector<double>, std::vector<double>> pipeline_reference(const PipelineFrame& f) {
    const auto& shape = f.shape;
    const std::size_t L = shape.height * shape.width, C = shape.channels;
    const auto ref_y = f.k.run();
    const auto scan_e = scan_bound(f.k);
    auto ref = saccade::cross_scan_merge(ref_y, shape.batch, C, shape.height, shape.width);
    std::vector<double> bound(ref.size());
    for (std::size_t i = 0; i < shape.batch; ++i)
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t p = 0; p < L; ++p) {
                double e = 0, mag = 0;
                for (std::size_t dir = 0; dir < 4; ++dir) {
                    const std::size_t at = ((dir * shape.batch + i) * L + saccade::cross_scan_source(dir, p, shape.height, shape.width)) * C + c;
                    e += scan_e[at];
                    mag += std::fabs(ref_y[at]);
                }
                bound[(i * C + c) * L + p] = (1 + gamma_n(4)) * e / 4 + gamma_n(4) * mag / 4 + kEta;
            }
    return {std::move(ref), std::move(bound)};
}

// The S02 API's output for the frame: the bit-exact oracle of the plugin path.
PipelineResult oracle(const PipelineFrame& f, cudaStream_t s) {
    DirectBackend direct;
    return direct.pipeline(f.shape, f.x, f.k, s);
}

constexpr CrossScanShape kP5{1, 256, 5, 5}, kP4{1, 256, 10, 10}, kP3{1, 256, 20, 20};
constexpr std::size_t kStateM = 16;

std::unique_ptr<saccade::ScanFrameSession> session(const CrossScanShape& shape, saccade::ScanFrameConfig config, bool has_d = true) {
    return std::make_unique<saccade::ScanFrameSession>(engines.plan(shape, kStateM, has_d), config, logger, &recorder);
}


void expect_outcome(const saccade::ScanFrameOutcome& o, std::int64_t frame, ResultStatus status, std::uint32_t frame_error,
                    std::uint32_t fail_code, const std::string& name) {
    check(o.frame == frame && o.status == status && bits(o.frame_error) == frame_error && bits(o.fail_code) == fail_code &&
              o.accepted == (status == ResultStatus::ok),
          name + ": frame " + std::to_string(o.frame) + " status " + std::to_string(static_cast<int>(o.status)) + " error " +
              hex(bits(o.frame_error)) + " fail " + hex(bits(o.fail_code)));
}

void expect_record_valid(const saccade::ScanFrameRecord& r, const std::string& name) {
    bool ok = r.frame_status == 0;
    for (unsigned i = 0; i < 3; ++i) ok = ok && r.flags[i] == 0 && r.done[i] == kScanDone;
    check(ok, name + ": the gate read three valid plugin statuses");
}

// ------------------------------------------------------------- test groups

// The plugin's TensorRT workspace request: 0 bytes for every shape.
bool rt_workspace_zero() {
    auto plugin = saccade::create_selective_scan_plugin(true);
    auto* build = static_cast<IPluginV3OneBuild*>(plugin->getCapabilityInterface(PluginCapabilityType::kBUILD));
    DynamicPluginTensorDesc in[6]{}, out[2]{};
    return build->getWorkspaceSize(in, 6, out, 2) == 0;
}
void contract(cudaStream_t s) {
    {
        saccade::ScanFrameConfig huge;
        huge.max_frames = std::numeric_limits<std::int64_t>::max();
        bool rejected = false;
        try { saccade::ScanFrameSession invalid({}, huge, logger, &recorder); }
        catch (const std::invalid_argument& error) { rejected = std::string_view(error.what()).find("allocation size overflow") != std::string_view::npos; }
        check(rejected, "frame capacity overflow is rejected before allocation or work");
    }
    // 1. Runtime shapes inside the profile that break the plugin contract:
    //    enqueueV3 returns false, the plugin reports why, and nothing runs
    //    (output and status sentinels survive).
    Context c(engines.scan(true));
    check(c.get().getErrorRecorder() == &recorder, "execution context inherits the runtime error recorder");
    const Case k = varied({1, 4, 4, 16, 1, 1, true});
    ScanBuffers b(k.shape);
    b.load(k, s);
    const auto rejected = [&](const std::string& name, const std::function<void()>& configure, std::string_view diagnostic) {
        put(b.status, {kStatusSentinel}, s);
        CUDA(cudaMemsetAsync(b.y.get(), 0xA5, b.y.bytes(), s));
        bind_scan(c, b);
        configure();
        check(!c.enqueue(s), name + ": enqueueV3 rejects the call");
        CUDA(cudaStreamSynchronize(s));
        expect_error(diagnostic);
        const auto st = fetch(b.status, s)[0];
        const auto y = fetch(b.y, s);
        bool untouched = true;
        for (const float v : y) untouched = untouched && std::bit_cast<std::uint32_t>(v) == 0xA5A5A5A5u;
        check(same_status(st, kStatusSentinel) && untouched, name + ": rejected call wrote no output or status");
        // TensorRT may add generic execution errors to the expected plugin
        // diagnostic. Retire them within this negative call's scope.
        logger.drain_errors(); recorder.drain();
        report << "contract\t" << name << "\trejected\tPASS\n";
    };
    DeviceBuffer<float> wide(4 * 4 * 32), a_wide(4 * 32);
    rejected("state_3", [&] { c.shape("A", {1, 3}); c.shape("B", {1, 4, 3}); }, "must be a power of two in [1, 32]");
    rejected("a_rows_2", [&] { c.shape("A", {2, 16}); c.address("A", a_wide.get()); }, "A rows must be 1 or channels");
    rejected("c_width_3", [&] { c.shape("C", {1, 4, 3}); c.address("C", wide.get()); }, "C width must be 1 or state");
    rejected("b_state_differs_from_a", [&] { c.shape("B", {1, 4, 8}); }, "is not (batch, length, state)");
    rejected("delta_differs_from_u", [&] { c.shape("delta", {1, 3, 4}); }, "differs from u");
    rejected("d_differs_from_channels", [&] { c.shape("D", {3}); }, "is not (channels)");
    // The contract holds again for a valid shape on the same context.
    put(b.status, {kStatusSentinel}, s);
    bind_scan(c, b);
    check(c.enqueue(s), "a valid shape after rejections runs");
    expect_valid(fetch(b.status, s)[0], "contract_recovery");
    compare("contract_recovery", fetch(b.y, s), k.run(), scan_bound(k));
    no_unexpected_errors("contract_recovery");

    // 2. Shapes outside the profile cannot be expressed: setInputShape refuses
    //    them (the S02 int32-capacity rejections map here).
    {
        Dims big{};
        big.nbDims = 2;
        big.d[0] = 1;
        big.d[1] = 64;
        check(!c.get().setInputShape("A", big), "setInputShape refuses state 64, beyond the profile");
        logger.drain_errors();
        recorder.drain();
        report << "contract\tprofile_bounds\tsetInputShape_refused\tPASS\n";
    }

    // 3. Implicit streams: the plugin rejects them before any work.
    for (const auto& [name, implicit] : {std::pair{"default", cudaStream_t{}}, std::pair{"legacy", cudaStreamLegacy},
                                         std::pair{"per_thread", cudaStreamPerThread}}) {
        put(b.status, {kStatusSentinel}, s);
        CUDA(cudaStreamSynchronize(s));
        bind_scan(c, b);
        check(!c.enqueue(implicit), std::string(name) + " stream: enqueueV3 rejects the call");
        CUDA(cudaDeviceSynchronize());
        expect_error("an explicit non-default stream is required");
        logger.drain_errors();
        check(same_status(fetch(b.status, s)[0], kStatusSentinel), std::string(name) + " stream: no work enqueued");
        report << "contract\t" << name << "_stream\trejected\tPASS\n";
    }

    // 4. Build-time rejections: FP16 input, wrong input count, constant shapes
    //    the plugin can never accept.
    const auto build_fails = [&](const std::string& name, const std::function<void(INetworkDefinition&, std::vector<std::unique_ptr<IPluginV3>>&)>& define) {
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(1U << static_cast<std::uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        config->setBuilderOptimizationLevel(0);
        std::vector<std::unique_ptr<IPluginV3>> plugins;
        define(*network, plugins);
        std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        check(plan == nullptr, name + ": the build fails");
        logger.drain_errors();
        recorder.drain();
        report << "contract\tbuild_" << name << "\trejected\tPASS\n";
    };
    const auto dims = [](std::initializer_list<std::int64_t> v) {
        Dims d{};
        d.nbDims = static_cast<std::int32_t>(v.size());
        int i = 0;
        for (const auto x : v) d.d[i++] = x;
        return d;
    };
    const auto scan_layer = [&](INetworkDefinition& n, std::vector<std::unique_ptr<IPluginV3>>& keep, DataType type, std::int64_t state, int inputs) {
        std::vector<ITensor*> in{n.addInput("u", type, dims({1, 4, 4})), n.addInput("delta", type, dims({1, 4, 4})),
                                 n.addInput("A", type, dims({1, state})), n.addInput("B", type, dims({1, 4, state})),
                                 n.addInput("C", type, dims({1, 4, 1})), n.addInput("D", type, dims({4}))};
        in.resize(static_cast<std::size_t>(inputs));
        keep.push_back(saccade::create_selective_scan_plugin(true));
        auto* layer = n.addPluginV3(in.data(), inputs, nullptr, 0, *keep.back());
        if (layer) {
            n.markOutput(*layer->getOutput(0));
            n.markOutput(*layer->getOutput(1));
        }
    };
    build_fails("fp16_inputs", [&](auto& n, auto& keep) { scan_layer(n, keep, DataType::kHALF, 16, 6); });
    build_fails("five_inputs_for_has_d", [&](auto& n, auto& keep) { scan_layer(n, keep, DataType::kFLOAT, 16, 5); });
    build_fails("constant_state_3", [&](auto& n, auto& keep) { scan_layer(n, keep, DataType::kFLOAT, 3, 6); });

    // 5. Creator fields: exactly the advertised INT32 fields.
    auto* creator = static_cast<IPluginCreatorV3One*>(getPluginRegistry()->getCreator(saccade::kSelectiveScanPluginName,
                                                                                         saccade::kSelectiveScanPluginVersion, saccade::kScanTrtNamespace));
    check(creator != nullptr, "SelectiveScan v2 creator is registered");
    const std::int32_t one = 1, two = 2;
    const float fone = 1;
    const auto creates = [&](std::vector<PluginField> fields) {
        const PluginFieldCollection fc{static_cast<std::int32_t>(fields.size()), fields.data()};
        std::unique_ptr<IPluginV3> p(creator->createPlugin("t", &fc, TensorRTPhase::kRUNTIME));
        return p != nullptr;
    };
    check(creates({PluginField("has_d", &one, PluginFieldType::kINT32, 1)}), "creator accepts has_d = 1");
    no_unexpected_errors("valid_creator");
    check(!creates({}), "creator rejects a missing has_d");
    check(!creates({PluginField("has_d", &two, PluginFieldType::kINT32, 1)}), "creator rejects has_d = 2");
    check(!creates({PluginField("has_d", &fone, PluginFieldType::kFLOAT32, 1)}), "creator rejects a non-INT32 has_d");
    check(!creates({PluginField("has_d", &one, PluginFieldType::kINT32, 1), PluginField("has_d", &one, PluginFieldType::kINT32, 1)}),
          "creator rejects a duplicate has_d");
    check(!creates({PluginField("has_d", &one, PluginFieldType::kINT32, 1), PluginField("N", &one, PluginFieldType::kINT32, 1)}),
          "creator rejects unknown fields (the old v1 fields included)");
    check(getPluginRegistry()->getCreator(saccade::kSelectiveScanPluginName, "1", saccade::kScanTrtNamespace) == nullptr,
          "no creator answers for the old SelectiveScan v1 identity");
    logger.drain_errors();
    report << "contract\tcreator_fields\tPASS\n";

    // 6. Kernel readiness belongs to a CUDA context: an unattached plugin and a
    //    stream of another CUDA context are refused before any work.
    {
        auto plugin = saccade::create_selective_scan_plugin(true);
        auto* rt = static_cast<IPluginV3OneRuntime*>(plugin->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        PluginTensorDesc in[6]{}, out[2]{};
        const auto desc = [&](PluginTensorDesc& d, DataType t, std::initializer_list<std::int64_t> v) {
            d.dims = dims(v);
            d.type = t;
            d.format = TensorFormat::kLINEAR;
        };
        desc(in[0], DataType::kFLOAT, {1, 4, 4}); desc(in[1], DataType::kFLOAT, {1, 4, 4}); desc(in[2], DataType::kFLOAT, {1, 16});
        desc(in[3], DataType::kFLOAT, {1, 4, 16}); desc(in[4], DataType::kFLOAT, {1, 4, 1}); desc(in[5], DataType::kFLOAT, {4});
        desc(out[0], DataType::kFLOAT, {1, 4, 4}); desc(out[1], DataType::kINT32, {2});
        const void* inputs[6] = {b.u.get(), b.delta.get(), b.a.get(), b.b.get(), b.c.get(), b.d.get()};
        void* outputs[2] = {b.y.get(), b.status.get()};
        put(b.status, {kStatusSentinel}, s);
        CUDA(cudaStreamSynchronize(s));
        check(rt->enqueue(in, out, inputs, outputs, nullptr, s) != 0, "an unattached plugin refuses to enqueue");
        expect_error("kernels are not loaded for any CUDA context");
        std::unique_ptr<IPluginV3> attached(rt->attachToContext(nullptr));
        check(attached != nullptr, "attachToContext binds the plugin to the current context");
        auto* art = static_cast<IPluginV3OneRuntime*>(attached->getCapabilityInterface(PluginCapabilityType::kRUNTIME));
        CUcontext primary{}, other{};
        check(cuCtxGetCurrent(&primary) == CUDA_SUCCESS && primary != nullptr, "primary context current");
        check(cuCtxCreate(&other, nullptr, 0, 0) == CUDA_SUCCESS, "second CUDA context created"); // pushes `other`
        CUstream foreign{};
        check(cuStreamCreate(&foreign, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS, "stream created in the second context");
        const int code = art->enqueue(in, out, inputs, outputs, nullptr, reinterpret_cast<cudaStream_t>(foreign));
        cuStreamDestroy(foreign);
        CUcontext popped{};
        cuCtxPopCurrent(&popped);
        cuCtxDestroy(other);
        check(code != 0, "a plugin bound to one CUDA context refuses a stream of another");
        expect_error("the kernels were loaded in another CUDA context");
        check(same_status(fetch(b.status, s)[0], kStatusSentinel), "refused plugin calls enqueued nothing");
        check(art->enqueue(in, out, inputs, outputs, nullptr, s) == 0, "the bound plugin runs on a stream of its context");
        expect_valid(fetch(b.status, s)[0], "bound_plugin_direct");
        report << "contract\tcontext_binding\tPASS\n";
    }
    check(rt_workspace_zero(), "getWorkspaceSize reports 0 bytes");
}

// Buffers for a static three-plugin engine, also used outside the harness so
// the stream/trace checks observe only enqueueV3 itself.
struct PipelineBuffers {
    PipelineFrame frame;
    Context context;
    ScanBuffers scan;
    DeviceBuffer<float> x, out;
    DeviceBuffer<ScanStatus> statuses{3};
    PipelineBuffers(const PipelineFrame& f)
        : frame(f), context(engines.pipeline(f.shape, f.k.shape.state, f.k.shape.has_d)), scan(f.k.shape), x(f.x.size()), out(f.x.size()) {
        context.address("x", x.get()); context.address("delta", scan.delta.get()); context.address("A", scan.a.get());
        context.address("B", scan.b.get()); context.address("C", scan.c.get());
        if (f.k.shape.has_d) context.address("D", scan.d.get());
        context.address("out", out.get()); context.address("expand_status", statuses.get());
        context.address("scan_status", statuses.get() + 1); context.address("merge_status", statuses.get() + 2);
    }
    void load(const PipelineFrame& f, cudaStream_t s) {
        put(x, f.x, s); scan.load(f.k, s); put(statuses, {kStatusSentinel, kStatusSentinel, kStatusSentinel}, s);
    }
    void enqueue(cudaStream_t s) { check(context.enqueue(s), "pipeline enqueueV3 succeeds"); }
    PipelineResult result(cudaStream_t s) {
        auto status = fetch(statuses, s);
        return {fetch(out, s), status[0], status[1], status[2]};
    }
};

void reset(cudaStream_t s) {
    const Case a = random_case({2, 17, 9, 8, 9, 8, true}, 0x5eed0300);
    Case bad = a; bad.delta[40] = std::numeric_limits<float>::quiet_NaN();
    const Case b = random_case(a.shape, 0x5eed0301);
    Context c(engines.scan(true)); ScanBuffers buffers(a.shape); bind_scan(c, buffers);
    const auto call = [&](const Case& k, bool clear) {
        buffers.load(k, s, clear); check(c.enqueue(s), "reset enqueue succeeds"); return buffers.result(s);
    };
    const auto a1 = call(a, true); expect_valid(a1.status, "reset_a1");
    const auto failed = call(bad, false);
    check(failed.status.done == kScanDone && (failed.status.flags & saccade::kScanInputFailures) == saccade::kScanNonFiniteDelta,
          "reset failure status: flags=" + hex(failed.status.flags) + " done=" + hex(failed.status.done));
    expect_poisoned(failed.y, "reset_failed");
    const auto a2 = call(a, false); expect_valid(a2.status, "reset_a2");
    check(same_bits(a1.y, a2.y), "A -> B(fail) -> A preserves exact output bits");
    const auto b1 = call(b, false); expect_valid(b1.status, "reset_b_valid"); compare("reset_b_valid", b1.y, b.run(), scan_bound(b));
    const auto a3 = call(a, false); expect_valid(a3.status, "reset_a3");
    check(same_bits(a1.y, a3.y), "A -> B(valid) -> A preserves exact output bits");
    report << "reset\tA_B(fail)_A_B(valid)_A\tPASS\n";
}

void isolation(cudaStream_t s) {
    const Stream other;
    const Case a = random_case({4, 100, 64, 16, 1, 1, true}, 0x5eed0310);
    const Case b = random_case({3, 50, 40, 8, 40, 8, true}, 0x5eed0311);
    Context c1(engines.scan(true)), c2(engines.scan(true));
    ScanBuffers b1(a.shape), b2(b.shape); b1.load(a, s); b2.load(b, other.get()); bind_scan(c1, b1); bind_scan(c2, b2);
    for (int i = 0; i < 4; ++i) { check(c1.enqueue(s), "isolated context 1 enqueues"); check(c2.enqueue(other.get()), "isolated context 2 enqueues"); }
    const auto r1 = b1.result(s), r2 = b2.result(other.get());
    expect_valid(r1.status, "isolated_1"); expect_valid(r2.status, "isolated_2");
    compare("isolated_1", r1.y, a.run(), scan_bound(a)); compare("isolated_2", r2.y, b.run(), scan_bound(b));
    DirectBackend direct; const auto d1 = direct.scan(a, s), d2 = direct.scan(b, other.get());
    check(same_bits(r1.y, d1.y) && same_bits(r2.y, d2.y), "interleaved contexts match isolated S02 calls bit for bit");
    report << "isolation\tsame_engine_two_contexts_two_streams\tPASS\n";
}

std::uint64_t digest(const std::vector<float>& out) { return saccade::scan_frame_digest(out.data(), out.size()); }
void retired(const saccade::ScanFrameRun& r, const std::string& name) {
    bool cleared = true;
    for (const auto& slot : r.slot_statuses)
        for (const auto& status : slot) cleared = cleared && status.flags == 0 && status.done == 0;
    check(cleared, name + ": every consumed status is retired to zero");
}
void frame(cudaStream_t s) {
    const auto a = pipeline_frame(kP5, kStateM, 0x5eed0320);
    const auto want = oracle(a, s);
    const auto [ref, bound] = pipeline_reference(a);
    compare("frame_p5_oracle", want.out, ref, bound);
    {
        auto sess = session(kP5, {});
        std::vector<std::vector<std::vector<float>>> frames;
        std::vector<std::uint64_t> wanted;
        for (unsigned i = 0; i < 4; ++i) {
            const auto f = pipeline_frame(kP5, kStateM, 0x5eed0321 + i);
            frames.push_back(f.inputs(*sess)); wanted.push_back(digest(oracle(f, s).out));
        }
        const auto run = sess->run(frames);
        check(run.published == std::vector<std::int64_t>({0, 1, 2, 3}) && run.submitted == 4, "four valid frames are all accepted");
        for (std::size_t i = 0; i < 4; ++i) {
            expect_outcome(run.outcomes[i], static_cast<std::int64_t>(i), ResultStatus::ok, 0, 0, "valid_frame");
            expect_record_valid(run.records[i], "valid_frame");
            check(run.records[i].published == 1 && run.records[i].digest == wanted[i], "published payload digest matches exact frame oracle");
        }
        check(run.control.committed == 3 && run.committed_payload_frame == 3 && digest(run.committed_payload) == wanted[3], "last valid frame is the committed result");
        retired(run, "valid_frames"); sess->close();
    }
    auto bad = a; bad.k.delta[17] = std::numeric_limits<float>::quiet_NaN();
    {
        saccade::ScanFrameConfig cfg; cfg.slots = 1; cfg.fixed_submit = true;
        auto sess = session(kP5, cfg);
        const auto run = sess->run({a.inputs(*sess), bad.inputs(*sess), a.inputs(*sess)});
        check(run.outcomes.size() == 3 && run.records.size() == 3, "fixed-submit preserves every packet and status record");
        expect_outcome(run.outcomes[0], 0, ResultStatus::ok, 0, 0, "failure_sequence");
        expect_outcome(run.outcomes[1], 1, ResultStatus::failed, bits(Error::numeric), bits(Error::numeric), "failure_sequence");
        expect_outcome(run.outcomes[2], 2, ResultStatus::skipped_after_failure, 0, bits(Error::numeric), "failure_sequence");
        expect_record_valid(run.records[2], "post_failure_valid_status");
        check(run.records[2].published == 0 && run.published == std::vector<std::int64_t>({0}), "sticky failure blocks publish despite valid later status");
        check(run.control.committed == 0 && run.control.fail_frame == 1 && run.committed_payload_frame == 0 && same_bits(run.committed_payload, want.out), "failure preserves previous committed payload bits");
        retired(run, "fixed_submit"); sess->close();
    }
    {
        saccade::ScanFrameConfig cfg; cfg.packet_slots = 1;
        auto sess = session(kP5, cfg);
        const auto run = sess->run({a.inputs(*sess), bad.inputs(*sess), a.inputs(*sess)});
        check(run.submitted == 2 && run.stopped_on_failure && run.outcomes.size() == 2 && run.published == std::vector<std::int64_t>({0}), "single packet slot stops before submitting the frame after FAILED");
        sess->close();
    }
    const std::tuple<const char*, unsigned, ScanStatus, std::uint32_t> injected[] = {
        {"flags_with_finite_payload", 1, {saccade::kScanDriveRange, kScanDone}, bits(Error::numeric)},
        {"missing_completion", 1, {0, 0}, bits(Error::protocol)},
        {"unknown_flag", 1, {1u << 31, kScanDone}, bits(Error::numeric) | bits(Error::protocol)},
        {"expand_flag", 0, {saccade::kCrossScanNonFiniteInput, kScanDone}, bits(Error::numeric)}};
    for (const auto& [name, at, status, error] : injected) {
        auto sess = session(kP5, {});
        const auto run = sess->run({a.inputs(*sess)}, [&](std::int64_t, const saccade::ScanFrameSlotView& slot, cudaStream_t stream) {
            CUDA(cudaMemcpyAsync(slot.statuses.at(at), &status, sizeof status, cudaMemcpyHostToDevice, stream));
        });
        expect_outcome(run.outcomes.at(0), 0, ResultStatus::failed, error, error, name);
        check(run.published.empty() && run.records[0].published == 0 && run.committed_payload_frame == -1, std::string(name) + ": invalid status blocks finite payload publication");
        check(same_bits(run.slot_payloads[0], want.out), std::string(name) + ": payload remains the finite valid oracle");
        retired(run, name); sess->close();
        report << "frame\t" << name << "\tPASS\n";
    }
    for (const bool nonfinite : {true, false}) {
        auto f = a;
        if (nonfinite) f.x[9] = std::numeric_limits<float>::quiet_NaN();
        else { f.k.delta[9] = std::numeric_limits<float>::max(); f.k.b[0] = std::numeric_limits<float>::max(); }
        const auto expected = oracle(f, s);
        auto sess = session(kP5, {});
        const auto run = sess->run({f.inputs(*sess)});
        const auto& r = run.records.at(0);
        check(r.flags[0] == expected.expand.flags && r.flags[1] == expected.scan.flags && r.flags[2] == expected.merge.flags &&
                  r.done[0] == expected.expand.done && r.done[1] == expected.scan.done && r.done[2] == expected.merge.done,
              "nonfinite/range frame statuses match S02 oracle");
        expect_outcome(run.outcomes.at(0), 0, ResultStatus::failed, bits(Error::numeric), bits(Error::numeric), nonfinite ? "nan_x" : "range");
        check(run.published.empty(), "failed numeric frame publishes nothing"); retired(run, "numeric_failure"); sess->close();
    }
    for (const auto shape : {kP4, kP3}) {
        const auto f = pipeline_frame(shape, kStateM, 0x5eed0330 + shape.height);
        const auto expected = oracle(f, s); const auto [r, e] = pipeline_reference(f);
        auto sess = session(shape, {}); const auto run = sess->run({f.inputs(*sess)});
        expect_outcome(run.outcomes.at(0), 0, ResultStatus::ok, 0, 0, "m_head");
        check(same_bits(run.committed_payload, expected.out), "m-head committed output is bit-identical to S02");
        compare("frame_m_head_L" + std::to_string(shape.height * shape.width), run.committed_payload, r, e); sess->close();
    }
    report << "frame\tall\tPASS\n";
}

void ownership(cudaStream_t s) {
    for (const int slots : {1, 2}) {
        saccade::ScanFrameConfig cfg; cfg.slots = slots; cfg.packet_slots = 10; cfg.gate_delay_cycles = 4000000;
        auto sess = session(kP5, cfg);
        std::vector<std::vector<std::vector<float>>> frames;
        std::vector<std::uint64_t> wanted;
        for (unsigned i = 0; i < 10; ++i) {
            const auto f = pipeline_frame(kP5, kStateM, 0x5eed0340 + i);
            frames.push_back(f.inputs(*sess)); wanted.push_back(digest(oracle(f, s).out));
        }
        const auto run = sess->run(frames);
        check(run.published.size() == 10, "ownership stress accepts all ten frames");
        for (std::size_t i = 0; i < 10; ++i) {
            expect_record_valid(run.records.at(i), "ownership");
            check(run.records[i].published == 1 && run.records[i].digest == wanted[i], "delayed gate sees the payload for its exact frame");
        }
        retired(run, "ownership"); sess->close();
        report << "ownership\tslots=" << slots << "\tframes=10\tdelay_cycles=4000000\tPASS\n";
    }
}

// Inspect the actual graph's topological indices and transitive dependencies;
// merely finding kernels in a text listing would miss an unordered graph.
void check_graph(const std::string& listing, int slot) {
    std::vector<std::string> types, names;
    std::vector<std::set<std::size_t>> ancestors;
    std::istringstream lines(listing);
    for (std::string line; std::getline(lines, line);) {
        report << "graph_node\tslot=" << slot << '\t' << line << '\n';
        std::istringstream fields(line); std::string field, type, name, deps;
        std::size_t index = 0;
        while (fields >> field) {
            if (field.starts_with("node=")) index = std::stoul(field.substr(5));
            if (field.starts_with("type=")) type = field.substr(5);
            if (field.starts_with("name=")) name = field.substr(5);
            if (field.starts_with("deps=")) deps = field.substr(5);
        }
        check(index == types.size(), "graph nodes have consecutive topological indices");
        check(type == "kernel" || type == "memset", "graph contains only kernels and memsets");
        std::set<std::size_t> upstream;
        std::replace(deps.begin(), deps.end(), ',', ' '); std::istringstream depstream(deps);
        for (std::size_t dep; depstream >> dep;) {
            check(dep < index, "graph dependency precedes its node");
            upstream.insert(dep); upstream.insert(ancestors.at(dep).begin(), ancestors.at(dep).end());
        }
        types.push_back(type); names.push_back(name); ancestors.push_back(std::move(upstream));
    }
    const std::vector<std::string> expected{"", "check_finite", "expand_kernel", "finalize", "", "check_finite", "scan_kernel", "finalize", "", "check_finite", "merge_kernel", "finalize"};
    check(types.size() == expected.size(), "three-plugin graph has 9 kernels and 3 memsets");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        check(expected[i].empty() ? types[i] == "memset" : (types[i] == "kernel" && names[i].find(expected[i]) != std::string::npos), "graph has expected kernel/memset order at " + std::to_string(i));
        if (i) check(ancestors[i].contains(i - 1), "graph preserves transitive producer-to-consumer order");
    }
}
void graph(cudaStream_t s) {
    auto a = pipeline_frame(kP5, kStateM, 0x5eed0350); auto bad = a;
    bad.k.delta[27] = std::numeric_limits<float>::quiet_NaN();
    saccade::ScanFrameConfig cfg; cfg.graphs = true; cfg.slots = 1; cfg.fixed_submit = true;
    auto sess = session(kP5, cfg); check_graph(sess->graph_nodes(0), 0);
    const auto run = sess->run({a.inputs(*sess), bad.inputs(*sess), a.inputs(*sess)});
    expect_outcome(run.outcomes.at(0), 0, ResultStatus::ok, 0, 0, "graph_sequence");
    expect_outcome(run.outcomes.at(1), 1, ResultStatus::failed, bits(Error::numeric), bits(Error::numeric), "graph_sequence");
    expect_outcome(run.outcomes.at(2), 2, ResultStatus::skipped_after_failure, 0, bits(Error::numeric), "graph_sequence");
    const auto expected = oracle(a, s);
    check(same_bits(run.committed_payload, expected.out) && run.records[0].digest == digest(expected.out), "graph output and digest match non-graph S02 oracle");
    expect_record_valid(run.records[2], "graph_post_failure");
    check(run.records[2].published == 0, "graph sticky failure blocks valid later frame"); retired(run, "graph");
    const auto replay = sess->run({a.inputs(*sess)});
    check(replay.outcomes[0].accepted && same_bits(replay.committed_payload, expected.out), "graph run reset permits valid replay");
    sess->close(); report << "graph\tall\tPASS\n";
}

struct Gate {
    std::mutex mutex; std::condition_variable changed; bool open = false, completed = false;
    static void CUDART_CB wait(void* self) {
        auto& g = *static_cast<Gate*>(self); std::unique_lock lock(g.mutex); g.changed.wait(lock, [&] { return g.open; });
        g.completed = true; g.changed.notify_all();
    }
    void release() { { std::lock_guard lock(mutex); open = true; } changed.notify_all(); }
    void join() { std::unique_lock lock(mutex); changed.wait(lock, [&] { return completed; }); }
};

// The watchdog releases the callback even when a mutated enqueue synchronizes.
// Joining it also makes its state lifetime independent of CUDA callback timing.
class HeldStream {
public:
    explicit HeldStream(cudaStream_t stream) {
        watchdog_ = std::thread([this] {
            std::unique_lock lock(mutex_);
            if (!changed_.wait_for(lock, std::chrono::seconds(30), [&] { return released_; })) {
                timed_out_ = true; gate_.release();
            }
        });
        try { CUDA(cudaLaunchHostFunc(stream, Gate::wait, &gate_)); queued_ = true; }
        catch (...) { release(); throw; }
    }
    ~HeldStream() { release(); }
    void release() {
        { std::lock_guard lock(mutex_); released_ = true; } changed_.notify_all(); gate_.release();
        if (watchdog_.joinable()) watchdog_.join();
        if (queued_) { gate_.join(); queued_ = false; }
    }
    bool timed_out() { std::lock_guard lock(mutex_); return timed_out_; }
private:
    Gate gate_; std::mutex mutex_; std::condition_variable changed_; bool released_ = false, timed_out_ = false, queued_ = false;
    std::thread watchdog_;
};

template <typename T> class PinnedCopy {
public:
    explicit PinnedCopy(const std::vector<T>& values) : count_(values.size()) {
        if (count_) { CUDA(cudaMallocHost(reinterpret_cast<void**>(&data_), count_ * sizeof(T))); std::memcpy(data_, values.data(), count_ * sizeof(T)); }
    }
    ~PinnedCopy() { if (data_) cudaFreeHost(data_); }
    void copy_to(T* device, cudaStream_t s) const {
        if (count_) CUDA(cudaMemcpyAsync(device, data_, count_ * sizeof(T), cudaMemcpyHostToDevice, s));
    }
private:
    T* data_ = nullptr; std::size_t count_ = 0;
};
void streams(cudaStream_t s) {
    if (!held_stream_test) { report << "skipped\tstreams\t--no-held-stream\n"; return; }
    const auto a = pipeline_frame(kP5, kStateM, 0x5eed0360);
    PipelineBuffers buf(a); buf.load(a, s); buf.enqueue(s); const auto expected = buf.result(s);
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    buf.enqueue(s); CUDA(cudaStreamEndCapture(s, &graph)); CUDA(cudaGraphInstantiate(&exec, graph, 0));
    CUDA(cudaGraphUpload(exec, s)); CUDA(cudaStreamSynchronize(s));
    const PinnedCopy x(a.x), delta(a.k.delta), aa(a.k.a), b(a.k.b), c(a.k.c), d(a.k.d);
    for (const bool replay : {false, true}) {
        CUDA(cudaMemsetAsync(buf.x.get(), 0xFF, buf.x.bytes(), s));
        CUDA(cudaMemsetAsync(buf.scan.delta.get(), 0xFF, buf.scan.delta.bytes(), s));
        CUDA(cudaMemsetAsync(buf.scan.a.get(), 0xFF, buf.scan.a.bytes(), s));
        CUDA(cudaMemsetAsync(buf.scan.b.get(), 0xFF, buf.scan.b.bytes(), s));
        CUDA(cudaMemsetAsync(buf.scan.c.get(), 0xFF, buf.scan.c.bytes(), s));
        CUDA(cudaMemsetAsync(buf.scan.d.get(), 0xFF, buf.scan.d.bytes(), s));
        CUDA(cudaStreamSynchronize(s));
        cudaEvent_t done{}; CUDA(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        cudaError_t queried{}; bool timed_out = false;
        {
            HeldStream hold(s);
            x.copy_to(buf.x.get(), s); delta.copy_to(buf.scan.delta.get(), s); aa.copy_to(buf.scan.a.get(), s);
            b.copy_to(buf.scan.b.get(), s); c.copy_to(buf.scan.c.get(), s); d.copy_to(buf.scan.d.get(), s);
            if (replay) CUDA(cudaGraphLaunch(exec, s)); else buf.enqueue(s);
            CUDA(cudaEventRecord(done, s));
            queried = cudaEventQuery(done);
            if (queried == cudaErrorNotReady) (void)cudaGetLastError();
            timed_out = hold.timed_out(); hold.release();
        }
        check(!timed_out, replay ? "cudaGraphLaunch returned before watchdog released stream" : "enqueueV3 returned before watchdog released stream");
        check(queried == cudaErrorNotReady, "held caller stream cannot complete before release");
        CUDA(cudaEventSynchronize(done)); CUDA(cudaEventDestroy(done));
        const auto result = buf.result(s);
        expect_valid(result.expand, "held_expand"); expect_valid(result.scan, "held_scan"); expect_valid(result.merge, "held_merge");
        check(same_bits(result.out, expected.out), "held-stream output matches values copied after callback release");
        report << "streams\t" << (replay ? "graph_launch" : "enqueueV3") << "\treturned_while_held\tPASS\n";
    }
    CUDA(cudaGraphExecDestroy(exec)); CUDA(cudaGraphDestroy(graph));
}

void inspect_trace(const std::vector<TracedCall>& calls, cudaStream_t stream, bool graph_launch) {
    std::size_t runtime_launches = 0, runtime_memsets = 0, runtime_graphs = 0;
    for (const auto& c : calls) {
        report << "api\t" << (graph_launch ? "graph" : "enqueue") << '\t' << (c.driver ? "driver" : "runtime") << '\t' << c.name
               << "\tstream=" << static_cast<const void*>(c.stream) << '\n';
        const auto has = [&](const char* needle) { return c.name.find(needle) != std::string::npos; };
        check(!has("Synchronize") && !has("Malloc") && !has("Free") && !has("Alloc") && !has("ModuleLoad") && !has("LibraryLoad"),
              "enqueue trace contains no synchronization, allocation/free or module/library load: " + c.name);
        check(!c.device_to_host, "enqueue trace contains no D2H: " + c.name);
        const bool memory = has("Memcpy") || has("Memset");
        check(!memory || has("Async"), "enqueue trace contains no synchronous copy or memset: " + c.name);
        if (c.stream_bearing) check(c.stream_known && c.stream == stream, "traced stream work uses the supplied explicit stream: " + c.name);
        if (!c.driver && has("LaunchKernel")) ++runtime_launches;
        if (!c.driver && has("MemsetAsync")) ++runtime_memsets;
        if (!c.driver && has("GraphLaunch")) ++runtime_graphs;
    }
    if (graph_launch) check(runtime_graphs == 1, "trace observes one graph launch");
    else check(runtime_launches == 27 && runtime_memsets == 9, "trace observes 9 launches and 3 memsets per enqueueV3 (three calls)");
    report << "api_trace\t" << (graph_launch ? "graph" : "enqueueV3_x3") << "\tcalls=" << calls.size() << "\tkernels=" << runtime_launches
           << "\tmemsets=" << runtime_memsets << "\tPASS\n";
}
void api_trace(cudaStream_t s) {
    const auto a = pipeline_frame(kP5, kStateM, 0x5eed0370);
    PipelineBuffers buf(a); buf.load(a, s); buf.enqueue(s); (void)buf.result(s);
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal)); buf.enqueue(s); CUDA(cudaStreamEndCapture(s, &graph));
    CUDA(cudaGraphInstantiate(&exec, graph, 0)); CUDA(cudaGraphUpload(exec, s)); CUDA(cudaStreamSynchronize(s));
    CudaApiTrace trace;
    trace.arm(); for (int i = 0; i < 3; ++i) buf.enqueue(s); const auto enqueues = trace.disarm();
    CUDA(cudaStreamSynchronize(s)); inspect_trace(enqueues, s, false);
    trace.arm(); CUDA(cudaGraphLaunch(exec, s)); const auto launch = trace.disarm();
    CUDA(cudaStreamSynchronize(s)); inspect_trace(launch, s, true);
    const auto result = buf.result(s); const auto expected = oracle(a, s);
    check(same_bits(result.out, expected.out), "API-traced execution matches S02 oracle");
    CUDA(cudaGraphExecDestroy(exec)); CUDA(cudaGraphDestroy(graph));
}

void async_fault(cudaStream_t) {
    const auto a = pipeline_frame(kP5, kStateM, 0x5eed0380);
    scan_trt_test_load_trap();
    saccade::ScanFrameConfig cfg; cfg.packet_slots = 1; cfg.slots = 1;
    auto sess = session(kP5, cfg);
    std::vector<std::vector<std::vector<float>>> frames(7, a.inputs(*sess));
    bool threw = false;
    try {
        (void)sess->run(frames, [](std::int64_t frame, const saccade::ScanFrameSlotView&, cudaStream_t stream) {
            if (frame == 3) scan_trt_test_trap(stream);
        });
    } catch (const std::runtime_error& error) { threw = true; report << "async_fault\tdiagnostic=" << error.what() << '\n'; }
    check(threw, "asynchronous fault propagates as a CUDA API exception");
    const auto& run = sess->last_run();
    check(run.published == std::vector<std::int64_t>({0, 1, 2}), "only frames accepted before the fault are published");
    for (const auto f : run.published) check(f < 3, "no frame at or after the fault was accepted");
    // Fatal device errors invalidate this process's context. Expected cleanup
    // errors are collected only in this isolated, explicitly selected group.
    sess.reset(); logger.drain_errors(); recorder.drain();
    report << "async_fault\tframe=3\tPASS\n";
}
void cross_scan(cudaStream_t s) { cross_scan_numeric(s); report << "cross_scan\tall\tPASS\n"; }

std::string identity() {
    int device = 0, rt = 0, drv = 0;
    CUDA(cudaGetDevice(&device)); CUDA(cudaRuntimeGetVersion(&rt)); CUDA(cudaDriverGetVersion(&drv));
    cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop, device));
    std::ostringstream out;
    out << saccade::scan_trt_plugin_identity() << "tensorrt=" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH
        << "\ncuda_runtime=" << rt << "\ncuda_driver_api=" << drv << "\ngpu=" << prop.name << "\ncompute_capability=" << prop.major << '.' << prop.minor << '\n';
    return out.str();
}
} // namespace

int main(int argc, char** argv) {
    try {
        CacheScope cache_scope;
        std::string only, output;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--no-held-stream") held_stream_test = false;
            else if (arg == "--group" && i + 1 < argc) only = argv[++i];
            else if (!arg.starts_with("--") && output.empty()) output = arg;
            else throw std::invalid_argument("usage: scan_trt_tests [--no-held-stream] [--group NAME] [REPORT.tsv]");
        }
        const auto id = identity(); std::istringstream lines(id);
        for (std::string line; std::getline(lines, line);) report << "identity\t" << line << '\n';
        const Stream stream;
        auto trt = std::make_unique<TrtBackend>();
        backend = trt.get();
        const std::pair<const char*, void (*)(cudaStream_t)> groups[] = {
            {"s01_values", s01_values}, {"coverage", coverage}, {"softplus", softplus}, {"intermediate_range", intermediate_range},
            {"failures", failures}, {"cross_scan", cross_scan}, {"contract", contract}, {"reset", reset}, {"isolation", isolation},
            {"frame", frame}, {"ownership", ownership}, {"graph", graph}, {"streams", streams}, {"api_trace", api_trace}, {"async_fault", async_fault}};
        bool matched = only.empty();
        for (const auto& [name, group] : groups) {
            if (only.empty() && (std::string_view(name) == "api_trace" || std::string_view(name) == "async_fault")) {
                report << "skipped\t" << name << "\texplicit_group_only\n"; continue;
            }
            if (!only.empty() && only != name) continue;
            matched = true; std::cerr << "group " << name << std::endl;
            group(stream.get()); no_unexpected_errors(name);
        }
        if (!matched) throw std::invalid_argument("unknown group: " + only);
        const auto calls = trt->calls;
        trt.reset();
        backend = nullptr;
        cache_scope.close();
        if (only == "async_fault") {
            // Only this isolated group deliberately invalidates its CUDA
            // context. Keep the resulting teardown diagnostics in the report.
            auto errors = logger.drain_errors();
            for (auto& e : recorder.drain()) errors.push_back(std::move(e));
            for (const auto& e : errors) report << "expected_fatal_teardown_error\t" << e << '\n';
        } else no_unexpected_errors("teardown");
        report << "plugin_s02_comparisons\t" << calls << "\nchecks\t" << checks << '\n';
        if (!only.empty()) report << "group_only\t" << only << '\n';
        if (!output.empty()) saccade::write_text_file(output, report.str());
        std::cout << report.str() << "scan_trt_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << report.str() << "scan_trt_tests: " << error.what() << '\n'; return 1;
    }
}

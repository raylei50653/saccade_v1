#include "scan_cuda_suite.hpp"
#include "scan_trt_api_trace.hpp"
#include "saccade/file_io.hpp"
#include "saccade/head_compare.hpp"
#include "saccade/head_reference.hpp"
#include "saccade/head_trt_network.hpp"
#include "saccade/protocol.hpp"
#include "saccade/scan_frame_kernels.hpp"
#include "saccade/scan_trt_frame.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/sha256.hpp"

#include <NvInfer.h>

#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

// S04 acceptance of the complete T=1 head on TensorRT (docs/HEAD_TRT.md §7):
// numeric comparison under docs/HEAD_POLICY.md, engine I/O contract, frame
// atomicity of the six outputs, status completeness, slot ownership, context
// isolation, reset, CUDA graphs, held streams and the enqueue API trace. The
// engine and the bundle are files whose SHA-256 the caller states.
namespace {
using namespace saccade::scan_test;
using namespace nvinfer1;
using saccade::Error;
using saccade::FrameIoSpec;
using saccade::ResultStatus;
using saccade::ScanFrameConfig;
using saccade::ScanFrameRun;
using saccade::ScanFrameSession;
using saccade::ScanFrameSlotView;

bool held_stream_test = true;
std::uint32_t bits(Error e) { return static_cast<std::uint32_t>(e); }

class Logger final : public ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity > Severity::kERROR) return;
        try { std::lock_guard lock(mutex_); errors_.emplace_back(message); }
        catch (...) {
        }
    }
    std::vector<std::string> drain() { std::lock_guard lock(mutex_); return std::exchange(errors_, {}); }
private:
    std::mutex mutex_;
    std::vector<std::string> errors_;
};
Logger logger;

void no_unexpected_errors(const std::string& group) {
    const auto errors = logger.drain();
    for (const auto& e : errors) report << "unexpected_error\t" << group << '\t' << e << '\n';
    check(errors.empty(), group + ": no unexpected TensorRT error" + (errors.empty() ? "" : ": " + errors.front()));
}

using Features = std::array<std::vector<float>, 3>;
std::vector<std::vector<float>> frame(const Features& f) { return {f[0], f[1], f[2]}; }

struct Fixture {
    std::string plan, engine_sha, bundle_sha;
    saccade::HeadWeights weights;
    FrameIoSpec spec = saccade::head_io_spec();
    std::vector<std::pair<std::string, Features>> inputs; // fixed acceptance inputs
    std::vector<std::vector<float>> expected;             // each input's accepted output arena (same engine)
    std::span<const std::uint8_t> bytes() const { return {reinterpret_cast<const std::uint8_t*>(plan.data()), plan.size()}; }
};
Fixture* fx = nullptr;

std::unique_ptr<ScanFrameSession> session(ScanFrameConfig cfg = {}, const FrameIoSpec* spec = nullptr) {
    return std::make_unique<ScanFrameSession>(fx->bytes(), cfg, logger, nullptr, spec ? spec : &fx->spec);
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::array<std::span<const float>, 6> split(const ScanFrameSession& s, const std::vector<float>& arena) {
    std::array<std::span<const float>, 6> out;
    for (std::size_t k = 0; k < 6; ++k) out[k] = std::span<const float>(arena).subspan(s.outputs().at(k).offset, s.outputs().at(k).count);
    return out;
}

void expect_outcome(const saccade::ScanFrameOutcome& o, std::int64_t frame, ResultStatus status, std::uint32_t frame_error, std::uint32_t fail_code,
                    const std::string& name) {
    check(o.frame == frame && o.status == status && bits(o.frame_error) == frame_error && bits(o.fail_code) == fail_code &&
              o.accepted == (status == ResultStatus::ok),
          name + ": frame " + std::to_string(o.frame) + " status " + std::to_string(static_cast<int>(o.status)) + " error " +
              hex(bits(o.frame_error)) + " fail " + hex(bits(o.fail_code)));
}

void retired(const ScanFrameRun& r, const std::string& name) {
    bool cleared = true;
    for (const auto& slot : r.slot_statuses)
        for (const auto& st : slot) cleared = cleared && st.flags == 0 && st.done == 0;
    check(cleared, name + ": every consumed status is retired to zero");
}

void expect_all_valid(const saccade::ScanFrameRecord& r, std::size_t statuses, const std::string& name) {
    bool ok = r.frame_status == 0;
    for (std::size_t i = 0; i < statuses; ++i) ok = ok && r.flags[i] == 0 && r.done[i] == kScanDone;
    check(ok, name + ": the gate read every status valid");
}

// ---------------------------------------------------------------- groups

void numeric(cudaStream_t) {
    auto s = session();
    report << "session_identity_begin\n" << s->identity() << "session_identity_end\n";
    for (const auto& [label, f] : fx->inputs) {
        const auto run = s->run({frame(f)});
        check(run.published == std::vector<std::int64_t>{0}, label + ": accepted");
        expect_all_valid(run.records.at(0), fx->spec.statuses.size() + 2, label);
        const auto reference = saccade::head_reference(fx->weights, {f[0], f[1], f[2]});
        const auto comparison = saccade::compare_head_outputs(split(*s, run.committed_payload), reference);
        report << comparison.report(label);
        check(comparison.pass, label + ": all six outputs within the policy");
        fx->expected.push_back(run.committed_payload);
    }
    s->close();
    // The scan step's own bound is S02's: identical numbers for the m shapes.
    for (const std::size_t L : {25u, 100u, 400u}) {
        auto k = random_case({4, L, 256, 16, 1, 1, true}, 0x5eed0400 + L);
        const auto want = scan_bound(k);
        namespace ref = saccade::head_ref;
        const auto L64 = static_cast<std::int64_t>(L);
        const saccade::HeadTensor d{{256}, k.d};
        const auto got = ref::selective_scan(ref::from_float(k.u, {4, L64, 256}), ref::from_float(k.delta, {4, L64, 256}), ref::from_float(k.a, {1, 16}),
                                             ref::from_float(k.b, {4, L64, 16}), ref::from_float(k.c, {4, L64, 1}), &d);
        check(got.bound == want && got.sigma == want, "head scan bound equals the S02 bound bit for bit (L=" + std::to_string(L) + ")");
        check(got.value == k.run(), "head scan value equals S01 (L=" + std::to_string(L) + ")");
    }
    report << "numeric\tscan_bound_is_s02\tL=25,100,400\tPASS\n";
}

// A debug engine marks the reference's intermediates as outputs: every one must
// lie inside lambda sigma too, which is how a first divergence is located.
void intermediates(cudaStream_t s) {
    saccade::HeadBuildOptions options;
    options.debug_intermediates = true;
    const auto t0 = std::chrono::steady_clock::now();
    const auto built = saccade::build_head_engine(logger, fx->weights, options);
    report << "debug_engine\tbytes=" << built.plan.size() << "\tbuild_seconds="
           << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() << "\tintermediates=" << built.debug_outputs.size() << '\n';
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(built.plan.data(), built.plan.size()));
    check(engine != nullptr, "debug engine deserializes");
    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
    check(context != nullptr, "debug engine context");
    const auto& f = fx->inputs.at(0).second;
    std::map<std::string, std::unique_ptr<DeviceBuffer<float>>> buffers;
    std::vector<std::unique_ptr<DeviceBuffer<ScanStatus>>> statuses;
    for (std::int32_t i = 0; i < engine->getNbIOTensors(); ++i) {
        const char* name = engine->getIOTensorName(i);
        const auto dims = engine->getTensorShape(name);
        std::size_t n = 1;
        for (int k = 0; k < dims.nbDims; ++k) n *= static_cast<std::size_t>(dims.d[k]);
        if (engine->getTensorDataType(name) == DataType::kINT32) {
            statuses.push_back(std::make_unique<DeviceBuffer<ScanStatus>>(1));
            check(context->setTensorAddress(name, statuses.back()->get()), "address");
            continue;
        }
        auto& b = buffers[name] = std::make_unique<DeviceBuffer<float>>(n);
        check(context->setTensorAddress(name, b->get()), "address");
    }
    for (int level = 0; level < 3; ++level) put(*buffers.at(saccade::kHeadInputNames[static_cast<std::size_t>(level)]), f[static_cast<std::size_t>(level)], s);
    check(context->enqueueV3(s), "debug enqueue");
    CUDA(cudaStreamSynchronize(s));
    for (const auto& st : statuses) expect_valid(fetch(*st, s)[0], "debug_status");
    const auto reference = saccade::head_reference(fx->weights, {f[0], f[1], f[2]}, {}, true);
    for (const auto& [name, t] : reference.intermediates) {
        const auto gpu = fetch(*buffers.at(name), s);
        check(gpu.size() == t.value.size(), name + ": intermediate size");
        double ratio = 0;
        std::size_t outside = 0;
        for (std::size_t i = 0; i < gpu.size(); ++i) {
            const double r = std::fabs(gpu[i] - t.value[i]) / (saccade::head_policy::kLambda * t.sigma[i] + saccade::head_policy::kFloor);
            ratio = std::max(ratio, r);
            outside += !(r <= 1);
        }
        report << "intermediate\t" << name << "\tcount=" << gpu.size() << std::setprecision(3) << "\tmax_err_over_tol=" << ratio << "\toutside=" << outside
               << '\t' << (outside ? "FAIL" : "PASS") << '\n';
        check(outside == 0, name + ": intermediate within lambda sigma");
    }
}

// The session refuses an engine whose I/O differs from the spec; the build
// refuses tactics outside the policy's model.
void io_contract(cudaStream_t) {
    const auto flagged = [](const std::string& tactic) { return !saccade::head_tactic_violations("{\"TacticName\": \"" + tactic + "\"}").empty(); };
    check(flagged("sm80_xmma_fprop_wngd_f32f32_f32_f32_nchwkcrs_nchw_tilesize8x16x64x8_warpsize8x1x1_wngd2x2"), "Winograd tactic rejected");
    check(flagged("sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x64x8"), "FP16 tactic rejected");
    check(flagged("sm86_xmma_fprop_implicit_gemm_tf32f32_tf32f32_f32_nhwckrsc"), "TF32 tactic rejected");
    check(!flagged("sm80_xmma_fprop_implicit_gemm_f32f32_f32f32_f32_nhwckrsc_nhwc_tilesize32x32x8_stage3_warpsize1x2x1_g1_ffma_aligna4_alignc4"),
          "FP32 implicit GEMM accepted");
    check(!flagged("__myl_Resh_0x7f7fcbe2a1be81f64f16364010a4c93a"), "generated kernel hashes are not read as precision markers");
    const auto rejects = [](FrameIoSpec spec, const std::string& needle, const std::string& label) {
        bool rejected = false;
        try { (void)session({}, &spec); }
        catch (const std::exception& error) { rejected = std::string(error.what()).find(needle) != std::string::npos; report << "io_contract\t" << label << '\t' << error.what() << '\n'; }
        check(rejected, "I/O spec mismatch rejected: " + label);
    };
    auto missing = fx->spec;
    missing.statuses.erase(missing.statuses.begin() + 4); // p4.scan_status not described
    rejects(missing, "engine tensor not in the I/O spec: p4.scan_status", "status_missing_from_spec");
    auto extra = fx->spec;
    extra.statuses.push_back({"p6.scan_status", "none"});
    rejects(extra, "engine lacks the status p6.scan_status", "status_absent_from_engine");
    auto dup = fx->spec;
    dup.outputs.push_back(dup.outputs[0]);
    rejects(dup, "names a tensor twice", "duplicate_output");
    auto shape = fx->spec;
    std::swap(shape.outputs[1].name, shape.outputs[2].name); // cls_p4 <-> cls_p5 names with the other's shape
    rejects(shape, "engine output shape differs from the spec", "output_order_swapped");
    auto inputs = fx->spec;
    std::swap(inputs.inputs[1].name, inputs.inputs[2].name);
    rejects(inputs, "engine input shape differs from the spec", "feature_order_swapped");
    auto role = fx->spec;
    role.outputs[3].name = "p3.merge_status";
    role.statuses[2].name = "reg_p3";
    rejects(role, "wrong data type", "status_and_output_roles_swapped");
    report << "io_contract\tall\tPASS\n";
}

// Any failing status or check makes the whole frame fail: none of the six
// outputs is published and the committed result does not move.
void atomicity(cudaStream_t) {
    ScanFrameConfig cfg;
    cfg.slots = 1;
    cfg.fixed_submit = true;
    auto s = session(cfg);
    const auto& a = fx->inputs.at(0).second;
    const auto& b = fx->inputs.at(1).second;
    const auto n = s->statuses().size();
    check(n == 11, "nine plugin statuses and two harness checks");
    const auto sequence = [&](const std::string& name, std::uint32_t error, const saccade::ScanFrameHook& hook, const saccade::ScanFrameHook& after) {
        const auto r = s->run({frame(a), frame(b), frame(a)},
                              [&](std::int64_t k, const ScanFrameSlotView& v, cudaStream_t d) { if (k == 1 && hook) hook(k, v, d); },
                              [&](std::int64_t k, const ScanFrameSlotView& v, cudaStream_t d) { if (k == 1 && after) after(k, v, d); });
        check(r.outcomes.size() == 3, name + ": three packets");
        expect_outcome(r.outcomes[0], 0, ResultStatus::ok, 0, 0, name);
        expect_outcome(r.outcomes[1], 1, ResultStatus::failed, error, error, name);
        expect_outcome(r.outcomes[2], 2, ResultStatus::skipped_after_failure, 0, error, name);
        check(r.published == std::vector<std::int64_t>{0} && r.records[1].published == 0 && r.records[2].published == 0,
              name + ": nothing of the failing or later frames is published");
        check(r.control.committed == 0 && r.committed_payload_frame == 0 && same_bits(r.committed_payload, fx->expected.at(0)),
              name + ": the committed six outputs stay frame 0's bits");
        retired(r, name);
    };
    for (std::size_t i = 0; i < n; ++i) {
        const auto& who = s->statuses()[i].name;
        const std::tuple<const char*, ScanStatus, std::uint32_t> kinds[] = {
            {"flags_with_finite_payload", {saccade::kScanDriveRange, kScanDone}, bits(Error::numeric)},
            {"missing_done", {0, 0}, bits(Error::protocol)},
            {"unknown_flag", {1u << 31, kScanDone}, bits(Error::numeric) | bits(Error::protocol)}};
        for (const auto& [kind, value, error] : kinds) {
            sequence(who + "/" + kind, error, {}, [&, i, value](std::int64_t, const ScanFrameSlotView& v, cudaStream_t d) {
                CUDA(cudaMemcpyAsync(v.statuses.at(i), &value, sizeof value, cudaMemcpyHostToDevice, d));
            });
        }
        report << "atomicity\tstatus=" << who << "\tproducer=" << s->statuses()[i].producer << "\tfinite_payload_flag,missing_done,unknown_bit\tPASS\n";
    }
    // A non-finite element in any one output, with every plugin status valid.
    for (std::size_t k = 0; k < 6; ++k) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        sequence(std::string("nan_in_") + saccade::kHeadOutputNames[k], bits(Error::numeric), [&, k](std::int64_t, const ScanFrameSlotView& v, cudaStream_t d) {
            CUDA(cudaMemcpyAsync(v.outputs.at(k) + s->outputs().at(k).count / 2, &nan, sizeof nan, cudaMemcpyHostToDevice, d));
        }, {});
        report << "atomicity\toutput_check\t" << saccade::kHeadOutputNames[k] << "\tPASS\n";
    }
    // A non-finite feature: the harness input check and the level's Cross-Scan expand both fire.
    for (int level = 0; level < 3; ++level) {
        auto bad = b;
        bad[static_cast<std::size_t>(level)][7] = std::numeric_limits<float>::infinity();
        const auto r = s->run({frame(a), frame(bad), frame(a)});
        expect_outcome(r.outcomes.at(1), 1, ResultStatus::failed, bits(Error::numeric), bits(Error::numeric), "nonfinite_input");
        const auto& rec = r.records.at(1);
        check((rec.flags[n - 2] & saccade::kFrameNonFiniteInput) && (rec.flags[static_cast<std::size_t>(3 * level)] & saccade::kCrossScanNonFiniteInput),
              "input check and the level's expand status both report the non-finite feature");
        check(r.published == std::vector<std::int64_t>{0} && same_bits(r.committed_payload, fx->expected.at(0)), "non-finite feature publishes nothing");
        report << "atomicity\tinput\t" << saccade::kHeadInputNames[static_cast<std::size_t>(level)] << "\tinput_check+expand_status\tPASS\n";
    }
    s->close();
}

void ownership(cudaStream_t) {
    for (const int slots : {1, 2}) {
        ScanFrameConfig cfg;
        cfg.slots = slots;
        cfg.packet_slots = 6;
        cfg.gate_delay_cycles = 4000000;
        auto s = session(cfg);
        std::vector<std::vector<std::vector<float>>> frames;
        std::vector<std::uint64_t> want;
        for (std::size_t i = 0; i < 6; ++i) {
            frames.push_back(frame(fx->inputs[i % 3].second));
            want.push_back(saccade::scan_frame_digest(fx->expected[i % 3].data(), fx->expected[i % 3].size()));
        }
        const auto r = s->run(frames);
        check(r.published.size() == 6, "ownership stress accepts six frames");
        for (std::size_t i = 0; i < 6; ++i)
            check(r.records.at(i).published == 1 && r.records[i].digest == want[i], "delayed gate publishes the six outputs of its own frame");
        check(same_bits(r.committed_payload, fx->expected[2]), "last committed result is frame 5's");
        retired(r, "ownership");
        s->close();
        report << "ownership\tslots=" << slots << "\tframes=6\tdelay_cycles=4000000\tPASS\n";
    }
}

// Two sessions (engines, contexts, buffers) interleaved; reset after failure.
void isolation(cudaStream_t) {
    auto one = session(), two = session();
    const std::size_t order[][2] = {{0, 1}, {2, 0}, {1, 2}};
    for (const auto& [x, y] : order) {
        const auto r1 = one->run({frame(fx->inputs[x].second)});
        const auto r2 = two->run({frame(fx->inputs[y].second)});
        check(same_bits(r1.committed_payload, fx->expected[x]) && same_bits(r2.committed_payload, fx->expected[y]), "interleaved sessions keep their results");
    }
    auto bad = fx->inputs[1].second;
    bad[0][3] = std::numeric_limits<float>::quiet_NaN();
    const auto failed = one->run({frame(bad)});
    expect_outcome(failed.outcomes.at(0), 0, ResultStatus::failed, bits(Error::numeric), bits(Error::numeric), "reset_failure");
    const auto again = one->run({frame(fx->inputs[0].second)});
    check(again.published == std::vector<std::int64_t>{0} && same_bits(again.committed_payload, fx->expected[0]), "a new run after a failure is clean");
    retired(failed, "reset");
    one->close();
    two->close();
    report << "isolation\ttwo_sessions_interleaved+reset\tPASS\n";
}

void graph(cudaStream_t) {
    ScanFrameConfig cfg;
    cfg.graphs = true;
    cfg.slots = 2;
    cfg.fixed_submit = true;
    auto s = session(cfg);
    std::map<std::string, std::size_t> types;
    std::size_t typed = 0;
    for (const int slot : {0, 1}) {
        std::istringstream lines(s->graph_nodes(slot));
        for (std::string line; std::getline(lines, line);) {
            const auto t = line.substr(line.find("type=") + 5, line.find(' ', line.find("type=")) - line.find("type=") - 5);
            if (slot == 0) ++types[t];
            check(t == "kernel" || t == "memset", "graph contains only kernels and memsets: " + line);
            typed += slot == 0 && line.find("typed_edges=") != std::string::npos;
        }
    }
    report << "graph\tnodes";
    for (const auto& [t, c] : types) report << '\t' << t << '=' << c;
    report << "\tnodes_with_typed_edges=" << typed << '\n';
    check(types["memset"] == 9 + 0, "graph has the nine plugin status memsets");
    for (std::size_t i = 0; i < 3; ++i) {
        const auto r = s->run({frame(fx->inputs[i].second)});
        check(r.published == std::vector<std::int64_t>{0} && same_bits(r.committed_payload, fx->expected[i]), "graph replay equals enqueueV3 bitwise");
    }
    auto bad = fx->inputs[2].second;
    bad[2][100] = std::numeric_limits<float>::quiet_NaN();
    const auto r = s->run({frame(fx->inputs[0].second), frame(bad), frame(fx->inputs[1].second)});
    expect_outcome(r.outcomes.at(1), 1, ResultStatus::failed, bits(Error::numeric), bits(Error::numeric), "graph_failure");
    expect_outcome(r.outcomes.at(2), 2, ResultStatus::skipped_after_failure, 0, bits(Error::numeric), "graph_failure");
    check(same_bits(r.committed_payload, fx->expected[0]), "graph failure keeps frame 0's six outputs");
    s->close();
    report << "graph\treplay+failure\tPASS\n";
}

// The engine's enqueue and graph launch return while the stream is held.
struct HeldStream {
    std::mutex mutex;
    std::condition_variable cv;
    bool open = false, done = false, timed_out = false;
    std::thread watchdog;
    static void CUDART_CB wait(void* self) {
        auto& h = *static_cast<HeldStream*>(self);
        std::unique_lock lock(h.mutex);
        h.cv.wait(lock, [&] { return h.open; });
        h.done = true;
        h.cv.notify_all();
    }
    explicit HeldStream(cudaStream_t s) {
        watchdog = std::thread([this] {
            std::unique_lock lock(mutex);
            if (!cv.wait_for(lock, std::chrono::seconds(30), [&] { return open; })) { timed_out = true; open = true; cv.notify_all(); }
        });
        CUDA(cudaLaunchHostFunc(s, wait, this));
    }
    void release() {
        { std::lock_guard lock(mutex); open = true; }
        cv.notify_all();
        if (watchdog.joinable()) watchdog.join();
        std::unique_lock lock(mutex);
        cv.wait(lock, [&] { return done; });
    }
    ~HeldStream() { if (watchdog.joinable()) release(); }
};

struct DirectHead {
    std::unique_ptr<IRuntime> runtime{createInferRuntime(logger)};
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> inputs, outputs;
    std::vector<std::unique_ptr<DeviceBuffer<ScanStatus>>> statuses;
    DirectHead() {
        engine.reset(runtime->deserializeCudaEngine(fx->plan.data(), fx->plan.size()));
        check(engine != nullptr, "direct engine");
        context.reset(engine->createExecutionContext());
        check(context != nullptr, "direct context");
        for (const auto& t : fx->spec.inputs) bind(inputs, t);
        for (const auto& t : fx->spec.outputs) bind(outputs, t);
        for (const auto& st : fx->spec.statuses) {
            statuses.push_back(std::make_unique<DeviceBuffer<ScanStatus>>(1));
            check(context->setTensorAddress(st.name.c_str(), statuses.back()->get()), "status address");
        }
    }
    void bind(std::vector<std::unique_ptr<DeviceBuffer<float>>>& list, const saccade::FrameTensorSpec& t) {
        std::size_t n = 1;
        for (const auto d : t.shape) n *= static_cast<std::size_t>(d);
        list.push_back(std::make_unique<DeviceBuffer<float>>(n));
        check(context->setTensorAddress(t.name.c_str(), list.back()->get()), "address " + t.name);
    }
    std::vector<float> arena(cudaStream_t s) const {
        std::vector<float> out;
        for (const auto& o : outputs) {
            const auto v = fetch(*o, s);
            out.insert(out.end(), v.begin(), v.end());
        }
        return out;
    }
};

void streams(cudaStream_t s) {
    if (!held_stream_test) { report << "skipped\tstreams\t--no-held-stream\n"; return; }
    DirectHead h;
    const auto& f = fx->inputs[1].second;
    for (int level = 0; level < 3; ++level) put(*h.inputs[static_cast<std::size_t>(level)], f[static_cast<std::size_t>(level)], s);
    check(h.context->enqueueV3(s), "warmup enqueue");
    CUDA(cudaStreamSynchronize(s));
    cudaGraph_t g{};
    cudaGraphExec_t exec{};
    CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    const bool captured = h.context->enqueueV3(s);
    CUDA(cudaStreamEndCapture(s, &g));
    check(captured, "capture enqueue");
    CUDA(cudaGraphInstantiate(&exec, g, 0));
    CUDA(cudaGraphUpload(exec, s));
    CUDA(cudaStreamSynchronize(s));
    for (const bool replay : {false, true}) {
        for (const auto& o : h.outputs) CUDA(cudaMemsetAsync(o->get(), 0xFF, o->bytes(), s));
        CUDA(cudaStreamSynchronize(s));
        cudaEvent_t done{};
        CUDA(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        cudaError_t queried{};
        bool timed_out = false;
        {
            HeldStream hold(s);
            if (replay) CUDA(cudaGraphLaunch(exec, s));
            else check(h.context->enqueueV3(s), "held enqueue");
            CUDA(cudaEventRecord(done, s));
            queried = cudaEventQuery(done);
            if (queried == cudaErrorNotReady) (void)cudaGetLastError();
            hold.release();
            timed_out = hold.timed_out;
        }
        check(!timed_out, replay ? "graph launch returned before the watchdog" : "enqueueV3 returned before the watchdog");
        check(queried == cudaErrorNotReady, "held stream cannot complete before release");
        CUDA(cudaEventSynchronize(done));
        CUDA(cudaEventDestroy(done));
        for (const auto& st : h.statuses) expect_valid(fetch(*st, s)[0], "held_status");
        check(same_bits(h.arena(s), fx->expected[1]), "held-stream outputs equal the session's bits");
        report << "streams\t" << (replay ? "graph_launch" : "enqueueV3") << "\treturned_while_held\tPASS\n";
    }
    CUDA(cudaGraphExecDestroy(exec));
    CUDA(cudaGraphDestroy(g));
}

void api_trace(cudaStream_t s) {
    DirectHead h;
    const auto& f = fx->inputs[2].second;
    for (int level = 0; level < 3; ++level) put(*h.inputs[static_cast<std::size_t>(level)], f[static_cast<std::size_t>(level)], s);
    check(h.context->enqueueV3(s), "warmup");
    CUDA(cudaStreamSynchronize(s));
    CudaApiTrace trace;
    trace.arm();
    for (int i = 0; i < 3; ++i) check(h.context->enqueueV3(s), "traced enqueue");
    const auto calls = trace.disarm();
    CUDA(cudaStreamSynchronize(s));
    std::size_t launches = 0, memsets = 0;
    for (const auto& c : calls) {
        const auto has = [&](const char* needle) { return c.name.find(needle) != std::string::npos; };
        check(!has("Synchronize") && !has("Malloc") && !has("Free") && !has("Alloc") && !has("ModuleLoad") && !has("LibraryLoad"),
              "head enqueue makes no synchronization, allocation or module load: " + c.name);
        check(!c.device_to_host, "head enqueue makes no D2H: " + c.name);
        check(!(has("Memcpy") || has("Memset")) || has("Async"), "head enqueue makes no synchronous copy or memset: " + c.name);
        if (c.stream_bearing) check(c.stream_known && c.stream == s, "head enqueue work uses the supplied stream: " + c.name);
        if (!c.driver && has("LaunchKernel")) ++launches;
        if (!c.driver && has("MemsetAsync")) ++memsets;
    }
    check(memsets == 27, "nine plugin status memsets per enqueue");
    check(same_bits(h.arena(s), fx->expected[2]), "traced execution equals the session's bits");
    report << "api_trace\tenqueueV3_x3\tcalls=" << calls.size() << "\truntime_kernel_launches=" << launches << "\truntime_memsets=" << memsets << "\tPASS\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string only, output;
        std::vector<std::string> positional;
        std::set<std::string> skip;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--no-held-stream") held_stream_test = false;
            else if (arg == "--group" && i + 1 < argc) only = argv[++i];
            else if (arg == "--skip" && i + 1 < argc) skip.insert(argv[++i]);
            else if (!arg.starts_with("--")) positional.push_back(arg);
            else throw std::invalid_argument("unknown option " + arg);
        }
        if (positional.size() != 4 && positional.size() != 5)
            throw std::invalid_argument("usage: head_trt_tests ENGINE ENGINE_SHA256 BUNDLE BUNDLE_SHA256 [REPORT.tsv] [--group NAME] [--skip NAME]... [--no-held-stream]");
        if (positional.size() == 5) output = positional[4];
        Fixture fixture;
        fixture.plan = saccade::read_text_file(positional[0]);
        fixture.engine_sha = positional[1];
        check(saccade::sha256_hex(fixture.plan) == fixture.engine_sha, "engine SHA-256 matches");
        const auto bundle = saccade::read_text_file(positional[2]);
        fixture.bundle_sha = positional[3];
        check(saccade::sha256_hex(bundle) == fixture.bundle_sha, "bundle SHA-256 matches");
        fixture.weights = saccade::decode_head_bundle(bundle);
        fixture.inputs = {{"asymmetric/1", saccade::head_test_features("asymmetric", 1)},
                          {"random/1", saccade::head_test_features("random", 1)},
                          {"random/2", saccade::head_test_features("random", 2)}};
        fx = &fixture;
        saccade::register_scan_trt_plugins(*getPluginRegistry());
        report << "identity\tengine_sha256=" << fixture.engine_sha << "\nidentity\tbundle_sha256=" << fixture.bundle_sha << "\nidentity\tpolicy="
               << saccade::kHeadPolicyVersion << '\n';
        const Stream stream;
        const std::pair<const char*, void (*)(cudaStream_t)> groups[] = {
            {"numeric", numeric}, {"intermediates", intermediates}, {"io_contract", io_contract}, {"atomicity", atomicity}, {"ownership", ownership},
            {"isolation", isolation}, {"graph", graph}, {"streams", streams}, {"api_trace", api_trace}};
        bool matched = only.empty();
        for (const auto& [name, group] : groups) {
            if (only.empty() && std::string_view(name) == "api_trace") { report << "skipped\tapi_trace\texplicit_group_only\n"; continue; }
            if (skip.contains(name)) { report << "skipped\t" << name << "\t--skip\n"; continue; }
            // Every group but numeric needs the accepted outputs that numeric records.
            if (!only.empty() && only != name && std::string_view(name) != "numeric") continue;
            matched = matched || only == name;
            std::cerr << "group " << name << std::endl;
            group(stream.get());
            no_unexpected_errors(name);
        }
        if (!matched) throw std::invalid_argument("unknown group: " + only);
        report << "checks\t" << checks << '\n';
        if (!only.empty()) report << "group_only\t" << only << '\n';
        if (!output.empty()) saccade::write_text_file(output, report.str());
        std::cout << report.str() << "head_trt_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << report.str() << "head_trt_tests: " << error.what() << '\n';
        return 1;
    }
}

#include "saccade/file_io.hpp"
#include "saccade/scan_trt_frame.hpp"
#include "saccade/scan_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/sha256.hpp"
#include "../tests/scan_cuda_suite.hpp"

#include <NvInfer.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>

namespace {
using namespace nvinfer1;
using namespace saccade;
using namespace saccade::scan_test;

constexpr CrossScanShape kShape{1, 256, 10, 10};
constexpr std::size_t kState = 16;

class Logger final : public ILogger {
public:
    std::atomic<unsigned> errors{0};
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) ++errors;
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::span<const std::uint8_t> bytes(const std::string& value) {
    return {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()};
}

struct Frame {
    std::vector<float> x;
    Case scan;
    std::vector<std::vector<float>> inputs(const std::vector<std::string>& names) const {
        std::vector<std::vector<float>> out;
        for (const auto& name : names) {
            if (name == "x") out.push_back(x);
            else if (name == "delta") out.push_back(scan.delta);
            else if (name == "A") out.push_back(scan.a);
            else if (name == "B") out.push_back(scan.b);
            else if (name == "C") out.push_back(scan.c);
            else if (name == "D") out.push_back(scan.d);
            else throw std::runtime_error("unexpected input: " + name);
        }
        return out;
    }
};

Frame frame(std::uint64_t seed) {
    Frame f{std::vector<float>(kShape.channels * kShape.height * kShape.width),
            random_case({4, 100, 256, kState, 1, 1, true}, seed)};
    Random rng{seed + 1};
    for (auto& value : f.x) value = rng.uniform(-2, 2);
    for (std::size_t n = 0; n < kState; ++n) f.scan.a[n] = -static_cast<float>(n + 1);
    return f;
}

bool equal(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The callback holds all work submitted after it. The watchdog bounds a
// plugin that synchronizes or lazily loads a kernel during its first enqueue.
struct Hold {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, released = false, timed_out = false;
    std::thread watchdog;
    Hold() : watchdog([this] {
        std::unique_lock lock(mutex);
        if (!cv.wait_for(lock, std::chrono::seconds(30), [this] { return released; })) {
            timed_out = released = true;
            cv.notify_all();
        }
    }) {}
    static void CUDART_CB callback(void* data) {
        auto& h = *static_cast<Hold*>(data);
        std::unique_lock lock(h.mutex);
        h.entered = true;
        h.cv.notify_all();
        h.cv.wait(lock, [&h] { return h.released; });
    }
    void release() {
        std::lock_guard lock(mutex);
        released = true;
        cv.notify_all();
    }
    ~Hold() { release(); if (watchdog.joinable()) watchdog.join(); }
};

struct FirstResult {
    std::vector<float> payload;
    std::vector<std::string> statuses;
    std::vector<ScanStatus> values;
    std::int64_t enqueue_microseconds = 0;
};

FirstResult first_enqueue(ICudaEngine& engine, const Frame& f) {
    // No warmup, kernel preload, ScanFrameSession or DirectBackend has run in
    // this process. createExecutionContext must establish kernel readiness.
    const auto memory_size = engine.getDeviceMemorySizeV2();
    require(memory_size >= 0, "negative context memory size");
    DeviceBuffer<std::uint8_t> memory(static_cast<std::size_t>(memory_size));
    Stream stream;
    std::vector<std::string> input_names;
    std::vector<std::unique_ptr<DeviceBuffer<float>>> inputs;
    std::vector<std::unique_ptr<DeviceBuffer<ScanStatus>>> statuses;
    std::unique_ptr<DeviceBuffer<float>> output;
    FirstResult result;
    // Declare the context after its caller-owned resources so an exception
    // also destroys it before freeing those resources.
    std::unique_ptr<IExecutionContext> context(engine.createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    require(context != nullptr, "first context creation failed");
    context->setDeviceMemoryV2(memory.get(), memory_size);
    for (int i = 0; i < engine.getNbIOTensors(); ++i) {
        const char* name = engine.getIOTensorName(i);
        require(!engine.isShapeInferenceIO(name), std::string("unexpected shape inference tensor: ") + name);
        const auto shape = engine.getTensorShape(name);
        std::size_t count = 1;
        for (int d = 0; d < shape.nbDims; ++d) {
            require(shape.d[d] > 0, "static nonempty P4 tensors required");
            count *= static_cast<std::size_t>(shape.d[d]);
        }
        void* address = nullptr;
        if (engine.getTensorIOMode(name) == TensorIOMode::kINPUT) {
            require(engine.getTensorDataType(name) == DataType::kFLOAT, "FP32 input required");
            input_names.emplace_back(name);
            inputs.push_back(std::make_unique<DeviceBuffer<float>>(count));
            address = inputs.back()->get();
        } else if (engine.getTensorDataType(name) == DataType::kFLOAT) {
            require(output == nullptr, "multiple payload outputs");
            output = std::make_unique<DeviceBuffer<float>>(count);
            address = output->get();
        } else {
            require(engine.getTensorDataType(name) == DataType::kINT32 && shape.nbDims == 1 && count == 2,
                    "INT32[2] status output required");
            result.statuses.emplace_back(name);
            statuses.push_back(std::make_unique<DeviceBuffer<ScanStatus>>(1));
            address = statuses.back()->get();
        }
        require(context->setTensorAddress(name, address), std::string("setTensorAddress failed: ") + name);
    }
    require(output != nullptr && statuses.size() == 3, "P4 pipeline outputs missing");
    const auto host = f.inputs(input_names);
    for (std::size_t i = 0; i < inputs.size(); ++i) put(*inputs[i], host[i], stream.get());
    for (auto& status : statuses) put(*status, {kStatusSentinel}, stream.get());
    CUDA(cudaStreamSynchronize(stream.get()));
    cudaEvent_t event{};
    CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    Hold hold;
    CUDA(cudaLaunchHostFunc(stream.get(), Hold::callback, &hold));
    {
        std::unique_lock lock(hold.mutex);
        hold.cv.wait(lock, [&hold] { return hold.entered || hold.timed_out; });
    }
    const auto start = std::chrono::steady_clock::now();
    const bool enqueued = context->enqueueV3(stream.get());
    result.enqueue_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
    const cudaError_t recorded = cudaEventRecord(event, stream.get());
    const cudaError_t queried = recorded == cudaSuccess ? cudaEventQuery(event) : recorded;
    // NotReady is an expected observation; do not leave it as the thread's
    // last runtime error for the plugin's launch checks.
    cudaGetLastError();
    hold.release();
    const cudaError_t drained = cudaStreamSynchronize(stream.get());
    const cudaError_t destroyed = cudaEventDestroy(event);
    require(!hold.timed_out, "first enqueueV3 blocked on the held stream (30 second watchdog)");
    require(enqueued, "first enqueueV3 returned false");
    CUDA(recorded);
    require(queried == cudaErrorNotReady, "first enqueue did not leave work pending behind the held stream");
    CUDA(drained);
    CUDA(destroyed);
    result.payload = fetch(*output, stream.get());
    for (auto& status : statuses) result.values.push_back(fetch(*status, stream.get())[0]);
    // Context lifetime ends before its externally supplied buffers.
    context.reset();
    return result;
}

ScanStatus named_status(const PipelineResult& oracle, const std::string& name) {
    if (name == "expand_status") return oracle.expand;
    if (name == "scan_status") return oracle.scan;
    if (name == "merge_status") return oracle.merge;
    throw std::runtime_error("unexpected status: " + name);
}

void run(const std::string& plan, const std::filesystem::path& report_path, Logger& logger) {
    register_scan_trt_plugins(*getPluginRegistry());
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    require(runtime != nullptr, "runtime creation failed");
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    require(engine != nullptr, "registered deserialization failed");
    const auto a = frame(1001);
    const auto first = first_enqueue(*engine, a);
    // The oracle deliberately starts after the first held-stream enqueue.
    Stream stream;
    DirectBackend direct;
    const auto oracle_a = direct.pipeline(kShape, a.x, a.scan, stream.get());
    require(equal(first.payload, oracle_a.out), "first enqueue payload differs bitwise from S02");
    for (std::size_t i = 0; i < first.statuses.size(); ++i) {
        const auto expected = named_status(oracle_a, first.statuses[i]);
        require(first.values[i].flags == expected.flags && first.values[i].done == expected.done,
                "first enqueue status differs from S02: " + first.statuses[i]);
    }
    std::ostringstream out;
    const char* loading = std::getenv("CUDA_MODULE_LOADING");
    out << "engine_sha256=" << sha256_hex(plan) << "\nengine_bytes=" << plan.size() << '\n'
        << "cuda_module_loading=" << (loading ? loading : "unset") << '\n'
        << scan_trt_plugin_identity() << describe_scan_engine(*engine)
        << "first_enqueue_held_stream=PASS\nfirst_enqueue_before_warmup=true\nfirst_enqueue_s02_bit_exact=PASS\n"
        << "first_enqueue_microseconds=" << first.enqueue_microseconds << '\n';
    ScanFrameConfig cfg;
    cfg.slots = 1;
    cfg.packet_slots = 3;
    cfg.fixed_submit = true;
    ScanFrameSession session(bytes(plan), cfg, logger);
    auto b = frame(1002);
    b.scan.delta[17] = std::numeric_limits<float>::quiet_NaN();
    const auto oracle_b = direct.pipeline(kShape, b.x, b.scan, stream.get());
    const auto ok = session.run({a.inputs(session.input_names()), a.inputs(session.input_names())});
    require(ok.published == std::vector<std::int64_t>({0, 1}) && equal(ok.committed_payload, oracle_a.out),
            "OK frames differ from S02 or were not accepted");
    const auto failed = session.run({a.inputs(session.input_names()), b.inputs(session.input_names()), a.inputs(session.input_names())});
    require(failed.outcomes.size() == 3 && failed.published == std::vector<std::int64_t>{0}, "failure publication set differs");
    require(failed.outcomes[0].status == ResultStatus::ok && failed.outcomes[1].status == ResultStatus::failed &&
                failed.outcomes[1].frame_error == Error::numeric && failed.outcomes[2].status == ResultStatus::skipped_after_failure,
            "A / failure / A packet sequence differs");
    require(failed.committed_payload_frame == 0 && equal(failed.committed_payload, oracle_a.out), "failure advanced the committed result");
    const ScanStatus expected_b[] = {oracle_b.expand, oracle_b.scan, oracle_b.merge};
    for (std::size_t i = 0; i < 3; ++i)
        require(failed.records[1].flags[i] == expected_b[i].flags && failed.records[1].done[i] == expected_b[i].done,
                "failed frame status differs from the S02 oracle");
    for (const auto& r : ok.records) require(r.digest == scan_frame_digest(oracle_a.out.data(), oracle_a.out.size()), "OK frame digest differs");
    require(failed.records[0].digest == scan_frame_digest(oracle_a.out.data(), oracle_a.out.size()) &&
                failed.records[1].published == 0 && failed.records[2].published == 0,
            "failure or skipped frame published a payload");
    out << session.identity() << "ok_frames_s02_bit_exact=PASS\nfailure_status_s02_exact=PASS\nfailure_packet_sequence=OK,FAILED,SKIPPED\n"
        << "committed_payload_s02_bit_exact=PASS\nresult=PASS\n";
    session.close();
    require(logger.errors == 0, "unexpected TensorRT errors in registered engine run");
    if (!report_path.parent_path().empty()) std::filesystem::create_directories(report_path.parent_path());
    write_text_file(report_path, out.str());
    std::cout << out.str();
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 3, "usage: scan_trt_engine build ENGINE | run ENGINE REPORT | run-unregistered ENGINE");
        const std::string mode = argv[1];
        const std::filesystem::path path = argv[2];
        Logger logger;
        CUDA(cudaSetDevice(0));
        if (mode == "build" && argc == 3) {
            const auto plan = build_cross_scan_pipeline_engine(logger, kShape, static_cast<std::int32_t>(kState));
            if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
            const std::string hash = sha256_hex({reinterpret_cast<const char*>(plan.data()), plan.size()});
            write_binary_file(path, plan);
            const auto saved = read_text_file(path);
            require(saved.size() == plan.size() && sha256_hex(saved) == hash, "saved engine hash differs");
            require(logger.errors == 0, "unexpected TensorRT errors in engine build");
            std::cout << "engine_bytes=" << saved.size() << "\nengine_sha256=" << hash << "\nresult=PASS\n";
        } else if (mode == "run" && argc == 4) {
            run(read_text_file(path), argv[3], logger);
        } else if (mode == "run-unregistered" && argc == 3) {
            require(getPluginRegistry()->getCreator(kSelectiveScanPluginName, kSelectiveScanPluginVersion, kScanTrtNamespace) == nullptr,
                    "SelectiveScan unexpectedly registered before negative control");
            require(getPluginRegistry()->getCreator(kCrossScanExpandPluginName, kCrossScanExpandPluginVersion, kScanTrtNamespace) == nullptr &&
                        getPluginRegistry()->getCreator(kCrossScanMergePluginName, kCrossScanMergePluginVersion, kScanTrtNamespace) == nullptr,
                    "Cross-Scan unexpectedly registered before negative control");
            const auto plan = read_text_file(path);
            std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
            require(runtime != nullptr, "runtime creation failed");
            std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
            if (engine) {
                std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
                require(context == nullptr, "unregistered engine unexpectedly created an execution context");
                std::cout << "unregistered_deserialization=deferred\nunregistered_context=rejected\nresult=PASS\n";
            } else std::cout << "unregistered_deserialization=rejected\nresult=PASS\n";
        } else throw std::runtime_error("invalid mode or argument count");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "scan_trt_engine: " << error.what() << '\n';
        return 1;
    }
}

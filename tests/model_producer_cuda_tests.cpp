#include "saccade/model_producer_cuda.hpp"
#include "saccade/postprocess_cuda.hpp"
#include "saccade/cleanup.hpp"
#include "scan_trt_api_trace.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace saccade;
unsigned checks = 0;
void check(bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void cu(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class F> void reject(F&& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "invalid model producer arguments accepted");
}
struct Device {
    void* pointer{};
    explicit Device(std::size_t bytes) { cu(cudaMalloc(&pointer, bytes)); }
    ~Device() { if (pointer) (void)cudaFree(pointer); }
    Device(const Device&) = delete;
    template<class T> T* as() const { return static_cast<T*>(pointer); }
    void close(CleanupReport& report) { report.release(pointer, [](void* p) { cu(cudaFree(p)); }); }
};
struct Stream {
    cudaStream_t value{};
    Stream() { cu(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) (void)cudaStreamDestroy(value); }
    void close(CleanupReport& report) { report.release(value, [](cudaStream_t s) { cu(cudaStreamDestroy(s)); }); }
};
struct Fixture {
    std::vector<float> images = std::vector<float>(kModelProducerInputElements, 0.5f);
    std::array<std::vector<float>, 3> cls, reg;
    Fixture() {
        for (unsigned l = 0; l < 3; ++l) {
            const auto plane = kPostprocessSizes[l] * kPostprocessSizes[l];
            cls[l].assign(80 * plane, -20.0f);
            reg[l].assign(4 * plane, 0.125f);
        }
        cls[0][0] = 20.0f;
        cls[2][399] = 20.0f;
    }
    PostprocessRawView view() const {
        PostprocessRawView v;
        for (unsigned l = 0; l < 3; ++l) {
            v.cls[l] = cls[l].data(); v.reg[l] = reg[l].data();
            v.cls_elements[l] = cls[l].size(); v.reg_elements[l] = reg[l].size();
        }
        return v;
    }
};
constexpr std::size_t kRawBytes = 8400 * 84 * sizeof(float);
constexpr std::size_t kPluginBytes = kModelProducerPlugins * sizeof(ScanStatus);
struct Harness {
    Stream stream;
    Device images{kModelProducerInputElements * sizeof(float)}, raw{kRawBytes};
    Device plugins{kPluginBytes}, producer{sizeof(PostprocessProducerStatus)};
    Device workspace{model_producer_workspace_bytes()};
    Device output{sizeof(DetectionCandidates)}, post_workspace{postprocess_workspace_bytes()};
    ScanStatus* staging{};
    PostprocessRawView view;
    ModelProducerPlugins status{};
    Harness() {
        cu(cudaMallocHost(reinterpret_cast<void**>(&staging), kPluginBytes));
        std::size_t offset = 0;
        for (unsigned l = 0; l < 3; ++l) {
            view.cls_elements[l] = 80 * kPostprocessSizes[l] * kPostprocessSizes[l];
            view.cls[l] = raw.as<float>() + offset; offset += view.cls_elements[l];
        }
        for (unsigned l = 0; l < 3; ++l) {
            view.reg_elements[l] = 4 * kPostprocessSizes[l] * kPostprocessSizes[l];
            view.reg[l] = raw.as<float>() + offset; offset += view.reg_elements[l];
        }
        check(offset * sizeof(float) == kRawBytes, "raw arena extent");
        for (unsigned i = 0; i < status.size(); ++i) status[i] = plugins.as<ScanStatus>() + i;
    }
    ~Harness() { if (staging) (void)cudaFreeHost(staging); }
    void upload(const Fixture& f) {
        cu(cudaMemcpyAsync(images.pointer, f.images.data(), f.images.size() * sizeof(float), cudaMemcpyHostToDevice, stream.value));
        for (unsigned l = 0; l < 3; ++l) {
            cu(cudaMemcpyAsync(const_cast<float*>(view.cls[l]), f.cls[l].data(), f.cls[l].size() * sizeof(float), cudaMemcpyHostToDevice, stream.value));
            cu(cudaMemcpyAsync(const_cast<float*>(view.reg[l]), f.reg[l].data(), f.reg[l].size() * sizeof(float), cudaMemcpyHostToDevice, stream.value));
        }
    }
    void fresh() {
        for (unsigned i = 0; i < status.size(); ++i) staging[i] = {0, kScanDone};
    }
    void publish_statuses() {
        cu(cudaMemcpyAsync(plugins.pointer, staging, kPluginBytes, cudaMemcpyHostToDevice, stream.value));
    }
    void launch() {
        model_producer_cuda(images.as<float>(), kModelProducerInputElements, view, status,
                            producer.as<PostprocessProducerStatus>(), workspace.pointer,
                            model_producer_workspace_bytes(), stream.value);
    }
    PostprocessProducerStatus read() {
        PostprocessProducerStatus result{};
        cu(cudaMemcpyAsync(&result, producer.pointer, sizeof(result), cudaMemcpyDeviceToHost, stream.value));
        cu(cudaStreamSynchronize(stream.value));
        return result;
    }
    void retired() {
        std::array<ScanStatus, kModelProducerPlugins> observed{};
        cu(cudaMemcpy(observed.data(), plugins.pointer, kPluginBytes, cudaMemcpyDeviceToHost));
        for (const auto& s : observed) check(s.flags == 0 && s.done == 0, "producer failed to retire every plugin status");
    }
    void postprocess(const PostprocessRequest& request) {
        postprocess_cuda(view, request, producer.as<PostprocessProducerStatus>(), output.as<DetectionCandidates>(),
                         sizeof(DetectionCandidates), post_workspace.pointer, postprocess_workspace_bytes(), stream.value);
    }
    DetectionCandidates read_candidates() {
        DetectionCandidates result{};
        cu(cudaMemcpyAsync(&result, output.pointer, sizeof(result), cudaMemcpyDeviceToHost, stream.value));
        cu(cudaStreamSynchronize(stream.value));
        return result;
    }
    void close() {
        cu(cudaStreamSynchronize(stream.value));
        CleanupReport report;
        report.release(staging, [](ScanStatus* p) { cu(cudaFreeHost(p)); });
        output.close(report); post_workspace.close(report); workspace.close(report);
        producer.close(report); plugins.close(report); raw.close(report); images.close(report); stream.close(report);
        report.throw_if_failed();
    }
};
void outcome(const PostprocessProducerStatus& status, std::uint32_t flags, const std::string& name) {
    check(status.flags == flags, name + ": aggregate flags");
    check(status.done == kPostprocessProducerDone, name + ": aggregate done");
}

void status_tests() {
    Fixture f; Harness h; h.upload(f);
    h.fresh(); h.publish_statuses(); h.launch(); outcome(h.read(), 0, "success"); h.retired();
    // A second gate cannot reuse success from the retired source words.
    h.launch(); outcome(h.read(), kPostprocessProtocol, "omitted-producer"); h.retired();
    for (unsigned i = 0; i < kModelProducerPlugins; ++i) {
        const bool scan = i % 3 == 1;
        const auto valid_bits = scan ? kScanInputFailures | kScanRangeFailures : kCrossScanNonFiniteInput;
        const auto wrong_role = scan ? kCrossScanNonFiniteInput : kScanNonFiniteU;
        const std::array<std::pair<ScanStatus, std::uint32_t>, 5> cases{{
            {{valid_bits, kScanDone}, kPostprocessNumeric},
            {{0, 0}, kPostprocessProtocol},
            {{0, kScanDone ^ 1u}, kPostprocessProtocol},
            {{1u << 31, kScanDone}, kPostprocessNumeric | kPostprocessProtocol},
            {{wrong_role, kScanDone}, kPostprocessNumeric | kPostprocessProtocol},
        }};
        for (const auto& [source, expected] : cases) {
            h.fresh(); h.staging[i] = source; h.publish_statuses(); h.launch();
            outcome(h.read(), expected, "plugin " + std::to_string(i)); h.retired();
        }
    }
    h.fresh(); h.staging[0] = {kCrossScanNonFiniteInput, kScanDone}; h.staging[8] = {0, 0};
    h.publish_statuses(); h.launch(); outcome(h.read(), kPostprocessNumeric | kPostprocessProtocol, "mixed-failures"); h.retired();
    h.fresh(); h.publish_statuses(); h.launch(); outcome(h.read(), 0, "failure-to-success"); h.retired();
    h.close(); std::cout << "statuses=PASS\n";
}

void finite_tests() {
    Fixture f; Harness h;
    std::array<std::vector<float>*, 7> tensors{&f.images, &f.cls[0], &f.cls[1], &f.cls[2], &f.reg[0], &f.reg[1], &f.reg[2]};
    for (unsigned tensor = 0; tensor < tensors.size(); ++tensor) {
        auto& values = *tensors[tensor];
        for (const auto index : {std::size_t{0}, values.size() - 1}) {
            const float original = values[index];
            for (const float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()}) {
                values[index] = bad; h.upload(f); h.fresh(); h.publish_statuses(); h.launch();
                outcome(h.read(), kPostprocessNumeric, "finite tensor " + std::to_string(tensor)); h.retired();
            }
            values[index] = original;
        }
    }
    // Finite values at the FP32 limits must not be misclassified as NaN/Inf.
    f.images.back() = std::numeric_limits<float>::max(); f.reg[2].back() = -std::numeric_limits<float>::max();
    h.upload(f); h.fresh(); h.publish_statuses(); h.launch(); outcome(h.read(), 0, "finite-extremes");
    h.close(); std::cout << "finite=PASS\n";
}

void integration_tests() {
    Fixture f; Harness h; h.upload(f);
    PostprocessRequest request; request.epoch = 0x200000003ULL; request.frame = 0x100000005ULL;
    for (const auto failure : {false, true, false}) {
        h.fresh(); if (failure) h.staging[4].flags = kScanDriveRange;
        h.publish_statuses(); h.launch(); h.postprocess(request);
        const auto actual = h.read_candidates();
        const PostprocessProducerStatus expected_producer{failure ? kPostprocessNumeric : 0u, kPostprocessProducerDone};
        const auto expected = postprocess_reference(f.view(), request, expected_producer);
        check(std::memcmp(&actual, &expected, sizeof(actual)) == 0, "F03 aggregate to F04 full output mismatch");
        check(actual.header.flags == (failure ? kPostprocessUpstream : 0u), "F04 upstream gate classification");
        if (!failure) check(actual.header.count == 2, "integration fixture has two detections");
        h.retired(); ++request.frame;
    }
    h.close(); std::cout << "f04-integration=PASS\n";
}

void graph_tests() {
    Fixture f; Harness h; h.upload(f); cu(cudaStreamSynchronize(h.stream.value));
    h.fresh();
    cudaGraph_t graph{}; cudaGraphExec_t executable{};
    cu(cudaStreamBeginCapture(h.stream.value, cudaStreamCaptureModeThreadLocal));
    // Status publication is part of the graph, so every replay has fresh words.
    h.publish_statuses(); h.launch();
    cu(cudaStreamEndCapture(h.stream.value, &graph));
    cu(cudaGraphInstantiate(&executable, graph, 0));
    for (const bool failure : {false, true, false}) {
        h.fresh(); if (failure) h.staging[7] = {kScanStateRange, kScanDone};
        cu(cudaGraphLaunch(executable, h.stream.value));
        outcome(h.read(), failure ? kPostprocessNumeric : 0u, "graph replay"); h.retired();
    }
    CleanupReport report;
    report.release(executable, [](cudaGraphExec_t p) { cu(cudaGraphExecDestroy(p)); });
    report.release(graph, [](cudaGraph_t p) { cu(cudaGraphDestroy(p)); }); report.throw_if_failed();
    h.close(); std::cout << "graph=PASS\n";
}

void argument_tests() {
    Fixture f; Harness h; h.upload(f); h.fresh(); h.publish_statuses();
    cu(cudaMemsetAsync(h.producer.pointer, 0xa5, sizeof(PostprocessProducerStatus), h.stream.value));
    cu(cudaStreamSynchronize(h.stream.value));
    const auto wb = model_producer_workspace_bytes();
    const auto invoke = [&](const float* images, std::size_t n, const PostprocessRawView& raw,
                            const ModelProducerPlugins& plugins, PostprocessProducerStatus* producer,
                            void* workspace, std::size_t bytes, cudaStream_t stream) {
        reject([&] { model_producer_cuda(images, n, raw, plugins, producer, workspace, bytes, stream); });
    };
    auto* images = h.images.as<float>(); auto* producer = h.producer.as<PostprocessProducerStatus>();
    const auto n = kModelProducerInputElements; auto* w = h.workspace.pointer; auto stream = h.stream.value;
    for (auto s : {cudaStream_t{}, cudaStreamLegacy, cudaStreamPerThread}) invoke(images, n, h.view, h.status, producer, w, wb, s);
    invoke(nullptr, n, h.view, h.status, producer, w, wb, stream);
    invoke(f.images.data(), n, h.view, h.status, producer, w, wb, stream);
    invoke(images, n - 1, h.view, h.status, producer, w, wb, stream);
    invoke(images, n + 1, h.view, h.status, producer, w, wb, stream);
    invoke(images, n, h.view, h.status, nullptr, w, wb, stream);
    invoke(images, n, h.view, h.status, producer, nullptr, wb, stream);
    invoke(images, n, h.view, h.status, producer, w, wb - 1, stream);
    invoke(images, n, h.view, h.status, producer, static_cast<char*>(w) + 1, wb, stream);
    invoke(images, n, h.view, h.status, reinterpret_cast<PostprocessProducerStatus*>(h.raw.pointer), w, wb, stream);
    invoke(images, n, h.view, h.status, reinterpret_cast<PostprocessProducerStatus*>(w), w, wb, stream);
    invoke(images, n, h.view, h.status, reinterpret_cast<PostprocessProducerStatus*>(static_cast<char*>(h.producer.pointer) + 1), w, wb, stream);
    for (unsigned i = 0; i < 9; ++i) {
        auto statuses = h.status; statuses[i] = nullptr;
        invoke(images, n, h.view, statuses, producer, w, wb, stream);
        statuses = h.status; statuses[i] = reinterpret_cast<ScanStatus*>(static_cast<char*>(h.plugins.pointer) + 1);
        invoke(images, n, h.view, statuses, producer, w, wb, stream);
        statuses = h.status; statuses[i] = reinterpret_cast<ScanStatus*>(h.producer.pointer);
        invoke(images, n, h.view, statuses, producer, w, wb, stream);
    }
    auto statuses = h.status; statuses[8] = statuses[0]; invoke(images, n, h.view, statuses, producer, w, wb, stream);
    Device small(4);
    invoke(small.as<float>(), n, h.view, h.status, producer, w, wb, stream);
    invoke(images + 1, n, h.view, h.status, producer, w, wb, stream);
    invoke(images, n, h.view, h.status, small.as<PostprocessProducerStatus>(), w, wb, stream);
    invoke(images, n, h.view, h.status, producer, small.pointer, wb, stream);
    statuses = h.status; statuses[8] = small.as<ScanStatus>(); invoke(images, n, h.view, statuses, producer, w, wb, stream);
    for (unsigned l = 0; l < 3; ++l) {
        auto raw = h.view; raw.cls[l] = nullptr; invoke(images, n, raw, h.status, producer, w, wb, stream);
        raw = h.view; --raw.cls_elements[l]; invoke(images, n, raw, h.status, producer, w, wb, stream);
        raw = h.view; ++raw.reg_elements[l]; invoke(images, n, raw, h.status, producer, w, wb, stream);
        raw = h.view; raw.reg[l] = raw.cls[l]; invoke(images, n, raw, h.status, producer, w, wb, stream);
        raw = h.view; raw.cls[l] = small.as<float>(); invoke(images, n, raw, h.status, producer, w, wb, stream);
        raw = h.view; raw.reg[l] = images; invoke(images, n, raw, h.status, producer, w, wb, stream);
    }
    // Host rejection must not write producer or retire otherwise-fresh statuses.
    const auto untouched = h.read();
    check(untouched.flags == 0xa5a5a5a5u && untouched.done == 0xa5a5a5a5u, "argument rejection wrote aggregate");
    std::array<ScanStatus, 9> observed{}; cu(cudaMemcpy(observed.data(), h.plugins.pointer, kPluginBytes, cudaMemcpyDeviceToHost));
    for (const auto& source : observed) check(source.flags == 0 && source.done == kScanDone, "argument rejection retired source statuses");
    CleanupReport report; small.close(report); report.throw_if_failed();
    h.close(); std::cout << "arguments=PASS\n";
}

void api_trace_tests() {
    Fixture f; Harness h; h.upload(f); h.fresh(); h.publish_statuses(); cu(cudaStreamSynchronize(h.stream.value));
    CudaApiTrace trace; trace.arm(); h.launch(); const auto calls = trace.disarm();
    check(!calls.empty(), "empty CUDA API trace");
    unsigned launches = 0;
    for (const auto& call : calls) {
        const auto has = [&](const char* fragment) { return call.name.find(fragment) != std::string::npos; };
        check(!has("Alloc") && !has("Free") && !has("Synchronize") && !has("Memcpy") && !has("Event") && !has("Memset"),
              "unexpected frame API: " + call.name);
        if (call.stream_bearing) check(call.stream_known && call.stream == h.stream.value, "frame work escaped explicit stream: " + call.name);
        if (!call.driver && has("LaunchKernel")) ++launches;
    }
    check(launches > 0, "no runtime kernel launch traced");
    outcome(h.read(), 0, "API-traced launch"); h.close();
    std::cout << "api-trace=PASS calls=" << calls.size() << " launches=" << launches << '\n';
}

struct HoldState { std::atomic<bool> started{false}, released{false}; };
struct Hold {
    std::shared_ptr<HoldState> state = std::make_shared<HoldState>();
    ~Hold() { state->released.store(true); }
};
void held(void* data) {
    // CUDA owns this shared-state reference until the callback returns. Early
    // test failures release the hold without leaving a dangling stack pointer.
    const std::unique_ptr<std::shared_ptr<HoldState>> owned(static_cast<std::shared_ptr<HoldState>*>(data));
    auto& hold = **owned; hold.started.store(true);
    while (!hold.released.load()) std::this_thread::yield();
}
void launch_hold(cudaStream_t stream, Hold& hold) {
    auto state = std::make_unique<std::shared_ptr<HoldState>>(hold.state);
    cu(cudaLaunchHostFunc(stream, held, state.get()));
    (void)state.release();
}
template<class F> void while_held(Hold& hold, F&& enqueue) {
    const auto start_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!hold.state->started.load() && std::chrono::steady_clock::now() < start_deadline) std::this_thread::yield();
    if (!hold.state->started.load()) { hold.state->released.store(true); throw std::runtime_error("held-stream callback did not start"); }
    std::atomic<bool> returned{false}; std::exception_ptr error;
    std::thread submit([&] {
        try { cu(cudaSetDevice(0)); enqueue(); } catch (...) { error = std::current_exception(); }
        returned.store(true);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!returned.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool asynchronous = returned.load(); hold.state->released.store(true); submit.join();
    if (error) std::rethrow_exception(error);
    check(asynchronous, "model producer synchronized with pending producer/consumer");
}
void stream_tests(bool no_held) {
    Fixture f; Harness h; h.upload(f); h.fresh(); h.publish_statuses(); cu(cudaStreamSynchronize(h.stream.value));
    if (!no_held) {
        Hold hold; launch_hold(h.stream.value, hold);
        while_held(hold, [&] { h.launch(); }); outcome(h.read(), 0, "held detect stream"); h.retired();
    }
    Stream upstream, consumer; Device stage(kPluginBytes), saved(sizeof(PostprocessProducerStatus));
    h.fresh(); cu(cudaMemcpy(stage.pointer, h.staging, kPluginBytes, cudaMemcpyHostToDevice));
    cudaEvent_t ready{}, consumed{};
    cu(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming)); cu(cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming));
    Hold producer_hold;
    if (!no_held) launch_hold(upstream.value, producer_hold);
    cu(cudaMemcpyAsync(h.plugins.pointer, stage.pointer, kPluginBytes, cudaMemcpyDeviceToDevice, upstream.value));
    cu(cudaEventRecord(ready, upstream.value));
    const auto receive = [&] { cu(cudaStreamWaitEvent(h.stream.value, ready)); h.launch(); };
    if (!no_held) while_held(producer_hold, receive); else receive();
    outcome(h.read(), 0, "pending upstream statuses"); h.retired();
    cu(cudaEventRecord(ready, h.stream.value)); cu(cudaStreamWaitEvent(consumer.value, ready));
    Hold consumer_hold;
    if (!no_held) launch_hold(consumer.value, consumer_hold);
    cu(cudaMemcpyAsync(saved.pointer, h.producer.pointer, sizeof(PostprocessProducerStatus), cudaMemcpyDeviceToDevice, consumer.value));
    cu(cudaEventRecord(consumed, consumer.value));
    const auto reuse = [&] { cu(cudaStreamWaitEvent(h.stream.value, consumed)); h.launch(); };
    if (!no_held) while_held(consumer_hold, reuse); else reuse();
    outcome(h.read(), kPostprocessProtocol, "reuse waits for last consumer");
    PostprocessProducerStatus previous{}; cu(cudaMemcpy(&previous, saved.pointer, sizeof(previous), cudaMemcpyDeviceToHost));
    outcome(previous, 0, "consumer retained old success");
    cu(cudaStreamSynchronize(upstream.value)); cu(cudaStreamSynchronize(consumer.value));
    CleanupReport report;
    report.release(consumed, [](cudaEvent_t e) { cu(cudaEventDestroy(e)); });
    report.release(ready, [](cudaEvent_t e) { cu(cudaEventDestroy(e)); });
    stage.close(report); saved.close(report); upstream.close(report); consumer.close(report); report.throw_if_failed();
    h.close(); std::cout << "streams=PASS held=" << (no_held ? "UNEXECUTED" : "PASS") << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool no_held = false, no_trace = false, streams_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--no-held-stream") no_held = true;
            else if (argument == "--no-api-trace") no_trace = true;
            else if (argument == "--streams-only") streams_only = true;
            else throw std::invalid_argument("unknown argument: " + argument);
        }
        cu(cudaSetDevice(0));
        std::cout << model_producer_cuda_identity() << model_producer_cuda_load_kernels();
        std::cout << postprocess_cuda_load_kernels();
        if (!streams_only) {
            status_tests(); finite_tests(); integration_tests(); graph_tests(); argument_tests();
            if (!no_trace) api_trace_tests(); else std::cout << "api-trace=UNEXECUTED\n";
        }
        stream_tests(no_held);
        std::cout << "PASS checks=" << checks << '\n'; return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL: " << error.what() << '\n'; return 1; }
}

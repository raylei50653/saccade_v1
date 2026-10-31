#include "saccade/postprocess_cuda.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace saccade;
unsigned checks = 0;
float worst_score_error = 0.0f;
void require(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
void cu(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class F> void reject(F&& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid F04 argument accepted");
}
struct Device {
    void* p = nullptr;
    explicit Device(std::size_t n) { cu(cudaMalloc(&p, n)); }
    ~Device() { if (p) (void)cudaFree(p); }
    Device(const Device&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
    void close(CleanupReport& report) { report.release(p, [](void* x) { cu(cudaFree(x)); }); }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cu(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) (void)cudaStreamDestroy(value); }
    void close(CleanupReport& report) {
        report.release(value, [](cudaStream_t x) { cu(cudaStreamDestroy(x)); });
    }
};
struct Raw {
    std::array<std::vector<float>, 3> cls, reg;
    Raw() {
        for (unsigned l = 0; l < 3; ++l) {
            const auto n = kPostprocessSizes[l] * kPostprocessSizes[l];
            cls[l].assign(80 * n, -20.0f); reg[l].assign(4 * n, 0.125f);
        }
    }
    PostprocessRawView view() const {
        PostprocessRawView v;
        for (unsigned l = 0; l < 3; ++l) {
            v.cls[l] = cls[l].data(); v.reg[l] = reg[l].data();
            v.cls_elements[l] = cls[l].size(); v.reg_elements[l] = reg[l].size();
        }
        return v;
    }
    std::pair<unsigned, unsigned> locate(unsigned id) const {
        for (unsigned l = 0; l < 3; ++l) {
            const auto n = kPostprocessSizes[l] * kPostprocessSizes[l];
            if (id < n) return {l, id};
            id -= n;
        }
        throw std::runtime_error("invalid fixture ID");
    }
    void logit(unsigned id, unsigned c, float x) {
        const auto [l, i] = locate(id);
        cls[l][c * kPostprocessSizes[l] * kPostprocessSizes[l] + i] = x;
    }
    void distances(unsigned id, std::array<float, 4> values) {
        const auto [l, i] = locate(id);
        const auto n = kPostprocessSizes[l] * kPostprocessSizes[l];
        for (unsigned c = 0; c < 4; ++c) reg[l][c * n + i] = values[c];
    }
    void box(unsigned id, float x1, float y1, float x2, float y2) {
        const auto [l, i] = locate(id);
        const float stride = static_cast<float>(kPostprocessStrides[l]);
        const float x = static_cast<float>(i % kPostprocessSizes[l]) + 0.5f;
        const float y = static_cast<float>(i / kPostprocessSizes[l]) + 0.5f;
        distances(id, {x - x1 / stride, y - y1 / stride, x2 / stride - x, y2 / stride - y});
    }
};
PostprocessRequest request(unsigned width = 640, unsigned height = 640) {
    PostprocessRequest r;
    r.epoch = 0x123456789ULL; r.frame = 0x200000003ULL;
    r.mapping = {static_cast<int>(width), static_cast<int>(height), 640, 640,
                 640.0 / width, 640.0 / height};
    return r;
}
const PostprocessProducerStatus success{0, kPostprocessProducerDone};
using Setup = std::function<void(Raw&, PostprocessRequest&, PostprocessProducerStatus&)>;
struct Case { const char* name; Setup setup; };
std::vector<Case> cases() {
    return {
        {"empty", [](Raw&, auto&, auto&) {}},
        {"single-asymmetric-map-no-clip", [](Raw& x, auto& r, auto&) {
            x.logit(0, 0, 0); x.distances(0, {1, 2, 3, 4}); r = request(1919, 1079);
        }},
        {"level-first-last-layout", [](Raw& x, auto& r, auto&) {
            for (unsigned id : {0u, 6399u, 6400u, 7999u, 8000u, 8399u}) x.logit(id, 0, 0);
            r = request(65535, 1);
        }},
        {"class-tie-lowest", [](Raw& x, auto&, auto&) { x.logit(3, 0, 1); x.logit(3, 1, 1); }},
        {"class-rounded-sigmoid-saturation", [](Raw& x, auto&, auto&) {
            x.logit(0, 0, 20); x.logit(0, 1, 30);
        }},
        {"top300-before-person", [](Raw& x, auto&, auto&) {
            for (unsigned id = 0; id < 300; ++id) x.logit(id, 1, 2);
            x.logit(600, 0, 1);
        }},
        {"top300-tie-id", [](Raw& x, auto&, auto&) {
            for (unsigned id = 0; id < 301; ++id) x.logit(id, 0, 0);
        }},
        {"degenerate-occupies-top300", [](Raw& x, auto&, auto&) {
            for (unsigned id = 0; id < 300; ++id) { x.logit(id, 0, 2); x.distances(id, {0, 0, 0, 0}); }
            x.logit(600, 0, 1);
        }},
        {"threshold-adjacent-logits", [](Raw& x, auto&, auto&) {
            const float at = std::log(kPostprocessScoreFloor / (1.0f - kPostprocessScoreFloor));
            x.logit(0, 0, std::nextafter(at, -INFINITY)); x.logit(1, 0, at);
            x.logit(2, 0, std::nextafter(at, INFINITY));
        }},
        {"nms-equal-score-id-suppressed-pool", [](Raw& x, auto&, auto&) {
            for (unsigned id : {17u, 9u, 2u}) { x.logit(id, 0, 0); x.box(id, 0, 0, 16, 16); }
        }},
        {"nms-iou-exact-half", [](Raw& x, auto&, auto&) {
            x.logit(0, 0, 2); x.logit(1, 0, 1);
            x.box(0, 0, 0, 16, 16); x.box(1, 0, 0, 8, 16);
        }},
        {"nms-iou-above-half", [](Raw& x, auto&, auto&) {
            x.logit(0, 0, 2); x.logit(1, 0, 1);
            x.box(0, 0, 0, 16, 16); x.box(1, 0, 0, 8.0000152587890625f, 16);
        }},
        {"nms-iou-below-half", [](Raw& x, auto&, auto&) {
            x.logit(0, 0, 2); x.logit(1, 0, 1);
            x.box(0, 0, 0, 16, 16); x.box(1, 0, 0, 7.9999847412109375f, 16);
        }},
        {"nms-greedy-chain", [](Raw& x, auto&, auto&) {
            for (unsigned i = 0; i < 3; ++i) { x.logit(i, 0, 3.0f - i); x.box(i, 4.0f * i, 0, 16.0f + 4.0f * i, 16); }
        }},
        {"negative-ltrb-valid-and-degenerate", [](Raw& x, auto&, auto&) {
            x.logit(0, 0, 1); x.distances(0, {-1, -1, 2, 2});
            x.logit(1, 0, 0); x.distances(1, {-2, 1, 1, 1});
        }},
        {"finite-extreme-logits", [](Raw& x, auto&, auto&) {
            for (auto& level : x.cls) std::fill(level.begin(), level.end(), -std::numeric_limits<float>::max());
            x.logit(8399, 0, std::numeric_limits<float>::max());
        }},
        {"minimum-anchor-adjacent-valid-box", [](Raw& x, auto&, auto&) {
            x.logit(0, 0, 0);
            const float negative_right = -std::nextafter(0.5f, 0.0f);
            x.distances(0, {0.5f, 0.5f, negative_right, negative_right});
        }},
        {"capacity-zero-empty", [](Raw&, auto& r, auto&) { r.capacity = 0; }},
        {"capacity-zero-overflow", [](Raw& x, auto& r, auto&) { x.logit(0, 0, 0); r.capacity = 0; }},
        {"capacity-exact", [](Raw& x, auto& r, auto&) { x.logit(0, 0, 0); x.logit(3, 0, 0); r.capacity = 2; }},
        {"capacity-one-short", [](Raw& x, auto& r, auto&) { x.logit(0, 0, 0); x.logit(3, 0, 0); r.capacity = 1; }},
        {"nan-low-score-logit", [](Raw& x, auto&, auto&) { x.logit(8399, 79, NAN); }},
        {"inf-low-score-reg", [](Raw& x, auto&, auto&) { x.distances(8399, {1, INFINITY, 1, 1}); }},
        {"decode-overflow-low-score", [](Raw& x, auto&, auto&) { x.distances(8399, {1e38f, 1, 1, 1}); }},
        {"nms-area-overflow", [](Raw& x, auto&, auto&) { x.logit(0, 0, 0); x.box(0, -1e20f, -1e20f, 1e20f, 1e20f); }},
        {"nms-union-overflow", [](Raw& x, auto&, auto&) {
            for (unsigned i = 0; i < 2; ++i) { x.logit(i, 0, 0); x.box(i, 0, 0, 1.4e19f, 1.4e19f); }
        }},
        {"mapping-overflow", [](Raw& x, auto& r, auto&) {
            x.logit(0, 0, 0); x.box(0, 1e36f, 0, 4e36f, 0.000001f); r = request(65535, 65535);
        }},
        {"producer-not-done", [](Raw&, auto&, auto& p) { p.done = 0; }},
        {"producer-upstream-failure", [](Raw&, auto&, auto& p) { p.flags = 0x100; }},
        {"producer-both-failures", [](Raw&, auto&, auto& p) { p = {1, 0}; }},
        {"deterministic-dense-random", [](Raw& x, auto& r, auto&) {
            std::uint32_t state = 0x78ac431u;
            for (auto& l : x.cls) for (auto& v : l) {
                state = state * 1664525u + 1013904223u;
                v = -12.0f + static_cast<float>((state >> 8) & 65535u) / 8192.0f;
            }
            for (auto& l : x.reg) for (auto& v : l) {
                state = state * 1664525u + 1013904223u; v = static_cast<float>(state % 500u) / 128.0f;
            }
            r = request(117, 993);
        }},
    };
}
void invariant(const DetectionCandidates& output, const std::string& name) {
    const auto& h = output.header;
    std::uint32_t flags = 0;
    if (name == "nan-low-score-logit" || name == "inf-low-score-reg" || name == "decode-overflow-low-score" ||
        name == "nms-area-overflow" || name == "nms-union-overflow" || name == "mapping-overflow") flags = kPostprocessNumeric;
    if (name == "capacity-zero-overflow" || name == "capacity-one-short") flags = kPostprocessOverflow;
    if (name == "producer-not-done") flags = kPostprocessProtocol;
    if (name == "producer-upstream-failure") flags = kPostprocessUpstream;
    if (name == "producer-both-failures") flags = kPostprocessProtocol | kPostprocessUpstream;
    require(h.flags == flags && h.done == kPostprocessDone, name + ": fixture intended status");
    if (flags) require(h.count == 0, name + ": failure published rows");
    if (name == "minimum-anchor-adjacent-valid-box") require(h.count == 1 && output.rows[0].x1 == 0 &&
        output.rows[0].x2 == std::ldexp(1.0f, -22) && output.rows[0].y2 == std::ldexp(1.0f, -22), name + ": minimal anchor difference");
    if (name == "class-rounded-sigmoid-saturation" || name == "class-tie-lowest")
        require(h.count == 1 && output.rows[0].class_id == 0, name + ": lowest rounded class");
    if (name == "top300-before-person") require(h.count == 0 && h.class_dropped == 300 && h.topk_dropped == 1, name + ": global top-k");
    if (name == "degenerate-occupies-top300") require(h.count == 0 && h.degenerate == 300 && h.topk_dropped == 1, name + ": invalid top-k no refill");
    if (name == "top300-tie-id") require(h.count == 300 && output.rows[299].candidate_id == 299 && h.topk_dropped == 1, name + ": tie at top-k boundary");
    if (name == "nms-iou-exact-half" || name == "nms-iou-below-half") require(h.count == 2 && h.kept == 2, name + ": IoU boundary");
    if (name == "nms-iou-above-half") require(h.count == 2 && h.kept == 1 && h.suppressed == 1, name + ": IoU strict above");
    if (name == "nms-equal-score-id-suppressed-pool") require(h.count == 3 && h.kept == 1 && h.suppressed == 2 && output.rows[0].candidate_id == 2, name + ": complete stable pool");
    if (name == "nms-greedy-chain") require(h.count == 3 && h.kept == 2 && output.rows[1].flags == kDetectionSuppressed && output.rows[2].flags == kDetectionKept, name + ": kept-only suppression");
}
void compare(const DetectionCandidates& actual, const DetectionCandidates& reference, const std::string& name) {
    if (std::memcmp(&actual.header, &reference.header, sizeof(actual.header)) != 0) {
        std::cerr << name << " got count=" << actual.header.count << " flags=" << actual.header.flags
                  << " expected count=" << reference.header.count << " flags=" << reference.header.flags << '\n';
    }
    require(std::memcmp(&actual.header, &reference.header, sizeof(actual.header)) == 0, name + ": header differs");
    for (unsigned i = 0; i < kPostprocessCapacity; ++i) {
        auto a = actual.rows[i], b = reference.rows[i];
        if (i < reference.header.count) {
            const float error = std::abs(a.score - b.score);
            require(std::isfinite(a.score) && error <= 4 * std::numeric_limits<float>::epsilon(), name + ": score exceeds fixed bound");
            worst_score_error = std::max(worst_score_error, error); a.score = b.score = 0;
        }
        require(std::memcmp(&a, &b, sizeof(a)) == 0, name + ": row " + std::to_string(i) + " bits differ");
    }
}
struct Harness {
    Stream stream;
    Device raw{2822400}, producer{sizeof(PostprocessProducerStatus)}, output{sizeof(DetectionCandidates)}, workspace{postprocess_workspace_bytes()};
    PostprocessRawView device_view() const {
        PostprocessRawView v;
        std::size_t at = 0;
        for (unsigned l = 0; l < 3; ++l) {
            const auto n = kPostprocessSizes[l] * kPostprocessSizes[l];
            v.cls[l] = raw.as<float>() + at; v.cls_elements[l] = 80 * n; at += 80 * n;
            v.reg[l] = raw.as<float>() + at; v.reg_elements[l] = 4 * n; at += 4 * n;
        }
        return v;
    }
    void upload(const Raw& input, PostprocessProducerStatus status) {
        const auto view = device_view();
        for (unsigned l = 0; l < 3; ++l) {
            cu(cudaMemcpyAsync(const_cast<float*>(view.cls[l]), input.cls[l].data(), input.cls[l].size() * 4, cudaMemcpyHostToDevice, stream.value));
            cu(cudaMemcpyAsync(const_cast<float*>(view.reg[l]), input.reg[l].data(), input.reg[l].size() * 4, cudaMemcpyHostToDevice, stream.value));
        }
        // This is test input staging, not part of production F04 enqueue.
        cu(cudaMemcpyAsync(producer.p, &status, sizeof(status), cudaMemcpyHostToDevice, stream.value));
        cu(cudaStreamSynchronize(stream.value));
    }
    void launch(const PostprocessRequest& r, cudaStream_t st = nullptr) {
        postprocess_cuda(device_view(), r, producer.as<PostprocessProducerStatus>(), output.as<DetectionCandidates>(),
            sizeof(DetectionCandidates), workspace.p, postprocess_workspace_bytes(), st ? st : stream.value);
    }
    DetectionCandidates read() {
        DetectionCandidates out;
        cu(cudaMemcpyAsync(&out, output.p, sizeof(out), cudaMemcpyDeviceToHost, stream.value));
        cu(cudaStreamSynchronize(stream.value)); return out;
    }
    void close() {
        cu(cudaStreamSynchronize(stream.value)); CleanupReport report;
        workspace.close(report); output.close(report); producer.close(report); raw.close(report); stream.close(report);
        report.throw_if_failed();
    }
};
void directed_tests(bool graph) {
    Harness h;
    for (const auto& item : cases()) {
        Raw raw; auto r = request(); auto p = success; item.setup(raw, r, p);
        const auto expected = postprocess_reference(raw.view(), r, p);
        invariant(expected, item.name);
        h.upload(raw, p); cu(cudaMemsetAsync(h.output.p, 0xa5, sizeof(DetectionCandidates), h.stream.value));
        h.launch(r); const auto result = h.read(); compare(result, expected, item.name);
        invariant(result, item.name);
        h.launch(r); const auto repeat = h.read();
        require(std::memcmp(&result, &repeat, sizeof(result)) == 0, std::string(item.name) + ": repeat differs");
        if (graph) {
            cudaGraph_t capture = nullptr; cudaGraphExec_t executable = nullptr;
            cu(cudaStreamBeginCapture(h.stream.value, cudaStreamCaptureModeThreadLocal)); h.launch(r);
            cu(cudaStreamEndCapture(h.stream.value, &capture)); cu(cudaGraphInstantiate(&executable, capture, 0));
            cu(cudaGraphLaunch(executable, h.stream.value)); const auto replay = h.read();
            require(std::memcmp(&result, &replay, sizeof(result)) == 0, std::string(item.name) + ": graph differs");
            cu(cudaGraphExecDestroy(executable)); cu(cudaGraphDestroy(capture));
        }
        std::cout << "case=" << item.name << " count=" << result.header.count << " kept=" << result.header.kept
                  << " suppressed=" << result.header.suppressed << " flags=" << result.header.flags << " PASS\n";
    }
    // Hand-computed invariants independent of reference sorting/NMS.
    Raw raw; auto r = request(1280, 320); raw.logit(0, 0, 0); raw.distances(0, {1, 2, 3, 4});
    h.upload(raw, success); h.launch(r); const auto single = h.read();
    require(single.header.count == 1 && single.rows[0].x1 == -8 && single.rows[0].y1 == -6 &&
        single.rows[0].x2 == 56 && single.rows[0].y2 == 18 && single.rows[0].score == 0.5f, "manual anchor/ltrb/map/no-clip");
    for (unsigned i = 0; i < 3; ++i) { raw.logit(i, 0, 3.0f - i); raw.box(i, 4.0f * i, 0, 16.0f + 4.0f * i, 16); }
    h.upload(raw, success); h.launch(request()); const auto chain = h.read();
    require(chain.header.count == 3 && chain.rows[0].flags == kDetectionKept &&
        chain.rows[1].flags == kDetectionSuppressed && chain.rows[2].flags == kDetectionKept, "manual greedy chain");
    h.close();
}
void argument_tests() {
    Harness h; Raw raw; h.upload(raw, success); const auto good = h.device_view(); const auto r = request();
    const auto invoke = [&](auto view, auto req, const auto* producer, auto* output, std::size_t ob, void* workspace, std::size_t wb, cudaStream_t stream) {
        postprocess_cuda(view, req, producer, output, ob, workspace, wb, stream);
    };
    const auto call = [&](auto view, auto req, const PostprocessProducerStatus* producer,
                           DetectionCandidates* output, std::size_t ob, void* workspace, std::size_t wb, cudaStream_t stream) {
        reject([&] { invoke(view, req, producer, output, ob, workspace, wb, stream); });
    };
    auto* p = h.producer.as<PostprocessProducerStatus>(); auto* o = h.output.as<DetectionCandidates>();
    const auto ob = sizeof(DetectionCandidates), wb = postprocess_workspace_bytes(); auto* w = h.workspace.p; auto st = h.stream.value;
    cu(cudaMemsetAsync(o, 0xab, ob, st)); cu(cudaStreamSynchronize(st));
    for (auto stream : {cudaStream_t{}, cudaStreamLegacy, cudaStreamPerThread}) call(good, r, p, o, ob, w, wb, stream);
    call(good, r, nullptr, o, ob, w, wb, st); call(good, r, p, nullptr, ob, w, wb, st);
    call(good, r, p, o, ob - 1, w, wb, st); call(good, r, p, o, ob, nullptr, wb, st);
    call(good, r, p, o, ob, w, wb - 1, st);
    call(good, r, p, o, ob, o, wb, st);
    call(good, r, p, reinterpret_cast<DetectionCandidates*>(h.raw.p), ob, w, wb, st);
    call(good, r, reinterpret_cast<PostprocessProducerStatus*>(w), o, ob, w, wb, st);
    call(good, r, p, reinterpret_cast<DetectionCandidates*>(static_cast<char*>(h.output.p) + 1), ob, w, wb, st);
    call(good, r, p, o, ob, static_cast<char*>(w) + 1, wb, st);
    auto v = good; v.cls[0] = raw.cls[0].data(); call(v, r, p, o, ob, w, wb, st);
    v = good; v.cls[0] = nullptr; call(v, r, p, o, ob, w, wb, st);
    v = good; --v.cls_elements[0]; call(v, r, p, o, ob, w, wb, st);
    v = good; v.reg[0] = v.cls[0]; call(v, r, p, o, ob, w, wb, st);
    Device small(4); v = good; v.cls[0] = small.as<float>(); call(v, r, p, o, ob, w, wb, st);
    call(good, r, p, small.as<DetectionCandidates>(), ob, w, wb, st);
    call(good, r, small.as<PostprocessProducerStatus>(), o, ob, w, wb, st);
    call(good, r, p, o, ob, small.p, wb, st);
    v = good; v.cls[0] = good.cls[0] + 1; call(v, r, p, o, ob, w, wb, st);
    auto bad = r; bad.capacity = 301; call(good, bad, p, o, ob, w, wb, st);
    bad = r; bad.mapping.scale_x = NAN; call(good, bad, p, o, ob, w, wb, st);
    const auto untouched = h.read();
    std::array<unsigned char, sizeof(DetectionCandidates)> sentinel; sentinel.fill(0xab);
    require(std::memcmp(&untouched, sentinel.data(), sentinel.size()) == 0, "argument rejection enqueued output writes");
    CleanupReport cleanup; small.close(cleanup); cleanup.throw_if_failed(); h.close();
    std::cout << "arguments=PASS\n";
}
struct Hold { std::atomic<bool> started{false}, released{false}; };
void held(void* arg) {
    auto& hold = *static_cast<Hold*>(arg); hold.started.store(true);
    while (!hold.released.load()) std::this_thread::yield();
}
template<class F> void while_held(Hold& hold, F&& enqueue) {
    while (!hold.started.load()) std::this_thread::yield();
    std::atomic<bool> returned{false}; std::exception_ptr error;
    std::thread submit([&] {
        // Runtime device/context selection is per submitting host thread.
        try { cu(cudaSetDevice(0)); enqueue(); } catch (...) { error = std::current_exception(); }
        returned.store(true);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!returned.load() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool asynchronous = returned.load(); hold.released.store(true); submit.join();
    if (error) std::rethrow_exception(error);
    require(asynchronous, "F04 enqueue synchronized with a pending producer/consumer");
}
void stream_tests(bool skip_held) {
    Harness h; Raw raw; raw.logit(6400, 0, 1); h.upload(raw, success); auto r = request();
    // Producer-ready and consumer-consumed protect one slot across two streams.
    Stream consumer; cudaEvent_t ready = nullptr, consumed = nullptr;
    cu(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming)); cu(cudaEventCreateWithFlags(&consumed, cudaEventDisableTiming));
    Device saved(sizeof(DetectionCandidates));
    h.launch(r); cu(cudaEventRecord(ready, h.stream.value)); cu(cudaStreamWaitEvent(consumer.value, ready));
    Hold consumer_hold;
    if (!skip_held) cu(cudaLaunchHostFunc(consumer.value, held, &consumer_hold));
    cu(cudaMemcpyAsync(saved.p, h.output.p, sizeof(DetectionCandidates), cudaMemcpyDeviceToDevice, consumer.value));
    cu(cudaEventRecord(consumed, consumer.value));
    r.frame++;
    const auto reuse = [&] { cu(cudaStreamWaitEvent(h.stream.value, consumed)); h.launch(r); };
    if (!skip_held) while_held(consumer_hold, reuse); else reuse();
    const auto next = h.read();
    DetectionCandidates previous{}; cu(cudaMemcpy(&previous, saved.p, sizeof(previous), cudaMemcpyDeviceToHost));
    require(previous.header.frame + 1 == next.header.frame, "consumer lease lost frame identity");
    compare(previous, postprocess_reference(raw.view(), request(), success), "cross-stream-consumer");
    cu(cudaStreamSynchronize(consumer.value)); cu(cudaEventDestroy(consumed)); cu(cudaEventDestroy(ready));
    if (!skip_held) {
        Hold hold;
        cu(cudaLaunchHostFunc(h.stream.value, held, &hold));
        while_held(hold, [&] { h.launch(r); });
        compare(h.read(), postprocess_reference(raw.view(), r, success), "held-stream");
    }
    // Upstream raw predictions and status are still pending on another stream
    // when F04 is submitted. Stage with D2D to avoid pageable host copy waits.
    Harness source; Raw produced; produced.logit(8399, 0, 2); source.upload(produced, success);
    Raw old; h.upload(old, PostprocessProducerStatus{});
    cudaEvent_t producer_ready = nullptr; cu(cudaEventCreateWithFlags(&producer_ready, cudaEventDisableTiming));
    Hold producer_hold;
    if (!skip_held) cu(cudaLaunchHostFunc(consumer.value, held, &producer_hold));
    cu(cudaMemcpyAsync(h.raw.p, source.raw.p, 2822400, cudaMemcpyDeviceToDevice, consumer.value));
    cu(cudaMemcpyAsync(h.producer.p, source.producer.p, sizeof(PostprocessProducerStatus), cudaMemcpyDeviceToDevice, consumer.value));
    cu(cudaEventRecord(producer_ready, consumer.value));
    const auto consume = [&] { cu(cudaStreamWaitEvent(h.stream.value, producer_ready)); h.launch(r); };
    if (!skip_held) while_held(producer_hold, consume); else consume();
    compare(h.read(), postprocess_reference(produced.view(), r, success), "pending-upstream-producer");
    cu(cudaStreamSynchronize(consumer.value)); cu(cudaEventDestroy(producer_ready)); source.close();
    h.upload(raw, success);
    // A separate instance can overlap without sharing any scratch/output state.
    Harness other; Raw empty; other.upload(empty, success); other.launch(request(31, 991));
    h.launch(r); compare(other.read(), postprocess_reference(empty.view(), request(31, 991), success), "independent-instance");
    compare(h.read(), postprocess_reference(raw.view(), r, success), "independent-instance-original");
    CleanupReport cleanup; saved.close(cleanup); consumer.close(cleanup); cleanup.throw_if_failed(); other.close(); h.close();
    std::cout << "streams=PASS held=" << (skip_held ? "UNEXECUTED" : "PASS") << '\n';
}
void replay_tests(const std::string& directory) {
    // Independent pre-existing training-side decode of these exact native raw
    // dumps: evidence/t01/2026-10-30/f03/capture/f03-diagnostics.json.
    // This also guards the file schema: CPU/GPU agreement alone cannot detect
    // a parser that gives both implementations the same misordered tensors.
    constexpr std::array<unsigned, 6> historical_kept{32, 28, 39, 2, 3, 1};
    constexpr std::array<const char*, 6> hashes{
        "12df20281dd2bd1b4dd2aefbffcb2ce598115e993f530b1d3b82ae2dbdf763e5",
        "02558978a3e7437cc82c24c3532386588d596c2f9f54a6760d811a4543a9cf16",
        "b69dc2f3a468540d507a35ee5da767d7607c88948c68418b1dfa8296b55ea85f",
        "c273efb9986a957a79ff29ec2c9ffa2fb98469fe284f2d11c4f3fa63b559c815",
        "a6c72c191da4fad43ae298ea0f677037c1f8ae36d04618a1e7305890f37def57",
        "ccd755c5f7a087bce0168e36076e377053fc6678fe8b5dc0ad981fcd52e2e771"};
    Harness h;
    for (unsigned i = 0; i < hashes.size(); ++i) {
        const auto file = directory + "/input-" + std::to_string(i) + ".f32";
        const auto data = read_text_file(file);
        require(data.size() == 2822400 && sha256_hex(data) == hashes[i], "F03 replay raw identity mismatch: " + file);
        Raw raw; std::size_t at = 0;
        // model_io_spec()/F03 dump order is cls_p3, cls_p4, cls_p5,
        // reg_p3, reg_p4, reg_p5 (not interleaved cls/reg level pairs).
        for (unsigned kind = 0; kind < 2; ++kind) for (unsigned l = 0; l < 3; ++l) {
            auto& tensor = kind == 0 ? raw.cls[l] : raw.reg[l];
            std::memcpy(tensor.data(), data.data() + at, tensor.size() * 4); at += tensor.size() * 4;
        }
        auto r = i < 3 ? request(1920, 1080) : request(); r.frame = i;
        const auto reference = postprocess_reference(raw.view(), r, success);
        require(reference.header.flags == 0 && reference.header.kept == historical_kept[i],
                "F03 replay disagrees with independent historical kept count: input " + std::to_string(i));
        h.upload(raw, success); h.launch(r); const auto first = h.read();
        compare(first, reference, "F03/" + std::to_string(i));
        require(first.header.flags == 0, "recorded F03 raw must produce valid candidates");
        h.launch(r); const auto second = h.read();
        require(std::memcmp(&first, &second, sizeof(first)) == 0, "F03 replay rerun differs");
        std::cout << "f03=" << i << " raw_sha256=" << hashes[i] << " output_sha256="
                  << sha256_hex(std::string_view(reinterpret_cast<const char*>(&first), sizeof(first)))
                  << " count=" << first.header.count << " kept=" << first.header.kept << " suppressed=" << first.header.suppressed << " PASS\n";
    }
    h.close();
}
} // namespace
int main(int argc, char** argv) {
    try {
        bool streams_only = false, skip_held = false, list = false, check_cases = false; std::string dumps;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--streams-only") streams_only = true;
            else if (arg == "--no-held-stream") skip_held = true;
            else if (arg == "--list-cases") list = true;
            else if (arg == "--check-fixtures") check_cases = true;
            else if (arg == "--f03-dumps" && i + 1 < argc) dumps = argv[++i];
            else throw std::invalid_argument("usage: postprocess_cuda_tests [--streams-only] [--no-held-stream] [--list-cases] [--f03-dumps DIR]");
        }
        if (list) { for (const auto& c : cases()) std::cout << c.name << '\n'; return 0; }
        if (check_cases) {
            for (const auto& c : cases()) { Raw raw; auto r = request(); auto p = success; c.setup(raw, r, p);
                invariant(postprocess_reference(raw.view(), r, p), c.name); }
            std::cout << "CPU fixture invariants=" << checks << " PASS\n"; return 0;
        }
        cu(cudaSetDevice(0)); std::cout << postprocess_cuda_load_kernels() << postprocess_cuda_identity();
        if (!streams_only) { directed_tests(true); argument_tests(); }
        stream_tests(skip_held);
        if (!dumps.empty()) replay_tests(dumps);
        std::cout << "checks=" << checks << " max_score_abs_error=" << worst_score_error << " PASS\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

#include "saccade/detection_session.hpp"
#include "detection_session_test_access.hpp"
#include "saccade/file_io.hpp"
#include "saccade/model_producer_cuda.hpp"
#include "saccade/model_trt_network.hpp"
#include "saccade/postprocess_cuda.hpp"
#include "saccade/preprocess.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/sha256.hpp"

#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// This executable uses real, frozen JPEGs. The independent serial harness
// observes F01/F02 and raw predictions, then compares F04 to its CPU reference.
// Those diagnostic readbacks are outside the session's normal data path.
namespace {
using namespace saccade;
using namespace nvinfer1;
std::size_t checks = 0;
void check(bool ok, const std::string& why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
void cu(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class F> void rejects(F&& f, const std::string& label) {
    bool threw = false;
    try { f(); } catch (const std::exception& e) {
        threw = true;
        std::cout << "rejection=" << label << " error=" << e.what() << '\n';
    }
    check(threw, "expected rejection: " + label);
}
template<class T> std::string hash(const T* p, std::size_t n) {
    return sha256_hex(std::string_view(reinterpret_cast<const char*>(p), n * sizeof(T)));
}
template<class T> void save(const std::filesystem::path& path, const T& data) {
    write_text_file(path, std::string_view(reinterpret_cast<const char*>(&data), sizeof(data)));
}
struct Device {
    void* p{};
    explicit Device(std::size_t size) { cu(cudaMalloc(&p, size)); }
    ~Device() { if (p) (void)cudaFree(p); }
    Device(const Device&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
};
template<class T> struct Pinned {
    T* p{};
    explicit Pinned(std::size_t count) { cu(cudaMallocHost(reinterpret_cast<void**>(&p), count * sizeof(T))); }
    ~Pinned() { if (p) (void)cudaFreeHost(p); }
    Pinned(const Pinned&) = delete;
};
struct Stream {
    cudaStream_t s{};
    Stream() { cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { if (s) { (void)cudaStreamSynchronize(s); (void)cudaStreamDestroy(s); } }
};
class Logger final : public ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << "TensorRT: " << message << '\n';
    }
};
struct CorpusCase {
    const char* label;
    int width, height;
    const char* jpeg_sha;
    const char* rgb_sha;
    const char* images_sha;
};
constexpr std::array<CorpusCase, 7> kCorpus{{
    {"MOT17-02-SDP/1", 1920, 1080, "41ff65408c5338f4a83035fd12e9f5f7b658ee82feacf4512655677032636597", "87cf5700dc92b3827ebe883f5bebabe9206a7257935cb5969c20a3bb5a1d6137", "a99482f804318d2348a49903b69f85bc0a3aa610d2880e223f591e3118fedfa2"},
    {"MOT17-10-SDP/1", 1920, 1080, "bc655b86e7dcd9993df950217534f2f6cbf85217faba544dce956b834fb053a0", "6137445b8be2856f9b7ad667fddc44732ec2f3321c0d5d88ad50257efa99b40c", "ca9388d1e5042961d708d103681a5e2fc54c5e4f20940c655a775df936e2cb57"},
    {"MOT17-13-SDP/400", 1920, 1080, "eab49b69eeb4816aaac41f2699479a4ead316672e6d1fef49efd912721eea05a", "f945138302ca2ca6b504116cd89f78f360f150dbbcba10729215461993a44fdc", "4a30d9421e3c414bfab6d08f7db83b020da0046abd8a49158e32b3394fc1f738"},
    {"MOT17-04-SDP/1", 1920, 1080, "53f3d86aad1aa4912b2919bae344919b54d193e72c1c7169ce8bc4685408053c", "f766efc6a166f27a8a2f54305463aad32a119c0a44105fc2ffbc89cc384157ac", "3fc9b2d6f7877fc4c3566f9b3d0d4fe1fda0e1a7ad2631164b50bbbae03b34b5"},
    {"MOT17-05-SDP/1", 640, 480, "9f4f112b29e93846501db27d2bcc2c99795715aa6376d5b5662d75f2e2f196d7", "66fb615e61936a227873612121304c9fe41923750b4759415a0d0036cb108bb1", "5da331cb78240c138a53e4fb25c27e31ae18134220fee1d034ba137b569b5a84"},
    {"MOT17-09-SDP/1", 1920, 1080, "e4b1c40c44d53489f191e1099156ad57d3963d8c434615750fba0065a8a572af", "f2ae8a4639cba20cfb08beb099d5490d6ecdbbb92280dacab3b9cbc5a0356ea2", "61f60829230cd8d8fa1c0764aee5bdab1cb3dec788a6a647ee8c7d8c007c4065"},
    {"MOT17-11-SDP/1", 1920, 1080, "45c999f9f2bff3d308cf8cd877ff63b95394f53e7319c1b9c444acff77d73b3c", "c4f358dc4ae246c143fab4e0211e114c4518189878e687d89196aee17dc92fab", "301a813069e5c96d3db4a3fccad3921641e55e506ac0b7a2c15045ba611d6942"},
}};
constexpr const char* kEngineSha = "c6cff1936194e679ce11c3736cab552e4a73a2b1a9ae5508bc3e5d1b3fbab9c5";
constexpr const char* kManifestSha = "a3b08bdfe20f13f122e63a0314969832622daf8cedf0e29330e4b325471af5ff";
constexpr std::array<const char*, 3> kRawSha{
    "12df20281dd2bd1b4dd2aefbffcb2ce598115e993f530b1d3b82ae2dbdf763e5",
    "02558978a3e7437cc82c24c3532386588d596c2f9f54a6760d811a4543a9cf16",
    "b69dc2f3a468540d507a35ee5da767d7607c88948c68418b1dfa8296b55ea85f"};
constexpr std::array<unsigned, 3> kKept{32, 28, 39};
struct Corpus {
    std::array<std::vector<std::uint8_t>, 7> jpeg;
    explicit Corpus(const std::filesystem::path& root) {
        check(sha256_hex(read_text_file(root / "manifest.json")) == kManifestSha, "frozen reference manifest");
        for (std::size_t i = 0; i < kCorpus.size(); ++i) {
            const auto name = "case-00" + std::to_string(i);
            const auto encoded = read_text_file(root / (name + ".jpg"));
            check(sha256_hex(encoded) == kCorpus[i].jpeg_sha, "pinned JPEG: " + name);
            check(sha256_hex(read_text_file(root / (name + ".rgb"))) == kCorpus[i].rgb_sha, "pinned RGB: " + name);
            check(sha256_hex(read_text_file(root / (name + ".f32"))) == kCorpus[i].images_sha, "pinned FP32: " + name);
            jpeg[i] = {encoded.begin(), encoded.end()};
            std::cout << "corpus=" << i << " label=" << kCorpus[i].label << " jpeg_sha256=" << kCorpus[i].jpeg_sha << '\n';
        }
    }
};
std::size_t elements(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (const auto d : shape) { check(d > 0, "static I/O dimensions"); n *= static_cast<std::size_t>(d); }
    return n;
}
void cpu_equal(const DetectionCandidates& a, const DetectionCandidates& b) {
    check(std::memcmp(&a.header, &b.header, sizeof(a.header)) == 0, "independent CPU reference full header");
    for (unsigned i = 0; i < kPostprocessCapacity; ++i) {
        auto x = a.rows[i], y = b.rows[i];
        if (i < b.header.count) {
            check(std::isfinite(x.score) && std::abs(x.score - y.score) <= 4 * std::numeric_limits<float>::epsilon(),
                  "independent CPU reference score bound");
            x.score = y.score = 0;
        }
        check(std::memcmp(&x, &y, sizeof(x)) == 0, "independent CPU reference complete row/padding");
    }
}
std::array<DetectionCandidates, 7> independent(const std::string& engine_path, const Corpus& corpus,
                                             const std::filesystem::path& out, unsigned count) {
    const auto plan = read_text_file(engine_path);
    check(sha256_hex(plan) == kEngineSha, "accepted pinned F03 engine");
    Logger logger;
    register_scan_trt_plugins(*getPluginRegistry());
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    check(runtime != nullptr, "reference runtime");
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    check(engine != nullptr && engine->getNbAuxStreams() == 0, "reference engine with zero aux streams");
    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
    check(context != nullptr, "reference execution context");
    Stream d;
    NativePreprocessor preprocess(d.s);
    Device raw(705600 * sizeof(float)), statuses(9 * 256), aggregate(sizeof(PostprocessProducerStatus));
    Device producer_scratch(model_producer_workspace_bytes()), f04_scratch(postprocess_workspace_bytes());
    Device candidates(sizeof(DetectionCandidates));
    Pinned<float> hraw(705600), himages(kModelProducerInputElements);
    Pinned<std::uint8_t> hrgb(3 * 1920 * 1080);
    Pinned<DetectionCandidates> hresult(1);
    Pinned<PostprocessProducerStatus> haggregate(1);
    const auto spec = model_io_spec();
    ModelProducerPlugins plugins{};
    PostprocessRawView view, hview;
    std::size_t offset = 0;
    for (unsigned i = 0; i < 6; ++i) {
        const auto n = elements(spec.outputs[i].shape);
        check(context->setTensorAddress(spec.outputs[i].name.c_str(), raw.as<float>() + offset), "reference raw binding");
        if (i < 3) {
            view.cls[i] = raw.as<float>() + offset; view.cls_elements[i] = n;
            hview.cls[i] = hraw.p + offset; hview.cls_elements[i] = n;
        } else {
            view.reg[i - 3] = raw.as<float>() + offset; view.reg_elements[i - 3] = n;
            hview.reg[i - 3] = hraw.p + offset; hview.reg_elements[i - 3] = n;
        }
        offset += n;
    }
    check(offset == 705600, "reference raw layout");
    for (unsigned i = 0; i < 9; ++i) {
        plugins[i] = reinterpret_cast<ScanStatus*>(statuses.as<std::uint8_t>() + i * 256);
        check(context->setTensorAddress(spec.statuses[i].name.c_str(), plugins[i]), "reference plugin binding");
    }
    cu(cudaMemsetAsync(statuses.p, 0, 9 * 256, d.s));
    model_producer_cuda_load_kernels(); postprocess_cuda_load_kernels();
    std::array<DetectionCandidates, 7> outputs{};
    for (unsigned i = 0; i < count; ++i) {
        const auto frame = preprocess.enqueue(corpus.jpeg[i], i, kCorpus[i].width, kCorpus[i].height);
        check(context->setTensorAddress(spec.inputs[0].name.c_str(), frame.images), "reference image binding");
        cu(cudaMemcpyAsync(hrgb.p, frame.rgb, 3 * kCorpus[i].width * kCorpus[i].height, cudaMemcpyDeviceToHost, d.s));
        cu(cudaMemcpyAsync(himages.p, frame.images, kModelProducerInputElements * sizeof(float), cudaMemcpyDeviceToHost, d.s));
        check(context->enqueueV3(d.s), "reference F03 enqueue");
        model_producer_cuda(frame.images, kModelProducerInputElements, view, plugins,
                            aggregate.as<PostprocessProducerStatus>(), producer_scratch.p, model_producer_workspace_bytes(), d.s);
        PostprocessRequest request;
        request.epoch = 0; request.frame = 0; request.mapping = frame.mapping;
        postprocess_cuda(view, request, aggregate.as<PostprocessProducerStatus>(), candidates.as<DetectionCandidates>(),
                         sizeof(DetectionCandidates), f04_scratch.p, postprocess_workspace_bytes(), d.s);
        cu(cudaMemcpyAsync(hraw.p, raw.p, 705600 * sizeof(float), cudaMemcpyDeviceToHost, d.s));
        cu(cudaMemcpyAsync(hresult.p, candidates.p, sizeof(DetectionCandidates), cudaMemcpyDeviceToHost, d.s));
        cu(cudaMemcpyAsync(haggregate.p, aggregate.p, sizeof(PostprocessProducerStatus), cudaMemcpyDeviceToHost, d.s));
        cu(cudaStreamSynchronize(d.s));
        check(hash(hrgb.p, static_cast<std::size_t>(3) * kCorpus[i].width * kCorpus[i].height) == kCorpus[i].rgb_sha,
              "native full RGB equals frozen independent decoder reference");
        check(hash(himages.p, kModelProducerInputElements) == kCorpus[i].images_sha,
              "native complete FP32 equals frozen training input");
        check(haggregate.p->flags == 0 && haggregate.p->done == kPostprocessProducerDone, "reference producer status");
        cpu_equal(*hresult.p, postprocess_reference(hview, request, *haggregate.p));
        if (i < kRawSha.size()) {
            check(hash(hraw.p, 705600) == kRawSha[i], "historical raw prediction bytes");
            check(hresult.p->header.kept == kKept[i], "independent historical kept count");
        }
        outputs[i] = *hresult.p;
        save(out / ("reference-" + std::to_string(i) + ".bin"), outputs[i]);
        std::cout << "independent_input=" << i << " kept=" << outputs[i].header.kept
                  << " suppressed=" << outputs[i].header.suppressed << " raw_sha256=" << hash(hraw.p, 705600)
                  << " candidates_sha256=" << hash(&outputs[i], 1) << '\n';
    }
    preprocess.close();
    return outputs;
}

// The session-specific verification below deliberately compares all bytes,
// including suppressed detections, unused rows and reserved header fields.
constexpr std::array<unsigned, 13> kSequence{0, 4, 1, 4, 2, 4, 3, 4, 5, 4, 6, 4, 0};
void identity(const DetectionSessionResult& result, std::uint64_t epoch, std::uint64_t frame,
              std::uint32_t capacity = kPostprocessCapacity) {
    check(result.epoch == epoch && result.frame == frame, "session result epoch/frame");
    check(result.reserved == 0, "session result reserved bytes");
    const auto& h = result.candidates.header;
    check(h.epoch == epoch && h.frame == frame, "candidate/result identity agreement");
    check(h.schema == kPostprocessSchema && h.done == kPostprocessDone && h.capacity == capacity && h.reserved == 0,
          "candidate fixed schema/completion/capacity/reserved");
}
void accepted(const DetectionSessionResult& result, const DetectionCandidates& reference,
              std::uint64_t epoch, std::uint64_t frame) {
    identity(result, epoch, frame);
    check(result.status == DetectionSessionStatus::Ok && result.fail_code == 0 &&
          result.fail_frame == kDetectionSessionNoFrame && result.committed_frame == frame, "successful session commit");
    auto actual = result.candidates;
    actual.header.epoch = reference.header.epoch; actual.header.frame = reference.header.frame;
    check(std::memcmp(&actual, &reference, sizeof(actual)) == 0, "complete session candidates equal independent serial baseline");
}
void failed(const DetectionSessionResult& result, std::uint64_t epoch, std::uint64_t frame,
            std::uint64_t fail_frame, std::uint64_t committed, std::uint32_t code, std::uint32_t capacity = kPostprocessCapacity) {
    identity(result, epoch, frame, capacity);
    check(result.status == (frame == fail_frame ? DetectionSessionStatus::Failed : DetectionSessionStatus::SkippedAfterFailure),
          "device failure/skip status");
    check(result.fail_frame == fail_frame && result.committed_frame == committed && result.fail_code == code,
          "first failure and last committed frame are preserved");
    DetectionCandidates expected{};
    expected.header.epoch = epoch; expected.header.frame = frame;
    expected.header.schema = kPostprocessSchema; expected.header.capacity = capacity;
    expected.header.done = kPostprocessDone; expected.header.flags = code;
    check(std::memcmp(&result.candidates, &expected, sizeof(expected)) == 0, "failure/skip clears all rows and statistics");
}
void normal_sessions(const std::string& engine, const Corpus& corpus, const std::array<DetectionCandidates, 7>& baseline,
                     const std::filesystem::path& out, bool sanitizer) {
    const unsigned count = sanitizer ? 3 : static_cast<unsigned>(kSequence.size());
    for (const unsigned slots : {1u, 2u, 3u}) {
        DetectionSessionConfig config;
        config.epoch = 0xF0100000u + slots;
        config.candidate_slots = slots;
        config.result_slots = slots == 1 ? 1 : 5;
        DetectionSession session(engine, config);
        check(session.state() == DetectionSessionState::Running && session.pending() == 0, "new session starts empty and running");
        rejects([&] { (void)session.receive(); }, "receive-empty");
        const auto consume = [&](unsigned frame) {
            const auto source = sanitizer ? frame : kSequence[frame];
            const auto result = session.receive();
            accepted(result, baseline[source], config.epoch, frame);
            save(out / ("slots-" + std::to_string(slots) + "-frame-" + std::to_string(frame) + ".bin"), result);
        };
        unsigned next_receive = 0;
        for (unsigned frame = 0; frame < count; ++frame) {
            if (session.pending() == config.result_slots) consume(next_receive++);
            const auto source = sanitizer ? frame : kSequence[frame];
            auto copied = corpus.jpeg[source];
            session.enqueue(copied, frame, kCorpus[source].width, kCorpus[source].height);
            std::fill(copied.begin(), copied.end(), 0); // caller bytes are borrowed only during enqueue
            check(session.pending() == frame + 1 - next_receive, "pending includes all unreceived frames");
            if (slots == 1) consume(next_receive++); // true serial receipt baseline
        }
        session.finish(); session.finish();
        check(session.state() == DetectionSessionState::Finished, "normal finish reports EOF after checked drain");
        while (next_receive < count) consume(next_receive++);
        check(session.pending() == 0, "normal finish retains every unreceived result");
        rejects([&] { session.enqueue(corpus.jpeg[0], count); }, "enqueue-after-finish");
        const auto runtime_identity = session.identity();
        check(runtime_identity.find(kDetectionSessionPolicy) != std::string::npos, "session policy in runtime identity");
        write_text_file(out / ("slots-" + std::to_string(slots) + "-identity.txt"), runtime_identity);
        session.close(); session.close();
        check(session.state() == DetectionSessionState::Closed, "checked close consumes ownership");
        rejects([&] { session.enqueue(corpus.jpeg[0], count); }, "enqueue-after-close");
        rejects([&] { (void)session.receive(); }, "receive-after-close");
        std::cout << "case=normal slots=" << slots << " result_slots=" << config.result_slots
                  << " frames=" << count << " serial_queued_full_bytes=PASS\n";
    }
}
void backpressure(const std::string& engine, const Corpus& corpus, const DetectionCandidates& reference) {
    DetectionSessionConfig config;
    config.epoch = 0xF0200001; config.candidate_slots = 1; config.result_slots = 2;
    DetectionSession session(engine, config);
    session.enqueue(corpus.jpeg[0], 0); session.enqueue(corpus.jpeg[0], 1);
    rejects([&] { session.enqueue(corpus.jpeg[0], 2); }, "result-ring-full");
    check(session.pending() == 2 && session.state() == DetectionSessionState::Running,
          "backpressure is recoverable and cannot discard/submit frame");
    accepted(session.receive(), reference, config.epoch, 0);
    session.enqueue(corpus.jpeg[0], 2);
    session.finish();
    accepted(session.receive(), reference, config.epoch, 1);
    accepted(session.receive(), reference, config.epoch, 2);
    session.close();
    std::cout << "case=result-ring-backpressure no-drop/retry-same-frame=PASS\n";
}
void device_failures(const std::string& engine, const Corpus& corpus, const DetectionCandidates& reference,
                     const std::filesystem::path& out) {
    for (unsigned scenario = 0; scenario < 3; ++scenario) {
        DetectionSessionConfig config;
        config.epoch = 0xF0300010 + scenario; config.candidate_slots = 1; config.result_slots = 4;
        if (scenario == 2) config.output_capacity = 1;
        DetectionSession session(engine, config);
        const auto fail_frame = scenario == 2 ? 0u : 1u;
        const auto first_code = scenario == 0 ? kPostprocessUpstream :
                                scenario == 1 ? kPostprocessProtocol : kPostprocessOverflow;
        if (scenario < 2) session.enqueue(corpus.jpeg[0], 0);
        DetectionSessionTestAccess::enqueue(session, corpus.jpeg[0], fail_frame,
                                            scenario == 0 ? static_cast<std::uint32_t>(kPostprocessNumeric) : 0u, scenario == 1);
        session.enqueue(corpus.jpeg[0], fail_frame + 1);
        DetectionSessionTestAccess::enqueue(session, corpus.jpeg[0], fail_frame + 2, kPostprocessProtocol);
        session.finish();
        if (scenario < 2) accepted(session.receive(), reference, config.epoch, 0);
        for (unsigned frame = fail_frame; frame <= fail_frame + 2; ++frame) {
            const auto result = session.receive();
            failed(result, config.epoch, frame, fail_frame,
                   fail_frame ? 0 : kDetectionSessionNoFrame, first_code, config.output_capacity);
            save(out / ("failure-" + std::to_string(scenario) + "-frame-" + std::to_string(frame) + ".bin"), result);
        }
        check(session.state() == DetectionSessionState::Failed, "device failure cannot become normal EOF");
        rejects([&] { session.enqueue(corpus.jpeg[0], fail_frame + 3); }, "enqueue-after-observed-device-failure");
        session.close(); session.close();
        std::cout << "case=device-failure scenario=" << scenario << " first_code=" << first_code
                  << " first-failure/skip/zero-output=PASS\n";
    }
}
void host_failures(const std::string& engine, const Corpus& corpus, const DetectionCandidates& reference) {
    for (unsigned scenario = 0; scenario < 5; ++scenario) {
        DetectionSessionConfig config;
        config.epoch = 0xF0400000 + scenario; config.result_slots = 3;
        DetectionSession session(engine, config);
        session.enqueue(corpus.jpeg[0], 0);
        switch (scenario) {
            case 0: rejects([&] { session.enqueue({}, 1); }, "empty-jpeg"); break;
            case 1: {
                const std::array<std::uint8_t, 7> malformed{'n', 'o', 't', 'j', 'p', 'e', 'g'};
                rejects([&] { session.enqueue(malformed, 1); }, "invalid-jpeg"); break;
            }
            case 2: rejects([&] { session.enqueue(corpus.jpeg[0], 1, 1921, 1080); }, "metadata-mismatch"); break;
            case 3: rejects([&] { session.enqueue(corpus.jpeg[0], 2); }, "missing-frame"); break;
            default: rejects([&] { session.enqueue(corpus.jpeg[0], 0); }, "duplicate-frame"); break;
        }
        check(session.state() == DetectionSessionState::Failed && session.pending() == 1, "host errors latch without inventing a result");
        rejects([&] { session.enqueue(corpus.jpeg[0], 1); }, "enqueue-after-host-failure");
        session.finish();
        accepted(session.receive(), reference, config.epoch, 0);
        check(session.state() == DetectionSessionState::Failed, "host failure survives terminal drain");
        session.close(); session.close();
    }
    std::cout << "case=host-failures malformed/metadata/missing/duplicate/latch/preserve-prefix=PASS\n";
}
void lifecycle(const std::string& engine, const Corpus& corpus, const DetectionCandidates& reference,
               const std::filesystem::path& out) {
    DetectionSessionConfig config;
    config.epoch = 0xF0500001;
    {
        DetectionSession session(engine, config);
        session.finish(); session.finish();
        check(session.state() == DetectionSessionState::Finished && session.pending() == 0, "empty normal EOF");
        session.close(); session.close();
    }
    {
        DetectionSession session(engine, config);
        session.enqueue(corpus.jpeg[0], 0);
        session.cancel(); session.cancel();
        check(session.state() == DetectionSessionState::Cancelled && session.pending() == 1,
              "cancellation drains but remains distinct from EOF");
        accepted(session.receive(), reference, config.epoch, 0);
        rejects([&] { session.enqueue(corpus.jpeg[0], 1); }, "enqueue-after-cancel");
        session.close();
    }
    {
        DetectionSession session(engine, config);
        session.enqueue(corpus.jpeg[0], 0);
        // Checked close must drain even if the caller does not receive first.
        session.close(); session.close();
        check(session.state() == DetectionSessionState::Closed, "close pending unreceived work");
    }
    {
        DetectionSession session(engine, config);
        session.enqueue(corpus.jpeg[0], 0); // destructor owns both in-flight streams
    }
    // A new epoch after failure/cancel/close/destructor starts without stale
    // plugin markers, candidate generations or a previous session's failure.
    config.epoch = 0xF0500002;
    DetectionSession fresh(engine, config);
    fresh.enqueue(corpus.jpeg[0], 0); fresh.finish();
    const auto result = fresh.receive();
    accepted(result, reference, config.epoch, 0);
    save(out / "fresh-after-lifecycle.bin", result);
    fresh.close();
    auto invalid = config; invalid.candidate_slots = 0;
    rejects([&] { DetectionSession bad(engine, invalid); }, "zero-candidate-ring");
    invalid = config; invalid.result_slots = 0;
    rejects([&] { DetectionSession bad(engine, invalid); }, "zero-result-ring");
    invalid = config; invalid.preprocess.output_width = 32;
    rejects([&] { DetectionSession bad(engine, invalid); }, "wrong-model-shape");
    invalid = config; invalid.output_capacity = kPostprocessCapacity + 1;
    rejects([&] { DetectionSession bad(engine, invalid); }, "output-capacity-range");
    rejects([&] { DetectionSession bad((out / "missing.engine").string(), config); }, "missing-engine");
    write_text_file(out / "malformed.engine", "not a TensorRT engine");
    rejects([&] { DetectionSession bad((out / "malformed.engine").string(), config); }, "malformed-engine");
    std::cout << "case=lifecycle empty-EOF/cancel/finish/checked-close/destructor/fresh-epoch/config/engine=PASS\n";
}
struct GateState {
    std::atomic<bool> started{false}, release{false}, finished{false}, timed_out{false};
};
void held_callback(void* argument) {
    std::unique_ptr<std::shared_ptr<GateState>> owner(static_cast<std::shared_ptr<GateState>*>(argument));
    const auto state = *owner;
    state->started.store(true);
    while (!state->release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    state->finished.store(true);
}
struct Gate {
    std::shared_ptr<GateState> state = std::make_shared<GateState>();
    std::jthread watchdog;
    explicit Gate(cudaStream_t stream) {
        auto* payload = new std::shared_ptr<GateState>(state);
        const auto rc = cudaLaunchHostFunc(stream, held_callback, payload);
        if (rc != cudaSuccess) { delete payload; cu(rc); }
        watchdog = std::jthread([s = state](std::stop_token stop) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!s->release.load() && !stop.stop_requested()) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    s->timed_out.store(true); s->release.store(true); break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!state->started.load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        check(state->started.load(), "consumer gate callback starts");
    }
    ~Gate() { state->release.store(true); watchdog.request_stop(); }
    void release() { state->release.store(true); }
};
void pending_consumer(const std::string& engine, const Corpus& corpus, const DetectionCandidates& reference) {
    for (unsigned slots : {1u, 2u}) {
        DetectionSessionConfig config;
        config.epoch = 0xF0600000 + slots; config.candidate_slots = slots; config.result_slots = 5;
        DetectionSession session(engine, config);
        // Warm the real nvJPEG handle once before perturbing steady-state
        // scheduling; first-use library work belongs to initialization.
        session.enqueue(corpus.jpeg[0], 0);
        accepted(session.receive(), reference, config.epoch, 0);
        Gate gate(DetectionSessionTestAccess::consumer_stream(session));
        // The final enqueue reuses a device slot while M is held. Correct D
        // queues its consumed wait without blocking H or overwriting the slot.
        for (unsigned frame = 1; frame <= slots + 1; ++frame) session.enqueue(corpus.jpeg[0], frame);
        check(!gate.state->release.load(), "enqueue returns while candidate consumer remains held");
        check(cudaStreamQuery(DetectionSessionTestAccess::consumer_stream(session)) == cudaErrorNotReady,
              "directed pending consumer observed at reuse");
        gate.release();
        session.finish();
        check(gate.state->finished.load() && !gate.state->timed_out.load(), "consumer gate completed without watchdog release");
        for (unsigned frame = 1; frame <= slots + 1; ++frame) accepted(session.receive(), reference, config.epoch, frame);
        session.close();
        std::cout << "case=pending-consumer slots=" << slots << " reuse-before-consumed=PASS\n";
    }
}
void detect_only_tail(const std::string& engine, const Corpus& corpus, bool sanitizer) {
    for (unsigned terminal = 0; terminal < 4; ++terminal) {
        DetectionSessionConfig config;
        config.epoch = 0xF0700000 + terminal;
        auto session = std::make_unique<DetectionSession>(engine, config);
        session->enqueue(corpus.jpeg[0], 0);
        const auto primed = session->receive();
        check(primed.status == DetectionSessionStatus::Ok, "D-only tail starts after a healthy primed frame");
        std::unique_ptr<Gate> gate;
        const auto detect = DetectionSessionTestAccess::detect_stream(*session);
        if (!sanitizer) gate = std::make_unique<Gate>(detect);
        rejects([&] { DetectionSessionTestAccess::enqueue_d_only(*session, corpus.jpeg[0], 1); }, "D-only-tail-after-F02");
        check(session->state() == DetectionSessionState::Failed && session->pending() == 0,
              "D-only failed submission cannot create an M result");
        if (gate) check(cudaStreamQuery(detect) == cudaErrorNotReady, "D-only tail is outstanding independently of M");
        std::jthread release;
        if (gate) release = std::jthread([state = gate->state] {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
            state->release.store(true);
        });
        switch (terminal) {
            case 0: session->finish(); break;
            case 1: session->cancel(); break;
            case 2: session->close(); break;
            default: session.reset(); break;
        }
        if (gate) check(gate->state->finished.load() && !gate->state->timed_out.load(), "terminal lifecycle drains D-only tail");
        if (session && terminal < 2) {
            check(session->state() == DetectionSessionState::Failed, "D-only failure is retained after drain");
            check(cudaStreamQuery(detect) == cudaSuccess, "D terminal includes all tail work");
            session->close();
        }
        std::cout << "case=D-only-tail terminal=" << terminal << " independent-D-drain=PASS callback_gate="
                  << (sanitizer ? "UNEXECUTED_SANITIZER" : "PASS") << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool sanitizer = argc == 5 && std::string(argv[4]) == "--sanitizer";
        if (argc != 4 && !sanitizer)
            throw std::invalid_argument("usage: detection_session_tests ENGINE REFERENCE_DIRECTORY OUTPUT_DIRECTORY [--sanitizer]");
        const std::filesystem::path out(argv[3]);
        std::filesystem::create_directories(out);
        cu(cudaSetDevice(0));
        const Corpus corpus(argv[2]);
        const auto baseline = independent(argv[1], corpus, out, sanitizer ? 3 : 7);
        normal_sessions(argv[1], corpus, baseline, out, sanitizer);
        backpressure(argv[1], corpus, baseline[0]);
        device_failures(argv[1], corpus, baseline[0], out);
        host_failures(argv[1], corpus, baseline[0]);
        lifecycle(argv[1], corpus, baseline[0], out);
        detect_only_tail(argv[1], corpus, sanitizer);
        if (!sanitizer) pending_consumer(argv[1], corpus, baseline[0]);
        std::cout << "coverage=independent-reference,serial,queued,rings123,backpressure,first-failure,skip,overflow,identity,"
                     "malformed,metadata,missing,duplicate,cancel,EOF,D-only-tail,drain,close,destructor,fresh-epoch\n";
        std::cout << "callback_gates=" << (sanitizer ? "UNEXECUTED_SANITIZER" : "PASS") << '\n';
        std::cout << "detection_session=PASS checks=" << checks << " mode=" << (sanitizer ? "sanitizer" : "normal") << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "detection_session=FAIL checks=" << checks << " error=" << e.what() << '\n';
        return 1;
    }
}

#include "saccade/tracking_session.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/file_io.hpp"
#include "saccade/model_producer_cuda.hpp"
#include "saccade/model_trt_network.hpp"
#include "saccade/postprocess_cuda.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/sha256.hpp"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace saccade {
namespace {
void cu(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc));
}
void require(bool ok, const char* what) { if (!ok) throw std::invalid_argument(what); }
std::size_t elements(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) { require(d > 0, "session requires static positive model dimensions"); n *= static_cast<std::size_t>(d); }
    return n;
}
void validate_tensor(const nvinfer1::ICudaEngine& engine, const std::string& name,
                     nvinfer1::DataType type, nvinfer1::TensorIOMode mode,
                     const std::vector<std::int64_t>& shape, std::set<std::string>& names) {
    require(names.insert(name).second, "duplicate model tensor specification");
    require(engine.getTensorDataType(name.c_str()) == type && engine.getTensorIOMode(name.c_str()) == mode,
            "session model tensor mode/type mismatch");
    const auto d = engine.getTensorShape(name.c_str());
    require(d.nbDims == static_cast<int>(shape.size()), "session model tensor rank mismatch");
    for (int i = 0; i < d.nbDims; ++i) require(d.d[i] == shape[static_cast<std::size_t>(i)], "session model tensor shape mismatch");
    require(engine.getTensorLocation(name.c_str()) == nvinfer1::TensorLocation::kDEVICE &&
            engine.getTensorFormat(name.c_str()) == nvinfer1::TensorFormat::kLINEAR && !engine.isShapeInferenceIO(name.c_str()),
            "session requires linear device execution tensors");
}
class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << "TrackingSession TensorRT: " << message << '\n';
    }
};
}

struct TrackingSession::Impl {
    struct CandidateSlot {
        DetectionCandidates* data = nullptr;
        cudaEvent_t ready = nullptr, consumed = nullptr;
        bool recorded = false;
    };
    struct HostSlot {
        PacketBytes* data = nullptr;
        cudaEvent_t done = nullptr;
    };
    TrackingSessionConfig config;
    TrackingSessionState lifecycle = TrackingSessionState::Running;
    std::thread::id owner = std::this_thread::get_id();
    FrameId submitted = 0, received = 0;
    bool host_failed = false;
    cudaStream_t detect = nullptr, consumer = nullptr;
    cudaEvent_t terminal_detect = nullptr, terminal_consumer = nullptr;
    cudaEvent_t exp_rgb = nullptr; bool exp_pending = false;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::unique_ptr<NativePreprocessor> preprocess;
    std::unique_ptr<MotionCuda> motion;
    std::vector<CandidateSlot> candidates;
    std::vector<HostSlot> results;
    void *raw = nullptr, *statuses = nullptr, *producer_scratch = nullptr, *postprocess_scratch = nullptr;
    float* warmup_images = nullptr;
    PostprocessProducerStatus* producer = nullptr;
    detail::TrackingSessionControl* control = nullptr;
    std::uint8_t* device_result = nullptr;
    DeviceTrackState* tracks = nullptr;
    DevicePredicted* predicted = nullptr;
    DeviceBatch* batch = nullptr;
    DeviceTrackingRows* rows = nullptr;
    TrackingCudaScratch* tracking_scratch = nullptr;
    TrackerConfig* tracker_config = nullptr;
    MotionContext* motion_context = nullptr;
    DeviceMotionContext* tracker_motion = nullptr;
    ModelProducerPlugins plugins{};
    PostprocessRawView raw_view;
    std::string engine_hash, configuration, input_name;

    Impl(const std::string& engine_path, const TrackingSessionConfig& cfg) : config(cfg) {
        require(cfg.device >= 0, "device must be nonnegative");
        require(cfg.candidate_slots >= 1 && cfg.motion_slots >= 1, "candidate and motion slots must be positive");
        require(cfg.result_slots >= 1, "result_slots must be positive");
        validate_tracker_config(cfg.tracker);
        const auto key = motion_image_key(cfg.original_width, cfg.original_height, cfg.motion);
        const auto& profile = model_profile(cfg.profile);
        require((cfg.preprocess.output_width == kT01ModelProfile.width && cfg.preprocess.output_height == kT01ModelProfile.height) ||
                (cfg.preprocess.output_width == profile.width && cfg.preprocess.output_height == profile.height),
                "preprocessing dimensions differ from the selected model profile");
        config.preprocess.output_width = static_cast<int>(profile.width);
        config.preprocess.output_height = static_cast<int>(profile.height);
        try {
            cu(cudaSetDevice(cfg.device), "session cudaSetDevice");
            cu(cudaStreamCreateWithFlags(&detect, cudaStreamNonBlocking), "create detect stream");
            cu(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), "create consumer stream");
            cu(cudaEventCreateWithFlags(&terminal_detect, cudaEventDisableTiming), "create D terminal event");
            cu(cudaEventCreateWithFlags(&terminal_consumer, cudaEventDisableTiming), "create M terminal event");
            (void)model_producer_cuda_load_kernels();
            (void)postprocess_cuda_load_kernels();
            detail::tracking_session_load_kernels();
            register_scan_trt_plugins(*getPluginRegistry());
            const auto plan = read_text_file(engine_path);
            engine_hash = sha256_hex(plan);
            runtime.reset(nvinfer1::createInferRuntime(logger));
            require(runtime != nullptr, "create session TensorRT runtime");
            engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
            require(engine != nullptr && engine->getNbAuxStreams() == 0, "session requires model engine with zero auxiliary streams");
            const auto spec = model_io_spec(cfg.profile);
            input_name = spec.inputs.at(0).name;
            require(spec.inputs.size() == 1 && spec.outputs.size() == 6 && spec.statuses.size() == 9 &&
                    spec.check_inputs && spec.check_outputs, "unexpected model_io_spec");
            std::set<std::string> names;
            for (const auto& t : spec.inputs) validate_tensor(*engine,t.name,nvinfer1::DataType::kFLOAT,nvinfer1::TensorIOMode::kINPUT,t.shape,names);
            for (const auto& t : spec.outputs) validate_tensor(*engine,t.name,nvinfer1::DataType::kFLOAT,nvinfer1::TensorIOMode::kOUTPUT,t.shape,names);
            for (const auto& t : spec.statuses) validate_tensor(*engine,t.name,nvinfer1::DataType::kINT32,nvinfer1::TensorIOMode::kOUTPUT,{2},names);
            require(engine->getNbIOTensors() == static_cast<int>(names.size()), "session model tensor count mismatch");
            for (int i = 0; i < engine->getNbIOTensors(); ++i) require(names.contains(engine->getIOTensorName(i)), "unexpected session model tensor");
            context.reset(engine->createExecutionContext());
            require(context != nullptr, "create session execution context");
            preprocess = std::make_unique<NativePreprocessor>(detect, config.preprocess);
            motion = std::make_unique<MotionCuda>(key, cfg.motion, cfg.epoch, cfg.motion_slots);
            cu(cudaMalloc(&raw, profile.raw_elements() * sizeof(float)), "allocate raw tensors");
            cu(cudaMalloc(&statuses, 9 * 256), "allocate plugin statuses");
            cu(cudaMalloc(&producer, sizeof(*producer)), "allocate aggregate");
            cu(cudaMalloc(&producer_scratch, model_producer_workspace_bytes()), "allocate aggregate scratch");
            cu(cudaMalloc(&postprocess_scratch, postprocess_workspace_bytes()), "allocate F04 scratch");
            cu(cudaMemsetAsync(raw, 0, profile.raw_elements() * sizeof(float), detect), "initialize raw");
            cu(cudaMemsetAsync(producer, 0, sizeof(*producer), detect), "initialize producer");
            cu(cudaMemsetAsync(producer_scratch, 0, model_producer_workspace_bytes(), detect), "initialize producer scratch");
            cu(cudaMemsetAsync(postprocess_scratch, 0, postprocess_workspace_bytes(), detect), "initialize F04 scratch");
            cu(cudaMalloc(&control, sizeof(*control)), "allocate session control");
            cu(cudaMalloc(&device_result, packet_bytes), "allocate device packet");
            if (cfg.perturb_seed) detail::tracking_session_perturb_initialization(consumer);
            allocate(tracks, 2); allocate(predicted); allocate(batch); allocate(rows);
            allocate(tracking_scratch); allocate(tracker_config); allocate(motion_context); allocate(tracker_motion);
            // cfg.tracker remains alive through the initialization drain below.
            cu(cudaMemcpyAsync(tracker_config, &cfg.tracker, sizeof(cfg.tracker), cudaMemcpyHostToDevice, consumer), "upload tracker config");
            cu(cudaMemsetAsync(device_result, 0, packet_bytes, consumer), "initialize device packet");
            std::size_t offset = 0;
            for (unsigned i = 0; i < 6; ++i) {
                auto* p = static_cast<float*>(raw) + offset;
                const auto n = elements(spec.outputs[i].shape);
                require(context->setTensorAddress(spec.outputs[i].name.c_str(), p), "bind session raw output");
                if (i < 3) { raw_view.cls[i] = p; raw_view.cls_elements[i] = n; }
                else { raw_view.reg[i-3] = p; raw_view.reg_elements[i-3] = n; }
                offset += n;
            }
            require(offset == profile.raw_elements(), "session raw tensor extent mismatch");
            for (unsigned i = 0; i < 9; ++i) {
                plugins[i] = reinterpret_cast<ScanStatus*>(static_cast<unsigned char*>(statuses) + i*256);
                require(context->setTensorAddress(spec.statuses[i].name.c_str(), plugins[i]), "bind session plugin output");
            }
            candidates.resize(cfg.candidate_slots);
            for (auto& slot : candidates) {
                cu(cudaMalloc(&slot.data, sizeof(*slot.data)), "allocate candidate slot");
                cu(cudaEventCreateWithFlags(&slot.ready,cudaEventDisableTiming), "create candidate ready event");
                cu(cudaEventCreateWithFlags(&slot.consumed,cudaEventDisableTiming), "create candidate consumed event");
                DetectionCandidates initial{};
                initial.header.frame = static_cast<std::uint64_t>(no_frame);
                cu(cudaMemcpy(slot.data, &initial, sizeof(initial), cudaMemcpyHostToDevice), "initialize candidate slot");
            }
            results.resize(cfg.result_slots);
            for (auto& slot : results) {
                cu(cudaMallocHost(&slot.data,sizeof(*slot.data)), "allocate pinned result slot");
                *slot.data = {};
                cu(cudaEventCreateWithFlags(&slot.done,cudaEventDisableTiming), "create result event");
            }
            cu(cudaMemsetAsync(statuses,0,9*256,detect), "initialize plugin statuses");
            // Load the exact context's kernels before accepting frame 0.
            cu(cudaMalloc(&warmup_images,profile.input_elements()*sizeof(float)), "allocate warmup input");
            cu(cudaMemsetAsync(warmup_images,0,profile.input_elements()*sizeof(float),detect), "initialize warmup input");
            require(context->setTensorAddress(input_name.c_str(),warmup_images), "bind warmup input");
            if (!context->enqueueV3(detect)) throw std::runtime_error("session engine warmup failed");
            cu(cudaStreamSynchronize(detect), "session engine warmup drain");
            auto* retired_warmup = warmup_images;
            warmup_images = nullptr;
            cu(cudaFree(retired_warmup), "free warmup input");
            cu(cudaMemsetAsync(statuses,0,9*256,detect), "invalidate warmup plugin statuses");
            detail::tracking_session_initialize(control, tracks, cfg.epoch, consumer);
            enqueue_predict_tracks(tracks, tracker_motion, &control->value.current_frame, tracker_config, predicted,
                                   tracking_scratch, consumer, &control->value.run_stage);
            enqueue_assemble_detection_batch(candidates[0].data, predicted, tracker_config, batch,
                                            tracking_scratch, consumer, &control->value.run_stage);
            enqueue_associate_and_update(predicted, batch, tracker_motion, tracker_config, tracks, rows,
                                         tracking_scratch, consumer, &control->value.run_stage);
            cu(cudaDeviceSynchronize(), "tracking session initialization drain");
            std::ostringstream cfg_text;
            cfg_text << "profile=" << profile.name << "\nepoch=" << cfg.epoch << "\ndevice=" << cfg.device << "\ncandidate_slots=" << cfg.candidate_slots
                     << "\nmotion_slots=" << cfg.motion_slots << "\nresult_slots=" << cfg.result_slots
                     << "\noriginal_width=" << cfg.original_width << "\noriginal_height=" << cfg.original_height
                     << "\nmode=" << (cfg.mode == TrackingSessionMode::Serial ? "serial" : "pipelined")
                     << "\nperturb_seed=" << cfg.perturb_seed << "\nmotion.downscale=" << cfg.motion.downscale
                     << "\nmotion.pcr_threshold=" << std::hexfloat << cfg.motion.pcr_threshold
                     << "\nmotion.max_shift_fraction=" << cfg.motion.max_shift_fraction << std::defaultfloat << '\n'
                     << canonical_tracker_config(cfg.tracker) << preprocess->identity();
            configuration = cfg_text.str();
        } catch (...) { release(); throw; }
    }
    template<class T> void allocate(T*& p, std::size_t count = 1) {
        cu(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)), "allocate tracking storage");
        cu(cudaMemsetAsync(p, 0, count * sizeof(T), consumer), "initialize tracking storage");
    }
    void check_access() const {
        require(lifecycle != TrackingSessionState::Closed, "tracking session is closed");
        require(owner == std::this_thread::get_id(), "session requires its single host submitter");
        int current = -1;
        cu(cudaGetDevice(&current), "session current device");
        require(current == config.device, "session requires its original CUDA device");
    }
    void drain() {
        CleanupReport report;
        int token = 1;
        report.release(token,[&](int) {
            cu(cudaEventRecord(terminal_detect,detect), "record D terminal");
            cu(cudaEventSynchronize(terminal_detect), "wait D terminal");
        });
        token = 1;
        report.release(token,[&](int) {
            cu(cudaEventRecord(terminal_consumer,consumer), "record M terminal");
            cu(cudaEventSynchronize(terminal_consumer), "wait M terminal");
        });
        report.throw_if_failed();
    }
    void observe_pending() {
        for (FrameId i = received; i < submitted; ++i)
            if (decode_packet(*results[i % config.result_slots].data).status != ResultStatus::ok)
                lifecycle = TrackingSessionState::Failed;
    }
    CleanupReport release() noexcept {
        CleanupReport report;
        if (lifecycle == TrackingSessionState::Closed) return report;
        lifecycle = TrackingSessionState::Closed;
        int previous = -1, token = 1;
        report.release(token,[&](int) { cu(cudaGetDevice(&previous),"cleanup get device"); cu(cudaSetDevice(config.device),"cleanup set device"); });
        // Drain both streams independently, including a D-only tail.
        if (detect) { token=1; report.release(token,[&](int) { cu(cudaStreamSynchronize(detect),"cleanup D drain"); }); }
        if (consumer) { token=1; report.release(token,[&](int) { cu(cudaStreamSynchronize(consumer),"cleanup M drain"); }); }
        if (auto* p = preprocess.release()) {
            report.release(p,[](auto* h) { std::unique_ptr<NativePreprocessor> hold(h); h->close(); });
        }
        if (auto* p = motion.release()) {
            report.release(p, [](auto* h) { std::unique_ptr<MotionCuda> hold(h); h->close(); });
        }
        context.reset(); engine.reset(); runtime.reset();
        for (auto& slot : candidates) {
            report.release(slot.ready,[](auto p) { cu(cudaEventDestroy(p),"destroy ready"); });
            report.release(slot.consumed,[](auto p) { cu(cudaEventDestroy(p),"destroy consumed"); });
            report.release(slot.data,[](auto p) { cu(cudaFree(p),"free candidates"); });
        }
        for (auto& slot : results) {
            report.release(slot.done,[](auto p) { cu(cudaEventDestroy(p),"destroy result event"); });
            report.release(slot.data,[](auto p) { cu(cudaFreeHost(p),"free pinned result"); });
        }
        report.release(warmup_images,[](auto p) { cu(cudaFree(p),"free warmup input"); });
        report.release(raw,[](auto p) { cu(cudaFree(p),"free raw"); });
        report.release(statuses,[](auto p) { cu(cudaFree(p),"free statuses"); });
        report.release(producer,[](auto p) { cu(cudaFree(p),"free producer"); });
        report.release(producer_scratch,[](auto p) { cu(cudaFree(p),"free producer scratch"); });
        report.release(postprocess_scratch,[](auto p) { cu(cudaFree(p),"free postprocess scratch"); });
        report.release(control,[](auto p) { cu(cudaFree(p),"free control"); });
        report.release(device_result,[](auto p) { cu(cudaFree(p),"free device packet"); });
        report.release(tracks,[](auto p) { cu(cudaFree(p),"free TS"); });
        report.release(predicted,[](auto p) { cu(cudaFree(p),"free PREDICTED"); });
        report.release(batch,[](auto p) { cu(cudaFree(p),"free BATCH"); });
        report.release(rows,[](auto p) { cu(cudaFree(p),"free ROWS"); });
        report.release(tracking_scratch,[](auto p) { cu(cudaFree(p),"free tracker scratch"); });
        report.release(tracker_config,[](auto p) { cu(cudaFree(p),"free tracker config"); });
        report.release(motion_context,[](auto p) { cu(cudaFree(p),"free motion context"); });
        report.release(tracker_motion,[](auto p) { cu(cudaFree(p),"free tracker motion"); });
        report.release(terminal_detect,[](auto p) { cu(cudaEventDestroy(p),"destroy D terminal"); });
        report.release(terminal_consumer,[](auto p) { cu(cudaEventDestroy(p),"destroy M terminal"); });
        report.release(detect,[](auto p) { cu(cudaStreamDestroy(p),"destroy D"); });
        report.release(consumer,[](auto p) { cu(cudaStreamDestroy(p),"destroy M"); });
        if (previous >= 0 && previous != config.device) {
            token=1; report.release(token,[&](int) { cu(cudaSetDevice(previous),"cleanup restore device"); });
        }
        return report;
    }
    ~Impl() { release(); }
};

TrackingSession::TrackingSession(const std::string& path,const TrackingSessionConfig& config) : impl_(std::make_unique<Impl>(path,config)) {}
TrackingSession::~TrackingSession() = default;
void TrackingSession::enqueue(std::span<const std::uint8_t> jpeg, FrameId frame) {
    enqueue_for_test(jpeg, frame, 0);
}
void TrackingSession::enqueue_for_test(std::span<const std::uint8_t> jpeg, FrameId frame,
                                        std::uint32_t override_flags, bool d_only) {
    auto& s = *impl_;
    s.check_access();
    require(s.lifecycle == TrackingSessionState::Running, "session is not accepting frames");
    if (s.submitted - s.received == s.config.result_slots) throw std::length_error("session result ring full; receive first");
    try {
        require(frame == s.submitted, "session frame must be contiguous and zero-based");
        unsigned boundary = 0;
        const auto d_delay = [&] { detail::tracking_session_perturb(s.config.perturb_seed, frame, boundary++, s.detect); };
        const auto m_delay = [&] { detail::tracking_session_perturb(s.config.perturb_seed, frame, boundary++, s.consumer); };
        d_delay();
        if (!s.exp_rgb) cu(cudaEventCreateWithFlags(&s.exp_rgb, cudaEventDisableTiming), "exp event");
        if (s.exp_pending) cu(cudaEventSynchronize(s.exp_rgb), "exp wait F05");
        const auto input = s.preprocess->enqueue(jpeg, frame, s.config.original_width, s.config.original_height);
        d_delay();
        s.motion->enqueue_image(s.detect, input.rgb, frame);
        cu(cudaEventRecord(s.exp_rgb, s.detect), "exp record F05"); s.exp_pending = true;
        d_delay();
        if (d_only) throw std::runtime_error("directed D-only tail after F05");
        require(s.context->setTensorAddress(s.input_name.c_str(), input.images), "bind session model input");
        if (!s.context->enqueueV3(s.detect)) throw std::runtime_error("session F03 enqueueV3 failed");
        d_delay();
        model_producer_cuda(input.images, model_profile(s.config.profile).input_elements(), s.raw_view, s.plugins, s.producer,
                            s.producer_scratch, model_producer_workspace_bytes(), s.detect, s.config.profile);
        if (override_flags) detail::tracking_session_producer_flag(s.producer, override_flags, s.detect);
        d_delay();
        auto& candidate = s.candidates[frame % s.config.candidate_slots];
        if (candidate.recorded) cu(cudaStreamWaitEvent(s.detect, candidate.consumed, 0), "D wait CS consumed");
        PostprocessRequest request;
        request.epoch = s.config.epoch; request.frame = frame; request.mapping = input.mapping; request.profile = s.config.profile;
        postprocess_cuda(s.raw_view, request, s.producer, candidate.data, sizeof(*candidate.data), s.postprocess_scratch,
                         postprocess_workspace_bytes(), s.detect);
        d_delay();
        cu(cudaEventRecord(candidate.ready, s.detect), "record CS ready");
        detail::tracking_session_begin(s.control, s.config.epoch, frame, s.consumer);
        m_delay();
        s.motion->enqueue_estimate(s.consumer, frame, s.motion_context, &s.control->value.run_stage, &s.control->value);
        m_delay();
        detail::tracking_session_stage(s.control, nullptr, s.motion_context, s.tracker_motion, s.consumer);
        for (unsigned i = 0; i < 2; ++i)
            enqueue_predict_tracks(s.tracks + 1 - i, s.tracker_motion, &s.control->value.current_frame,
                                   s.tracker_config, s.predicted, s.tracking_scratch, s.consumer, &s.control->slot_run[i]);
        m_delay();
        detail::tracking_session_stage(s.control, &s.predicted->error, nullptr, nullptr, s.consumer);
        cu(cudaStreamWaitEvent(s.consumer, candidate.ready, 0), "M wait CS ready");
        detail::tracking_session_candidates(s.control, candidate.data, s.consumer);
        enqueue_assemble_detection_batch(candidate.data, s.predicted, s.tracker_config, s.batch,
                                         s.tracking_scratch, s.consumer, &s.control->value.run_stage);
        detail::tracking_session_candidates(s.control, candidate.data, s.consumer);
        cu(cudaEventRecord(candidate.consumed, s.consumer), "record CS consumed");
        m_delay();
        detail::tracking_session_stage(s.control, &s.batch->error, nullptr, nullptr, s.consumer);
        for (unsigned i = 0; i < 2; ++i)
            enqueue_associate_and_update(s.predicted, s.batch, s.tracker_motion, s.tracker_config, s.tracks + i,
                                         s.rows, s.tracking_scratch, s.consumer, &s.control->slot_run[i]);
        m_delay();
        detail::tracking_session_stage(s.control, &s.rows->error, nullptr, nullptr, s.consumer);
        detail::tracking_session_commit(s.control, s.consumer);
        m_delay();
        detail::tracking_session_packet(&s.control->value, s.rows, s.device_result, s.consumer);
        auto& host = s.results[frame % s.config.result_slots];
        cu(cudaMemcpyAsync(host.data, s.device_result, packet_bytes, cudaMemcpyDeviceToHost, s.consumer), "copy packet");
        cu(cudaEventRecord(host.done, s.consumer), "record packet ready");
        candidate.recorded = true;
        ++s.submitted;
        if (s.config.mode == TrackingSessionMode::Serial) s.drain();
    } catch (...) { s.host_failed = true; s.lifecycle = TrackingSessionState::Failed; throw; }
}
TrackingSessionResult TrackingSession::receive() {
    auto& s = *impl_;
    s.check_access();
    if (s.received == s.submitted) throw std::out_of_range("no pending tracking result");
    auto& slot = s.results[s.received % s.config.result_slots];
    try {
        cu(cudaEventSynchronize(slot.done), "wait tracking packet");
        TrackingSessionResult result;
        result.bytes = *slot.data;
        result.packet = decode_packet(result.bytes);
        require(result.packet.epoch == s.config.epoch && result.packet.frame == s.received, "tracking packet identity mismatch");
        ++s.received;
        if (result.packet.status != ResultStatus::ok) s.lifecycle = TrackingSessionState::Failed;
        return result;
    } catch (...) { s.host_failed = true; s.lifecycle = TrackingSessionState::Failed; throw; }
}
std::size_t TrackingSession::pending() const { return static_cast<std::size_t>(impl_->submitted-impl_->received); }
void TrackingSession::finish() {
    auto& s=*impl_;
    s.check_access();
    if (s.lifecycle==TrackingSessionState::Finished || s.lifecycle==TrackingSessionState::Cancelled) return;
    try {
        s.drain();
        s.observe_pending();
        if (!s.host_failed && s.lifecycle!=TrackingSessionState::Failed) s.lifecycle=TrackingSessionState::Finished;
    } catch (...) { s.host_failed=true; s.lifecycle=TrackingSessionState::Failed; throw; }
}
void TrackingSession::cancel() {
    auto& s=*impl_;
    s.check_access();
    if (s.lifecycle==TrackingSessionState::Finished || s.lifecycle==TrackingSessionState::Cancelled) return;
    try {
        s.drain(); s.observe_pending();
        if (s.lifecycle!=TrackingSessionState::Failed) s.lifecycle=TrackingSessionState::Cancelled;
    } catch (...) { s.host_failed=true; s.lifecycle=TrackingSessionState::Failed; throw; }
}
void TrackingSession::close() { impl_->release().throw_if_failed(); }
TrackingSessionState TrackingSession::state() const { return impl_->lifecycle; }
std::string TrackingSession::identity() const {
    const auto& s=*impl_;
    std::ostringstream out;
    out << "contract=f01-f09-session/1\nmotion_profile=" << kMotionProfile << "\ntracker_profile=" << kTrackerProfile
        << "\npacket_profile=v1\nengine_sha256=" << s.engine_hash
        << "\ntracker_config_sha256=" << sha256_hex(canonical_tracker_config(s.config.tracker))
        << "\nsource_commit=" << SACCADE_BUILD_COMMIT << "\nsource_dirty=" << SACCADE_BUILD_DIRTY << '\n'
        << s.configuration << "session_config_sha256=" << sha256_hex(s.configuration) << '\n';
    return out.str();
}
detail::TrackingSessionDiagnostic TrackingSession::diagnostic_for_test() const {
    auto& s = *impl_;
    s.check_access();
    s.drain();
    detail::TrackingSessionDiagnostic out{};
    cu(cudaMemcpy(&out.control, &s.control->value, sizeof(out.control), cudaMemcpyDeviceToHost), "terminal CTL");
    cu(cudaMemcpy(out.tracks, s.tracks, sizeof(out.tracks), cudaMemcpyDeviceToHost), "terminal TS");
    cu(cudaMemcpy(&out.motion, s.motion_context, sizeof(out.motion), cudaMemcpyDeviceToHost), "terminal motion");
    if (s.submitted)
        cu(cudaMemcpy(&out.candidates, s.candidates[(s.submitted - 1) % s.config.candidate_slots].data,
                      sizeof(out.candidates), cudaMemcpyDeviceToHost), "terminal candidates");
    return out;
}
} // namespace saccade

#include "saccade/detection_session.hpp"
#include "saccade/detail/detection_session_kernels.hpp"
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
        if (severity <= Severity::kERROR) std::cerr << "DetectionSession TensorRT: " << message << '\n';
    }
};
}

struct DetectionSession::Impl {
    struct CandidateSlot {
        DetectionCandidates* data = nullptr;
        cudaEvent_t ready = nullptr, consumed = nullptr;
        bool recorded = false;
    };
    struct HostSlot {
        DetectionSessionResult* data = nullptr;
        cudaEvent_t done = nullptr;
    };
    DetectionSessionConfig config;
    DetectionSessionState lifecycle = DetectionSessionState::Running;
    std::thread::id owner = std::this_thread::get_id();
    std::uint64_t submitted = 0, received = 0;
    bool host_failed = false;
    cudaStream_t detect = nullptr, consumer = nullptr;
    cudaEvent_t terminal_detect = nullptr, terminal_consumer = nullptr;
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::unique_ptr<NativePreprocessor> preprocess;
    std::vector<CandidateSlot> candidates;
    std::vector<HostSlot> results;
    void *raw = nullptr, *statuses = nullptr, *producer_scratch = nullptr, *postprocess_scratch = nullptr;
    float* warmup_images = nullptr;
    PostprocessProducerStatus* producer = nullptr;
    detail::DetectionSessionControl* control = nullptr;
    DetectionSessionResult* device_result = nullptr;
    ModelProducerPlugins plugins{};
    PostprocessRawView raw_view;
    std::string engine_hash, configuration, input_name;

    Impl(const std::string& engine_path, const DetectionSessionConfig& cfg) : config(cfg) {
        require(cfg.device >= 0, "device must be nonnegative");
        require(cfg.candidate_slots >= 1 && cfg.candidate_slots <= 16, "candidate_slots must be in [1,16]");
        require(cfg.result_slots >= 1 && cfg.result_slots <= 64, "result_slots must be in [1,64]");
        require(cfg.output_capacity >= 1 && cfg.output_capacity <= kPostprocessCapacity, "output_capacity must be in [1,300]");
        require(cfg.preprocess.output_width == 640 && cfg.preprocess.output_height == 640, "F03 requires 640x640 preprocessing");
        try {
            cu(cudaSetDevice(cfg.device), "session cudaSetDevice");
            cu(cudaStreamCreateWithFlags(&detect, cudaStreamNonBlocking), "create detect stream");
            cu(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking), "create consumer stream");
            cu(cudaEventCreateWithFlags(&terminal_detect, cudaEventDisableTiming), "create D terminal event");
            cu(cudaEventCreateWithFlags(&terminal_consumer, cudaEventDisableTiming), "create M terminal event");
            (void)model_producer_cuda_load_kernels();
            (void)postprocess_cuda_load_kernels();
            detail::detection_session_load_kernels();
            register_scan_trt_plugins(*getPluginRegistry());
            const auto plan = read_text_file(engine_path);
            engine_hash = sha256_hex(plan);
            runtime.reset(nvinfer1::createInferRuntime(logger));
            require(runtime != nullptr, "create session TensorRT runtime");
            engine.reset(runtime->deserializeCudaEngine(plan.data(), plan.size()));
            require(engine != nullptr && engine->getNbAuxStreams() == 0, "session requires model engine with zero auxiliary streams");
            const auto spec = model_io_spec();
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
            preprocess = std::make_unique<NativePreprocessor>(detect, cfg.preprocess);
            cu(cudaMalloc(&raw, 705600 * sizeof(float)), "allocate raw tensors");
            cu(cudaMalloc(&statuses, 9 * 256), "allocate plugin statuses");
            cu(cudaMalloc(&producer, sizeof(*producer)), "allocate aggregate");
            cu(cudaMalloc(&producer_scratch, model_producer_workspace_bytes()), "allocate aggregate scratch");
            cu(cudaMalloc(&postprocess_scratch, postprocess_workspace_bytes()), "allocate F04 scratch");
            cu(cudaMalloc(&control, sizeof(*control)), "allocate session control");
            cu(cudaMalloc(&device_result, sizeof(*device_result)), "allocate device result");
            std::size_t offset = 0;
            for (unsigned i = 0; i < 6; ++i) {
                auto* p = static_cast<float*>(raw) + offset;
                const auto n = elements(spec.outputs[i].shape);
                require(context->setTensorAddress(spec.outputs[i].name.c_str(), p), "bind session raw output");
                if (i < 3) { raw_view.cls[i] = p; raw_view.cls_elements[i] = n; }
                else { raw_view.reg[i-3] = p; raw_view.reg_elements[i-3] = n; }
                offset += n;
            }
            require(offset == 705600, "session raw tensor extent mismatch");
            for (unsigned i = 0; i < 9; ++i) {
                plugins[i] = reinterpret_cast<ScanStatus*>(static_cast<unsigned char*>(statuses) + i*256);
                require(context->setTensorAddress(spec.statuses[i].name.c_str(), plugins[i]), "bind session plugin output");
            }
            candidates.resize(cfg.candidate_slots);
            for (auto& slot : candidates) {
                cu(cudaMalloc(&slot.data, sizeof(*slot.data)), "allocate candidate slot");
                cu(cudaEventCreateWithFlags(&slot.ready,cudaEventDisableTiming), "create candidate ready event");
                cu(cudaEventCreateWithFlags(&slot.consumed,cudaEventDisableTiming), "create candidate consumed event");
                cu(cudaMemsetAsync(slot.data, 0, sizeof(*slot.data), detect), "initialize candidate slot");
            }
            results.resize(cfg.result_slots);
            for (auto& slot : results) {
                cu(cudaMallocHost(&slot.data,sizeof(*slot.data)), "allocate pinned result slot");
                *slot.data = {};
                cu(cudaEventCreateWithFlags(&slot.done,cudaEventDisableTiming), "create result event");
            }
            cu(cudaMemsetAsync(statuses,0,9*256,detect), "initialize plugin statuses");
            // TensorRT may lazily load internal kernels on its first enqueue.
            // Exercise the exact context at initialization, before a caller can
            // hold M pending, so frame submissions do not rely on first-use
            // device synchronization. The zero input is warmup scratch only.
            cu(cudaMalloc(&warmup_images,kModelProducerInputElements*sizeof(float)), "allocate warmup input");
            cu(cudaMemsetAsync(warmup_images,0,kModelProducerInputElements*sizeof(float),detect), "initialize warmup input");
            require(context->setTensorAddress(input_name.c_str(),warmup_images), "bind warmup input");
            if (!context->enqueueV3(detect)) throw std::runtime_error("session engine warmup failed");
            cu(cudaStreamSynchronize(detect), "session engine warmup drain");
            auto* retired_warmup = warmup_images;
            warmup_images = nullptr;
            cu(cudaFree(retired_warmup), "free warmup input");
            cu(cudaMemsetAsync(statuses,0,9*256,detect), "invalidate warmup plugin statuses");
            detail::detection_session_initialize(control,consumer);
            cu(cudaStreamSynchronize(detect), "initialize detect stream");
            cu(cudaStreamSynchronize(consumer), "initialize consumer stream");
            std::ostringstream cfg_text;
            cfg_text << "epoch=" << cfg.epoch << "\ndevice=" << cfg.device << "\ncandidate_slots=" << cfg.candidate_slots
                     << "\nresult_slots=" << cfg.result_slots << "\noutput_capacity=" << cfg.output_capacity << '\n'
                     << preprocess->identity();
            configuration = cfg_text.str();
        } catch (...) { release(); throw; }
    }
    void check_access() const {
        require(lifecycle != DetectionSessionState::Closed, "detection session is closed");
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
        for (std::uint64_t i = received; i < submitted; ++i)
            if (results[i % config.result_slots].data->status != DetectionSessionStatus::Ok)
                lifecycle = DetectionSessionState::Failed;
    }
    CleanupReport release() noexcept {
        CleanupReport report;
        if (lifecycle == DetectionSessionState::Closed) return report;
        lifecycle = DetectionSessionState::Closed;
        int previous = -1, token = 1;
        report.release(token,[&](int) { cu(cudaGetDevice(&previous),"cleanup get device"); cu(cudaSetDevice(config.device),"cleanup set device"); });
        // Both streams must be drained even when D has work without an M
        // packet. Attempt M independently if D reports an asynchronous error.
        if (detect) { token=1; report.release(token,[&](int) { cu(cudaStreamSynchronize(detect),"cleanup D drain"); }); }
        if (consumer) { token=1; report.release(token,[&](int) { cu(cudaStreamSynchronize(consumer),"cleanup M drain"); }); }
        if (auto* p = preprocess.release()) {
            report.release(p,[](auto* h) { std::unique_ptr<NativePreprocessor> hold(h); h->close(); });
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
        report.release(device_result,[](auto p) { cu(cudaFree(p),"free device result"); });
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

DetectionSession::DetectionSession(const std::string& path,const DetectionSessionConfig& config) : impl_(std::make_unique<Impl>(path,config)) {}
DetectionSession::~DetectionSession() = default;
void DetectionSession::enqueue(std::span<const std::uint8_t> jpeg,std::uint64_t frame,int ew,int eh) {
    enqueue_for_test(jpeg,frame,ew,eh,0,false);
}
void DetectionSession::enqueue_for_test(std::span<const std::uint8_t> jpeg,std::uint64_t frame,int ew,int eh,
                                        std::uint32_t override_flags,bool corrupt,bool d_only) {
    auto& s = *impl_;
    s.check_access();
    require(s.lifecycle == DetectionSessionState::Running, "session is not accepting frames");
    // Bounded backpressure is recoverable and cannot discard an unread result.
    if (s.submitted - s.received == s.config.result_slots) throw std::length_error("session result ring full; receive first");
    try {
        require(frame == s.submitted && frame != kDetectionSessionNoFrame, "session frame must be contiguous and zero-based");
        const auto input = s.preprocess->enqueue(jpeg,frame,ew,eh);
        if (d_only) throw std::runtime_error("directed D-only tail after preprocessing");
        require(s.context->setTensorAddress(s.input_name.c_str(),input.images), "bind session model input");
        if (!s.context->enqueueV3(s.detect)) throw std::runtime_error("session F03 enqueueV3 failed");
        model_producer_cuda(input.images,kModelProducerInputElements,s.raw_view,s.plugins,s.producer,s.producer_scratch,
                            model_producer_workspace_bytes(),s.detect);
        if (override_flags) detail::detection_session_test_override(s.producer,override_flags,s.detect);
        auto& candidate = s.candidates[frame % s.config.candidate_slots];
        if (candidate.recorded) cu(cudaStreamWaitEvent(s.detect,candidate.consumed,0),"D wait candidate consumed");
        PostprocessRequest request;
        request.epoch=s.config.epoch; request.frame=frame; request.mapping=input.mapping; request.capacity=s.config.output_capacity;
        postprocess_cuda(s.raw_view,request,s.producer,candidate.data,sizeof(*candidate.data),s.postprocess_scratch,
                         postprocess_workspace_bytes(),s.detect);
        if (corrupt) detail::detection_session_test_corrupt(candidate.data,s.detect);
        cu(cudaEventRecord(candidate.ready,s.detect),"record candidates ready");
        cu(cudaStreamWaitEvent(s.consumer,candidate.ready,0),"M wait candidates ready");
        detail::detection_session_finalize(candidate.data,s.device_result,s.control,s.config.epoch,frame,s.config.output_capacity,s.consumer);
        cu(cudaEventRecord(candidate.consumed,s.consumer),"record candidates consumed");
        auto& host=s.results[frame % s.config.result_slots];
        cu(cudaMemcpyAsync(host.data,s.device_result,sizeof(*host.data),cudaMemcpyDeviceToHost,s.consumer),"copy session result");
        cu(cudaEventRecord(host.done,s.consumer),"record session result ready");
        candidate.recorded=true;
        ++s.submitted;
    } catch (...) { s.host_failed=true; s.lifecycle=DetectionSessionState::Failed; throw; }
}
DetectionSessionResult DetectionSession::receive() {
    auto& s=*impl_;
    s.check_access();
    if (s.received == s.submitted) throw std::out_of_range("no pending detection result");
    auto& slot=s.results[s.received % s.config.result_slots];
    try {
        cu(cudaEventSynchronize(slot.done),"wait session result");
        const auto result=*slot.data;
        require(result.epoch==s.config.epoch && result.frame==s.received && result.reserved==0, "session result identity mismatch");
        ++s.received;
        if (result.status!=DetectionSessionStatus::Ok) s.lifecycle=DetectionSessionState::Failed;
        return result;
    } catch (...) { s.host_failed=true; s.lifecycle=DetectionSessionState::Failed; throw; }
}
std::size_t DetectionSession::pending() const { return static_cast<std::size_t>(impl_->submitted-impl_->received); }
void DetectionSession::finish() {
    auto& s=*impl_;
    s.check_access();
    if (s.lifecycle==DetectionSessionState::Finished || s.lifecycle==DetectionSessionState::Cancelled) return;
    try {
        s.drain();
        s.observe_pending();
        if (!s.host_failed && s.lifecycle!=DetectionSessionState::Failed) s.lifecycle=DetectionSessionState::Finished;
    } catch (...) { s.host_failed=true; s.lifecycle=DetectionSessionState::Failed; throw; }
}
void DetectionSession::cancel() {
    auto& s=*impl_;
    s.check_access();
    if (s.lifecycle==DetectionSessionState::Finished || s.lifecycle==DetectionSessionState::Cancelled) return;
    try {
        s.drain(); s.observe_pending();
        if (s.lifecycle!=DetectionSessionState::Failed) s.lifecycle=DetectionSessionState::Cancelled;
    } catch (...) { s.host_failed=true; s.lifecycle=DetectionSessionState::Failed; throw; }
}
void DetectionSession::close() { impl_->release().throw_if_failed(); }
DetectionSessionState DetectionSession::state() const { return impl_->lifecycle; }
std::string DetectionSession::identity() const {
    const auto& s=*impl_;
    std::ostringstream out;
    out << "contract=" << kDetectionSessionPolicy << "\nengine_sha256=" << s.engine_hash
        << "\nsource_commit=" << SACCADE_BUILD_COMMIT << "\nsource_dirty=" << SACCADE_BUILD_DIRTY << '\n'
        << s.configuration << "session_config_sha256=" << sha256_hex(s.configuration) << '\n';
    return out.str();
}
cudaStream_t DetectionSession::consumer_stream_for_test() const { impl_->check_access(); return impl_->consumer; }
cudaStream_t DetectionSession::detect_stream_for_test() const { impl_->check_access(); return impl_->detect; }
} // namespace saccade

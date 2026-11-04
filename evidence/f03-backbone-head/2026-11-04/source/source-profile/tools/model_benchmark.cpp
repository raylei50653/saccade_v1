// Development F03 benchmark. Uses an existing serialized engine and frozen
// FP32 inputs; model-only timings exclude H2D/D2H, validation and initialization.
#include "saccade/file_io.hpp"
#include "saccade/head_trt_network.hpp"
#include "saccade/model_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/selective_scan_cuda.hpp"
#include "saccade/sha256.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace saccade;
using namespace nvinfer1;
using Clock = std::chrono::steady_clock;
constexpr const char* usage = "usage: model_benchmark ENGINE IMAGES_FILE NEW_OUT_DIR [--profile hr-1280x768] [--iterations 80] [--warmup 20] [--rounds 2] [--persistent-cache-bytes 0] [--nvtx engine|none|detailed] [--layer-iterations 0]";
void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void cuda_check(cudaError_t error, const char* call) {
    if (error != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(error));
}
#define CUDA(call) cuda_check((call), #call)
struct Logger final : ILogger {
    unsigned errors = 0;
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) ++errors;
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};
struct LayerTotals { std::uint64_t calls = 0; double total_ms = 0; };
struct LayerProfiler final : IProfiler {
    std::map<std::string, LayerTotals> layers;
    bool failed = false;
    void reportLayerTime(const char* name, float ms) noexcept override {
        if (!name || !std::isfinite(ms) || ms < 0) { failed = true; return; }
        try {
            auto& totals = layers[name];
            ++totals.calls;
            totals.total_ms += ms;
        } catch (...) { failed = true; }
    }
};
const char* nvtx_name(ProfilingVerbosity verbosity) {
    switch (verbosity) {
    case ProfilingVerbosity::kNONE: return "none";
    case ProfilingVerbosity::kLAYER_NAMES_ONLY: return "layer-names-only";
    case ProfilingVerbosity::kDETAILED: return "detailed";
    }
    throw std::runtime_error("unknown NVTX verbosity");
}
std::string csv_text(std::string_view value) {
    std::string result = "\"";
    for (const char c : value) {
        if (c == '"') result += '"';
        result += c;
    }
    return result + '"';
}
unsigned number(std::string_view text) {
    unsigned value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    require(error == std::errc{} && end == text.data() + text.size(), "invalid positive integer");
    return value;
}
std::size_t elements(const std::vector<std::int64_t>& shape) {
    std::size_t value = 1;
    for (const auto d : shape) {
        require(d > 0 && static_cast<std::uint64_t>(d) <= std::numeric_limits<std::size_t>::max() / value, "invalid static shape");
        value *= static_cast<std::size_t>(d);
    }
    return value;
}
bool same_shape(Dims actual, const std::vector<std::int64_t>& expected) {
    if (actual.nbDims != static_cast<int>(expected.size())) return false;
    for (int i = 0; i < actual.nbDims; ++i) if (actual.d[i] != expected[static_cast<std::size_t>(i)]) return false;
    return true;
}
struct Resources {
    cudaStream_t stream = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    std::vector<void*> allocations;
    ~Resources() {
        if (stream) cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        for (auto* allocation : allocations) cudaFree(allocation);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
        if (stream) cudaStreamDestroy(stream);
    }
    void* allocate(std::size_t bytes) {
        void* result = nullptr;
        CUDA(cudaMalloc(&result, bytes));
        allocations.push_back(result);
        return result;
    }
    void close() {
        CUDA(cudaStreamSynchronize(stream));
        CUDA(cudaGraphExecDestroy(executable)); executable = nullptr;
        CUDA(cudaGraphDestroy(graph)); graph = nullptr;
        for (auto*& allocation : allocations) { CUDA(cudaFree(allocation)); allocation = nullptr; }
        CUDA(cudaEventDestroy(begin)); begin = nullptr;
        CUDA(cudaEventDestroy(end)); end = nullptr;
        CUDA(cudaStreamDestroy(stream)); stream = nullptr;
    }
};
struct ProfileGraph {
    cudaStream_t stream;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    bool capturing = false;
    explicit ProfileGraph(cudaStream_t value) : stream(value) {}
    ProfileGraph(const ProfileGraph&) = delete;
    ProfileGraph& operator=(const ProfileGraph&) = delete;
    ~ProfileGraph() {
        if (capturing) {
            cudaGraph_t discarded = nullptr;
            cudaStreamEndCapture(stream, &discarded);
            if (discarded) cudaGraphDestroy(discarded);
        }
        cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
    void capture_begin() {
        CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        capturing = true;
    }
    void capture_end() {
        const auto result = cudaStreamEndCapture(stream, &graph);
        capturing = false;
        cuda_check(result, "cudaStreamEndCapture(profile)");
        CUDA(cudaGraphInstantiate(&executable, graph, 0));
        CUDA(cudaGraphUpload(executable, stream));
        CUDA(cudaStreamSynchronize(stream));
    }
    void close() {
        CUDA(cudaGraphExecDestroy(executable)); executable = nullptr;
        CUDA(cudaGraphDestroy(graph)); graph = nullptr;
    }
};
struct Tensor { std::string name; std::size_t count; void* device; };
struct Snapshot { std::string payload, statuses; };
double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}
double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<std::size_t>(std::ceil(static_cast<double>(values.size()) * fraction));
    return values[std::max<std::size_t>(1, rank) - 1];
}
} // namespace

int main(int argc, char** argv) {
    std::filesystem::path output;
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") { std::cout << usage << '\n'; return 0; }
        require(argc >= 4, usage);
        ModelProfile profile = ModelProfile::Hr1280x768;
        unsigned iterations = 80, warmup = 20, rounds = 2;
        unsigned persistent_cache_bytes = 0, layer_iterations = 0;
        std::string nvtx = "engine";
        for (int i = 4; i < argc; ++i) {
            const std::string_view key = argv[i];
            require(i + 1 < argc, usage);
            const std::string_view value = argv[++i];
            if (key == "--profile") profile = parse_model_profile(value);
            else if (key == "--iterations") iterations = number(value);
            else if (key == "--warmup") warmup = number(value);
            else if (key == "--rounds") rounds = number(value);
            else if (key == "--persistent-cache-bytes") persistent_cache_bytes = number(value);
            else if (key == "--nvtx") nvtx = value;
            else if (key == "--layer-iterations") layer_iterations = number(value);
            else throw std::invalid_argument("unknown option: " + std::string(key));
        }
        require(iterations > 0 && iterations <= 100000 && warmup > 0 && warmup <= 10000 && rounds > 0 && rounds <= 100, "invalid replay bounds");
        require(layer_iterations <= 100000, "invalid layer profiling bound");
        require(nvtx == "engine" || nvtx == "none" || nvtx == "detailed", "invalid NVTX verbosity");
        require(!std::filesystem::exists(argv[3]), "output directory must be new");
        require(std::filesystem::create_directory(argv[3]), "cannot create output directory");
        output = argv[3];
        const auto plan = read_text_file(argv[1]), images = read_text_file(argv[2]);
        const auto frame_bytes = model_profile(profile).input_elements() * sizeof(float);
        require(!images.empty() && images.size() % frame_bytes == 0, "images must contain complete profile-sized FP32 frames");
        const std::size_t frames = images.size() / frame_bytes;
        const auto spec = model_io_spec(profile);
        require(spec.inputs.size() == 1 && spec.outputs.size() == 6 && spec.statuses.size() == 9, "unexpected F03 I/O specification");
        CUDA(cudaSetDevice(0));
        Logger logger;
        register_scan_trt_plugins(*getPluginRegistry());
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        require(runtime != nullptr, "createInferRuntime failed");
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
        require(engine != nullptr && engine->getNbAuxStreams() == 0, "engine missing or auxiliary streams enabled");
        require(engine->getNbIOTensors() == 16, "engine I/O count differs from F03 specification");
        LayerProfiler layer_profiler;
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
        require(context != nullptr, "createExecutionContext failed");
        context->setPersistentCacheLimit(persistent_cache_bytes);
        const auto actual_cache_bytes = context->getPersistentCacheLimit();
        if (nvtx != "engine")
            require(context->setNvtxVerbosity(nvtx == "none" ? ProfilingVerbosity::kNONE : ProfilingVerbosity::kDETAILED), "setNvtxVerbosity failed");
        const std::string actual_nvtx = nvtx_name(context->getNvtxVerbosity());
        Resources resources;
        CUDA(cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking));
        CUDA(cudaEventCreate(&resources.begin)); CUDA(cudaEventCreate(&resources.end));
        std::map<std::string, TensorIOMode> expected_names;
        std::vector<Tensor> outputs, statuses;
        const auto bind = [&](const std::string& name, const std::vector<std::int64_t>& shape, TensorIOMode mode, DataType type) {
            require(expected_names.emplace(name, mode).second, "duplicate expected tensor name");
            require(engine->getTensorIOMode(name.c_str()) == mode && engine->getTensorDataType(name.c_str()) == type &&
                    engine->getTensorLocation(name.c_str()) == TensorLocation::kDEVICE && engine->getTensorFormat(name.c_str()) == TensorFormat::kLINEAR &&
                    same_shape(engine->getTensorShape(name.c_str()), shape), "engine I/O mismatch: " + name);
            Tensor tensor{name, elements(shape), resources.allocate(elements(shape) * sizeof(float))};
            require(context->setTensorAddress(name.c_str(), tensor.device), "setTensorAddress failed: " + name);
            return tensor;
        };
        const auto input = bind(spec.inputs[0].name, spec.inputs[0].shape, TensorIOMode::kINPUT, DataType::kFLOAT);
        for (const auto& tensor : spec.outputs) outputs.push_back(bind(tensor.name, tensor.shape, TensorIOMode::kOUTPUT, DataType::kFLOAT));
        for (const auto& tensor : spec.statuses) statuses.push_back(bind(tensor.name, {2}, TensorIOMode::kOUTPUT, DataType::kINT32));
        for (int i = 0; i < engine->getNbIOTensors(); ++i)
            require(expected_names.contains(engine->getIOTensorName(i)), "unexpected engine tensor");
        const auto enqueue = [&](bool graph) {
            if (graph) CUDA(cudaGraphLaunch(resources.executable, resources.stream));
            else require(context->enqueueV3(resources.stream), "enqueueV3 failed");
        };
        const auto upload = [&](std::size_t frame) {
            CUDA(cudaMemcpyAsync(input.device, images.data() + frame * frame_bytes, frame_bytes, cudaMemcpyHostToDevice, resources.stream));
            CUDA(cudaStreamSynchronize(resources.stream));
        };
        const auto poison = [&] {
            for (const auto& tensor : outputs) CUDA(cudaMemsetAsync(tensor.device, 0xff, tensor.count * sizeof(float), resources.stream));
            for (const auto& tensor : statuses) CUDA(cudaMemsetAsync(tensor.device, 0xff, sizeof(ScanStatus), resources.stream));
        };
        const auto snapshot = [&] {
            Snapshot result;
            for (const auto& tensor : outputs) {
                std::vector<float> values(tensor.count);
                CUDA(cudaMemcpyAsync(values.data(), tensor.device, values.size() * sizeof(float), cudaMemcpyDeviceToHost, resources.stream));
                CUDA(cudaStreamSynchronize(resources.stream));
                for (const float value : values) require(std::isfinite(value), "nonfinite output: " + tensor.name);
                result.payload.append(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            }
            for (const auto& tensor : statuses) {
                ScanStatus value{};
                CUDA(cudaMemcpyAsync(&value, tensor.device, sizeof(value), cudaMemcpyDeviceToHost, resources.stream));
                CUDA(cudaStreamSynchronize(resources.stream));
                require(value.flags == 0 && value.done == kScanDone, "plugin status failed: " + tensor.name);
                result.statuses.append(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            return result;
        };
        const auto equal = [](const Snapshot& a, const Snapshot& b) {
            require(a.payload == b.payload && a.statuses == b.statuses, "ordinary/repeat/graph output or status bits differ");
        };
        upload(0);
        for (unsigned i = 0; i < warmup; ++i) enqueue(false);
        CUDA(cudaStreamSynchronize(resources.stream));
        CUDA(cudaStreamBeginCapture(resources.stream, cudaStreamCaptureModeGlobal));
        require(context->enqueueV3(resources.stream), "capture enqueueV3 failed");
        CUDA(cudaStreamEndCapture(resources.stream, &resources.graph));
        CUDA(cudaGraphInstantiate(&resources.executable, resources.graph, 0));
        CUDA(cudaGraphUpload(resources.executable, resources.stream));
        CUDA(cudaStreamSynchronize(resources.stream));
        std::ostringstream verification, layout;
        verification << "input,payload_sha256,status_sha256,ordinary_repeat_bits,graph_bits,status,finite\n";
        layout << "kind,name,offset_elements,elements\n";
        std::size_t offset = 0;
        for (const auto& tensor : outputs) { layout << "payload," << tensor.name << ',' << offset << ',' << tensor.count << '\n'; offset += tensor.count; }
        offset = 0;
        for (const auto& tensor : statuses) { layout << "status," << tensor.name << ',' << offset << ",2\n"; offset += 2; }
        Snapshot first;
        for (std::size_t frame = 0; frame < frames; ++frame) {
            upload(frame); poison(); enqueue(false); const auto expected = snapshot();
            poison(); enqueue(false); equal(expected, snapshot());
            poison(); enqueue(true); equal(expected, snapshot());
            write_text_file(output / ("input-" + std::to_string(frame) + ".f32"), expected.payload);
            write_text_file(output / ("input-" + std::to_string(frame) + ".status"), expected.statuses);
            verification << frame << ',' << sha256_hex(expected.payload) << ',' << sha256_hex(expected.statuses) << ",PASS,PASS,PASS,PASS\n";
            if (frame == 0) first = expected;
        }
        write_text_file(output / "verification.csv", verification.str());
        write_text_file(output / "tensor-layout.csv", layout.str());
        upload(0); // The benchmark holds this input fixed; transfers are outside timings.
        std::ostringstream samples, runs;
        samples << "round,ordinal,mode,iteration,gpu_ms,wall_ms\n" << std::fixed << std::setprecision(6);
        runs << "round,ordinal,mode,gpu_mean_ms,gpu_p50_ms,gpu_p95_ms,wall_mean_ms\n" << std::fixed << std::setprecision(6);
        std::vector<double> ordinary, graph;
        for (unsigned round = 0; round < rounds; ++round) {
            for (unsigned ordinal = 0; ordinal < 4; ++ordinal) {
                const bool use_graph = ordinal == 1 || ordinal == 2;
                const char* mode = use_graph ? "graph" : "ordinary";
                for (unsigned i = 0; i < warmup; ++i) enqueue(use_graph);
                CUDA(cudaStreamSynchronize(resources.stream));
                std::vector<double> gpu, wall;
                for (unsigned i = 0; i < iterations; ++i) {
                    const auto started = Clock::now();
                    CUDA(cudaEventRecord(resources.begin, resources.stream));
                    enqueue(use_graph);
                    CUDA(cudaEventRecord(resources.end, resources.stream));
                    CUDA(cudaEventSynchronize(resources.end));
                    float elapsed = 0;
                    CUDA(cudaEventElapsedTime(&elapsed, resources.begin, resources.end));
                    const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
                    gpu.push_back(elapsed); wall.push_back(wall_ms);
                    (use_graph ? graph : ordinary).push_back(elapsed);
                    samples << round << ',' << ordinal << ',' << mode << ',' << i << ',' << elapsed << ',' << wall_ms << '\n';
                }
                equal(first, snapshot());
                runs << round << ',' << ordinal << ',' << mode << ',' << mean(gpu) << ',' << percentile(gpu, .5) << ',' << percentile(gpu, .95) << ',' << mean(wall) << '\n';
                std::cout << "round=" << round << " ordinal=" << ordinal << " mode=" << mode << " gpu_mean_ms=" << mean(gpu) << " output_bits=PASS\n" << std::flush;
            }
        }
        CUDA(cudaStreamSynchronize(resources.stream));
        if (layer_iterations > 0) {
            // Capture profiling events separately after all benchmark samples.
            const bool emits_profile = context->getEnqueueEmitsProfile();
            context->setProfiler(&layer_profiler);
            context->setEnqueueEmitsProfile(false);
            const auto report_layers = [&] {
                require(context->reportToProfiler(), "reportToProfiler failed");
                require(!layer_profiler.failed, "layer profiler callback failed");
            };
            for (unsigned i = 0; i < warmup; ++i) {
                enqueue(false);
                CUDA(cudaStreamSynchronize(resources.stream));
                report_layers();
            }
            layer_profiler.layers.clear();
            ProfileGraph profile_graph(resources.stream);
            profile_graph.capture_begin();
            enqueue(false);
            profile_graph.capture_end();
            constexpr unsigned graph_warmup = 5;
            for (unsigned i = 0; i < graph_warmup; ++i) {
                CUDA(cudaGraphLaunch(profile_graph.executable, resources.stream));
                CUDA(cudaStreamSynchronize(resources.stream));
                report_layers();
            }
            layer_profiler.layers.clear();
            std::vector<double> profiled_graph;
            profiled_graph.reserve(layer_iterations);
            for (unsigned i = 0; i < layer_iterations; ++i) {
                CUDA(cudaEventRecord(resources.begin, resources.stream));
                CUDA(cudaGraphLaunch(profile_graph.executable, resources.stream));
                CUDA(cudaEventRecord(resources.end, resources.stream));
                CUDA(cudaEventSynchronize(resources.end));
                float elapsed = 0;
                CUDA(cudaEventElapsedTime(&elapsed, resources.begin, resources.end));
                profiled_graph.push_back(elapsed);
                report_layers();
            }
            // No further enqueues follow; the profiler outlives the context.
            context->setEnqueueEmitsProfile(emits_profile);
            require(!layer_profiler.layers.empty(), "layer profiler returned no layers");
            equal(first, snapshot());
            std::ostringstream layers;
            layers << "layer,calls,total_ms,mean_ms\n" << std::fixed << std::setprecision(9);
            double sum_layer_mean_ms = 0;
            for (const auto& [name, totals] : layer_profiler.layers) {
                const double layer_mean = totals.total_ms / static_cast<double>(totals.calls);
                layers << csv_text(name) << ',' << totals.calls << ',' << totals.total_ms << ','
                       << layer_mean << '\n';
                sum_layer_mean_ms += layer_mean;
            }
            write_text_file(output / "layers.csv", layers.str());
            write_text_file(output / "engine-layers.json", head_engine_layers(*engine));
            std::ostringstream layer_summary;
            layer_summary << std::fixed << std::setprecision(9)
                          << "{\n  \"layer_profile_mode\": \"graph\",\n  \"layer_iterations\": " << layer_iterations
                          << ",\n  \"ordinary_warmup_iterations\": " << warmup << ",\n  \"graph_warmup_iterations\": " << graph_warmup
                          << ",\n  \"layer_count\": " << layer_profiler.layers.size()
                          << ",\n  \"profiled_graph_mean_ms\": " << mean(profiled_graph)
                          << ",\n  \"profiled_graph_p50_ms\": " << percentile(profiled_graph, .5)
                          << ",\n  \"profiled_graph_p95_ms\": " << percentile(profiled_graph, .95)
                          << ",\n  \"sum_layer_mean_ms\": " << sum_layer_mean_ms
                          << ",\n  \"timing_includes_profiler_events\": true,\n  \"included_in_benchmark_statistics\": false\n}\n";
            write_text_file(output / "layer-profile-summary.json", layer_summary.str());
            profile_graph.close();
        }
        require(logger.errors == 0, "TensorRT reported errors");
        std::ostringstream identity;
        identity << "capture_kind=development-optimization\nformal_acceptance=false\nengine_sha256=" << sha256_hex(plan)
                 << "\nimages_sha256=" << sha256_hex(images) << "\nprofile=" << model_profile(profile).name
                 << "\nframes=" << frames << "\ntiming_input=0\ntiming_excludes=H2D,D2H,external_verification,initialization,warmup,layer_profiling\n"
                 << "persistent_cache_bytes_requested=" << persistent_cache_bytes << "\npersistent_cache_bytes=" << actual_cache_bytes
                 << "\nnvtx_requested=" << nvtx << "\nnvtx=" << actual_nvtx << "\nlayer_iterations=" << layer_iterations << '\n'
                 << scan_trt_plugin_identity();
        write_text_file(output / "identity.txt", identity.str());
        write_text_file(output / "samples.csv", samples.str());
        write_text_file(output / "runs.csv", runs.str());
        std::ostringstream summary;
        summary << std::fixed << std::setprecision(6)
                << "{\n  \"status\": \"PASS\",\n  \"capture_kind\": \"development-optimization\",\n  \"formal_acceptance\": false,\n"
                << "  \"engine_sha256\": \"" << sha256_hex(plan) << "\",\n  \"images_sha256\": \"" << sha256_hex(images) << "\",\n"
                << "  \"frames_verified\": " << frames << ",\n  \"timing_input\": 0,\n  \"iterations_per_run\": " << iterations
                << ",\n  \"warmup_per_run\": " << warmup << ",\n  \"rounds\": " << rounds
                << ",\n  \"persistent_cache_bytes_requested\": " << persistent_cache_bytes << ",\n  \"persistent_cache_bytes\": " << actual_cache_bytes
                << ",\n  \"nvtx_requested\": \"" << nvtx << "\",\n  \"nvtx\": \"" << actual_nvtx << "\",\n  \"layer_iterations\": " << layer_iterations
                << ",\n  \"ordinary_gpu_mean_ms\": " << mean(ordinary) << ",\n  \"ordinary_gpu_p50_ms\": " << percentile(ordinary, .5)
                << ",\n  \"ordinary_gpu_p95_ms\": " << percentile(ordinary, .95) << ",\n  \"graph_gpu_mean_ms\": " << mean(graph)
                << ",\n  \"graph_gpu_p50_ms\": " << percentile(graph, .5) << ",\n  \"graph_gpu_p95_ms\": " << percentile(graph, .95)
                << ",\n  \"ordinary_repeat_and_graph_bits_equal\": true\n}\n";
        resources.close();
        write_text_file(output / "summary.json", summary.str());
        std::cout << summary.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_benchmark: " << error.what() << '\n';
        if (!output.empty()) {
            try { write_text_file(output / "FAILURE.txt", error.what()); } catch (...) {}
        }
        return 1;
    }
}

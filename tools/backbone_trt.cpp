// T01 backbone/neck engine (docs/BACKBONE_POLICY.md §5):
//   build ONNX ONNX_SHA256 ENGINE BUILD_REPORT
//   run   ENGINE ENGINE_SHA256 IMAGES IMAGES_SHA256 REPORT DUMP_DIR
// `build` imports the ONNX with the pinned nvonnxparser, builds under the
// head's builder settings and tactic check, writes the plan with checked I/O,
// reads it back and records the builder options, identity and per-layer
// tactics. `run` is a new process: deserialization, one execution context on a
// non-default stream, every input of IMAGES (FP32 little-endian [1,3,640,640]
// frames back to back) with poisoned outputs, a bit-identical rerun and a
// CUDA graph replay bit-identical to enqueueV3; DUMP_DIR/input-<i>.f32 holds
// p3_feat, p4_feat, p5_feat of input i for the policy comparison
// (training/t01/t01_backbone_ref.py compare).
#include "saccade/file_io.hpp"
#include "saccade/head_reference.hpp"
#include "saccade/head_trt_network.hpp"
#include "saccade/model_trt_network.hpp"
#include "saccade/scan_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/sha256.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
using namespace nvinfer1;
using namespace saccade;

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

void cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

std::span<const std::uint8_t> bytes(const std::string& value) { return {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()}; }

std::string checked_read(const std::string& path, const std::string& sha, const char* what) {
    auto data = read_text_file(path);
    const auto got = sha256_hex(data);
    require(got == sha, std::string(what) + " SHA-256 " + got + " differs from the expected " + sha);
    return data;
}

std::size_t count(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (const auto d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

double seconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::string device_identity() {
    int runtime = 0, driver = 0, device = 0;
    cuda(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    cuda(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    cuda(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp p{};
    cuda(cudaGetDeviceProperties(&p, device), "cudaGetDeviceProperties");
    char bus[32] = {};
    cuda(cudaDeviceGetPCIBusId(bus, sizeof bus, device), "cudaDeviceGetPCIBusId");
    std::ostringstream id;
    id << "cuda_runtime=" << runtime << "\ncuda_driver_api=" << driver << "\ngpu=" << p.name << "\nsm=" << p.major << '.' << p.minor << "\npci_bus=" << bus
       << '\n';
    return id.str();
}

void build(const std::string& onnx_path, const std::string& onnx_sha, const std::string& engine_path, const std::string& report_path, Logger& logger) {
    const auto onnx = checked_read(onnx_path, onnx_sha, "ONNX");
    const auto t0 = seconds();
    const auto engine = build_backbone_engine(logger, bytes(onnx));
    const auto build_seconds = seconds() - t0;
    write_binary_file(engine_path, engine.plan);
    const auto readback = read_text_file(engine_path);
    require(readback.size() == engine.plan.size() && std::memcmp(readback.data(), engine.plan.data(), readback.size()) == 0,
            "engine readback differs from the serialized plan");
    std::ostringstream report;
    {
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        require(runtime != nullptr, "createInferRuntime failed");
        std::unique_ptr<ICudaEngine> deserialized(runtime->deserializeCudaEngine(readback.data(), readback.size()));
        require(deserialized != nullptr, "deserializeCudaEngine of the written plan failed");
        report << "onnx_sha256=" << onnx_sha << "\nengine_bytes=" << readback.size() << "\nengine_sha256=" << sha256_hex(readback)
               << "\nbuild_seconds=" << build_seconds << '\n' << engine.build_options << onnx_parser_identity() << scan_trt_plugin_identity()
               << device_identity() << describe_scan_engine(*deserialized) << "layers_json=" << head_engine_layers(*deserialized) << '\n';
    }
    require(logger.errors == 0, "TensorRT reported errors");
    write_text_file(report_path, report.str());
    std::cout << "engine_sha256=" << sha256_hex(readback) << "\nengine_bytes=" << readback.size() << '\n';
}

// Device and pinned buffers of the engine's four tensors, released in the destructor.
struct Buffers {
    std::array<float*, 4> device{};
    std::array<float*, 4> host{};
    std::array<std::size_t, 4> counts{};
    ~Buffers() {
        for (auto* p : device) if (p) cudaFree(p);
        for (auto* p : host) if (p) cudaFreeHost(p);
    }
};

void run(const std::string& engine_path, const std::string& engine_sha, const std::string& images_path, const std::string& images_sha,
         const std::string& report_path, const std::string& dump_dir, Logger& logger) {
    static_assert(std::endian::native == std::endian::little, "image files are little-endian FP32");
    const auto plan = checked_read(engine_path, engine_sha, "engine");
    const auto images = checked_read(images_path, images_sha, "images");
    const auto frame_values = count(model_input_shape());
    require(!images.empty() && images.size() % (frame_values * sizeof(float)) == 0, "images: not a whole number of [1,3,640,640] frames");
    const auto frames = images.size() / (frame_values * sizeof(float));
    std::ostringstream report;
    report << "engine_sha256=" << engine_sha << "\nimages_file=" << images_path << "\nimages_sha256=" << images_sha << "\nimages_frames=" << frames
           << "\npolicy=t01-backbone-fp32/1\n" << scan_trt_plugin_identity() << onnx_parser_identity() << device_identity();
    bool pass = true;
    try {
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        require(runtime != nullptr, "createInferRuntime failed");
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
        require(engine != nullptr, "deserializeCudaEngine failed");
        require(engine->getNbAuxStreams() == 0 && engine->getNbIOTensors() == 4, "the engine must have 4 I/O tensors and no auxiliary streams");
        report << describe_scan_engine(*engine);
        const std::array<std::string, 4> names{kModelInputName, kBackboneOutputNames[0], kBackboneOutputNames[1], kBackboneOutputNames[2]};
        const std::array<std::vector<std::int64_t>, 4> shapes{model_input_shape(), head_input_shape(0), head_input_shape(1), head_input_shape(2)};
        Buffers b;
        std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
        require(context != nullptr, "createExecutionContext failed");
        for (std::size_t t = 0; t < 4; ++t) {
            const char* name = names[t].c_str();
            const auto dims = engine->getTensorShape(name);
            bool same = dims.nbDims == 4;
            for (int k = 0; same && k < 4; ++k) same = dims.d[k] == shapes[t][static_cast<std::size_t>(k)];
            require(same && engine->getTensorDataType(name) == DataType::kFLOAT && engine->getTensorFormat(name) == TensorFormat::kLINEAR &&
                        engine->getTensorLocation(name) == TensorLocation::kDEVICE &&
                        (engine->getTensorIOMode(name) == TensorIOMode::kINPUT) == (t == 0),
                    "engine tensor " + names[t] + " differs from the backbone I/O");
            b.counts[t] = count(shapes[t]);
            cuda(cudaMalloc(reinterpret_cast<void**>(&b.device[t]), b.counts[t] * sizeof(float)), "cudaMalloc");
            cuda(cudaMallocHost(reinterpret_cast<void**>(&b.host[t]), b.counts[t] * sizeof(float)), "cudaMallocHost");
            require(context->setTensorAddress(name, b.device[t]), "setTensorAddress " + names[t]);
        }
        cudaStream_t stream = nullptr;
        cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        const auto destroy_stream = [&] { if (stream) cudaStreamDestroy(stream); stream = nullptr; };
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        try {
            // One step: input H2D, outputs poisoned (all bits set: NaN), the engine (or its graph), D2H, synchronize.
            const auto step = [&](std::size_t frame, bool use_graph) {
                std::memcpy(b.host[0], images.data() + frame * frame_values * sizeof(float), frame_values * sizeof(float));
                cuda(cudaMemcpyAsync(b.device[0], b.host[0], frame_values * sizeof(float), cudaMemcpyHostToDevice, stream), "H2D");
                for (std::size_t t = 1; t < 4; ++t) cuda(cudaMemsetAsync(b.device[t], 0xFF, b.counts[t] * sizeof(float), stream), "poison");
                if (use_graph) cuda(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
                else require(context->enqueueV3(stream), "enqueueV3 failed");
                std::vector<float> out;
                for (std::size_t t = 1; t < 4; ++t)
                    cuda(cudaMemcpyAsync(b.host[t], b.device[t], b.counts[t] * sizeof(float), cudaMemcpyDeviceToHost, stream), "D2H");
                cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
                for (std::size_t t = 1; t < 4; ++t) out.insert(out.end(), b.host[t], b.host[t] + b.counts[t]);
                return out;
            };
            std::vector<std::vector<float>> results;
            for (std::size_t i = 0; i < frames; ++i) {
                auto out = step(i, false);
                std::size_t nonfinite = 0;
                for (const float v : out) nonfinite += !std::isfinite(v);
                const auto again = step(i, false);
                const bool same = again.size() == out.size() && std::memcmp(again.data(), out.data(), out.size() * sizeof(float)) == 0;
                const std::string_view input(images.data() + i * frame_values * sizeof(float), frame_values * sizeof(float));
                const std::string_view blob(reinterpret_cast<const char*>(out.data()), out.size() * sizeof(float));
                const auto path = dump_dir + "/input-" + std::to_string(i) + ".f32";
                write_binary_file(path, bytes(std::string(blob)));
                report << "input\t" << i << "\tsha256=" << sha256_hex(input) << "\tdump=" << path << "\tsha256=" << sha256_hex(blob)
                       << "\tnonfinite=" << nonfinite << '\n';
                report << "determinism\t" << i << "\trerun_bit_identical\t" << (same ? "PASS" : "FAIL") << '\n';
                pass = pass && same && nonfinite == 0;
                results.push_back(std::move(out));
            }
            // The engine alone captured on the stream after the warm enqueues above.
            cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
            const bool enqueued = context->enqueueV3(stream);
            const auto ended = cudaStreamEndCapture(stream, &graph);
            require(enqueued && ended == cudaSuccess && graph != nullptr, "graph capture of enqueueV3 failed");
            cuda(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
            std::size_t nodes = 0;
            cuda(cudaGraphGetNodes(graph, nullptr, &nodes), "cudaGraphGetNodes");
            bool replay = true;
            for (std::size_t i = 0; i < frames; ++i) {
                const auto out = step(i, true);
                replay = replay && std::memcmp(out.data(), results[i].data(), out.size() * sizeof(float)) == 0;
            }
            report << "graph_replay\t" << frames << "_inputs\tnodes=" << nodes << "\tbit_identical_to_enqueueV3\t" << (replay ? "PASS" : "FAIL") << '\n';
            pass = pass && replay;
            cuda(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
            exec = nullptr;
            cuda(cudaGraphDestroy(graph), "cudaGraphDestroy");
            graph = nullptr;
            cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
            stream = nullptr;
        } catch (...) {
            if (stream) cudaStreamSynchronize(stream);
            if (exec) cudaGraphExecDestroy(exec);
            if (graph) cudaGraphDestroy(graph);
            destroy_stream();
            throw;
        }
        context.reset();
        engine.reset();
        cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        require(logger.errors == 0, "TensorRT reported errors");
    } catch (const std::exception& error) {
        report << "error\t" << error.what() << "\nresult\tFAIL\n";
        write_text_file(report_path, report.str());
        std::cout << report.str();
        throw;
    }
    report << "result\t" << (pass ? "PASS" : "FAIL") << " (execution checks; numeric judgement by t01_backbone_ref.py compare)\n";
    write_text_file(report_path, report.str());
    std::cout << report.str();
    require(pass, "execution checks failed");
}

} // namespace

int main(int argc, char** argv) {
    Logger logger;
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "build" && argc == 6) build(argv[2], argv[3], argv[4], argv[5], logger);
        else if (mode == "run" && argc == 8) run(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], logger);
        else throw std::invalid_argument("usage: backbone_trt build ONNX SHA ENGINE REPORT | run ENGINE SHA IMAGES SHA REPORT DUMP_DIR");
        std::cout.flush();
        require(static_cast<bool>(std::cout), "cannot write standard output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "backbone_trt: " << error.what() << '\n';
        return 1;
    }
}

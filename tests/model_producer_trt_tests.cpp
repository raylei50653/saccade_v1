#include "saccade/cleanup.hpp"
#include "saccade/file_io.hpp"
#include "saccade/model_producer_cuda.hpp"
#include "saccade/model_trt_network.hpp"
#include "saccade/postprocess_cuda.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/sha256.hpp"

#include <NvInfer.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Bounded live F03 -> producer -> F04 integration. Input staging and terminal
// diagnostic readbacks are test operations; the pipeline has no intermediate
// host count/status readback. This is not a complete asynchronous session.
namespace {
using namespace saccade;
using namespace nvinfer1;
unsigned checks = 0;
void require(bool ok, const std::string& why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
void cu(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
class Logger final : public ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) std::cerr << "TensorRT: " << message << '\n';
    }
};
struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { cu(cudaMalloc(&p, bytes)); }
    ~Device() { if (p) (void)cudaFree(p); }
    Device(const Device&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
    void close(CleanupReport& report) { report.release(p, [](void* x) { cu(cudaFree(x)); }); }
};
struct Stream {
    cudaStream_t s = nullptr;
    Stream() { cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)); }
    ~Stream() { if (s) { (void)cudaStreamSynchronize(s); (void)cudaStreamDestroy(s); } }
    void close(CleanupReport& report) {
        report.release(s, [](cudaStream_t x) { cu(cudaStreamDestroy(x)); });
    }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Graph() { if (exec) (void)cudaGraphExecDestroy(exec); if (graph) (void)cudaGraphDestroy(graph); }
    void close(CleanupReport& report) {
        report.release(exec, [](cudaGraphExec_t x) { cu(cudaGraphExecDestroy(x)); });
        report.release(graph, [](cudaGraph_t x) { cu(cudaGraphDestroy(x)); });
    }
};
std::size_t elements(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (auto d : shape) { require(d > 0, "non-static model I/O shape"); n *= static_cast<std::size_t>(d); }
    return n;
}
void tensor(const ICudaEngine& engine, const std::string& name, DataType type,
            TensorIOMode mode, const std::vector<std::int64_t>& shape, std::set<std::string>& names) {
    require(names.insert(name).second, "duplicate tensor in model_io_spec");
    require(engine.getTensorDataType(name.c_str()) == type && engine.getTensorIOMode(name.c_str()) == mode,
            "model tensor mode/type mismatch: " + name);
    const auto d = engine.getTensorShape(name.c_str());
    require(d.nbDims == static_cast<int>(shape.size()), "model tensor rank mismatch: " + name);
    for (int i = 0; i < d.nbDims; ++i)
        require(d.d[i] == shape[static_cast<std::size_t>(i)], "model tensor shape mismatch: " + name);
    require(engine.getTensorLocation(name.c_str()) == TensorLocation::kDEVICE &&
            engine.getTensorFormat(name.c_str()) == TensorFormat::kLINEAR && !engine.isShapeInferenceIO(name.c_str()),
            "model tensor must be a linear device execution tensor: " + name);
}
void compare(const DetectionCandidates& actual, const DetectionCandidates& reference, const std::string& label) {
    require(std::memcmp(&actual.header, &reference.header, sizeof(actual.header)) == 0, label + ": full header differs");
    for (unsigned i = 0; i < kPostprocessCapacity; ++i) {
        auto a = actual.rows[i], b = reference.rows[i];
        if (i < reference.header.count) {
            require(std::isfinite(a.score) && std::abs(a.score - b.score) <= 4 * std::numeric_limits<float>::epsilon(),
                    label + ": score differs");
            a.score = b.score = 0;
        }
        require(std::memcmp(&a, &b, sizeof(a)) == 0, label + ": complete row/unused bytes differ");
    }
}
std::string float_hash(const std::vector<float>& data) {
    return sha256_hex(std::string_view(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float)));
}
constexpr const char* kEngineHash = "c6cff1936194e679ce11c3736cab552e4a73a2b1a9ae5508bc3e5d1b3fbab9c5";
constexpr const char* kInputsHash = "14dda6867e23ed8a2d83086224d71a445a7fdcb606c7507a3f39c0e5faf60aa0";
constexpr std::array<unsigned, 6> kHistoricalKept{32, 28, 39, 2, 3, 1};
constexpr std::array<const char*, 6> kHistoricalRawHashes{
    "12df20281dd2bd1b4dd2aefbffcb2ce598115e993f530b1d3b82ae2dbdf763e5",
    "02558978a3e7437cc82c24c3532386588d596c2f9f54a6760d811a4543a9cf16",
    "b69dc2f3a468540d507a35ee5da767d7607c88948c68418b1dfa8296b55ea85f",
    "c273efb9986a957a79ff29ec2c9ffa2fb98469fe284f2d11c4f3fa63b559c815",
    "a6c72c191da4fad43ae298ea0f677037c1f8ae36d04618a1e7305890f37def57",
    "ccd755c5f7a087bce0168e36076e377053fc6678fe8b5dc0ad981fcd52e2e771"};

void run(const std::string& engine_path, std::filesystem::path input_path, unsigned input_count) {
    if (std::filesystem::is_directory(input_path)) input_path /= "f03-inputs.f32";
    const auto plan = read_text_file(engine_path), input_bytes = read_text_file(input_path);
    require(sha256_hex(plan) == kEngineHash, "engine differs from the pinned accepted F03 engine");
    require(sha256_hex(input_bytes) == kInputsHash, "input corpus differs from the six pinned F03 inputs");
    require(input_bytes.size() == 6 * kModelProducerInputElements * sizeof(float), "fixed input corpus length");
    require(input_count >= 1 && input_count <= 6, "--inputs must be in [1,6]");
    std::cout << "engine_sha256=" << sha256_hex(plan) << "\ninputs_sha256=" << sha256_hex(input_bytes) << '\n';
    cu(cudaSetDevice(0));
    std::cout << model_producer_cuda_identity() << postprocess_cuda_identity();
    std::cout << model_producer_cuda_load_kernels() << postprocess_cuda_load_kernels();
    Logger logger;
    register_scan_trt_plugins(*getPluginRegistry());
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    require(runtime != nullptr, "createInferRuntime");
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    require(engine != nullptr && engine->getNbAuxStreams() == 0, "deserialize F03 with zero auxiliary streams");
    const auto spec = model_io_spec();
    require(spec.inputs.size() == 1 && spec.outputs.size() == 6 && spec.statuses.size() == 9 &&
            spec.check_inputs && spec.check_outputs, "fixed model_io_spec contract");
    std::set<std::string> names;
    for (const auto& t : spec.inputs) tensor(*engine, t.name, DataType::kFLOAT, TensorIOMode::kINPUT, t.shape, names);
    for (const auto& t : spec.outputs) tensor(*engine, t.name, DataType::kFLOAT, TensorIOMode::kOUTPUT, t.shape, names);
    for (const auto& t : spec.statuses) tensor(*engine, t.name, DataType::kINT32, TensorIOMode::kOUTPUT, {2}, names);
    require(engine->getNbIOTensors() == static_cast<int>(names.size()), "unexpected model I/O tensor count");
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
        require(names.contains(engine->getIOTensorName(i)), "engine I/O not in model_io_spec");
    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
    require(context != nullptr, "create F03 execution context");
    Stream stream;
    Device images{kModelProducerInputElements * sizeof(float)}, raw{705600 * sizeof(float)};
    // TensorRT I/O addresses use 256-byte aligned arena subranges. Only the
    // first sizeof(ScanStatus) bytes in each subrange belong to that status.
    Device statuses{9 * 256}, producer{sizeof(PostprocessProducerStatus)};
    Device producer_workspace{model_producer_workspace_bytes()}, output{sizeof(DetectionCandidates)};
    Device f04_workspace{postprocess_workspace_bytes()};
    ModelProducerPlugins plugins{};
    PostprocessRawView raw_view;
    require(context->setTensorAddress(spec.inputs[0].name.c_str(), images.p), "bind images");
    std::size_t offset = 0;
    for (unsigned i = 0; i < 6; ++i) {
        auto* p = raw.as<float>() + offset;
        const auto n = elements(spec.outputs[i].shape);
        require(context->setTensorAddress(spec.outputs[i].name.c_str(), p), "bind raw output");
        if (i < 3) { raw_view.cls[i] = p; raw_view.cls_elements[i] = n; }
        else { raw_view.reg[i - 3] = p; raw_view.reg_elements[i - 3] = n; }
        offset += n;
    }
    require(offset == 705600, "six raw output extent");
    for (unsigned i = 0; i < 9; ++i) {
        plugins[i] = reinterpret_cast<ScanStatus*>(statuses.as<unsigned char>() + i * 256);
        require(context->setTensorAddress(spec.statuses[i].name.c_str(), plugins[i]), "bind plugin status");
    }
    PostprocessRequest request;
    request.epoch = 0xF0300001;
    request.frame = 0;
    const auto launch = [&] {
        require(context->enqueueV3(stream.s), "F03 enqueueV3");
        model_producer_cuda(images.as<float>(), kModelProducerInputElements, raw_view, plugins,
                            producer.as<PostprocessProducerStatus>(), producer_workspace.p,
                            model_producer_workspace_bytes(), stream.s);
        postprocess_cuda(raw_view, request, producer.as<PostprocessProducerStatus>(),
                         output.as<DetectionCandidates>(), sizeof(DetectionCandidates),
                         f04_workspace.p, postprocess_workspace_bytes(), stream.s);
    };
    std::vector<float> host_raw(705600);
    const auto snapshot = [&] {
        DetectionCandidates result{};
        PostprocessProducerStatus aggregate{};
        std::array<ScanStatus, 9> retired{};
        cu(cudaMemcpyAsync(&result, output.p, sizeof(result), cudaMemcpyDeviceToHost, stream.s));
        cu(cudaMemcpyAsync(&aggregate, producer.p, sizeof(aggregate), cudaMemcpyDeviceToHost, stream.s));
        for (unsigned i = 0; i < 9; ++i)
            cu(cudaMemcpyAsync(&retired[i], plugins[i], sizeof(ScanStatus), cudaMemcpyDeviceToHost, stream.s));
        cu(cudaMemcpyAsync(host_raw.data(), raw.p, host_raw.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.s));
        cu(cudaStreamSynchronize(stream.s));
        for (const auto& s : retired) require(s.flags == 0 && s.done == 0, "all plugin statuses retired");
        require(aggregate.done == kPostprocessProducerDone, "aggregate completion marker");
        return std::make_pair(result, aggregate);
    };
    const auto host_view = [&] {
        auto view = raw_view;
        for (unsigned i = 0; i < 3; ++i) {
            view.cls[i] = host_raw.data() + (raw_view.cls[i] - raw.as<float>());
            view.reg[i] = host_raw.data() + (raw_view.reg[i] - raw.as<float>());
        }
        return view;
    };
    const auto stage = [&](unsigned i) {
        cu(cudaMemcpyAsync(images.p, input_bytes.data() + i * kModelProducerInputElements * sizeof(float),
                           kModelProducerInputElements * sizeof(float), cudaMemcpyHostToDevice, stream.s));
    };
    // Slot initialization establishes invalid statuses before the first engine
    // producer; subsequent successful/failed adapter calls retire them again.
    cu(cudaMemsetAsync(statuses.p, 0, 9 * 256, stream.s));
    // Warm up the exact context and pointers before graph capture.
    stage(0); launch(); cu(cudaStreamSynchronize(stream.s));
    for (unsigned i = 0; i < input_count; ++i) {
        request.frame = i;
        stage(i); launch();
        const auto [actual, aggregate] = snapshot();
        require(aggregate.flags == 0 && actual.header.flags == 0, "live input accepted");
        require(float_hash(host_raw) == kHistoricalRawHashes[i], "live raw differs from historical accepted bytes");
        compare(actual, postprocess_reference(host_view(), request, aggregate), "live/" + std::to_string(i));
        require(actual.header.kept == kHistoricalKept[i], "independent historical kept count");
        launch();
        const auto [repeat, repeat_status] = snapshot();
        require(repeat_status.flags == 0 && std::memcmp(&actual, &repeat, sizeof(actual)) == 0,
                "live same-input rerun complete output differs");
        std::cout << "live_input=" << i << " kept=" << actual.header.kept << " suppressed=" << actual.header.suppressed
                  << " raw_sha256=" << float_hash(host_raw) << '\n';
    }
    request.frame = 100;
    stage(0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    cu(cudaMemcpyAsync(images.as<float>() + kModelProducerInputElements - 1, &nan, sizeof(nan), cudaMemcpyHostToDevice, stream.s));
    launch();
    const auto [failed, failed_status] = snapshot();
    require(failed_status.flags == kPostprocessNumeric && failed.header.flags == kPostprocessUpstream,
            "non-finite input must propagate through aggregate to F04");
    compare(failed, postprocess_reference(host_view(), request, failed_status), "live/nonfinite");
    stage(0); launch();
    const auto [recovered, recovered_status] = snapshot();
    require(recovered_status.flags == 0 && recovered.header.flags == 0 && recovered.header.kept == kHistoricalKept[0],
            "primitive failure-to-success reuse");
    compare(recovered, postprocess_reference(host_view(), request, recovered_status), "live/recovered");
    Graph graph;
    cu(cudaStreamBeginCapture(stream.s, cudaStreamCaptureModeThreadLocal));
    launch();
    cu(cudaStreamEndCapture(stream.s, &graph.graph));
    cu(cudaGraphInstantiate(&graph.exec, graph.graph, 0));
    for (unsigned replay = 0; replay < 3; ++replay) {
        cu(cudaGraphLaunch(graph.exec, stream.s));
        const auto [actual, aggregate] = snapshot();
        require(aggregate.flags == 0 && std::memcmp(&actual, &recovered, sizeof(actual)) == 0,
                "live engine+aggregate+F04 graph replay differs");
    }
    CleanupReport cleanup;
    cu(cudaStreamSynchronize(stream.s));
    graph.close(cleanup);
    context.reset(); engine.reset(); runtime.reset();
    f04_workspace.close(cleanup); output.close(cleanup); producer_workspace.close(cleanup);
    producer.close(cleanup); statuses.close(cleanup); raw.close(cleanup); images.close(cleanup);
    stream.close(cleanup);
    cleanup.throw_if_failed();
    std::cout << "live_integration=PASS inputs=" << input_count << " checks=" << checks
              << " full_async_session=UNEXECUTED P01=OPEN\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 5) throw std::invalid_argument("usage: model_producer_trt_tests ENGINE INPUTS_OR_DIRECTORY [--inputs N]");
        unsigned inputs = 6;
        if (argc == 5) {
            require(std::string(argv[3]) == "--inputs", "unknown option");
            inputs = static_cast<unsigned>(std::stoul(argv[4]));
        }
        run(argv[1], argv[2], inputs);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

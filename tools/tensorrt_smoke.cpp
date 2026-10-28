#include "saccade/cleanup.hpp"
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace nvinfer1;
constexpr std::size_t count = 256;
constexpr std::size_t bytes = count * sizeof(float);
constexpr std::size_t builder_workspace = 16 * 1024 * 1024;
// Fixed before building/running. These binary fractions and operations have
// exactly representable FP32 answers throughout the chosen input range.
constexpr float atol = 0.0F, rtol = 0.0F;

void require(bool condition, const std::string& why) {
    if (!condition) throw std::runtime_error(why);
}
void cuda_check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
class Logger final : public ILogger {
public:
    std::atomic_bool error{false};
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) error = true;
        try { std::cerr << "TensorRT[" << static_cast<int>(severity) << "] " << message << '\n'; }
        catch (...) { error = true; }
    }
};
struct Resources {
    cudaStream_t stream{};
    void *input{}, *output{}, *workspace{};
    float *host_input{}, *host_output{};
    void close() {
        saccade::CleanupReport report;
        // Caller drains the stream and destroys the execution context first.
        report.release(workspace, [](void* p) { cuda_check(cudaFree(p), "cudaFree(workspace)"); });
        report.release(output, [](void* p) { cuda_check(cudaFree(p), "cudaFree(output)"); });
        report.release(input, [](void* p) { cuda_check(cudaFree(p), "cudaFree(input)"); });
        report.release(host_output, [](float* p) { cuda_check(cudaFreeHost(p), "cudaFreeHost(output)"); });
        report.release(host_input, [](float* p) { cuda_check(cudaFreeHost(p), "cudaFreeHost(input)"); });
        report.release(stream, [](cudaStream_t s) { cuda_check(cudaStreamDestroy(s), "cudaStreamDestroy"); });
        report.throw_if_failed();
    }
};

std::string loaded_libraries() {
    const std::string maps = saccade::read_text_file("/proc/self/maps");
    std::istringstream lines(maps);
    std::set<std::string> paths;
    for (std::string line; std::getline(lines, line);) {
        const auto start = line.find('/');
        if (start == std::string::npos) continue;
        const auto path = line.substr(start);
        if (path.find("libnvinfer") != std::string::npos || path.find("libcuda") != std::string::npos ||
            path.find("libcublas") != std::string::npos || path.find("libnvrtc") != std::string::npos)
            paths.insert(path);
    }
    std::string result;
    for (const auto& path : paths) result += "loaded_library=" + path + '\n';
    return result;
}

void run(const std::filesystem::path& engine_path, const std::filesystem::path& report_path) {
    require(std::filesystem::absolute(engine_path).lexically_normal() !=
            std::filesystem::absolute(report_path).lexically_normal(), "engine and report paths must differ");
    if (std::filesystem::exists(engine_path) && std::filesystem::exists(report_path))
        require(!std::filesystem::equivalent(engine_path, report_path), "engine and report paths alias");
    require(getInferLibVersion() == NV_TENSORRT_VERSION && getInferLibBuildVersion() == NV_TENSORRT_BUILD,
            "TensorRT runtime/header version mismatch (requires 10.16.1.11)");
    Dl_info library{};
    require(dladdr(reinterpret_cast<void*>(&getInferLibVersion), &library) != 0 && library.dli_fname,
            "dladdr(getInferLibVersion) failed");
    require(std::filesystem::equivalent(library.dli_fname, SACCADE_TRT_EXPECTED_LIBRARY),
            "loaded TensorRT library differs from configured SDK: " + std::string(library.dli_fname));
    int runtime_version{}, driver_version{};
    cuda_check(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
    cuda_check(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
    require(runtime_version >= 13020 && runtime_version < 14000 && driver_version >= runtime_version,
            "CUDA runtime/driver incompatible with pinned CUDA 13.x SDK");
    cuda_check(cudaSetDevice(0), "cudaSetDevice(0)");
    cudaDeviceProp gpu{};
    cuda_check(cudaGetDeviceProperties(&gpu, 0), "cudaGetDeviceProperties");
    require(gpu.major >= 7 && (gpu.major != 7 || gpu.minor >= 5), "GPU SM must be at least 7.5");
    char pci_bus[32]{};
    cuda_check(cudaDeviceGetPCIBusId(pci_bus, sizeof(pci_bus), 0), "cudaDeviceGetPCIBusId");

    Logger logger;
    std::unique_ptr<IHostMemory> plan;
    {
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
        require(builder != nullptr, "createInferBuilder failed");
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(
            1U << static_cast<std::uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        require(network != nullptr, "createNetworkV2 failed");
        auto* x = network->addInput("x", DataType::kFLOAT, Dims4{1, 1, 1, static_cast<int>(count)});
        require(x != nullptr, "addInput(x) failed");
        const float two = 2.0F, one = 1.0F;
        auto* c2 = network->addConstant(Dims4{1, 1, 1, 1}, Weights{DataType::kFLOAT, &two, 1});
        auto* c1 = network->addConstant(Dims4{1, 1, 1, 1}, Weights{DataType::kFLOAT, &one, 1});
        require(c2 && c1, "addConstant failed");
        auto* product = network->addElementWise(*x, *c2->getOutput(0), ElementWiseOperation::kPROD);
        require(product != nullptr, "addElementWise(PROD) failed");
        auto* sum = network->addElementWise(*product->getOutput(0), *c1->getOutput(0), ElementWiseOperation::kSUM);
        require(sum != nullptr, "addElementWise(SUM) failed");
        sum->getOutput(0)->setName("y");
        network->markOutput(*sum->getOutput(0));
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        require(config != nullptr, "createBuilderConfig failed");
        config->clearFlag(BuilderFlag::kTF32);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, builder_workspace);
        config->setMaxAuxStreams(0);
        config->setBuilderOptimizationLevel(0);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        require(!config->getFlag(BuilderFlag::kTF32), "TF32 disable failed");
        plan.reset(builder->buildSerializedNetwork(*network, *config));
        require(plan != nullptr && plan->size() > 0 && !logger.error, "buildSerializedNetwork failed");
    } // Builder and network are gone before persisted plan is reloaded.
    const std::string plan_hash = saccade::sha256_hex(
        std::string_view(static_cast<const char*>(plan->data()), plan->size()));
    saccade::write_binary_file(engine_path, {static_cast<const std::uint8_t*>(plan->data()), plan->size()});
    const auto saved_size = plan->size();
    plan.reset();
    const auto saved = saccade::read_text_file(engine_path);
    require(saved.size() == saved_size && saccade::sha256_hex(saved) == plan_hash,
            "persisted engine size/hash differs from serialized engine: " + engine_path.string());
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    require(runtime != nullptr, "createInferRuntime failed");
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(saved.data(), saved.size()));
    require(engine != nullptr && !logger.error, "deserializeCudaEngine failed");
    require(engine->getNbIOTensors() == 2 && engine->getNbAuxStreams() == 0, "unexpected engine tensor/stream count");
    for (const char* name : {"x", "y"}) {
        const auto shape = engine->getTensorShape(name);
        require(shape.nbDims == 4 && shape.d[0] == 1 && shape.d[1] == 1 && shape.d[2] == 1 &&
                shape.d[3] == static_cast<int>(count) && engine->getTensorDataType(name) == DataType::kFLOAT &&
                engine->getTensorLocation(name) == TensorLocation::kDEVICE &&
                engine->getTensorFormat(name) == TensorFormat::kLINEAR, "engine I/O contract mismatch: " + std::string(name));
    }
    require(engine->getTensorIOMode("x") == TensorIOMode::kINPUT &&
            engine->getTensorIOMode("y") == TensorIOMode::kOUTPUT, "engine tensor direction mismatch");
    std::string engine_information;
    {
        std::unique_ptr<IEngineInspector> inspector(engine->createEngineInspector());
        require(inspector != nullptr, "createEngineInspector failed");
        const char* information = inspector->getEngineInformation(LayerInformationFormat::kJSON);
        require(information != nullptr, "getEngineInformation(JSON) failed");
        engine_information = information;
    }
    const auto workspace_bytes = engine->getDeviceMemorySizeV2();
    require(workspace_bytes >= 0 && static_cast<std::uint64_t>(workspace_bytes) <=
            std::numeric_limits<std::size_t>::max(), "invalid execution workspace byte size");
    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED));
    require(context != nullptr, "createExecutionContext(USER_MANAGED) failed");
    Resources r;
    std::string comparison, libraries;
    try {
        cuda_check(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        require(r.stream != nullptr, "explicit CUDA stream must be non-default");
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&r.host_input), bytes), "cudaMallocHost(input)");
        cuda_check(cudaMallocHost(reinterpret_cast<void**>(&r.host_output), bytes), "cudaMallocHost(output)");
        cuda_check(cudaMalloc(&r.input, bytes), "cudaMalloc(input)");
        cuda_check(cudaMalloc(&r.output, bytes), "cudaMalloc(output)");
        if (workspace_bytes > 0) cuda_check(cudaMalloc(&r.workspace, static_cast<std::size_t>(workspace_bytes)), "cudaMalloc(workspace)");
        context->setDeviceMemoryV2(r.workspace, workspace_bytes);
        require(context->setTensorAddress("x", r.input) && context->setTensorAddress("y", r.output), "setTensorAddress failed");
        for (std::size_t i = 0; i < count; ++i) r.host_input[i] = static_cast<float>(static_cast<int>(i) - 128) / 8.0F;
        cuda_check(cudaMemcpyAsync(r.input, r.host_input, bytes, cudaMemcpyHostToDevice, r.stream), "cudaMemcpyAsync(H2D)");
        cuda_check(cudaMemsetAsync(r.output, 0xff, bytes, r.stream), "cudaMemsetAsync(output poison)");
        require(context->enqueueV3(r.stream), "enqueueV3 failed");
        cuda_check(cudaGetLastError(), "enqueue CUDA launch status");
        cuda_check(cudaMemcpyAsync(r.host_output, r.output, bytes, cudaMemcpyDeviceToHost, r.stream), "cudaMemcpyAsync(D2H)");
        cuda_check(cudaStreamSynchronize(r.stream), "cudaStreamSynchronize");
        require(!logger.error, "TensorRT reported an error during execution");
        std::ostringstream values;
        values << std::setprecision(std::numeric_limits<float>::max_digits10);
        float max_error = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const float expected = 2.0F * r.host_input[i] + 1.0F;
            const float actual = r.host_output[i];
            const float error = std::abs(actual - expected);
            require(std::isfinite(actual) && error <= atol + rtol * std::abs(expected),
                    "output comparison failed at index " + std::to_string(i));
            max_error = std::max(max_error, error);
            values << "value=" << i << ',' << r.host_input[i] << ',' << expected << ',' << actual << '\n';
        }
        values << "compared_elements=" << count << "\nmax_abs_error=" << max_error << "\ncomparison=PASS\n";
        comparison = values.str();
        libraries = loaded_libraries();
        context.reset();
        engine.reset();
        runtime.reset();
        r.close();
    } catch (...) {
        const auto original = std::current_exception();
        if (r.stream) {
            try { cuda_check(cudaStreamSynchronize(r.stream), "failure-path cudaStreamSynchronize"); }
            catch (const std::exception& e) { std::cerr << e.what() << '\n'; }
        }
        context.reset();
        engine.reset();
        runtime.reset();
        try { r.close(); }
        catch (const std::exception& e) { std::cerr << "cleanup: " << e.what() << '\n'; }
        std::rethrow_exception(original);
    }
    require(!logger.error, "TensorRT reported a teardown error");
    std::ostringstream report;
    report << "schema=1\npurpose=synthetic-native-sdk-smoke-only\nmodel_readiness=NOT_ASSESSED\n"
           << "build_commit=" << SACCADE_TRT_BUILD_COMMIT << "\nbuild_dirty=" << SACCADE_TRT_BUILD_DIRTY << '\n'
           << "compiler=" << SACCADE_TRT_COMPILER << "\nheader=" << SACCADE_TRT_HEADER_PATH << '\n'
           << "header_sha256=" << saccade::sha256_hex(saccade::read_text_file(SACCADE_TRT_HEADER_PATH)) << '\n'
           << "tensorrt=10.16.1.11\nloaded_nvinfer=" << std::filesystem::canonical(library.dli_fname).string() << '\n'
           << "cuda_runtime=" << runtime_version << "\ncuda_driver_api=" << driver_version << '\n'
           << "gpu=" << gpu.name << "\ngpu_pci_bus=" << pci_bus << "\ngpu_sm=" << gpu.major << '.' << gpu.minor << '\n'
           << "network=y=2x+1\nshape=1,1,1,256\ndtype=float32\ntyping=strong\ntf32=false\n"
           << "builder_optimization_level=0\nbuilder_workspace_limit=" << builder_workspace << '\n'
           << "timing_cache=builder-default-no-external-import-or-export\nplugin=none\naux_streams=0\nstream=explicit-nonblocking\n"
           << "execution_workspace_bytes=" << workspace_bytes << "\natol=0\nrtol=0\n"
           << "engine_bytes=" << saved.size() << "\nengine_sha256=" << plan_hash << '\n'
           << "engine_inspector_sha256=" << saccade::sha256_hex(engine_information) << '\n'
           << "engine_inspector_json_begin\n" << engine_information << "\nengine_inspector_json_end\n"
           << libraries << comparison << "cleanup=PASS\n";
    saccade::write_text_file(report_path, report.str());
    std::cout << "PASS native TensorRT build/save/reload/enqueue; elements=" << count
              << " max_abs_error=0 engine_sha256=" << plan_hash << '\n';
    std::cout.flush();
    require(static_cast<bool>(std::cout), "cannot write stdout");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("usage: tensorrt_smoke ENGINE_PATH REPORT_PATH");
        run(argv[1], argv[2]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tensorrt_smoke: " << e.what() << '\n';
        return 1;
    }
}

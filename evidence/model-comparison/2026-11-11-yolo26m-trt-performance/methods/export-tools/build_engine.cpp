#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt = nvinfer1;
static_assert(NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR == 16 && NV_TENSORRT_PATCH == 1 && NV_TENSORRT_BUILD == 11);
struct Logger : trt::ILogger {
    int errors = 0;
    void log(Severity s, const char* m) noexcept override {
        if (s <= Severity::kERROR) ++errors;
        if (s <= Severity::kWARNING) std::cerr << "TensorRT: " << m << '\n';
    }
};
void require(bool b, const std::string& m) { if (!b) throw std::runtime_error(m); }
void write(const std::string& p, const char* b, std::size_t n) {
    std::ofstream f(p, std::ios::binary); f.write(b, static_cast<std::streamsize>(n));
    f.close(); require(bool(f), "write failed: " + p);
}
std::string read(const std::string& p) {
    std::ifstream f(p, std::ios::binary); require(bool(f), "read failed: " + p);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
std::string dims(const trt::Dims& d) {
    std::ostringstream o; o << '[';
    for (int i = 0; i < d.nbDims; ++i) { if (i) o << ','; o << d.d[i]; }
    o << ']'; return o.str();
}
int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: build_engine MODEL.onnx MODEL.engine BUILD.json LAYERS.json");
        Logger logger;
        require(getInferLibVersion() == NV_TENSORRT_VERSION, "TensorRT library version mismatch");
        const auto onnx = read(argv[1]);
        const auto start = std::chrono::steady_clock::now();
        std::unique_ptr<trt::IBuilder> builder(trt::createInferBuilder(logger));
        require(bool(builder), "createInferBuilder failed");
        std::unique_ptr<trt::INetworkDefinition> network(builder->createNetworkV2(0));
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        require(parser->parse(onnx.data(), onnx.size()), "ONNX parse failed");
        require(network->getNbInputs() == 1, "expected one input");
        require(std::string(network->getInput(0)->getName()) == "images" &&
                dims(network->getInput(0)->getDimensions()) == "[1,3,768,1280]" &&
                network->getInput(0)->getType() == trt::DataType::kFLOAT, "unexpected input contract");
        require(network->getNbOutputs() == 6 || network->getNbOutputs() == 1, "unexpected output count");
        for (int i = 0; i < network->getNbOutputs(); ++i)
            network->getOutput(i)->setType(trt::DataType::kFLOAT);
        std::unique_ptr<trt::IBuilderConfig> config(builder->createBuilderConfig());
        config->clearFlag(trt::BuilderFlag::kTF32);
        config->setFlag(trt::BuilderFlag::kFP16);
        config->setMemoryPoolLimit(trt::MemoryPoolType::kWORKSPACE, 256ULL * 1024 * 1024);
        config->setBuilderOptimizationLevel(3);
        config->setMaxAuxStreams(0);
        config->setProfilingVerbosity(trt::ProfilingVerbosity::kDETAILED);
        std::unique_ptr<trt::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        require(bool(plan), "buildSerializedNetwork failed");
        write(argv[2], static_cast<const char*>(plan->data()), plan->size());
        const auto bytes = read(argv[2]);
        require(bytes.size() == plan->size(), "plan readback size differs");
        std::unique_ptr<trt::IRuntime> runtime(trt::createInferRuntime(logger));
        std::unique_ptr<trt::ICudaEngine> engine(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        require(bool(engine) && engine->getNbAuxStreams() == 0, "plan deserialization / aux validation failed");
        std::unique_ptr<trt::IEngineInspector> inspector(engine->createEngineInspector());
        const auto* layers = inspector->getEngineInformation(trt::LayerInformationFormat::kJSON);
        require(layers != nullptr, "engine inspection failed");
        write(argv[4], layers, std::char_traits<char>::length(layers));
        cudaDeviceProp gpu{}; int runtime_version = 0, driver_version = 0;
        require(cudaGetDeviceProperties(&gpu, 0) == cudaSuccess, "GPU identity failed");
        cudaRuntimeGetVersion(&runtime_version); cudaDriverGetVersion(&driver_version);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::ostringstream o;
        o << "{\"status\":\"PASS\",\"network\":\"weakly_typed\",\"onnx_parser_version\":" << getNvOnnxParserVersion()
          << ",\"tensorrt_version\":" << getInferLibVersion() << ",\"tensorrt_build\":" << getInferLibBuildVersion()
          << ",\"fp16\":" << (config->getFlag(trt::BuilderFlag::kFP16) ? "true" : "false")
          << ",\"tf32\":" << (config->getFlag(trt::BuilderFlag::kTF32) ? "true" : "false")
          << ",\"bf16\":" << (config->getFlag(trt::BuilderFlag::kBF16) ? "true" : "false")
          << ",\"int8\":" << (config->getFlag(trt::BuilderFlag::kINT8) ? "true" : "false")
          << ",\"optimization_level\":" << config->getBuilderOptimizationLevel()
          << ",\"workspace_bytes\":" << config->getMemoryPoolLimit(trt::MemoryPoolType::kWORKSPACE)
          << ",\"max_aux_streams\":" << config->getMaxAuxStreams() << ",\"actual_aux_streams\":" << engine->getNbAuxStreams()
          << ",\"profiling_verbosity\":\"detailed\",\"timing_cache\":\"none supplied\",\"engine_bytes\":" << bytes.size()
          << ",\"build_wall_seconds\":" << seconds << ",\"cuda_runtime\":" << runtime_version << ",\"cuda_driver_api\":" << driver_version
          << ",\"gpu\":\"" << gpu.name << "\",\"sm\":\"" << gpu.major << '.' << gpu.minor << "\",\"io\":[";
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const auto* name = engine->getIOTensorName(i);
            require(engine->getTensorDataType(name) == trt::DataType::kFLOAT, "I/O unexpectedly not FP32");
            if (i) o << ',';
            o << "{\"name\":\"" << name << "\",\"mode\":\""
              << (engine->getTensorIOMode(name) == trt::TensorIOMode::kINPUT ? "input" : "output")
              << "\",\"dtype\":\"FP32\",\"shape\":" << dims(engine->getTensorShape(name)) << '}';
        }
        o << "],\"tensor_rt_errors\":" << logger.errors << "}\n";
        require(logger.errors == 0, "TensorRT reported errors");
        const auto report = o.str(); write(argv[3], report.data(), report.size());
        std::cout << report;
    } catch (const std::exception& e) {
        std::cerr << "build_engine: " << e.what() << '\n'; return 1;
    }
}

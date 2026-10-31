#include "saccade/model_trt_network.hpp"
#include "saccade/head_reference.hpp"
#include "saccade/scan_trt_plugin.hpp"

#include <NvOnnxParser.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace saccade {
namespace {
using namespace nvinfer1;

void require(bool condition, const std::string& why) {
    if (!condition) throw std::runtime_error("model engine build: " + why);
}

bool same_dims(const Dims& dims, const std::vector<std::int64_t>& want) {
    if (dims.nbDims != static_cast<std::int32_t>(want.size())) return false;
    for (int k = 0; k < dims.nbDims; ++k)
        if (dims.d[k] != want[static_cast<std::size_t>(k)]) return false;
    return true;
}

// Parses the ONNX into `network` and checks its I/O; returns the three outputs in order.
std::array<ITensor*, 3> parse_backbone(INetworkDefinition& network, ILogger& logger, std::span<const std::uint8_t> onnx,
                                       std::unique_ptr<nvonnxparser::IParser>& parser) {
    parser.reset(nvonnxparser::createParser(network, logger));
    require(parser != nullptr, "createParser failed");
    if (!parser->parse(onnx.data(), onnx.size())) {
        std::string errors;
        for (std::int32_t i = 0; i < parser->getNbErrors(); ++i) errors += std::string("\n  ") + parser->getError(i)->desc();
        require(false, "ONNX parse failed:" + errors);
    }
    require(network.getNbInputs() == 1, "the ONNX has " + std::to_string(network.getNbInputs()) + " inputs");
    const auto* input = network.getInput(0);
    require(std::string(input->getName()) == kModelInputName && input->getType() == DataType::kFLOAT &&
                same_dims(input->getDimensions(), model_input_shape()),
            "the ONNX input is not images FP32 [1,3,640,640]");
    require(network.getNbOutputs() == 3, "the ONNX has " + std::to_string(network.getNbOutputs()) + " outputs");
    std::array<ITensor*, 3> out{};
    for (int k = 0; k < 3; ++k) {
        auto* t = network.getOutput(k);
        require(std::string(t->getName()) == kBackboneOutputNames[static_cast<std::size_t>(k)] && t->getType() == DataType::kFLOAT &&
                    same_dims(t->getDimensions(), head_input_shape(k)),
                std::string("ONNX output ") + std::to_string(k) + " is not " + kBackboneOutputNames[static_cast<std::size_t>(k)] +
                    " FP32 with the head input shape");
        out[static_cast<std::size_t>(k)] = t;
    }
    return out;
}

void require_policy_tactics(ILogger& logger, const std::vector<std::uint8_t>& plan) {
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    require(runtime != nullptr, "createInferRuntime failed");
    std::unique_ptr<ICudaEngine> built(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    require(built != nullptr, "deserializeCudaEngine of the built plan failed");
    const auto violations = head_tactic_violations(head_engine_layers(*built));
    std::string listed;
    for (const auto& v : violations) listed += (listed.empty() ? "" : ", ") + v;
    require(violations.empty(), "tactics outside the FP32 direct-summation model: " + listed);
}

} // namespace

std::vector<std::int64_t> model_input_shape() { return {1, 3, 640, 640}; }

std::string onnx_parser_identity() {
    std::ostringstream id;
    id << "onnx_parser_library=" << getNvOnnxParserVersion() << "\nonnx_parser_header_api=" << NV_ONNX_PARSER_VERSION << '\n';
    return id.str();
}

HeadEngine build_backbone_engine(ILogger& logger, std::span<const std::uint8_t> onnx) {
    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    require(builder != nullptr, "createInferBuilder failed");
    std::unique_ptr<INetworkDefinition> network(
        builder->createNetworkV2(1U << static_cast<std::uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    require(network != nullptr, "createNetworkV2 failed");
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    require(config != nullptr, "createBuilderConfig failed");
    config->clearFlag(BuilderFlag::kTF32);
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, kHeadBuilderWorkspace);
    config->setMaxAuxStreams(0);
    config->setBuilderOptimizationLevel(kHeadBuilderOptimizationLevel);
    config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
    HeadEngine engine;
    {
        std::unique_ptr<nvonnxparser::IParser> parser;
        parse_backbone(*network, logger, onnx, parser);
        std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        require(plan != nullptr && plan->size() > 0, "buildSerializedNetwork failed");
        const auto* bytes = static_cast<const std::uint8_t*>(plan->data());
        engine.plan.assign(bytes, bytes + plan->size());
    }
    std::ostringstream o;
    o << "network=strongly_typed_fp32\ntf32=off\nfp16=off\nint8=off\nworkspace_bytes=" << kHeadBuilderWorkspace
      << "\noptimization_level=" << kHeadBuilderOptimizationLevel << "\naux_streams=0\ntiming_cache=none\nprofiling_verbosity=detailed"
      << "\nimport=nvonnxparser\n";
    engine.build_options = o.str();
    require_policy_tactics(logger, engine.plan);
    return engine;
}

HeadEngine build_model_engine(ILogger& logger, std::span<const std::uint8_t> onnx, const HeadWeights& weights) {
    // Owns the parsed weights until the head's build has serialized the plan.
    std::unique_ptr<nvonnxparser::IParser> parser;
    HeadBuildOptions options;
    options.features = [&](INetworkDefinition& network) {
        auto out = parse_backbone(network, logger, onnx, parser);
        for (auto* t : out) network.unmarkOutput(*t);
        return out;
    };
    auto engine = build_head_engine(logger, weights, options);
    engine.build_options += "import=nvonnxparser (backbone), network API (head)\n";
    return engine;
}

FrameIoSpec model_io_spec() {
    auto spec = head_io_spec();
    spec.inputs = {{kModelInputName, model_input_shape()}};
    return spec;
}

} // namespace saccade

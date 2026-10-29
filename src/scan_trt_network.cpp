#include "saccade/scan_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace saccade {
namespace {
using namespace nvinfer1;

void require(bool condition, const std::string& why) {
    if (!condition) throw std::runtime_error("scan engine build: " + why);
}

struct Network {
    std::unique_ptr<IBuilder> builder;
    std::unique_ptr<INetworkDefinition> network;
    std::unique_ptr<IBuilderConfig> config;
    std::vector<std::unique_ptr<IPluginV3>> plugins; // owned by the caller of addPluginV3 until the build ends
    IOptimizationProfile* profile = nullptr;

    explicit Network(ILogger& logger) {
        register_scan_trt_plugins(*getPluginRegistry());
        builder.reset(createInferBuilder(logger));
        require(builder != nullptr, "createInferBuilder failed");
        network.reset(builder->createNetworkV2(1U << static_cast<std::uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        require(network != nullptr, "createNetworkV2 failed");
        config.reset(builder->createBuilderConfig());
        require(config != nullptr, "createBuilderConfig failed");
        config->clearFlag(BuilderFlag::kTF32);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, kScanTrtBuilderWorkspace);
        config->setMaxAuxStreams(0);
        config->setBuilderOptimizationLevel(0);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
    }
    ITensor* input(const char* name, Dims dims) {
        auto* t = network->addInput(name, DataType::kFLOAT, dims);
        require(t != nullptr, std::string("addInput ") + name);
        return t;
    }
    // Dynamic input: dims with -1 where the profile gives min/opt/max.
    ITensor* input(const char* name, Dims dims, Dims min, Dims opt, Dims max) {
        if (!profile) profile = builder->createOptimizationProfile();
        require(profile != nullptr, "createOptimizationProfile failed");
        require(profile->setDimensions(name, OptProfileSelector::kMIN, min) && profile->setDimensions(name, OptProfileSelector::kOPT, opt) &&
                    profile->setDimensions(name, OptProfileSelector::kMAX, max),
                std::string("profile for ") + name);
        return input(name, dims);
    }
    // Adds the plugin layer and names its payload and status outputs.
    ITensor* plugin(std::unique_ptr<IPluginV3> plugin, std::vector<ITensor*> inputs, const char* payload, const char* status, bool mark_payload) {
        auto* layer = network->addPluginV3(inputs.data(), static_cast<std::int32_t>(inputs.size()), nullptr, 0, *plugin);
        require(layer != nullptr && layer->getNbOutputs() == 2, std::string("addPluginV3 for ") + payload);
        plugins.push_back(std::move(plugin));
        layer->getOutput(0)->setName(payload);
        layer->getOutput(1)->setName(status);
        if (mark_payload) network->markOutput(*layer->getOutput(0));
        network->markOutput(*layer->getOutput(1));
        return layer->getOutput(0);
    }
    std::vector<std::uint8_t> build() {
        if (profile) require(config->addOptimizationProfile(profile) >= 0, "addOptimizationProfile failed");
        std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        require(plan != nullptr && plan->size() > 0, "buildSerializedNetwork failed");
        const auto* bytes = static_cast<const std::uint8_t*>(plan->data());
        return {bytes, bytes + plan->size()};
    }
};

Dims d1(std::int64_t a) { Dims d{}; d.nbDims = 1; d.d[0] = a; return d; }
Dims d2(std::int64_t a, std::int64_t b) { Dims d{}; d.nbDims = 2; d.d[0] = a; d.d[1] = b; return d; }
Dims d3(std::int64_t a, std::int64_t b, std::int64_t c) { Dims d{}; d.nbDims = 3; d.d[0] = a; d.d[1] = b; d.d[2] = c; return d; }
Dims d4(std::int64_t a, std::int64_t b, std::int64_t c, std::int64_t e) { Dims d{}; d.nbDims = 4; d.d[0] = a; d.d[1] = b; d.d[2] = c; d.d[3] = e; return d; }

const char* text(DataType t) {
    switch (t) {
    case DataType::kFLOAT: return "FP32";
    case DataType::kINT32: return "INT32";
    case DataType::kHALF: return "FP16";
    default: return "other";
    }
}
} // namespace

std::vector<std::uint8_t> build_selective_scan_engine(ILogger& logger, bool has_d, const ScanProfile& p) {
    Network n(logger);
    const auto& B = p.batch;
    const auto& L = p.length;
    const auto& Dc = p.channels;
    const auto& N = p.state;
    // A rows range over 1..Dc and C width over 1..N; the plugin keeps only {1, Dc} and {1, N}.
    std::vector<ITensor*> in{
        n.input("u", d3(-1, -1, -1), d3(B[0], L[0], Dc[0]), d3(B[1], L[1], Dc[1]), d3(B[2], L[2], Dc[2])),
        n.input("delta", d3(-1, -1, -1), d3(B[0], L[0], Dc[0]), d3(B[1], L[1], Dc[1]), d3(B[2], L[2], Dc[2])),
        n.input("A", d2(-1, -1), d2(1, N[0]), d2(1, N[1]), d2(Dc[2], N[2])),
        n.input("B", d3(-1, -1, -1), d3(B[0], L[0], N[0]), d3(B[1], L[1], N[1]), d3(B[2], L[2], N[2])),
        n.input("C", d3(-1, -1, -1), d3(B[0], L[0], 1), d3(B[1], L[1], 1), d3(B[2], L[2], N[2]))};
    if (has_d) in.push_back(n.input("D", d1(-1), d1(Dc[0]), d1(Dc[1]), d1(Dc[2])));
    n.plugin(create_selective_scan_plugin(has_d), in, "y", "scan_status", true);
    return n.build();
}

std::vector<std::uint8_t> build_cross_scan_expand_engine(ILogger& logger) {
    Network n(logger);
    auto* x = n.input("x", d4(-1, -1, -1, -1), d4(1, 1, 1, 1), d4(1, 128, 20, 20), d4(8, 512, 64, 64));
    n.plugin(create_cross_scan_expand_plugin(), {x}, "tokens", "expand_status", true);
    return n.build();
}

std::vector<std::uint8_t> build_cross_scan_merge_engine(ILogger& logger, std::int32_t height, std::int32_t width) {
    Network n(logger);
    const std::int64_t L = std::int64_t{height} * width;
    auto* y = n.input("y", d3(-1, L, -1), d3(4, L, 1), d3(4, L, 128), d3(32, L, 512));
    n.plugin(create_cross_scan_merge_plugin(height, width), {y}, "out", "merge_status", true);
    return n.build();
}

std::vector<std::uint8_t> build_cross_scan_pipeline_engine(ILogger& logger, const CrossScanShape& s, std::int32_t state, bool has_d) {
    Network n(logger);
    const auto B = static_cast<std::int64_t>(s.batch), C = static_cast<std::int64_t>(s.channels);
    const auto H = static_cast<std::int64_t>(s.height), W = static_cast<std::int64_t>(s.width);
    auto* x = n.input("x", d4(B, C, H, W));
    auto* tokens = n.plugin(create_cross_scan_expand_plugin(), {x}, "tokens", "expand_status", false);
    std::vector<ITensor*> in{tokens, n.input("delta", d3(4 * B, H * W, C)), n.input("A", d2(1, state)),
                             n.input("B", d3(4 * B, H * W, state)), n.input("C", d3(4 * B, H * W, 1))};
    if (has_d) in.push_back(n.input("D", d1(C)));
    auto* y = n.plugin(create_selective_scan_plugin(has_d), in, "y", "scan_status", false);
    n.plugin(create_cross_scan_merge_plugin(static_cast<std::int32_t>(H), static_cast<std::int32_t>(W)), {y}, "out", "merge_status", true);
    return n.build();
}

std::string describe_scan_engine(const ICudaEngine& engine) {
    std::ostringstream out;
    for (std::int32_t i = 0; i < engine.getNbIOTensors(); ++i) {
        const char* name = engine.getIOTensorName(i);
        const auto dims = engine.getTensorShape(name);
        out << "tensor=" << name << " mode=" << (engine.getTensorIOMode(name) == TensorIOMode::kINPUT ? "input" : "output")
            << " dtype=" << text(engine.getTensorDataType(name))
            << " format=" << (engine.getTensorFormat(name) == TensorFormat::kLINEAR ? "linear" : "other")
            << " location=" << (engine.getTensorLocation(name) == TensorLocation::kDEVICE ? "device" : "host") << " shape=(";
        for (int k = 0; k < dims.nbDims; ++k) out << (k ? "," : "") << dims.d[k];
        out << ") shape_inference_io=" << (engine.isShapeInferenceIO(name) ? "true" : "false") << '\n';
    }
    out << "aux_streams=" << engine.getNbAuxStreams() << "\nprofiles=" << engine.getNbOptimizationProfiles()
        << "\ndevice_memory_bytes=" << engine.getDeviceMemorySizeV2() << '\n';
    return out.str();
}

} // namespace saccade

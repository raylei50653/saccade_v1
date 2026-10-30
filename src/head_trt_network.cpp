#include "saccade/head_trt_network.hpp"
#include "saccade/head_reference.hpp"
#include "saccade/scan_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"

#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace saccade {
namespace {
using namespace nvinfer1;

void require(bool condition, const std::string& why) {
    if (!condition) throw std::runtime_error("head engine build: " + why);
}

Dims shape(std::initializer_list<std::int64_t> d) {
    Dims out{};
    out.nbDims = static_cast<std::int32_t>(d.size());
    int i = 0;
    for (const auto v : d) out.d[i++] = v;
    return out;
}
Permutation permutation(std::initializer_list<std::int32_t> order) {
    Permutation p{};
    int i = 0;
    for (const auto v : order) p.order[i++] = v;
    return p;
}
Weights weights(const HeadTensor& t) { return Weights{DataType::kFLOAT, t.values.data(), static_cast<std::int64_t>(t.values.size())}; }

std::string status_producer(const char* plugin, const std::string& level, const std::string& detail) {
    return std::string(kScanTrtNamespace) + "::" + plugin + " (" + level + (detail.empty() ? "" : ", " + detail) + ")";
}

// The network of SCAN_CONTRACT §2, one level at a time; every tensor that the
// reference keeps as an intermediate gets the same name.
class HeadNetwork {
public:
    HeadNetwork(ILogger& logger, const HeadWeights& w, const HeadBuildOptions& o) : w_(w), options_(o) {
        register_scan_trt_plugins(*getPluginRegistry());
        builder_.reset(createInferBuilder(logger));
        require(builder_ != nullptr, "createInferBuilder failed");
        network_.reset(builder_->createNetworkV2(1U << static_cast<std::uint32_t>(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        require(network_ != nullptr, "createNetworkV2 failed");
        config_.reset(builder_->createBuilderConfig());
        require(config_ != nullptr, "createBuilderConfig failed");
        config_->clearFlag(BuilderFlag::kTF32);
        config_->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, kHeadBuilderWorkspace);
        config_->setMaxAuxStreams(0);
        config_->setBuilderOptimizationLevel(kHeadBuilderOptimizationLevel);
        config_->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
    }

    HeadEngine build() {
        std::vector<ITensor*> outputs(6);
        for (int level = 0; level < kHeadLevels; ++level) {
            const auto shape_in = head_input_shape(level);
            auto* x = network_->addInput(kHeadInputNames[static_cast<std::size_t>(level)], DataType::kFLOAT,
                                         shape({shape_in[0], shape_in[1], shape_in[2], shape_in[3]}));
            require(x != nullptr, "addInput");
            const auto [cls, reg] = head_level(level, x);
            outputs[static_cast<std::size_t>(level)] = cls;
            outputs[static_cast<std::size_t>(3 + level)] = reg;
        }
        for (std::size_t k = 0; k < 6; ++k) {
            outputs[k]->setName(kHeadOutputNames[k]);
            network_->markOutput(*outputs[k]);
        }
        for (auto* s : statuses_) network_->markOutput(*s);
        for (auto* t : debug_) network_->markOutput(*t);
        std::unique_ptr<IHostMemory> plan(builder_->buildSerializedNetwork(*network_, *config_));
        require(plan != nullptr && plan->size() > 0, "buildSerializedNetwork failed");
        HeadEngine engine;
        const auto* bytes = static_cast<const std::uint8_t*>(plan->data());
        engine.plan.assign(bytes, bytes + plan->size());
        for (auto* s : statuses_) engine.statuses.emplace_back(s->getName());
        for (auto* t : debug_) engine.debug_outputs.emplace_back(t->getName());
        std::ostringstream o;
        o << "network=strongly_typed_fp32\ntf32=off\nfp16=off\nint8=off\nworkspace_bytes=" << kHeadBuilderWorkspace
          << "\noptimization_level=" << kHeadBuilderOptimizationLevel << "\naux_streams=0\ntiming_cache=none\nprofiling_verbosity=detailed"
          << "\ndebug_intermediates=" << (options_.debug_intermediates ? "true" : "false") << '\n';
        engine.build_options = o.str();
        return engine;
    }

private:
    const HeadTensor& W(const std::string& key) const { return w_.at(prefix_ + key); }
    template <typename L> L* named(L* layer, const std::string& what) {
        require(layer != nullptr, "TensorRT rejected " + prefix_ + what);
        layer->setName((prefix_ + what).c_str());
        return layer;
    }
    // Names the tensor after the reference intermediate and, for a debug
    // engine, marks it as an output.
    ITensor* keep(ITensor* t, const char* step) {
        t->setName((prefix_ + step).c_str());
        if (options_.debug_intermediates) debug_.push_back(t);
        return t;
    }

    ITensor* conv(ITensor* x, const std::string& key, int k, int stride, int pad) {
        const auto& wt = W(key + ".weight");
        auto* l = named(network_->addConvolutionNd(*x, static_cast<std::int64_t>(wt.shape[0]), DimsHW{k, k}, weights(wt), weights(W(key + ".bias"))), key);
        l->setStrideNd(DimsHW{stride, stride});
        l->setPaddingNd(DimsHW{pad, pad});
        return l->getOutput(0);
    }
    ITensor* constant(const HeadTensor& t, Dims d, const std::string& what) {
        return named(network_->addConstant(d, weights(t)), what)->getOutput(0);
    }
    // x (4, L, I) times W^T, W (O, I) as stored: the bundle keeps PyTorch's
    // (out, in) layout and the transpose is the matrix-multiply operation.
    ITensor* linear(ITensor* x, const std::string& key) {
        const auto& wt = W(key);
        auto* c = constant(wt, shape({1, wt.shape[0], wt.shape[1]}), key);
        return named(network_->addMatrixMultiply(*x, MatrixOperation::kNONE, *c, MatrixOperation::kTRANSPOSE), key + ".matmul")->getOutput(0);
    }
    ITensor* silu(ITensor* x, const std::string& what) {
        auto* s = named(network_->addActivation(*x, ActivationType::kSIGMOID), what + ".sigmoid")->getOutput(0);
        return named(network_->addElementWise(*x, *s, ElementWiseOperation::kPROD), what + ".silu")->getOutput(0);
    }
    ITensor* slice(ITensor* x, Dims start, Dims size, const std::string& what) {
        Dims stride = size;
        for (int i = 0; i < stride.nbDims; ++i) stride.d[i] = 1;
        return named(network_->addSlice(*x, start, size, stride), what)->getOutput(0);
    }
    ITensor* plugin(std::unique_ptr<IPluginV3> p, std::vector<ITensor*> in, const char* step) {
        auto* layer = named(network_->addPluginV3(in.data(), static_cast<std::int32_t>(in.size()), nullptr, 0, *p), step);
        require(layer->getNbOutputs() == 2, "plugin output count");
        plugins_.push_back(std::move(p));
        auto* status = layer->getOutput(1);
        status->setName((prefix_ + step + "_status").c_str());
        statuses_.push_back(status);
        return layer->getOutput(0);
    }

    std::pair<ITensor*, ITensor*> head_level(int level, ITensor* x) {
        prefix_ = std::string(kHeadLevelNames[static_cast<std::size_t>(level)]) + ".";
        const auto size = kHeadSize[static_cast<std::size_t>(level)], small = size / kHeadReduction, L = small * small;
        // 1-2: input projection and stride-4 downsample.
        auto* a = keep(conv(x, "input_proj", 1, 1, 0), "input_proj");
        auto* down = keep(conv(a, "downsample", static_cast<int>(kHeadReduction), static_cast<int>(kHeadReduction), 0), "downsample");
        // 3: Cross-Scan expand -> (4, L, 128).
        auto* tokens = keep(plugin(create_cross_scan_expand_plugin(), {down}, "expand"), "tokens");
        // 4a: in_proj, x = [:256], z = [256:].
        auto* xz = keep(linear(tokens, "mamba.in_proj.weight"), "in_proj");
        auto* xs = slice(xz, shape({0, 0, 0}), shape({4, L, kHeadDInner}), "mamba.x_half");
        auto* z = slice(xz, shape({0, 0, kHeadDInner}), shape({4, L, kHeadDInner}), "mamba.z_half");
        // 4b: depthwise causal Conv1d, k = 4, pad 3 on both sides, keep the first L, + bias.
        auto* to_conv = named(network_->addShuffle(*xs), "mamba.conv1d.to_nchw");
        to_conv->setFirstTranspose(permutation({0, 2, 1}));
        to_conv->setReshapeDimensions(shape({4, kHeadDInner, 1, L}));
        const auto& cw = W("mamba.conv1d.weight"); // (256, 1, 4) bytes = depthwise (256, 1, 1, 4) kernel
        auto* conv1d = named(network_->addConvolutionNd(*to_conv->getOutput(0), kHeadDInner, DimsHW{1, static_cast<std::int64_t>(kHeadConvKernel)},
                                                        weights(cw), weights(W("mamba.conv1d.bias"))),
                             "mamba.conv1d");
        conv1d->setNbGroups(kHeadDInner);
        conv1d->setPaddingNd(DimsHW{0, kHeadConvKernel - 1});
        auto* causal = slice(conv1d->getOutput(0), shape({0, 0, 0, 0}), shape({4, kHeadDInner, 1, L}), "mamba.conv1d.causal");
        auto* back = named(network_->addShuffle(*causal), "mamba.conv1d.to_nlc");
        back->setReshapeDimensions(shape({4, kHeadDInner, L}));
        back->setSecondTranspose(permutation({0, 2, 1}));
        auto* conv_out = keep(back->getOutput(0), "conv1d");
        auto* u = keep(silu(conv_out, "mamba.u"), "u");
        // 4c-4d: x_proj -> dt [:16], B [16:32], C [32:33]; delta = dt_proj(dt) + bias.
        auto* dbc = keep(linear(u, "mamba.x_proj.weight"), "x_proj");
        auto* dt = slice(dbc, shape({0, 0, 0}), shape({4, L, kHeadDState}), "mamba.dt");
        auto* bm = slice(dbc, shape({0, 0, kHeadDState}), shape({4, L, kHeadDState}), "mamba.B");
        auto* cm = slice(dbc, shape({0, 0, 2 * kHeadDState}), shape({4, L, kHeadCRank}), "mamba.C");
        auto* dt_bias = constant(W("mamba.dt_proj.bias"), shape({1, 1, kHeadDInner}), "mamba.dt_proj.bias");
        auto* delta = keep(named(network_->addElementWise(*linear(dt, "mamba.dt_proj.weight"), *dt_bias, ElementWiseOperation::kSUM), "mamba.delta")
                               ->getOutput(0),
                           "delta");
        // 4e: A = -exp(A_log), computed in the graph.
        auto* a_log = constant(W("mamba.A_log"), shape({1, kHeadDState}), "mamba.A_log");
        auto* a_exp = named(network_->addUnary(*a_log, UnaryOperation::kEXP), "mamba.A_exp")->getOutput(0);
        auto* am = named(network_->addUnary(*a_exp, UnaryOperation::kNEG), "mamba.A")->getOutput(0);
        auto* d = constant(W("mamba.D"), shape({kHeadDInner}), "mamba.D");
        // 4f: SelectiveScan v2 (u, delta, A, B, C, D).
        auto* y = keep(plugin(create_selective_scan_plugin(true), {u, delta, am, bm, cm, d}, "scan"), "scan");
        // 4g: gate, out_proj, residual with the tokens.
        auto* gated = keep(named(network_->addElementWise(*y, *silu(z, "mamba.z"), ElementWiseOperation::kPROD), "mamba.gate")->getOutput(0), "gate");
        auto* o = keep(linear(gated, "mamba.out_proj.weight"), "out_proj");
        auto* r = keep(named(network_->addElementWise(*o, *tokens, ElementWiseOperation::kSUM), "mamba.residual")->getOutput(0), "residual");
        // 5: Cross-Scan merge -> (1, 128, Hs, Ws).
        auto* merged = keep(plugin(create_cross_scan_merge_plugin(static_cast<std::int32_t>(small), static_cast<std::int32_t>(small)), {r}, "merge"),
                            "merge");
        // 6: upsample conv 3x3, PixelShuffle 4 (CRD: channel c*16 + i*4 + j -> (c, h*4+i, w*4+j)).
        auto* up = keep(conv(merged, "upsample", 3, 1, 1), "upsample");
        auto* split = named(network_->addShuffle(*up), "pixel_shuffle.split");
        split->setReshapeDimensions(shape({1, kHeadDModel, kHeadReduction, kHeadReduction, small, small}));
        split->setSecondTranspose(permutation({0, 1, 4, 2, 5, 3}));
        auto* join = named(network_->addShuffle(*split->getOutput(0)), "pixel_shuffle.join");
        join->setReshapeDimensions(shape({1, kHeadDModel, size, size}));
        auto* shuffled = keep(join->getOutput(0), "pixel_shuffle");
        // 7: concat [input_proj, upsampled], then the cls and reg branches.
        ITensor* parts[] = {a, shuffled};
        auto* cat_layer = named(network_->addConcatenation(parts, 2), "concat");
        cat_layer->setAxis(1);
        auto* cat = cat_layer->getOutput(0);
        auto* cls_hidden = keep(silu(conv(cat, "cls.conv0", 3, 1, 1), "cls.hidden"), "cls_hidden");
        auto* cls = conv(cls_hidden, "cls.conv1", 1, 1, 0);
        auto* reg_hidden = keep(silu(conv(cat, "reg.conv0", 3, 1, 1), "reg.hidden"), "reg_hidden");
        auto* reg = conv(reg_hidden, "reg.conv1", 1, 1, 0);
        return {cls, reg};
    }

    const HeadWeights& w_;
    HeadBuildOptions options_;
    std::unique_ptr<IBuilder> builder_;
    std::unique_ptr<INetworkDefinition> network_;
    std::unique_ptr<IBuilderConfig> config_;
    std::vector<std::unique_ptr<IPluginV3>> plugins_; // kept alive until the build ends
    std::vector<ITensor*> statuses_, debug_;
    std::string prefix_;
};

} // namespace

FrameIoSpec head_io_spec() {
    FrameIoSpec spec;
    for (int level = 0; level < kHeadLevels; ++level)
        spec.inputs.push_back({kHeadInputNames[static_cast<std::size_t>(level)], head_input_shape(level)});
    for (int k = 0; k < 6; ++k) spec.outputs.push_back({kHeadOutputNames[static_cast<std::size_t>(k)], head_output_shape(k)});
    for (int level = 0; level < kHeadLevels; ++level) {
        const std::string p = kHeadLevelNames[static_cast<std::size_t>(level)];
        const auto small = std::to_string(kHeadSize[static_cast<std::size_t>(level)] / kHeadReduction);
        spec.statuses.push_back({p + ".expand_status", status_producer(kCrossScanExpandPluginName, p, "v1")});
        spec.statuses.push_back({p + ".scan_status", status_producer(kSelectiveScanPluginName, p, "v2, has_d=1")});
        spec.statuses.push_back({p + ".merge_status", status_producer(kCrossScanMergePluginName, p, "v1, " + small + "x" + small)});
    }
    spec.check_inputs = spec.check_outputs = true;
    return spec;
}

HeadEngine build_head_engine(ILogger& logger, const HeadWeights& weights, const HeadBuildOptions& options) {
    validate_head_weights(weights);
    auto engine = HeadNetwork(logger, weights, options).build();
    const auto spec = head_io_spec();
    require(engine.statuses.size() == spec.statuses.size(), "the network created " + std::to_string(engine.statuses.size()) + " statuses");
    for (std::size_t i = 0; i < spec.statuses.size(); ++i)
        require(engine.statuses[i] == spec.statuses[i].name, "status " + std::to_string(i) + " is " + engine.statuses[i] + ", expected " + spec.statuses[i].name);
    // The built kernels must stay inside the policy's operation model.
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    require(runtime != nullptr, "createInferRuntime failed");
    std::unique_ptr<ICudaEngine> built(runtime->deserializeCudaEngine(engine.plan.data(), engine.plan.size()));
    require(built != nullptr, "deserializeCudaEngine of the built plan failed");
    const auto violations = head_tactic_violations(head_engine_layers(*built));
    std::string listed;
    for (const auto& v : violations) listed += (listed.empty() ? "" : ", ") + v;
    require(violations.empty(), "tactics outside the FP32 direct-summation model: " + listed);
    return engine;
}

std::vector<std::string> head_tactic_violations(const std::string& layers_json) {
    const std::string key = "\"TacticName\"";
    std::vector<std::string> out;
    for (auto at = layers_json.find(key); at != std::string::npos; at = layers_json.find(key, at + key.size())) {
        const auto open = layers_json.find('"', layers_json.find(':', at + key.size()));
        const auto close = open == std::string::npos ? open : layers_json.find('"', open + 1);
        if (close == std::string::npos) break;
        const auto name = layers_json.substr(open + 1, close - open - 1);
        // Check the name's '_'-separated tokens; "0x..." tokens are generated
        // kernel hashes, whose hex digits carry no meaning.
        bool bad = false;
        std::size_t begin = 0;
        while (!bad && begin <= name.size()) {
            const auto end = std::min(name.find('_', begin), name.size());
            std::string token = name.substr(begin, end - begin);
            for (auto& c : token) c = static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
            if (!token.starts_with("0x"))
                for (const char* marker : {"wngd", "winograd", "fft", "f16", "tf32", "hmma", "i8", "int8"})
                    bad = bad || token.find(marker) != std::string::npos;
            begin = end + 1;
        }
        if (bad) out.push_back(name);
    }
    return out;
}

std::string head_engine_layers(const ICudaEngine& engine) {
    std::unique_ptr<IEngineInspector> inspector(engine.createEngineInspector());
    if (!inspector) throw std::runtime_error("createEngineInspector failed");
    const char* info = inspector->getEngineInformation(LayerInformationFormat::kJSON);
    return info ? std::string(info) : std::string();
}

} // namespace saccade

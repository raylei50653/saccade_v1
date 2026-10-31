#include "saccade/scan_trt_plugin.hpp"
#include "saccade/selective_scan_cuda.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace saccade {
namespace {
using namespace nvinfer1;

static_assert(sizeof(ScanStatus) == kScanStatusWords * sizeof(std::int32_t), "status output is the ScanStatus words");

enum class Op { scan, expand, merge };

struct Identity { const char* name; const char* version; };
constexpr Identity identity_of(Op op) {
    return op == Op::scan ? Identity{kSelectiveScanPluginName, kSelectiveScanPluginVersion}
         : op == Op::expand ? Identity{kCrossScanExpandPluginName, kCrossScanExpandPluginVersion}
                            : Identity{kCrossScanMergePluginName, kCrossScanMergePluginVersion};
}

// TensorRT learns of a failure from the return code; the message goes to the
// execution context's error recorder, or to TensorRT's logger when there is none.
void report(IErrorRecorder* recorder, ErrorCode code, const std::string& message) noexcept {
    try {
        if (recorder) recorder->reportError(code, message.c_str());
        else if (auto* logger = getLogger()) logger->log(ILogger::Severity::kERROR, message.c_str());
    } catch (...) {
    }
}

std::string text(const Dims& d) {
    std::ostringstream out;
    out << '(';
    for (int i = 0; i < d.nbDims; ++i) out << (i ? "," : "") << d.d[i];
    out << ')';
    return out.str();
}

bool equal(const Dims& d, std::initializer_list<std::int64_t> want) {
    if (d.nbDims != static_cast<int>(want.size())) return false;
    int i = 0;
    for (const auto w : want)
        if (d.d[i++] != w) return false;
    return true;
}

bool known_differ(const IDimensionExpr* a, const IDimensionExpr* b) {
    return a->isConstant() && b->isConstant() && a->getConstantValue() != b->getConstantValue();
}

// Concrete shapes of one invocation, derived from TensorRT's descriptors.
struct Call {
    SelectiveScanShape scan;
    CrossScanShape cross;
};

class ScanPlugin final : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild, public IPluginV3OneRuntime {
public:
    ScanPlugin(Op op, std::int32_t has_d, std::int32_t height, std::int32_t width)
        : op_(op), has_d_(has_d), height_(height), width_(width) { describe_fields(); }
    // Clones keep the semantic attributes and the context binding; the binding
    // only lets enqueue run on a stream of that same CUDA context.
    ScanPlugin(const ScanPlugin& other)
        : op_(other.op_), has_d_(other.has_d_), height_(other.height_), width_(other.width_),
          bound_context_(other.bound_context_), recorder_(other.recorder_) { describe_fields(); }
    ScanPlugin& operator=(const ScanPlugin&) = delete;

    // ------------------------------------------------------------ IPluginV3
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override {
        try { return new ScanPlugin(*this); }
        catch (...) { return nullptr; }
    }

    // ------------------------------------------------------- IPluginV3OneCore
    const char* getPluginName() const noexcept override { return identity_of(op_).name; }
    const char* getPluginVersion() const noexcept override { return identity_of(op_).version; }
    const char* getPluginNamespace() const noexcept override { return kScanTrtNamespace; }

    // ------------------------------------------------------ IPluginV3OneBuild
    std::int32_t getNbOutputs() const noexcept override { return 2; }

    std::int32_t getOutputDataTypes(DataType* out, std::int32_t nb_out, const DataType* in, std::int32_t nb_in) const noexcept override {
        if (nb_in != inputs() || nb_out != 2) return fail(ErrorCode::kINVALID_ARGUMENT, "input or output count");
        for (std::int32_t i = 0; i < nb_in; ++i)
            if (in[i] != DataType::kFLOAT) return fail(ErrorCode::kINVALID_ARGUMENT, "input " + std::to_string(i) + " is not FP32");
        out[0] = DataType::kFLOAT;
        out[1] = DataType::kINT32;
        return 0;
    }

    std::int32_t getOutputShapes(const DimsExprs* in, std::int32_t nb_in, const DimsExprs*, std::int32_t nb_shape_in, DimsExprs* out,
                                 std::int32_t nb_out, IExprBuilder& expr) noexcept override {
        if (nb_in != inputs() || nb_out != 2 || nb_shape_in != 0) return fail(ErrorCode::kINVALID_ARGUMENT, "input or output count");
        out[1].nbDims = 1;
        out[1].d[0] = expr.constant(kScanStatusWords);
        switch (op_) {
        case Op::scan: {
            const int ranks[6] = {3, 3, 2, 3, 3, 1};
            for (std::int32_t i = 0; i < nb_in; ++i)
                if (in[i].nbDims != ranks[i]) return fail(ErrorCode::kINVALID_ARGUMENT, "input " + std::to_string(i) + " rank");
            // u, delta (B,L,Dc); A (rows,N); B (B,L,N); C (B,L,cw); D (Dc).
            const auto &u = in[0], &delta = in[1], &a = in[2], &b = in[3], &c = in[4];
            bool differ = known_differ(b.d[2], a.d[1]);
            for (int k = 0; k < 3; ++k) differ = differ || known_differ(u.d[k], delta.d[k]);
            for (int k = 0; k < 2; ++k) differ = differ || known_differ(u.d[k], b.d[k]) || known_differ(u.d[k], c.d[k]);
            if (has_d_) differ = differ || known_differ(in[5].d[0], u.d[2]);
            if (differ) return fail(ErrorCode::kINVALID_ARGUMENT, "input dimensions disagree");
            // Constant dimensions that can never be valid fail the build already.
            const auto value = [](const IDimensionExpr* e) { return e->isConstant() ? e->getConstantValue() : -1; };
            const auto n = value(a.d[1]), rows = value(a.d[0]), width = value(c.d[2]), channels = value(u.d[2]);
            if (n >= 0 && (n < 1 || n > 32 || (n & (n - 1)) != 0)) return fail(ErrorCode::kINVALID_ARGUMENT, "state must be a power of two in [1, 32]");
            if (rows > 1 && channels >= 0 && rows != channels) return fail(ErrorCode::kINVALID_ARGUMENT, "A rows must be 1 or channels");
            if (width > 1 && n >= 0 && width != n) return fail(ErrorCode::kINVALID_ARGUMENT, "C width must be 1 or state");
            out[0] = u;
            return 0;
        }
        case Op::expand: {
            const auto& x = in[0];
            if (x.nbDims != 4) return fail(ErrorCode::kINVALID_ARGUMENT, "x must be (batch, channels, height, width)");
            out[0].nbDims = 3;
            out[0].d[0] = expr.operation(DimensionOperation::kPROD, *expr.constant(4), *x.d[0]);
            out[0].d[1] = expr.operation(DimensionOperation::kPROD, *x.d[2], *x.d[3]);
            out[0].d[2] = x.d[1];
            return 0;
        }
        case Op::merge: {
            const auto& y = in[0];
            if (y.nbDims != 3) return fail(ErrorCode::kINVALID_ARGUMENT, "y must be (4 * batch, height * width, channels)");
            if (y.d[1]->isConstant() && y.d[1]->getConstantValue() != std::int64_t{height_} * width_)
                return fail(ErrorCode::kINVALID_ARGUMENT, "y length differs from height * width");
            out[0].nbDims = 4;
            out[0].d[0] = expr.operation(DimensionOperation::kFLOOR_DIV, *y.d[0], *expr.constant(4));
            out[0].d[1] = y.d[2];
            out[0].d[2] = expr.constant(height_);
            out[0].d[3] = expr.constant(width_);
            return 0;
        }
        }
        return -1;
    }

    // Dense row-major FP32 everywhere except the INT32 status output.
    bool supportsFormatCombination(std::int32_t pos, const DynamicPluginTensorDesc* io, std::int32_t nb_in, std::int32_t nb_out) noexcept override {
        if (nb_in != inputs() || nb_out != 2 || pos < 0 || pos >= nb_in + nb_out) return false;
        const auto& d = io[pos].desc;
        return d.format == TensorFormat::kLINEAR && d.type == (pos == nb_in + 1 ? DataType::kINT32 : DataType::kFLOAT);
    }

    // Build phase only (never on the frame path): check the formats and load the
    // kernels in the builder's CUDA context, in case TensorRT times the plugin.
    std::int32_t configurePlugin(const DynamicPluginTensorDesc* in, std::int32_t nb_in, const DynamicPluginTensorDesc* out,
                                 std::int32_t nb_out) noexcept override {
        if (nb_in != inputs() || nb_out != 2) return fail(ErrorCode::kINVALID_CONFIG, "input or output count");
        for (std::int32_t i = 0; i < nb_in + nb_out; ++i) {
            const auto& d = i < nb_in ? in[i].desc : out[i - nb_in].desc;
            if (d.format != TensorFormat::kLINEAR || d.type != (i == nb_in + 1 ? DataType::kINT32 : DataType::kFLOAT))
                return fail(ErrorCode::kINVALID_CONFIG, "unsupported type or format for tensor " + std::to_string(i));
        }
        return bind();
    }

    // The recurrence state lives in registers and the status in the status
    // output, so TensorRT's workspace is not needed: 0 bytes, no alignment.
    std::size_t getWorkspaceSize(const DynamicPluginTensorDesc*, std::int32_t, const DynamicPluginTensorDesc*, std::int32_t) const noexcept override {
        return 0;
    }

    // ---------------------------------------------------- IPluginV3OneRuntime
    // Host-only shape check; TensorRT calls it inside enqueueV3 after a shape
    // or address change, so it makes no CUDA call.
    std::int32_t onShapeChange(const PluginTensorDesc* in, std::int32_t nb_in, const PluginTensorDesc* out, std::int32_t nb_out) noexcept override {
        if (nb_in != inputs() || nb_out != 2) return fail(ErrorCode::kINVALID_ARGUMENT, "input or output count");
        Call call;
        const auto why = check(in, out, call);
        return why.empty() ? 0 : fail(ErrorCode::kINVALID_ARGUMENT, why);
    }

    // Enqueues the S02 call on TensorRT's stream with TensorRT's buffers. The S02
    // host checks run first and throw before any GPU work, so a rejected call
    // leaves the outputs, including the status, untouched.
    std::int32_t enqueue(const PluginTensorDesc* in_desc, const PluginTensorDesc* out_desc, const void* const* in, void* const* out,
                         void*, cudaStream_t stream) noexcept override {
        try {
            require_bound(stream);
            Call call;
            const auto why = check(in_desc, out_desc, call);
            if (!why.empty()) throw std::invalid_argument(why);
            auto* status = static_cast<ScanStatus*>(out[1]);
            switch (op_) {
            case Op::scan: {
                const auto* f = reinterpret_cast<const float* const*>(in);
                const SelectiveScanDeviceInputs inputs{f[0], f[1], f[2], f[3], f[4], has_d_ ? f[5] : nullptr};
                selective_scan_cuda(call.scan, inputs, static_cast<float*>(out[0]), status, stream);
                break;
            }
            case Op::expand:
                cross_scan_expand_cuda(call.cross, static_cast<const float*>(in[0]), static_cast<float*>(out[0]), status, stream);
                break;
            case Op::merge:
                cross_scan_merge_cuda(call.cross, static_cast<const float*>(in[0]), static_cast<float*>(out[0]), status, stream);
                break;
            }
            return 0;
        } catch (const std::invalid_argument& error) {
            return fail(ErrorCode::kINVALID_ARGUMENT, error.what());
        } catch (const std::exception& error) {
            return fail(ErrorCode::kFAILED_EXECUTION, error.what());
        }
    }

    // Initialization stage: every execution context gets its own clone, which
    // loads the kernels in the CUDA context current at context creation.
    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override {
        try {
            auto attached = std::make_unique<ScanPlugin>(*this);
            attached->recorder_ = context ? context->getErrorRecorder() : nullptr;
            if (attached->bind() != 0) return nullptr;
            return attached.release();
        } catch (...) {
            return nullptr;
        }
    }

    const PluginFieldCollection* getFieldsToSerialize() noexcept override { return &collection_; }

private:
    std::int32_t inputs() const { return op_ == Op::scan ? 5 + has_d_ : 1; }

    std::int32_t fail(ErrorCode code, const std::string& why) const noexcept {
        report(recorder_, code, std::string(kScanTrtNamespace) + "::" + identity_of(op_).name + ": " + why);
        return -1;
    }

    void describe_fields() {
        fields_.clear();
        if (op_ == Op::scan) fields_.emplace_back("has_d", &has_d_, PluginFieldType::kINT32, 1);
        if (op_ == Op::merge) {
            fields_.emplace_back("height", &height_, PluginFieldType::kINT32, 1);
            fields_.emplace_back("width", &width_, PluginFieldType::kINT32, 1);
        }
        collection_.nbFields = static_cast<std::int32_t>(fields_.size());
        collection_.fields = fields_.data();
    }

    // Kernel readiness belongs to a CUDA context: load every S02 kernel in the
    // current context and remember that context's process-unique ID.
    std::int32_t bind() noexcept {
        try {
            (void)selective_scan_cuda_load_kernels();
            CUcontext context{};
            unsigned long long id = 0;
            if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr || cuCtxGetId(context, &id) != CUDA_SUCCESS)
                throw std::runtime_error("no current CUDA context to load the kernels in");
            bound_context_ = id;
            return 0;
        } catch (const std::exception& error) {
            return fail(ErrorCode::kFAILED_INITIALIZATION, error.what());
        }
    }

    // Host-only: enqueue never loads kernels (lazy loading would wait for work
    // already on the device), so the stream must belong to the bound context.
    void require_bound(cudaStream_t stream) const {
        if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
            throw std::invalid_argument("an explicit non-default stream is required");
        if (bound_context_ == 0) throw std::invalid_argument("kernels are not loaded for any CUDA context");
        CUcontext of_stream{}, current{};
        unsigned long long id = 0;
        if (cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &of_stream) != CUDA_SUCCESS || cuCtxGetCurrent(&current) != CUDA_SUCCESS ||
            of_stream == nullptr || of_stream != current || cuCtxGetId(of_stream, &id) != CUDA_SUCCESS)
            throw std::invalid_argument("the stream does not belong to the current CUDA context");
        if (id != bound_context_) throw std::invalid_argument("the kernels were loaded in another CUDA context");
    }

    // Returns why the descriptors break the plugin contract, or "" with `call`
    // filled. Shape rules the S02 API checks itself (state, broadcast widths,
    // capacity) are left to it; this checks what it cannot see: types, formats,
    // ranks and agreement between the tensors' dimensions.
    std::string check(const PluginTensorDesc* in, const PluginTensorDesc* out, Call& call) const {
        for (std::int32_t i = 0; i < inputs(); ++i)
            if (in[i].type != DataType::kFLOAT || in[i].format != TensorFormat::kLINEAR) return "input " + std::to_string(i) + " is not linear FP32";
        if (out[0].type != DataType::kFLOAT || out[0].format != TensorFormat::kLINEAR) return "output 0 is not linear FP32";
        if (out[1].type != DataType::kINT32 || out[1].format != TensorFormat::kLINEAR || !equal(out[1].dims, {kScanStatusWords}))
            return "status output is not linear INT32[2]";
        for (std::int32_t i = 0; i < inputs(); ++i)
            for (int k = 0; k < in[i].dims.nbDims; ++k)
                if (in[i].dims.d[k] < 0) return "input " + std::to_string(i) + " has an unresolved dimension";
        const auto& d0 = in[0].dims;
        switch (op_) {
        case Op::scan: {
            if (d0.nbDims != 3 || in[2].dims.nbDims != 2) return "u must be (batch, length, channels) and A (rows, state)";
            const std::int64_t batch = d0.d[0], length = d0.d[1], channels = d0.d[2];
            const std::int64_t rows = in[2].dims.d[0], state = in[2].dims.d[1];
            const auto& c = in[4].dims;
            if (!equal(in[1].dims, {batch, length, channels})) return "delta " + text(in[1].dims) + " differs from u " + text(d0);
            if (!equal(in[3].dims, {batch, length, state})) return "B " + text(in[3].dims) + " is not (batch, length, state)";
            if (c.nbDims != 3 || c.d[0] != batch || c.d[1] != length) return "C " + text(c) + " is not (batch, length, width)";
            if (has_d_ && !equal(in[5].dims, {channels})) return "D " + text(in[5].dims) + " is not (channels)";
            if (!equal(out[0].dims, {batch, length, channels})) return "y " + text(out[0].dims) + " differs from u";
            call.scan = {static_cast<std::size_t>(batch), static_cast<std::size_t>(length), static_cast<std::size_t>(channels),
                         static_cast<std::size_t>(state), static_cast<std::size_t>(rows), static_cast<std::size_t>(c.d[2]), has_d_ != 0};
            return "";
        }
        case Op::expand: {
            if (d0.nbDims != 4) return "x " + text(d0) + " is not (batch, channels, height, width)";
            if (!equal(out[0].dims, {4 * d0.d[0], d0.d[2] * d0.d[3], d0.d[1]})) return "tokens " + text(out[0].dims) + " do not match x " + text(d0);
            call.cross = {static_cast<std::size_t>(d0.d[0]), static_cast<std::size_t>(d0.d[1]), static_cast<std::size_t>(d0.d[2]),
                          static_cast<std::size_t>(d0.d[3])};
            return "";
        }
        case Op::merge: {
            if (d0.nbDims != 3 || d0.d[0] % 4 != 0 || d0.d[1] != std::int64_t{height_} * width_)
                return "y " + text(d0) + " is not (4 * batch, " + std::to_string(height_) + " * " + std::to_string(width_) + ", channels)";
            if (!equal(out[0].dims, {d0.d[0] / 4, d0.d[2], height_, width_})) return "out " + text(out[0].dims) + " does not match y " + text(d0);
            call.cross = {static_cast<std::size_t>(d0.d[0] / 4), static_cast<std::size_t>(d0.d[2]), static_cast<std::size_t>(height_),
                          static_cast<std::size_t>(width_)};
            return "";
        }
        }
        return "unknown plugin";
    }

    Op op_;
    std::int32_t has_d_, height_, width_;
    unsigned long long bound_context_ = 0; // 0: kernels not loaded for any context
    IErrorRecorder* recorder_ = nullptr;
    std::vector<PluginField> fields_;
    PluginFieldCollection collection_{};
};

class ScanCreator final : public IPluginCreatorV3One {
public:
    explicit ScanCreator(Op op) : op_(op) {
        if (op == Op::scan) names_.emplace_back("has_d", nullptr, PluginFieldType::kINT32, 1);
        if (op == Op::merge) {
            names_.emplace_back("height", nullptr, PluginFieldType::kINT32, 1);
            names_.emplace_back("width", nullptr, PluginFieldType::kINT32, 1);
        }
        collection_.nbFields = static_cast<std::int32_t>(names_.size());
        collection_.fields = names_.data();
    }

    // Accepts exactly the advertised fields, each one INT32 value: the same
    // parsing for addPluginV3 and for deserialization.
    IPluginV3* createPlugin(const char*, const PluginFieldCollection* fc, TensorRTPhase) noexcept override {
        try {
            std::array<std::int32_t, 2> value{};
            std::array<bool, 2> seen{};
            const std::int32_t n = fc ? fc->nbFields : 0;
            if (n < 0) throw std::invalid_argument("negative field count");
            if (n > 0 && fc->fields == nullptr) throw std::invalid_argument("null field list");
            for (std::int32_t i = 0; i < n; ++i) {
                const auto& f = fc->fields[i];
                std::size_t k = 0;
                while (k < names_.size() && (f.name == nullptr || std::strcmp(f.name, names_[k].name) != 0)) ++k;
                if (k == names_.size()) throw std::invalid_argument(std::string("unknown field ") + (f.name ? f.name : "(null)"));
                if (seen[k]) throw std::invalid_argument(std::string("duplicate field ") + f.name);
                if (f.type != PluginFieldType::kINT32 || f.length != 1 || f.data == nullptr)
                    throw std::invalid_argument(std::string("field ") + f.name + " must hold one INT32");
                std::memcpy(&value[k], f.data, sizeof(std::int32_t));
                seen[k] = true;
            }
            for (std::size_t k = 0; k < names_.size(); ++k)
                if (!seen[k]) throw std::invalid_argument(std::string("missing field ") + names_[k].name);
            if (op_ == Op::scan && value[0] != 0 && value[0] != 1) throw std::invalid_argument("has_d must be 0 or 1");
            if (op_ == Op::merge && (value[0] < 1 || value[1] < 1)) throw std::invalid_argument("height and width must be positive");
            return new ScanPlugin(op_, op_ == Op::scan ? value[0] : 1, op_ == Op::merge ? value[0] : 0, op_ == Op::merge ? value[1] : 0);
        } catch (const std::exception& error) {
            report(nullptr, ErrorCode::kINVALID_ARGUMENT,
                   std::string(kScanTrtNamespace) + "::" + identity_of(op_).name + " creator: " + error.what());
            return nullptr;
        }
    }
    const PluginFieldCollection* getFieldNames() noexcept override { return &collection_; }
    const char* getPluginName() const noexcept override { return identity_of(op_).name; }
    const char* getPluginVersion() const noexcept override { return identity_of(op_).version; }
    const char* getPluginNamespace() const noexcept override { return kScanTrtNamespace; }

private:
    Op op_;
    std::vector<PluginField> names_;
    PluginFieldCollection collection_{};
};

ScanCreator& creator(Op op) {
    static ScanCreator scan(Op::scan), expand(Op::expand), merge(Op::merge);
    return op == Op::scan ? scan : op == Op::expand ? expand : merge;
}

std::unique_ptr<IPluginV3> create(Op op, std::vector<PluginField> fields) {
    const auto id = identity_of(op);
    auto* found = getPluginRegistry()->getCreator(id.name, id.version, kScanTrtNamespace);
    if (found == nullptr || std::string_view(found->getInterfaceInfo().kind) != "PLUGIN CREATOR_V3ONE")
        throw std::runtime_error(std::string("plugin creator not registered: ") + kScanTrtNamespace + "::" + id.name + " v" + id.version);
    const PluginFieldCollection fc{static_cast<std::int32_t>(fields.size()), fields.data()};
    std::unique_ptr<IPluginV3> plugin(static_cast<IPluginCreatorV3One*>(found)->createPlugin(id.name, &fc, TensorRTPhase::kBUILD));
    if (!plugin) throw std::runtime_error(std::string("plugin creator rejected the fields: ") + id.name);
    return plugin;
}

} // namespace

void register_scan_trt_plugins(IPluginRegistry& registry) {
    for (const Op op : {Op::scan, Op::expand, Op::merge}) {
        const auto id = identity_of(op);
        if (registry.getCreator(id.name, id.version, kScanTrtNamespace) != nullptr) continue;
        if (!registry.registerCreator(creator(op), kScanTrtNamespace))
            throw std::runtime_error(std::string("TensorRT refused plugin creator ") + kScanTrtNamespace + "::" + id.name + " v" + id.version);
    }
}

std::unique_ptr<IPluginV3> create_selective_scan_plugin(bool has_d) {
    const std::int32_t value = has_d ? 1 : 0;
    return create(Op::scan, {PluginField("has_d", &value, PluginFieldType::kINT32, 1)});
}

std::unique_ptr<IPluginV3> create_cross_scan_expand_plugin() { return create(Op::expand, {}); }

std::unique_ptr<IPluginV3> create_cross_scan_merge_plugin(std::int32_t height, std::int32_t width) {
    return create(Op::merge, {PluginField("height", &height, PluginFieldType::kINT32, 1),
                              PluginField("width", &width, PluginFieldType::kINT32, 1)});
}

std::string scan_trt_plugin_identity() {
    std::ostringstream id;
    id << "tensorrt_headers=" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.' << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD
       << "\ntensorrt_library=" << getInferLibVersion() << '.' << getInferLibBuildVersion()
       << "\nplugin_api=IPluginV3+IPluginV3OneCore/Build/Runtime,IPluginCreatorV3One\nplugin_namespace=" << kScanTrtNamespace
       << "\nplugins=" << kSelectiveScanPluginName << ":v" << kSelectiveScanPluginVersion << ',' << kCrossScanExpandPluginName << ":v"
       << kCrossScanExpandPluginVersion << ',' << kCrossScanMergePluginName << ":v" << kCrossScanMergePluginVersion << '\n'
       << selective_scan_cuda_identity();
    return id.str();
}

} // namespace saccade

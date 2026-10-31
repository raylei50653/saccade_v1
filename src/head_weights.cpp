#include "saccade/head_weights.hpp"
#include "saccade/sha256.hpp"
#include "saccade/torch_checkpoint.hpp"

#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace saccade {
namespace {

static_assert(std::endian::native == std::endian::little, "the bundle stores little-endian FP32 and is read by memcpy");

std::string dims(const std::vector<std::int64_t>& shape, char separator = ',') {
    std::string out;
    for (std::size_t i = 0; i < shape.size(); ++i) out += (i ? std::string(1, separator) : "") + std::to_string(shape[i]);
    return out;
}

std::size_t elements(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (const auto d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

std::string_view float_bytes(const std::vector<float>& v) {
    return {reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float)};
}

std::vector<HeadParam> make_schema() {
    std::vector<HeadParam> s;
    for (int i = 0; i < kHeadLevels; ++i) {
        const std::string p = kHeadLevelNames[static_cast<std::size_t>(i)], n = std::to_string(i);
        const auto cin = kHeadInChannels[static_cast<std::size_t>(i)];
        const auto add = [&](std::string name, std::string layout, std::string key, std::vector<std::int64_t> shape) {
            s.push_back({p + "." + name, std::move(layout), std::move(key), std::move(shape)});
        };
        add("input_proj.weight", "OIHW", "input_proj." + n + ".weight", {kHeadDModel, cin, 1, 1});
        add("input_proj.bias", "O", "input_proj." + n + ".bias", {kHeadDModel});
        add("downsample.weight", "OIHW", "downsample." + n + ".weight", {kHeadDModel, kHeadDModel, kHeadReduction, kHeadReduction});
        add("downsample.bias", "O", "downsample." + n + ".bias", {kHeadDModel});
        const auto m = "mamba_blocks." + n + ".0.";
        add("mamba.in_proj.weight", "OI", m + "in_proj.weight", {2 * kHeadDInner, kHeadDModel});
        add("mamba.conv1d.weight", "OIK", m + "conv1d.weight", {kHeadDInner, 1, kHeadConvKernel});
        add("mamba.conv1d.bias", "O", m + "conv1d.bias", {kHeadDInner});
        add("mamba.x_proj.weight", "OI", m + "x_proj.weight", {2 * kHeadDState + kHeadCRank, kHeadDInner});
        add("mamba.dt_proj.weight", "OI", m + "dt_proj.weight", {kHeadDInner, kHeadDState});
        add("mamba.dt_proj.bias", "O", m + "dt_proj.bias", {kHeadDInner});
        add("mamba.A_log", "RN", m + "A_log", {1, kHeadDState});
        add("mamba.D", "C", m + "D", {kHeadDInner});
        add("mamba.out_proj.weight", "OI", m + "out_proj.weight", {kHeadDModel, kHeadDInner});
        add("upsample.weight", "OIHW", "upsample." + n + ".0.weight", {kHeadDModel * kHeadReduction * kHeadReduction, kHeadDModel, 3, 3});
        add("upsample.bias", "O", "upsample." + n + ".0.bias", {kHeadDModel * kHeadReduction * kHeadReduction});
        add("cls.conv0.weight", "OIHW", "cls_head." + n + ".0.weight", {kHeadDModel, 2 * kHeadDModel, 3, 3});
        add("cls.conv0.bias", "O", "cls_head." + n + ".0.bias", {kHeadDModel});
        add("cls.conv1.weight", "OIHW", "cls_head." + n + ".2.weight", {kHeadClasses, kHeadDModel, 1, 1});
        add("cls.conv1.bias", "O", "cls_head." + n + ".2.bias", {kHeadClasses});
        add("reg.conv0.weight", "OIHW", "reg_head." + n + ".0.weight", {kHeadDModel, 2 * kHeadDModel, 3, 3});
        add("reg.conv0.bias", "O", "reg_head." + n + ".0.bias", {kHeadDModel});
        add("reg.conv1.weight", "OIHW", "reg_head." + n + ".2.weight", {kHeadRegs, kHeadDModel, 1, 1});
        add("reg.conv1.bias", "O", "reg_head." + n + ".2.bias", {kHeadRegs});
    }
    return s;
}

std::vector<std::pair<std::string, std::string>> make_excluded() {
    std::vector<std::pair<std::string, std::string>> x;
    const char* temporal = "temporal Mamba block: the head runs it only when T_frames > 1; the T=1 forward bypasses it";
    const char* gate = "flow gate of the temporal path: needs T_frames > 1 and optical flow; not on the T=1 forward";
    for (int i = 0; i < kHeadLevels; ++i) {
        const auto n = std::to_string(i);
        for (const char* k : {"A_log", "D", "in_proj.weight", "conv1d.weight", "conv1d.bias", "x_proj.weight", "dt_proj.weight", "dt_proj.bias",
                              "out_proj.weight"})
            x.emplace_back("temporal_blocks." + n + ".0." + k, temporal);
        for (const char* k : {"0.weight", "0.bias", "2.weight", "2.bias"}) x.emplace_back("flow_gate_conv." + n + "." + k, gate);
    }
    return x;
}

[[noreturn]] void reject(const std::string& why) { throw std::invalid_argument("head checkpoint: " + why); }

const PickleValue& need(const PickleValue& dict, const char* key, const char* where) {
    const auto* v = dict.find(key);
    if (!v) reject(std::string(where) + " lacks " + key);
    return *v;
}

std::int64_t integer(const PickleValue& v, const std::string& where) {
    if (v.kind != PickleValue::Kind::integer) reject(where + " is not an integer");
    return v.integer;
}
bool flag(const PickleValue& v, const std::string& where) {
    if (v.kind != PickleValue::Kind::boolean) reject(where + " is not a boolean");
    return v.boolean;
}

// The forward-relevant head configuration; anything else in mamba_args is
// training metadata and is copied into the mapping report unchecked.
void check_config(const PickleValue& args, std::ostringstream& report) {
    const auto want_int = [&](const char* key, std::int64_t value) {
        const auto got = integer(need(args, key, "mamba_args"), std::string("mamba_args.") + key);
        if (got != value) reject(std::string("mamba_args.") + key + " = " + std::to_string(got) + ", expected " + std::to_string(value));
        report << "config\t" << key << '\t' << got << "\tchecked\n";
    };
    const auto want_flag = [&](const char* key, bool value) {
        const auto got = flag(need(args, key, "mamba_args"), std::string("mamba_args.") + key);
        if (got != value) reject(std::string("mamba_args.") + key + " = " + (got ? "true" : "false") + ", expected " + (value ? "true" : "false"));
        report << "config\t" << key << '\t' << (got ? "true" : "false") << "\tchecked\n";
    };
    const auto& channels = need(args, "in_channels", "mamba_args");
    if ((channels.kind != PickleValue::Kind::list && channels.kind != PickleValue::Kind::tuple) || channels.items.size() != 3)
        reject("mamba_args.in_channels is not a 3-element list");
    for (std::size_t i = 0; i < 3; ++i)
        if (integer(*channels.items[i], "mamba_args.in_channels") != kHeadInChannels[i]) reject("mamba_args.in_channels is not [256,512,512]");
    report << "config\tin_channels\t256,512,512\tchecked\n";
    want_int("d_model", kHeadDModel);
    want_int("d_state", kHeadDState);
    want_int("num_blocks", 1);
    want_int("spatial_reduction", kHeadReduction);
    want_int("num_classes", kHeadClasses);
    want_flag("use_cross_scan", true);
    want_flag("use_pixel_shuffle", true);
    want_flag("per_channel_a", false);
    want_flag("legacy_n1_scan", false);
    want_flag("use_hybrid_head", false);
    want_flag("use_temporal_attention", false);
    want_flag("use_detail_fusion", false);
    want_flag("use_strip_detail", false);
    // Keys added after this checkpoint was written: absent means the shapes
    // decide (reg 4 outputs -> reg_max 1; one 3x3 conv per branch, no BN ->
    // head_depth 1; 4x4 stride-4 conv weights -> reduction_variant conv).
    for (const auto& [key, value] : {std::pair{"reg_max", 1}, std::pair{"head_depth", 1}}) {
        if (const auto* v = args.find(key)) {
            if (integer(*v, std::string("mamba_args.") + key) != value) reject(std::string("mamba_args.") + key + " differs from " + std::to_string(value));
            report << "config\t" << key << '\t' << value << "\tchecked\n";
        } else report << "config\t" << key << "\tabsent\tderived from parameter shapes: " << value << '\n';
    }
    if (const auto* v = args.find("reduction_variant")) {
        if (v->kind != PickleValue::Kind::text || v->text != "conv") reject("mamba_args.reduction_variant is not conv");
        report << "config\treduction_variant\tconv\tchecked\n";
    } else report << "config\treduction_variant\tabsent\tderived from parameter shapes: conv\n";
    const auto* temporal = args.find("use_temporal_mamba");
    report << "config\tuse_temporal_mamba\t" << (temporal && temporal->kind == PickleValue::Kind::boolean ? (temporal->boolean ? "true" : "false") : "absent")
           << "\trecorded; T=1 bypasses the temporal blocks\n";
}

void put_line(std::string& out, const std::string& line) { out += line; out += '\n'; }

} // namespace

const std::vector<HeadParam>& head_parameter_schema() {
    static const auto schema = make_schema();
    return schema;
}

const std::vector<std::pair<std::string, std::string>>& head_excluded_keys() {
    static const auto excluded = make_excluded();
    return excluded;
}

const HeadTensor& HeadWeights::at(std::string_view name) const {
    const auto it = tensors.find(name);
    if (it == tensors.end()) throw std::invalid_argument("head weights: missing " + std::string(name));
    return it->second;
}

void validate_head_weights(const HeadWeights& w) {
    const auto& schema = head_parameter_schema();
    if (w.schema != kHeadSchema) throw std::invalid_argument("head weights: schema " + w.schema + " is not " + kHeadSchema);
    if (w.tensors.size() != schema.size())
        throw std::invalid_argument("head weights: " + std::to_string(w.tensors.size()) + " tensors, schema has " + std::to_string(schema.size()));
    for (const auto& p : schema) {
        const auto it = w.tensors.find(p.name);
        if (it == w.tensors.end()) throw std::invalid_argument("head weights: missing " + p.name);
        if (it->second.shape != p.shape) throw std::invalid_argument("head weights: " + p.name + " has shape [" + dims(it->second.shape) + "], expected [" + dims(p.shape) + "]");
        if (it->second.values.size() != elements(p.shape)) throw std::invalid_argument("head weights: " + p.name + " value count differs from its shape");
        for (std::size_t i = 0; i < it->second.values.size(); ++i)
            if (!std::isfinite(it->second.values[i])) throw std::invalid_argument("head weights: " + p.name + " element " + std::to_string(i) + " is not finite");
    }
}

HeadConversion convert_head_checkpoint(std::string checkpoint, std::string_view expected_sha256) {
    HeadConversion out;
    const auto file_sha = sha256_hex(checkpoint);
    if (file_sha != expected_sha256) reject("SHA-256 " + file_sha + " differs from the expected " + std::string(expected_sha256));
    const TorchCheckpoint ckpt(std::move(checkpoint));
    std::ostringstream report;
    report << "checkpoint_sha256\t" << file_sha << "\narchive\t" << ckpt.archive() << '\n';
    const auto& root = ckpt.root();
    if (!root.dict()) reject("pickle root is not a dict");
    for (const auto& [k, v] : *root.dict()) {
        (void)v;
        const auto name = k->kind == PickleValue::Kind::text ? k->text : std::string("<non-text key>");
        const bool read = name == "epoch" || name == "selection" || name == "mamba_args" || name == "student";
        report << "root_key\t" << name << '\t' << (read ? "read" : "not read: training state or metadata, not part of the T=1 forward") << '\n';
    }
    const auto epoch = integer(need(root, "epoch", "checkpoint"), "epoch");
    report << "epoch\t" << epoch << '\n';
    if (const auto* selection = root.find("selection"))
        if (const auto* status = selection->find("status"); status && status->kind == PickleValue::Kind::text)
            report << "selection_status\t" << status->text << "\trecorded only; not a quality verdict\n";
    check_config(need(root, "mamba_args", "checkpoint"), report);

    const auto& student = need(root, "student", "checkpoint");
    const auto* pairs = student.dict();
    if (!pairs) reject("student is not a state dict");
    std::map<std::string, const PickleValue*> keys;
    for (const auto& [k, v] : *pairs) {
        if (k->kind != PickleValue::Kind::text) reject("student has a non-string key");
        if (!keys.emplace(k->text, v.get()).second) reject("student has duplicate key " + k->text);
    }
    std::map<std::string, const HeadParam*> active;
    for (const auto& p : head_parameter_schema()) active.emplace(p.source_key, &p);
    std::map<std::string, std::string> excluded(head_excluded_keys().begin(), head_excluded_keys().end());
    for (const auto& [key, value] : keys)
        if (!active.contains(key) && !excluded.contains(key)) reject("student has unknown key " + key + " (no mapping; refusing to guess)");
    for (const auto& [key, param] : active)
        if (!keys.contains(key)) reject("student lacks required key " + key);

    report << "#source_key\tdisposition\tstorage_type\tshape\tstride\tstorage_key\tstorage_offset\tstorage_numel\tlocation\tsource_sha256"
              "\tbundle_name\tbundle_layout\ttransform\tbundle_sha256\tnote\n";
    std::map<std::string, std::string> rows;
    std::map<std::string, std::vector<std::string>> same_bytes; // sha256 -> bundle names
    for (const auto& [key, value] : keys) {
        TorchTensorRef ref;
        try { ref = torch_tensor_ref(*value); }
        catch (const std::exception& error) { reject(key + ": " + error.what()); }
        std::ostringstream row;
        row << "student." << key << '\t' << (active.contains(key) ? "active" : "excluded") << '\t' << ref.storage_type << '\t' << dims(ref.shape) << '\t'
            << dims(ref.stride) << '\t' << ref.storage_key << '\t' << ref.offset << '\t' << ref.storage_numel << '\t' << ref.location << '\t';
        const auto storage = ckpt.storage(ref); // size-checked against storage_numel
        const auto element = torch_storage_element_bytes(ref.storage_type);
        std::string_view view;
        if (ref.contiguous()) view = storage.substr(static_cast<std::size_t>(ref.offset) * element, static_cast<std::size_t>(ref.numel()) * element);
        row << (ref.contiguous() ? sha256_hex(view) : std::string("non-contiguous")) << '\t';
        if (const auto it = active.find(key); it != active.end()) {
            const auto& p = *it->second;
            if (ref.storage_type != "FloatStorage") reject(key + ": dtype " + ref.storage_type + " is not FP32 (FloatStorage)");
            if (ref.shape != p.shape) reject(key + ": shape [" + dims(ref.shape) + "] differs from the head's [" + dims(p.shape) + "]");
            if (!ref.contiguous()) reject(key + ": strides [" + dims(ref.stride) + "] are not contiguous row-major");
            HeadTensor t{p.shape, std::vector<float>(static_cast<std::size_t>(ref.numel()))};
            std::memcpy(t.values.data(), view.data(), view.size());
            for (std::size_t i = 0; i < t.values.size(); ++i)
                if (!std::isfinite(t.values[i])) reject(key + ": element " + std::to_string(i) + " is not finite");
            const auto sha = sha256_hex(float_bytes(t.values));
            same_bytes[sha].push_back(p.name);
            const bool conv1d = p.layout == "OIK";
            row << p.name << '\t' << p.layout << '\t'
                << (conv1d ? "none in bundle; viewed as a (256,1,1,4) depthwise 2-D kernel when the network is built" : "none (bytes and layout preserved)")
                << '\t' << sha << '\t' << (ref.parameter ? "parameter" : "tensor") << '\n';
            out.weights.tensors.emplace(p.name, std::move(t));
        } else {
            row << "-\t-\t-\t-\t" << excluded.at(key) << '\n';
        }
        rows.emplace(key, row.str());
    }
    for (const auto& [key, row] : rows) report << row;
    for (const auto& [sha, names] : same_bytes) {
        if (names.size() < 2) continue;
        report << "identical_bytes\t" << sha;
        for (const auto& n : names) report << '\t' << n;
        report << "\tkept as separate bundle tensors (distinct checkpoint parameters); no dedup or aliasing\n";
    }
    out.weights.source_sha256 = file_sha;
    out.weights.source_archive = ckpt.archive();
    out.weights.source_epoch = epoch;
    validate_head_weights(out.weights);
    out.bundle = encode_head_bundle(out.weights);
    report << "bundle_bytes\t" << out.bundle.size() << "\nbundle_sha256\t" << sha256_hex(out.bundle) << '\n';
    out.mapping = report.str();
    return out;
}

std::string encode_head_bundle(const HeadWeights& w) {
    validate_head_weights(w);
    if (w.source_sha256.size() != 64 || w.source_sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::invalid_argument("head bundle: source_sha256 must be 64 lowercase hex digits");
    if (w.source_archive.empty() || w.source_archive.find_first_of(" \t\n") != std::string::npos || w.source_epoch < -1)
        throw std::invalid_argument("head bundle: invalid provenance fields");
    std::string header, data;
    put_line(header, kHeadBundleMagic);
    put_line(header, std::string("schema ") + kHeadSchema);
    put_line(header, "source_sha256 " + w.source_sha256);
    put_line(header, "source_archive " + w.source_archive);
    put_line(header, "source_epoch " + std::to_string(w.source_epoch));
    put_line(header, "tensors " + std::to_string(head_parameter_schema().size()));
    for (const auto& p : head_parameter_schema()) {
        const auto& t = w.tensors.at(p.name);
        const auto bytes = float_bytes(t.values);
        put_line(header, "tensor " + p.name + " f32 " + dims(t.shape, 'x') + ' ' + std::to_string(data.size()) + ' ' + std::to_string(bytes.size()) +
                             ' ' + sha256_hex(bytes));
        data.append(bytes);
        data.append((64 - data.size() % 64) % 64, '\0');
    }
    put_line(header, "end");
    header.append((64 - header.size() % 64) % 64, '\0');
    return header + data;
}

HeadWeights decode_head_bundle(std::string_view bytes) {
    const auto fail = [](const std::string& why) { throw std::invalid_argument("head bundle: " + why); };
    const auto end = bytes.find("\nend\n");
    if (end == std::string_view::npos) fail("no header terminator");
    std::istringstream lines(std::string(bytes.substr(0, end + 5)));
    const auto data_offset = (end + 5 + 63) / 64 * 64;
    if (data_offset > bytes.size()) fail("truncated header padding");
    const auto data = bytes.substr(data_offset);
    HeadWeights w;
    std::string line;
    const auto field = [&](const std::string& key) {
        if (!std::getline(lines, line) || line.compare(0, key.size() + 1, key + " ") != 0) fail("expected header field " + key);
        return line.substr(key.size() + 1);
    };
    if (!std::getline(lines, line) || line != kHeadBundleMagic) fail("bad magic or version");
    w.schema = field("schema");
    w.source_sha256 = field("source_sha256");
    w.source_archive = field("source_archive");
    const auto epoch = field("source_epoch");
    if (std::from_chars(epoch.data(), epoch.data() + epoch.size(), w.source_epoch).ec != std::errc{}) fail("bad source_epoch");
    if (field("tensors") != std::to_string(head_parameter_schema().size())) fail("tensor count differs from the schema");
    for (const auto& p : head_parameter_schema()) {
        std::istringstream f(field("tensor"));
        std::string name, dtype, shape, sha;
        std::uint64_t offset = 0, size = 0;
        if (!(f >> name >> dtype >> shape >> offset >> size >> sha) || !f.eof()) fail("malformed tensor line: " + line);
        if (name != p.name) fail("tensor " + name + " is out of order or unknown; expected " + p.name);
        if (dtype != "f32") fail(name + ": dtype " + dtype + " is not f32");
        if (shape != dims(p.shape, 'x')) fail(name + ": shape " + shape + " differs from the schema's " + dims(p.shape, 'x'));
        if (size != elements(p.shape) * sizeof(float) || offset > data.size() || size > data.size() - offset) fail(name + ": data range outside the bundle");
        HeadTensor t{p.shape, std::vector<float>(elements(p.shape))};
        std::memcpy(t.values.data(), data.data() + offset, size);
        if (sha256_hex(data.substr(offset, size)) != sha) fail(name + ": SHA-256 mismatch");
        w.tensors.emplace(name, std::move(t));
    }
    if (!std::getline(lines, line) || line != "end" || std::getline(lines, line)) fail("missing end line");
    validate_head_weights(w); // finite values
    if (encode_head_bundle(w) != bytes) fail("bytes are not the canonical encoding (offsets, padding or trailing data)");
    return w;
}

} // namespace saccade

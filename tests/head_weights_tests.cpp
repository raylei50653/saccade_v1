#include "saccade/head_reference.hpp"
#include "saccade/head_weights.hpp"
#include "saccade/sha256.hpp"
#include "saccade/torch_checkpoint.hpp"

#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// S04 weight delivery (docs/HEAD_TRT.md §2): the torch.save reader, the head
// checkpoint conversion and the bundle codec, on synthetic archives written
// here byte by byte in the torch.save layout. Each rejection case changes one
// thing of an otherwise valid archive and must be refused with its diagnostic.
namespace {
using saccade::HeadWeights;

std::size_t checks = 0;
void check(bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template <typename F> void expect_reject(F&& action, const std::string& needle) {
    try { action(); }
    catch (const std::exception& error) {
        check(std::string(error.what()).find(needle) != std::string::npos, "unexpected diagnostic: " + std::string(error.what()) + "; wanted " + needle);
        return;
    }
    check(false, "expected rejection: " + needle);
}

void le(std::string& out, std::uint64_t v, int n) {
    for (int i = 0; i < n; ++i) out += static_cast<char>((v >> (8 * i)) & 0xFF);
}

// --------------------------------------------------------- pickle writer
struct Pickle {
    std::string out = std::string("\x80\x02", 2);
    void text(std::string_view s) { out += 'X'; le(out, s.size(), 4); out += s; }
    void integer(std::int64_t v) {
        if (v >= 0 && v < 256) { out += 'K'; out += static_cast<char>(v); }
        else { out += 'J'; le(out, static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<std::int32_t>(v))), 4); }
    }
    void boolean(bool b) { out += b ? '\x88' : '\x89'; }
    void global(std::string_view module, std::string_view name) { out += 'c'; out += module; out += '\n'; out += name; out += '\n'; }
    void mark() { out += '('; }
    void tuple() { out += 't'; }
    void ordered_dict() { global("collections", "OrderedDict"); out += ')'; out += 'R'; }
    void ints(const std::vector<std::int64_t>& v) { mark(); for (const auto x : v) integer(x); tuple(); }
    void tensor(const std::string& storage_type, const std::string& key, std::int64_t numel, std::int64_t offset, const std::vector<std::int64_t>& shape,
                const std::vector<std::int64_t>& stride) {
        global("torch._utils", "_rebuild_tensor_v2");
        mark();
        mark(); text("storage"); global("torch", storage_type); text(key); text("cuda:0"); integer(numel); tuple(); out += 'Q';
        integer(offset); ints(shape); ints(stride); boolean(false); ordered_dict();
        tuple();
        out += 'R';
    }
};

// ------------------------------------------------------------ zip writer
struct ZipOptions { bool descriptors = true, zip64_end = true; int method = 0; };
std::string zip(const std::vector<std::pair<std::string, std::string>>& entries, const ZipOptions& o = {}) {
    std::string out, central;
    for (const auto& [name, data] : entries) {
        const auto crc = saccade::crc32(data);
        const std::uint64_t at = out.size();
        const std::uint16_t flags = o.descriptors ? 0x0808 : 0x0800;
        le(out, 0x04034b50, 4); le(out, 0, 2); le(out, flags, 2); le(out, static_cast<std::uint64_t>(o.method), 2); le(out, 0, 4);
        le(out, o.descriptors ? 0 : crc, 4); le(out, o.descriptors ? 0 : data.size(), 4); le(out, o.descriptors ? 0 : data.size(), 4);
        le(out, name.size(), 2); le(out, 0, 2); out += name; out += data;
        if (o.descriptors) { le(out, 0x08074b50, 4); le(out, crc, 4); le(out, data.size(), 4); le(out, data.size(), 4); }
        le(central, 0x02014b50, 4); le(central, 0, 2); le(central, 0, 2); le(central, flags, 2); le(central, static_cast<std::uint64_t>(o.method), 2);
        le(central, 0, 4); le(central, crc, 4); le(central, data.size(), 4); le(central, data.size(), 4); le(central, name.size(), 2);
        le(central, 0, 2); le(central, 0, 2); le(central, 0, 2); le(central, 0, 2); le(central, 0, 4); le(central, at, 4); central += name;
    }
    const std::uint64_t cd = out.size();
    out += central;
    if (o.zip64_end) {
        const std::uint64_t record = out.size();
        le(out, 0x06064b50, 4); le(out, 44, 8); le(out, 45, 2); le(out, 45, 2); le(out, 0, 4); le(out, 0, 4);
        le(out, entries.size(), 8); le(out, entries.size(), 8); le(out, central.size(), 8); le(out, cd, 8);
        le(out, 0x07064b50, 4); le(out, 0, 4); le(out, record, 8); le(out, 1, 4);
    }
    le(out, 0x06054b50, 4); le(out, 0, 2); le(out, 0, 2); le(out, entries.size(), 2); le(out, entries.size(), 2);
    le(out, central.size(), 4); le(out, cd, 4); le(out, 0, 2);
    return out;
}

// ---------------------------------------------- synthetic head checkpoint
struct Fixture {
    HeadWeights weights = saccade::synthetic_head_weights(7);
    std::map<std::string, std::string> config_ints{{"d_model", "128"}, {"d_state", "16"}, {"num_blocks", "1"}, {"spatial_reduction", "4"}, {"num_classes", "80"}};
    std::map<std::string, bool> config_flags{{"use_cross_scan", true}, {"use_pixel_shuffle", true}, {"per_channel_a", false}, {"legacy_n1_scan", false},
                                             {"use_hybrid_head", false}, {"use_temporal_attention", false}, {"use_detail_fusion", false},
                                             {"use_strip_detail", false}, {"use_temporal_mamba", true}};
    std::vector<std::int64_t> in_channels{256, 512, 512};
    // Per checkpoint key overrides of the tensor record.
    std::map<std::string, std::string> storage_type;
    std::map<std::string, std::vector<std::int64_t>> shape, stride;
    std::map<std::string, std::int64_t> offset, extra_numel, short_storage;
    std::map<std::string, float> poison;
    std::vector<std::string> drop, duplicate, unknown;
    ZipOptions zip_options;

    std::string build() const {
        std::map<std::string, std::pair<std::vector<std::int64_t>, std::vector<float>>> tensors; // checkpoint key -> shape, values
        for (const auto& p : saccade::head_parameter_schema()) tensors[p.source_key] = {p.shape, weights.at(p.name).values};
        for (const auto& [key, reason] : saccade::head_excluded_keys()) {
            (void)reason;
            const std::vector<std::int64_t> s = key.starts_with("flow") ? std::vector<std::int64_t>{128, 2} : std::vector<std::int64_t>{3, 5};
            tensors[key] = {s, std::vector<float>(static_cast<std::size_t>(s[0] * s[1]), 0.25f)};
        }
        for (const auto& k : unknown) tensors[k] = {{2}, {1.0f, 2.0f}};
        for (const auto& k : drop) tensors.erase(k);
        Pickle p;
        std::vector<std::pair<std::string, std::string>> records;
        p.out += '}'; p.mark();
        p.text("epoch"); p.integer(15);
        p.text("student"); p.ordered_dict(); p.mark();
        int key = 0;
        const auto put = [&](const std::string& name, const std::vector<std::int64_t>& s0, std::vector<float> values) {
            if (poison.contains(name)) values[1] = poison.at(name);
            const auto shp = shape.contains(name) ? shape.at(name) : s0;
            std::vector<std::int64_t> str(shp.size());
            std::int64_t step = 1;
            for (std::size_t n = shp.size(); n-- > 0;) { str[n] = step; step *= shp[n]; }
            if (stride.contains(name)) str = stride.at(name);
            const auto type = storage_type.contains(name) ? storage_type.at(name) : "FloatStorage";
            const auto off = offset.contains(name) ? offset.at(name) : 0;
            std::int64_t numel = static_cast<std::int64_t>(values.size()) + (extra_numel.contains(name) ? extra_numel.at(name) : 0);
            std::string bytes(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            if (type == "DoubleStorage") bytes += bytes;
            if (short_storage.contains(name)) bytes.resize(bytes.size() - static_cast<std::size_t>(short_storage.at(name)));
            const auto k = std::to_string(key++);
            p.text(name); p.tensor(type, k, numel, off, shp, str);
            records.emplace_back("best/data/" + k, bytes);
        };
        for (const auto& [name, t] : tensors) put(name, t.first, t.second);
        for (const auto& name : duplicate) put(name, tensors.at(name).first, tensors.at(name).second);
        p.out += 'u';
        p.text("mamba_args"); p.out += '}'; p.mark();
        p.text("in_channels"); p.out += ']'; p.mark(); for (const auto c : in_channels) p.integer(c); p.out += 'e';
        for (const auto& [k, v] : config_ints) { p.text(k); p.integer(std::stoll(v)); }
        for (const auto& [k, v] : config_flags) { p.text(k); p.boolean(v); }
        p.out += 'u';
        p.text("selection"); p.out += '}'; p.mark(); p.text("status"); p.text("diagnostic_train_loss"); p.out += 'u';
        p.out += 'u'; p.out += '.';
        std::vector<std::pair<std::string, std::string>> entries{{"best/data.pkl", p.out}, {"best/byteorder", "little"}};
        entries.insert(entries.end(), records.begin(), records.end());
        entries.emplace_back("best/version", "3\n");
        return zip(entries, zip_options);
    }
};

saccade::HeadConversion convert(const std::string& file) { return saccade::convert_head_checkpoint(file, saccade::sha256_hex(file)); }

// The schema's key mapping, checked against an independent statement of it:
// bundle level p3/p4/p5 is checkpoint index 0/1/2, plus literal pairs. The
// synthetic archives below are written from the schema itself, so they cannot
// catch a mapping error on their own.
void schema_mapping() {
    const auto& schema = saccade::head_parameter_schema();
    check(schema.size() == 69 && saccade::head_excluded_keys().size() == 39, "69 active and 39 excluded keys");
    for (const auto& p : schema) {
        const int level = p.name[1] - '3';
        const auto digit = p.source_key.find_first_of("0123456789");
        check(level >= 0 && level < 3 && digit != std::string::npos && p.source_key[digit] - '0' == level &&
                  p.source_key.compare(digit + 1, 1, ".") == 0,
              "bundle level of " + p.name + " is the checkpoint index of " + p.source_key);
    }
    const std::pair<const char*, const char*> literal[] = {
        {"p3.input_proj.weight", "input_proj.0.weight"}, {"p4.downsample.bias", "downsample.1.bias"},
        {"p4.mamba.x_proj.weight", "mamba_blocks.1.0.x_proj.weight"}, {"p5.mamba.A_log", "mamba_blocks.2.0.A_log"},
        {"p5.upsample.weight", "upsample.2.0.weight"}, {"p3.cls.conv1.bias", "cls_head.0.2.bias"}, {"p4.reg.conv0.weight", "reg_head.1.0.weight"}};
    for (const auto& [name, key] : literal) {
        bool found = false;
        for (const auto& p : schema) found = found || (p.name == name && p.source_key == key);
        check(found, std::string("schema maps ") + key + " to " + name);
    }
    std::cout << "schema_mapping\tPASS\n";
}

void conversion() {
    const Fixture valid;
    const auto file = valid.build();
    const auto result = convert(file);
    check(result.weights.tensors.size() == 69, "valid archive converts 69 tensors");
    for (const auto& p : saccade::head_parameter_schema())
        check(result.weights.at(p.name).values == valid.weights.at(p.name).values, "converted tensor equals the archive bytes: " + p.name);
    check(result.weights.source_sha256 == saccade::sha256_hex(file) && result.weights.source_epoch == 15 && result.weights.source_archive == "best",
          "bundle provenance records the checkpoint");
    check(saccade::decode_head_bundle(result.bundle).tensors.size() == 69 && saccade::encode_head_bundle(result.weights) == result.bundle,
          "bundle decodes and re-encodes to the same bytes");
    std::size_t active = 0, excluded = 0;
    std::istringstream lines(result.mapping);
    for (std::string line; std::getline(lines, line);) {
        active += line.find("\tactive\t") != std::string::npos;
        excluded += line.find("\texcluded\t") != std::string::npos;
    }
    check(active == 69 && excluded == 39, "mapping lists 69 active and 39 excluded checkpoint keys");
    check(result.mapping.find("identical_bytes") != std::string::npos, "mapping records byte-identical parameters without deduplicating them");
    {
        Fixture plain; plain.zip_options = {false, false, 0};
        // Only the provenance line (the archive's own SHA-256) may differ.
        const auto other = convert(plain.build());
        bool same = true;
        for (const auto& [name, t] : result.weights.tensors) same = same && other.weights.at(name).values == t.values;
        check(same, "archives without data descriptors or zip64 end records convert to the same tensors");
    }
    expect_reject([&] { (void)saccade::convert_head_checkpoint(file, std::string(64, '0')); }, "differs from the expected");

    const std::vector<std::pair<std::string, std::function<void(Fixture&)>>> cases{
        {"lacks required key mamba_blocks.1.0.x_proj.weight", [](Fixture& f) { f.drop.push_back("mamba_blocks.1.0.x_proj.weight"); }},
        {"unknown key emb_head.0.0.weight", [](Fixture& f) { f.unknown.push_back("emb_head.0.0.weight"); }},
        {"duplicate key upsample.2.0.bias", [](Fixture& f) { f.duplicate.push_back("upsample.2.0.bias"); }},
        {"differs from the head's", [](Fixture& f) { f.shape["downsample.0.weight"] = {128, 128, 2, 8}; }},
        {"is not FP32", [](Fixture& f) { f.storage_type["mamba_blocks.0.0.dt_proj.weight"] = "DoubleStorage"; }},
        {"not contiguous", [](Fixture& f) { f.stride["mamba_blocks.2.0.out_proj.weight"] = {1, 128}; }},
        {"view exceeds its storage", [](Fixture& f) { f.offset["cls_head.1.2.bias"] = 3; }},
        {"holds", [](Fixture& f) { f.short_storage["reg_head.0.2.bias"] = 4; }},
        {"is not finite", [](Fixture& f) { f.poison["mamba_blocks.0.0.D"] = std::numeric_limits<float>::infinity(); }},
        {"is not finite", [](Fixture& f) { f.poison["upsample.1.0.weight"] = std::numeric_limits<float>::quiet_NaN(); }},
        {"mamba_args.d_state", [](Fixture& f) { f.config_ints["d_state"] = "8"; }},
        {"mamba_args.per_channel_a", [](Fixture& f) { f.config_flags["per_channel_a"] = true; }},
        {"mamba_args.use_pixel_shuffle", [](Fixture& f) { f.config_flags["use_pixel_shuffle"] = false; }},
        {"mamba_args lacks legacy_n1_scan", [](Fixture& f) { f.config_flags.erase("legacy_n1_scan"); }},
        {"in_channels", [](Fixture& f) { f.in_channels = {128, 256, 512}; }},
        {"compression method 8", [](Fixture& f) { f.zip_options.method = 8; }},
    };
    for (const auto& [needle, mutate] : cases) {
        Fixture f;
        mutate(f);
        const auto bytes = f.build();
        expect_reject([&] { (void)convert(bytes); }, needle);
    }
    // Byte-level damage to the valid archive.
    auto crc = file;
    crc[crc.size() / 2] ^= 0x01;
    expect_reject([&] { (void)convert(crc); }, "CRC-32 mismatch");
    expect_reject([&] { (void)convert(file.substr(0, file.size() - 7)); }, "zip:");
    auto trailing = file + "x";
    expect_reject([&] { (void)convert(trailing); }, "end-of-central-directory");
    {
        // An opcode outside the container subset (EXT1) inside data.pkl.
        Pickle p; p.out += '}'; p.out += '\x82'; p.out += '\x01'; p.out += '.';
        const auto bytes = zip({{"best/data.pkl", p.out}});
        expect_reject([&] { (void)saccade::TorchCheckpoint(bytes); }, "unsupported opcode 0x82");
    }
    expect_reject([&] { (void)saccade::TorchCheckpoint(zip({{"best/data.pkl", "\x80\x02}."}, {"other/x", "1"}})); }, "outside archive directory");
    expect_reject([&] { (void)saccade::TorchCheckpoint(zip({{"best/data.pkl", "\x80\x02}."}, {"best/byteorder", "big"}})); }, "byteorder");
    std::cout << "conversion\tPASS\n";
}

void bundle() {
    const auto w = saccade::synthetic_head_weights(11);
    const auto bytes = saccade::encode_head_bundle(w);
    check(saccade::decode_head_bundle(bytes).at("p4.mamba.in_proj.weight").values == w.at("p4.mamba.in_proj.weight").values, "bundle round trip");
    const auto data = bytes.find("\nend\n");
    // The last tensor (4 floats) starts the last 64-byte block; its padding follows.
    auto flipped = bytes; flipped[flipped.size() - 64 + 1] ^= 0x10;
    expect_reject([&] { (void)saccade::decode_head_bundle(flipped); }, "SHA-256 mismatch");
    auto padding = bytes; padding[padding.size() - 3] ^= 0x10;
    expect_reject([&] { (void)saccade::decode_head_bundle(padding); }, "not the canonical encoding");
    expect_reject([&] { (void)saccade::decode_head_bundle(bytes.substr(0, bytes.size() - 64)); }, "data range outside the bundle");
    expect_reject([&] { (void)saccade::decode_head_bundle(bytes + std::string(64, '\0')); }, "not the canonical encoding");
    auto magic = bytes; magic[5] = '2';
    expect_reject([&] { (void)saccade::decode_head_bundle(magic); }, "bad magic or version");
    auto order = bytes;
    const auto a = order.find("tensor p3.input_proj.weight"), b = order.find("tensor p3.input_proj.bias");
    order.replace(a, 27, "tensor p3.input_proj.bias  ");
    (void)b;
    expect_reject([&] { (void)saccade::decode_head_bundle(order); }, "out of order or unknown");
    auto shape = bytes; shape.replace(shape.find("128x256x1x1"), 11, "128x256x1x2");
    expect_reject([&] { (void)saccade::decode_head_bundle(shape); }, "differs from the schema");
    auto dtype = bytes; dtype.replace(dtype.find(" f32 "), 5, " f16 ");
    expect_reject([&] { (void)saccade::decode_head_bundle(dtype); }, "is not f32");
    // A non-finite value with a consistent SHA-256 is still refused.
    auto nonfinite = w;
    nonfinite.tensors.at("p5.reg.conv1.bias").values[2] = std::numeric_limits<float>::infinity();
    expect_reject([&] { (void)saccade::encode_head_bundle(nonfinite); }, "is not finite");
    auto missing = w; missing.tensors.erase("p3.mamba.A_log");
    expect_reject([&] { (void)saccade::encode_head_bundle(missing); }, "tensors, schema has");
    auto wrong = w; wrong.tensors.at("p3.mamba.A_log").shape = {16, 1};
    expect_reject([&] { (void)saccade::encode_head_bundle(wrong); }, "has shape [16,1]");
    check(data != std::string::npos, "bundle header terminator present");
    std::cout << "bundle\tPASS\n";
}

} // namespace

int main() {
    try {
        schema_mapping();
        conversion();
        bundle();
        std::cout << "head_weights_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "head_weights_tests: " << error.what() << '\n';
        return 1;
    }
}

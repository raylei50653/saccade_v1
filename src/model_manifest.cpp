#include "saccade/model_manifest.hpp"
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <variant>

namespace saccade {
namespace {
[[noreturn]] void invalid(std::string_view path, std::string_view why) {
    throw std::invalid_argument("manifest " + std::string(path) + ": " + std::string(why));
}
struct Json {
    using Object = std::map<std::string, Json>;
    using Array = std::vector<Json>;
    std::variant<std::nullptr_t, bool, std::uint64_t, std::string, Object, Array> data;
};
class Parser {
public:
    explicit Parser(std::string_view text) : input(text) {}
    Json parse() {
        auto result = value("$", 0);
        space();
        if (position != input.size()) fail("$", "trailing input");
        return result;
    }
private:
    std::string_view input;
    std::size_t position = 0;
    [[noreturn]] void fail(std::string_view path, std::string_view message) const {
        invalid(path, std::string(message) + " at byte " + std::to_string(position));
    }
    void space() {
        while (position < input.size() && (input[position] == ' ' || input[position] == '\n' ||
               input[position] == '\r' || input[position] == '\t')) ++position;
    }
    bool take(char c) {
        space();
        if (position < input.size() && input[position] == c) { ++position; return true; }
        return false;
    }
    std::string string(std::string_view path) {
        if (!take('"')) fail(path, "expected string");
        std::string result;
        while (position < input.size()) {
            const unsigned char c = input[position++];
            if (c == '"') return result;
            if (c < 32 || c > 126) fail(path, "ASCII JSON string required");
            if (c != '\\') { result += static_cast<char>(c); continue; }
            if (position == input.size()) fail(path, "unfinished escape");
            const char e = input[position++];
            switch (e) {
            case '"': result += '"'; break;
            case '\\': result += '\\'; break;
            case '/': result += '/'; break;
            case 'b': result += '\b'; break;
            case 'f': result += '\f'; break;
            case 'n': result += '\n'; break;
            case 'r': result += '\r'; break;
            case 't': result += '\t'; break;
            case 'u': {
                unsigned int code = 0;
                for (int i = 0; i < 4; ++i) {
                    if (position == input.size()) fail(path, "unfinished unicode escape");
                    const char h = input[position++];
                    code *= 16;
                    if (h >= '0' && h <= '9') code += h - '0';
                    else if (h >= 'a' && h <= 'f') code += h - 'a' + 10;
                    else if (h >= 'A' && h <= 'F') code += h - 'A' + 10;
                    else fail(path, "invalid unicode escape");
                }
                if (code > 126) fail(path, "only ASCII unicode escapes supported");
                result += static_cast<char>(code);
                break;
            }
            default: fail(path, "invalid string escape");
            }
        }
        fail(path, "unterminated string");
    }
    Json value(const std::string path, unsigned int depth) {
        if (depth > 64) fail(path, "maximum nesting exceeded");
        space();
        if (position == input.size()) fail(path, "expected value");
        const char c = input[position];
        if (c == '"') return Json{string(path)};
        if (take('{')) {
            Json::Object object;
            if (take('}')) return Json{object};
            for (;;) {
                const auto key = string(path);
                const auto child = path + "." + key;
                if (!take(':')) fail(child, "expected colon");
                auto entry = value(child, depth + 1);
                if (!object.emplace(key, std::move(entry)).second) fail(child, "duplicate field");
                if (take('}')) return Json{object};
                if (!take(',')) fail(path, "expected comma or closing brace");
            }
        }
        if (take('[')) {
            Json::Array array;
            if (take(']')) return Json{array};
            for (;;) {
                array.push_back(value(path + "[" + std::to_string(array.size()) + "]", depth + 1));
                if (take(']')) return Json{array};
                if (!take(',')) fail(path, "expected comma or closing bracket");
            }
        }
        for (const auto& token : {std::string("true"), std::string("false"), std::string("null")}) {
            if (input.substr(position, token.size()) == token) {
                position += token.size();
                if (token == "null") return Json{nullptr};
                return Json{token == "true"};
            }
        }
        if (c < '0' || c > '9') fail(path, "expected unsigned integer, boolean, string, object, array or null");
        const auto start = position;
        while (position < input.size() && input[position] >= '0' && input[position] <= '9') ++position;
        if (position - start > 1 && input[start] == '0') fail(path, "leading zero");
        std::uint64_t number{};
        const auto [end, error] = std::from_chars(input.data() + start, input.data() + position, number);
        if (error != std::errc{} || end != input.data() + position) fail(path, "integer out of uint64 range");
        return Json{number};
    }
};
std::string quote(std::string_view text) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else result += static_cast<char>(c);
    }
    return result + '"';
}
std::string canonical(const Json& value) {
    if (std::holds_alternative<std::nullptr_t>(value.data)) return "null";
    if (const auto* p = std::get_if<bool>(&value.data)) return *p ? "true" : "false";
    if (const auto* p = std::get_if<std::uint64_t>(&value.data)) return std::to_string(*p);
    if (const auto* p = std::get_if<std::string>(&value.data)) return quote(*p);
    std::string result;
    if (const auto* p = std::get_if<Json::Object>(&value.data)) {
        result = "{";
        for (const auto& [key, item] : *p) {
            if (result.size() > 1) result += ',';
            result += quote(key) + ':' + canonical(item);
        }
        return result + '}';
    }
    result = "[";
    for (const auto& item : std::get<Json::Array>(value.data)) {
        if (result.size() > 1) result += ',';
        result += canonical(item);
    }
    return result + ']';
}
const Json::Object& object(const Json& value, std::string_view path,
                           std::initializer_list<std::string_view> keys) {
    const auto* result = std::get_if<Json::Object>(&value.data);
    if (!result) invalid(path, "expected object");
    for (const auto& [key, item] : *result) {
        (void)item;
        if (std::find(keys.begin(), keys.end(), key) == keys.end()) invalid(std::string(path) + "." + key, "unknown field");
    }
    for (const auto key : keys) if (!result->contains(std::string(key))) invalid(std::string(path) + "." + std::string(key), "missing required field");
    return *result;
}
const Json::Array& array(const Json& value, std::string_view path, bool allow_empty = false) {
    const auto* result = std::get_if<Json::Array>(&value.data);
    if (!result) invalid(path, "expected array");
    if (!allow_empty && result->empty()) invalid(path, "array must not be empty");
    return *result;
}
const std::string& str(const Json& value, std::string_view path, bool allow_empty = false) {
    const auto* result = std::get_if<std::string>(&value.data);
    if (!result) invalid(path, "expected string");
    if (!allow_empty && result->empty()) invalid(path, "string must not be empty");
    for (const unsigned char c : *result) if (c < 32 || c > 126) invalid(path, "printable ASCII required");
    return *result;
}
std::uint64_t number(const Json& value, const std::string path) {
    const auto* result = std::get_if<std::uint64_t>(&value.data);
    if (!result) invalid(path, "expected unsigned integer");
    return *result;
}
bool boolean(const Json& value, const std::string path) {
    const auto* result = std::get_if<bool>(&value.data);
    if (!result) invalid(path, "expected boolean");
    return *result;
}
void equals(const Json& value, const std::string path, std::string_view expected) {
    if (str(value, path) != expected) invalid(path, "expected " + std::string(expected));
}
void equals(const Json& value, const std::string path, std::uint64_t expected) {
    if (number(value, path) != expected) invalid(path, "expected " + std::to_string(expected));
}
const std::string& choice(const Json& value, std::string_view path,
                          std::initializer_list<std::string_view> choices) {
    const auto& result = str(value, path);
    if (std::find(choices.begin(), choices.end(), result) == choices.end()) invalid(path, "unsupported value '" + result + "'");
    return result;
}
void string_fields(const Json::Object& value, const std::string path,
                   std::initializer_list<std::string_view> fields) {
    for (const auto field : fields) str(value.at(std::string(field)), path + "." + std::string(field));
}
void strings(const Json& value, const std::string path, bool allow_empty = false) {
    const auto& entries = array(value, path, allow_empty);
    for (std::size_t i = 0; i < entries.size(); ++i) str(entries[i], path + "[" + std::to_string(i) + "]");
}
void decimal(const Json& value, const std::string path) {
    const auto& text = str(value, path);
    bool dot = false, digit = false;
    for (const char c : text) {
        if (c == '.' && !dot) { dot = true; continue; }
        if (c < '0' || c > '9') invalid(path, "expected nonnegative finite decimal string (no exponent)");
        digit = true;
    }
    if (!digit || text.front() == '.' || text.back() == '.') invalid(path, "expected decimal string");
    const auto integer_digits = text.find('.');
    const auto integer_size = integer_digits == std::string::npos ? text.size() : integer_digits;
    if ((integer_size > 1 && text.front() == '0') || (dot && text.back() == '0'))
        invalid(path, "decimal string must use canonical spelling (no leading integer or trailing fractional zeros)");
}
std::uint64_t multiply(std::uint64_t a, std::uint64_t b, const std::string path) {
    if (b == 0) invalid(path, "dimension must be positive");
    if (a > std::numeric_limits<std::uint64_t>::max() / b) invalid(path, "dimension product or byte size overflow");
    return a * b;
}
void relative_path(std::string_view value, const std::string path) {
    const std::filesystem::path file(value);
    if (file.empty() || file.is_absolute() || file.has_root_name() || file.has_root_directory()) invalid(path, "expected relative artifact path");
    if (value.find('\\') != std::string_view::npos) invalid(path, "use forward slash path separators");
    // No hidden cwd or traversal base: parent references are not needed in an
    // artifact bundle and would allow a relocated manifest to select other data.
    for (const auto& component : file) if (component == "..") invalid(path, "parent traversal is not permitted");
}
} // namespace

ModelManifest parse_model_manifest(std::string_view input) {
    const auto root = Parser(input).parse();
    const auto& m = object(root, "$", {"schema", "purpose", "model_family", "artifacts", "boundary", "tensors", "input_profile", "scan", "precision", "content_review"});
    equals(m.at("schema"), "$.schema", std::uint64_t{1});
    ModelManifest result;
    result.purpose = choice(m.at("purpose"), "$.purpose", {"test_fixture", "model_candidate"});
    equals(m.at("model_family"), "$.model_family", "YOLO26s-backbone-neck+Mamba-FPN-head");
    std::set<std::string> artifact_ids;
    std::map<std::string, std::pair<std::string, std::string>> artifact_kinds;
    const auto& artifacts = array(m.at("artifacts"), "$.artifacts");
    for (std::size_t i = 0; i < artifacts.size(); ++i) {
        const auto path = "$.artifacts[" + std::to_string(i) + "]";
        const auto& a = object(artifacts[i], path, {"id", "role", "path", "format", "source", "revision", "license", "generated_by", "size_bytes", "sha256"});
        string_fields(a, path, {"id", "source", "revision", "license", "generated_by"});
        const auto& id = str(a.at("id"), path + ".id");
        if (!artifact_ids.insert(id).second) invalid(path + ".id", "duplicate artifact id");
        const auto& role = choice(a.at("role"), path + ".role", {"backbone_neck", "mamba_fpn_head", "combined_model", "upstream_detector", "checkpoint", "specification", "fixture"});
        const auto& format = choice(a.at("format"), path + ".format", {"onnx", "checkpoint", "native_weights", "text", "fixture_bytes"});
        artifact_kinds.emplace(id, std::pair{role, format});
        if (result.purpose == "model_candidate" && (role == "fixture" || format == "fixture_bytes")) invalid(path, "test fixture cannot be a model candidate");
        const auto& file = str(a.at("path"), path + ".path");
        relative_path(file, path + ".path");
        const auto size = number(a.at("size_bytes"), path + ".size_bytes");
        const auto& hash = str(a.at("sha256"), path + ".sha256");
        if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdef") != std::string::npos) invalid(path + ".sha256", "expected 64 lowercase hexadecimal digits");
        result.artifacts.push_back({id, file, size, hash});
    }
    const auto& boundary = object(m.at("boundary"), "$.boundary", {"packaging", "components", "classification_owner", "regression_owner", "decode_owner"});
    const auto& packaging = choice(boundary.at("packaging"), "$.boundary.packaging", {"merged", "separate", "incomplete"});
    for (const auto key : {"classification_owner", "regression_owner", "decode_owner"})
        choice(boundary.at(key), "$.boundary." + std::string(key), {"head", "external", "unverified"});
    const auto& components = array(boundary.at("components"), "$.boundary.components");
    std::map<std::string, std::string> component_files;
    for (std::size_t i = 0; i < components.size(); ++i) {
        const auto path = "$.boundary.components[" + std::to_string(i) + "]";
        const auto& c = object(components[i], path, {"role", "artifact"});
        const auto role = choice(c.at("role"), path + ".role", {"backbone", "neck", "head"});
        std::string artifact;
        if (!std::holds_alternative<std::nullptr_t>(c.at("artifact").data)) {
            artifact = str(c.at("artifact"), path + ".artifact");
            if (!artifact_ids.contains(artifact)) invalid(path + ".artifact", "unknown artifact id");
            const auto& [artifact_role, artifact_format] = artifact_kinds.at(artifact);
            const bool fixture = result.purpose == "test_fixture" && artifact_role == "fixture" && artifact_format == "fixture_bytes";
            if (!fixture) {
                if (artifact_format == "text" || artifact_format == "fixture_bytes" || artifact_role == "specification" || artifact_role == "fixture")
                    invalid(path + ".artifact", "component must map to model content, not specification or fixture");
                if ((role == "head" && (artifact_role == "backbone_neck" || artifact_role == "upstream_detector")) ||
                    (role != "head" && artifact_role == "mamba_fpn_head"))
                    invalid(path + ".artifact", "artifact role incompatible with component role");
            }
        }
        if (!component_files.emplace(role, artifact).second) invalid(path + ".role", "duplicate component role");
    }
    if (component_files.size() != 3) invalid("$.boundary.components", "backbone, neck and head entries required (null artifact for missing component)");
    const auto& backbone = component_files.at("backbone");
    const auto& neck = component_files.at("neck");
    const auto& head = component_files.at("head");
    const bool missing = backbone.empty() || neck.empty() || head.empty();
    if ((packaging == "incomplete") != missing) invalid("$.boundary.packaging", "incomplete must match missing component mappings");
    if (packaging == "merged" && (backbone != neck || neck != head)) invalid("$.boundary.components", "merged packaging requires one shared artifact");
    if (packaging == "separate" && backbone == head && neck == head) invalid("$.boundary.components", "separate packaging requires multiple artifacts");

    const auto& tensors = array(m.at("tensors"), "$.tensors");
    std::set<std::string> tensor_names;
    std::map<std::string, std::uint64_t> next_order;
    std::string profile_tensor;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto path = "$.tensors[" + std::to_string(i) + "]";
        const auto& t = object(tensors[i], path, {"name", "order", "direction", "component", "shape", "dtype", "layout", "semantic"});
        const auto& name = str(t.at("name"), path + ".name");
        if (!tensor_names.insert(name).second) invalid(path + ".name", "duplicate tensor name");
        const auto& direction = choice(t.at("direction"), path + ".direction", {"input", "output", "intermediate"});
        const auto& component = choice(t.at("component"), path + ".component", {"model", "backbone", "neck", "head", "scan"});
        const auto key = component + ":" + direction;
        if (number(t.at("order"), path + ".order") != next_order[key]++) invalid(path + ".order", "order must be contiguous from zero within component/direction and agree with array order");
        const auto& dtype = choice(t.at("dtype"), path + ".dtype", {"float32", "float16", "int32", "int64", "bool"});
        const auto& layout = choice(t.at("layout"), path + ".layout", {"NCHW", "NC", "NCL", "NLC", "linear"});
        const auto& shape = array(t.at("shape"), path + ".shape");
        if (shape.size() > 8) invalid(path + ".shape", "rank must be 1..8");
        const std::size_t expected_rank = layout == "NCHW" ? 4 : layout == "NC" ? 2 : layout == "linear" ? 1 : 3;
        if (shape.size() != expected_rank) invalid(path + ".layout", "layout/rank mismatch");
        std::uint64_t elements = 1;
        for (std::size_t d = 0; d < shape.size(); ++d) {
            const auto dimension = number(shape[d], path + ".shape[" + std::to_string(d) + "]");
            if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) invalid(path + ".shape[" + std::to_string(d) + "]", "dimension exceeds native int64 range");
            elements = multiply(elements, dimension, path + ".shape");
        }
        const std::uint64_t bytes_per_element = dtype == "bool" ? 1 : dtype == "float16" ? 2 : dtype == "int64" ? 8 : 4;
        const auto bytes = multiply(elements, bytes_per_element, path + ".shape");
        if (bytes > std::numeric_limits<std::size_t>::max()) invalid(path + ".shape", "byte size exceeds native size_t");
        str(t.at("semantic"), path + ".semantic");
        if (component == "model" && direction == "input") {
            if (!profile_tensor.empty()) invalid(path, "schema 1 supports exactly one model image input");
            profile_tensor = name;
            if (dtype != "float32" || layout != "NCHW" || shape.size() != 4 ||
                number(shape[0], path + ".shape[0]") != 1 || number(shape[1], path + ".shape[1]") != 3 ||
                number(shape[2], path + ".shape[2]") != 640 || number(shape[3], path + ".shape[3]") != 640)
                invalid(path, "model input must be float32 NCHW [1,3,640,640]");
        }
    }
    const auto& profile = object(m.at("input_profile"), "$.input_profile", {"tensor", "batch", "color", "layout", "dtype", "range_min", "range_max", "width", "height"});
    if (profile_tensor.empty() || str(profile.at("tensor"), "$.input_profile.tensor") != profile_tensor) invalid("$.input_profile.tensor", "must name declared model image input");
    equals(profile.at("batch"), "$.input_profile.batch", std::uint64_t{1});
    equals(profile.at("color"), "$.input_profile.color", "RGB");
    equals(profile.at("layout"), "$.input_profile.layout", "CHW");
    equals(profile.at("dtype"), "$.input_profile.dtype", "float32");
    equals(profile.at("range_min"), "$.input_profile.range_min", std::uint64_t{0});
    equals(profile.at("range_max"), "$.input_profile.range_max", std::uint64_t{1});
    equals(profile.at("width"), "$.input_profile.width", std::uint64_t{640});
    equals(profile.at("height"), "$.input_profile.height", std::uint64_t{640});

    const auto& scan = object(m.at("scan"), "$.scan", {"status", "operator", "attributes_status", "attributes", "inputs", "direction", "flatten_order", "initial_state", "state_lifetime", "outputs", "formula_source", "formula_revision", "formula_locator", "semantics"});
    const bool specified = choice(scan.at("status"), "$.scan.status", {"unverified", "specified"}) == "specified";
    const auto& op = object(scan.at("operator"), "$.scan.operator", {"domain", "op_type", "opset", "plugin_name", "plugin_namespace", "plugin_version"});
    // Empty domain/namespace are meaningful. Unverified identity is represented
    // by explicit 'unverified' strings and opset 0, never guessed from Mamba.
    str(op.at("domain"), "$.scan.operator.domain", true);
    str(op.at("plugin_namespace"), "$.scan.operator.plugin_namespace", true);
    string_fields(op, "$.scan.operator", {"op_type", "plugin_name", "plugin_version"});
    const auto opset = number(op.at("opset"), "$.scan.operator.opset");
    const auto& attribute_status = choice(scan.at("attributes_status"), "$.scan.attributes_status", {"unverified", "complete"});
    const auto& attributes = array(scan.at("attributes"), "$.scan.attributes", true);
    std::set<std::string> attribute_names;
    for (std::size_t i = 0; i < attributes.size(); ++i) {
        const auto path = "$.scan.attributes[" + std::to_string(i) + "]";
        const auto& a = object(attributes[i], path, {"name", "type", "value"});
        const auto& name = str(a.at("name"), path + ".name");
        if (!attribute_names.insert(name).second) invalid(path + ".name", "duplicate attribute name");
        const auto& type = choice(a.at("type"), path + ".type", {"unsigned_integer", "boolean", "string", "decimal"});
        if (type == "unsigned_integer") number(a.at("value"), path + ".value");
        else if (type == "boolean") boolean(a.at("value"), path + ".value");
        else if (type == "decimal") decimal(a.at("value"), path + ".value");
        else str(a.at("value"), path + ".value", true);
    }
    const auto& scan_inputs = array(scan.at("inputs"), "$.scan.inputs", !specified);
    for (std::size_t i = 0; i < scan_inputs.size(); ++i) {
        const auto path = "$.scan.inputs[" + std::to_string(i) + "]";
        const auto& item = object(scan_inputs[i], path, {"tensor", "role", "stride", "broadcast"});
        string_fields(item, path, {"tensor", "role", "stride", "broadcast"});
        if (!tensor_names.contains(str(item.at("tensor"), path + ".tensor"))) invalid(path + ".tensor", "unknown tensor name");
    }
    string_fields(scan, "$.scan", {"direction", "flatten_order", "initial_state", "formula_source", "formula_revision", "formula_locator", "semantics"});
    equals(scan.at("state_lifetime"), "$.scan.state_lifetime", "per_frame_reset");
    strings(scan.at("outputs"), "$.scan.outputs", !specified);
    for (const auto& output : array(scan.at("outputs"), "$.scan.outputs", true))
        if (!tensor_names.contains(str(output, "$.scan.outputs"))) invalid("$.scan.outputs", "unknown tensor name");
    if (specified) {
        if (opset == 0 || attribute_status != "complete") invalid("$.scan", "specified scan requires positive opset and complete attributes");
        for (const auto key : {"direction", "flatten_order", "initial_state", "formula_source", "formula_revision", "formula_locator", "semantics"})
            if (str(scan.at(key), "$.scan." + std::string(key)) == "unverified") invalid("$.scan." + std::string(key), "specified scan cannot contain unverified semantics");
        for (const auto key : {"op_type", "plugin_name", "plugin_version"})
            if (str(op.at(key), "$.scan.operator." + std::string(key)) == "unverified") invalid("$.scan.operator." + std::string(key), "specified scan requires identity");
    }

    const auto& precision = object(m.at("precision"), "$.precision", {"compute_dtype", "accumulation_dtype", "tf32", "fp16", "int8", "builder_options", "comparison"});
    choice(precision.at("compute_dtype"), "$.precision.compute_dtype", {"float32", "float16"});
    choice(precision.at("accumulation_dtype"), "$.precision.accumulation_dtype", {"float32", "float64", "unverified"});
    for (const auto key : {"tf32", "fp16", "int8"}) boolean(precision.at(key), "$.precision." + std::string(key));
    if (str(precision.at("compute_dtype"), "$.precision.compute_dtype") == "float16" && !boolean(precision.at("fp16"), "$.precision.fp16"))
        invalid("$.precision.fp16", "float16 compute requires fp16=true");
    if (boolean(precision.at("int8"), "$.precision.int8")) invalid("$.precision.int8", "schema 1 does not support int8 quantization");
    const auto& options = array(precision.at("builder_options"), "$.precision.builder_options", true);
    std::set<std::string> option_names;
    for (std::size_t i = 0; i < options.size(); ++i) {
        const auto path = "$.precision.builder_options[" + std::to_string(i) + "]";
        const auto& option = object(options[i], path, {"name", "value"});
        string_fields(option, path, {"name", "value"});
        if (!option_names.insert(str(option.at("name"), path + ".name")).second) invalid(path + ".name", "duplicate build option");
    }
    const auto& comparison = object(precision.at("comparison"), "$.precision.comparison", {"status", "reference_source", "metric", "atol", "rtol"});
    const bool comparison_specified = choice(comparison.at("status"), "$.precision.comparison.status", {"unverified", "specified"}) == "specified";
    string_fields(comparison, "$.precision.comparison", {"reference_source", "metric"});
    decimal(comparison.at("atol"), "$.precision.comparison.atol");
    decimal(comparison.at("rtol"), "$.precision.comparison.rtol");
    if (comparison_specified && (str(comparison.at("reference_source"), "$.precision.comparison.reference_source") == "unverified" ||
        str(comparison.at("metric"), "$.precision.comparison.metric") == "unverified")) invalid("$.precision.comparison", "specified comparison requires source and metric");
    const auto& review = object(m.at("content_review"), "$.content_review", {"status", "method", "evidence_source", "unverified_fields"});
    const bool reviewed = choice(review.at("status"), "$.content_review.status", {"not_reviewed", "external_review"}) == "external_review";
    string_fields(review, "$.content_review", {"method", "evidence_source"});
    strings(review.at("unverified_fields"), "$.content_review.unverified_fields", reviewed);
    if (result.purpose == "test_fixture" && reviewed) invalid("$.content_review.status", "fixture cannot declare real-model content review");
    result.canonical_json = canonical(root) + '\n';
    return result;
}

std::string model_manifest_hash(const ModelManifest& manifest) { return sha256_hex(manifest.canonical_json); }

void verify_manifest_artifacts(const ModelManifest& manifest, const std::filesystem::path& manifest_path) {
    const auto base = std::filesystem::absolute(manifest_path).parent_path();
    for (const auto& artifact : manifest.artifacts) {
        const auto path = (base / artifact.path).lexically_normal();
        const auto label = "artifact " + artifact.id + " (" + path.string() + ")";
        std::error_code error;
        const auto status = std::filesystem::status(path, error);
        if (error || !std::filesystem::exists(status)) throw std::runtime_error(label + ": missing or inaccessible file");
        if (!std::filesystem::is_regular_file(status)) throw std::runtime_error(label + ": expected regular file");
        const auto size = std::filesystem::file_size(path, error);
        if (error) throw std::runtime_error(label + ": cannot determine file size: " + error.message());
        if (size != artifact.size_bytes) throw std::runtime_error(label + ": size_bytes mismatch; expected " + std::to_string(artifact.size_bytes) + ", got " + std::to_string(size));
        std::string bytes;
        try { bytes = read_text_file(path); }
        catch (const std::exception& failure) { throw std::runtime_error(label + ": " + failure.what()); }
        if (bytes.size() != artifact.size_bytes) throw std::runtime_error(label + ": size changed while reading");
        if (sha256_hex(bytes) != artifact.sha256) throw std::runtime_error(label + ": SHA-256 mismatch");
    }
}
} // namespace saccade

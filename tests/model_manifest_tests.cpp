#include "saccade/model_manifest.hpp"
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t checks = 0;
void check(bool value, std::string_view message) {
    ++checks;
    if (!value) throw std::runtime_error(std::string(message));
}
template <typename F> void reject(F&& action, std::string_view diagnostic) {
    try { action(); }
    catch (const std::exception& error) {
        check(std::string(error.what()).find(diagnostic) != std::string::npos,
              "unexpected diagnostic: " + std::string(error.what()) + "; wanted " + std::string(diagnostic));
        return;
    }
    check(false, "expected rejection: " + std::string(diagnostic));
}
std::string replace(std::string text, std::string_view from, std::string_view to) {
    const auto offset = text.find(from);
    if (offset == std::string::npos) throw std::runtime_error("test replacement not found: " + std::string(from));
    text.replace(offset, from.size(), to);
    return text;
}
struct Temp {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("saccade-manifest-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Temp() { if (!std::filesystem::create_directory(path)) throw std::runtime_error("cannot create temp directory"); }
    ~Temp() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
void tests(const std::filesystem::path& fixture) {
    using namespace saccade;
    const auto text = read_text_file(fixture / "contract.json");
    const auto manifest = parse_model_manifest(text);
    check(manifest.purpose == "test_fixture", "fixture purpose preserved");
    check(manifest.artifacts.size() == 1 && manifest.artifacts[0].size_bytes == 3, "artifact size and number");
    check(model_manifest_hash(manifest) == sha256_hex(manifest.canonical_json), "canonical hash");
    // Independent sha256sum result for the version-controlled fixture. This
    // pins canonical bytes as well as the hash API, not a self-comparison.
    check(model_manifest_hash(manifest) == "97d83da29be2693ba86a210a10c8a5454de8a135b5b31985feb3aee07152b0b7", "known canonical fixture SHA-256");
    check(parse_model_manifest(manifest.canonical_json).canonical_json == manifest.canonical_json, "canonical idempotence");
    check(manifest.canonical_json.back() == '\n' && manifest.canonical_json.find('\n') == manifest.canonical_json.size() - 1,
          "single JSON record with terminal newline");
    const auto reordered = replace(text, "\"schema\": 1,\n  \"purpose\": \"test_fixture\",", "\"purpose\":\"test_fixture\", \"schema\" : 1,");
    check(model_manifest_hash(parse_model_manifest(reordered)) == model_manifest_hash(manifest), "object ordering and insignificant whitespace do not alter hash");
    const auto escaped = replace(text, "\"schema\"", "\"sch\\u0065ma\"");
    check(model_manifest_hash(parse_model_manifest(escaped)) == model_manifest_hash(manifest), "equivalent string escapes normalize");
    check(model_manifest_hash(parse_model_manifest(replace(text, "fixture-v1", "fixture-v2"))) != model_manifest_hash(manifest), "semantic change changes hash");
    verify_manifest_artifacts(manifest, fixture / "contract.json");
    check(true, "actual fixture files verified");

    const auto bad = [&](std::string changed, std::string_view field) { reject([&] { parse_model_manifest(changed); }, field); };
    bad("", "$");
    bad("{}", "$.schema");
    bad(text + "true", "trailing input");
    bad(replace(text, "\"schema\": 1,", "\"schema\": 1, \"schema\": 1,"), "$.schema: duplicate field");
    bad(replace(text, "\"schema\": 1,", "\"schema\": 1, \"sch\\u0065ma\": 1,"), "$.schema: duplicate field");
    bad(replace(text, "\"schema\": 1,", "\"schema\": 1, \"unknown\": 1,"), "$.unknown: unknown field");
    bad(replace(text, "\"schema\": 1,", ""), "$.schema: missing required field");
    bad(replace(text, "\"schema\": 1", "\"schema\": 2"), "$.schema");
    bad(replace(text, "\"schema\": 1", "\"schema\": \"1\""), "$.schema: expected unsigned integer");
    bad(replace(text, "\"schema\": 1", "\"schema\": true"), "$.schema: expected unsigned integer");
    bad(replace(text, "\"schema\": 1", "\"schema\": 1.0"), "expected comma");
    bad(replace(text, "\"schema\": 1", "\"schema\": 01"), "leading zero");
    bad(replace(text, "\"schema\": 1", "\"schema\": -1"), "expected unsigned integer");
    bad(replace(text, "\"schema\": 1", "\"schema\": 18446744073709551616"), "integer out of uint64 range");
    bad(replace(text, "\"id\": \"fixture\",", "\"id\": \"fixture\", \"id\": \"other\","), "$.artifacts[0].id: duplicate field");
    bad(replace(text, "\"id\": \"fixture\",", "\"id\": \"fixture\", \"extra\": false,"), "$.artifacts[0].extra: unknown field");
    bad(replace(text, "\"generated_by\": \"literal ASCII abc, no newline\",", ""), "$.artifacts[0].generated_by: missing required field");
    bad(replace(text, "\"size_bytes\": 3", "\"size_bytes\": \"3\""), "$.artifacts[0].size_bytes: expected unsigned integer");
    bad(replace(text, "ba7816bf", "BA7816BF"), "$.artifacts[0].sha256");
    bad(replace(text, "\"path\": \"bytes.txt\"", "\"path\": \"/bytes.txt\""), "$.artifacts[0].path");
    bad(replace(text, "\"path\": \"bytes.txt\"", "\"path\": \"../bytes.txt\""), "$.artifacts[0].path");
    bad(replace(text, "\"path\": \"bytes.txt\"", "\"path\": \"sub\\\\bytes.txt\""), "$.artifacts[0].path");
    bad(replace(text, "\"purpose\": \"test_fixture\"", "\"purpose\": \"model_candidate\""), "fixture cannot be a model candidate");
    bad(replace(text, "\"artifact\": \"fixture\"", "\"artifact\": \"missing\""), "unknown artifact id");
    bad(replace(text, "\"packaging\": \"incomplete\"", "\"packaging\": \"merged\""), "$.boundary.packaging");
    bad(replace(text, "\"role\": \"head\", \"artifact\": null", "\"role\": \"neck\", \"artifact\": null"), "duplicate component role");
    bad(replace(text, "\"classification_owner\": \"unverified\"", "\"classification_owner\": \"backbone\""), "$.boundary.classification_owner");
    bad(replace(text, "\"name\": \"fixture_output\"", "\"name\": \"images\""), "duplicate tensor name");
    bad(replace(text, "\"order\": 0", "\"order\": 1"), "$.tensors[0].order");
    bad(replace(text, "\"dtype\": \"float32\"", "\"dtype\": \"float64\""), "$.tensors[0].dtype");
    bad(replace(text, "\"layout\": \"NCHW\"", "\"layout\": \"NHWC\""), "$.tensors[0].layout");
    bad(replace(text, "\"shape\": [1,3,640,640]", "\"shape\": []"), "$.tensors[0].shape");
    bad(replace(text, "\"shape\": [1,3,640,640]", "\"shape\": [1,3,0,640]"), "dimension must be positive");
    bad(replace(text, "\"shape\": [1,3,640,640]", "\"shape\": [1,3,-1,640]"), "$.tensors[0].shape[2]");
    bad(replace(text, "\"shape\": [1,3,640,640]", "\"shape\": [1,3,\"640\",640]"), "$.tensors[0].shape[2]");
    bad(replace(text, "\"shape\": [1,3,640,640]", "\"shape\": [1,3,640]"), "layout/rank mismatch");
    bad(replace(text, "\"shape\": [1,3,640,640]", "\"shape\": [1,3,9223372036854775807,2]"), "dimension product or byte size overflow");
    bad(replace(text, "\"shape\": [1,2]", "\"shape\": [1,4611686018427387904]"), "dimension product or byte size overflow");
    bad(replace(text, "\"shape\": [1,3,640,640]", "\"shape\": [1,3,320,320]"), "model input must be");
    bad(replace(text, "\"tensor\": \"images\"", "\"tensor\": \"missing\""), "$.input_profile.tensor");
    bad(replace(text, "\"color\": \"RGB\"", "\"color\": \"BGR\""), "$.input_profile.color");
    bad(replace(text, "\"tf32\": false", "\"tf32\": \"false\""), "$.precision.tf32: expected boolean");
    bad(replace(text, "\"atol\": \"0\"", "\"atol\": \"nan\""), "$.precision.comparison.atol");
    bad(replace(text, "\"atol\": \"0\"", "\"atol\": 0"), "$.precision.comparison.atol: expected string");
    bad(replace(text, "\"atol\": \"0\"", "\"atol\": \"0.0\""), "decimal string must use canonical spelling");
    bad(replace(text, "\"atol\": \"0\"", "\"atol\": \"00\""), "decimal string must use canonical spelling");
    bad(replace(text, "\"state_lifetime\": \"per_frame_reset\"", "\"state_lifetime\": \"persistent\""), "$.scan.state_lifetime");
    bad(replace(text, "\"status\": \"unverified\"", "\"status\": \"specified\""), "$.scan.inputs");
    bad(replace(text, "\"attributes\": []", "\"attributes\": [{\"name\":\"x\",\"type\":\"boolean\",\"value\":0}]"), "$.scan.attributes[0].value");
    bad(replace(text, "\"attributes\": []", "\"attributes\": [{\"name\":\"x\",\"type\":\"boolean\",\"value\":true},{\"name\":\"x\",\"type\":\"boolean\",\"value\":true}]"), "duplicate attribute name");
    bad(replace(text, "\"status\": \"not_reviewed\"", "\"status\": \"external_review\""), "fixture cannot declare real-model content review");
    bad(replace(text, "\"revision\": \"fixture-v1\"", "\"revision\": \"bad\\u0080\""), "only ASCII unicode escapes");
    bad(replace(text, "\"revision\": \"fixture-v1\"", "\"revision\": \"bad\\u0000\""), "printable ASCII");

    bad(replace(text, "\"shape\": [1,2]", "\"shape\": [1,18446744073709551615]"), "dimension exceeds native int64 range");
    bad(replace(text, "\"int8\": false", "\"int8\": true"), "schema 1 does not support int8");
    bad(replace(text, "\"compute_dtype\": \"float32\"", "\"compute_dtype\": \"float16\""), "float16 compute requires");
    const auto incompatible = replace(replace(replace(text, "\"role\": \"fixture\"", "\"role\": \"mamba_fpn_head\""), "\"format\": \"fixture_bytes\"", "\"format\": \"onnx\""), "\"purpose\": \"test_fixture\"", "\"purpose\": \"model_candidate\"");
    bad(incompatible, "artifact role incompatible with component role");
    bad(replace(replace(text, "\"role\": \"fixture\"", "\"role\": \"specification\""), "\"format\": \"fixture_bytes\"", "\"format\": \"text\""), "component must map to model content");
    Temp temp;
    const auto own_manifest = temp.path / "contract.json";
    write_text_file(own_manifest, text);
    reject([&] { verify_manifest_artifacts(manifest, own_manifest); }, "missing or inaccessible file");
    std::filesystem::create_directory(temp.path / "bytes.txt");
    reject([&] { verify_manifest_artifacts(manifest, own_manifest); }, "expected regular file");
    std::filesystem::remove(temp.path / "bytes.txt");
    write_text_file(temp.path / "bytes.txt", "ab");
    reject([&] { verify_manifest_artifacts(manifest, own_manifest); }, "size_bytes mismatch");
    write_text_file(temp.path / "bytes.txt", "abd");
    reject([&] { verify_manifest_artifacts(manifest, own_manifest); }, "SHA-256 mismatch");
    write_text_file(temp.path / "bytes.txt", "abc");
    verify_manifest_artifacts(manifest, own_manifest);
    check(true, "integrity succeeds only with exact bytes");
    const auto old_cwd = std::filesystem::current_path();
    try {
        std::filesystem::current_path(std::filesystem::temp_directory_path());
        verify_manifest_artifacts(manifest, own_manifest);
        std::filesystem::current_path(old_cwd);
    } catch (...) { std::filesystem::current_path(old_cwd); throw; }
    check(true, "absolute manifest path anchors relative artifact independent of cwd");
#if defined(__linux__)
    // /proc/self/mem is a regular file with nominal size 0; an ordinary read at
    // offset zero fails with EIO, even under root. This exercises the actual
    // checked-read failure after regular-file and size checks, without relying
    // on filesystem permission bits which root bypasses.
    if (std::filesystem::exists("/proc/self/mem")) {
        std::filesystem::create_symlink("/proc/self/mem", temp.path / "unreadable");
        auto unreadable = parse_model_manifest(replace(replace(text, "\"path\": \"bytes.txt\"", "\"path\": \"unreadable\""), "\"size_bytes\": 3", "\"size_bytes\": 0"));
        reject([&] { verify_manifest_artifacts(unreadable, own_manifest); }, "cannot read input");
    }
#endif
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("fixture directory required");
        tests(std::filesystem::absolute(argv[1]));
        std::cout << "model_manifest_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_manifest_tests: " << error.what() << '\n';
        return 1;
    }
}

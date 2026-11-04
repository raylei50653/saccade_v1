#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace saccade {

struct ManifestArtifact {
    std::string id;
    std::string path;
    std::uint64_t size_bytes;
    std::string sha256;
};

// A validated schema-1 (s-period image model) or schema-2 (S04 head-only) declaration. Actual model contents are not inspected by
// this module. A successful integrity check cannot confer model readiness.
struct ModelManifest {
    std::uint64_t schema = 1;
    std::string canonical_json;
    std::string purpose;
    std::vector<ManifestArtifact> artifacts;
};

// Strict schema, including nested fields; throws diagnostics with field paths.
// Parsing/canonicalization is independent of filesystem or execution directory.
ModelManifest parse_model_manifest(std::string_view input);
std::string model_manifest_hash(const ModelManifest& manifest);

// All artifact paths must be relative and resolve against the manifest's parent
// directory. Manifest path is made absolute exactly once by this operation.
// Reads through checked I/O and rejects missing, unreadable, size/hash mismatch.
void verify_manifest_artifacts(const ModelManifest& manifest,
                               const std::filesystem::path& manifest_path);

} // namespace saccade

#pragma once

#include "saccade/file_io.hpp"
#include "saccade/resolved_config.hpp"
#include "saccade/sha256.hpp"

#include <filesystem>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include <fcntl.h>
#include <unistd.h>

// Process-environment access belongs to CLI resolution, never tracker stages.
extern char** environ;

namespace saccade {

inline ConfigAssignments declared_tracking_environment() {
    std::set<std::string> declared;
    for (const auto& field : tracking_config_schema())
        if (!field.environment.empty()) declared.insert(field.environment);
    ConfigAssignments assignments;
    for (auto** entry = environ; entry && *entry; ++entry) {
        const std::string_view text(*entry);
        if (!text.starts_with("SACCADE_CFG_")) continue;
        const auto equals = text.find('=');
        const std::string name(text.substr(0, equals));
        if (equals == std::string_view::npos || !declared.contains(name))
            throw std::invalid_argument("undeclared tracking environment variable: " + name);
        assignments.emplace_back(name, text.substr(equals + 1));
    }
    return assignments;
}

inline std::string tracking_build_identity(const std::filesystem::path& engine = {}) {
    std::string identity = "build.schema=saccade-build-identity/1\n";
#ifdef SACCADE_BUILD_COMMIT
    identity += "build.source_commit=" SACCADE_BUILD_COMMIT "\n";
#else
    identity += "build.source_commit=unknown\n";
#endif
#ifdef SACCADE_BUILD_DIRTY
    identity += "build.source_dirty=" SACCADE_BUILD_DIRTY "\n";
#else
    identity += "build.source_dirty=unknown\n";
#endif
    identity += "build.executable_sha256=" + sha256_hex(read_text_file("/proc/self/exe")) + '\n';
    if (!engine.empty()) identity += "build.engine_sha256=" + sha256_hex(read_text_file(engine)) + '\n';
    return identity;
}

inline std::pair<std::string, std::string> tracking_cli_assignment(std::string_view text) {
    const auto equals = text.find('=');
    if (equals == std::string_view::npos || equals == 0 || equals + 1 == text.size())
        throw std::invalid_argument("--set requires KEY=VALUE");
    return {std::string(text.substr(0, equals)), std::string(text.substr(equals + 1))};
}

inline void write_exclusive_output_file(const std::filesystem::path& path,
                                        std::span<const std::uint8_t> bytes, std::string_view description) {
    const int descriptor = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (descriptor < 0) throw std::runtime_error(std::string(description) + " open failed: " + std::string(std::strerror(errno)));
    auto* output = ::fdopen(descriptor, "wb");
    if (!output) {
        (void)::close(descriptor);
        throw std::runtime_error(std::string(description) + " fdopen failed");
    }
    const bool written = bytes.empty() || std::fwrite(bytes.data(), 1, bytes.size(), output) == bytes.size();
    const bool closed = std::fclose(output) == 0;
    if (!written || !closed) throw std::runtime_error(std::string(description) + " write/close failed");
}
inline void write_exclusive_binary_file(const std::filesystem::path& path, std::span<const std::uint8_t> bytes) {
    write_exclusive_output_file(path, bytes, "diagnostic output");
}
inline void write_resolved_config_file(const std::filesystem::path& path, std::string_view text) {
    write_exclusive_output_file(path, {reinterpret_cast<const std::uint8_t*>(text.data()), text.size()}, "resolved config");
}

} // namespace saccade

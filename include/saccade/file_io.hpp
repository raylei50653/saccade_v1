#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace saccade {

// Byte-preserving reads; empty files are valid, read/open errors throw.
std::string read_text_file(const std::filesystem::path& path);

// Every write explicitly closes and checks the stream, detecting write/close
// errors without promising fsync durability. Empty outputs are valid; writing
// no bytes to a device may also succeed when the OS reports no error.
void write_text_file(const std::filesystem::path& path, std::string_view text);
void write_binary_file(const std::filesystem::path& path, std::span<const std::uint8_t> bytes);
void append_text_file(const std::filesystem::path& path, std::string_view text);

} // namespace saccade

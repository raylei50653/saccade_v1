#include "saccade/file_io.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <stdexcept>

namespace saccade {
namespace {
[[noreturn]] void failed(std::string_view action, const std::filesystem::path& path) {
    throw std::runtime_error(std::string(action) + ": " + path.string());
}
void write_file(const std::filesystem::path& path, std::string_view bytes, bool append) {
    std::ofstream output(path, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
    if (!output.is_open()) failed("cannot open output", path);
    // A bounded chunk avoids narrowing lengths larger than streamsize. Check
    // close as well: a small buffered write can fail only when it is flushed.
    constexpr std::size_t chunk_bytes = 65536;
    while (!bytes.empty()) {
        const auto count = std::min(bytes.size(), chunk_bytes);
        output.write(bytes.data(), static_cast<std::streamsize>(count));
        if (!output) break;
        bytes.remove_prefix(count);
    }
    output.close();
    if (!output) failed("cannot write or close output", path);
}
} // namespace

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) failed("cannot open input", path);
    std::string contents;
    std::array<char, 8192> chunk{};
    for (;;) {
        input.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        if (input.bad()) failed("cannot read input", path);
        const auto count = input.gcount();
        if (count > 0) contents.append(chunk.data(), static_cast<std::size_t>(count));
        if (input.eof()) break;
        if (input.fail()) failed("cannot read input", path);
    }
    // EOF is expected, but must not mask an error reported by close.
    input.clear();
    input.close();
    if (!input) failed("cannot close input", path);
    return contents;
}
void write_text_file(const std::filesystem::path& path, std::string_view text) {
    write_file(path, text, false);
}
void write_binary_file(const std::filesystem::path& path, std::span<const std::uint8_t> bytes) {
    const std::string_view contents = bytes.empty() ? std::string_view{} :
        std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    write_file(path, contents, false);
}
void append_text_file(const std::filesystem::path& path, std::string_view text) {
    write_file(path, text, true);
}

} // namespace saccade

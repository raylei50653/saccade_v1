#include "saccade/file_io.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
std::size_t checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template <typename F> void reject(F&& action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::runtime_error&) { rejected = true; }
    check(rejected, message);
}
struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("saccade-file-io-" + std::to_string(nonce));
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("cannot create temporary directory");
    }
    ~TemporaryDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};
void tests() {
    using namespace saccade;
    TemporaryDirectory temp;
    const auto file = temp.path / "text.cfg";
    write_text_file(file, "");
    check(read_text_file(file).empty(), "legitimate empty file reads as empty");
    write_text_file(file, "frames=16\nseed=7\n");
    check(read_text_file(file) == "frames=16\nseed=7\n", "regular text contents preserved");
    write_text_file(file, "x");
    check(read_text_file(file) == "x", "write replaces rather than appends");
    append_text_file(file, "yz\n");
    check(read_text_file(file) == "xyz\n", "append preserves previous contents");
    append_text_file(file, "");
    check(read_text_file(file) == "xyz\n", "empty append preserves previous contents");
    const auto append = temp.path / "new-append.txt";
    append_text_file(append, "first");
    check(read_text_file(append) == "first", "append creates new file");
    const auto empty_packets = temp.path / "empty-packets.bin";
    write_binary_file(empty_packets, {});
    check(std::filesystem::file_size(empty_packets) == 0 && read_text_file(empty_packets).empty(),
          "empty raw packet stream is a valid file");
    const std::array<std::uint8_t, 5> bytes{0, 1, 128, 255, 10};
    const auto binary = temp.path / "packets.bin";
    write_binary_file(binary, bytes);
    const std::string expected{static_cast<char>(0), static_cast<char>(1), static_cast<char>(128),
                               static_cast<char>(255), static_cast<char>(10)};
    check(read_text_file(binary) == expected, "binary bytes including NUL preserved");
    // Exercise exact chunk boundaries and multiple read/write iterations.
    for (const auto size : {8191, 8192, 8193, 65536, 65537}) {
        std::string payload(static_cast<std::size_t>(size), 'a');
        payload.front() = '\0'; payload.back() = static_cast<char>(255);
        write_text_file(file, payload);
        check(read_text_file(file) == payload, "chunked write/read boundary preserved");
    }
    reject([&] { read_text_file(temp.path); }, "directory must not parse as empty input");
    reject([&] { read_text_file(temp.path / "missing.cfg"); }, "missing input rejected");
    reject([&] { write_text_file(temp.path / "missing" / "file", "x"); }, "text open failure rejected");
    reject([&] { write_binary_file(temp.path / "missing" / "file", bytes); }, "binary open failure rejected");
    reject([&] { write_binary_file(temp.path / "missing" / "file", {}); }, "empty binary open failure rejected");
    reject([&] { append_text_file(temp.path, "x"); }, "append open failure rejected");
    reject([&] { write_text_file(temp.path, "x"); }, "directory output rejected");
    // Small writes remain buffered on ordinary libstdc++ streams. /dev/full
    // catches the write/flush error that appears only at explicit close.
    if (std::filesystem::exists("/dev/full")) {
        reject([] { write_text_file("/dev/full", "x"); }, "small text close/flush failure rejected");
        reject([&] { write_binary_file("/dev/full", bytes); }, "small binary close/flush failure rejected");
        reject([] { append_text_file("/dev/full", "x"); }, "small append close/flush failure rejected");
    }
}
} // namespace
int main() {
    try {
        tests();
        std::cout << "file_io_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "file_io_tests: " << error.what() << '\n';
        return 1;
    }
}

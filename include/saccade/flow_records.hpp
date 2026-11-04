#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace saccade::flow {
struct Vector { float dx = 0, dy = 0; bool operator==(const Vector&) const = default; };
// Numerical values match the API 2.0 NV_OF_PERF_LEVEL enum.
enum class Perf : std::int32_t { slow = 5, medium = 10, fast = 20 };
enum class Status : std::uint32_t { uninitialized = 0, valid = 1 };
struct Frame {
    std::int64_t frame = 0;
    Status status = Status::uninitialized;
    std::vector<Vector> cells;
};
struct Records {
    std::int32_t width = 0, height = 0, cell = 8, cols = 0, rows = 0;
    Perf perf = Perf::slow;
    std::vector<Frame> frames;
};
// Four S10.5 grid vectors -> one componentwise median cell, in pixels.
Vector reduce_cell(const std::array<std::array<std::int16_t, 2>, 4>& grid);
void write_records(const std::filesystem::path& path, const Records& records);
Records read_records(const std::filesystem::path& path);
} // namespace saccade::flow

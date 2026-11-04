#include "saccade/flow_records.hpp"

#include <algorithm>
#include <bit>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace saccade::flow {
namespace {
template<class T> void put(std::ostream& out, T value) {
    using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    const auto bits = std::bit_cast<U>(value);
    for (std::size_t i = 0; i < sizeof(T); ++i) out.put(static_cast<char>((bits >> (8 * i)) & 255));
}
template<class T> T get(std::istream& in) {
    using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    U bits = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        const auto byte = in.get();
        if (byte == std::char_traits<char>::eof()) throw std::runtime_error("truncated SCFL");
        bits |= static_cast<U>(static_cast<unsigned char>(byte)) << (8 * i);
    }
    return std::bit_cast<T>(bits);
}
std::size_t cells(const Records& r) {
    if (r.width < 8 || r.height < 8 || r.cell != 8 || r.cols != r.width / 8 || r.rows != r.height / 8)
        throw std::runtime_error("wrong SCFL dimensions");
    return static_cast<std::size_t>(r.cols) * r.rows;
}
}
Vector reduce_cell(const std::array<std::array<std::int16_t, 2>, 4>& grid) {
    std::array<int, 4> x{}, y{};
    for (int i = 0; i < 4; ++i) { x[i] = grid[i][0]; y[i] = grid[i][1]; }
    std::sort(x.begin(), x.end()); std::sort(y.begin(), y.end());
    return {float(x[1] + x[2]) / 64.f, float(y[1] + y[2]) / 64.f};
}
void write_records(const std::filesystem::path& path, const Records& r) {
    const auto n = cells(r);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot open SCFL output");
    out.write("SCFL", 4); put(out, std::uint32_t{1});
    for (auto v : {r.width, r.height, r.cell, r.cols, r.rows, static_cast<std::int32_t>(r.perf)}) put(out, v);
    put(out, static_cast<std::uint64_t>(r.frames.size()));
    for (const auto& f : r.frames) {
        if (f.cells.size() != n) throw std::runtime_error("wrong SCFL cell count");
        put(out, f.frame); put(out, static_cast<std::uint32_t>(f.status));
        for (const auto& v : f.cells) { put(out, v.dx); put(out, v.dy); }
    }
    out.close();
    if (!out) throw std::runtime_error("cannot write/close SCFL output");
}
Records read_records(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open SCFL input");
    std::array<char, 4> magic{};
    if (!in.read(magic.data(), 4)) throw std::runtime_error("truncated SCFL");
    if (magic != std::array<char, 4>{'S','C','F','L'}) throw std::runtime_error("unknown SCFL magic");
    if (get<std::uint32_t>(in) != 1) throw std::runtime_error("unknown SCFL version");
    Records r;
    r.width = get<std::int32_t>(in); r.height = get<std::int32_t>(in); r.cell = get<std::int32_t>(in);
    r.cols = get<std::int32_t>(in); r.rows = get<std::int32_t>(in); r.perf = static_cast<Perf>(get<std::int32_t>(in));
    const auto count = get<std::uint64_t>(in), n = cells(r);
    const auto stride = 12 + 8 * static_cast<std::uint64_t>(n);
    if (std::filesystem::file_size(path) < 40 || count != (std::filesystem::file_size(path) - 40) / stride ||
        (std::filesystem::file_size(path) - 40) % stride != 0) throw std::runtime_error("wrong SCFL size");
    r.frames.reserve(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        Frame f; f.frame = get<std::int64_t>(in);
        const auto status = get<std::uint32_t>(in);
        if (status > 1) throw std::runtime_error("unknown SCFL status");
        f.status = static_cast<Status>(status); f.cells.resize(n);
        for (auto& v : f.cells) { v.dx = get<float>(in); v.dy = get<float>(in); }
        r.frames.push_back(std::move(f));
    }
    return r;
}
} // namespace saccade::flow

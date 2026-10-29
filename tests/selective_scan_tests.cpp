#include "saccade/selective_scan.hpp"
#include "saccade/file_io.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Expected values come from fixtures/selective-scan/expected.tsv, derived by
// bc from closed forms or explicit sums (derive.sh), or from hand derivations
// in fixtures/selective-scan/README.md. The reference never produces its own
// expected values; equality cases compare two independent reference calls whose
// contract says the results must be identical.
namespace {
using saccade::SelectiveScanInputs;
using saccade::SelectiveScanShape;

std::size_t checks = 0;
std::ostringstream report;

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

// Fixed before execution (docs/SCAN_CONTRACT.md §5): |actual - expected| <=
// atol + rtol * |expected|, both binary64 error budgets with margin.
struct Tolerance { long double atol, rtol; const char* name; };
constexpr Tolerance unit{1e-10L, 1e-10L, "unit"};
constexpr Tolerance relative{0.0L, 1e-10L, "relative"};

using Table = std::map<std::string, std::vector<long double>>;
Table load_expected(const std::filesystem::path& path) {
    Table table;
    std::istringstream lines(saccade::read_text_file(path));
    std::string line;
    while (std::getline(lines, line)) {
        std::istringstream fields(line);
        std::string name, index, value, exp2;
        if (!std::getline(fields, name, '\t') || !std::getline(fields, index, '\t') ||
            !std::getline(fields, value, '\t') || !std::getline(fields, exp2) )
            throw std::runtime_error("malformed expected row: " + line);
        auto& column = table[name];
        if (std::stoul(index) != column.size()) throw std::runtime_error("non-consecutive expected index: " + line);
        char* end = nullptr;
        const long double v = std::strtold(value.c_str(), &end);
        if (end == value.c_str() || *end != '\0') throw std::runtime_error("bad expected value: " + line);
        column.push_back(std::ldexp(v, std::stoi(exp2)));
    }
    return table;
}

void compare(const std::string& name, const std::vector<double>& actual, const std::vector<long double>& expected, Tolerance tol) {
    check(actual.size() == expected.size(), name + ": output count matches expected count");
    long double max_abs = 0, max_rel = 0;
    std::size_t worst = 0;
    bool pass = true;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const long double err = std::fabs(static_cast<long double>(actual[i]) - expected[i]);
        const long double rel = expected[i] == 0 ? err : err / std::fabs(expected[i]);
        if (err > max_abs) { max_abs = err; worst = i; }
        if (rel > max_rel) max_rel = rel;
        if (!(err <= tol.atol + tol.rtol * std::fabs(expected[i]))) pass = false;
        if (actual.size() <= 32)
            report << "element\t" << name << '\t' << i << '\t' << std::setprecision(21) << expected[i] << '\t'
                   << std::setprecision(17) << actual[i] << '\t' << std::setprecision(3) << err << '\n';
    }
    report << "case\t" << name << '\t' << tol.name << "\tcount=" << actual.size() << std::setprecision(3)
           << "\tmax_abs=" << max_abs << "\tmax_rel=" << max_rel << "\tworst_index=" << worst
           << "\t" << (pass ? "PASS" : "FAIL") << '\n';
    check(pass, name + ": within " + tol.name + " tolerance");
}

struct Case {
    SelectiveScanShape shape;
    std::vector<float> u, delta, a, b, c, d;
    SelectiveScanInputs inputs() const { return {u, delta, a, b, c, d}; }
    std::vector<double> run() const { return saccade::selective_scan_reference(shape, inputs()); }
};

// Shapes are filled in (batch, length, channels, state, a_rows, c_width, has_d) order.
Case filled(SelectiveScanShape s, float u, float delta, float b, float c) {
    const auto tokens = s.batch * s.length;
    return {s, std::vector<float>(tokens * s.channels, u), std::vector<float>(tokens * s.channels, delta),
            std::vector<float>(s.a_rows * s.state, -1.0f), std::vector<float>(tokens * s.state, b),
            std::vector<float>(tokens * s.c_width, c), std::vector<float>(s.has_d ? s.channels : 0, 0.0f)};
}

float gen(int a, int m, int offset, float scale) { return static_cast<float>(a % m - offset) / scale; }

void contract_values(const Table& expected) {
    {
        Case k{{1, 1, 2, 2, 1, 2, true}, {2, -4}, {0, 21}, {-1, -2}, {0.5f, 0.75f}, {1, 2}, {0.5f, -1}};
        compare("single_token", k.run(), expected.at("single_token"), unit);
    }
    {
        Case k = filled({1, 3, 1, 1, 1, 1, false}, 0, 0, 1, 1);
        k.u = {1, 2, 3};
        compare("three_token_decay", k.run(), expected.at("three_token_decay"), unit);
    }
    {
        Case k = filled({1, 1, 6, 1, 1, 1, false}, 1, 0, 1, 1);
        k.delta = {-30, -20, -1, 0, 20, 20.5f};
        compare("softplus_branches", k.run(), expected.at("softplus_branches"), relative);
    }
    {
        Case k = filled({1, 400, 4, 16, 1, 1, true}, 0, 0, 1, 1);
        for (std::size_t t = 0; t < 400; ++t)
            for (std::size_t d = 0; d < 4; ++d) k.u[t * 4 + d] = static_cast<float>(d + 1) / 4;
        for (std::size_t n = 0; n < 16; ++n) k.a[n] = -static_cast<float>(n + 1);
        k.d.assign(4, 0.5f);
        compare("geometric_p3_length", k.run(), expected.at("geometric_p3_length"), unit);
    }
    {
        Case k = filled({1, 25, 3, 2, 3, 2, true}, 0, 0, 0, 0);
        k.a = {-1, -2, -0.5f, -4, 0, -1};
        k.d = {0, 0.25f, -1};
        const float u[3] = {1, -2, 0.5f}, b[2] = {1, -0.5f}, c[2] = {2, 1};
        for (std::size_t t = 0; t < 25; ++t) {
            for (std::size_t d = 0; d < 3; ++d) k.u[t * 3 + d] = u[d];
            for (std::size_t n = 0; n < 2; ++n) { k.b[t * 2 + n] = b[n]; k.c[t * 2 + n] = c[n]; }
        }
        compare("per_channel_a", k.run(), expected.at("per_channel_a"), unit);
    }
    {
        Case k = filled({1, 100, 2, 32, 1, 1, false}, 1, 0, 1, 0.5f);
        for (std::size_t n = 0; n < 32; ++n) k.a[n] = -static_cast<float>(n + 1) / 8;
        for (std::size_t t = 0; t < 100; ++t) k.u[t * 2 + 1] = -1;
        compare("state_32_p4_length", k.run(), expected.at("state_32_p4_length"), unit);
    }
    {
        Case k = filled({2, 7, 3, 4, 1, 4, true}, 0, 0, 0, 0);
        k.a = {-0.5f, -1, -2, -3};
        k.d = {-0.5f, 0, 0.5f};
        for (int b = 0; b < 2; ++b)
            for (int t = 0; t < 7; ++t) {
                const auto token = static_cast<std::size_t>(b * 7 + t);
                for (int d = 0; d < 3; ++d) {
                    k.delta[token * 3 + d] = gen(7 * b + 3 * t + 5 * d, 13, 6, 2);
                    k.u[token * 3 + d] = gen(5 * b + 7 * t + 3 * d, 11, 5, 4);
                }
                for (int n = 0; n < 4; ++n) {
                    k.b[token * 4 + n] = gen(3 * b + 5 * t + 7 * n, 9, 4, 4);
                    k.c[token * 4 + n] = gen(7 * b + 3 * t + 5 * n, 7, 3, 2);
                }
            }
        compare("varying_direct_sum", k.run(), expected.at("varying_direct_sum"), unit);
    }
    // The m head's three scan nodes: 4 directions x 1 image, d_inner 256,
    // N 16, shared A, rank-1 C, D present; L = 25 / 100 / 400.
    const auto& k_table = expected.at("m_head_k");
    for (const std::size_t length : {25u, 100u, 400u}) {
        Case k = filled({4, length, 256, 16, 1, 1, true}, 0, 0, 0, 0);
        for (std::size_t n = 0; n < 16; ++n) k.a[n] = -static_cast<float>(n + 1);
        k.d.resize(256);
        for (int d = 0; d < 256; ++d) k.d[d] = gen(7 * d, 5, 2, 4);
        std::vector<long double> want(k.u.size());
        for (int b = 0; b < 4; ++b)
            for (std::size_t t = 0; t < length; ++t) {
                const auto token = static_cast<std::size_t>(b) * length + t;
                k.c[token] = static_cast<float>(b + 1) / 2;
                for (std::size_t n = 0; n < 16; ++n) k.b[token * 16 + n] = static_cast<float>(n % 4 + 1) / 4;
                for (int d = 0; d < 256; ++d) {
                    const float u = gen(37 * b + 11 * d, 17, 8, 8);
                    k.u[token * 256 + d] = u;
                    want[token * 256 + d] = static_cast<long double>(k.c[token]) * u * k_table.at(t) +
                                            static_cast<long double>(k.d[d]) * u;
                }
            }
        compare("m_head_abi_L" + std::to_string(length), k.run(), want, unit);
    }
    const float big = std::ldexp(1.0f, 100), small = std::ldexp(1.0f, -100), top = std::ldexp(1.0f, 127);
    {
        Case k = filled({1, 3, 1, 1, 1, 1, false}, 0, big, small, 1);
        k.u = {3, -5, 7};
        compare("extreme_large_delta_memoryless", k.run(), expected.at("extreme_large_delta_memoryless"), relative);
    }
    {
        Case k = filled({1, 2, 1, 1, 1, 1, true}, 0, -big, 1, 1);
        k.u = {big, -big};
        k.d = {small};
        compare("extreme_negative_delta", k.run(), expected.at("extreme_negative_delta"), relative);
    }
    {
        Case k = filled({1, 1, 1, 1, 1, 1, false}, top, -100, 1, 1);
        compare("extreme_fp32_max_input", k.run(), expected.at("extreme_fp32_max_input"), relative);
    }
    {
        Case k = filled({1, 1, 1, 1, 1, 1, false}, top, 21, std::ldexp(1.0f, -5), 1);
        compare("extreme_near_fp32_limit_output", k.run(), expected.at("extreme_near_fp32_limit_output"), relative);
    }
    {
        Case k = filled({1, 2, 1, 1, 1, 1, false}, std::numeric_limits<float>::denorm_min(), 0, 1, 1);
        compare("extreme_subnormal_input", k.run(), expected.at("extreme_subnormal_input"), relative);
    }
    {
        Case k = filled({1, 3, 1, 1, 1, 1, false}, 0, 0, 1, 1);
        k.u = {1, 2, 3};
        k.a = {-std::numeric_limits<float>::max()};
        compare("extreme_most_negative_a", k.run(), expected.at("extreme_most_negative_a"), unit);
    }
}

// Deterministic non-degenerate inputs for the equality cases.
Case varied(SelectiveScanShape s) {
    Case k = filled(s, 0, 0, 0, 0);
    for (std::size_t i = 0; i < k.u.size(); ++i) {
        k.u[i] = gen(static_cast<int>(5 * i + 1), 11, 5, 4);
        k.delta[i] = gen(static_cast<int>(3 * i + 2), 13, 6, 2);
    }
    for (std::size_t i = 0; i < k.b.size(); ++i) k.b[i] = gen(static_cast<int>(7 * i + 3), 9, 4, 4);
    for (std::size_t i = 0; i < k.c.size(); ++i) k.c[i] = gen(static_cast<int>(3 * i + 1), 7, 3, 2);
    for (std::size_t i = 0; i < k.a.size(); ++i) k.a[i] = -static_cast<float>(i % s.state + 1) / 4;
    for (std::size_t i = 0; i < k.d.size(); ++i) k.d[i] = gen(static_cast<int>(i), 3, 1, 2);
    return k;
}

void equalities() {
    const SelectiveScanShape base{2, 9, 3, 8, 1, 1, true};
    const Case rank1 = varied(base);
    Case full = rank1;
    full.shape.c_width = 8;
    full.c.clear();
    for (const float c : rank1.c) full.c.insert(full.c.end(), 8, c);
    check(rank1.run() == full.run(), "rank-1 C equals the same value broadcast to every state");

    Case rows = rank1;
    rows.shape.a_rows = 3;
    rows.a.clear();
    for (int d = 0; d < 3; ++d) rows.a.insert(rows.a.end(), rank1.a.begin(), rank1.a.end());
    check(rank1.run() == rows.run(), "shared A equals per-channel A with identical rows");

    Case single = varied({1, 9, 3, 8, 1, 1, true});
    Case pair = single;
    pair.shape.batch = 2;
    // Row 0 drives large, long-lived state; row 1 must not observe it.
    pair.u.insert(pair.u.begin(), 27, 100.0f);
    pair.delta.insert(pair.delta.begin(), 27, 5.0f);
    pair.b.insert(pair.b.begin(), 72, 1.0f);
    pair.c.insert(pair.c.begin(), 9, 1.0f);
    pair.a.assign(8, -0.001f);
    single.a = pair.a;
    const auto paired = pair.run(), alone = single.run();
    check(std::vector<double>(paired.begin() + 27, paired.end()) == alone, "batch rows have independent zero-initialized state");

    const auto first = rank1.run();
    (void)pair.run();
    check(rank1.run() == first, "calls are reset: no state survives between invocations");

    Case no_d = rank1;
    no_d.shape.has_d = false;
    no_d.d.clear();
    Case zero_d = rank1;
    zero_d.d.assign(3, 0.0f);
    check(no_d.run() == zero_d.run(), "absent D equals zero D");

    Case empty = varied({2, 0, 3, 4, 1, 1, true});
    check(empty.u.empty() && empty.run().empty(), "zero-length sequence is accepted and yields an empty output");
    report << "equality\tall\tPASS\n";
}

void rejections() {
    const auto bad = [](Case k, std::string_view diagnostic) { reject([&] { (void)k.run(); }, diagnostic); };
    const Case ok = varied({1, 4, 4, 16, 1, 1, true});
    for (const std::size_t n : {0u, 3u, 64u}) {
        Case k = ok;
        k.shape.state = n;
        bad(k, "must be a power of two in [1, 32]");
    }
    { Case k = ok; k.shape.a_rows = 2; k.a.resize(32); bad(k, "A rows must be 1 or channels"); }
    { Case k = ok; k.shape.c_width = 4; k.c.resize(16); bad(k, "C width must be 1 or state"); }
    { Case k = ok; k.shape.batch = 0; bad(k, "batch and channels must be positive"); }
    { Case k = ok; k.shape.channels = 0; bad(k, "batch and channels must be positive"); }
    { Case k = ok; k.u.pop_back(); bad(k, "u has 15 values, shape needs 16"); }
    { Case k = ok; k.delta.push_back(0); bad(k, "delta has 17 values"); }
    { Case k = ok; k.a.pop_back(); bad(k, "A has 15 values"); }
    { Case k = ok; k.b.pop_back(); bad(k, "B has 63 values"); }
    { Case k = ok; k.c.push_back(0); bad(k, "C has 5 values"); }
    { Case k = ok; k.d.pop_back(); bad(k, "D has 3 values"); }
    { Case k = ok; k.shape.has_d = false; bad(k, "D has 4 values, shape needs 0"); }
    // Capacity is checked from the declared shape before any data is touched.
    { Case k{{1, 65536, 65536, 1, 1, 1, false}, {}, {}, {}, {}, {}, {}}; bad(k, "u/delta/y element count exceeds int32 capacity"); }
    { Case k{{1, std::size_t{1} << 27, 1, 32, 1, 1, false}, {}, {}, {}, {}, {}, {}}; bad(k, "B element count exceeds int32 capacity"); }
    { Case k{{2, std::numeric_limits<std::size_t>::max() / 2, 1, 1, 1, 1, false}, {}, {}, {}, {}, {}, {}}; bad(k, "exceeds int32 capacity"); }
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    { Case k = ok; k.u[5] = nan; bad(k, "u[5] is not finite"); }
    { Case k = ok; k.delta[0] = inf; bad(k, "delta[0] is not finite"); }
    { Case k = ok; k.a[3] = -inf; bad(k, "A[3] is not finite"); }
    { Case k = ok; k.b[63] = nan; bad(k, "B[63] is not finite"); }
    { Case k = ok; k.c[2] = inf; bad(k, "C[2] is not finite"); }
    { Case k = ok; k.d[1] = nan; bad(k, "D[1] is not finite"); }
    const float top = std::ldexp(1.0f, 127);
    { Case k = filled({1, 1, 1, 1, 1, 1, false}, top, 21, 4, 1); bad(k, "drive leaves FP32 range at batch 0, token 0"); }
    { Case k = filled({1, 1, 1, 1, 1, 1, false}, 1, 21, 1, 1); k.a = {100}; bad(k, "decay leaves FP32 range"); }
    { Case k = filled({1, 200, 1, 1, 1, 1, false}, 1, 0, 1, 1); k.a = {1}; bad(k, "state leaves FP32 range"); }
    { Case k = filled({1, 2, 1, 1, 1, 1, false}, 1, 21, std::ldexp(1.0f, 120), top); bad(k, "C*state leaves FP32 range"); }
    { Case k = filled({1, 1, 1, 1, 1, 1, true}, top, -100, 1, 1); k.d = {4}; bad(k, "D*u leaves FP32 range"); }
    { Case k = filled({1, 1, 1, 4, 1, 1, false}, top, 21, std::ldexp(1.0f, -5), 1); bad(k, "output leaves FP32 range"); }
    report << "rejection\tall\tPASS\n";
}

void cross_scan(const Table& expected) {
    using saccade::cross_scan_expand;
    using saccade::cross_scan_merge;
    using saccade::cross_scan_source;
    // x[c, h, w] = 100 c + 10 h + w; hand-listed token order per direction.
    std::vector<float> x(12);
    for (int c = 0; c < 2; ++c)
        for (int h = 0; h < 2; ++h)
            for (int w = 0; w < 3; ++w) x[(c * 2 + h) * 3 + w] = static_cast<float>(100 * c + 10 * h + w);
    const int order[4][6] = {{0, 1, 2, 10, 11, 12}, {12, 11, 10, 2, 1, 0}, {2, 1, 0, 12, 11, 10}, {10, 11, 12, 0, 1, 2}};
    const auto tokens = cross_scan_expand(x, 1, 2, 2, 3);
    check(tokens.size() == 48, "expanded tokens: 4 directions x 6 tokens x 2 channels");
    for (int k = 0; k < 4; ++k)
        for (int t = 0; t < 6; ++t)
            for (int c = 0; c < 2; ++c)
                check(tokens[(k * 6 + t) * 2 + c] == static_cast<float>(100 * c + order[k][t]), "direction token order");
    const std::size_t row[4][4] = {{0, 1, 2, 3}, {3, 2, 1, 0}, {3, 2, 1, 0}, {0, 1, 2, 3}};
    const std::size_t column[4][3] = {{0, 1, 2}, {2, 1, 0}, {0, 1, 2}, {2, 1, 0}};
    for (std::size_t k = 0; k < 4; ++k) {
        for (std::size_t t = 0; t < 4; ++t) check(cross_scan_source(k, t, 1, 4) == row[k][t], "single-row direction order");
        for (std::size_t t = 0; t < 3; ++t) check(cross_scan_source(k, t, 3, 1) == column[k][t], "single-column direction order");
    }
    std::vector<float> two(x);
    for (const float v : x) two.push_back(v + 1000);
    const auto batched = cross_scan_expand(two, 2, 2, 2, 3);
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 12; ++i) {
            check(batched[(k * 2) * 12 + i] == tokens[k * 12 + i], "direction-major rows: image 0");
            check(batched[(k * 2 + 1) * 12 + i] == tokens[k * 12 + i] + 1000, "direction-major rows: image 1");
        }
    const auto merged = cross_scan_merge(std::vector<double>(batched.begin(), batched.end()), 2, 2, 2, 3);
    check(merged == std::vector<double>(two.begin(), two.end()), "merge inverts every direction exactly");

    // Causal operator per direction: SelectiveScan with A = 0, delta = 0 is
    // ln2 * prefix sum; the hand-derived merge is in the fixture README.
    std::vector<float> ramp{1, 2, 3, 4, 5, 6};
    Case k = filled({4, 6, 1, 1, 1, 1, false}, 0, 0, 1, 1);
    k.u = cross_scan_expand(ramp, 1, 1, 2, 3);
    k.a = {0};
    compare("cross_scan_prefix_merge", cross_scan_merge(k.run(), 1, 1, 2, 3), expected.at("cross_scan_prefix_merge"), unit);

    reject([] { (void)cross_scan_expand(std::vector<float>{}, 1, 1, 0, 3); }, "dimensions must be positive");
    reject([&] { (void)cross_scan_expand(x, 1, 2, 2, 2); }, "input size does not match shape");
    reject([&] { auto v = x; v[4] = std::numeric_limits<float>::infinity(); (void)cross_scan_expand(v, 1, 2, 2, 3); }, "input is not finite");
    reject([&] { (void)cross_scan_merge(std::vector<double>(47), 1, 2, 2, 3); }, "merge input size does not match shape");
    reject([&] { auto v = std::vector<double>(48); v[7] = std::nan(""); (void)cross_scan_merge(v, 1, 2, 2, 3); }, "merge input is not finite");
    reject([] { (void)cross_scan_source(4, 0, 2, 3); }, "token or direction out of range");
    reject([] { (void)cross_scan_source(0, 6, 2, 3); }, "token or direction out of range");
    reject([] { (void)cross_scan_expand(std::vector<float>{}, std::size_t{1} << 20, 1, 1 << 10, 1 << 10); }, "exceeds int32 capacity");
    report << "cross_scan\tall\tPASS\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) throw std::invalid_argument("usage: selective_scan_tests FIXTURE_DIR [REPORT.tsv]");
        const auto expected = load_expected(std::filesystem::path(argv[1]) / "expected.tsv");
        check(std::fabs(saccade::selective_scan_softplus(20.0) - 20.0000000020611536203L) < 1e-13L, "softplus at the threshold keeps log1p(exp(x))");
        check(saccade::selective_scan_softplus(20.25) == 20.25, "softplus above the threshold is the identity");
        contract_values(expected);
        equalities();
        rejections();
        cross_scan(expected);
        report << "checks\t" << checks << '\n';
        if (argc == 3) saccade::write_text_file(argv[2], report.str());
        std::cout << report.str();
        std::cout << "selective_scan_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << report.str() << "selective_scan_tests: " << error.what() << '\n';
        return 1;
    }
}

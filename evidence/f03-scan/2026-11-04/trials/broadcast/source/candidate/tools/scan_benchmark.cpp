#include "saccade/selective_scan_cuda.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// An unchanged baseline CUDA translation unit is linked with its public
// function names suffixed. Shape/status types deliberately remain identical.
namespace saccade {
void selective_scan_cuda_baseline(const SelectiveScanShape&, const SelectiveScanDeviceInputs&,
                                  float*, ScanStatus*, cudaStream_t);
std::string selective_scan_cuda_load_kernels_baseline();
std::string selective_scan_cuda_identity_baseline();
}

namespace {
using Shape = saccade::SelectiveScanShape;
using Status = saccade::ScanStatus;
using Launch = decltype(&saccade::selective_scan_cuda);
using Clock = std::chrono::steady_clock;
constexpr std::size_t guard = 64;
constexpr std::uint32_t sentinel = 0xa5a5a5a5u;
constexpr const char* usage = "usage: scan_benchmark NEW_OUT_DIR [--verify-only|--benchmark-only] "
    "[--rounds 3] [--iterations 100] [--warmup 20]";

void cu(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void require(bool ok, const std::string& text) { if (!ok) throw std::runtime_error(text); }
template<class T> T positive(std::string_view text) {
    T n{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), n);
    if (ec != std::errc{} || end != text.data() + text.size() || n <= 0)
        throw std::invalid_argument("expected positive integer: " + std::string(text));
    return n;
}
void save(const std::filesystem::path& p, const std::string& text) {
    std::ofstream f(p); require(bool(f), "cannot open " + p.string());
    f << text; f.close(); require(bool(f), "cannot write " + p.string());
}
struct Stream {
    cudaStream_t p{};
    Stream() { cu(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking)); }
    ~Stream() { if (p) cudaStreamDestroy(p); }
};
template<class T> struct Device {
    T* p{};
    std::size_t n;
    explicit Device(std::size_t count) : n(count) {
        if (n) cu(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)));
    }
    ~Device() { if (p) cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    void put(const std::vector<T>& v, cudaStream_t s) {
        require(v.size() == n, "upload size mismatch");
        if (n) cu(cudaMemcpyAsync(p, v.data(), n * sizeof(T), cudaMemcpyHostToDevice, s));
    }
};
struct Case {
    Shape s;
    std::vector<float> u, delta, a, b, c, d;
};
struct Random {
    std::uint64_t state;
    float uniform(float lo, float hi) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        return lo + (hi - lo) * float(double(state >> 40) * 0x1p-24);
    }
};
Case filled(Shape s, float u = 1, float delta = 0, float b = 1, float c = 1) {
    const auto t = s.batch * s.length;
    return {s, std::vector<float>(t * s.channels, u), std::vector<float>(t * s.channels, delta),
        std::vector<float>(s.a_rows * s.state, -1), std::vector<float>(t * s.state, b),
        std::vector<float>(t * s.c_width, c), std::vector<float>(s.has_d ? s.channels : 0, 0)};
}
Case random_case(Shape s, std::uint64_t seed) {
    Case k = filled(s); Random g{seed};
    for (auto& v : k.u) v = g.uniform(-2, 2);
    for (auto& v : k.delta) v = g.uniform(-6, 3);
    for (auto& v : k.a) v = g.uniform(-4, -.05f);
    for (auto& v : k.b) v = g.uniform(-1.5f, 1.5f);
    for (auto& v : k.c) v = g.uniform(-1.5f, 1.5f);
    for (auto& v : k.d) v = g.uniform(-1, 1);
    return k;
}
struct Result { std::vector<float> y; Status status; };
struct Buffers {
    Shape shape;
    cudaStream_t stream;
    Device<float> u, delta, a, b, c, d, y;
    Device<Status> status{1};
    explicit Buffers(const Case& k, cudaStream_t s) : shape(k.s), stream(s), u(k.u.size()),
        delta(k.delta.size()), a(k.a.size()), b(k.b.size()), c(k.c.size()), d(k.d.size()),
        y(k.u.size() + 2 * guard) { load(k); poison(); }
    float* out() const { return y.p + guard; }
    void load(const Case& k) {
        u.put(k.u, stream); delta.put(k.delta, stream); a.put(k.a, stream);
        b.put(k.b, stream); c.put(k.c, stream); d.put(k.d, stream);
        cu(cudaStreamSynchronize(stream));
    }
    void poison() {
        cu(cudaMemsetAsync(y.p, 0xa5, y.n * sizeof(float), stream));
        cu(cudaMemsetAsync(status.p, 0xff, sizeof(Status), stream));
    }
    void launch(Launch f) { f(shape, {u.p, delta.p, a.p, b.p, c.p, d.p}, out(), status.p, stream); }
    Result read() {
        Result r{std::vector<float>(u.n), {}};
        std::vector<std::uint32_t> edges(2 * guard);
        if (u.n) cu(cudaMemcpyAsync(r.y.data(), out(), u.n * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cu(cudaMemcpyAsync(&r.status, status.p, sizeof(Status), cudaMemcpyDeviceToHost, stream));
        cu(cudaMemcpyAsync(edges.data(), y.p, guard * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cu(cudaMemcpyAsync(edges.data() + guard, out() + u.n, guard * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cu(cudaStreamSynchronize(stream));
        require(std::all_of(edges.begin(), edges.end(), [](auto x) { return x == sentinel; }), "output guard damaged");
        require(r.status.done == saccade::kScanDone, "missing completion status");
        for (float x : r.y) {
            if (r.status.flags) require(std::bit_cast<std::uint32_t>(x) == 0x7fc00000u, "failure did not poison whole output");
            else require(std::isfinite(x), "successful output contains nonfinite value");
        }
        return r;
    }
};
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    Graph(Launch f, const std::vector<Buffers*>& buffers, cudaStream_t s) {
        cu(cudaStreamSynchronize(s));
        cu(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
        for (auto* b : buffers) b->launch(f);
        cu(cudaStreamEndCapture(s, &graph));
        cu(cudaGraphInstantiate(&executable, graph, 0));
    }
    ~Graph() { if (executable) cudaGraphExecDestroy(executable); if (graph) cudaGraphDestroy(graph); }
};
void same(const Result& a, const Result& b, const std::string& context) {
    require(a.status.flags == b.status.flags && a.status.done == b.status.done,
        context + " status mismatch baseline_flags=" + std::to_string(a.status.flags) +
        " candidate_flags=" + std::to_string(b.status.flags));
    require(a.y.size() == b.y.size(), context + " output length mismatch");
    for (std::size_t i = 0; i < a.y.size(); ++i)
        if (std::bit_cast<std::uint32_t>(a.y[i]) != std::bit_cast<std::uint32_t>(b.y[i]))
            throw std::runtime_error(context + " output bits differ at " + std::to_string(i) +
                " baseline=" + std::to_string(std::bit_cast<std::uint32_t>(a.y[i])) +
                " candidate=" + std::to_string(std::bit_cast<std::uint32_t>(b.y[i])));
}
std::size_t cases_done = 0, elements_compared = 0;
void verify_case(const Case& k, const std::string& name, std::ofstream& log, cudaStream_t s) {
    log << "BEGIN," << name << '\n'; log.flush();
    Buffers b(k, s);
    b.poison(); b.launch(saccade::selective_scan_cuda_baseline); const auto baseline = b.read();
    Graph bg(saccade::selective_scan_cuda_baseline, {&b}, s);
    b.poison(); cu(cudaGraphLaunch(bg.executable, s)); same(baseline, b.read(), name + " baseline graph");
    b.poison(); b.launch(saccade::selective_scan_cuda); same(baseline, b.read(), name + " candidate ordinary");
    b.poison(); b.launch(saccade::selective_scan_cuda); same(baseline, b.read(), name + " candidate repeat");
    Graph cg(saccade::selective_scan_cuda, {&b}, s);
    b.poison(); cu(cudaGraphLaunch(cg.executable, s)); same(baseline, b.read(), name + " candidate graph");
    if (!k.u.empty()) {
        auto changed = k; changed.u[0] = std::numeric_limits<float>::infinity(); b.load(changed);
        b.poison(); b.launch(saccade::selective_scan_cuda_baseline); const auto invalid = b.read();
        b.poison(); cu(cudaGraphLaunch(cg.executable, s)); same(invalid, b.read(), name + " graph changed input");
        b.load(k); b.poison(); cu(cudaGraphLaunch(cg.executable, s)); same(baseline, b.read(), name + " graph reset A-B-A");
    }
    ++cases_done; elements_compared += k.u.size();
    log << "PASS," << name << ',' << k.s.batch << ',' << k.s.length << ',' << k.s.channels << ','
        << k.s.state << ',' << k.s.a_rows << ',' << k.s.c_width << ',' << k.s.has_d << ','
        << k.u.size() << ',' << baseline.status.flags << '\n'; log.flush();
}
void verify(const std::filesystem::path& out, cudaStream_t s) {
    std::ofstream log(out / "verification.csv"); require(bool(log), "cannot open verification.csv");
    log << "result,name,batch,length,channels,state,a_rows,c_width,has_d,elements,flags\n";
    std::uint64_t seed = 123;
    for (std::size_t n : {1, 2, 4, 8, 16, 32})
        for (bool shared : {true, false}) for (bool rank1 : {true, false}) for (bool has_d : {true, false})
            for (std::size_t length : {0, 1, 19}) {
                Shape sh{2, length, 17, n, shared ? 1u : 17u, rank1 ? 1u : n, has_d};
                ++seed;
                verify_case(random_case(sh, seed), "matrix-" + std::to_string(seed), log, s);
            }
    for (std::size_t length : {960, 240, 60, 400, 100, 25}) for (unsigned trial = 0; trial < 3; ++trial) {
        ++seed;
        verify_case(random_case({4, length, 256, 16, 1, 1, true}, seed),
                    "deploy-L" + std::to_string(length) + "-seed" + std::to_string(seed), log, s);
    }
    const std::vector<float> thresholds{-1100, -1024, std::nextafter(-1024.f, 0.f), -150, -100,
        std::nextafter(-80.f, -100.f), -80, std::nextafter(-80.f, 0.f), -18, -0.f, 0,
        std::nextafter(20.f, 0.f), 20, std::nextafter(20.f, 100.f), 100, 1000};
    for (std::size_t n : {1, 2, 4, 8, 16, 32}) {
        auto k = filled({1, thresholds.size(), 3, n, 1, 1, true});
        for (std::size_t i = 0; i < k.delta.size(); ++i) k.delta[i] = thresholds[i / 3];
        verify_case(k, "softplus-N" + std::to_string(n), log, s);
        k = filled({1, 3, 3, n, 1, n, true}, 0x1p100f, 0x1p100f, 0x1p-100f, 0x1p-100f);
        verify_case(k, "scaled-drive-N" + std::to_string(n), log, s);
        k = filled({1, 3, 3, n, 1, 1, true}, 0x1p100f, -150, 0x1p100f, 0x1p-100f);
        verify_case(k, "scaled-delta-N" + std::to_string(n), log, s);
        for (unsigned tensor = 0; tensor < 6; ++tensor) {
            k = random_case({1, 5, 3, n, 1, 1, true}, ++seed);
            std::array<std::vector<float>*, 6> inputs{&k.u, &k.delta, &k.a, &k.b, &k.c, &k.d};
            (*inputs[tensor])[0] = tensor % 2 ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
            verify_case(k, "invalid-N" + std::to_string(n) + "-tensor" + std::to_string(tensor), log, s);
        }
        for (unsigned failure = 0; failure < 6; ++failure) {
            k = filled({1, 3, 1, n, 1, 1, true});
            if (failure == 0) { k.a.assign(n, 1000); k.delta.assign(3, 1); }
            if (failure == 1) { k.u.assign(3, 0x1p100f); k.b.assign(3 * n, 0x1p100f); }
            if (failure == 2) { k.a.assign(n, 1); k.delta.assign(3, 40); k.u.assign(3, 0x1p60f); }
            if (failure == 3) { k.u.assign(3, 0x1p90f); k.c.assign(3, 0x1p90f); }
            if (failure == 4) { k.u.assign(3, 0x1p100f); k.d.assign(1, 0x1p100f); }
            if (failure == 5) { k.a.assign(n, -100); k.u.assign(3, 0x1p126f); k.b.assign(3 * n, 4); }
            verify_case(k, "range-N" + std::to_string(n) + "-kind" + std::to_string(failure), log, s);
        }
    }
    save(out / "verification.json", "{\n  \"status\": \"PASS\",\n  \"cases\": " + std::to_string(cases_done) +
        ",\n  \"elements_per_comparison\": " + std::to_string(elements_compared) +
        ",\n  \"comparison\": \"exact FP32 output bits and complete status words, ordinary/repeat/graph/changed-input/reset\",\n"
        "  \"formal_acceptance\": false\n}\n");
}
struct Event {
    cudaEvent_t p{};
    Event() { cu(cudaEventCreate(&p)); }
    ~Event() { if (p) cudaEventDestroy(p); }
};
struct Measurement { double gpu_us, wall_us; };
Measurement timed(Launch f, const std::vector<Buffers*>& buffers, Graph* graph,
                  cudaStream_t s, unsigned iterations, unsigned warmup) {
    const auto launch = [&] { if (graph) cu(cudaGraphLaunch(graph->executable, s)); else for (auto* b : buffers) b->launch(f); };
    for (unsigned i = 0; i < warmup; ++i) launch();
    cu(cudaStreamSynchronize(s)); Event start, stop;
    const auto begin = Clock::now(); cu(cudaEventRecord(start.p, s));
    for (unsigned i = 0; i < iterations; ++i) launch();
    cu(cudaEventRecord(stop.p, s)); cu(cudaEventSynchronize(stop.p));
    const auto end = Clock::now(); float ms{}; cu(cudaEventElapsedTime(&ms, start.p, stop.p));
    return {double(ms) * 1000 / iterations, std::chrono::duration<double, std::micro>(end - begin).count() / iterations};
}
void benchmark(const std::filesystem::path& out, cudaStream_t s, unsigned rounds, unsigned iterations, unsigned warmup) {
    std::ofstream csv(out / "timings.csv"); require(bool(csv), "cannot open timings.csv");
    csv << "group,mode,round,order,implementation,iterations,gpu_interval_us,wall_us\n" << std::setprecision(12);
    std::ofstream summary(out / "benchmark.json"); require(bool(summary), "cannot open benchmark.json");
    summary << "{\n  \"formal_acceptance\": false,\n  \"metric\": \"whole scan API pipeline CUDA event interval per replay, not scan-core-only or frame latency\",\n  \"results\": [\n";
    bool first = true;
    const std::vector<std::pair<std::string, std::vector<std::size_t>>> groups{
        {"HR-P3", {960}}, {"HR-P4", {240}}, {"HR-P5", {60}}, {"HR-total", {960, 240, 60}}, {"T01-total", {400, 100, 25}}};
    for (const auto& [name, lengths] : groups) {
        std::vector<std::unique_ptr<Buffers>> owned;
        std::vector<Buffers*> buffers;
        for (auto length : lengths) {
            owned.push_back(std::make_unique<Buffers>(random_case({4, length, 256, 16, 1, 1, true}, 456 + length), s));
            buffers.push_back(owned.back().get());
        }
        Graph bg(saccade::selective_scan_cuda_baseline, buffers, s), cg(saccade::selective_scan_cuda, buffers, s);
        for (bool graph_mode : {false, true}) {
            std::array<std::vector<double>, 2> times;
            for (unsigned round = 0; round < rounds; ++round) for (unsigned order = 0; order < 4; ++order) {
                const bool candidate = order == 1 || order == 2;
                auto f = candidate ? saccade::selective_scan_cuda : saccade::selective_scan_cuda_baseline;
                auto m = timed(f, buffers, graph_mode ? (candidate ? &cg : &bg) : nullptr, s, iterations, warmup);
                for (auto* b : buffers) require(b->read().status.flags == 0, "benchmark produced failed status");
                times[candidate].push_back(m.gpu_us);
                csv << name << ',' << (graph_mode ? "graph" : "direct") << ',' << round << ',' << order << ','
                    << (candidate ? "candidate" : "baseline") << ',' << iterations << ',' << m.gpu_us << ',' << m.wall_us << '\n';
                csv.flush();
            }
            const auto mean = [](const auto& v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
            const double a = mean(times[0]), b = mean(times[1]);
            if (!first) summary << ",\n";
            first = false;
            summary << "    {\"group\":\"" << name << "\",\"mode\":\"" << (graph_mode ? "graph" : "direct")
                << "\",\"baseline_us\":" << a << ",\"candidate_us\":" << b << ",\"speedup\":" << a / b << '}';
            std::cout << name << ' ' << (graph_mode ? "graph" : "direct") << ": " << a << " -> " << b << " us\n";
        }
    }
    summary << "\n  ]\n}\n"; summary.close(); require(bool(summary), "cannot write benchmark.json");
}
} // namespace

int main(int argc, char** argv) {
    std::filesystem::path out;
    bool created = false;
    try {
        require(argc >= 2, usage); out = argv[1];
        bool verify_only = false, benchmark_only = false;
        unsigned rounds = 3, iterations = 100, warmup = 20;
        for (int i = 2; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--verify-only") verify_only = true;
            else if (arg == "--benchmark-only") benchmark_only = true;
            else if ((arg == "--rounds" || arg == "--iterations" || arg == "--warmup") && i + 1 < argc) {
                const auto n = positive<unsigned>(argv[++i]);
                if (arg == "--rounds") rounds = n; else if (arg == "--iterations") iterations = n; else warmup = n;
            } else throw std::invalid_argument(usage);
        }
        require(!(verify_only && benchmark_only), "conflicting run modes");
        require(!std::filesystem::exists(out), "output directory must be new");
        std::filesystem::create_directories(out);
        created = true;
        save(out / "status.txt", "RUNNING\nformal_acceptance=false\n");
        save(out / "baseline-identity.txt", saccade::selective_scan_cuda_identity_baseline() + saccade::selective_scan_cuda_load_kernels_baseline());
        save(out / "candidate-identity.txt", saccade::selective_scan_cuda_identity() + saccade::selective_scan_cuda_load_kernels());
        Stream s;
        if (!benchmark_only) verify(out, s.p);
        if (!verify_only) benchmark(out, s.p, rounds, iterations, warmup);
        cu(cudaStreamSynchronize(s.p));
        save(out / "status.txt", "PASS\nformal_acceptance=false\n");
        std::cout << "scan_benchmark: PASS, cases=" << cases_done << ", elements_per_comparison=" << elements_compared << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "scan_benchmark: " << e.what() << '\n';
        // Preserve partial CSV evidence on failure; never overwrite a preexisting directory.
        if (created && std::filesystem::exists(out / "status.txt")) {
            try { save(out / "status.txt", std::string("FAIL\nformal_acceptance=false\nerror=") + e.what() + '\n'); } catch (...) {}
        }
        return 1;
    }
}

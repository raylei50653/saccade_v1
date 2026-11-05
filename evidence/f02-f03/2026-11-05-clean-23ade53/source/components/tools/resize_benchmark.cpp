#include "saccade/resize_cuda.hpp"

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
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Link an unchanged baseline resize.cu compiled with its six exported function
// names suffixed with _baseline. ResizeShape itself must not be renamed.
namespace saccade {
void resize_rgb_chw_cuda_baseline(const ResizeShape&, const std::uint8_t*, std::size_t,
                                float*, std::size_t, float*, std::size_t, cudaStream_t);
std::string resize_cuda_load_kernels_baseline();
std::string resize_cuda_identity_baseline();
}

namespace {
using Clock = std::chrono::steady_clock;
using Launch = decltype(&saccade::resize_rgb_chw_cuda);
constexpr std::size_t guard = 64;
constexpr std::uint32_t sentinel = 0xa5a5a5a5u;
constexpr const char* usage =
    "usage: resize_benchmark NEW_OUT_DIR [--verify-only|--benchmark-only] "
    "[--shape IW IH OW OH] [--rgb PLANAR_RGB] [--rounds 3] [--iterations 100] [--warmup 30]";

void cu(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void check(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}
template<class T> T number(std::string_view text) {
    T value{};
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || end != text.data() + text.size() || value <= 0)
        throw std::invalid_argument("expected positive integer: " + std::string(text));
    return value;
}
std::string label(const saccade::ResizeShape& s) {
    return std::to_string(s.input_width) + "x" + std::to_string(s.input_height) + "-" +
           std::to_string(s.output_width) + "x" + std::to_string(s.output_height);
}
void write(const std::filesystem::path& path, const void* data, std::size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open " + path.string());
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    out.close();
    if (!out) throw std::runtime_error("cannot write/close " + path.string());
}
void write(const std::filesystem::path& path, const std::string& text) {
    write(path, text.data(), text.size());
}
struct Allocation {
    void* p = nullptr;
    explicit Allocation(std::size_t bytes) { cu(cudaMalloc(&p, bytes)); }
    ~Allocation() { if (p) (void)cudaFree(p); }
    Allocation(const Allocation&) = delete;
};
struct Stream {
    cudaStream_t p = nullptr;
    Stream() { cu(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking)); }
    ~Stream() { if (p) (void)cudaStreamDestroy(p); }
};
struct Event {
    cudaEvent_t p = nullptr;
    Event() { cu(cudaEventCreate(&p)); }
    ~Event() { if (p) (void)cudaEventDestroy(p); }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    Graph(Launch launch, const saccade::ResizeShape& shape, const std::uint8_t* input,
          std::size_t ni, float* output, std::size_t no, float* workspace,
          std::size_t nw, cudaStream_t stream) {
        cu(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(shape, input, ni, output, no, workspace, nw, stream);
        cu(cudaStreamEndCapture(stream, &graph));
        cu(cudaGraphInstantiate(&executable, graph, 0));
    }
    ~Graph() {
        if (executable) (void)cudaGraphExecDestroy(executable);
        if (graph) (void)cudaGraphDestroy(graph);
    }
    Graph(const Graph&) = delete;
};

struct Buffers {
    saccade::ResizeShape shape;
    std::size_t ni, no, nw;
    Stream stream;
    Allocation input, output, workspace;
    explicit Buffers(const saccade::ResizeShape& s)
        : shape(s), ni(saccade::resize_input_bytes(s)), no(saccade::resize_output_elements(s)),
          nw(saccade::resize_workspace_elements(s)), input(ni),
          output((no + 2 * guard) * sizeof(float)), workspace((nw + 2 * guard) * sizeof(float)) {}
    ~Buffers() { (void)cudaStreamSynchronize(stream.p); }
    std::uint8_t* rgb() { return static_cast<std::uint8_t*>(input.p); }
    float* out() { return static_cast<float*>(output.p) + guard; }
    float* work() { return static_cast<float*>(workspace.p) + guard; }
    void load(const std::vector<std::uint8_t>& values) {
        check(values.size() == ni, "wrong RGB byte count");
        cu(cudaMemcpyAsync(input.p, values.data(), ni, cudaMemcpyHostToDevice, stream.p));
        cu(cudaStreamSynchronize(stream.p));
    }
    void poison() {
        cu(cudaMemsetAsync(output.p, 0xa5, (no + 2 * guard) * sizeof(float), stream.p));
        cu(cudaMemsetAsync(workspace.p, 0xa5, (nw + 2 * guard) * sizeof(float), stream.p));
        cu(cudaMemsetAsync(out(), 0xff, no * sizeof(float), stream.p));
        cu(cudaMemsetAsync(work(), 0xff, nw * sizeof(float), stream.p));
    }
    void launch(Launch f) { f(shape, rgb(), ni, out(), no, work(), nw, stream.p); }
    std::vector<float> read() {
        std::vector<std::uint32_t> edges(guard * 4);
        std::vector<float> values(no);
        cu(cudaMemcpyAsync(values.data(), out(), no * sizeof(float), cudaMemcpyDeviceToHost, stream.p));
        cu(cudaMemcpyAsync(edges.data(), output.p, guard * sizeof(float), cudaMemcpyDeviceToHost, stream.p));
        cu(cudaMemcpyAsync(edges.data() + guard, out() + no, guard * sizeof(float), cudaMemcpyDeviceToHost, stream.p));
        cu(cudaMemcpyAsync(edges.data() + 2 * guard, workspace.p, guard * sizeof(float), cudaMemcpyDeviceToHost, stream.p));
        cu(cudaMemcpyAsync(edges.data() + 3 * guard, work() + nw, guard * sizeof(float), cudaMemcpyDeviceToHost, stream.p));
        cu(cudaStreamSynchronize(stream.p));
        check(std::all_of(edges.begin(), edges.end(), [](auto v) { return v == sentinel; }), "output/workspace guard corrupted");
        check(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }), "output contains nonfinite/unwritten elements");
        return values;
    }
};

void same(const std::vector<float>& expected, const std::vector<float>& actual,
          const std::string& context) {
    check(expected.size() == actual.size(), "wrong output element count");
    if (std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0) return;
    std::size_t mismatches = 0, first = 0;
    float max_error = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(expected[i]) != std::bit_cast<std::uint32_t>(actual[i])) {
            if (mismatches++ == 0) first = i;
            max_error = std::max(max_error, std::abs(expected[i] - actual[i]));
        }
    }
    throw std::runtime_error(context + ": bitwise mismatch count=" + std::to_string(mismatches) +
        " first=" + std::to_string(first) + " baseline_bits=" + std::to_string(std::bit_cast<std::uint32_t>(expected[first])) +
        " candidate_bits=" + std::to_string(std::bit_cast<std::uint32_t>(actual[first])) +
        " max_absolute_error=" + std::to_string(max_error));
}

std::vector<std::uint8_t> pattern(const saccade::ResizeShape& s, unsigned kind) {
    std::vector<std::uint8_t> data(saccade::resize_input_bytes(s));
    std::uint32_t rng = 0x12345678u;
    for (int c = 0; c < 3; ++c) for (int y = 0; y < s.input_height; ++y)
        for (int x = 0; x < s.input_width; ++x) {
            std::uint8_t value = 0;
            if (kind == 0) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; value = static_cast<std::uint8_t>(rng); }
            else if (kind == 1) value = ((x + y + c) & 1) ? 255 : 0;
            else if (kind == 2) value = static_cast<std::uint8_t>((x * 17 + y * 29 + c * 53) & 255);
            else value = (x == 0 || y == 0 || x + 1 == s.input_width || y + 1 == s.input_height) ? 255 : 0;
            data[(static_cast<std::size_t>(c) * s.input_height + y) * s.input_width + x] = value;
        }
    return data;
}

std::size_t verify(const saccade::ResizeShape& shape, const std::vector<std::uint8_t>& data,
                   const std::string& name, const std::filesystem::path& out, bool dump) {
    const auto context = name + " " + label(shape);
    Buffers b(shape);
    b.load(data);
    b.poison(); b.launch(saccade::resize_rgb_chw_cuda_baseline);
    const auto baseline = b.read();
    Graph baseline_graph(saccade::resize_rgb_chw_cuda_baseline, shape, b.rgb(), b.ni, b.out(), b.no, b.work(), b.nw, b.stream.p);
    b.poison(); cu(cudaGraphLaunch(baseline_graph.executable, b.stream.p));
    same(baseline, b.read(), context + " baseline graph");
    b.poison(); b.launch(saccade::resize_rgb_chw_cuda);
    const auto candidate = b.read();
    same(baseline, candidate, context + " candidate ordinary");
    b.poison(); b.launch(saccade::resize_rgb_chw_cuda);
    same(baseline, b.read(), context + " candidate repeat");
    Graph candidate_graph(saccade::resize_rgb_chw_cuda, shape, b.rgb(), b.ni, b.out(), b.no, b.work(), b.nw, b.stream.p);
    b.poison(); cu(cudaGraphLaunch(candidate_graph.executable, b.stream.p));
    same(baseline, b.read(), context + " candidate graph");
    if (dump) {
        write(out / (name + "-baseline.f32"), baseline.data(), baseline.size() * sizeof(float));
        write(out / (name + "-candidate.f32"), candidate.data(), candidate.size() * sizeof(float));
    }
    std::cout << "verify," << name << ',' << label(shape) << ',' << b.no << ",PASS\n" << std::flush;
    return b.no;
}

struct Sample { unsigned round, ordinal; bool candidate, graph; double event_us, wall_us; };
std::vector<Sample> benchmark(const saccade::ResizeShape& shape, const std::vector<std::uint8_t>& data,
                              unsigned rounds, unsigned iterations, unsigned warmup) {
    Buffers b(shape); b.load(data); b.poison();
    b.launch(saccade::resize_rgb_chw_cuda_baseline); b.launch(saccade::resize_rgb_chw_cuda);
    cu(cudaStreamSynchronize(b.stream.p));
    Graph baseline_graph(saccade::resize_rgb_chw_cuda_baseline, shape, b.rgb(), b.ni, b.out(), b.no, b.work(), b.nw, b.stream.p);
    Graph candidate_graph(saccade::resize_rgb_chw_cuda, shape, b.rgb(), b.ni, b.out(), b.no, b.work(), b.nw, b.stream.p);
    std::vector<Sample> samples;
    // Each round uses ABBA for both ordinary enqueue and one-operation graph
    // replay. Event/host results are batch-average processing intervals, not
    // single-frame latency distributions or isolated kernel busy times.
    for (unsigned round = 0; round < rounds; ++round) for (const bool graph : {false, true}) {
        for (unsigned ordinal = 0; ordinal < 4; ++ordinal) {
            const bool candidate = ordinal == 1 || ordinal == 2;
            const auto launch = [&] {
                if (graph) cu(cudaGraphLaunch(candidate ? candidate_graph.executable : baseline_graph.executable, b.stream.p));
                else b.launch(candidate ? saccade::resize_rgb_chw_cuda : saccade::resize_rgb_chw_cuda_baseline);
            };
            for (unsigned i = 0; i < warmup; ++i) launch();
            cu(cudaStreamSynchronize(b.stream.p));
            Event begin, end;
            const auto started = Clock::now();
            cu(cudaEventRecord(begin.p, b.stream.p));
            for (unsigned i = 0; i < iterations; ++i) launch();
            cu(cudaEventRecord(end.p, b.stream.p));
            cu(cudaEventSynchronize(end.p));
            const double wall_us = std::chrono::duration<double, std::micro>(Clock::now() - started).count() / iterations;
            float elapsed_ms = 0; cu(cudaEventElapsedTime(&elapsed_ms, begin.p, end.p));
            samples.push_back({round, ordinal, candidate, graph, elapsed_ms * 1000.0 / iterations, wall_us});
            std::cout << "sample," << round << ',' << ordinal << ',' << (candidate ? "candidate" : "baseline") << ','
                      << (graph ? "graph" : "ordinary") << ',' << samples.back().event_us << ',' << wall_us << '\n' << std::flush;
        }
    }
    return samples;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") { std::cout << usage << '\n'; return 0; }
        if (argc < 2) throw std::invalid_argument(usage);
        const std::filesystem::path out = argv[1];
        bool verify_only = false, benchmark_only = false, shape_set = false;
        unsigned rounds = 3, iterations = 100, warmup = 30;
        saccade::ResizeShape shape{1920, 1080, 1280, 768};
        std::filesystem::path rgb_path;
        for (int i = 2; i < argc; ++i) {
            const std::string_view arg = argv[i];
            const auto next = [&]() -> std::string_view {
                if (++i >= argc) throw std::invalid_argument("missing argument for " + std::string(arg));
                return argv[i];
            };
            if (arg == "--verify-only") verify_only = true;
            else if (arg == "--benchmark-only") benchmark_only = true;
            else if (arg == "--rounds") rounds = number<unsigned>(next());
            else if (arg == "--iterations") iterations = number<unsigned>(next());
            else if (arg == "--warmup") warmup = number<unsigned>(next());
            else if (arg == "--rgb") rgb_path = next();
            else if (arg == "--shape") {
                shape.input_width = number<int>(next()); shape.input_height = number<int>(next());
                shape.output_width = number<int>(next()); shape.output_height = number<int>(next()); shape_set = true;
            } else throw std::invalid_argument("unknown argument " + std::string(arg));
        }
        check(!(verify_only && benchmark_only), "verification and benchmark modes are mutually exclusive");
        check(!std::filesystem::exists(out), "output directory must be new");
        std::filesystem::create_directories(out);
        cu(cudaSetDevice(0));
        cudaDeviceProp device{}; cu(cudaGetDeviceProperties(&device, 0));
        int runtime = 0, driver = 0; cu(cudaRuntimeGetVersion(&runtime)); cu(cudaDriverGetVersion(&driver));
        std::ostringstream identity;
        identity << "baseline:\n" << saccade::resize_cuda_identity_baseline() << saccade::resize_cuda_load_kernels_baseline()
                 << "candidate:\n" << saccade::resize_cuda_identity() << saccade::resize_cuda_load_kernels()
                 << "device=" << device.name << "\ncompute_capability=" << device.major << '.' << device.minor
                 << "\ncuda_runtime=" << runtime << "\ncuda_driver=" << driver << "\nshape=" << label(shape)
                 << "\nrounds=" << rounds << "\niterations=" << iterations << "\nwarmup=" << warmup
                 << "\nbenchmark_scope=isolated-resize-batch-average-processing-interval\n";
        write(out / "identity.txt", identity.str());
        std::cout << std::fixed << std::setprecision(6);
        std::vector<std::uint8_t> rgb;
        if (!rgb_path.empty()) {
            const auto bytes = std::filesystem::file_size(rgb_path);
            check(bytes == saccade::resize_input_bytes(shape), "--rgb file size does not match --shape");
            rgb.resize(static_cast<std::size_t>(bytes));
            std::ifstream file(rgb_path, std::ios::binary);
            file.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
            check(bool(file), "cannot read RGB input");
        } else rgb = pattern(shape, 0);
        std::size_t verified_cases = 0, verified_elements = 0;
        if (!benchmark_only) {
            if (!rgb_path.empty()) {
                verified_elements += verify(shape, rgb, "real-rgb", out, true); ++verified_cases;
            } else {
                std::vector<saccade::ResizeShape> shapes;
                if (shape_set) shapes.push_back(shape);
                else shapes = {
                    {1,1,1,1}, {2,2,1,1}, {3,3,2,2}, {7,5,4,3}, {7,5,7,5},
                    {7,5,13,9}, {13,9,7,5}, {31,17,16,9}, {33,19,17,10},
                    {63,33,32,17}, {65,35,33,18}, {127,63,64,32}, {129,65,65,33},
                    {255,127,128,64}, {257,129,129,65}, {259,131,130,66},
                    {511,257,256,129}, {513,259,257,130},
                    {1,33,1,17}, {33,1,17,1}, {1,1,17,13},
                    {99,77,17,13}, {17,99,13,17}, {99,17,17,13},
                    {65535,1,32768,1}, {1,65535,1,32768},
                    {1920,1080,1280,768}, {1920,1080,640,640},
                    {640,480,1280,768}, {1280,720,1280,768},
                    {1919,1079,1279,767}, {1921,1081,1281,769}
                };
                for (const auto& s : shapes) for (unsigned kind = 0; kind < 4; ++kind) {
                    verified_elements += verify(s, pattern(s, kind), "pattern-" + std::to_string(kind), out, false);
                    ++verified_cases;
                }
            }
        }
        std::ostringstream summary;
        summary << "{\n  \"verified_cases\": " << verified_cases << ",\n  \"verified_output_elements\": " << verified_elements
                << ",\n  \"bitwise_equivalence\": " << (verified_cases ? "true" : "null");
        if (!verify_only) {
            const auto samples = benchmark(shape, rgb, rounds, iterations, warmup);
            std::ostringstream csv;
            csv << "round,ordinal,backend,mode,event_us_per_operation,wall_us_per_operation\n" << std::fixed << std::setprecision(6);
            for (const auto& s : samples)
                csv << s.round << ',' << s.ordinal << ',' << (s.candidate ? "candidate" : "baseline") << ','
                    << (s.graph ? "graph" : "ordinary") << ',' << s.event_us << ',' << s.wall_us << '\n';
            write(out / "samples.csv", csv.str());
            for (const bool graph : {false, true}) for (const bool candidate : {false, true}) {
                double event_total = 0, wall_total = 0; std::size_t count = 0;
                for (const auto& s : samples) if (s.graph == graph && s.candidate == candidate) {
                    event_total += s.event_us; wall_total += s.wall_us; ++count;
                }
                const std::string key = std::string(graph ? "graph_" : "ordinary_") + (candidate ? "candidate" : "baseline");
                summary << ",\n  \"" << key << "_event_us_per_operation\": " << event_total / static_cast<double>(count)
                        << ",\n  \"" << key << "_wall_us_per_operation\": " << wall_total / static_cast<double>(count);
            }
        }
        summary << "\n}\n";
        write(out / "summary.json", summary.str());
        std::cout << summary.str();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "resize_benchmark: " << error.what() << '\n';
        return 1;
    }
}

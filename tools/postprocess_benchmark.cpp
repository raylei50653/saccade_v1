// Complete F04 API timing, from already-resident raw tensors to candidates.
// No TensorRT/decode/resize traffic is included. See docs/POSTPROCESS.md.
#include "saccade/postprocess_cuda.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace saccade;
void cu(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
struct Device {
    void* p = nullptr;
    explicit Device(std::size_t n) { cu(cudaMalloc(&p, n)); }
    ~Device() { if (p) (void)cudaFree(p); }
    Device(const Device&) = delete;
    template<class T> T* as() const { return static_cast<T*>(p); }
    void close(CleanupReport& report) { report.release(p, [](void* x) { cu(cudaFree(x)); }); }
};
struct Stream {
    cudaStream_t p = nullptr;
    Stream() { cu(cudaStreamCreateWithFlags(&p, cudaStreamNonBlocking)); }
    ~Stream() { if (p) (void)cudaStreamDestroy(p); }
};
struct Raw {
    std::array<std::vector<float>, 3> cls, reg;
    Raw() {
        for (unsigned l = 0; l < 3; ++l) {
            const auto spatial = kPostprocessSizes[l] * kPostprocessSizes[l];
            cls[l].assign(80 * spatial, -20.0f); reg[l].assign(4 * spatial, 0.125f);
        }
    }
    PostprocessRawView view() const {
        PostprocessRawView out;
        for (unsigned l = 0; l < 3; ++l) {
            out.cls[l] = cls[l].data(); out.reg[l] = reg[l].data();
            out.cls_elements[l] = cls[l].size(); out.reg_elements[l] = reg[l].size();
        }
        return out;
    }
};
void compare(const DetectionCandidates& actual, const DetectionCandidates& expected) {
    require(std::memcmp(&actual.header, &expected.header, sizeof(actual.header)) == 0, "benchmark header differs from reference");
    for (unsigned i = 0; i < kPostprocessCapacity; ++i) {
        auto a = actual.rows[i], b = expected.rows[i];
        if (i < expected.header.count) {
            require(std::isfinite(a.score) && std::abs(a.score - b.score) <= 4 * std::numeric_limits<float>::epsilon(),
                    "benchmark score exceeds policy");
            a.score = b.score = 0;
        }
        require(std::memcmp(&a, &b, sizeof(a)) == 0, "benchmark candidate bytes differ from reference");
    }
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end()); return values[values.size() / 2];
}
double p95(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(std::ceil(values.size() * 0.95)) - 1];
}
constexpr std::array<const char*, 6> hashes{
    "12df20281dd2bd1b4dd2aefbffcb2ce598115e993f530b1d3b82ae2dbdf763e5",
    "02558978a3e7437cc82c24c3532386588d596c2f9f54a6760d811a4543a9cf16",
    "b69dc2f3a468540d507a35ee5da767d7607c88948c68418b1dfa8296b55ea85f",
    "c273efb9986a957a79ff29ec2c9ffa2fb98469fe284f2d11c4f3fa63b559c815",
    "a6c72c191da4fad43ae298ea0f677037c1f8ae36d04618a1e7305890f37def57",
    "ccd755c5f7a087bce0168e36076e377053fc6678fe8b5dc0ad981fcd52e2e771"};
Raw load(const std::string& directory, unsigned index) {
    const auto blob = read_text_file(directory + "/input-" + std::to_string(index) + ".f32");
    require(blob.size() == 2822400 && sha256_hex(blob) == hashes[index], "benchmark F03 input identity mismatch");
    Raw raw; std::size_t offset = 0;
    // The original dump has all three cls tensors followed by all three reg.
    for (unsigned kind = 0; kind < 2; ++kind) for (unsigned l = 0; l < 3; ++l) {
        auto& tensor = kind == 0 ? raw.cls[l] : raw.reg[l];
        std::memcpy(tensor.data(), blob.data() + offset, tensor.size() * sizeof(float));
        offset += tensor.size() * sizeof(float);
    }
    return raw;
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2 && argc <= 4, "usage: postprocess_benchmark F03_DUMP_DIR [ITERATIONS=32] [ROUNDS=7]");
        const int iterations = argc > 2 ? std::stoi(argv[2]) : 32;
        const int rounds = argc > 3 ? std::stoi(argv[3]) : 7;
        require(iterations > 0 && iterations <= 1024 && rounds > 0 && rounds <= 100, "invalid benchmark iteration count");
        cu(cudaSetDevice(0));
        std::cerr << postprocess_cuda_load_kernels() << postprocess_cuda_identity()
                  << "iterations=" << iterations << "\nrounds=" << rounds
                  << "\nscope=complete F04 API; resident raw tensors; excludes H2D/D2H, F01-F03 and downstream tracking"
                  << "\npercentiles=across round mean latencies, not per-frame tails"
                  << "\ngraph_batch=one graph containing ITERATIONS complete F04 calls\n";
        Stream stream;
        Device data(2822400), producer(sizeof(PostprocessProducerStatus)), output(sizeof(DetectionCandidates)), workspace(postprocess_workspace_bytes());
        const PostprocessProducerStatus status{0, kPostprocessProducerDone};
        cu(cudaMemcpyAsync(producer.p, &status, sizeof(status), cudaMemcpyHostToDevice, stream.p));
        PostprocessRawView device;
        std::size_t offset = 0;
        for (unsigned l = 0; l < 3; ++l) {
            const auto spatial = kPostprocessSizes[l] * kPostprocessSizes[l];
            device.cls[l] = data.as<float>() + offset; device.cls_elements[l] = spatial * 80; offset += spatial * 80;
            device.reg[l] = data.as<float>() + offset; device.reg_elements[l] = spatial * 4; offset += spatial * 4;
        }
        const auto read = [&] {
            DetectionCandidates result;
            cu(cudaMemcpyAsync(&result, output.p, sizeof(result), cudaMemcpyDeviceToHost, stream.p));
            cu(cudaStreamSynchronize(stream.p)); return result;
        };
        std::cout << "sample,qualified,mode,kernels,other_nodes,iterations,rounds,gpu_median_us,round_p95_us,wall_median_us,output_sha256,correct\n";
        constexpr std::array<unsigned, 6> kept{32, 28, 39, 2, 3, 1};
        for (unsigned index = 0; index < 6; ++index) {
            const auto raw = load(argv[1], index);
            for (unsigned l = 0; l < 3; ++l) {
                cu(cudaMemcpyAsync(const_cast<float*>(device.cls[l]), raw.cls[l].data(), raw.cls[l].size() * sizeof(float), cudaMemcpyHostToDevice, stream.p));
                cu(cudaMemcpyAsync(const_cast<float*>(device.reg[l]), raw.reg[l].data(), raw.reg[l].size() * sizeof(float), cudaMemcpyHostToDevice, stream.p));
            }
            PostprocessRequest request;
            request.epoch = 0x123456789ULL; request.frame = index;
            if (index < 3) request.mapping = {1920, 1080, 640, 640, 640.0 / 1920, 640.0 / 1080};
            const auto expected = postprocess_reference(raw.view(), request, status);
            require(expected.header.flags == 0 && expected.header.kept == kept[index], "benchmark independent historical kept count mismatch");
            const auto launch = [&] {
                postprocess_cuda(device, request, producer.as<PostprocessProducerStatus>(), output.as<DetectionCandidates>(),
                                 sizeof(DetectionCandidates), workspace.p, postprocess_workspace_bytes(), stream.p);
            };
            for (unsigned warmup = 0; warmup < 4; ++warmup) launch();
            cu(cudaStreamSynchronize(stream.p));
            const auto baseline = read(); compare(baseline, expected);
            const auto digest = sha256_hex(std::string_view(reinterpret_cast<const char*>(&baseline), sizeof(baseline)));
            // Count one complete invocation's actual graph nodes.
            cudaGraph_t one = nullptr;
            cu(cudaStreamBeginCapture(stream.p, cudaStreamCaptureModeThreadLocal)); launch();
            cu(cudaStreamEndCapture(stream.p, &one));
            std::size_t node_count = 0; cu(cudaGraphGetNodes(one, nullptr, &node_count));
            std::vector<cudaGraphNode_t> nodes(node_count); cu(cudaGraphGetNodes(one, nodes.data(), &node_count));
            std::size_t kernels = 0;
            for (const auto node : nodes) { cudaGraphNodeType type{}; cu(cudaGraphNodeGetType(node, &type)); kernels += type == cudaGraphNodeTypeKernel; }
            cu(cudaGraphDestroy(one));
            for (const bool graph_mode : {false, true}) {
                cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
                if (graph_mode) {
                    cu(cudaStreamBeginCapture(stream.p, cudaStreamCaptureModeThreadLocal));
                    for (int i = 0; i < iterations; ++i) launch();
                    cu(cudaStreamEndCapture(stream.p, &graph)); cu(cudaGraphInstantiate(&executable, graph, 0));
                    for (unsigned warmup = 0; warmup < 3; ++warmup) cu(cudaGraphLaunch(executable, stream.p));
                    cu(cudaStreamSynchronize(stream.p));
                }
                cudaEvent_t begin = nullptr, end = nullptr;
                cu(cudaEventCreate(&begin)); cu(cudaEventCreate(&end));
                std::vector<double> gpu, wall;
                for (int round = 0; round < rounds; ++round) {
                    cu(cudaEventRecord(begin, stream.p));
                    const auto started = std::chrono::steady_clock::now();
                    if (graph_mode) cu(cudaGraphLaunch(executable, stream.p));
                    else for (int i = 0; i < iterations; ++i) launch();
                    cu(cudaEventRecord(end, stream.p)); cu(cudaEventSynchronize(end));
                    const auto finished = std::chrono::steady_clock::now();
                    float milliseconds = 0; cu(cudaEventElapsedTime(&milliseconds, begin, end));
                    gpu.push_back(milliseconds * 1000.0 / iterations);
                    wall.push_back(std::chrono::duration<double, std::micro>(finished - started).count() / iterations);
                    const auto result = read(); compare(result, expected);
                    require(std::memcmp(&result, &baseline, sizeof(result)) == 0, "benchmark repeat/graph bytes differ");
                }
                std::cout << "f03-" << index << ',' << kPostprocessPositions - expected.header.low_score << ','
                          << (graph_mode ? "graph_batch" : "plain") << ',' << kernels << ',' << node_count - kernels << ','
                          << iterations << ',' << rounds << ',' << std::fixed << std::setprecision(5)
                          << median(gpu) << ',' << p95(gpu) << ',' << median(wall) << ',' << digest << ",PASS\n";
                cu(cudaEventDestroy(begin)); cu(cudaEventDestroy(end));
                if (executable) cu(cudaGraphExecDestroy(executable));
                if (graph) cu(cudaGraphDestroy(graph));
            }
        }
        cu(cudaStreamSynchronize(stream.p)); CleanupReport cleanup;
        workspace.close(cleanup); output.close(cleanup); producer.close(cleanup); data.close(cleanup);
        cleanup.release(stream.p, [](cudaStream_t value) { cu(cudaStreamDestroy(value)); }); cleanup.throw_if_failed();
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

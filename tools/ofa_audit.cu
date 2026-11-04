// TRACKING §20.6: direct API 2.0 audit, independent of TrackingSession and
// flow::Engine. All measured work uses one non-default IO stream. Preparation
// and observation synchronize; only the explicitly named control synchronizes
// between the operations whose ordering is under test.
#include "nvOpticalFlowCuda.h"
#include "saccade/sha256.hpp"
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int width = 1920, height = 1080, grid_width = width / 4, grid_height = height / 4;
constexpr int trials = 20;
using Bytes = std::vector<std::uint8_t>;
void ck(cudaError_t s, const char* op) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(s));
}
void driver(CUresult s, const char* op) {
    if (s != CUDA_SUCCESS) throw std::runtime_error(std::string(op) + ": CUresult=" + std::to_string(s));
}
__global__ void spin(unsigned long long ns) {
    unsigned long long start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); } while (now - start < ns);
}
__global__ void copy_image(const std::uint8_t* src, std::uint8_t* dst, std::size_t pitch, int w, int h) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < std::size_t(w) * h) dst[(i / w) * pitch + i % w] = src[i];
}
__global__ void snapshot(const std::uint8_t* src, std::size_t pitch, std::uint8_t* dst, int w, int h) {
    const auto i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
    if (i < std::size_t(w) * h) dst[i] = src[(i / w) * pitch + i % w];
}
Bytes texture() {
    Bytes image(std::size_t(width) * height);
    for (std::size_t i = 0; i < image.size(); ++i) {
        std::uint32_t v = static_cast<std::uint32_t>(i) + 0x9e3779b9u;
        v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
        image[i] = static_cast<std::uint8_t>(v >> 24);
    }
    return image;
}
Bytes shift(const Bytes& image, int dx, int dy) {
    Bytes out(image.size());
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x)
        out[std::size_t(y) * width + x] = image[std::size_t(std::clamp(y - dy, 0, height - 1)) * width +
                                                std::clamp(x - dx, 0, width - 1)];
    return out;
}
std::string sha(const Bytes& v) {
    return saccade::sha256_hex({reinterpret_cast<const char*>(v.data()), v.size()});
}
void save(const std::filesystem::path& path, const Bytes& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()));
    f.close();
    if (!f) throw std::runtime_error("cannot write " + path.string());
}
std::size_t differences(const Bytes& a, const Bytes& b) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
    return n;
}
double quantile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const double position = (v.size() - 1) * p;
    const auto lo = static_cast<std::size_t>(position), hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (position - lo);
}
struct Audit {
    struct Buffer { NvOFGPUBufferHandle handle = nullptr; std::uint8_t* ptr = nullptr; std::size_t pitch = 0; };
    CUcontext context = nullptr;
    cudaStream_t stream = nullptr;
    void* library = nullptr;
    NV_OF_CUDA_API_FUNCTION_LIST api{};
    NvOFHandle handle = nullptr;
    Buffer input, reference, output;
    std::uint8_t *replacement = nullptr, *snap = nullptr;
    const std::size_t output_bytes = std::size_t(grid_width) * grid_height * sizeof(short2);
    std::filesystem::path dir;
    Bytes a = texture(), b = shift(a, 8, 4), c = shift(a, -16, 12);
    Bytes ab, cb;
    std::vector<double> all_host;
    std::ofstream csv;

    void of(NV_OF_STATUS s, const char* op) {
        if (s == NV_OF_SUCCESS) return;
        char message[1024]{}; std::uint32_t n = sizeof(message);
        if (handle && api.nvOFGetLastError) api.nvOFGetLastError(handle, message, &n);
        throw std::runtime_error(std::string(op) + ": NV_OF_STATUS=" + std::to_string(s) + " " + message);
    }
    void allocate(Buffer& buf, int w, int h, NV_OF_BUFFER_USAGE usage, NV_OF_BUFFER_FORMAT format) {
        const NV_OF_BUFFER_DESCRIPTOR desc{std::uint32_t(w), std::uint32_t(h), usage, format};
        of(api.nvOFCreateGPUBufferCuda(handle, &desc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &buf.handle), "create buffer");
        buf.ptr = reinterpret_cast<std::uint8_t*>(api.nvOFGPUBufferGetCUdeviceptr(buf.handle));
        NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
        of(api.nvOFGPUBufferGetStrideInfo(buf.handle, &stride), "get stride");
        buf.pitch = stride.strideInfo[0].strideXInBytes;
    }
    void wait() { ck(cudaStreamSynchronize(stream), "synchronize audit IO stream"); }
    void upload(const Bytes& image) {
        ck(cudaMemcpy2DAsync(input.ptr, input.pitch, image.data(), width, width, height,
                             cudaMemcpyHostToDevice, stream), "prepare input");
    }
    double execute() {
        NV_OF_EXECUTE_INPUT_PARAMS in{};
        in.inputFrame = input.handle; in.referenceFrame = reference.handle;
        in.disableTemporalHints = NV_OF_TRUE;
        NV_OF_EXECUTE_OUTPUT_PARAMS out{}; out.outputBuffer = output.handle;
        const auto start = std::chrono::steady_clock::now();
        of(api.nvOFExecute(handle, &in, &out), "NvOFExecute");
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    Bytes read() {
        Bytes raw(output_bytes);
        ck(cudaMemcpy2DAsync(raw.data(), grid_width * sizeof(short2), output.ptr, output.pitch,
                             grid_width * sizeof(short2), grid_height, cudaMemcpyDeviceToHost, stream), "read raw output");
        wait(); return raw;
    }
    explicit Audit(const std::filesystem::path& path) : dir(path), csv(path / "trials.csv") {
        if (!csv) throw std::runtime_error("cannot open trials.csv");
        driver(cuInit(0), "cuInit");
        CUdevice device;
        driver(cuDeviceGet(&device, 0), "cuDeviceGet");
        driver(cuDevicePrimaryCtxRetain(&context, device), "retain primary context");
        driver(cuCtxSetCurrent(context), "set context");
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create non-default stream");
        library = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        auto create = reinterpret_cast<decltype(&NvOFAPICreateInstanceCuda)>(dlsym(library, "NvOFAPICreateInstanceCuda"));
        if (!create) throw std::runtime_error("missing NvOFAPICreateInstanceCuda");
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void*>(create), &info)) throw std::runtime_error("dladdr failed");
        std::cout << "NVOF library=" << std::filesystem::canonical(info.dli_fname).string()
                  << " API=2.0 width=1920 height=1080 grid=4 perf=SLOW temporal_hints=disabled"
                  << " IO=input=output=same_nondefault_nonblocking_stream\n";
        of(create(NV_OF_API_VERSION, &api), "create API");
        of(api.nvCreateOpticalFlowCuda(context, &handle), "create optical flow");
        of(api.nvOFSetIOCudaStreams(handle, stream, stream), "set IO streams");
        NV_OF_INIT_PARAMS init{};
        init.width = width; init.height = height; init.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
        init.mode = NV_OF_MODE_OPTICALFLOW; init.perfLevel = NV_OF_PERF_LEVEL_SLOW;
        of(api.nvOFInit(handle, &init), "NvOFInit");
        allocate(input, width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8);
        allocate(reference, width, height, NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8);
        allocate(output, grid_width, grid_height, NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2);
        ck(cudaMalloc(&replacement, c.size()), "allocate replacement");
        ck(cudaMalloc(&snap, output_bytes), "allocate snapshot");
        ck(cudaMemcpyAsync(replacement, c.data(), c.size(), cudaMemcpyHostToDevice, stream), "prepare replacement");
        ck(cudaMemcpy2DAsync(reference.ptr, reference.pitch, b.data(), width, width, height,
                             cudaMemcpyHostToDevice, stream), "prepare reference");
        // Load every audit kernel before measuring, including CUDA lazy loading.
        cudaFuncAttributes attributes{};
        ck(cudaFuncGetAttributes(&attributes, spin), "load spin");
        ck(cudaFuncGetAttributes(&attributes, copy_image), "load overwrite");
        ck(cudaFuncGetAttributes(&attributes, snapshot), "load snapshot");
        wait();
        for (const auto* image : {&a, &c}) {
            upload(*image); wait(); (void)execute(); wait();
            (image == &a ? ab : cb) = read();
        }
        if (ab == cb) throw std::runtime_error("audit input pairs have indistinguishable raw flow");
        save(dir / "input-A.gray", a); save(dir / "reference-B.gray", b); save(dir / "replacement-C.gray", c);
        save(dir / "baseline-AB.raw", ab); save(dir / "baseline-CB.raw", cb);
        std::cout << "input_A_sha256=" << sha(a) << " reference_B_sha256=" << sha(b)
                  << " replacement_C_sha256=" << sha(c) << '\n'
                  << "baseline_AB_sha256=" << sha(ab) << " baseline_CB_sha256=" << sha(cb)
                  << " baseline_different_bytes=" << differences(ab, cb) << '\n';
        csv << "item,spin_ms,control,trial,host_ms,different_bytes,equals_AB,equals_CB,observed_sha256,expected_sha256\n"
            << std::setprecision(9);
    }
    ~Audit() {
        if (stream) cudaStreamSynchronize(stream);
        if (snap) cudaFree(snap);
        if (replacement) cudaFree(replacement);
        for (auto* buf : {&input, &reference, &output}) if (buf->handle) api.nvOFDestroyGPUBufferCuda(buf->handle);
        if (handle) api.nvOFDestroy(handle);
        if (library) dlclose(library);
        if (stream) cudaStreamDestroy(stream);
        if (context) cuDevicePrimaryCtxRelease(0);
    }
    bool group(int item, double ms, bool control) {
        std::size_t bad = 0;
        std::vector<double> host;
        for (int trial = 0; trial < trials; ++trial) {
            // Baseline input is fully prepared before the tested interval.
            upload(a);
            ck(cudaMemset2DAsync(output.ptr, output.pitch, 0x5a, grid_width * sizeof(short2), grid_height, stream), "poison output");
            ck(cudaMemsetAsync(snap, 0xa5, output_bytes, stream), "poison snapshot");
            wait();
            spin<<<1, 1, 0, stream>>>(static_cast<unsigned long long>(ms * 1e6));
            if (item == 1) copy_image<<<(width * height + 255) / 256, 256, 0, stream>>>(replacement, input.ptr, input.pitch, width, height);
            ck(cudaGetLastError(), "enqueue pre-Execute audit kernels");
            // Control closes exactly the dependency being challenged.
            if (control && (item == 1 || item == 4)) wait();
            const double elapsed = execute();
            host.push_back(elapsed); all_host.push_back(elapsed);
            if (control && (item == 2 || item == 3)) wait();
            if (item == 2) copy_image<<<(width * height + 255) / 256, 256, 0, stream>>>(replacement, input.ptr, input.pitch, width, height);
            if (item == 3) snapshot<<<(output_bytes + 255) / 256, 256, 0, stream>>>(output.ptr, output.pitch, snap, grid_width * sizeof(short2), grid_height);
            ck(cudaGetLastError(), "enqueue post-Execute audit kernels");
            wait();
            const auto completed = read();
            Bytes observed = completed;
            if (item == 3) {
                ck(cudaMemcpyAsync(observed.data(), snap, output_bytes, cudaMemcpyDeviceToHost, stream), "read snapshot");
                wait();
            }
            const auto& expected = item == 1 ? cb : (item == 3 ? completed : ab);
            const auto different = differences(observed, expected);
            bad += different != 0;
            csv << 'A' << item << ',' << ms << ',' << control << ',' << trial << ',' << elapsed << ',' << different
                << ',' << (observed == ab) << ',' << (observed == cb) << ',' << sha(observed) << ',' << sha(expected) << '\n';
            csv.flush();
            std::cout << 'A' << item << " spin_ms=" << ms << " control=" << control << " trial=" << trial
                      << " host_ms=" << elapsed << " different_bytes=" << different
                      << " equals_AB=" << (observed == ab) << " equals_CB=" << (observed == cb) << '\n';
            if (different) {
                const auto pos = std::mismatch(observed.begin(), observed.end(), expected.begin()).first - observed.begin();
                std::cout << "FIRST_DIFFERENCE byte=" << pos << " grid_x=" << (pos / 4) % grid_width
                          << " grid_y=" << pos / (grid_width * 4) << " observed=" << unsigned(observed[pos])
                          << " expected=" << unsigned(expected[pos]) << '\n';
                if (bad == 1) {
                    const auto label = "A" + std::to_string(item) + "-" + std::to_string(ms) + (control ? "-control" : "-order");
                    save(dir / (label + "-first-observed.raw"), observed);
                    save(dir / (label + "-first-expected.raw"), expected);
                }
            }
        }
        std::cout << "SUMMARY A" << item << " spin_ms=" << ms << " control=" << control
                  << " violations=" << bad << '/' << trials << " A5_host_median_ms=" << quantile(host, .5)
                  << " A5_host_p95_ms=" << quantile(host, .95) << std::endl;
        return bad == 0;
    }
};
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: ofa_audit NEW_BUILD_PAYLOAD_DIRECTORY");
        if (!std::filesystem::create_directory(argv[1])) throw std::runtime_error("output directory must be new");
        Audit audit(argv[1]);
        std::cout << std::fixed << std::setprecision(6);
        for (int item = 1; item <= 4; ++item) {
            for (int group = 0; group < 3; ++group) {
                if (!audit.group(item, group == 0 ? .2 : 20., group == 2)) {
                    std::cout << "A5 executed_calls=" << audit.all_host.size()
                              << " host_median_ms=" << quantile(audit.all_host, .5)
                              << " host_p95_ms=" << quantile(audit.all_host, .95) << '\n'
                              << "AUDIT FAIL; stop at first violating group; remaining groups unexecuted\n";
                    return 3;
                }
            }
        }
        std::cout << "A5 executed_calls=" << audit.all_host.size()
                  << " host_median_ms=" << quantile(audit.all_host, .5)
                  << " host_p95_ms=" << quantile(audit.all_host, .95) << '\n'
                  << "A1-A4 synthetic PASS; A4 MOT17-10 flow_sequence comparison remains required\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ofa_audit: " << e.what() << '\n'; return 1;
    }
}

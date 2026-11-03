// Standalone stream-order check for nvjpegDecodeBatched (CUDA + nvJPEG only).
//
// Per trial, on one non-blocking stream S:
//   1. decode A into DST; record an event; host waits for it (A is complete)
//   2. enqueue: spin kernel (SPIN_MS) -> copy kernel DST -> SNAP
//   3. without any further wait, call nvjpegDecodeBatched(B -> DST, S)
//   4. synchronize S; download SNAP
// CUDA stream order puts the copy in step 2 before every GPU write of step 3,
// so SNAP must equal decode(A). A SNAP that differs from A means decode B
// wrote DST before earlier work on S had finished.
// mode "control" adds cudaStreamSynchronize(S) before step 3.
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void cu(cudaError_t s, const char* op) {
    if (s != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(s));
}
static void nj(nvjpegStatus_t s, const char* op) {
    if (s != NVJPEG_STATUS_SUCCESS) throw std::runtime_error(std::string(op) + ": nvjpeg status " + std::to_string(s));
}
static std::vector<unsigned char> read_file(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("open ") + path);
    return {std::istreambuf_iterator<char>(in), {}};
}
__global__ void spin(unsigned long long ns) {
    unsigned long long start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now)); } while (now - start < ns);
}
__global__ void copy(const unsigned char* src, unsigned char* dst, std::size_t n) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n; i += std::size_t(gridDim.x) * blockDim.x)
        dst[i] = src[i];
}

int main(int argc, char** argv) {
    try {
        if (argc != 8) throw std::runtime_error("usage: repro_order hw|hybrid default|no-pipeline order|control SPIN_MS TRIALS A.jpg B.jpg");
        const std::string backend = argv[1], flags = argv[2], mode = argv[3];
        if ((backend != "hw" && backend != "hybrid") || (flags != "default" && flags != "no-pipeline") ||
            (mode != "order" && mode != "control")) throw std::runtime_error("invalid backend, flags or mode");
        const double spin_ms = std::atof(argv[4]);
        const int trials = std::atoi(argv[5]);
        if (spin_ms <= 0 || trials < 1) throw std::runtime_error("SPIN_MS and TRIALS must be positive");
        const auto jpeg_a = read_file(argv[6]), jpeg_b = read_file(argv[7]);

        cudaStream_t s{};
        cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        nvjpegHandle_t handle{};
        nvjpegJpegState_t state{};
        nj(nvjpegCreateEx(backend == "hw" ? NVJPEG_BACKEND_HARDWARE : NVJPEG_BACKEND_HYBRID, nullptr, nullptr,
                          flags == "default" ? NVJPEG_FLAGS_DEFAULT : NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE, &handle), "nvjpegCreateEx");
        nj(nvjpegJpegStateCreate(handle, &state), "nvjpegJpegStateCreate");
        nj(nvjpegDecodeBatchedInitialize(handle, state, 1, 1, NVJPEG_OUTPUT_RGB), "nvjpegDecodeBatchedInitialize");

        int components = 0, w[NVJPEG_MAX_COMPONENT]{}, h[NVJPEG_MAX_COMPONENT]{}, wb[NVJPEG_MAX_COMPONENT]{}, hb[NVJPEG_MAX_COMPONENT]{};
        nvjpegChromaSubsampling_t sub{};
        nj(nvjpegGetImageInfo(handle, jpeg_a.data(), jpeg_a.size(), &components, &sub, w, h), "info A");
        nj(nvjpegGetImageInfo(handle, jpeg_b.data(), jpeg_b.size(), &components, &sub, wb, hb), "info B");
        if (w[0] != wb[0] || h[0] != hb[0]) throw std::runtime_error("A and B must have the same dimensions");
        const std::size_t plane = std::size_t(w[0]) * h[0], bytes = 3 * plane;

        unsigned char *in_a{}, *in_b{}, *dst{}, *snap{}, *host{};
        cu(cudaMallocHost(&in_a, jpeg_a.size()), "pinned A");
        cu(cudaMallocHost(&in_b, jpeg_b.size()), "pinned B");
        cu(cudaMallocHost(&host, bytes), "pinned download");
        cu(cudaMalloc(&dst, bytes), "DST");
        cu(cudaMalloc(&snap, bytes), "SNAP");
        std::memcpy(in_a, jpeg_a.data(), jpeg_a.size());
        std::memcpy(in_b, jpeg_b.data(), jpeg_b.size());
        nvjpegImage_t out{};
        for (int c = 0; c < 3; ++c) { out.channel[c] = dst + c * plane; out.pitch[c] = std::size_t(w[0]); }
        const unsigned char* a_ptr = in_a; const unsigned char* b_ptr = in_b;
        const std::size_t a_len = jpeg_a.size(), b_len = jpeg_b.size();
        auto decode = [&](const unsigned char* const* p, const std::size_t* n) {
            nj(nvjpegDecodeBatched(handle, state, p, n, &out, s), "nvjpegDecodeBatched");
        };
        auto download = [&](const unsigned char* src) {
            cu(cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost, s), "download");
            cu(cudaStreamSynchronize(s), "download sync");
            return std::vector<unsigned char>(host, host + bytes);
        };

        decode(&a_ptr, &a_len); cu(cudaStreamSynchronize(s), "ref A");
        const auto ref_a = download(dst);
        decode(&b_ptr, &b_len); cu(cudaStreamSynchronize(s), "ref B");
        const auto ref_b = download(dst);
        std::size_t ab = 0;
        for (std::size_t i = 0; i < bytes; ++i) ab += ref_a[i] != ref_b[i];

        int major = 0, minor = 0, patch = 0, driver = 0, runtime = 0;
        nvjpegGetProperty(MAJOR_VERSION, &major); nvjpegGetProperty(MINOR_VERSION, &minor); nvjpegGetProperty(PATCH_LEVEL, &patch);
        cudaDriverGetVersion(&driver); cudaRuntimeGetVersion(&runtime);
        cudaDeviceProp prop{}; cu(cudaGetDeviceProperties(&prop, 0), "device");
        std::printf("device=\"%s\" sm=%d%d cuda_driver=%d cuda_runtime=%d nvjpeg=%d.%d.%d\n", prop.name, prop.major, prop.minor, driver, runtime, major, minor, patch);
        std::printf("backend=%s flags=%s mode=%s spin_ms=%g trials=%d image=%dx%d bytes=%zu bytes_differing_A_B=%zu\n",
                    backend.c_str(), flags.c_str(), mode.c_str(), spin_ms, trials, w[0], h[0], bytes, ab);

        cudaEvent_t a_done{};
        cu(cudaEventCreateWithFlags(&a_done, cudaEventDisableTiming), "event");
        int violated = 0;
        for (int t = 0; t < trials; ++t) {
            decode(&a_ptr, &a_len);
            cu(cudaEventRecord(a_done, s), "record A");
            cu(cudaEventSynchronize(a_done), "wait A");
            spin<<<1, 1, 0, s>>>(static_cast<unsigned long long>(spin_ms * 1e6));
            copy<<<256, 256, 0, s>>>(dst, snap, bytes);
            cu(cudaGetLastError(), "launch");
            if (mode == "control") cu(cudaStreamSynchronize(s), "control sync");
            const auto t0 = std::chrono::steady_clock::now();
            decode(&b_ptr, &b_len);
            const double call_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            cu(cudaStreamSynchronize(s), "trial sync");
            const auto got = download(snap);
            std::size_t not_a = 0, equal_b = 0;
            for (std::size_t i = 0; i < bytes; ++i) if (got[i] != ref_a[i]) { ++not_a; equal_b += got[i] == ref_b[i]; }
            violated += not_a != 0;
            std::printf("trial=%d decode_B_call_ms=%.3f snap_bytes_not_A=%zu of_which_equal_B=%zu %s\n",
                        t, call_ms, not_a, equal_b, not_a ? "ORDER_VIOLATED" : "ordered");
        }
        std::printf("result=%s violated=%d/%d\n", violated ? "ORDER_VIOLATED" : "ordered", violated, trials);
        cu(cudaStreamSynchronize(s), "final sync");
        cudaEventDestroy(a_done); cudaFree(snap); cudaFree(dst); cudaFreeHost(host); cudaFreeHost(in_b); cudaFreeHost(in_a);
        nvjpegJpegStateDestroy(state); nvjpegDestroy(handle); cudaStreamDestroy(s);
        return violated ? 3 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

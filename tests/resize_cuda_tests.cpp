#include "saccade/resize_cuda.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void cu(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class F> void reject(F call) {
    bool caught = false;
    try { call(); } catch (const std::invalid_argument&) { caught = true; }
    check(caught, "invalid resize argument accepted");
}
struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { cu(cudaMalloc(&p, bytes)); }
    ~Device() { if (p != nullptr) (void)cudaFree(p); }
    Device(const Device&) = delete;
    template<class T> T* as() { return static_cast<T*>(p); }
};
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cu(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { (void)cudaStreamDestroy(value); }
};

// Independent binary64 mathematical reference. Construct every contributing
// source coordinate directly, normalize the clipped triangle and convolve the
// two dimensions. The production kernels use FP32 and a horizontal intermediate.
std::vector<double> reference(const saccade::ResizeShape& s, const std::vector<std::uint8_t>& input) {
    std::vector<double> out(saccade::resize_output_elements(s), 0.0);
    const double sx = static_cast<double>(s.input_width) / s.output_width;
    const double sy = static_cast<double>(s.input_height) / s.output_height;
    const auto weights = [](int index, int size, double scale) {
        std::vector<std::pair<int, double>> result;
        const double center = scale * (index + 0.5), support = std::max(scale, 1.0);
        double total = 0.0;
        const int first = std::max(static_cast<int>(center - support + 0.5), 0);
        const int last = std::min(static_cast<int>(center + support + 0.5), size);
        for (int p = first; p < last; ++p) {
            const double w = std::max(0.0, 1.0 - std::abs((p + 0.5 - center) / support));
            result.emplace_back(p, w); total += w;
        }
        for (auto& item : result) item.second /= total;
        return result;
    };
    std::vector<std::vector<std::pair<int, double>>> wx, wy;
    for (int x = 0; x < s.output_width; ++x) wx.push_back(weights(x, s.input_width, sx));
    for (int y = 0; y < s.output_height; ++y) wy.push_back(weights(y, s.input_height, sy));
    for (int c = 0; c < 3; ++c) for (int y = 0; y < s.output_height; ++y)
        for (int x = 0; x < s.output_width; ++x) {
            double sum = 0.0;
            for (const auto& [iy, yw] : wy[y]) for (const auto& [ix, xw] : wx[x])
                sum += input[(static_cast<std::size_t>(c) * s.input_height + iy) * s.input_width + ix] / 255.0 * xw * yw;
            out[(static_cast<std::size_t>(c) * s.output_height + y) * s.output_width + x] = sum;
        }
    return out;
}

std::vector<float> run(const saccade::ResizeShape& s, const std::vector<std::uint8_t>& input, bool graph = false) {
    Stream stream;
    const auto ni = saccade::resize_input_bytes(s), no = saccade::resize_output_elements(s), nw = saccade::resize_workspace_elements(s);
    Device di(ni), dout(no * sizeof(float)), dw(nw * sizeof(float));
    cu(cudaMemcpyAsync(di.p, input.data(), ni, cudaMemcpyHostToDevice, stream.value));
    cu(cudaMemsetAsync(dout.p, 0xff, no * sizeof(float), stream.value));
    const auto launch = [&] { saccade::resize_rgb_chw_cuda(s, di.as<std::uint8_t>(), ni,
        dout.as<float>(), no, dw.as<float>(), nw, stream.value); };
    launch();
    std::vector<float> out(no), repeat(no);
    cu(cudaMemcpyAsync(out.data(), dout.p, no * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    cu(cudaStreamSynchronize(stream.value));
    launch();
    cu(cudaMemcpyAsync(repeat.data(), dout.p, no * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    cu(cudaStreamSynchronize(stream.value));
    check(std::memcmp(out.data(), repeat.data(), no * sizeof(float)) == 0, "resize repeat differs");
    if (graph) {
        cudaGraph_t capture = nullptr; cudaGraphExec_t executable = nullptr;
        cu(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal));
        launch();
        cu(cudaStreamEndCapture(stream.value, &capture));
        cu(cudaGraphInstantiate(&executable, capture, 0));
        cu(cudaGraphLaunch(executable, stream.value));
        cu(cudaMemcpyAsync(repeat.data(), dout.p, no * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
        cu(cudaStreamSynchronize(stream.value));
        check(std::memcmp(out.data(), repeat.data(), no * sizeof(float)) == 0, "resize CUDA graph differs");
        cu(cudaGraphExecDestroy(executable)); cu(cudaGraphDestroy(capture));
    }
    return out;
}

void argument_tests() {
    const saccade::ResizeShape s{7, 5, 3, 2};
    const auto ni = saccade::resize_input_bytes(s), no = saccade::resize_output_elements(s), nw = saccade::resize_workspace_elements(s);
    Device di(ni), dout(no * sizeof(float)), dw(nw * sizeof(float)); Stream stream;
    const auto invoke = [&](const saccade::ResizeShape& shape, const std::uint8_t* input, std::size_t ic,
                            float* output, std::size_t oc, float* work, std::size_t wc, cudaStream_t st) {
        saccade::resize_rgb_chw_cuda(shape, input, ic, output, oc, work, wc, st);
    };
    for (auto bad : {saccade::ResizeShape{0, 5, 3, 2}, saccade::ResizeShape{-1, 5, 3, 2},
                     saccade::ResizeShape{7, 5, 65536, 2}, saccade::ResizeShape{65535, 65535, 3, 2}})
        reject([&] { (void)saccade::resize_input_bytes(bad); });
    reject([&] { invoke(s, di.as<std::uint8_t>(), ni-1, dout.as<float>(), no, dw.as<float>(), nw, stream.value); });
    reject([&] { invoke(s, di.as<std::uint8_t>(), ni, dout.as<float>(), no-1, dw.as<float>(), nw, stream.value); });
    reject([&] { invoke(s, di.as<std::uint8_t>(), ni, dout.as<float>(), no, dw.as<float>(), nw-1, stream.value); });
    for (auto st : {cudaStream_t{}, cudaStreamLegacy, cudaStreamPerThread})
        reject([&] { invoke(s, di.as<std::uint8_t>(), ni, dout.as<float>(), no, dw.as<float>(), nw, st); });
    reject([&] { invoke(s, nullptr, ni, dout.as<float>(), no, dw.as<float>(), nw, stream.value); });
    reject([&] { invoke(s, di.as<std::uint8_t>(), ni, nullptr, no, dw.as<float>(), nw, stream.value); });
    reject([&] { invoke(s, di.as<std::uint8_t>(), ni, dout.as<float>(), no, nullptr, nw, stream.value); });
    reject([&] { invoke(s, di.as<std::uint8_t>(), ni, di.as<float>(), no, dw.as<float>(), nw, stream.value); });
    reject([&] { invoke(s, di.as<std::uint8_t>(), ni, dout.as<float>(), no, dout.as<float>(), nw, stream.value); });
    reject([&] { invoke(s, di.as<std::uint8_t>(), ni, reinterpret_cast<float*>(dout.as<std::uint8_t>()+1), no, dw.as<float>(), nw, stream.value); });
    std::vector<std::uint8_t> host(ni);
    reject([&] { invoke(s, host.data(), ni, dout.as<float>(), no, dw.as<float>(), nw, stream.value); });
}

void held_callback(void* payload) {
    auto& release = *static_cast<std::atomic<bool>*>(payload);
    while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
}
void held_stream_test() {
    const saccade::ResizeShape s{8, 8, 4, 4};
    const auto ni = saccade::resize_input_bytes(s), no = saccade::resize_output_elements(s), nw = saccade::resize_workspace_elements(s);
    Device di(ni), dout(no*sizeof(float)), dw(nw*sizeof(float)); Stream stream;
    cu(cudaMemsetAsync(di.p, 127, ni, stream.value)); cu(cudaStreamSynchronize(stream.value));
    cudaEvent_t done = nullptr; cu(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    std::atomic<bool> release{false};
    cu(cudaLaunchHostFunc(stream.value, held_callback, &release));
    try {
        saccade::resize_rgb_chw_cuda(s, di.as<std::uint8_t>(), ni, dout.as<float>(), no, dw.as<float>(), nw, stream.value);
        cu(cudaEventRecord(done, stream.value));
        const auto state = cudaEventQuery(done);
        release.store(true, std::memory_order_release);
        check(state == cudaErrorNotReady, "resize did not remain ordered behind supplied stream");
        cu(cudaStreamSynchronize(stream.value));
        cu(cudaEventDestroy(done));
    } catch (...) {
        release.store(true, std::memory_order_release);
        (void)cudaStreamSynchronize(stream.value); (void)cudaEventDestroy(done); throw;
    }
}

void coordinate_contraction_regression() {
    // Pinned torch 2.11.0+cu130 CUDA reference: uint8 (y%2)*173 in all
    // channels, float().div(255), bilinear AA align_corners=False to (640,1).
    // RGB SHA256 df6d6e96b6f74be87485fcfd4c5c21a51d04c772e6ff9d63c1fa7d3b5612bc69
    // FP32 SHA256 58cbc7cf049cedc76c4adf914af315ce035c7f43576286dac99331eeb5dcae9b
    // Rounding center before xmin-center suppresses nvcc's legal FMA and makes
    // this case fail the policy, despite integer-ratio real images agreeing.
    const saccade::ResizeShape s{1, 721, 1, 640};
    std::vector<std::uint8_t> input(saccade::resize_input_bytes(s));
    const std::size_t plane = static_cast<std::size_t>(s.input_width)*s.input_height;
    for (int y = 0; y < s.input_height; ++y) for (int x = 0; x < s.input_width; ++x) {
        const auto p = static_cast<std::size_t>(y)*s.input_width+x;
        input[p] = static_cast<std::uint8_t>(y%2*173);
        input[plane+p] = static_cast<std::uint8_t>(y%2*173);
        input[2*plane+p] = static_cast<std::uint8_t>(y%2*173);
    }
    const auto out = run(s, input);
    struct Expected { int y, x; std::uint32_t bits; };
    const Expected expected[]{{635,0,0x3ec0f4c8}, {455,0,0x3ed709a8},
                              {639,0,0x3dd2828e}, {320,0,0x3ebf37a8}, {361,0,0x3ef39fb3}};
    const double nu = 52*std::ldexp(1.0,-24), atol = 2*nu/(1-nu);
    for (const auto& point : expected) {
        const float value = out[2*std::size_t{640}+point.y];
        check(std::abs(double(value)-std::bit_cast<float>(point.bits)) <= atol,
              "odd-ratio coordinate contraction differs from fixed PyTorch reference");
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool skip_held = argc == 2 && std::string(argv[1]) == "--no-held-stream";
        check(argc == 1 || skip_held, "usage: resize_cuda_tests [--no-held-stream]");
        std::cout << saccade::resize_cuda_identity() << saccade::resize_cuda_load_kernels();
        argument_tests();
        std::size_t cases = 0; double worst = 0.0;
        const std::vector<saccade::ResizeShape> shapes{
            {1,1,1,1}, {1,1,11,13}, {1,17,7,3}, {19,1,3,7}, {2,2,7,9},
            {4,1,2,1}, {2,1,4,1}, {31,17,31,17}, {31,17,31,5}, {31,17,9,17},
            {193,257,37,41}, {101,77,19,13}, {5,7,23,3}, {65535,1,1,1},
            {1920,1080,640,640}, {720,576,640,640}, {37,29,640,640}};
        for (const auto& shape : shapes) for (int pattern = 0; pattern < 4; ++pattern) {
            std::vector<std::uint8_t> input(saccade::resize_input_bytes(shape));
            std::uint32_t random = 0xcafef00d;
            for (std::size_t i = 0; i < input.size(); ++i) {
                random = 1664525u * random + 1013904223u;
                if (pattern == 0) input[i] = static_cast<std::uint8_t>((i / (static_cast<std::size_t>(shape.input_width)*shape.input_height))*91);
                if (pattern == 1) input[i] = static_cast<std::uint8_t>(((i % shape.input_width + i / shape.input_width) % 2) * 255);
                if (pattern == 2) input[i] = i % 17 == 0 ? 255 : 0;
                if (pattern == 3) input[i] = static_cast<std::uint8_t>(random >> 24);
            }
            const auto actual = run(shape, input, pattern == 3 && shape.input_width == 193);
            const auto expected = reference(shape, input);
            double max_error = 0.0;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                check(std::isfinite(actual[i]), "nonfinite or unwritten resize output");
                max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
            }
            worst = std::max(worst, max_error);
            // This mathematical reference also computes coordinates in double.
            // Bound FP32 center displacement separately from accumulation;
            // the training-reference policy fixes the SAME FP32 coordinates.
            const double u = std::ldexp(1.0, -24);
            const double taps = std::ceil(2*std::max(1.0, static_cast<double>(shape.input_width)/shape.output_width))
                              + std::ceil(2*std::max(1.0, static_cast<double>(shape.input_height)/shape.output_height)) + 2;
            const double bound = 4*u*(shape.input_width+shape.input_height) + 8*u*(taps+16);
            check(max_error < bound, "resize differs from binary64 triangle reference");
            ++cases;
        }
        const auto hand = run({4,1,2,1}, {0,0,255,255, 0,0,255,255, 0,0,255,255});
        check(std::abs(hand[0]-1.0/7.0)<1e-7 && std::abs(hand[1]-6.0/7.0)<1e-7, "antialias truncated edge hand case");
        const auto up = run({2,1,4,1}, {0,255,0,255,0,255});
        for (int i = 0; i < 4; ++i) check(std::abs(up[i] - std::vector<double>{0,0.25,0.75,1}[i]) < 1e-7, "upsample half-pixel hand case");
        coordinate_contraction_regression();
        if (!skip_held) held_stream_test();
        std::cout << "resize_cuda_tests PASS cases=" << cases << " max_abs_binary64=" << worst
                  << " held_stream=" << (skip_held ? "UNEXECUTED" : "PASS") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "resize_cuda_tests FAIL: " << error.what() << '\n'; return 1;
    }
}

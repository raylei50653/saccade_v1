#include "saccade/preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
unsigned checks = 0;
void check(bool ok, const char* why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
void cu(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
template<class F> void reject(F&& call) {
    bool caught = false;
    try { call(); } catch (const std::invalid_argument&) { caught = true; }
    check(caught, "invalid RGB input was accepted");
}
struct Stream {
    cudaStream_t value = nullptr;
    Stream() { cu(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) { (void)cudaStreamSynchronize(value); (void)cudaStreamDestroy(value); } }
};
template<class T> struct Pinned {
    T* data = nullptr;
    explicit Pinned(std::size_t n) { cu(cudaMallocHost(&data, n * sizeof(T))); }
    ~Pinned() { if (data) (void)cudaFreeHost(data); }
};

// Independent scalar binary64 triangle-filter reference; the runtime's F02
// computes FP32 separable weights. This catches channel/shape/upload errors
// without using the production resize kernel as its own oracle.
std::vector<double> reference(const std::vector<std::uint8_t>& rgb, int w, int h, int ow, int oh) {
    const auto weights = [](int index, int size, double scale) {
        std::vector<std::pair<int, double>> result;
        const double center = scale * (index + 0.5), support = std::max(scale, 1.0);
        const int first = std::max(static_cast<int>(center - support + 0.5), 0);
        const int last = std::min(static_cast<int>(center + support + 0.5), size);
        double sum = 0;
        for (int p = first; p < last; ++p) {
            const double weight = std::max(0.0, 1.0 - std::abs((p + 0.5 - center) / support));
            result.emplace_back(p, weight); sum += weight;
        }
        for (auto& item : result) item.second /= sum;
        return result;
    };
    std::vector<double> out(static_cast<std::size_t>(3) * ow * oh);
    for (int c = 0; c < 3; ++c) for (int y = 0; y < oh; ++y) for (int x = 0; x < ow; ++x) {
        double value = 0;
        for (const auto& [iy, yw] : weights(y, h, static_cast<double>(h) / oh))
            for (const auto& [ix, xw] : weights(x, w, static_cast<double>(w) / ow))
                value += rgb[(static_cast<std::size_t>(c) * h + iy) * w + ix] / 255.0 * xw * yw;
        out[(static_cast<std::size_t>(c) * oh + y) * ow + x] = value;
    }
    return out;
}

void hold(void* value) {
    const auto& release = *static_cast<std::atomic<bool>*>(value);
    while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
}

saccade::PreprocessConfig config() {
    saccade::PreprocessConfig cfg;
    cfg.max_width = 31; cfg.max_height = 19;
    cfg.output_width = 7; cfg.output_height = 5;
    cfg.max_jpeg_bytes = 4; // RGB capacity is independent of compressed capacity.
    cfg.input = saccade::PreprocessInput::HostRgb;
    return cfg;
}

void reuse_test(bool held) {
    const auto cfg = config();
    Stream stream;
    saccade::NativePreprocessor preprocess(stream.value, cfg);
    check(preprocess.identity().find("backend=HOST_RGB_UPLOAD") != std::string::npos, "RGB identity backend");
    check(preprocess.identity().find("input=host-planar-rgb8") != std::string::npos, "RGB identity input");
    Pinned<std::uint8_t> actual_rgb(static_cast<std::size_t>(3) * cfg.max_width * cfg.max_height);
    Pinned<float> actual_images(static_cast<std::size_t>(3) * cfg.output_width * cfg.output_height);
    std::atomic<bool> release{false};
    if (held) cu(cudaLaunchHostFunc(stream.value, hold, &release));
    try {
        for (std::uint64_t frame = 0; frame < 32; ++frame) {
            const int w = frame % 2 ? 2 : 31, h = frame % 2 ? 3 : 19;
            std::vector<std::uint8_t> input(static_cast<std::size_t>(3) * w * h);
            for (std::size_t i = 0; i < input.size(); ++i)
                input[i] = static_cast<std::uint8_t>((i * 37 + frame * 13) % 256);
            const auto expected = input;
            const auto resized = reference(expected, w, h, cfg.output_width, cfg.output_height);
            const auto result = preprocess.enqueue_rgb(input, frame, w, h);
            check(result.frame == frame && result.mapping.original_width == w && result.mapping.original_height == h,
                  "RGB frame/mapping identity");
            check(result.mapping.model_width == cfg.output_width && result.mapping.model_height == cfg.output_height,
                  "RGB model shape");
            // Host input is owned before return even while upload is held.
            std::fill(input.begin(), input.end(), 0);
            release.store(true, std::memory_order_release);
            cu(cudaMemcpyAsync(actual_rgb.data, result.rgb, expected.size(), cudaMemcpyDeviceToHost, stream.value));
            cu(cudaMemcpyAsync(actual_images.data, result.images, resized.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
            preprocess.wait();
            cu(cudaStreamSynchronize(stream.value));
            check(std::equal(expected.begin(), expected.end(), actual_rgb.data), "RGB copy/channel ownership");
            for (std::size_t i = 0; i < resized.size(); ++i)
                check(std::abs(actual_images.data[i] - resized[i]) <= 3e-6, "RGB F02 differs from independent reference");
        }
        preprocess.close();
        preprocess.close();
        reject([&] { preprocess.wait(); });
    } catch (...) {
        release.store(true, std::memory_order_release);
        (void)cudaStreamSynchronize(stream.value);
        throw;
    }
}

// Queue an old-frame consumer before the next upload and receive it only after
// another frame is submitted. D ordering must protect both RGB and F02 output.
void delayed_consumer_test() {
    auto cfg = config();
    Stream stream;
    saccade::NativePreprocessor preprocess(stream.value, cfg);
    const std::size_t bytes = static_cast<std::size_t>(3) * cfg.max_width * cfg.max_height;
    const std::size_t elements = static_cast<std::size_t>(3) * cfg.output_width * cfg.output_height;
    Pinned<std::uint8_t> before(bytes);
    Pinned<float> before_images(elements);
    const std::vector<std::uint8_t> first(bytes, 255), second(bytes, 0);
    const auto old = preprocess.enqueue_rgb(first, 0, cfg.max_width, cfg.max_height);
    cu(cudaMemcpyAsync(before.data, old.rgb, bytes, cudaMemcpyDeviceToHost, stream.value));
    cu(cudaMemcpyAsync(before_images.data, old.images, elements * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    const auto current = preprocess.enqueue_rgb(second, 1, cfg.max_width, cfg.max_height);
    preprocess.wait();
    cu(cudaStreamSynchronize(stream.value));
    check(std::all_of(before.data, before.data + bytes, [](auto v) { return v == 255; }), "old RGB consumed after overwrite");
    check(std::all_of(before_images.data, before_images.data + elements, [](auto v) { return std::abs(v - 1.0f) < 1e-6f; }),
          "old model input consumed after overwrite");
    cu(cudaMemcpyAsync(before_images.data, current.images, elements * sizeof(float), cudaMemcpyDeviceToHost, stream.value));
    cu(cudaStreamSynchronize(stream.value));
    check(std::all_of(before_images.data, before_images.data + elements, [](auto v) { return v == 0.0f; }), "new frame not uploaded");
    preprocess.close();
}

void rejection_tests() {
    const auto cfg = config();
    Stream stream;
    const std::vector<std::uint8_t> rgb(3 * 2 * 3, 127);
    for (int test = 0; test < 7; ++test) {
        saccade::NativePreprocessor preprocess(stream.value, cfg);
        reject([&] {
            switch (test) {
                case 0: (void)preprocess.enqueue_rgb(rgb, 1, 2, 3); break;
                case 1: (void)preprocess.enqueue_rgb(rgb, 0, 0, 3); break;
                case 2: (void)preprocess.enqueue_rgb(rgb, 0, 32, 3); break;
                case 3: (void)preprocess.enqueue_rgb(rgb, 0, 2, 20); break;
                case 4: (void)preprocess.enqueue_rgb({rgb.data(), rgb.size() - 1}, 0, 2, 3); break;
                case 5: (void)preprocess.enqueue(rgb, 0); break;
                case 6: (void)preprocess.enqueue_rgb(rgb, 0, -1, 3); break;
            }
        });
        reject([&] { (void)preprocess.enqueue_rgb(rgb, 0, 2, 3); });
        preprocess.close();
    }
    auto bad = cfg;
    bad.input = static_cast<saccade::PreprocessInput>(27);
    reject([&] { saccade::NativePreprocessor invalid(stream.value, bad); });
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool skip_held = argc == 2 && std::string(argv[1]) == "--no-held-stream";
        check(argc == 1 || skip_held, "usage: host_rgb_preprocess_tests [--no-held-stream]");
        cu(cudaSetDevice(0));
        reuse_test(!skip_held);
        delayed_consumer_test();
        rejection_tests();
        std::cout << "host_rgb_preprocess_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "host_rgb_preprocess_tests: " << error.what() << '\n';
        return 1;
    }
}

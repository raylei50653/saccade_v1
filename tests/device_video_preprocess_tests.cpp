#include "saccade/preprocess.hpp"
#include "saccade/video_color_cuda.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace saccade;
unsigned checks = 0;
void check(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
void cu(cudaError_t error) { if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
template<class F> void reject(F call) {
    bool caught = false;
    try { call(); } catch (const std::invalid_argument&) { caught = true; }
    check(caught, "invalid device video input accepted");
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
struct Surface {
    DeviceVideoFrame view;
    std::vector<std::uint8_t> y, uv;
    cudaEvent_t ready = nullptr;
    explicit Surface(int w, int h, VideoSurfaceFormat fmt, cudaStream_t producer) {
        view.width = w; view.height = h; view.format = fmt;
        view.stream = producer;
        CUcontext context = nullptr;
        check(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context, "surface current context");
        view.context = context;
        const std::size_t sample = fmt == VideoSurfaceFormat::P010 ? 2 : 1;
        view.pitches = {static_cast<std::size_t>(w) * sample + 16,
                        static_cast<std::size_t>((w + 1) / 2) * 2 * sample + 16};
        y.assign(view.pitches[0] * h, 0xcd); uv.assign(view.pitches[1] * ((h + 1) / 2), 0xef);
        void* plane = nullptr;
        cu(cudaMalloc(&plane, y.size())); view.planes[0] = static_cast<std::uint8_t*>(plane);
        try {
            cu(cudaMalloc(&plane, uv.size())); view.planes[1] = static_cast<std::uint8_t*>(plane);
            cu(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming)); view.ready_event = ready;
        } catch (...) { (void)cudaFree(const_cast<std::uint8_t*>(view.planes[0])); throw; }
        const unsigned max = sample == 2 ? 1023 : 255;
        const auto put = [&](std::vector<std::uint8_t>& bytes, std::size_t offset, unsigned value) {
            if (sample == 1) bytes[offset] = static_cast<std::uint8_t>(value);
            else { const std::uint16_t word = static_cast<std::uint16_t>((value << 6) | 37); std::memcpy(bytes.data() + offset, &word, 2); }
        };
        for (int row = 0; row < h; ++row) for (int x = 0; x < w; ++x)
            put(y, row * view.pitches[0] + x * sample, static_cast<unsigned>((x * 173 + row * 97) % (max + 1)));
        for (int row = 0; row < (h + 1) / 2; ++row) for (int x = 0; x < (w + 1) / 2; ++x) {
            put(uv, row * view.pitches[1] + 2 * x * sample, static_cast<unsigned>((x * 149 + row * 223) % (max + 1)));
            put(uv, row * view.pitches[1] + (2 * x + 1) * sample, static_cast<unsigned>((max - x * 73 - row * 43) % (max + 1)));
        }
        cu(cudaMemcpy(const_cast<std::uint8_t*>(view.planes[0]), y.data(), y.size(), cudaMemcpyHostToDevice));
        cu(cudaMemcpy(const_cast<std::uint8_t*>(view.planes[1]), uv.data(), uv.size(), cudaMemcpyHostToDevice));
        cu(cudaEventRecord(ready, producer));
    }
    ~Surface() {
        if (ready) (void)cudaEventDestroy(ready);
        if (view.planes[1]) (void)cudaFree(const_cast<std::uint8_t*>(view.planes[1]));
        if (view.planes[0]) (void)cudaFree(const_cast<std::uint8_t*>(view.planes[0]));
    }
};

std::vector<std::uint8_t> reference(const Surface& surface) {
    const auto& f = surface.view;
    const bool ten = f.format == VideoSurfaceFormat::P010;
    const auto load = [&](const std::vector<std::uint8_t>& plane, std::size_t offset) {
        if (!ten) return static_cast<double>(plane[offset]);
        std::uint16_t value = 0; std::memcpy(&value, plane.data() + offset, 2);
        return static_cast<double>(value >> 6);
    };
    double kr = 0.299, kb = 0.114;
    if (f.matrix == VideoColorMatrix::Bt709) { kr = 0.2126; kb = 0.0722; }
    if (f.matrix == VideoColorMatrix::Bt2020) { kr = 0.2627; kb = 0.0593; }
    const double multiplier = ten ? 4 : 1, maximum = ten ? 1023 : 255;
    const std::size_t sample = ten ? 2 : 1, pixels = static_cast<std::size_t>(f.width) * f.height;
    std::vector<std::uint8_t> output(pixels * 3);
    for (int y = 0; y < f.height; ++y) for (int x = 0; x < f.width; ++x) {
        const double yy = (load(surface.y, y * f.pitches[0] + x * sample) - (f.full_range ? 0 : 16 * multiplier)) /
                           (f.full_range ? maximum : 219 * multiplier);
        const double cb = (load(surface.uv, (y / 2) * f.pitches[1] + (2 * (x / 2)) * sample) - 128 * multiplier) /
                           (f.full_range ? maximum : 224 * multiplier);
        const double cr = (load(surface.uv, (y / 2) * f.pitches[1] + (2 * (x / 2) + 1) * sample) - 128 * multiplier) /
                           (f.full_range ? maximum : 224 * multiplier);
        const double r = yy + 2 * (1 - kr) * cr, b = yy + 2 * (1 - kb) * cb;
        // Recover G through the luma equation, independently of kernel's
        // expanded coefficient expression.
        const double g = (yy - kr * r - kb * b) / (1 - kr - kb);
        const auto p = static_cast<std::size_t>(y) * f.width + x;
        for (const auto& [channel, value] : {std::pair{0, r}, std::pair{1, g}, std::pair{2, b}})
            output[channel * pixels + p] = static_cast<std::uint8_t>(std::nearbyint(std::clamp(value * 255, 0.0, 255.0)));
    }
    return output;
}

PreprocessConfig config() {
    PreprocessConfig cfg;
    cfg.max_width = 31; cfg.max_height = 19;
    cfg.output_width = 5; cfg.output_height = 3;
    cfg.input = PreprocessInput::DeviceVideo;
    return cfg;
}

void numeric_tests() {
    Stream producer, consumer;
    Pinned<std::uint8_t> rgb(3 * 31 * 19);
    Pinned<float> images(3 * 5 * 3);
    for (const auto format : {VideoSurfaceFormat::Nv12, VideoSurfaceFormat::P010})
      for (const auto matrix : {VideoColorMatrix::Bt601, VideoColorMatrix::Bt709, VideoColorMatrix::Bt2020})
       for (const bool full : {false, true}) {
        NativePreprocessor preprocess(consumer.value, config());
        check(preprocess.identity().find("backend=CUDA_DEVICE_VIDEO") != std::string::npos, "device input identity");
        for (std::uint64_t k = 0; k < 3; ++k) {
            auto surface = std::make_shared<Surface>(5, 3, format, producer.value);
            surface->view.matrix = matrix; surface->view.full_range = full;
            auto input = surface->view; input.owner = surface;
            const auto expected = reference(*surface);
            const auto output = preprocess.enqueue_device(input, k);
            input.owner.reset(); surface.reset();
            cu(cudaMemcpyAsync(rgb.data, output.rgb, expected.size(), cudaMemcpyDeviceToHost, consumer.value));
            cu(cudaMemcpyAsync(images.data, output.images, expected.size() * sizeof(float), cudaMemcpyDeviceToHost, consumer.value));
            preprocess.wait(); cu(cudaStreamSynchronize(consumer.value));
            check(output.frame == k && output.mapping.original_width == 5 && output.mapping.original_height == 3, "device mapping");
            for (std::size_t i = 0; i < expected.size(); ++i) {
                check(std::abs(static_cast<int>(rgb.data[i]) - expected[i]) <= 1, "device color differs from scalar reference");
                check(std::abs(images.data[i] - rgb.data[i] / 255.0f) <= 1e-6f, "device RGB did not enter F02");
            }
        }
        preprocess.close(); preprocess.close();
      }
}

void held(void* data) {
    const auto& release = *static_cast<std::atomic<bool>*>(data);
    while (!release.load(std::memory_order_acquire)) std::this_thread::yield();
}
void producer_and_owner_test() {
    Stream producer, consumer;
    auto surface = std::make_shared<Surface>(5, 3, VideoSurfaceFormat::Nv12, producer.value);
    cu(cudaStreamSynchronize(producer.value));
    NativePreprocessor preprocess(consumer.value, config());
    auto input = surface->view; input.owner = surface;
    std::weak_ptr<Surface> lifetime = surface;
    std::atomic<bool> release{false};
    cu(cudaLaunchHostFunc(producer.value, held, &release));
    try {
        // All YUV planes become neutral limited-range black after the held
        // producer. Missing ready-event wait would read the old test pattern.
        cu(cudaMemsetAsync(const_cast<std::uint8_t*>(input.planes[0]), 16, surface->y.size(), producer.value));
        cu(cudaMemsetAsync(const_cast<std::uint8_t*>(input.planes[1]), 128, surface->uv.size(), producer.value));
        cu(cudaEventRecord(surface->ready, producer.value));
        const auto output = preprocess.enqueue_device(input, 0);
        input.owner.reset(); surface.reset();
        check(!lifetime.expired(), "device surface owner released before completion");
        release.store(true, std::memory_order_release);
        preprocess.wait();
        check(lifetime.expired(), "device surface owner not retired after completion");
        Pinned<std::uint8_t> actual(45);
        cu(cudaMemcpyAsync(actual.data, output.rgb, 45, cudaMemcpyDeviceToHost, consumer.value));
        cu(cudaStreamSynchronize(consumer.value));
        check(std::all_of(actual.data, actual.data + 45, [](auto x) { return x == 0; }), "consumer missed producer readiness event");
        preprocess.close();
    } catch (...) {
        release.store(true, std::memory_order_release);
        (void)cudaStreamSynchronize(producer.value); (void)cudaStreamSynchronize(consumer.value);
        throw;
    }
}

void invalid_tests() {
    Stream producer, consumer;
    auto surface = std::make_shared<Surface>(5, 3, VideoSurfaceFormat::Nv12, producer.value);
    std::vector<std::uint8_t> host(45);
    for (int test = 0; test < 9; ++test) {
        NativePreprocessor preprocess(consumer.value, config());
        auto input = surface->view; input.owner = surface;
        switch (test) {
            case 0: input.owner.reset(); break;
            case 1: input.ready_event = nullptr; break;
            case 2: input.pitches[0] = 1; break;
            case 3: input.pitches[1] = 1; break;
            case 4: input.context = reinterpret_cast<void*>(1); break;
            case 5: input.device = -1; break;
            case 6: input.planes[0] = host.data(); break;
            case 7: input.width = 32; break;
            case 8: input.matrix = static_cast<VideoColorMatrix>(71); break;
        }
        reject([&] { (void)preprocess.enqueue_device(input, 0); });
        input = surface->view; input.owner = surface;
        reject([&] { (void)preprocess.enqueue_device(input, 0); });
        preprocess.close();
    }
    {
        NativePreprocessor preprocess(consumer.value, config());
        auto input = surface->view; input.owner = surface;
        reject([&] { (void)preprocess.enqueue_device(input, 1); });
        preprocess.close();
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const bool skip_held = argc == 2 && std::string(argv[1]) == "--no-held-stream";
        check(argc == 1 || skip_held, "usage: device_video_preprocess_tests [--no-held-stream]");
        cu(cudaSetDevice(0));
        numeric_tests();
        if (!skip_held) producer_and_owner_test();
        invalid_tests();
        std::cout << "device_video_preprocess_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "device_video_preprocess_tests: " << error.what() << '\n';
        return 1;
    }
}

#include "saccade/preprocess.hpp"
#include "saccade/file_io.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t checks = 0;
void check(bool ok, const std::string& why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
void cuda(cudaError_t rc, const char* call) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(rc));
}
template<class F> void rejects(F&& action, const std::string& contains) {
    bool threw = false;
    try { action(); }
    catch (const std::exception& e) {
        threw = true;
        check(std::string(e.what()).find(contains) != std::string::npos,
              "wrong error, expected " + contains + ", got " + e.what());
    }
    check(threw, "operation must reject: " + contains);
}
struct Stream {
    cudaStream_t value{};
    Stream() { cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create test stream"); }
    ~Stream() { if (value) (void)cudaStreamDestroy(value); }
    void close() { auto s = value; value = nullptr; cuda(cudaStreamDestroy(s), "destroy test stream"); }
};
struct Readback {
    unsigned char* rgb{};
    float* images{};
    std::size_t rgb_count, image_count;
    Readback(std::size_t r, std::size_t f) : rgb_count(r), image_count(f) {
        cuda(cudaMallocHost(&rgb, rgb_count), "allocate test RGB readback");
        try { cuda(cudaMallocHost(&images, image_count * sizeof(float)), "allocate test images readback"); }
        catch (...) { (void)cudaFreeHost(rgb); throw; }
    }
    ~Readback() { if (images) (void)cudaFreeHost(images); if (rgb) (void)cudaFreeHost(rgb); }
    void enqueue(const saccade::PreprocessedFrame& frame, cudaStream_t stream) {
        cuda(cudaMemcpyAsync(rgb, frame.rgb, rgb_count, cudaMemcpyDeviceToHost, stream), "copy test RGB");
        cuda(cudaMemcpyAsync(images, frame.images, image_count * sizeof(float), cudaMemcpyDeviceToHost, stream), "copy test images");
    }
    void close() {
        auto* f = images; images = nullptr; cuda(cudaFreeHost(f), "free test images");
        auto* b = rgb; rgb = nullptr; cuda(cudaFreeHost(b), "free test RGB");
    }
};
std::vector<std::uint8_t> bytes(const std::filesystem::path& path) {
    const auto s = saccade::read_text_file(path);
    return {s.begin(), s.end()};
}
saccade::PreprocessConfig small_config() {
    saccade::PreprocessConfig c;
    c.max_width = 48; c.max_height = 32;
    c.output_width = 35; c.output_height = 27;
    c.max_jpeg_bytes = 8192;
    return c;
}
void same(const Readback& a, const Readback& b, const std::string& label) {
    check(a.rgb_count == b.rgb_count && a.image_count == b.image_count, label + " dimensions");
    check(std::memcmp(a.rgb, b.rgb, a.rgb_count) == 0, label + " RGB bitwise");
    check(std::memcmp(a.images, b.images, a.image_count * sizeof(float)) == 0, label + " images bitwise");
}
void positive(const std::filesystem::path& root, Stream& stream) {
    auto encoded = bytes(root / "baseline.jpg");
    const auto preserved = encoded;
    const auto other = bytes(root / "alternate.jpg");
    const std::size_t count = 3 * 35 * 27;
    Readback first(count, count), repeated(count, count), interleaved(3 * 48 * 32, count), last(count, count);
    saccade::NativePreprocessor p(stream.value, small_config());
    const auto f0 = p.enqueue(encoded, 0, 35, 27);
    check(f0.frame == 0, "frame identity");
    check(f0.mapping.original_width == 35 && f0.mapping.original_height == 27 &&
          f0.mapping.model_width == 35 && f0.mapping.model_height == 27, "mapping dimensions");
    check(f0.mapping.scale_x == 1 && f0.mapping.scale_y == 1, "identity mapping");
    // The caller may immediately change/drop its vector; the session owns JPEG bytes.
    std::fill(encoded.begin(), encoded.end(), 0);
    first.enqueue(f0, stream.value);
    // These readbacks are queued AFTER p's F02 event, so the next enqueue must
    // preserve them by same-stream ordering, not just by waiting that event.
    repeated.enqueue(p.enqueue(preserved, 1), stream.value);
    interleaved.enqueue(p.enqueue(other, 2, 48, 32), stream.value);
    last.enqueue(p.enqueue(preserved, 3), stream.value);
    cuda(cudaStreamSynchronize(stream.value), "finish positive frame observations");
    p.wait();
    same(first, repeated, "repeated frame");
    same(first, last, "reuse after different-size frame");
    for (std::size_t i = 0; i < count; ++i) {
        const float expected = static_cast<float>(first.rgb[i]) / 255.0f;
        check(std::isfinite(first.images[i]) && std::abs(first.images[i] - expected) <= 1.3e-7f,
              "identity resize is float32 RGB normalization");
    }
    const std::size_t plane = 35 * 27;
    const auto sum = [&](std::size_t c) {
        return std::accumulate(first.rgb + c * plane, first.rgb + (c + 1) * plane, std::size_t{});
    };
    check(sum(0) > sum(1) && sum(1) > sum(2), "fixture channel order is planar R G B");
    const auto identity = p.identity();
    check(identity.find("backend=NVJPEG_BACKEND_HARDWARE") != std::string::npos &&
          identity.find("output=NVJPEG_OUTPUT_RGB") != std::string::npos &&
          identity.find("nvjpeg=") != std::string::npos, "runtime identity includes actual decode backend and version");
    p.close(); p.close();
    rejects([&] { (void)p.enqueue(preserved, 4); }, "closed");
    rejects([&] { p.wait(); }, "closed");
    first.close(); repeated.close(); interleaved.close(); last.close();
    std::cout << "case=positive frames=4 odd-size=35x27 alternate=48x32 bitwise-repeat=PASS\n";
}
void independent(const std::filesystem::path& root, Stream& a, Stream& b) {
    const auto data = bytes(root / "baseline.jpg");
    auto cfg = small_config(); cfg.output_width = 16; cfg.output_height = 12;
    saccade::NativePreprocessor x(a.value, cfg), y(b.value, cfg);
    Readback ra(3 * 35 * 27, 3 * 16 * 12), rb(3 * 35 * 27, 3 * 16 * 12);
    const auto fa = x.enqueue(data, 0);
    const auto fb = y.enqueue(data, 0);
    check(std::abs(fa.mapping.scale_x - 16.0/35) < 1e-15 &&
          std::abs(fa.mapping.scale_y - 12.0/27) < 1e-15 &&
          fa.mapping.scale_x != fa.mapping.scale_y, "stretch preserves distinct x/y edge scales");
    check(fa.rgb != fb.rgb && fa.images != fb.images, "sessions own independent buffers");
    ra.enqueue(fa, a.value); rb.enqueue(fb, b.value);
    cuda(cudaStreamSynchronize(a.value), "finish first independent stream");
    cuda(cudaStreamSynchronize(b.value), "finish second independent stream");
    same(ra, rb, "independent sessions");
    x.close(); y.close(); ra.close(); rb.close();
    std::cout << "case=independent-streams mapping=nonuniform bitwise=PASS\n";
}
void negatives(const std::filesystem::path& root, Stream& stream) {
    const auto good = bytes(root / "baseline.jpg");
    auto bad_case = [&](const std::string& label, auto&& action, saccade::PreprocessConfig cfg = small_config()) {
        saccade::NativePreprocessor p(stream.value, cfg);
        rejects([&] { action(p); }, label);
        rejects([&] { (void)p.enqueue(good, 0); }, "sequence has failed");
        rejects([&] { p.wait(); }, "sequence has failed");
        p.close(); p.close();
    };
    bad_case("empty", [&](auto& p) { (void)p.enqueue({}, 0); });
    const std::vector<std::uint8_t> non_jpeg{'n','o','t','j','p','e','g'};
    bad_case("SOI/EOI", [&](auto& p) { (void)p.enqueue(non_jpeg, 0); });
    auto truncated = good; truncated.resize(truncated.size() - 2);
    bad_case("SOI/EOI", [&](auto& p) { (void)p.enqueue(truncated, 0); });
    const std::vector<std::uint8_t> malformed{0xff, 0xd8, 0xff, 0xd9};
    bad_case("nvjpegJpegStreamParseHeader", [&](auto& p) { (void)p.enqueue(malformed, 0); });
    bad_case("baseline", [&](auto& p) { (void)p.enqueue(bytes(root / "progressive.jpg"), 0); });
    for (const auto* fixture : {"gray.jpg", "cmyk.jpg", "444.jpg"})
        bad_case("three-component", [&](auto& p) { (void)p.enqueue(bytes(root / fixture), 0); });
    auto cfg = small_config(); cfg.max_width = 34;
    bad_case("capacity", [&](auto& p) { (void)p.enqueue(good, 0); }, cfg);
    cfg = small_config(); cfg.max_height = 26;
    bad_case("capacity", [&](auto& p) { (void)p.enqueue(good, 0); }, cfg);
    cfg = small_config(); cfg.max_jpeg_bytes = good.size() - 1;
    bad_case("capacity", [&](auto& p) { (void)p.enqueue(good, 0); }, cfg);
    bad_case("sequence metadata", [&](auto& p) { (void)p.enqueue(good, 0, 34, 27); });
    bad_case("both", [&](auto& p) { (void)p.enqueue(good, 0, 35, 0); });
    bad_case("both", [&](auto& p) { (void)p.enqueue(good, 0, -1, 27); });
    bad_case("contiguous", [&](auto& p) { (void)p.enqueue(good, 1); });
    bad_case("contiguous", [&](auto& p) { (void)p.enqueue(good, 0); (void)p.enqueue(good, 0); });
    bad_case("contiguous", [&](auto& p) { (void)p.enqueue(good, 0); (void)p.enqueue(good, 2); });
    rejects([&] { saccade::NativePreprocessor p(nullptr, small_config()); }, "non-default");
    rejects([&] { saccade::NativePreprocessor p(cudaStreamLegacy, small_config()); }, "non-default");
    rejects([&] { saccade::NativePreprocessor p(cudaStreamPerThread, small_config()); }, "non-default");
    const auto invalid_cfg = [&](const saccade::PreprocessConfig& invalid, const char* why) {
        rejects([&] { saccade::NativePreprocessor p(stream.value, invalid); }, why);
    };
    cfg = small_config(); cfg.max_width = 0; invalid_cfg(cfg, "max dimensions");
    cfg = small_config(); cfg.max_height = 8193; invalid_cfg(cfg, "max dimensions");
    cfg = small_config(); cfg.output_width = 0; invalid_cfg(cfg, "output dimensions");
    cfg = small_config(); cfg.output_height = 4097; invalid_cfg(cfg, "output dimensions");
    cfg = small_config(); cfg.max_jpeg_bytes = 0; invalid_cfg(cfg, "max_jpeg_bytes");
    cfg = small_config(); cfg.max_jpeg_bytes = 256U * 1024 * 1024 + 1; invalid_cfg(cfg, "max_jpeg_bytes");
    std::cout << "case=host-rejections malformed/unsupported/capacity/metadata/order/config/default-stream/latch/closed=PASS\n";
}
void capture_rejected(const std::filesystem::path& root, Stream& stream) {
    const auto good = bytes(root / "baseline.jpg");
    saccade::NativePreprocessor p(stream.value, small_config());
    cudaGraph_t graph{};
    cuda(cudaStreamBeginCapture(stream.value, cudaStreamCaptureModeThreadLocal), "begin rejection capture");
    rejects([&] { (void)p.enqueue(good, 0); }, "graph capture");
    cuda(cudaStreamEndCapture(stream.value, &graph), "end rejection capture");
    if (graph) cuda(cudaGraphDestroy(graph), "destroy rejection capture");
    p.close();
    std::cout << "case=capture-rejected=PASS\n";
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: preprocess_tests FIXTURE_DIRECTORY");
        Stream a, b;
        const std::filesystem::path root(argv[1]);
        positive(root, a);
        independent(root, a, b);
        negatives(root, a);
        capture_rejected(root, a);
        a.close(); b.close();
        std::cout << "preprocess_tests checks=" << checks << " PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "preprocess_tests FAILED: " << e.what() << '\n';
        return 1;
    }
}

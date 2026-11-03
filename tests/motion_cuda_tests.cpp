#include "saccade/motion_cuda.hpp"
#include "motion_fixtures.hpp"

#ifdef SACCADE_MOTION_TEST_PREPROCESS
#include "saccade/preprocess.hpp"
#include "saccade/file_io.hpp"
#endif
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace saccade;
std::size_t checks = 0;
void ck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
bool bits(float a, float b) { return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b); }
struct Streams {
    cudaStream_t d = nullptr, m = nullptr;
    Streams() {
        ck(cudaStreamCreateWithFlags(&d, cudaStreamNonBlocking), "create D");
        ck(cudaStreamCreateWithFlags(&m, cudaStreamNonBlocking), "create M");
    }
    ~Streams() {
        if (d) { cudaStreamSynchronize(d); cudaStreamDestroy(d); }
        if (m) { cudaStreamSynchronize(m); cudaStreamDestroy(m); }
    }
};
template<class T> struct Buffer {
    T *device = nullptr, *host = nullptr;
    std::size_t count;
    explicit Buffer(std::size_t size) : count(size) {
        ck(cudaMalloc(&device, size * sizeof(T)), "allocate device buffer");
        ck(cudaMallocHost(&host, size * sizeof(T)), "allocate pinned buffer");
    }
    ~Buffer() { if (device) cudaFree(device); if (host) cudaFreeHost(host); }
    void upload(cudaStream_t stream) { ck(cudaMemcpyAsync(device, host, count * sizeof(T), cudaMemcpyHostToDevice, stream), "upload"); }
    void download(cudaStream_t stream) { ck(cudaMemcpyAsync(host, device, count * sizeof(T), cudaMemcpyDeviceToHost, stream), "download"); }
};
void compare(const MotionContext& actual, const MotionContext& expected) {
    check(actual.epoch == expected.epoch && actual.frame == expected.frame && actual.reference_frame == expected.reference_frame, "F06 identity differs");
    check(actual.status == expected.status, "F06 status differs");
    check(actual.peak_x == expected.peak_x && actual.peak_y == expected.peak_y, "F06 peak differs");
    if (!(std::abs(actual.pcr - expected.pcr) <= 1e-4f * std::abs(expected.pcr))) {
        std::cerr << "PCR GPU=" << actual.pcr << " CPU=" << expected.pcr << '\n';
        check(false, "F06 PCR relative error exceeds 1e-4");
    }
    check(true, "PCR tolerance");
    check(std::abs(actual.confidence - expected.confidence) <= 1e-4f, "F06 confidence differs");
    if (expected.confidence == 1)
        check(bits(actual.tx, expected.tx) && bits(actual.ty, expected.ty), "F06 saturated t differs bitwise");
    else
        check(std::abs(actual.tx - expected.tx) <= 1e-4f * std::max(1.f, std::abs(expected.tx)) &&
              std::abs(actual.ty - expected.ty) <= 1e-4f * std::max(1.f, std::abs(expected.ty)), "F06 attenuated t differs");
}
void image_case(const MotionImageKey& key, const std::vector<std::uint8_t>& input, Streams& streams) {
    const auto expected = motion_image_reference(input, key);
    const auto wx = hanning_window(key.width), wy = hanning_window(key.height);
    Buffer<std::uint8_t> rgb(input.size());
    Buffer<float> image(expected.size()), dx(wx.size()), dy(wy.size());
    std::copy(input.begin(), input.end(), rgb.host); std::copy(wx.begin(), wx.end(), dx.host); std::copy(wy.begin(), wy.end(), dy.host);
    rgb.upload(streams.d); dx.upload(streams.d); dy.upload(streams.d);
    motion_image_cuda(rgb.device, image.device, key, dx.device, dy.device, streams.d);
    image.download(streams.d); ck(cudaStreamSynchronize(streams.d), "wait F05");
    check(std::memcmp(image.host, expected.data(), expected.size() * sizeof(float)) == 0, "F05 not bitwise equal");
}
void f05(Streams& streams) {
    for (const auto shape : {std::array{32, 32, 4}, std::array{35, 39, 4}, std::array{17, 19, 2},
                             std::array{19, 21, 1}, std::array{101, 83, 3}}) {
        MotionConfig config; config.downscale = shape[2];
        const auto key = motion_image_key(shape[0], shape[1], config);
        image_case(key, motion_test::rgb(key), streams);
        image_case(key, std::vector<std::uint8_t>(static_cast<std::size_t>(shape[0]) * shape[1] * 3, 255), streams);
    }
}
std::vector<std::uint8_t> impulse(const MotionImageKey& key, int x, int y) {
    const auto plane = static_cast<std::size_t>(key.original_width) * key.original_height;
    std::vector<std::uint8_t> rgb(3 * plane);
    for (int c = 0; c < 3; ++c)
        for (int dy = 0; dy < key.downscale; ++dy)
            for (int dx = 0; dx < key.downscale; ++dx)
                rgb[c * plane + (y * key.downscale + dy) * key.original_width + x * key.downscale + dx] = 255;
    return rgb;
}
void sequence_case(Streams& streams, MotionConfig config, unsigned slots) {
    const auto key = motion_image_key(64, 64, config);
    const auto plane3 = static_cast<std::size_t>(64 * 64 * 3);
    const std::vector<std::array<int, 2>> positions{{4, 4}, {7, 4}, {4, 4}, {4, 7}, {4, 4}, {6, 2}, {2, 6},
                                                {7, 6}, {6, 1}, {6, 9}, {6, 1}, {14, 1}, {6, 1}, {-1, -1}, {-1, -1}, {4, 4}};
    Buffer<std::uint8_t> images(plane3 * positions.size());
    Buffer<MotionContext> outputs(positions.size());
    MotionCuda motion(key, config, 0x123456789abcdef0ULL, slots);
    std::vector<MotionContext> expected;
    std::vector<float> previous;
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto xy = positions[i];
        const auto rgb = xy[0] < 0 ? std::vector<std::uint8_t>(plane3) : impulse(key, xy[0], xy[1]);
        const auto current = motion_image_reference(rgb, key);
        expected.push_back(phase_correlation_reference(previous, current, key, config, 0x123456789abcdef0ULL, i));
        std::copy(rgb.begin(), rgb.end(), images.host + i * plane3);
        previous = current;
    }
    // Reuse one RAW allocation and MI slots while all host input staging stays immutable.
    for (std::size_t i = 0; i < positions.size(); ++i) {
        ck(cudaMemcpyAsync(images.device, images.host + i * plane3, plane3, cudaMemcpyHostToDevice, streams.d), "copy RAW");
        motion.enqueue_image(streams.d, images.device, i);
        motion.enqueue_estimate(streams.m, i, outputs.device + i);
        motion.enqueue_commit(streams.m, i);
    }
    outputs.download(streams.m); ck(cudaStreamSynchronize(streams.m), "wait sequence");
    for (std::size_t i = 0; i < positions.size(); ++i) compare(outputs.host[i], expected[i]);
    check(outputs.host[0].status == MotionStatus::uninitialized && outputs.host[0].pcr == 0 && outputs.host[0].tx == 0, "first frame");
    check(outputs.host[1].peak_x == 3 && outputs.host[2].peak_x == -3 && outputs.host[3].peak_y == 3 && outputs.host[4].peak_y == -3, "known signed impulse shifts");
    check(outputs.host[9].peak_y == 8 && outputs.host[10].peak_y == 8 && outputs.host[11].peak_x == 8 && outputs.host[12].peak_x == 8, "strict half wrap");
    check(outputs.host[14].status == MotionStatus::rejected && outputs.host[14].peak_x == 0 && outputs.host[14].peak_y == 0, "zero image/tie");
    check(outputs.host[15].reference_frame == 14, "rejected frame committed as reference");
    motion.close(); motion.close(); check(true, "close twice");
}
void numeric(Streams& streams) {
    const MotionConfig config{4, std::numeric_limits<float>::denorm_min(), .25f};
    const auto key = motion_image_key(64, 64, config);
    Buffer<std::uint8_t> rgb(3 * 64 * 64);
    Buffer<MotionContext> out(2);
    MotionCuda motion(key, config, 19);
    const auto input = impulse(key, 4, 4);
    std::copy(input.begin(), input.end(), rgb.host); rgb.upload(streams.d);
    for (int frame = 0; frame < 2; ++frame) {
        motion.enqueue_image(streams.d, rgb.device, frame);
        motion.enqueue_estimate(streams.m, frame, out.device + frame);
        if (frame == 0) motion.enqueue_commit(streams.m, frame);
    }
    out.download(streams.m); ck(cudaStreamSynchronize(streams.m), "wait numeric");
    check(out.host[0].status == MotionStatus::uninitialized && out.host[0].pcr == 0, "first numeric config");
    check(!std::isfinite(out.host[1].pcr), "nonfinite arithmetic did not report NUMERIC");
    motion.close();
}
void uncommitted(Streams& streams) {
    const MotionConfig config;
    const auto key = motion_image_key(64, 64, config);
    Buffer<std::uint8_t> rgb(3 * 64 * 64);
    Buffer<MotionContext> out(3);
    MotionCuda motion(key, config, 7);
    const std::array<std::array<int, 2>, 3> positions{{{4, 4}, {6, 4}, {4, 7}}};
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const auto input = impulse(key, positions[i][0], positions[i][1]);
        std::copy(input.begin(), input.end(), rgb.host); rgb.upload(streams.d);
        motion.enqueue_image(streams.d, rgb.device, i);
        motion.enqueue_estimate(streams.m, i, out.device + i);
        if (i != 1) motion.enqueue_commit(streams.m, i);
        ck(cudaStreamSynchronize(streams.d), "release host RGB");
    }
    out.download(streams.m); ck(cudaStreamSynchronize(streams.m), "wait uncommitted control");
    check(out.host[2].reference_frame == 0 && out.host[2].peak_x == 0 && out.host[2].peak_y == 3, "working estimate overwrote committed spectrum");
    motion.close();
}
#ifdef SACCADE_MOTION_TEST_PREPROCESS
void real(Streams& streams) {
    const MotionConfig config;
    const auto key = motion_image_key(1920, 1080, config);
    const auto plane3 = static_cast<std::size_t>(1920 * 1080 * 3);
    NativePreprocessor preprocess(streams.d);
    MotionCuda motion(key, config, 17);
    Buffer<std::uint8_t> rgb(plane3);
    Buffer<MotionContext> output(1);
    std::vector<float> previous;
    for (int frame = 0; frame < 2; ++frame) {
        const auto path = std::string("/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP/img1/") + (frame == 0 ? "000001.jpg" : "000002.jpg");
        const auto jpeg = read_text_file(path);
        const auto decoded = preprocess.enqueue({reinterpret_cast<const std::uint8_t*>(jpeg.data()), jpeg.size()}, frame, 1920, 1080);
        motion.enqueue_image(streams.d, decoded.rgb, frame);
        ck(cudaMemcpyAsync(rgb.host, decoded.rgb, plane3, cudaMemcpyDeviceToHost, streams.d), "copy decoded RGB");
        motion.enqueue_estimate(streams.m, frame, output.device);
        motion.enqueue_commit(streams.m, frame);
        output.download(streams.m);
        ck(cudaStreamSynchronize(streams.d), "wait RGB"); ck(cudaStreamSynchronize(streams.m), "wait real motion");
        const std::vector<std::uint8_t> input(rgb.host, rgb.host + plane3);
        image_case(key, input, streams);
        const auto current = motion_image_reference(input, key);
        const auto expected = phase_correlation_reference(previous, current, key, config, 17, frame);
        compare(output.host[0], expected);
        previous = current;
    }
    motion.close(); preprocess.close();
}
#endif
}
int main() {
    try {
        Streams streams;
        f05(streams);
        for (unsigned slots : {1u, 2u, 3u}) sequence_case(streams, {}, slots);
        sequence_case(streams, {4, 5, .5f}, 2);
        sequence_case(streams, {4, 100, .25f}, 2);
        uncommitted(streams); numeric(streams);
#ifdef SACCADE_MOTION_TEST_PREPROCESS
        real(streams);
#endif
        std::cout << "motion_cuda_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "motion_cuda_tests: " << e.what() << '\n'; return 1; }
}

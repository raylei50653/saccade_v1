#include "saccade/motion_cuda.hpp"
#include "saccade/cleanup.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace saccade {
namespace {
void ck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
void fft(cufftResult result, const char* operation) {
    if (result != CUFFT_SUCCESS) throw std::runtime_error(std::string(operation) + ": cuFFT " + std::to_string(result));
}
void explicit_stream(cudaStream_t stream) {
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        throw std::invalid_argument("motion requires an explicit non-default stream");
}
constexpr int kThreads = 256;
__global__ void image_kernel(const std::uint8_t* rgb, float* output, MotionImageKey key, const float* wx, const float* wy, const bool* run_stage) {
    if (run_stage && !*run_stage) return;
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::size_t>(key.width) * key.height) return;
    const int x = i % key.width, y = i / key.width, d = key.downscale;
    const auto plane = static_cast<std::size_t>(key.original_width) * key.original_height;
    float sum = 0;
    for (int dy = 0; dy < d; ++dy) {
        for (int dx = 0; dx < d; ++dx) {
            const auto p = static_cast<std::size_t>(y * d + dy) * key.original_width + x * d + dx;
            const float g = (0.299f * float(rgb[p]) + 0.587f * float(rgb[plane + p])) + 0.114f * float(rgb[2 * plane + p]);
            sum += g;
        }
    }
    const float v = (sum / float(d * d)) / 255.0f;
    output[i] = (v * wx[x]) * wy[y];
}
__global__ void reset_numeric(int* numeric) { *numeric = 0; }
__global__ void deliver(const cufftComplex* temporary, cufftComplex* working, std::size_t n, int* numeric,
                         const bool* run_stage, const Control* control) {
    if (run_stage && !*run_stage) return;
    if (control) working += control->working_slot * n;
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto v = temporary[i];
    working[i] = v;
    if (!isfinite(v.x) || !isfinite(v.y)) atomicOr(numeric, 1);
}
__global__ void cross_power(const cufftComplex* previous, const cufftComplex* current, cufftComplex* cross,
                            std::size_t n, int* numeric, const bool* run_stage, const Control* control) {
    if (run_stage && !*run_stage) return;
    if (control) {
        if (control->committed_frame < 0) return;
        previous += control->committed_slot * n;
        current += control->working_slot * n;
    }
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto a = previous[i], b = current[i];
    const double real = double(a.x) * b.x + double(a.y) * b.y;
    const double imag = double(a.x) * b.y - double(a.y) * b.x;
    const double magnitude = sqrt(real * real + imag * imag);
    const cufftComplex r{float(real / (magnitude + 1e-6)), float(imag / (magnitude + 1e-6))};
    if (!isfinite(real) || !isfinite(imag) || !isfinite(magnitude) || !isfinite(r.x) || !isfinite(r.y)) atomicOr(numeric, 1);
    cross[i] = r;
}
struct Reduction {
    double squares;
    float maximum;
    std::size_t index;
    int numeric;
};
__device__ Reduction combine(Reduction a, Reduction b) {
    a.squares += b.squares;
    if (b.maximum > a.maximum || (b.maximum == a.maximum && b.index < a.index)) {
        a.maximum = b.maximum; a.index = b.index;
    }
    a.numeric |= b.numeric;
    return a;
}
__global__ void reduce_correlation(const float* correlation, std::size_t n, Reduction* partial, const bool* run_stage) {
    if (run_stage && !*run_stage) return;
    __shared__ Reduction values[kThreads];
    const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    Reduction value{0, -CUDART_INF_F, i, 0};
    if (i < n) {
        const float r = correlation[i];
        value = {double(r) * r, r, i, !isfinite(r)};
    }
    values[threadIdx.x] = value;
    __syncthreads();
    for (int offset = kThreads / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) values[threadIdx.x] = combine(values[threadIdx.x], values[threadIdx.x + offset]);
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = values[0];
}
__global__ void context_kernel(const Reduction* partial, unsigned blocks, const int* numeric, MotionImageKey key,
                               MotionConfig config, std::uint64_t epoch, std::int64_t frame, std::int64_t reference,
                               MotionContext* output, const bool* run_stage, const Control* control) {
    if (run_stage && !*run_stage) return;
    if (control) reference = control->committed_frame;
    MotionContext result{};
    result.epoch = epoch; result.frame = frame; result.reference_frame = reference;
    if (reference >= 0) {
        Reduction total{0, -CUDART_INF_F, 0, 0};
        for (unsigned i = 0; i < blocks; ++i) total = combine(total, partial[i]);
        result.peak_x = total.index % key.width; result.peak_y = total.index / key.width;
        if (result.peak_x > key.width / 2) result.peak_x -= key.width;
        if (result.peak_y > key.height / 2) result.peak_y -= key.height;
        result.status = MotionStatus::rejected;
        const double rms = sqrt(total.squares / (double(key.width) * key.height));
        if (total.numeric || !isfinite(rms)) result.pcr = CUDART_NAN_F;
        else if (rms > 0) {
            result.pcr = float(total.maximum / rms);
            const float scaled = result.pcr / config.pcr_threshold;
            result.confidence = fminf(1.0f, scaled);
            if (abs(result.peak_x) <= floorf(config.max_shift_fraction * float(key.width)) &&
                abs(result.peak_y) <= floorf(config.max_shift_fraction * float(key.height))) {
                result.status = MotionStatus::valid;
                result.tx = float(result.peak_x * key.downscale) * result.confidence;
                result.ty = float(result.peak_y * key.downscale) * result.confidence;
            }
            if (!isfinite(result.pcr) || !isfinite(scaled) || !isfinite(result.confidence) ||
                !isfinite(result.tx) || !isfinite(result.ty)) result.pcr = CUDART_NAN_F;
        }
    }
    if (*numeric) result.pcr = CUDART_NAN_F;
    *output = result;
}
struct ImageHeader { std::uint64_t epoch; std::int64_t frame; };
__global__ void image_header(ImageHeader* header, std::uint64_t epoch, std::int64_t frame) {
    *header = {epoch, frame};
}
__global__ void motion_stage(Control* control, const int* numeric, const ImageHeader* header,
                             std::uint64_t epoch, std::int64_t frame) {
    if (!control) return;
    if (header && (header->epoch != epoch || header->frame != frame))
        control->frame_status = static_cast<Error>(static_cast<unsigned>(control->frame_status) | static_cast<unsigned>(Error::protocol));
    if (*numeric) control->frame_status = static_cast<Error>(static_cast<unsigned>(control->frame_status) | static_cast<unsigned>(Error::numeric));
    control->run_stage = control->fail_code == Error::none && control->frame_status == Error::none;
}
template<class Kernel> void load(Kernel kernel) {
    cudaFuncAttributes attributes{};
    ck(cudaFuncGetAttributes(&attributes, kernel), "load motion kernel");
}
}
void motion_image_cuda(const std::uint8_t* rgb, float* image, const MotionImageKey& key,
                       const float* wx, const float* wy, cudaStream_t detect, const bool* run_stage) {
    explicit_stream(detect);
    const auto n = static_cast<std::size_t>(key.width) * key.height;
    image_kernel<<<(n + kThreads - 1) / kThreads, kThreads, 0, detect>>>(rgb, image, key, wx, wy, run_stage);
    ck(cudaGetLastError(), "motion F05");
}

struct MotionCuda::Impl {
    struct Slot {
        float* image = nullptr;
        ImageHeader* header = nullptr;
        cudaEvent_t ready = nullptr, consumed = nullptr;
        bool used = false;
    };
    MotionImageKey key;
    MotionConfig config;
    std::uint64_t epoch;
    std::vector<Slot> slots;
    std::size_t n, nf;
    unsigned blocks;
    float *wx = nullptr, *wy = nullptr, *correlation = nullptr;
    cufftComplex *temporary = nullptr, *spectra = nullptr, *cross = nullptr;
    Reduction* partial = nullptr;
    int* numeric = nullptr;
    cufftHandle forward = 0, inverse = 0;
    void* workspace = nullptr;
    cudaStream_t initialization = nullptr, detect = nullptr, main = nullptr;
    int committed_slot = 1;
    std::int64_t reference_frame = -1;
    bool closed = false;
    CleanupReport cleanup;

    int working_slot() const { return 1 - committed_slot; }
    void publish(std::int64_t frame) { committed_slot = working_slot(); reference_frame = frame; }
    template<class T> void allocate(T*& pointer, std::size_t count) {
        ck(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)), "allocate motion");
        ck(cudaMemsetAsync(pointer, 0, count * sizeof(T), initialization), "initialize motion");
    }
    Impl(const MotionImageKey& k, const MotionConfig& c, std::uint64_t e, unsigned count)
        : key(k), config(c), epoch(e), slots(count), n(static_cast<std::size_t>(k.width) * k.height),
          nf(static_cast<std::size_t>(k.width / 2 + 1) * k.height), blocks((n + kThreads - 1) / kThreads) {
        if (key != motion_image_key(key.original_width, key.original_height, config)) throw std::invalid_argument("inconsistent motion key/config");
        if (count == 0) throw std::invalid_argument("motion needs at least one image slot");
        try {
            ck(cudaStreamCreateWithFlags(&initialization, cudaStreamNonBlocking), "create initialization stream");
            load(image_header); load(motion_stage); load(image_kernel); load(reset_numeric); load(deliver); load(cross_power); load(reduce_correlation); load(context_kernel);
            for (auto& slot : slots) {
                allocate(slot.image, n);
                allocate(slot.header, 1);
                image_header<<<1, 1, 0, initialization>>>(slot.header, epoch, -1);
                ck(cudaGetLastError(), "initialize MI header");
                ck(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming), "create MI ready");
                ck(cudaEventCreateWithFlags(&slot.consumed, cudaEventDisableTiming), "create MI consumed");
            }
            allocate(wx, key.width); allocate(wy, key.height);
            allocate(correlation, n); allocate(temporary, nf); allocate(spectra, 2 * nf); allocate(cross, nf);
            allocate(partial, blocks); allocate(numeric, 1);
            ck(cudaStreamSynchronize(initialization), "initialize tables before upload");
            const auto host_wx = hanning_window(key.width), host_wy = hanning_window(key.height);
            // Constructor-only synchronous uploads; frame work never borrows host tables.
            ck(cudaMemcpy(wx, host_wx.data(), key.width * sizeof(float), cudaMemcpyHostToDevice), "upload wx");
            ck(cudaMemcpy(wy, host_wy.data(), key.height * sizeof(float), cudaMemcpyHostToDevice), "upload wy");
            fft(cufftCreate(&forward), "create forward plan"); fft(cufftCreate(&inverse), "create inverse plan");
            fft(cufftSetAutoAllocation(forward, 0), "disable forward autoallocation");
            fft(cufftSetAutoAllocation(inverse, 0), "disable inverse autoallocation");
            std::size_t forward_bytes = 0, inverse_bytes = 0;
            fft(cufftMakePlan2d(forward, key.height, key.width, CUFFT_R2C, &forward_bytes), "plan R2C");
            fft(cufftMakePlan2d(inverse, key.height, key.width, CUFFT_C2R, &inverse_bytes), "plan C2R");
            ck(cudaMalloc(&workspace, std::max<std::size_t>({forward_bytes, inverse_bytes, 1})), "allocate FFT workspace");
            fft(cufftSetWorkArea(forward, workspace), "forward workspace"); fft(cufftSetWorkArea(inverse, workspace), "inverse workspace");
            fft(cufftSetStream(forward, initialization), "forward initialization stream");
            fft(cufftSetStream(inverse, initialization), "inverse initialization stream");
            fft(cufftExecR2C(forward, slots[0].image, temporary), "warm forward");
            fft(cufftExecC2R(inverse, cross, correlation), "warm inverse");
            ck(cudaDeviceSynchronize(), "motion initialization drain");
            cleanup.release(initialization, [](auto s) { ck(cudaStreamDestroy(s), "destroy initialization stream"); });
            cleanup.throw_if_failed();
        } catch (...) { release(); throw; }
    }
    ~Impl() { release(); }
    void release() noexcept {
        if (closed) return;
        closed = true;
        for (const auto stream : {initialization, detect, main}) {
            if (stream) { int pending = 1; cleanup.release(pending, [&](int) { ck(cudaStreamSynchronize(stream), "drain motion stream"); }); }
        }
        cleanup.release(forward, [](auto p) { fft(cufftDestroy(p), "destroy forward"); });
        cleanup.release(inverse, [](auto p) { fft(cufftDestroy(p), "destroy inverse"); });
        cleanup.release(workspace, [](auto p) { ck(cudaFree(p), "free FFT workspace"); });
        for (auto& slot : slots) {
            cleanup.release(slot.header, [](auto p) { ck(cudaFree(p), "free MI header"); });
            cleanup.release(slot.image, [](auto p) { ck(cudaFree(p), "free MI"); });
            cleanup.release(slot.ready, [](auto p) { ck(cudaEventDestroy(p), "destroy ready"); });
            cleanup.release(slot.consumed, [](auto p) { ck(cudaEventDestroy(p), "destroy consumed"); });
        }
        cleanup.release(wx, [](auto p) { ck(cudaFree(p), "free wx"); });
        cleanup.release(wy, [](auto p) { ck(cudaFree(p), "free wy"); });
        cleanup.release(correlation, [](auto p) { ck(cudaFree(p), "free correlation"); });
        cleanup.release(temporary, [](auto p) { ck(cudaFree(p), "free FFT_TMP"); });
        cleanup.release(spectra, [](auto p) { ck(cudaFree(p), "free SPEC"); });
        cleanup.release(cross, [](auto p) { ck(cudaFree(p), "free cross"); });
        cleanup.release(partial, [](auto p) { ck(cudaFree(p), "free reduction"); });
        cleanup.release(numeric, [](auto p) { ck(cudaFree(p), "free numeric"); });
        cleanup.release(initialization, [](auto p) { ck(cudaStreamDestroy(p), "destroy initialization stream"); });
    }
};
MotionCuda::MotionCuda(const MotionImageKey& key, const MotionConfig& config, std::uint64_t epoch, unsigned slots)
    : impl_(std::make_unique<Impl>(key, config, epoch, slots)) {}
MotionCuda::~MotionCuda() = default;
void MotionCuda::enqueue_image(cudaStream_t detect, const std::uint8_t* rgb, std::int64_t frame) {
    explicit_stream(detect);
    auto& p = *impl_; p.detect = detect;
    auto& slot = p.slots[static_cast<std::uint64_t>(frame) % p.slots.size()];
    if (slot.used) ck(cudaStreamWaitEvent(detect, slot.consumed, 0), "wait MI consumed");
    motion_image_cuda(rgb, slot.image, p.key, p.wx, p.wy, detect);
    image_header<<<1, 1, 0, detect>>>(slot.header, p.epoch, frame); ck(cudaGetLastError(), "publish MI header");
    ck(cudaEventRecord(slot.ready, detect), "record MI ready");
}
void MotionCuda::enqueue_estimate(cudaStream_t main, std::int64_t frame, MotionContext* output,
                                  const bool* run_stage, Control* control) {
    explicit_stream(main);
    auto& p = *impl_; p.main = main;
    auto& slot = p.slots[static_cast<std::uint64_t>(frame) % p.slots.size()];
    ck(cudaStreamWaitEvent(main, slot.ready, 0), "wait MI ready");
    reset_numeric<<<1, 1, 0, main>>>(p.numeric); ck(cudaGetLastError(), "reset numeric");
    motion_stage<<<1, 1, 0, main>>>(control, p.numeric, slot.header, p.epoch, frame);
    ck(cudaGetLastError(), "motion pre FFT control");
    fft(cufftSetStream(p.forward, main), "forward main stream");
    fft(cufftExecR2C(p.forward, slot.image, p.temporary), "motion forward");
    motion_stage<<<1, 1, 0, main>>>(control, p.numeric, slot.header, p.epoch, frame);
    ck(cudaGetLastError(), "motion delivery control");
    auto* working = p.spectra + (control ? 0 : p.working_slot() * p.nf);
    deliver<<<(p.nf + kThreads - 1) / kThreads, kThreads, 0, main>>>(p.temporary, working, p.nf, p.numeric, run_stage, control);
    ck(cudaGetLastError(), "deliver spectrum");
    motion_stage<<<1, 1, 0, main>>>(control, p.numeric, slot.header, p.epoch, frame);
    ck(cudaGetLastError(), "motion cross control");
    ck(cudaEventRecord(slot.consumed, main), "record MI consumed"); slot.used = true;
    if (control || p.reference_frame >= 0) {
        const auto* committed = p.spectra + (control ? 0 : p.committed_slot * p.nf);
        cross_power<<<(p.nf + kThreads - 1) / kThreads, kThreads, 0, main>>>(committed, working, p.cross, p.nf, p.numeric, run_stage, control);
        ck(cudaGetLastError(), "motion cross power");
        motion_stage<<<1, 1, 0, main>>>(control, p.numeric, nullptr, p.epoch, frame);
        ck(cudaGetLastError(), "motion inverse control");
        fft(cufftSetStream(p.inverse, main), "inverse main stream");
        fft(cufftExecC2R(p.inverse, p.cross, p.correlation), "motion inverse");
        reduce_correlation<<<p.blocks, kThreads, 0, main>>>(p.correlation, p.n, p.partial, run_stage);
        ck(cudaGetLastError(), "motion reduction");
    }
    motion_stage<<<1, 1, 0, main>>>(control, p.numeric, nullptr, p.epoch, frame);
    ck(cudaGetLastError(), "motion context control");
    context_kernel<<<1, 1, 0, main>>>(p.partial, p.blocks, p.numeric, p.key, p.config, p.epoch, frame, p.reference_frame, output, run_stage, control);
    ck(cudaGetLastError(), "motion context");
}
void MotionCuda::enqueue_commit(cudaStream_t main, std::int64_t frame) {
    explicit_stream(main);
    impl_->main = main;
    impl_->publish(frame);
}
void MotionCuda::close() { impl_->release(); impl_->cleanup.throw_if_failed(); }
} // namespace saccade

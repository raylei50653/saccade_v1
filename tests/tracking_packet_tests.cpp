#include "saccade/tracking_session.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
using namespace saccade;
std::uint64_t checks = 0;
void check(bool value, const char* what) {
    ++checks;
    if (!value) throw std::runtime_error(what);
}
void ck(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
template<class T> struct Buffer {
    T *device = nullptr, *host = nullptr;
    Buffer() { ck(cudaMalloc(&device, sizeof(T))); ck(cudaMallocHost(&host, sizeof(T))); *host = {}; }
    ~Buffer() { cudaFree(device); cudaFreeHost(host); }
    void upload(cudaStream_t stream) { ck(cudaMemcpyAsync(device, host, sizeof(T), cudaMemcpyHostToDevice, stream)); }
    void download(cudaStream_t stream) { ck(cudaMemcpyAsync(host, device, sizeof(T), cudaMemcpyDeviceToHost, stream)); }
};
void run() {
    cudaStream_t stream{};
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
        Buffer<Control> control;
        Buffer<DeviceTrackingRows> rows;
        Buffer<PacketBytes> bytes;
        detail::tracking_session_load_kernels();
        std::mt19937_64 random(0x53435250);
        for (unsigned sample = 0; sample < 400; ++sample) {
            ResultPacket p;
            p.epoch = random(); p.frame = 20 + random() % 10000;
            const auto status = sample % 4 == 0 ? ResultStatus::failed :
                                sample % 4 == 1 ? ResultStatus::skipped_after_failure : ResultStatus::ok;
            p.status = status;
            p.committed_frame = status == ResultStatus::ok ? p.frame : p.frame - 3;
            if (status != ResultStatus::ok) {
                p.fail_frame = status == ResultStatus::failed ? p.frame : p.frame - 2;
                p.fail_code = static_cast<Error>(1 + random() % 31);
                p.frame_error = status == ResultStatus::failed ? p.fail_code : Error::none;
            } else {
                const unsigned count = sample % 4 == 2 ? packet_capacity : random() % (packet_capacity + 1);
                for (unsigned i = 0; i < count; ++i) {
                    TrackRow r;
                    r.uid = 1 + random(); r.output_id = 1 + random(); r.class_id = random();
                    r.flags = random() % 2 ? 1 : 2;
                    for (auto& f : r.xyxy) f = static_cast<float>(static_cast<int>(random() % 200000) - 100000) / 32;
                    r.xyxy[i % 4] = -0.0f;
                    if (r.flags == 1) {
                        r.observed_score = static_cast<float>(random() % 1025) / 1024;
                        r.association_score = static_cast<float>(static_cast<int>(random() % 40000) - 20000) / 512;
                    } else r.observed_score = r.association_score = -0.0f;
                    p.tracks.push_back(r);
                }
            }
            *control.host = {};
            control.host->epoch = p.epoch; control.host->current_frame = p.frame;
            control.host->committed_frame = p.committed_frame; control.host->fail_frame = p.fail_frame;
            control.host->fail_code = p.fail_code; control.host->frame_status = p.frame_error;
            *rows.host = {};
            rows.host->count = p.tracks.size();
            for (unsigned i = 0; i < p.tracks.size(); ++i) {
                const auto& a = p.tracks[i]; auto& b = rows.host->rows[i];
                b.uid = a.uid; b.output_id = a.output_id; b.class_id = a.class_id; b.flags = a.flags;
                std::copy(a.xyxy.begin(), a.xyxy.end(), b.xyxy);
                b.observed_score = a.observed_score; b.association_score = a.association_score;
            }
            control.upload(stream); rows.upload(stream);
            ck(cudaMemsetAsync(bytes.device, 0xa5, packet_bytes, stream));
            detail::tracking_session_packet(control.device, rows.device, reinterpret_cast<std::uint8_t*>(bytes.device), stream);
            bytes.download(stream); ck(cudaStreamSynchronize(stream));
            check(*bytes.host == encode_packet(p), "device writer differs from encode_packet");
            check(encode_packet(decode_packet(*bytes.host)) == *bytes.host, "canonical roundtrip");
        }
        // Each inactive M2 entry point leaves its outputs and scratch untouched.
        Buffer<detail::TrackingSessionControl> ctl;
        Buffer<DeviceTrackState> state;
        Buffer<DeviceMotionContext> motion;
        Buffer<FrameId> frame;
        Buffer<TrackerConfig> config;
        Buffer<DevicePredicted> predicted;
        Buffer<DeviceBatch> batch;
        Buffer<DetectionCandidates> candidates;
        Buffer<TrackingCudaScratch> scratch;
        ctl.upload(stream); state.upload(stream); motion.upload(stream); frame.upload(stream);
        config.upload(stream); candidates.upload(stream);
        ck(cudaMemsetAsync(predicted.device, 0xa5, sizeof(DevicePredicted), stream));
        ck(cudaMemsetAsync(batch.device, 0xa5, sizeof(DeviceBatch), stream));
        ck(cudaMemsetAsync(rows.device, 0xa5, sizeof(DeviceTrackingRows), stream));
        ck(cudaMemsetAsync(scratch.device, 0xa5, sizeof(TrackingCudaScratch), stream));
        enqueue_predict_tracks(state.device, motion.device, frame.device, config.device, predicted.device, scratch.device,
                               stream, &ctl.device->value.run_stage);
        enqueue_assemble_detection_batch(candidates.device, predicted.device, config.device, batch.device, scratch.device,
                                         stream, &ctl.device->value.run_stage);
        enqueue_associate_and_update(predicted.device, batch.device, motion.device, config.device, state.device, rows.device,
                                     scratch.device, stream, &ctl.device->value.run_stage);
        predicted.download(stream); batch.download(stream); rows.download(stream); scratch.download(stream);
        ck(cudaStreamSynchronize(stream));
        const auto untouched = [](const auto* p) {
            const auto* b = reinterpret_cast<const unsigned char*>(p);
            return std::all_of(b, b + sizeof(*p), [](auto v) { return v == 0xa5; });
        };
        check(untouched(predicted.host), "predict guard"); check(untouched(batch.host), "batch guard");
        check(untouched(rows.host), "update guard"); check(untouched(scratch.host), "scratch guard");
    }
    ck(cudaStreamDestroy(stream));
}
}
int main() {
    try { run(); std::cout << "tracking_packet_tests: " << checks << " checks passed\n"; return 0; }
    catch (const std::exception& e) { std::cerr << "tracking_packet_tests: " << e.what() << '\n'; return 1; }
}

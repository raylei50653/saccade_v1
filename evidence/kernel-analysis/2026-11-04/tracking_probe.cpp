#include "saccade/tracking_cuda.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

static void check(cudaError_t r) {
    if (r != cudaSuccess) throw std::runtime_error(cudaGetErrorString(r));
}
template<class T> T read_frame(const std::filesystem::path& path, std::size_t frame) {
    const auto size = std::filesystem::file_size(path);
    if (size % sizeof(T) || size < (frame + 1) * sizeof(T))
        throw std::runtime_error("incompatible ABI or insufficient records: " + path.string());
    std::ifstream f(path, std::ios::binary);
    f.seekg(frame * sizeof(T));
    T value{};
    f.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!f) throw std::runtime_error("record read failed: " + path.string());
    return value;
}
template<class T> struct Buffer {
    T* p = nullptr;
    Buffer() { check(cudaMalloc(reinterpret_cast<void**>(&p), sizeof(T))); check(cudaMemset(p, 0, sizeof(T))); }
    ~Buffer() { if (p) cudaFree(p); }
    void upload(const T& value, cudaStream_t stream) {
        check(cudaMemcpyAsync(p, &value, sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    T download(cudaStream_t stream) {
        T value{};
        check(cudaMemcpyAsync(&value, p, sizeof(T), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        return value;
    }
};
// Disposable isolated cache probe. Frame 20 is from previously saved HR
// MOT17-10 replay data. Every iteration reads the same immutable inputs.
// Update reads the saved reference batch, not the output of this probe's batch
// kernel. This matches the independent primitive replay contract. No tracker
// state is advanced, no JPEG/model is run, and no GT is read.
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: tracking_probe REPLAY_DIRECTORY");
        using namespace saccade;
        constexpr std::size_t frame = 20;
        const std::filesystem::path root = argv[1];
        const auto predicted = read_frame<DevicePredicted>(root / "predicted.bin", frame);
        const auto candidates = read_frame<DetectionCandidates>(root / "candidates.bin", frame);
        const auto batch = read_frame<DeviceBatch>(root / "batch.bin", frame);
        const auto motion = read_frame<DeviceMotionContext>(root / "motion.bin", frame);
        if (predicted.frame != frame || candidates.header.frame != frame || batch.frame != frame || motion.frame != frame ||
            predicted.epoch != candidates.header.epoch || predicted.epoch != batch.epoch || predicted.epoch != motion.epoch ||
            predicted.count > kTrackCapacity || predicted.prior_count > kTrackCapacity || candidates.header.count > kPostprocessCapacity ||
            batch.count > kBatchCapacity || predicted.error != Error::none || batch.error != Error::none || candidates.header.flags)
            throw std::runtime_error("selected replay input violates identity/count/error contract");
        unsigned suppressed = 0;
        for (unsigned i = 0; i < candidates.header.count; ++i)
            suppressed += bool(candidates.rows[i].flags & kDetectionSuppressed);
        check(cudaSetDevice(0));
        cudaStream_t stream = nullptr;
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        {
            Buffer<DevicePredicted> dp;
            Buffer<DetectionCandidates> dc;
            Buffer<DeviceBatch> db_input, db_output;
            Buffer<DeviceMotionContext> dm;
            Buffer<TrackerConfig> dcfg;
            Buffer<TrackingCudaScratch> scratch;
            Buffer<DeviceTrackState> next;
            Buffer<DeviceTrackingRows> rows;
            const TrackerConfig config{};
            dp.upload(predicted, stream); dc.upload(candidates, stream); db_input.upload(batch, stream);
            dm.upload(motion, stream); dcfg.upload(config, stream);
            check(cudaStreamSynchronize(stream));
            for (unsigned i = 0; i < 21; ++i) {
                enqueue_assemble_detection_batch(dc.p, dp.p, dcfg.p, db_output.p, scratch.p, stream);
                enqueue_associate_and_update(dp.p, db_input.p, dm.p, dcfg.p, next.p, rows.p, scratch.p, stream);
            }
            check(cudaStreamSynchronize(stream));
            const auto got = db_output.download(stream);
            const auto got_next = next.download(stream);
            const auto got_rows = rows.download(stream);
            if (got.error != Error::none || got_rows.error != Error::none || got_next.count > kTrackCapacity || got_rows.count > packet_capacity ||
                got.count != batch.count || got.epoch != batch.epoch || got.frame != batch.frame)
                throw std::runtime_error("probe output failed error/count/identity validation");
            for (unsigned i = 0; i < got.count; ++i) {
                const auto& a = got.rows[i]; const auto& b = batch.rows[i];
                if (std::memcmp(a.xyxy, b.xyxy, sizeof(a.xyxy)) || std::memcmp(&a.observed_score, &b.observed_score, sizeof(float)) ||
                    std::memcmp(&a.association_score, &b.association_score, sizeof(float)) || a.class_id != b.class_id ||
                    a.candidate_id != b.candidate_id || a.can_continue != b.can_continue || a.can_birth != b.can_birth)
                    throw std::runtime_error("batch semantic bits differ from saved reference");
            }
            std::cout << "{\"frame\":20,\"iterations\":21,\"predicted_tracks\":" << predicted.count
                      << ",\"priors\":" << predicted.prior_count << ",\"candidates\":" << candidates.header.count
                      << ",\"suppressed\":" << suppressed << ",\"batch_rows\":" << batch.count
                      << ",\"next_tracks\":" << got_next.count << ",\"output_rows\":" << got_rows.count
                      << ",\"batch_reference_semantic_bits_equal\":true,\"output_errors\":0}"
                      << std::endl;
        }
        check(cudaStreamDestroy(stream));
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

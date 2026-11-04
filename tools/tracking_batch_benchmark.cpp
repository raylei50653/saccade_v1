#include "saccade/tracking_cuda.hpp"
#include "saccade/file_io.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/sha256.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <bit>
#include <charconv>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {
using namespace saccade;
using Bytes = std::vector<std::uint8_t>;
constexpr const char* usage = "usage: tracking_batch_benchmark RECS.scdr RECS.scmo NEW_OUT_DIR [--frames 140] [--iterations 32] [--rounds 7]";
void ck(cudaError_t rc, const char* operation) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(rc));
}
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
template<class T> T number(std::string_view value) {
    T out{};
    const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), out);
    if (ec != std::errc{} || end != value.data() + value.size()) throw std::invalid_argument("invalid integer");
    return out;
}
template<class T> void integer(Bytes& out, T value) {
    using U = std::make_unsigned_t<T>;
    U bits = static_cast<U>(value);
    for (unsigned i = 0; i < sizeof(T); ++i) { out.push_back(static_cast<std::uint8_t>(bits)); bits >>= 8; }
}
void real(Bytes& out, float value) { integer(out, std::bit_cast<std::uint32_t>(value)); }
void track(Bytes& out, const Track& t) {
    integer(out,t.uid); integer(out,t.output_id); integer(out,static_cast<std::uint32_t>(t.lifecycle));
    for (float value : t.x) real(out,value);
    for (float value : t.P) real(out,value);
    integer(out,t.age); integer(out,t.hit_streak); real(out,t.score_sum); integer(out,t.class_id);
    real(out,t.observed_score); real(out,t.association_score); integer(out,t.birth_frame); integer(out,t.last_observed_frame);
    for (const auto& h : t.history) { real(out,h.cx); real(out,h.cy); real(out,h.h); }
    integer(out,t.history_count); real(out,t.h_ema);
}
// Serialize every logical field, preserving float bits and excluding device
// padding and unused capacity, which are not part of the tracker contract.
Bytes semantic(const DetectionBatch& b) {
    Bytes out;
    integer(out,b.epoch); integer(out,b.frame); integer(out,static_cast<std::uint32_t>(b.error));
    integer(out,static_cast<std::uint32_t>(b.rows.size()));
    for (const auto& d : b.rows) {
        for (float value : d.xyxy) real(out,value);
        real(out,d.observed_score); real(out,d.association_score); integer(out,d.class_id); integer(out,d.candidate_id);
        integer(out,static_cast<std::uint8_t>(d.can_continue)); integer(out,static_cast<std::uint8_t>(d.can_birth));
    }
    return out;
}
Bytes semantic(const UpdateResult& u) {
    Bytes out;
    integer(out,static_cast<std::uint32_t>(u.error));
    if (u.error == Error::none) {
        integer(out,u.next.epoch); integer(out,u.next.committed_frame); integer(out,u.next.next_uid);
        integer(out,static_cast<std::uint32_t>(u.next.tracks.size()));
        for (const auto& t : u.next.tracks) track(out,t);
    }
    integer(out,static_cast<std::uint32_t>(u.rows.size()));
    for (const auto& r : u.rows) {
        integer(out,r.uid); integer(out,r.output_id); integer(out,r.class_id); integer(out,r.flags);
        for (float value : r.xyxy) real(out,value);
        real(out,r.observed_score); real(out,r.association_score);
    }
    return out;
}
void compare(const Bytes& actual, const Bytes& expected, std::size_t frame, const char* stage) {
    const auto mismatch = std::mismatch(actual.begin(),actual.end(),expected.begin(),expected.end());
    if (mismatch.first != actual.end() || mismatch.second != expected.end())
        throw std::runtime_error("CPU bit comparison failed: frame=" + std::to_string(frame) + " stage=" + stage +
                                 " semantic_byte=" + std::to_string(mismatch.first - actual.begin()));
}
std::string hash(const Bytes& data) { return sha256_hex({reinterpret_cast<const char*>(data.data()),data.size()}); }
template<class T> void append_raw(Bytes& out, const T& value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(),bytes,bytes + sizeof(T));
}
struct Replay {
    DevicePredicted predicted;
    DetectionCandidates candidates;
    DeviceBatch batch;
    DeviceMotionContext motion;
    Bytes batch_expected, update_expected;
};
template<class T> struct Buffer {
    T *device = nullptr, *host = nullptr;
    Buffer() {
        ck(cudaMalloc(&device,sizeof(T)),"allocate device replay");
        try { ck(cudaMallocHost(&host,sizeof(T)),"allocate pinned replay"); }
        catch (...) { cudaFree(device); throw; }
    }
    ~Buffer() { cudaFree(device); cudaFreeHost(host); }
    void upload(cudaStream_t stream) { ck(cudaMemcpyAsync(device,host,sizeof(T),cudaMemcpyHostToDevice,stream),"upload replay"); }
    void download(cudaStream_t stream) { ck(cudaMemcpyAsync(host,device,sizeof(T),cudaMemcpyDeviceToHost,stream),"download replay"); }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::size_t nodes = 0;
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
};
struct Harness {
    Buffer<DevicePredicted> predicted;
    Buffer<DetectionCandidates> candidates;
    Buffer<DeviceBatch> batch;
    Buffer<DeviceMotionContext> motion;
    Buffer<DeviceTrackState> next;
    Buffer<DeviceTrackingRows> rows;
    Buffer<TrackingCudaScratch> scratch;
    Buffer<TrackerConfig> config;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    Harness() {
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create replay stream");
        ck(cudaEventCreate(&start),"create start event"); ck(cudaEventCreate(&stop),"create stop event");
        *config.host = TrackerConfig{};
        config.upload(stream);
    }
    ~Harness() {
        if (stream) cudaStreamSynchronize(stream);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
    }
    void wait() { ck(cudaStreamSynchronize(stream),"wait replay"); }
    void upload(const Replay& r) {
        *predicted.host = r.predicted; *candidates.host = r.candidates; *batch.host = r.batch; *motion.host = r.motion;
        predicted.upload(stream); candidates.upload(stream); batch.upload(stream); motion.upload(stream);
        wait();
    }
    void enqueue(bool update) {
        if (update) enqueue_associate_and_update(predicted.device,batch.device,motion.device,config.device,next.device,rows.device,scratch.device,stream);
        else enqueue_assemble_detection_batch(candidates.device,predicted.device,config.device,batch.device,scratch.device,stream);
    }
    void capture(Graph& g, bool update, unsigned iterations) {
        enqueue(update); wait(); // Complete lazy module loading before capture.
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"begin replay capture");
        try {
            for (unsigned i = 0; i < iterations; ++i) enqueue(update);
            ck(cudaStreamEndCapture(stream,&g.graph),"end replay capture");
        } catch (...) { cudaGraph_t abandoned = nullptr; cudaStreamEndCapture(stream,&abandoned); if (abandoned) cudaGraphDestroy(abandoned); throw; }
        ck(cudaGraphInstantiate(&g.exec,g.graph,0),"instantiate replay");
        ck(cudaGraphGetNodes(g.graph,nullptr,&g.nodes),"count replay nodes");
    }
    double time(bool update, const Graph* g, unsigned iterations) {
        if (g) ck(cudaGraphLaunch(g->exec,stream),"warm replay");
        else for (unsigned i = 0; i < iterations; ++i) enqueue(update);
        wait();
        ck(cudaEventRecord(start,stream),"record replay start");
        if (g) ck(cudaGraphLaunch(g->exec,stream),"launch replay");
        else for (unsigned i = 0; i < iterations; ++i) enqueue(update);
        ck(cudaEventRecord(stop,stream),"record replay stop"); ck(cudaEventSynchronize(stop),"wait replay stop");
        float ms = 0; ck(cudaEventElapsedTime(&ms,start,stop),"replay elapsed");
        return static_cast<double>(ms) * 1000 / iterations;
    }
    void check(const Replay& r, std::size_t frame, bool update) {
        if (update) {
            next.download(stream); rows.download(stream); wait();
            require(next.host->count <= kTrackCapacity && rows.host->count <= packet_capacity,"invalid replay output count");
            compare(semantic(to_host(*next.host,*rows.host)),r.update_expected,frame,"update");
        } else {
            batch.download(stream); wait();
            require(batch.host->count <= kBatchCapacity,"invalid replay batch count");
            compare(semantic(to_host(*batch.host)),r.batch_expected,frame,"F04b");
        }
    }
};
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") { std::cout << usage << '\n'; return 0; }
        if (argc < 4) throw std::invalid_argument(usage);
        std::size_t frames = 140;
        unsigned iterations = 32, rounds = 7;
        for (int i = 4; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (i + 1 == argc) throw std::invalid_argument(usage);
            const std::string_view value = argv[++i];
            if (arg == "--frames") frames = number<std::size_t>(value);
            else if (arg == "--iterations") iterations = number<unsigned>(value);
            else if (arg == "--rounds") rounds = number<unsigned>(value);
            else throw std::invalid_argument(usage);
        }
        if (frames == 0 || frames > 10000 || iterations == 0 || iterations > 10000 || rounds == 0 || rounds > 100)
            throw std::invalid_argument("frames/iterations/rounds must be positive and within 10000/10000/100");
        const auto detections = read_detection_records(argv[1]);
        const auto motions = read_motion_records(argv[2]);
        require(detections.epoch == motions.epoch && detections.frames.size() >= frames && motions.frames.size() >= frames,
                "replay records epoch or size mismatch");
        const std::filesystem::path out = argv[3];
        require(std::filesystem::create_directory(out),"output directory must be new");
        std::ostringstream manifest;
        manifest << sha256_hex(read_text_file(argv[1])) << "  " << argv[1] << '\n'
                 << sha256_hex(read_text_file(argv[2])) << "  " << argv[2] << '\n';
        write_text_file(out / "inputs.sha256",manifest.str());
        const TrackerConfig cfg;
        auto state = initial_track_state(detections.epoch);
        std::vector<Replay> replays;
        replays.reserve(frames);
        Bytes all_predicted, all_candidates, all_batches, all_motion, all_expected;
        std::ostringstream cases;
        cases << "frame,predicted_tracks,priors,candidates,suppressed,batch_rows\n";
        unsigned max_priors = 0, max_candidates = 0, max_suppressed = 0;
        for (std::size_t k = 0; k < frames; ++k) {
            const auto& d = detections.frames[k]; const auto& m = motions.frames[k];
            require(d.header.frame == k && m.frame == static_cast<FrameId>(k),"noncontiguous replay identity");
            const auto p = predict_tracks(state,m,k,cfg);
            const auto b = assemble_detection_batch(d,p,cfg);
            auto u = associate_and_update(p,b,m,cfg);
            require(p.error == Error::none && b.error == Error::none && u.error == Error::none,
                    "CPU reference failed at frame=" + std::to_string(k));
            replays.push_back({to_device(p),d,to_device(b),to_device(m),semantic(b),semantic(u)});
            const auto& r = replays.back();
            append_raw(all_predicted,r.predicted); append_raw(all_candidates,r.candidates);
            append_raw(all_batches,r.batch); append_raw(all_motion,r.motion);
            integer(all_expected,static_cast<std::uint32_t>(r.batch_expected.size()));
            all_expected.insert(all_expected.end(),r.batch_expected.begin(),r.batch_expected.end());
            integer(all_expected,static_cast<std::uint32_t>(r.update_expected.size()));
            all_expected.insert(all_expected.end(),r.update_expected.begin(),r.update_expected.end());
            unsigned suppressed = 0;
            for (unsigned j = 0; j < d.header.count; ++j) suppressed += bool(d.rows[j].flags & kDetectionSuppressed);
            cases << k << ',' << p.tracks.size() << ',' << p.priors.size() << ',' << d.header.count << ',' << suppressed << ',' << b.rows.size() << '\n';
            max_priors = std::max(max_priors,static_cast<unsigned>(p.priors.size()));
            max_candidates = std::max(max_candidates,d.header.count); max_suppressed = std::max(max_suppressed,suppressed);
            state = std::move(u.next);
        }
        write_binary_file(out / "predicted.bin",all_predicted); write_binary_file(out / "candidates.bin",all_candidates);
        write_binary_file(out / "batch.bin",all_batches); write_binary_file(out / "motion.bin",all_motion);
        write_binary_file(out / "expected-semantic.bin",all_expected); write_text_file(out / "cases.csv",cases.str());
        Harness h;
        h.upload(replays.front());
        Graph batch_graph, update_graph;
        h.capture(batch_graph,false,iterations); h.capture(update_graph,true,iterations);
        std::ostringstream raw, selected;
        raw << "frame,stage,mode,round,round_mean_us\n" << std::fixed << std::setprecision(6);
        selected << "frame,stage,mode,mean_of_round_means_us\n" << std::fixed << std::setprecision(6);
        double totals[2][2]{};
        for (std::size_t frame = 0; frame < frames; ++frame) {
            const auto& r = replays[frame];
            h.upload(r);
            for (unsigned stage = 0; stage < 2; ++stage) {
                const bool update = stage != 0;
                double per_frame[2]{};
                for (unsigned round = 0; round < rounds; ++round) {
                    for (unsigned order = 0; order < 2; ++order) {
                        const unsigned mode = (round % 2) ^ order;
                        const Graph* graph = mode ? (update ? &update_graph : &batch_graph) : nullptr;
                        const double us = h.time(update,graph,iterations);
                        h.check(r,frame,update); // Outside timing; every replay mode and round checked.
                        raw << frame << ',' << (update ? "update" : "F04b") << ',' << (mode ? "graph" : "ordinary") << ',' << round << ',' << us << '\n';
                        totals[stage][mode] += us; per_frame[mode] += us;
                    }
                }
                if (frame == 0 || frame == 20 || frame == 50 || frame == 100 || frame == 139)
                    for (unsigned mode = 0; mode < 2; ++mode)
                        selected << frame << ',' << (update ? "update" : "F04b") << ',' << (mode ? "graph" : "ordinary") << ',' << per_frame[mode] / rounds << '\n';
            }
            if (frame % 20 == 0) std::cout << "replay frame=" << frame << " priors=" << r.predicted.prior_count << " candidates=" << r.candidates.header.count << " CPU bit comparison PASS\n" << std::flush;
        }
        write_text_file(out / "timings.csv",raw.str()); write_text_file(out / "selected.csv",selected.str());
        std::ostringstream summary;
        summary << std::fixed << std::setprecision(6) << "{\n  \"schema\":\"tracking-batch-replay/1\",\n  \"status\":\"PASS\",\n"
                << "  \"frames\":" << frames << ",\n  \"iterations\":" << iterations << ",\n  \"rounds\":" << rounds
                << ",\n  \"timing\":\"CUDA event batch means; excludes upload, capture, instantiate, CPU comparison; no single-frame p95\",\n"
                << "  \"comparison\":\"all logical batch/state/row fields preserve CPU bits on every frame, mode and round; padding and unused capacity excluded\",\n"
                << "  \"max_priors\":" << max_priors << ",\n  \"max_candidates\":" << max_candidates << ",\n  \"max_suppressed\":" << max_suppressed
                << ",\n  \"predicted_sha256\":\"" << hash(all_predicted) << "\",\n  \"candidates_sha256\":\"" << hash(all_candidates)
                << "\",\n  \"expected_semantic_sha256\":\"" << hash(all_expected) << "\",\n  \"batch_graph_nodes\":" << batch_graph.nodes
                << ",\n  \"update_graph_nodes\":" << update_graph.nodes << ",\n  \"prefix_means_us\":{\n";
        for (unsigned stage = 0; stage < 2; ++stage) for (unsigned mode = 0; mode < 2; ++mode) {
            if (stage || mode) summary << ",\n";
            summary << "    \"" << (stage ? "update" : "F04b") << '_' << (mode ? "graph" : "ordinary") << "\":" << totals[stage][mode] / (frames * rounds);
        }
        summary << "\n  }\n}\n";
        write_text_file(out / "summary.json",summary.str());
        std::cout << summary.str();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tracking_batch_benchmark: " << e.what() << '\n'; return 1;
    }
}

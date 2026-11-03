#include "saccade/tracking_cuda.hpp"
#include "saccade/tracking_math.hpp"
#include "saccade/mot_output.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace saccade;
namespace math = tracking_math;
std::uint64_t checks = 0;
std::string context;
void check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(context + ": " + message);
}
void ck(cudaError_t code, const char* operation) {
    if (code != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
void same(float a, float b, const char* field) {
    ++checks;
    if (std::bit_cast<std::uint32_t>(a) != std::bit_cast<std::uint32_t>(b)) {
        std::cerr << context << ' ' << field << " GPU=" << std::hexfloat << a << " CPU=" << b << std::defaultfloat << '\n';
        throw std::runtime_error(context + ": float bits differ: " + field);
    }
}
void compare(const Track& a, const Track& b) {
    check(a.uid == b.uid && a.output_id == b.output_id && a.lifecycle == b.lifecycle, "track identity/lifecycle");
    for (unsigned i = 0; i < 8; ++i) same(a.x[i], b.x[i], "track.x");
    for (unsigned i = 0; i < 64; ++i) same(a.P[i], b.P[i], "track.P");
    check(a.age == b.age && a.hit_streak == b.hit_streak && a.class_id == b.class_id &&
          a.birth_frame == b.birth_frame && a.last_observed_frame == b.last_observed_frame &&
          a.history_count == b.history_count, "track bookkeeping");
    same(a.score_sum, b.score_sum, "score_sum"); same(a.observed_score, b.observed_score, "observed_score");
    same(a.association_score, b.association_score, "association_score"); same(a.h_ema, b.h_ema, "h_ema");
    for (unsigned i = 0; i < kBridgeHistory; ++i) {
        same(a.history[i].cx, b.history[i].cx, "history.cx");
        same(a.history[i].cy, b.history[i].cy, "history.cy"); same(a.history[i].h, b.history[i].h, "history.h");
    }
}
void compare(const MotionContext& a, const MotionContext& b) {
    check(a.epoch == b.epoch && a.frame == b.frame && a.reference_frame == b.reference_frame &&
          a.status == b.status && a.peak_x == b.peak_x && a.peak_y == b.peak_y, "motion fields");
    same(a.pcr, b.pcr, "pcr"); same(a.confidence, b.confidence, "confidence");
    same(a.tx, b.tx, "tx"); same(a.ty, b.ty, "ty");
}
void compare(const TrackState& a, const TrackState& b) {
    check(a.epoch == b.epoch && a.committed_frame == b.committed_frame && a.next_uid == b.next_uid, "state header");
    check(a.tracks.size() == b.tracks.size(), "state track count");
    for (unsigned i = 0; i < a.tracks.size(); ++i) compare(a.tracks[i], b.tracks[i]);
}
void compare(const PredictedState& a, const PredictedState& b) {
    check(a.epoch == b.epoch && a.frame == b.frame && a.source_committed_frame == b.source_committed_frame &&
          a.next_uid == b.next_uid && a.error == b.error, "predict header/error");
    compare(a.motion, b.motion);
    check(a.tracks.size() == b.tracks.size() && a.priors.size() == b.priors.size(), "predict counts");
    for (unsigned i = 0; i < a.tracks.size(); ++i) compare(a.tracks[i], b.tracks[i]);
    for (unsigned i = 0; i < a.priors.size(); ++i) {
        check(a.priors[i].uid == b.priors[i].uid && a.priors[i].class_id == b.priors[i].class_id, "prior identity");
        for (unsigned j = 0; j < 4; ++j) same(a.priors[i].xyxy[j], b.priors[i].xyxy[j], "prior.xyxy");
    }
}
void compare(const DetectionBatch& a, const DetectionBatch& b) {
    check(a.epoch == b.epoch && a.frame == b.frame && a.error == b.error && a.rows.size() == b.rows.size(), "batch header/count/error");
    for (unsigned i = 0; i < a.rows.size(); ++i) {
        const auto& x = a.rows[i]; const auto& y = b.rows[i];
        check(x.candidate_id == y.candidate_id && x.class_id == y.class_id &&
              x.can_continue == y.can_continue && x.can_birth == y.can_birth, "batch row fields");
        for (unsigned j = 0; j < 4; ++j) same(x.xyxy[j], y.xyxy[j], "batch.xyxy");
        same(x.observed_score, y.observed_score, "batch.observed"); same(x.association_score, y.association_score, "batch.association");
    }
}
void compare(const UpdateResult& a, const UpdateResult& b) {
    check(a.error == b.error && a.rows.size() == b.rows.size(), "update count/error");
    if (a.error == Error::none) compare(a.next, b.next);
    for (unsigned i = 0; i < a.rows.size(); ++i) {
        const auto& x = a.rows[i]; const auto& y = b.rows[i];
        check(x.uid == y.uid && x.output_id == y.output_id && x.class_id == y.class_id && x.flags == y.flags, "packet row fields");
        for (unsigned j = 0; j < 4; ++j) same(x.xyxy[j], y.xyxy[j], "packet.xyxy");
        same(x.observed_score, y.observed_score, "packet.observed"); same(x.association_score, y.association_score, "packet.association");
    }
    if (a.error == Error::none) {
        ResultPacket x, y;
        x.epoch = a.next.epoch; y.epoch = b.next.epoch;
        x.frame = x.committed_frame = a.next.committed_frame;
        y.frame = y.committed_frame = b.next.committed_frame;
        x.tracks = a.rows; y.tracks = b.rows;
        check(encode_packet(x) == encode_packet(y), "encoded packet bytes");
    }
}
template<class T> struct Buffer {
    T *device = nullptr, *host = nullptr;
    Buffer() {
        ck(cudaMalloc(&device, sizeof(T)), "allocate device");
        ck(cudaMallocHost(&host, sizeof(T)), "allocate pinned");
    }
    ~Buffer() { cudaFree(device); cudaFreeHost(host); }
    void upload(cudaStream_t stream) { ck(cudaMemcpyAsync(device, host, sizeof(T), cudaMemcpyHostToDevice, stream), "upload"); }
    void download(cudaStream_t stream) { ck(cudaMemcpyAsync(host, device, sizeof(T), cudaMemcpyDeviceToHost, stream), "download"); }
};
struct Harness {
    cudaStream_t stream = nullptr;
    Buffer<DeviceTrackState> state, next;
    Buffer<DeviceMotionContext> motion;
    Buffer<FrameId> frame;
    Buffer<TrackerConfig> config;
    Buffer<DetectionCandidates> candidates;
    Buffer<DevicePredicted> predicted;
    Buffer<DeviceBatch> batch;
    Buffer<DeviceTrackingRows> rows;
    Buffer<TrackingCudaScratch> scratch;
    std::array<cudaEvent_t, 4> events{};
    std::array<double, 3> times{};
    std::size_t timed_frames = 0;
    Harness() {
        ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        for (auto& event : events) ck(cudaEventCreate(&event), "create event");
    }
    ~Harness() {
        cudaStreamSynchronize(stream);
        for (auto event : events) cudaEventDestroy(event);
        cudaStreamDestroy(stream);
    }
    void wait() { ck(cudaStreamSynchronize(stream), "wait tracker"); }
    void reset(const TrackState& s, const TrackerConfig& c = {}) {
        *state.host = to_device(s); *config.host = c; state.upload(stream); config.upload(stream); wait();
    }
    UpdateResult step(const TrackState& cpu, const DetectionCandidates& d, const MotionContext& m, FrameId k, bool time = false) {
        *motion.host = to_device(m); *frame.host = k; *candidates.host = d;
        motion.upload(stream); frame.upload(stream); candidates.upload(stream);
        if (time) ck(cudaEventRecord(events[0], stream), "time predict start");
        enqueue_predict_tracks(state.device, motion.device, frame.device, config.device, predicted.device, scratch.device, stream);
        if (time) ck(cudaEventRecord(events[1], stream), "time predict end");
        enqueue_assemble_detection_batch(candidates.device, predicted.device, config.device, batch.device, scratch.device, stream);
        if (time) ck(cudaEventRecord(events[2], stream), "time batch end");
        enqueue_associate_and_update(predicted.device, batch.device, motion.device, config.device, next.device, rows.device, scratch.device, stream);
        if (time) ck(cudaEventRecord(events[3], stream), "time update end");
        predicted.download(stream); batch.download(stream); next.download(stream); rows.download(stream); wait();
        const auto p = predict_tracks(cpu, m, k, *config.host);
        const auto b = assemble_detection_batch(d, p, *config.host);
        auto u = associate_and_update(p, b, m, *config.host);
        context += " F08a"; compare(to_host(*predicted.host), p);
        context += " F04b"; compare(to_host(*batch.host), b);
        context += " F07/F08b"; compare(to_host(*next.host, *rows.host), u);
        if (u.error == Error::none) std::swap(state.device, next.device);
        if (time) {
            for (unsigned i = 0; i < 3; ++i) {
                float ms = 0; ck(cudaEventElapsedTime(&ms, events[i], events[i + 1]), "elapsed time"); times[i] += ms;
            }
            ++timed_frames;
        }
        return u;
    }
    UpdateResult update(const PredictedState& p, const DetectionBatch& b, const MotionContext& m, const TrackerConfig& c = {}) {
        *predicted.host = to_device(p); *batch.host = to_device(b); *motion.host = to_device(m); *config.host = c;
        predicted.upload(stream); batch.upload(stream); motion.upload(stream); config.upload(stream);
        enqueue_associate_and_update(predicted.device, batch.device, motion.device, config.device, next.device, rows.device, scratch.device, stream);
        next.download(stream); rows.download(stream); wait();
        auto u = associate_and_update(p, b, m, c); compare(to_host(*next.host, *rows.host), u); return u;
    }
    DetectionBatch assemble(const DetectionCandidates& d, const PredictedState& p, const TrackerConfig& c = {}) {
        *candidates.host = d; *predicted.host = to_device(p); *config.host = c;
        candidates.upload(stream); predicted.upload(stream); config.upload(stream);
        enqueue_assemble_detection_batch(candidates.device, predicted.device, config.device, batch.device, scratch.device, stream);
        batch.download(stream); wait();
        auto b = assemble_detection_batch(d, p, c); compare(to_host(*batch.host), b); return b;
    }
};
constexpr Epoch epoch = 17;
MotionContext motion(FrameId k, MotionStatus status = MotionStatus::uninitialized, float tx = 0, float ty = 0) {
    MotionContext m; m.epoch = epoch; m.frame = k; m.reference_frame = k - 1; m.status = status; m.tx = tx; m.ty = ty; return m;
}
DetectionCandidate candidate(unsigned id, float x = 0, float score = .8f, unsigned flags = kDetectionKept,
                             float w = 10, float h = 20, float y = 0) {
    return {x, y, x + w, y + h, score, 0, id, flags};
}
DetectionCandidates candidates(FrameId k, const std::vector<DetectionCandidate>& v = {}) {
    DetectionCandidates out{}; out.header.epoch = epoch; out.header.frame = k; out.header.schema = kPostprocessSchema;
    out.header.count = static_cast<unsigned>(v.size()); out.header.capacity = kPostprocessCapacity; out.header.done = kPostprocessDone;
    std::copy(v.begin(), v.end(), out.rows.begin()); return out;
}
Track track(std::uint64_t uid, float x = 0, bool confirmed = true) {
    Track t; t.uid = t.output_id = uid; t.lifecycle = confirmed ? TrackLifecycle::confirmed : TrackLifecycle::tentative;
    const float box[4] = {x, 0, x + 10, 20}; math::kalman_init(box, t.x.data(), t.P.data(), .05f, .00625f);
    t.age = 1; t.hit_streak = 2; t.score_sum = 1.6f; t.observed_score = t.association_score = .8f;
    t.birth_frame = 0; t.last_observed_frame = 2; t.history_count = 1; t.history[0] = {x + 5, 10, 20}; t.h_ema = 20; return t;
}
PredictedState predicted(FrameId k, std::vector<Track> tracks = {}) {
    PredictedState p; p.epoch = epoch; p.frame = k; p.source_committed_frame = k - 1; p.motion = motion(k); p.tracks = std::move(tracks);
    for (const auto& t : p.tracks) p.next_uid = std::max(p.next_uid, t.uid + 1);
    return p;
}
BatchDetection detection(unsigned id, float x = 0, float score = .8f, bool birth = false) {
    return {{x, 0, x + 10, 20}, score, score, 0, id, true, birth};
}
DetectionBatch batch(FrameId k, std::vector<BatchDetection> rows = {}) {
    return {epoch, k, std::move(rows), Error::none};
}
void stream_tests(Harness& h) {
    context = "explicit streams";
    for (auto stream : {cudaStream_t{}, cudaStreamLegacy, cudaStreamPerThread}) {
        for (int stage = 0; stage < 3; ++stage) {
            bool rejected = false;
            try {
                if (stage == 0) enqueue_predict_tracks(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, stream);
                if (stage == 1) enqueue_assemble_detection_batch(nullptr, nullptr, nullptr, nullptr, nullptr, stream);
                if (stage == 2) enqueue_associate_and_update(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, stream);
            } catch (const std::invalid_argument&) { rejected = true; }
            check(rejected, "default stream accepted");
        }
    }
    h.reset(initial_track_state(epoch));
    context = "first empty frame"; auto u = h.step(initial_track_state(epoch), candidates(0), motion(0), 0);
    check(u.error == Error::none && u.next.committed_frame == 0, "empty commit");
}
void sequences(Harness& h) {
    auto state = initial_track_state(epoch); h.reset(state);
    for (int k = 0; k < 10; ++k) {
        context = "bridge sequence " + std::to_string(k);
        std::vector<DetectionCandidate> v;
        if (k <= 3 || k >= 6) v.push_back(candidate(5));
        const float camera = k >= 4 ? 100.f : 0.f;
        v.push_back(candidate(2, 200 + camera + k)); v.push_back(candidate(9, 400 + camera - k));
        auto u = h.step(state, candidates(k, v), motion(k, k == 4 ? MotionStatus::valid : MotionStatus::rejected, k == 4 ? 100.f : 0.f), k);
        check(u.error == Error::none, "bridge sequence error"); state = std::move(u.next);
    }
    check(state.tracks.size() == 3 && state.tracks.back().uid == 4 && state.tracks.back().output_id == 2, "bridge not exercised");
    std::mt19937 rng(0x5accadeu);
    for (unsigned run = 0; run < 3; ++run) {
        TrackerConfig c; c.assoc.top_k = run == 0 ? 1 : run == 1 ? 3 : 8;
        state = initial_track_state(epoch); h.reset(state, c);
        float camera_x = 0, camera_y = 0;
        for (int k = 0; k < 96; ++k) {
            const float tx = k % 9 == 0 ? 1.25f : 0, ty = k % 13 == 0 ? -.75f : 0;
            camera_x += tx; camera_y += ty;
            std::vector<DetectionCandidate> v;
            for (unsigned j = 0; j < 135; ++j) {
                if ((k + j) % 19 < 3) continue;
                const float x = static_cast<float>(j % 27) * 7 + camera_x + static_cast<float>(rng() % 5) * .125f;
                const float y = static_cast<float>(j / 27) * 12 + camera_y;
                const float score[] = {.8f, .45f, .28f, .1f, .05f, .02f};
                v.push_back(candidate(j * 2, x, score[(j + k / 7) % 6], k % 11 == 0 && j % 3 == 0 ? kDetectionSuppressed : kDetectionKept, 10, 20, y));
                if (j % 3 == 0) v.push_back(candidate(j * 2 + 1, x + 3, .6f, kDetectionSuppressed, 10, 20, y));
            }
            std::shuffle(v.begin(), v.end(), rng);
            context = "crowded seed run " + std::to_string(run) + " frame " + std::to_string(k);
            auto u = h.step(state, candidates(k, v), motion(k, k == 0 ? MotionStatus::uninitialized : MotionStatus::valid, tx, ty), k);
            check(u.error == Error::none, "crowded sequence error"); state = std::move(u.next);
        }
    }
}
void capacity_tests(Harness& h) {
    TrackerConfig c; c.life.confirm_streak = 1;
    for (unsigned n : {511u, 512u}) {
        std::vector<Track> tracks;
        for (unsigned i = 1; i <= n; ++i) tracks.push_back(track(i, -100));
        context = "track capacity " + std::to_string(n + 1);
        auto u = h.update(predicted(3, tracks), batch(3, {detection(1, 0, .8f, true)}), motion(3), c);
        check(u.error == (n == 511 ? Error::none : Error::overflow), "track capacity boundary");
    }
    for (unsigned n : {256u, 257u, 300u}) {
        std::vector<BatchDetection> rows;
        for (unsigned i = 0; i < n; ++i) rows.push_back(detection(n - i, static_cast<float>(i * 20), .8f, true));
        context = "row capacity " + std::to_string(n);
        auto u = h.update(predicted(0), batch(0, rows), motion(0), c);
        check(u.error == (n == 256 ? Error::none : Error::overflow), "row capacity boundary");
    }
    // Two observed groups reach exactly 512 active tracks without overflowing rows.
    auto state = initial_track_state(epoch); h.reset(state);
    for (int k = 0; k < 2; ++k) {
        std::vector<DetectionCandidate> v;
        for (unsigned i = 0; i < 256; ++i) v.push_back(candidate(i, static_cast<float>((i + k * 256) * 20)));
        context = "512 births frame " + std::to_string(k);
        auto u = h.step(state, candidates(k, v), motion(k), k); check(u.error == Error::none, "512 active births"); state = std::move(u.next);
        // Keep the first group confirmed so it survives the next empty match.
        if (k == 0) { for (auto& t : state.tracks) t.lifecycle = TrackLifecycle::confirmed; h.reset(state); }
    }
    check(state.tracks.size() == 512 && state.next_uid == 513, "512 births not reached");
    context = "513th birth full chain";
    auto u = h.step(state, candidates(2, {candidate(1, 20000)}), motion(2), 2);
    // Unmatched tentative tracks are removed before births in the reference.
    check(u.error == Error::none, "tentative removal before capacity");
    for (auto& t : state.tracks) t.lifecycle = TrackLifecycle::confirmed;
    h.reset(state); context = "513th birth confirmed full chain";
    u = h.step(state, candidates(2, {candidate(1, 20000)}), motion(2), 2);
    check(u.error == Error::overflow, "full chain track overflow");
}
void error_tests(Harness& h) {
    for (unsigned kind = 0; kind < 11; ++kind) {
        auto d = candidates(0, {candidate(1)});
        if (kind == 0) ++d.header.epoch;
        if (kind == 1) ++d.header.frame;
        if (kind == 2) d.header.done = 0;
        if (kind == 3) d.header.count = 301;
        if (kind == 4) d.header.flags = kPostprocessOverflow;
        if (kind == 5) d.header.flags = kPostprocessNumeric;
        if (kind == 6) d.header.flags = kPostprocessUpstream;
        if (kind == 7) d.header.flags = kPostprocessProtocol;
        if (kind == 8) d.header.flags = kPostprocessOverflow | kPostprocessUpstream;
        if (kind == 9) { d.header.count = 301; d.header.flags = kPostprocessOverflow | 0x80000000u; }
        if (kind == 10) { d.header.schema = 99; d.header.capacity = 0; }
        auto state = initial_track_state(epoch); h.reset(state); context = "header case " + std::to_string(kind);
        auto u = h.step(state, d, motion(0), 0);
        check((u.error == Error::none) == (kind == 10), "reference ignores schema/capacity");
    }
    for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        for (unsigned field = 0; field < 5; ++field) {
            auto d = candidates(0, {candidate(1), candidate(2, 20, .001f, 0)});
            float* fields[] = {&d.rows[1].x1, &d.rows[1].y1, &d.rows[1].x2, &d.rows[1].y2, &d.rows[1].score}; *fields[field] = value;
            auto state = initial_track_state(epoch); h.reset(state); context = "nonfinite candidate " + std::to_string(field);
            check(h.step(state, d, motion(0), 0).error == Error::numeric, "candidate numeric");
        }
    }
    for (unsigned kind = 0; kind < 9; ++kind) {
        auto p = predicted(3, {track(1)}); auto b = batch(3); auto m = motion(3);
        if (kind == 0) ++b.epoch;
        if (kind == 1) ++b.frame;
        if (kind == 2) ++m.epoch;
        if (kind == 3) ++m.frame;
        if (kind == 4) ++p.motion.epoch;
        if (kind == 5) ++p.motion.frame;
        if (kind == 6) --p.source_committed_frame;
        if (kind == 7) ++p.epoch;
        if (kind == 8) ++p.frame;
        context = "F07 identity " + std::to_string(kind);
        check(h.update(p, b, m).error == Error::protocol, "identity protocol");
    }
    for (unsigned kind = 0; kind < 12; ++kind) {
        auto p = predicted(3, {track(1)}); auto b = batch(3, {detection(1)});
        const float inf = std::numeric_limits<float>::infinity();
        if (kind == 0) p.tracks[0].x[0] = inf;
        if (kind == 1) p.tracks[0].P[1] = inf;
        if (kind == 2) p.tracks[0].score_sum = inf;
        if (kind == 3) p.tracks[0].h_ema = inf;
        if (kind == 4) p.tracks[0].observed_score = inf;
        if (kind == 5) p.tracks[0].association_score = inf;
        if (kind == 6) p.tracks[0].history[0].cx = inf;
        if (kind == 7) b.rows[0].observed_score = inf;
        if (kind == 8) b.rows[0].association_score = inf;
        if (kind == 9) b.rows[0].xyxy[0] = inf;
        if (kind == 10) p.error = Error::numeric;
        if (kind == 11) { p.error = Error::numeric; b.error = Error::producer; ++b.frame; }
        context = "update error " + std::to_string(kind);
        check(h.update(p, b, motion(3)).error != Error::none, "update input error");
    }
    for (unsigned kind = 0; kind < 9; ++kind) {
        TrackState state{epoch, 2, 4, {track(1), track(2), track(3)}};
        if (kind == 0) state.tracks[1].x[0] = std::numeric_limits<float>::infinity();
        if (kind == 1) state.tracks[1].P[0] = std::numeric_limits<float>::quiet_NaN();
        if (kind == 2) state.tracks[1].history[0].h = std::numeric_limits<float>::infinity();
        if (kind == 3) state.tracks[1].x[3] = .99f;
        if (kind == 4) state.tracks[1].x[2] = 0;
        if (kind == 5) state.tracks[1].age = 30;
        if (kind == 6) state.tracks[1].x[3] = std::numeric_limits<float>::max();
        if (kind == 7) state.tracks[1].x[4] = std::numeric_limits<float>::denorm_min();
        if (kind == 8) state.tracks[1].x[4] = -0.0f;
        h.reset(state); context = "predict edge " + std::to_string(kind);
        h.step(state, candidates(3), motion(3), 3);
    }
}
void decision_tests(Harness& h) {
    for (unsigned top_k : {1u, 2u, 3u, 8u}) {
        TrackerConfig c; c.assoc.top_k = top_k;
        for (float score : {.8f, .45f, .28f, .1f, .05f, .049f}) {
            for (bool confirmed : {false, true}) {
                context = "auction score/topK/UID ties";
                auto p = predicted(3, {track(1, 0, confirmed), track(2, 0, confirmed), track(3, 1, confirmed)});
                auto d9 = detection(9, 0, score), d2 = detection(2, 0, score); d9.class_id = 9; d2.class_id = 2;
                h.update(p, batch(3, {d9, d2, detection(7, -1, .1f)}), motion(3), c);
                c.assoc.stability_bid_w = 0;
                h.update(p, batch(3, {d9}), motion(3), c);
            }
        }
    }
    for (unsigned kind = 0; kind < 18; ++kind) {
        context = "bridge gates/ties " + std::to_string(kind);
        auto lost = track(1, -100), fresh = track(2);
        lost.output_id = 71; lost.age = 7; lost.last_observed_frame = 3; lost.history_count = 4;
        for (auto& sample : lost.history) sample = {5, 10, 20};
        fresh.birth_frame = 7; fresh.last_observed_frame = 9; fresh.hit_streak = 3; fresh.score_sum = 2.4f; fresh.history_count = 3;
        for (auto& sample : fresh.history) sample = {5, 10, 20};
        TrackerConfig c;
        if (kind == 1) fresh.hit_streak = 2;
        if (kind == 2) fresh.birth_frame = 6;
        if (kind == 3) fresh.output_id = 99;
        if (kind == 4) lost.age = 1;
        if (kind == 5) lost.age = 31;
        if (kind == 6) lost.last_observed_frame = fresh.birth_frame;
        if (kind == 7) lost.history_count = 3;
        if (kind == 8) lost.lifecycle = TrackLifecycle::tentative;
        if (kind == 9) lost.history[3].cx = 100;
        if (kind == 10) lost.h_ema = .59f * 20;
        if (kind == 11) lost.h_ema = 1.71f * 20;
        if (kind == 12) c.bridge.enabled = false;
        auto p = predicted(10, {lost, fresh}); auto b = batch(10, {detection(1)});
        if (kind == 13 || kind == 14) {
            auto other = lost; other.uid = 3; other.output_id = 73; other.x[0] = -195;
            p.tracks.push_back(other); p.next_uid = 4;
            if (kind == 14) c.bridge.margin = 0;
        }
        if (kind >= 15) {
            auto receiver = fresh; receiver.uid = receiver.output_id = 3; receiver.x[0] = 35;
            p.tracks.push_back(receiver); p.next_uid = 4;
            b.rows.push_back(detection(2, 30, kind == 15 ? .9f : .8f));
            if (kind == 17) {
                p.tracks[0].history[0].cx = 2; p.tracks[0].history[1].cx = 3; p.tracks[0].history[2].cx = 4;
            }
        }
        auto u = h.update(p, b, motion(10), c);
        if (kind == 0) check(u.next.tracks.size() == 1 && u.next.tracks[0].output_id == 71, "bridge fixture not exercised");
    }
    auto p = predicted(0); p.priors = {{1, {0, 0, 10, 20}, 0}};
    for (unsigned kind = 0; kind < 10; ++kind) {
        context = "continuation gates " + std::to_string(kind);
        TrackerConfig c; c.continuation.prior_iou = .5f; c.continuation.dedup_iou = .5f;
        auto d = candidates(0, {candidate(8, 0, .9f, kDetectionSuppressed, 5), candidate(1, 0, .01f)});
        auto pp = p;
        if (kind == 1) c.continuation.prior_iou = std::nextafter(.5f, 1.f);
        if (kind == 2) c.continuation.dedup_iou = std::nextafter(.5f, 0.f);
        if (kind == 3) c.continuation.max_candidates = 0;
        if (kind == 4) c.continuation.enabled = false;
        if (kind == 5) pp.priors.clear();
        if (kind == 6) pp.error = Error::numeric;
        if (kind == 7) pp.priors[0].xyxy[0] = std::numeric_limits<float>::infinity();
        if (kind == 8) d.rows[0].score = .099f;
        if (kind == 9) d.rows[0].flags |= kDetectionKept;
        auto b = h.assemble(d, pp, c);
        h.update(predicted(0), b, motion(0), c);
    }
    for (unsigned cap : {0u, 2u, 50u, 300u}) {
        TrackerConfig c; c.continuation.max_candidates = cap; c.continuation.dedup_iou = 1;
        std::vector<DetectionCandidate> v;
        for (unsigned i = 0; i < 300; ++i) v.push_back(candidate(300 - i, 0, i % 2 ? .8f : .6f, kDetectionSuppressed));
        context = "continuation cap " + std::to_string(cap);
        auto b = h.assemble(candidates(0, v), p, c);
        check(b.rows.size() == cap, "continuation cap not exercised");
    }
    for (unsigned cap : {0u, 1u, 2u}) {
        TrackerConfig c; c.continuation.max_candidates = cap; c.continuation.dedup_iou = 1;
        const auto d = candidates(0, {candidate(1, 0, .9f, kDetectionSuppressed),
            candidate(2, 0, .8f, kDetectionSuppressed, 1e30f, 1e30f)});
        context = "deferred continuation NUMERIC cap " + std::to_string(cap);
        auto b = h.assemble(d, p, c);
        check(b.error == (cap == 2 ? Error::numeric : Error::none), "unvisited gate must not propagate NUMERIC");
    }
    for (bool match : {false, true}) {
        auto d = candidates(0, {candidate(1, match ? 0.f : 100.f, .9f, kDetectionSuppressed),
            candidate(2, 0, .01f, kDetectionKept, 1e30f, 1e30f)});
        context = "kept NUMERIC only after matching prior";
        auto b = h.assemble(d, p);
        check(b.error == (match ? Error::numeric : Error::none), "unvisited dedup must not propagate NUMERIC");
    }

}
void recorded(Harness& h, const std::filesystem::path& dir) {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const auto name = entry.path().filename().string();
        if (name.starts_with("MOT17-") && name.ends_with("-SDP.scdr")) paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
    check(!paths.empty(), "no MOT17 SDP records");
    std::size_t total = 0;
    for (const auto& path : paths) {
        auto motion_path = path; motion_path.replace_extension(".scmo");
        const auto detections = read_detection_records(path);
        const auto motions = read_motion_records(motion_path);
        check(detections.epoch == motions.epoch && detections.frames.size() == motions.frames.size(), "record identity/count");
        auto state = initial_track_state(detections.epoch); h.reset(state);
        const bool time = path.stem() == "MOT17-04-SDP";
        for (std::size_t k = 0; k < detections.frames.size(); ++k) {
            context = path.stem().string() + " frame " + std::to_string(k);
            auto u = h.step(state, detections.frames[k], motions.frames[k], static_cast<FrameId>(k), time);
            check(u.error == Error::none, "recorded frame failed"); state = std::move(u.next);
        }
        total += detections.frames.size();
        std::cout << path.stem().string() << ": " << detections.frames.size() << " frames, zero mismatches\n";
    }
    std::cout << "MOT17 total: " << paths.size() << " sequences, " << total << " frames, zero mismatches\n";
    if (h.timed_frames) {
        std::cout << std::fixed << std::setprecision(3) << "MOT17-04-SDP mean GPU time/frame (" << h.timed_frames << " frames): "
                  << "F08a=" << h.times[0] * 1000 / h.timed_frames << " us, F04b=" << h.times[1] * 1000 / h.timed_frames
                  << " us, F07/F08b=" << h.times[2] * 1000 / h.timed_frames << " us\n";
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        Harness h;
        stream_tests(h); sequences(h); capacity_tests(h); error_tests(h); decision_tests(h);
        if (argc > 1) recorded(h, argv[1]);
        std::cout << "tracking_cuda_tests: " << checks << " checks passed\n"; return 0;
    } catch (const std::exception& e) {
        std::cerr << "tracking_cuda_tests: " << e.what() << '\n'; return 1;
    }
}

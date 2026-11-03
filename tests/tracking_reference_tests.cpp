#include "saccade/tracking.hpp"
#include "saccade/tracking_math.hpp"
#include "saccade/sha256.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace saccade;
namespace math = saccade::tracking_math;
int checks = 0;
void check(bool ok, const std::string& message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
void near(float actual, double expected, double tolerance, const std::string& message) {
    check(std::abs(static_cast<double>(actual) - expected) <= tolerance, message);
}
constexpr Epoch epoch = 17;
MotionContext motion(FrameId frame, MotionStatus status = MotionStatus::uninitialized, float tx = 0, float ty = 0) {
    MotionContext m;
    m.epoch = epoch;
    m.frame = frame;
    m.reference_frame = frame - 1;
    m.status = status;
    m.tx = tx;
    m.ty = ty;
    return m;
}
DetectionCandidate candidate(std::uint32_t id, float x = 0, float score = 0.8f,
                             std::uint32_t flags = kDetectionKept, float w = 10, float h = 20, float y = 0) {
    return {x, y, x + w, y + h, score, 0, id, flags};
}
DetectionCandidates candidates(FrameId frame, const std::vector<DetectionCandidate>& rows = {}) {
    DetectionCandidates c{};
    c.header.epoch = epoch;
    c.header.frame = static_cast<std::uint64_t>(frame);
    c.header.schema = kPostprocessSchema;
    c.header.capacity = kPostprocessCapacity;
    c.header.count = static_cast<std::uint32_t>(rows.size());
    c.header.done = kPostprocessDone;
    std::copy(rows.begin(), rows.end(), c.rows.begin());
    return c;
}
BatchDetection detection(std::uint32_t id, float x = 0, float score = 0.8f, float w = 10, float h = 20) {
    return {{x, 0, x + w, h}, score, score, 0, id, true, false};
}
Track track(std::uint64_t uid, float x = 0, bool confirmed = true, float w = 10, float h = 20) {
    Track t;
    t.uid = t.output_id = uid;
    t.lifecycle = confirmed ? TrackLifecycle::confirmed : TrackLifecycle::tentative;
    const float box[4] = {x, 0, x + w, h};
    // Fixture initialization is explicit, independent of kalman_init.
    t.x = {x + w / 2, h / 2, w / h, h, 0, 0, 0, 0};
    const float diag[] = {4, 4, 1e-4f, 4, 1.5625f, 1.5625f, 1e-10f, 1.5625f};
    for (int i = 0; i < 8; ++i) t.P[i * 8 + i] = diag[i];
    t.age = 1;
    t.hit_streak = 2;
    t.score_sum = 1.6f;
    t.observed_score = t.association_score = 0.8f;
    t.birth_frame = 0;
    t.last_observed_frame = 2;
    t.history_count = 1;
    t.history[0] = {(box[0] + box[2]) / 2, h / 2, h};
    t.h_ema = h;
    return t;
}
PredictedState predicted(FrameId frame, std::vector<Track> tracks = {}) {
    PredictedState p;
    p.epoch = epoch;
    p.frame = frame;
    p.source_committed_frame = frame - 1;
    p.motion = motion(frame);
    p.tracks = std::move(tracks);
    p.next_uid = 1;
    for (const auto& t : p.tracks) p.next_uid = std::max(p.next_uid, t.uid + 1);
    return p;
}
UpdateResult update(PredictedState p, std::vector<BatchDetection> rows, const TrackerConfig& c = {}) {
    DetectionBatch b;
    b.epoch = epoch;
    b.frame = p.frame;
    b.rows = std::move(rows);
    return associate_and_update(p, b, p.motion, c);
}
const Track& find(const TrackState& state, std::uint64_t uid) {
    const auto it = std::find_if(state.tracks.begin(), state.tracks.end(), [&](const auto& t) { return t.uid == uid; });
    if (it == state.tracks.end()) throw std::runtime_error("fixture track absent: " + std::to_string(uid));
    return *it;
}
void math_tests() {
    for (int i = 0; i <= 1750000; ++i) {
        const float x = -87.0f + static_cast<float>(i) * 0.0001f;
        const float expected = static_cast<float>(std::exp(static_cast<double>(x)));
        const float actual = math::det_expf(x);
        const auto a = std::bit_cast<std::uint32_t>(actual), b = std::bit_cast<std::uint32_t>(expected);
        check((a > b ? a - b : b - a) <= 2, "det_expf dense ULP at " + std::to_string(x));
    }
    check(math::det_expf(0) == 1 && math::det_expf(-0.0f) == 1, "exp signed zero");
    check(math::det_expf(-100) == math::det_expf(-87) && math::det_expf(100) == math::det_expf(88), "exp clamp");
    check(math::float_bits(math::det_expf(1)) == 0x402df854u, "exp(1) exact rounded point");
    check(math::float_bits(math::det_expf(-1)) == 0x3ebc5ab2u, "exp(-1) exact rounded point");
    float x[8]{}, box[4] = {2, 4, 12, 24}, back[4];
    math::box_to_state(box, x);
    check(x[0] == 7 && x[1] == 14 && x[2] == 0.5f && x[3] == 20, "box to state");
    math::state_to_box(x, back);
    check(std::equal(box, box + 4, back), "state to box");
    const float a[4] = {0, 0, 10, 10}, b[4] = {0, 0, 5, 10}, empty[4] = {0, 0, 0, 0}, separate[4] = {10, 0, 20, 10};
    check(math::iou(a, b) == 0.5f && math::iou(a, separate) == 0 && math::iou(empty, empty) == 0, "IoU area/boundaries");
    check(math::aspect_gain(0.15f) == 1 && math::aspect_gain(0.8f) == 1, "aspect central boundaries");
    near(math::aspect_gain(0.1f), 0.75, 1e-7, "aspect narrow");
    near(math::aspect_gain(1), 0.8, 1e-7, "aspect wide");
    check(math::aspect_gain(0) == 0.5f && math::aspect_gain(3) == 0.5f, "aspect lower bound");
    check(math::height_similarity(20, 10) == 0.5f && math::height_similarity(0, 0) == 1, "height stability");
    check(math::association_cost(1, 20, 20, 10, 0.2f) == 0, "cost lower clamp");
    check(math::association_cost(0, 20, 20, 10, 0.2f) == 1, "cost upper boundary");
    near(math::association_cost(0.5f, 20, 10, 0.5f, 0.2f), 1 - 0.5 * std::exp(0.1), 1e-7, "reward lambda floor");
    near(math::association_value(0.5f, 10, 20, 10), std::exp(-5.0), 1e-8, "p formula");
    near(math::auction_bid(0.8f, 0.3f, true, 0.01f, 0.1f, 20, 10), 0.56, 1e-7, "two-option bid");
    near(math::auction_bid(0.8f, 0.3f, false, 0.01f, 0.1f, 20, 10), 0.06, 1e-7, "one-option bid");
    check(math::regression_velocity(0, 2, 4, 6) == 2, "four-point regression");
    check(math::bridge_distance(0, 0, 4, 0, 2, 0, 2, 0, 2, 20, 20, 0.01f) == 0, "moving bridge distance");
    check(math::bridge_distance(0, 0, 5, 0, 0, 0, 0, 0, 2, 20, 20, 0.12f) == 0.25f, "stationary bridge distance");
    near(math::bridge_distance(0, 0, 5, 0, 0.25f, 0, 0.25f, 0, 2, 20, 20, 0.05f), 0.2375, 1e-7, "fractional speed weight");
    check(math::bridge_distance(0, 0, 1, 0, 0, 0, 0, 0, 1, 0.5f, 0.5f, 1) == 1, "bridge reference height floor");
}

void kalman_tests() {
    const float box[4] = {0, 0, 10, 20};
    float x[8], P[64];
    math::kalman_init(box, x, P, 0.05f, 0.00625f);
    const double diagonal[8] = {4, 4, 1e-4, 4, 1.5625, 1.5625, 1e-10, 1.5625};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) near(P[i * 8 + j], i == j ? diagonal[i] : 0, 1e-10, "init covariance");
    check(x[0] == 5 && x[1] == 10 && x[2] == 0.5f && x[3] == 20 && x[4] == 0, "Kalman init state");
    x[4] = 2; x[5] = 3; x[6] = 0.01f; x[7] = 4;
    double dx[8], dp[64], F[64]{}, R[16]{};
    for (int i = 0; i < 8; ++i) { dx[i] = x[i]; F[i * 8 + i] = 1; if (i < 4) F[i * 8 + i + 4] = 1; }
    for (int i = 0; i < 64; ++i) dp[i] = P[i];
    double xp[8]{}, pp[64]{};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) xp[i] += F[i * 8 + j] * dx[j];
        for (int j = 0; j < 8; ++j)
            for (int k = 0; k < 8; ++k)
                for (int l = 0; l < 8; ++l) pp[i * 8 + j] += F[i * 8 + k] * dp[k * 8 + l] * F[j * 8 + l];
    }
    const double sp = static_cast<double>(0.05f) * xp[3], sv = static_cast<double>(0.00625f) * xp[3];
    const double q[8] = {sp * sp, sp * sp, 1e-4, sp * sp, sv * sv, sv * sv, 1e-10, sv * sv};
    for (int i = 0; i < 8; ++i) pp[i * 8 + i] += q[i];
    math::kalman_predict(x, P, 0.05f, 0.00625f);
    for (int i = 0; i < 8; ++i) near(x[i], xp[i], 2e-6, "predict state binary64");
    for (int i = 0; i < 64; ++i) near(P[i], pp[i], 2e-6, "predict covariance binary64");
    const float obs[4] = {3, 4, 15, 30};
    const double z[4] = {9, 17, 12.0 / 26, 26};
    for (int i = 0; i < 4; ++i) R[i * 4 + i] = (i == 2 ? 0.01 : sp * sp) * 3.5;
    // This covariance has independent position/velocity pairs, so S is diagonal.
    double K[32]{};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) K[i * 4 + j] = pp[i * 8 + j] / (pp[j * 8 + j] + R[j * 4 + j]);
    double expected_x[8], expected_p[64];
    for (int i = 0; i < 8; ++i) {
        expected_x[i] = xp[i];
        for (int j = 0; j < 4; ++j) expected_x[i] += K[i * 4 + j] * (z[j] - xp[j]);
        for (int j = 0; j < 8; ++j) {
            expected_p[i * 8 + j] = pp[i * 8 + j];
            for (int k = 0; k < 4; ++k) expected_p[i * 8 + j] -= K[i * 4 + k] * pp[k * 8 + j];
        }
    }
    math::kalman_update(x, P, obs, 0.05f, 3.5f);
    for (int i = 0; i < 8; ++i) near(x[i], expected_x[i], 3e-6, "update state binary64");
    for (int i = 0; i < 64; ++i) near(P[i], expected_p[i], 3e-6, "update covariance binary64");

    // Correlated S = D + 11^T has a closed-form Sherman-Morrison inverse.
    const double original_x[8] = {5, 10, 0.5, 20, 0, 0, 0, 0};
    double original_p[64]{};
    for (int i = 0; i < 8; ++i) {
        x[i] = static_cast<float>(original_x[i]);
        for (int j = 0; j < 8; ++j) {
            const double value = i < 4 && j < 4 ? (i == j ? 4 : 1) :
                i == j ? 1.5625 : (i == j + 4 || j == i + 4) ? 0.25 : 0;
            original_p[i * 8 + j] = value;
            P[i * 8 + j] = static_cast<float>(value);
        }
    }
    const double D[4] = {6.5, 6.5, 3.035, 6.5};
    double denominator = 1;
    for (double v : D) denominator += 1 / v;
    double inverse[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            inverse[i * 4 + j] = (i == j ? 1 / D[i] : 0) - (1 / D[i]) * (1 / D[j]) / denominator;
    const float correlated_box[4] = {3, 2, 15, 24};
    const double correlated_z[4] = {9, 13, 12.0 / 22, 22};
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 4; ++j) {
            K[i * 4 + j] = 0;
            for (int k = 0; k < 4; ++k) K[i * 4 + j] += original_p[i * 8 + k] * inverse[k * 4 + j];
        }
    for (int i = 0; i < 8; ++i) {
        expected_x[i] = original_x[i];
        for (int j = 0; j < 4; ++j) expected_x[i] += K[i * 4 + j] * (correlated_z[j] - original_x[j]);
        for (int j = 0; j < 8; ++j) {
            expected_p[i * 8 + j] = original_p[i * 8 + j];
            for (int k = 0; k < 4; ++k) expected_p[i * 8 + j] -= K[i * 4 + k] * original_p[k * 8 + j];
        }
    }
    math::kalman_update(x, P, correlated_box, 0.05f, 3.5f);
    for (int i = 0; i < 8; ++i) near(x[i], expected_x[i], 3e-6, "correlated state closed form");
    for (int i = 0; i < 64; ++i) near(P[i], expected_p[i], 3e-6, "correlated covariance closed form");
}

void predict_tests() {
    auto state = initial_track_state(epoch);
    check(state.committed_frame == -1 && state.next_uid == 1 && state.tracks.empty(), "initial state");
    auto first = predict_tracks(state, motion(0), 0, {});
    check(first.tracks.empty() && first.priors.empty() && first.source_committed_frame == -1 && first.frame == 0, "first prediction");
    auto t = track(1);
    t.x[4] = 2; t.x[5] = -1; t.age = 0;
    state.tracks = {t}; state.committed_frame = 0;
    for (auto status : {MotionStatus::uninitialized, MotionStatus::rejected, MotionStatus::disabled, MotionStatus::valid}) {
        auto p = predict_tracks(state, motion(1, status, 8, 4), 1, {});
        check(p.error == Error::none && p.tracks.size() == 1 && p.priors.size() == 1, "predict survives");
        check(p.tracks[0].x[0] == (status == MotionStatus::valid ? 15 : 7) &&
              p.tracks[0].x[1] == (status == MotionStatus::valid ? 13 : 9) && p.tracks[0].age == 1, "control plus one predict");
        auto u = update(p, {});
        check(u.next.tracks[0].x == p.tracks[0].x && u.next.tracks[0].age == 1, "F08b does not predict again");
    }
    state.tracks = {track(1), track(2), track(3), track(4), track(5), track(6)};
    state.tracks[0].age = 29; state.tracks[1].age = 30;
    state.tracks[2].x[3] = 0.999f; state.tracks[3].x[2] = 0;
    state.tracks[4].x[2] = -0.1f; state.tracks[5].x[3] = 1;
    auto p = predict_tracks(state, motion(1), 1, {});
    check(p.tracks.size() == 2 && p.tracks[0].uid == 1 && p.tracks[0].age == 30 && p.tracks[1].uid == 6, "removal exact boundaries");
    state.tracks = {track(1, 0, false)};
    check(predict_tracks(state, motion(1), 1, {}).priors.empty(), "tentative excluded from prior");
    for (int kind = 0; kind < 4; ++kind) {
        state.tracks = {track(1)};
        if (kind == 0) state.tracks[0].x[0] = std::numeric_limits<float>::quiet_NaN();
        if (kind == 1) state.tracks[0].P[0] = std::numeric_limits<float>::infinity();
        if (kind == 2) state.tracks[0].x[4] = std::numeric_limits<float>::max();
        if (kind == 2) state.tracks[0].x[0] = std::numeric_limits<float>::max();
        auto m = motion(1, MotionStatus::valid);
        if (kind == 3) m.tx = std::numeric_limits<float>::infinity();
        check(predict_tracks(state, m, 1, {}).error == Error::numeric, "nonfinite predict numeric");
    }
}

void batch_tests() {
    auto p = predicted(0);
    auto input = candidates(0, {candidate(5, 100, 0.05f), candidate(2, 0, std::nextafter(0.05f, 0.0f))});
    auto b = assemble_detection_batch(input, p, {});
    check(b.rows.size() == 1 && b.rows[0].candidate_id == 5 && b.rows[0].can_birth && b.rows[0].can_continue &&
          b.rows[0].observed_score == b.rows[0].association_score, "kept threshold equality");
    for (int kind = 0; kind < 6; ++kind) {
        auto bad = input;
        if (kind == 0) ++bad.header.epoch;
        if (kind == 1) ++bad.header.frame;
        if (kind == 2) bad.header.done = 0;
        if (kind == 3) bad.header.count = kPostprocessCapacity + 1;
        if (kind == 4) bad.header.flags = kPostprocessNumeric;
        if (kind == 5) bad.header.flags = kPostprocessOverflow;
        b = assemble_detection_batch(bad, p, {});
        check(b.rows.empty() && b.error == (kind < 4 ? Error::protocol : kind == 4 ? Error::producer : Error::overflow), "header error mapping");
    }
    input.header.flags = kPostprocessOverflow | kPostprocessUpstream;
    check(assemble_detection_batch(input, p, {}).error == (Error::overflow | Error::producer), "header combined bits");
    TrackerConfig c;
    c.continuation.prior_iou = 0.5f; c.continuation.dedup_iou = 0.5f;
    p.priors = {{1, {0, 0, 10, 20}, 0}};
    // Suppressed occupies half of prior and overlaps kept by exactly 1/2.
    input = candidates(0, {candidate(8, 0, 0.9f, kDetectionSuppressed, 5), candidate(1, 0, 0.01f)});
    b = assemble_detection_batch(input, p, c);
    check(b.rows.size() == 1 && b.rows[0].candidate_id == 8 && !b.rows[0].can_birth && b.rows[0].can_continue &&
          b.rows[0].observed_score == 0.9f && b.rows[0].association_score == c.life.new_track_thresh, "continuation equality and clamp");
    auto no_birth = update(predicted(0), b.rows);
    check(no_birth.next.tracks.empty(), "high observed continuation cannot birth");
    c.continuation.prior_iou = std::nextafter(0.5f, 1.0f);
    check(assemble_detection_batch(input, p, c).rows.empty(), "prior below threshold");
    c.continuation.prior_iou = 0.5f; c.continuation.dedup_iou = std::nextafter(0.5f, 0.0f);
    check(assemble_detection_batch(input, p, c).rows.empty(), "dedup above threshold against low kept");
    c.continuation.dedup_iou = 1;
    input = candidates(0, {candidate(9, 0, 0.2f, kDetectionSuppressed), candidate(4, 0, 0.2f, kDetectionSuppressed),
                           candidate(2, 0, 0.3f, kDetectionSuppressed), candidate(1, 0, 0.1f, kDetectionSuppressed)});
    c.continuation.max_candidates = 2;
    b = assemble_detection_batch(input, p, c);
    check(b.rows.size() == 2 && b.rows[0].candidate_id == 2 && b.rows[1].candidate_id == 4 &&
          b.rows[1].association_score == 0.2f, "continuation cap and score/id order");
    c.continuation.max_candidates = 4;
    check(assemble_detection_batch(input, p, c).rows.size() == 4, "continuation min score equality");
    input.rows[3].score = std::nextafter(0.1f, 0.0f);
    check(assemble_detection_batch(input, p, c).rows.size() == 3, "continuation min score below");
    c.continuation.dedup_iou = 0.7f;
    check(assemble_detection_batch(input, p, c).rows.size() == 1, "accepted continuations deduplicate later ones");
    c.continuation.max_candidates = 0;
    check(assemble_detection_batch(input, p, c).rows.empty(), "zero continuation cap");
    c.continuation.max_candidates = 50; c.continuation.enabled = false;
    check(assemble_detection_batch(input, p, c).rows.empty(), "continuation disabled");
    c.continuation.enabled = true; p.priors.clear();
    check(assemble_detection_batch(input, p, c).rows.empty(), "continuation needs prior");
    p.priors = {{1, {0, 0, 10, 20}, 0}};
    c.continuation.prior_iou = c.continuation.dedup_iou = 0.5f;
    input = candidates(0, {candidate(1, 0, 0.9f, kDetectionSuppressed, 5),
                           candidate(2, 0, 0.8f, kDetectionSuppressed)});
    check(assemble_detection_batch(input, p, c).rows.size() == 2, "accepted continuation dedup equality");
    c.continuation.dedup_iou = std::nextafter(0.5f, 0.0f);
    check(assemble_detection_batch(input, p, c).rows.size() == 1, "accepted continuation dedup above cap");
    input = candidates(0, {candidate(9), candidate(3), candidate(5)});
    b = assemble_detection_batch(input, p, c);
    check(b.rows[0].candidate_id == 3 && b.rows[1].candidate_id == 5 && b.rows[2].candidate_id == 9, "batch ID order");
}

void association_tests() {
    TrackerConfig c;
    c.assoc.stability_cost_w = 0;
    for (const float boundary : {c.assoc.high_thresh, c.assoc.mid_thresh}) {
        auto first = detection(1, 1, boundary), second = detection(2, 0, c.batch.track_thresh);
        first.class_id = 1; second.class_id = 2;
        if (boundary == c.assoc.high_thresh) second.association_score = c.assoc.mid_thresh;
        auto staged = update(predicted(3, {track(1)}), {first, second}, c);
        check(staged.next.tracks[0].class_id == 1, "score equality enters earlier stage");
        first.association_score = std::nextafter(boundary, 0.0f);
        staged = update(predicted(3, {track(1)}), {first, second}, c);
        check(staged.next.tracks[0].class_id == 2, "score below boundary enters later stage");
    }
    for (int stage = 0; stage < 4; ++stage) {
        const bool confirmed = stage != 2;
        const float lower = stage == 0 ? c.assoc.high_thresh : stage == 3 ? c.batch.track_thresh : c.assoc.mid_thresh;
        auto d = detection(1, 0, lower, 5);
        auto u = update(predicted(3, {track(1, 0, confirmed)}), {d}, c);
        check(u.error == Error::none && u.next.tracks.size() == 1 && u.next.tracks[0].age == 0, "stage lower score and cost equality");
        if (stage == 3) c.assoc.stage2_match_thresh = std::nextafter(0.5f, 0.0f);
        else c.assoc.match_thresh = std::nextafter(0.5f, 0.0f);
        u = update(predicted(3, {track(1, 0, confirmed)}), {d}, c);
        check(confirmed ? u.next.tracks[0].age == 1 : u.next.tracks.empty(), "stage cost above cap");
        c.assoc.match_thresh = c.assoc.stage2_match_thresh = 0.5f;
    }
    for (const float score : {std::nextafter(c.batch.track_thresh, 0.0f), 0.0f}) {
        auto u = update(predicted(3, {track(1)}), {detection(1, 0, score)}, c);
        check(u.next.tracks[0].age == 1, "below all score ranges");
    }
    auto u = update(predicted(3, {track(1, 0, false)}), {detection(1, 0, std::nextafter(c.assoc.mid_thresh, 0.0f))}, c);
    check(u.next.tracks.empty(), "tentative not eligible for S4");
    u = update(predicted(3, {track(1), track(2, 0, false)}), {detection(1, 0, c.assoc.high_thresh)}, c);
    check(u.next.tracks.size() == 1 && u.next.tracks[0].uid == 1, "S1 before S3");
    u = update(predicted(3, {track(1), track(2, 0, false)}), {detection(1, 0, c.assoc.mid_thresh)}, c);
    check(u.next.tracks.size() == 1 && u.next.tracks[0].uid == 1, "S2 before S3");
    // Confirmed cannot take low detections until S4; tentative misses in S3.
    u = update(predicted(3, {track(1), track(2, 0, false)}), {detection(1, 0, c.batch.track_thresh)}, c);
    check(u.next.tracks.size() == 1 && u.next.tracks[0].age == 0, "S4 confirmed only");
    u = update(predicted(3, {track(1), track(2)}), {detection(1), detection(2, 0, c.assoc.mid_thresh)}, c);
    check(u.next.tracks.size() == 2 && find(u.next, 1).observed_score == 0.8f &&
          find(u.next, 2).observed_score == c.assoc.mid_thresh, "loser matches in later stage");
    u = update(predicted(3, {track(1), track(2)}), {detection(1), detection(2)}, c);
    check(find(u.next, 1).age == 0 && find(u.next, 2).age == 1, "auction loser does not retry same stage");
    u = update(predicted(3, {track(1)}), {detection(9), detection(2)}, c);
    check(u.next.tracks[0].age == 0, "top-K equal p uses candidate ID");
    auto d9 = detection(9), d2 = detection(2);
    d9.class_id = 9; d2.class_id = 2;
    u = update(predicted(3, {track(1)}), {d9, d2}, c);
    check(u.next.tracks[0].class_id == 2, "equal-value candidate ID and person class ignored");
    c.assoc.stability_bid_w = 0;
    u = update(predicted(3, {track(1, 1), track(2)}), {detection(1)}, c);
    check(find(u.next, 2).age == 0 && find(u.next, 1).age == 1, "equal-bid tie resolves p before UID");
    u = update(predicted(3, {track(1), track(2)}), {detection(1)}, c);
    check(find(u.next, 1).age == 0 && find(u.next, 2).age == 1, "equal-bid equal-p resolves UID");
    // Track 1 has one admissible option; track 2 has a close second. Delta wins over p/UID.
    auto t1 = track(1, -2), t2 = track(2);
    u = update(predicted(3, {t1, t2}), {detection(1, 0), detection(2, 2)}, c);
    check(find(u.next, 2).age == 0, "two-candidate delta bid");
    c.assoc.top_k = 1;
    u = update(predicted(3, {track(1), track(2, 1)}), {detection(1), detection(2, -1)}, c);
    check(find(u.next, 1).age == 0 && find(u.next, 2).age == 1, "top-K one uses epsilon-only bid");
    c.assoc.top_k = 3;
    // A second equal-valued option eliminates track 1's bid advantage.
    u = update(predicted(3, {track(1), track(2, 1)}), {detection(1), detection(2)}, c);
    check(find(u.next, 1).age == 0, "top-K deterministic equal options");
    c.assoc.top_k = 1;
    u = update(predicted(3, {track(1), track(2, 2)}), {detection(1, 1), detection(2, -1)}, c);
    check(find(u.next, 1).age == 0 && find(u.next, 2).age == 1, "top-K one removes second-price delta");
    c.assoc.top_k = 2;
    u = update(predicted(3, {track(1), track(2, 2)}), {detection(1, 1), detection(2, -1)}, c);
    check(find(u.next, 2).age == 0 && find(u.next, 1).age == 1, "top-K second value changes winner");
    u = update(predicted(3, {track(1)}), {detection(1, 10)}, c);
    check(find(u.next, 1).age == 1, "zero IoU ineligible even with cap one");
    c.assoc.match_thresh = 1;
    u = update(predicted(3, {track(1)}), {detection(1, 10)}, c);
    check(find(u.next, 1).age == 1, "zero overlap with maximum cost cap");
}

void lifecycle_tests() {
    TrackerConfig c;
    c.life.confirm_streak = 3;
    auto t = track(1, 0, false);
    t.hit_streak = 1; t.score_sum = 0.5f;
    auto d = detection(1, 0, 0.5f);
    auto u = update(predicted(2, {t}), {d}, c);
    check(u.rows.empty() && u.next.tracks[0].hit_streak == 2 && u.next.tracks[0].lifecycle == TrackLifecycle::tentative, "streak below confirm");
    u = update(predicted(3, u.next.tracks), {d}, c);
    check(u.rows.size() == 1 && u.next.tracks[0].lifecycle == TrackLifecycle::confirmed &&
          u.next.tracks[0].score_sum == 1.5f, "streak and mean equality confirm");
    t.hit_streak = 2; t.score_sum = std::nextafter(1.0f, 0.0f);
    d.observed_score = std::nextafter(0.5f, 0.0f);
    u = update(predicted(3, {t}), {d}, c);
    check(u.rows.empty() && u.next.tracks[0].lifecycle == TrackLifecycle::tentative, "mean below confirmation");
    u = update(predicted(3, {track(1, 0, false), track(2)}), {}, c);
    check(u.next.tracks.size() == 1 && u.next.tracks[0].uid == 2 && u.next.tracks[0].hit_streak == 0 &&
          u.next.tracks[0].score_sum == 0 && u.next.tracks[0].age == 1 && u.rows.empty(), "miss removes tentative and resets confirmed streak");
    t = track(1); t.history_count = 4;
    for (int i = 0; i < 4; ++i) t.history[i] = {static_cast<float>(i), 10, 18};
    d = detection(8, 1, 0.7f, 10, 22); d.class_id = 7; d.association_score = 0.2f;
    u = update(predicted(3, {t}), {d}, c);
    const auto& a = u.next.tracks[0];
    check(a.age == 0 && a.last_observed_frame == 3 && a.observed_score == 0.7f &&
          a.association_score == 0.2f && a.class_id == 7, "match bookkeeping");
    check(a.history_count == 4 && a.history[0].cx == 1 && a.history[3].cx == 6 && a.history[3].h == 22, "history oldest-first ring");
    near(a.h_ema, 20.1, 2e-6, "height EMA");
    check(u.rows[0].flags == 1 && u.rows[0].class_id == 7 && u.rows[0].observed_score == 0.7f &&
          u.rows[0].association_score == 0.2f && u.rows[0].xyxy[0] != d.xyxy[0], "rows use updated Kalman box");
    auto birth = detection(3, 0, c.life.new_track_thresh); birth.can_birth = true;
    u = update(predicted(0), {birth}, c);
    check(u.next.tracks.size() == 1 && u.next.tracks[0].uid == 1 && u.next.tracks[0].hit_streak == 1 &&
          u.next.tracks[0].history_count == 1 && u.next.tracks[0].birth_frame == 0 && u.next.next_uid == 2 && u.rows.empty(), "birth threshold equality and state");
    birth.observed_score = std::nextafter(c.life.new_track_thresh, 0.0f);
    check(update(predicted(0), {birth}, c).next.tracks.empty(), "birth below threshold");
    c.life.confirm_streak = 1; c.life.confirm_score = c.life.new_track_thresh;
    birth.observed_score = c.life.new_track_thresh;
    check(update(predicted(0), {birth}, c).rows.size() == 1, "birth applies confirmation");
    auto b9 = detection(9, 90), b2 = detection(2, 20), b5 = detection(5, 50);
    b9.can_birth = b2.can_birth = b5.can_birth = true;
    u = update(predicted(0), {b9, b2, b5}, c);
    check(u.next.tracks[0].x[0] == 25 && u.next.tracks[1].x[0] == 55 && u.next.tracks[2].x[0] == 95 &&
          u.next.tracks[2].uid == 3 && u.next.next_uid == 4, "birth ID order and UID sequence");
}

struct BridgeFixture {
    TrackerConfig config;
    PredictedState p;
    std::vector<BatchDetection> rows;
};
BridgeFixture bridge_fixture() {
    BridgeFixture f;
    auto lost = track(1, -100), fresh = track(2);
    lost.output_id = 71;
    lost.age = 7;
    lost.last_observed_frame = 3;
    lost.history_count = 4;
    for (auto& h : lost.history) h = {5, 10, 20};
    fresh.birth_frame = 7;
    fresh.last_observed_frame = 9;
    fresh.hit_streak = 3;
    fresh.score_sum = 2.4f;
    fresh.history_count = 3;
    for (auto& h : fresh.history) h = {5, 10, 20};
    f.p = predicted(10, {lost, fresh});
    f.rows = {detection(1)};
    return f;
}
UpdateResult bridge_update(const BridgeFixture& f) { return update(f.p, f.rows, f.config); }
bool bridged(const UpdateResult& u, std::uint64_t uid = 2) {
    return u.error == Error::none && find(u.next, uid).output_id == 71;
}
void bridge_tests() {
    auto f = bridge_fixture();
    auto u = bridge_update(f);
    check(bridged(u) && u.next.tracks.size() == 1 && u.rows[0].uid == 2 && u.rows[0].output_id == 71, "bridge takes output ID and removes lost");
    auto disabled = f; disabled.config.bridge.enabled = false;
    const auto unbridged = bridge_update(disabled);
    check(u.next.tracks[0].x == find(unbridged.next, 2).x && u.next.tracks[0].P == find(unbridged.next, 2).P &&
          u.next.tracks[0].history[3].cx == find(unbridged.next, 2).history[3].cx, "bridge preserves winner state/history");
    check(!bridged(unbridged), "bridge disabled");
    for (int gate = 0; gate < 11; ++gate) {
        f = bridge_fixture();
        auto& lost = f.p.tracks[0]; auto& fresh = f.p.tracks[1];
        if (gate == 0) fresh.hit_streak = 2;
        if (gate == 1) fresh.birth_frame = 6;
        if (gate == 2) fresh.output_id = 99;
        if (gate == 3) lost.age = 1;
        if (gate == 4) lost.age = 31;
        if (gate == 5) lost.last_observed_frame = fresh.birth_frame;
        if (gate == 6) lost.history_count = 3;
        if (gate == 7) lost.lifecycle = TrackLifecycle::tentative;
        if (gate == 8) lost.history[3].cx = 100;
        if (gate == 9) lost.h_ema = 0.59f * 20;
        if (gate == 10) lost.h_ema = 1.71f * 20;
        check(!bridged(bridge_update(f)), "bridge gate " + std::to_string(gate));
    }
    f = bridge_fixture(); f.rows.clear();
    check(!bridged(bridge_update(f)), "bridge receiver must hit");
    for (const std::uint32_t age : {2u, 30u}) {
        f = bridge_fixture(); f.p.tracks[0].age = age;
        check(bridged(bridge_update(f)), "bridge age gate equality");
    }
    f = bridge_fixture(); f.rows.push_back(detection(2, -100));
    check(!bridged(bridge_update(f)), "bridge excludes lost matched this frame");
    f = bridge_fixture(); f.p.tracks[1].hit_streak = 4;
    check(!bridged(bridge_update(f)), "bridge trigger is exact equality");
    for (const float ratio : {0.6f, 1.7f}) {
        f = bridge_fixture(); f.p.tracks[0].h_ema = ratio * 20;
        check(bridged(bridge_update(f)), "height ratio gate equality");
    }
    f = bridge_fixture();
    for (auto& h : f.p.tracks[0].history) h.cx = 13;
    f.config.bridge.max_distance = 0.4f;
    check(bridged(bridge_update(f)), "bridge distance equality");
    f.config.bridge.max_distance = std::nextafter(0.4f, 0.0f);
    check(!bridged(bridge_update(f)), "bridge distance just over");
    f = bridge_fixture();
    auto other = f.p.tracks[0]; other.uid = 3; other.output_id = 73; other.x[0] = -195;
    f.p.tracks.push_back(other); f.p.next_uid = 4;
    check(!bridged(bridge_update(f)), "bridge ambiguous second candidate abandons");
    f.config.bridge.margin = 0;
    check(bridged(bridge_update(f)), "equal-distance lost UID tie with zero margin");
    f = bridge_fixture();
    other = f.p.tracks[0]; other.uid = 3; other.output_id = 73; other.x[0] = -195;
    for (auto& h : other.history) h.cx = 6;
    f.p.tracks.push_back(other); f.p.next_uid = 4;
    check(bridged(bridge_update(f)), "bridge margin equality accepts");
    f.config.bridge.margin = std::nextafter(0.05f, 1.0f);
    check(!bridged(bridge_update(f)), "bridge margin below configured boundary");
    // Two receivers claim the same lost; the loser has a valid second choice but no retry.
    f = bridge_fixture();
    auto receiver = f.p.tracks[1]; receiver.uid = receiver.output_id = 3; receiver.x[0] = 35;
    for (auto& h : receiver.history) h.cx = 5;
    other = f.p.tracks[0]; other.uid = 4; other.output_id = 74; other.x[0] = -195;
    for (auto& h : other.history) h.cx = 9;
    f.p.tracks.push_back(receiver); f.p.tracks.push_back(other); f.p.next_uid = 5;
    f.rows.push_back(detection(2, 30, 0.9f));
    u = bridge_update(f);
    check(bridged(u, 3) && find(u.next, 2).output_id == 2 && find(u.next, 4).output_id == 74, "bridge claim observed score and no retry");
    f.rows[1].observed_score = f.rows[0].observed_score;
    u = bridge_update(f);
    check(bridged(u, 2) && find(u.next, 3).output_id == 3, "bridge claim UID tie-break");
}

void capacity_protocol_tests() {
    TrackerConfig c;
    c.life.confirm_streak = 1;
    std::vector<Track> tracks;
    for (std::uint64_t uid = 1; uid <= 511; ++uid) tracks.push_back(track(uid, -100));
    auto d = detection(1); d.can_birth = true;
    auto u = update(predicted(3, tracks), {d}, c);
    check(u.error == Error::none && u.next.tracks.size() == 512 && u.next.next_uid == 513, "track capacity exactly 512");
    tracks.push_back(track(512, -100));
    u = update(predicted(3, tracks), {d}, c);
    check(u.error == Error::overflow && u.rows.empty(), "track capacity 513 overflow");
    std::vector<BatchDetection> rows;
    for (std::uint32_t id = 1; id <= 256; ++id) {
        auto birth = detection(id, static_cast<float>(id) * 20); birth.can_birth = true;
        rows.push_back(birth);
    }
    u = update(predicted(0), rows, c);
    check(u.error == Error::none && u.rows.size() == 256, "packet capacity exactly 256");
    auto last = detection(257, 6000); last.can_birth = true; rows.push_back(last);
    u = update(predicted(0), rows, c);
    check(u.error == Error::overflow && u.rows.empty(), "packet capacity above 256");
    auto p = predicted(3, {track(1)});
    DetectionBatch b; b.epoch = epoch; b.frame = 3;
    for (int mismatch = 0; mismatch < 9; ++mismatch) {
        auto badp = p; auto badb = b; auto badm = motion(3);
        if (mismatch == 0) ++badb.epoch;
        if (mismatch == 1) ++badb.frame;
        if (mismatch == 2) ++badm.epoch;
        if (mismatch == 3) ++badm.frame;
        if (mismatch == 4) ++badp.motion.epoch;
        if (mismatch == 5) ++badp.motion.frame;
        if (mismatch == 6) --badp.source_committed_frame;
        if (mismatch == 7) ++badp.epoch;
        if (mismatch == 8) ++badp.frame;
        u = associate_and_update(badp, badb, badm, c);
        check(u.error == Error::protocol && u.rows.empty(), "F07 identity " + std::to_string(mismatch));
    }
    p.error = Error::numeric;
    check(associate_and_update(p, b, p.motion, c).error == Error::numeric, "predict error blocks association");
    p.error = Error::none; b.error = Error::producer;
    check(associate_and_update(p, b, p.motion, c).error == Error::producer, "batch error blocks association");
    p = predicted(0); p.source_committed_frame = 0; b = {}; b.epoch = epoch; b.frame = 0;
    check(associate_and_update(p, b, p.motion, c).error == Error::protocol, "first frame source must be -1");
    for (int kind = 0; kind < 6; ++kind) {
        p = predicted(3, {track(1)}); d = detection(1);
        if (kind == 0) p.tracks[0].x[0] = std::numeric_limits<float>::quiet_NaN();
        if (kind == 1) p.tracks[0].P[1] = std::numeric_limits<float>::infinity();
        if (kind == 2) d.observed_score = std::numeric_limits<float>::infinity();
        if (kind == 3) d.association_score = std::numeric_limits<float>::quiet_NaN();
        if (kind == 4) d.xyxy[0] = std::numeric_limits<float>::infinity();
        if (kind == 5) p.tracks[0].score_sum = std::numeric_limits<float>::infinity();
        u = update(p, {d}, c);
        check(u.error == Error::numeric && u.rows.empty(), "nonfinite update input");
    }
    // Fail after a successful commit, then prove that even malformed later input is skipped.
    TrackerReference session(epoch, c);
    auto packet = session.step(candidates(0, {candidate(1)}), motion(0));
    check(packet.status == ResultStatus::ok && packet.committed_frame == 0 && packet.tracks.size() == 1, "first packet OK");
    encode_packet(packet);
    const auto frozen = session.committed();
    auto bad = candidates(1); bad.header.flags = kPostprocessNumeric;
    packet = session.step(bad, motion(1));
    check(packet.status == ResultStatus::failed && packet.fail_frame == 1 && packet.frame_error == Error::producer &&
          packet.fail_code == Error::producer && packet.committed_frame == 0 && packet.tracks.empty(), "FAILED packet");
    encode_packet(packet);
    packet = session.step({}, {});
    check(packet.status == ResultStatus::skipped_after_failure && packet.frame == 2 && packet.fail_frame == 1 &&
          packet.fail_code == Error::producer && packet.frame_error == Error::none && packet.committed_frame == 0 &&
          packet.tracks.empty(), "SKIPPED packet");
    encode_packet(packet);
    check(session.failed() && session.committed().committed_frame == frozen.committed_frame &&
          session.committed().next_uid == frozen.next_uid && session.committed().tracks[0].x == frozen.tracks[0].x &&
          session.committed().tracks[0].P == frozen.tracks[0].P && session.committed().tracks[0].age == frozen.tracks[0].age,
          "failed session frozen state");
    TrackerReference numeric(epoch, c);
    auto nan = candidates(0, {candidate(1)}); nan.rows[0].score = std::numeric_limits<float>::quiet_NaN();
    packet = numeric.step(nan, motion(0));
    check(packet.status == ResultStatus::failed && packet.fail_code == Error::numeric && packet.committed_frame == -1, "first-frame NUMERIC packet");
    encode_packet(packet);
    TrackerReference overflow(epoch, c);
    std::vector<DetectionCandidate> many;
    for (int i = 0; i < 257; ++i) many.push_back(candidate(static_cast<std::uint32_t>(i), static_cast<float>(i * 20)));
    packet = overflow.step(candidates(0, many), motion(0));
    check(packet.status == ResultStatus::failed && packet.fail_code == Error::overflow && overflow.committed().tracks.empty(), "packet overflow freezes session");
    encode_packet(packet);
    TrackerReference empty(epoch);
    packet = empty.step(candidates(0), motion(0));
    check(packet.status == ResultStatus::ok && packet.tracks.empty() && empty.committed().committed_frame == 0, "empty frame commits");
    encode_packet(packet);
}

std::vector<float*> float_fields(TrackerConfig& c) {
    return {&c.kalman.r_scale, &c.kalman.std_position, &c.kalman.std_velocity, &c.batch.track_thresh,
        &c.continuation.min_score, &c.continuation.prior_iou, &c.continuation.dedup_iou,
        &c.assoc.high_thresh, &c.assoc.mid_thresh, &c.assoc.match_thresh, &c.assoc.stage2_match_thresh,
        &c.assoc.lambda, &c.assoc.stability_cost_w, &c.assoc.auction_eps, &c.assoc.stability_bid_w,
        &c.life.new_track_thresh, &c.life.confirm_score, &c.bridge.max_distance, &c.bridge.height_ratio_lo,
        &c.bridge.height_ratio_hi, &c.bridge.margin, &c.bridge.speed_norm};
}
void rejected(const TrackerConfig& c, const std::string& message) {
    bool did_reject = false;
    try { validate_tracker_config(c); } catch (const std::invalid_argument&) { did_reject = true; }
    check(did_reject, message);
}
void config_tests() {
    validate_tracker_config({}); check(true, "default config accepted");
    TrackerConfig defaults;
    for (std::size_t i = 0; i < float_fields(defaults).size(); ++i) {
        for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
            TrackerConfig c; *float_fields(c)[i] = value;
            rejected(c, "nonfinite config field " + std::to_string(i));
        }
    }
    for (std::size_t index : {3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 15u, 16u})
        for (float value : {-0.01f, 1.01f}) {
            TrackerConfig c; *float_fields(c)[index] = value;
            rejected(c, "threshold outside [0,1]");
        }
    const std::vector<std::function<void(TrackerConfig&)>> invalid = {
        [](auto& c) { c.batch.track_thresh = 0.2f; },
        [](auto& c) { c.assoc.mid_thresh = 0.6f; },
        [](auto& c) { c.assoc.lambda = 0; },
        [](auto& c) { c.kalman.r_scale = 0; },
        [](auto& c) { c.kalman.std_position = 0; },
        [](auto& c) { c.kalman.std_velocity = 0; },
        [](auto& c) { c.assoc.auction_eps = 0; },
        [](auto& c) { c.bridge.speed_norm = 0; },
        [](auto& c) { c.assoc.stability_cost_w = -1; },
        [](auto& c) { c.assoc.stability_bid_w = -1; },
        [](auto& c) { c.bridge.margin = -1; },
        [](auto& c) { c.bridge.max_distance = -1; },
        [](auto& c) { c.assoc.top_k = 0; },
        [](auto& c) { c.assoc.top_k = 9; },
        [](auto& c) { c.life.confirm_streak = 0; },
        [](auto& c) { c.life.track_buffer = 0; },
        [](auto& c) { c.bridge.trigger_streak = 3; },
        [](auto& c) { c.bridge.min_lost = 0; },
        [](auto& c) { c.bridge.height_ratio_lo = 0; },
        [](auto& c) { c.bridge.height_ratio_hi = c.bridge.height_ratio_lo; },
        [](auto& c) { c.bridge.height_ratio_hi = 0.5f; },
        [](auto& c) { c.continuation.max_candidates = 301; }};
    for (const auto& change : invalid) { TrackerConfig c; change(c); rejected(c, "config rule rejection"); }
    TrackerConfig edge;
    edge.assoc.top_k = 8; edge.continuation.max_candidates = 300;
    edge.batch.track_thresh = edge.assoc.mid_thresh = edge.assoc.high_thresh = 0;
    edge.assoc.stability_cost_w = edge.assoc.stability_bid_w = edge.bridge.margin = edge.bridge.max_distance = 0;
    edge.life.confirm_streak = edge.life.track_buffer = edge.bridge.min_lost = 1;
    validate_tracker_config(edge); check(true, "config inclusive boundaries accepted");
    const std::string expected =
        "kalman.r_scale=0x1.cp+1\nkalman.std_position=0x1.99999ap-5\nkalman.std_velocity=0x1.99999ap-8\n"
        "batch.track_thresh=0x1.99999ap-5\ncontinuation.enabled=1\ncontinuation.min_score=0x1.99999ap-4\n"
        "continuation.prior_iou=0x1.333334p-2\ncontinuation.dedup_iou=0x1.666666p-1\ncontinuation.max_candidates=50\n"
        "assoc.high_thresh=0x1.ccccccp-2\nassoc.mid_thresh=0x1.99999ap-4\nassoc.match_thresh=0x1p-1\n"
        "assoc.stage2_match_thresh=0x1p-1\nassoc.lambda=0x1.4p+3\nassoc.stability_cost_w=0x1.99999ap-3\n"
        "assoc.top_k=3\nassoc.auction_eps=0x1.47ae14p-7\nassoc.stability_bid_w=0x1.99999ap-4\n"
        "life.new_track_thresh=0x1.1eb852p-2\nlife.confirm_streak=3\nlife.confirm_score=0x1p-1\nlife.track_buffer=30\n"
        "bridge.enabled=1\nbridge.trigger_streak=4\nbridge.min_lost=2\nbridge.max_distance=0x1.99999ap-2\n"
        "bridge.height_ratio_lo=0x1.333334p-1\nbridge.height_ratio_hi=0x1.b33334p+0\nbridge.margin=0x1.99999ap-5\n"
        "bridge.speed_norm=0x1.eb851ep-4\n";
    check(canonical_tracker_config({}) == expected, "canonical config exact text");
    check(sha256_hex(canonical_tracker_config({})) == sha256_hex(expected), "stable config hash");
    struct Comma : std::numpunct<char> { char do_decimal_point() const override { return ','; } };
    const auto previous = std::locale();
    std::locale::global(std::locale(previous, new Comma));
    const auto localized = canonical_tracker_config({});
    std::locale::global(previous);
    check(localized == expected, "canonical floats independent of locale");
}

void sequence_tests() {
    // Constant camera shift of +100 at frame 4 separates the predicted old track
    // from its re-entry at frame 6. Four stationary observations allow a bridge.
    std::vector<std::vector<DetectionCandidate>> frames;
    for (int k = 0; k < 10; ++k) {
        std::vector<DetectionCandidate> rows;
        if (k <= 3 || k >= 6) rows.push_back(candidate(5, 0));
        const float camera = k >= 4 ? 100.0f : 0.0f;
        rows.push_back(candidate(2, 200 + camera + static_cast<float>(k)));
        rows.push_back(candidate(9, 400 + camera - static_cast<float>(k)));
        frames.push_back(rows);
    }
    const auto run = [&](bool reverse) {
        TrackerReference tracker(epoch);
        std::vector<PacketBytes> stream;
        // Independent binary64 1D Kalman recurrence for the two moving boxes.
        struct Axis {
            double position, velocity = 0, pp = 4, pv = 0, vp = 0, vv = 1.5625;
            void observe(double measurement, double camera) {
                position += camera + velocity;
                const double a = pp + pv + vp + vv + 1;
                const double b = pv + vv, c = vp + vv, d = vv + 0.015625;
                const double kp = a / (a + 3.5), kv = c / (a + 3.5);
                const double innovation = measurement - position;
                position += kp * innovation;
                velocity += kv * innovation;
                pp = (1 - kp) * a; pv = (1 - kp) * b;
                vp = c - kv * a; vv = d - kv * b;
            }
        } forward{205}, backward{405};
        for (int k = 0; k < 10; ++k) {
            auto rows = frames[k]; if (reverse) std::reverse(rows.begin(), rows.end());
            const auto m = motion(k, k == 4 ? MotionStatus::valid : MotionStatus::uninitialized, k == 4 ? 100.0f : 0.0f);
            const auto packet = tracker.step(candidates(k, rows), m);
            if (k > 0) {
                const double camera = k >= 4 ? 100 : 0;
                forward.observe(205 + camera + k, k == 4 ? 100 : 0);
                backward.observe(405 + camera - k, k == 4 ? 100 : 0);
            }
            check(packet.status == ResultStatus::ok && packet.frame == k && packet.committed_frame == k &&
                  packet.fail_frame == -1 && packet.fail_code == Error::none, "synthetic packet headers");
            const std::size_t count = k < 2 ? 0 : k <= 3 || k >= 8 ? 3 : 2;
            check(packet.tracks.size() == count, "synthetic packet count frame " + std::to_string(k));
            if (k >= 2) {
                std::vector<std::uint64_t> expected_uids = k <= 3 ? std::vector<std::uint64_t>{1, 2, 3} :
                    k >= 8 ? std::vector<std::uint64_t>{1, 3, 4} : std::vector<std::uint64_t>{1, 3};
                for (std::size_t i = 0; i < packet.tracks.size(); ++i) {
                    const auto& row = packet.tracks[i];
                    const auto expected_uid = expected_uids[i];
                    check(row.uid == expected_uid && row.output_id == (k == 9 && expected_uid == 4 ? 2 : expected_uid),
                          "synthetic full identity stream");
                    const double cx = expected_uid == 1 ? forward.position : expected_uid == 3 ? backward.position : 5;
                    near(row.xyxy[0], cx - 5, 0.0001, "synthetic independent left");
                    near(row.xyxy[1], 0, 0.0001, "synthetic independent top");
                    near(row.xyxy[2], cx + 5, 0.0001, "synthetic independent right");
                    near(row.xyxy[3], 20, 0.0001, "synthetic independent bottom");
                    check(row.flags == 1 && row.class_id == 0 && row.observed_score == 0.8f &&
                          row.association_score == 0.8f && row.xyxy[2] > row.xyxy[0], "synthetic full row fields");
                }
            }
            if (k == 8) check(packet.tracks.back().output_id == 4, "re-entry confirmed before bridge");
            if (k == 9) check(packet.tracks.back().uid == 4 && packet.tracks.back().output_id == 2, "re-entry bridge retains UID");
            stream.push_back(encode_packet(packet));
            check(encode_packet(decode_packet(stream.back())) == stream.back(), "full packet wire round trip");
        }
        check(tracker.committed().tracks.size() == 3 && tracker.committed().next_uid == 5, "synthetic bridge removes lost identity");
        return stream;
    };
    const auto first = run(false), second = run(false), permuted = run(true);
    check(first == second && first == permuted, "full sequence byte-identical and candidate permutation invariant");
}
} // namespace

int main() {
    try {
        math_tests();
        kalman_tests();
        predict_tests();
        batch_tests();
        association_tests();
        lifecycle_tests();
        bridge_tests();
        capacity_protocol_tests();
        config_tests();
        sequence_tests();
        std::cout << "tracking_reference_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "tracking_reference_tests: " << e.what() << '\n';
        return 1;
    }
}

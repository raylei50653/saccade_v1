#include "saccade/occlusion_experiment.hpp"
#include "saccade/tracking_math.hpp"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace saccade;
namespace math = tracking_math;
int checks = 0;
void check(bool value, const std::string& message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void near(float actual, float expected, const std::string& message) {
    check(std::fabs(actual - expected) < 1e-6f, message);
}
Track track(std::uint64_t uid, float left = 0, float top = 0) {
    Track t;
    t.uid = t.output_id = uid;
    t.lifecycle = TrackLifecycle::confirmed;
    const float box[4] = {left, top, left + 100, top + 100};
    math::kalman_init(box, t.x.data(), t.P.data(), 0.05f, 0.00625f);
    t.age = 1;
    return t;
}
PredictedState prediction(FrameId frame, std::vector<Track> tracks, Epoch epoch = 1) {
    PredictedState p;
    p.epoch = p.motion.epoch = epoch;
    p.frame = p.motion.frame = frame;
    p.source_committed_frame = frame - 1;
    p.tracks = std::move(tracks);
    for (const auto& t : p.tracks) if (t.uid >= p.next_uid) p.next_uid = t.uid + 1;
    return p;
}
BatchDetection detection(float top = 0) {
    return {{0, top, 100, top + 100}, .8f, .8f, 0, 1, true, false};
}
OcclusionExperimentConfig enabled(bool oao, bool front) {
    OcclusionExperimentConfig c;
    c.oao_enabled = oao;
    c.front_enabled = front;
    return c;
}

void duration_and_identity_tests() {
    OcclusionExperiment experiment({}, enabled(true, false));
    auto first = track(1), second = track(2, 20);
    first.lifecycle = TrackLifecycle::tentative;
    second.age = 15; // lost tracks remain in the overlap pool
    for (FrameId frame = 0; frame < 30; ++frame) {
        auto p = prediction(frame, {first, second});
        check(experiment.prepare(p) == Error::none, "consecutive prepare");
        const auto* state = experiment.state(1);
        check(state && state->overlap_duration == frame + 1, "tentative duration advances against lost partner");
        near(state->max_overlap, 2.0f / 3.0f, "maximum overlap");
        near(state->oao_coefficient, (2.0f / 3.0f) * math::min(1, (frame + 1) / 25.0f), "ramp and saturation");
    }
    check(experiment.prepare(prediction(29, {first, second})) == Error::none &&
          experiment.state(1)->overlap_duration == 30 && experiment.diagnostics().frames == 30,
          "duplicate prepare does not age state or counters");
    second.x[0] += 300;
    check(experiment.prepare(prediction(30, {first, second})) == Error::none &&
          experiment.state(1)->overlap_duration == 0 && experiment.state(1)->oao_coefficient == 0,
          "duration resets immediately without overlap");
    second.x[0] -= 300;
    check(experiment.prepare(prediction(31, {first, second})) == Error::none &&
          experiment.state(1)->overlap_duration == 1, "new overlap starts at one");
    auto replacement = track(3);
    replacement.output_id = first.output_id; // bridge identity must not transfer temporal state
    check(experiment.prepare(prediction(32, {second, replacement})) == Error::none &&
          !experiment.state(1) && experiment.state(3)->overlap_duration == 1,
          "removed UID erased and bridge output ID cannot inherit state");
    check(experiment.prepare(prediction(34, {second, replacement})) == Error::protocol &&
          experiment.state(3)->overlap_duration == 1 && experiment.diagnostics().frames == 33,
          "skipped frame rejected without mutation");
    check(experiment.prepare(prediction(0, {first, second}, 2)) == Error::none &&
          experiment.state(1)->overlap_duration == 1 && !experiment.state(3), "epoch clears UID state");
}

void front_tests() {
    OcclusionExperiment experiment({}, enabled(false, true));
    const auto back = track(1), front = track(2, 0, 10), beyond = track(3, 0, 20);
    check(experiment.prepare(prediction(0, {back, front, beyond})) == Error::none, "front prepare");
    check(experiment.state(2)->partner_uid == 1 && experiment.state(2)->front_ttl == 4,
          "max overlap tie chooses lowest UID and only lower-foot member latches");
    check(experiment.state(1)->front_ttl == 0, "rear track cannot become front");

    OcclusionExperiment reverse({}, enabled(false, true));
    check(reverse.prepare(prediction(0, {track(1, 0, 20), front, track(3)})) == Error::none &&
          reverse.state(2)->partner_uid == 1 && reverse.state(2)->front_ttl == 0,
          "tie partner controls front decision deterministically");

    OcclusionExperiment far_depth({}, enabled(false, true));
    check(far_depth.prepare(prediction(0, {back, track(2, 0, 20)})) == Error::none &&
          far_depth.state(2)->front_ttl == 0, "different depth exceeds foot-gap band");
    OcclusionExperiment weak({}, enabled(false, true));
    check(weak.prepare(prediction(0, {back, track(2, 70, 10)})) == Error::none &&
          weak.state(2)->front_ttl == 0, "weak IoU cannot latch");

    auto separated = front;
    separated.x[0] += 400;
    for (FrameId frame = 1; frame <= 4; ++frame) {
        check(experiment.prepare(prediction(frame, {back, separated})) == Error::none,
              "separation prepare");
        check(experiment.state(2)->front_ttl == 4 - frame, "TTL includes latch frame and expires");
    }
    check(experiment.diagnostics().front_latched_track_frames == 5,
          "front diagnostic counts each latched track-frame, including other initial front");
}

void cost_and_hook_tests() {
    TrackerConfig tracker;
    OcclusionExperiment experiment(tracker, enabled(true, true));
    auto back = track(1), front = track(2, 0, 10);
    auto p = prediction(0, {back, front});
    check(experiment.prepare(p) == Error::none, "cost prepare");
    const auto d = detection();
    const float overlap = 90.0f / 110.0f;
    const float oao = .5f * overlap / 25;
    const float front_penalty = .5f * .1f;
    const float reward = .2f / 10;
    const float expected = 1 - overlap * math::det_expf(-(oao + front_penalty - reward));
    near(experiment.cost(front, d, overlap), expected, "penalties enter shared multiplicative exponent");
    const auto same_front = detection(10);
    const float only_oao = 1 - math::det_expf(-(oao - reward));
    near(experiment.cost(front, same_front, 1), math::max(0, only_oao), "front penalty is directional");
    check(experiment.diagnostics().cost_calls == 2 && experiment.diagnostics().oao_positive_calls == 2 &&
          experiment.diagnostics().front_positive_calls == 1, "penalty diagnostics count actual positive terms");

    OcclusionExperiment disabled(tracker);
    check(disabled.prepare(p) == Error::none, "disabled prepare");
    for (float iou : {0.0f, .01f, .5f, .99f, 1.0f}) {
        for (float top : {-10.0f, 0.0f, 10.0f}) {
            const auto candidate = detection(top);
            const float stock = math::association_cost(iou, front.x[3], 100, tracker.assoc.lambda,
                                                       tracker.assoc.stability_cost_w);
            check(math::float_bits(disabled.cost(front, candidate, iou)) == math::float_bits(stock),
                  "disabled cost preserves stock bits");
        }
    }
    OcclusionExperiment isolated(tracker, enabled(true, true));
    check(isolated.prepare(prediction(0, {front})) == Error::none, "isolated prepare");
    check(math::float_bits(isolated.cost(front, d, overlap)) ==
          math::float_bits(math::association_cost(overlap, 100, 100, 10, .2f)),
          "enabled zero-penalty cost preserves stock bits");

    DetectionBatch batch;
    batch.epoch = p.epoch;
    batch.frame = p.frame;
    batch.rows = {same_front};
    int calls = 0;
    const auto blocked = associate_and_update(p, batch, p.motion, tracker,
        [&](const Track&, const BatchDetection&, float) { ++calls; return 1.0f; });
    check(calls > 0 && blocked.error == Error::none && blocked.rows.empty(),
          "association hook changes the actual matching decision");
    const auto numeric = associate_and_update(p, batch, p.motion, tracker,
        [](const Track&, const BatchDetection&, float) { return math::bits_float(0x7fc00000u); });
    check(numeric.error == Error::numeric, "association rejects nonfinite hook cost");
}

void validation_and_numeric_tests() {
    validate_occlusion_experiment_config({});
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<std::function<void(OcclusionExperimentConfig&)>> invalid = {
        [nan](auto& c) { c.oao_tau = nan; }, [](auto& c) { c.oao_tau = -1; },
        [](auto& c) { c.oao_ramp_frames = 0; }, [](auto& c) { c.front_iou_thresh = 1.01f; },
        [](auto& c) { c.front_iou_thresh = -1; }, [inf](auto& c) { c.front_iou_thresh = inf; },
        [nan](auto& c) { c.front_foot_gap = nan; }, [](auto& c) { c.front_foot_gap = -1; },
        [](auto& c) { c.front_ttl = 0; }, [](auto& c) { c.front_cost_weight = -1; },
        [inf](auto& c) { c.front_cost_weight = inf; }
    };
    for (const auto& change : invalid) {
        OcclusionExperimentConfig c;
        change(c);
        bool rejected = false;
        try { OcclusionExperiment invalid_experiment({}, c); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "invalid experiment configuration rejected");
    }
    OcclusionExperiment experiment({}, enabled(true, true));
    auto bad = track(1);
    bad.x[0] = nan;
    check(experiment.prepare(prediction(0, {bad})) == Error::numeric && !experiment.state(1) &&
          experiment.diagnostics().frames == 0, "numeric prepare leaves state untouched");
    check(experiment.prepare(prediction(0, {track(2), track(1)})) == Error::protocol,
          "unsorted UID input rejected");
    check(experiment.prepare(prediction(0, {track(1), track(1)})) == Error::protocol,
          "duplicate UID input rejected");
    check(!math::finite(experiment.cost(track(1), detection(), .5f)),
          "unprepared enabled track cost signals numeric failure");
    check(!math::finite(experiment.cost(track(1), detection(), nan)),
          "nonfinite overlap signals numeric failure");
}
} // namespace

int main() {
    try {
        duration_and_identity_tests();
        front_tests();
        cost_and_hook_tests();
        validation_and_numeric_tests();
        std::cout << "occlusion experiment checks: " << checks << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "occlusion experiment test failed: " << e.what() << '\n';
        return 1;
    }
}

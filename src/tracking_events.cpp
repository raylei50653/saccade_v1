#include "saccade/tracking_events.hpp"
#include "saccade/sha256.hpp"
#include "saccade/tracking_math.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace saccade {
namespace {
using Bytes = std::vector<std::uint8_t>;
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void u32(Bytes& out, std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void u64(Bytes& out, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void f32(Bytes& out, float value) {
    require(std::isfinite(value), "nonfinite event payload");
    u32(out, std::bit_cast<std::uint32_t>(value));
}
struct Reader {
    std::span<const std::uint8_t> bytes;
    std::size_t at = 0;
    std::span<const std::uint8_t> take(std::size_t n) {
        require(n <= bytes.size() - at, "truncated tracking events");
        auto result = bytes.subspan(at, n); at += n; return result;
    }
    std::uint32_t u32() {
        const auto b = take(4); std::uint32_t v = 0;
        for (unsigned i = 0; i < 4; ++i) v |= static_cast<std::uint32_t>(b[i]) << (8 * i);
        return v;
    }
    std::uint64_t u64() {
        const auto b = take(8); std::uint64_t v = 0;
        for (unsigned i = 0; i < 8; ++i) v |= static_cast<std::uint64_t>(b[i]) << (8 * i);
        return v;
    }
    float f32() { const float v = std::bit_cast<float>(u32()); require(std::isfinite(v), "nonfinite event payload"); return v; }
    bool boolean() { const auto v = u32(); require(v <= 1, "noncanonical event bool"); return v != 0; }
};
void encode_track(Bytes& out, const Track& t) {
    require(t.uid != 0 && t.output_id != 0, "zero event track identity");
    require(t.lifecycle == TrackLifecycle::tentative || t.lifecycle == TrackLifecycle::confirmed, "invalid event lifecycle");
    require(t.history_count <= kBridgeHistory, "invalid event history count");
    u64(out, t.uid); u64(out, t.output_id); u32(out, static_cast<std::uint32_t>(t.lifecycle));
    for (float v : t.x) f32(out, v);
    for (float v : t.P) f32(out, v);
    u32(out, t.age); u32(out, t.hit_streak); f32(out, t.score_sum); u32(out, t.class_id);
    f32(out, t.observed_score); f32(out, t.association_score);
    u64(out, static_cast<std::uint64_t>(t.birth_frame)); u64(out, static_cast<std::uint64_t>(t.last_observed_frame));
    u32(out, t.history_count);
    for (std::uint32_t i = 0; i < t.history_count; ++i) {
        f32(out, t.history[i].cx); f32(out, t.history[i].cy); f32(out, t.history[i].h);
    }
    f32(out, t.h_ema);
}
Track decode_track(Reader& in) {
    Track t;
    t.uid = in.u64(); t.output_id = in.u64(); t.lifecycle = static_cast<TrackLifecycle>(in.u32());
    for (auto& v : t.x) v = in.f32();
    for (auto& v : t.P) v = in.f32();
    t.age = in.u32(); t.hit_streak = in.u32(); t.score_sum = in.f32(); t.class_id = in.u32();
    t.observed_score = in.f32(); t.association_score = in.f32();
    t.birth_frame = std::bit_cast<FrameId>(in.u64()); t.last_observed_frame = std::bit_cast<FrameId>(in.u64());
    t.history_count = in.u32(); require(t.history_count <= kBridgeHistory, "invalid event history count");
    for (std::uint32_t i = 0; i < t.history_count; ++i) t.history[i] = {in.f32(), in.f32(), in.f32()};
    t.h_ema = in.f32();
    Bytes check; encode_track(check, t);
    return t;
}
void encode_detection(Bytes& out, const BatchDetection& d) {
    for (float v : d.xyxy) f32(out, v);
    f32(out, d.observed_score); f32(out, d.association_score);
    u32(out, d.class_id); u32(out, d.candidate_id); u32(out, d.can_continue); u32(out, d.can_birth);
}
BatchDetection decode_detection(Reader& in) {
    BatchDetection d;
    for (auto& v : d.xyxy) v = in.f32();
    d.observed_score = in.f32(); d.association_score = in.f32(); d.class_id = in.u32(); d.candidate_id = in.u32();
    d.can_continue = in.boolean(); d.can_birth = in.boolean(); return d;
}
void encode_row(Bytes& out, const TrackRow& r) {
    u64(out, r.uid); u64(out, r.output_id); u32(out, r.class_id); u32(out, r.flags);
    for (float v : r.xyxy) f32(out, v);
    f32(out, r.observed_score); f32(out, r.association_score);
}
TrackRow decode_row(Reader& in) {
    TrackRow r;
    r.uid = in.u64(); r.output_id = in.u64(); r.class_id = in.u32(); r.flags = in.u32();
    for (auto& v : r.xyxy) v = in.f32();
    r.observed_score = in.f32(); r.association_score = in.f32(); return r;
}
void check_common(const TrackingEvent& e) {
    require(e.frame >= 0, "negative event frame");
    require(static_cast<std::uint32_t>(e.kind) >= 1 && static_cast<std::uint32_t>(e.kind) <= 14, "unknown event kind");
    require(static_cast<std::uint32_t>(e.phase) <= 7, "unknown event phase");
    require(static_cast<std::uint32_t>(e.stage) <= 4, "unknown event stage");
}
Bytes encode_event(const TrackingEvent& e) {
    check_common(e); Bytes out;
    u64(out, e.source); u64(out, e.epoch); u64(out, static_cast<std::uint64_t>(e.frame));
    u32(out, static_cast<std::uint32_t>(e.kind)); u32(out, static_cast<std::uint32_t>(e.phase)); u32(out, static_cast<std::uint32_t>(e.stage));
    switch (e.kind) {
    case TrackingEventKind::frame_begin: u64(out, static_cast<std::uint64_t>(e.committed_frame)); break;
    case TrackingEventKind::batch_ready: u32(out, e.count); break;
    case TrackingEventKind::candidate: encode_detection(out, e.detection); break;
    case TrackingEventKind::predict:
    case TrackingEventKind::confirm:
    case TrackingEventKind::lost: encode_track(out, e.before); encode_track(out, e.after); break;
    case TrackingEventKind::associate: encode_track(out, e.before); encode_track(out, e.after); encode_detection(out, e.detection); u32(out, e.association_stage); break;
    case TrackingEventKind::birth: encode_track(out, e.after); encode_detection(out, e.detection); break;
    case TrackingEventKind::remove:
        require(e.reason == TrackingEventReason::unmatched || e.reason == TrackingEventReason::expired || e.reason == TrackingEventReason::invalid_geometry, "invalid removal reason");
        encode_track(out, e.before); u32(out, static_cast<std::uint32_t>(e.reason)); break;
    case TrackingEventKind::relink: encode_track(out, e.before); encode_track(out, e.after); u64(out, e.related_uid); break;
    case TrackingEventKind::emit: encode_row(out, e.row); break;
    case TrackingEventKind::frame_commit: u64(out, static_cast<std::uint64_t>(e.committed_frame)); u32(out, e.count); break;
    case TrackingEventKind::frame_fail:
    case TrackingEventKind::frame_skipped:
        require(static_cast<std::uint32_t>(e.error) > 0 && static_cast<std::uint32_t>(e.error) <= 31, "invalid event error");
        u64(out, static_cast<std::uint64_t>(e.committed_frame)); u32(out, static_cast<std::uint32_t>(e.error)); break;
    }
    return out;
}
TrackingEvent decode_event(Reader& in) {
    TrackingEvent e;
    e.source = in.u64(); e.epoch = in.u64(); e.frame = std::bit_cast<FrameId>(in.u64());
    e.kind = static_cast<TrackingEventKind>(in.u32()); e.phase = static_cast<TrackingEventPhase>(in.u32()); e.stage = static_cast<TrackingEventStage>(in.u32());
    check_common(e);
    switch (e.kind) {
    case TrackingEventKind::frame_begin: e.committed_frame = std::bit_cast<FrameId>(in.u64()); break;
    case TrackingEventKind::batch_ready: e.count = in.u32(); break;
    case TrackingEventKind::candidate: e.detection = decode_detection(in); break;
    case TrackingEventKind::predict:
    case TrackingEventKind::confirm:
    case TrackingEventKind::lost: e.before = decode_track(in); e.after = decode_track(in); break;
    case TrackingEventKind::associate: e.before = decode_track(in); e.after = decode_track(in); e.detection = decode_detection(in); e.association_stage = in.u32(); break;
    case TrackingEventKind::birth: e.after = decode_track(in); e.detection = decode_detection(in); break;
    case TrackingEventKind::remove: e.before = decode_track(in); e.reason = static_cast<TrackingEventReason>(in.u32()); break;
    case TrackingEventKind::relink: e.before = decode_track(in); e.after = decode_track(in); e.related_uid = in.u64(); break;
    case TrackingEventKind::emit: e.row = decode_row(in); break;
    case TrackingEventKind::frame_commit: e.committed_frame = std::bit_cast<FrameId>(in.u64()); e.count = in.u32(); break;
    case TrackingEventKind::frame_fail:
    case TrackingEventKind::frame_skipped: e.committed_frame = std::bit_cast<FrameId>(in.u64()); e.error = static_cast<Error>(in.u32()); break;
    }
    return e;
}
bool same_track(const Track& a, const Track& b) { Bytes x, y; encode_track(x, a); encode_track(y, b); return x == y; }
bool same_detection(const BatchDetection& a, const BatchDetection& b) { Bytes x, y; encode_detection(x, a); encode_detection(y, b); return x == y; }
bool same_row(const TrackRow& a, const TrackRow& b) { Bytes x, y; encode_row(x, a); encode_row(y, b); return x == y; }
bool confirmable(const Track& t, const TrackerConfig& c) {
    return t.lifecycle == TrackLifecycle::tentative && t.hit_streak >= c.life.confirm_streak &&
        t.score_sum / static_cast<float>(t.hit_streak) >= c.life.confirm_score;
}
bool bad_geometry(const Track& t) { return t.x[3] < 1 || t.x[2] <= 0; }
void push_history(Track& t, const BatchDetection& d) {
    if (t.history_count == kBridgeHistory) {
        for (std::uint32_t i = 1; i < kBridgeHistory; ++i) t.history[i-1] = t.history[i];
        --t.history_count;
    }
    float z[4]; tracking_math::box_to_state(d.xyxy.data(), z);
    t.history[t.history_count++] = {z[0], z[1], z[3]};
}
struct Session {
    std::map<std::uint64_t, Track> committed, working;
    std::uint64_t next_uid = 1, working_next_uid = 1;
    FrameId next_frame = 0, frame = no_frame, committed_frame = no_frame;
    Error failure = Error::none;
    bool active = false, batch_ready = false;
    std::uint32_t batch_count = 0;
    TrackingEventPhase phase = TrackingEventPhase::frame;
    std::set<std::uint64_t> original, predicted, updated, matched, emitted;
    std::map<std::uint32_t, BatchDetection> candidates;
    std::set<std::uint32_t> used;
    std::uint64_t pending_confirm = 0, last_emit_uid = 0, last_predict_uid = 0, last_update_uid = 0, last_donor_uid = 0;
};
void validate_stage(const TrackingEvent& e) {
    const auto is = [&](TrackingEventPhase p, TrackingEventStage s) { return e.phase == p && e.stage == s; };
    switch (e.kind) {
    case TrackingEventKind::frame_begin: require(is(TrackingEventPhase::frame, TrackingEventStage::control), "wrong frame_begin phase/stage"); break;
    case TrackingEventKind::predict: require(is(TrackingEventPhase::predict, TrackingEventStage::f08a), "wrong predict phase/stage"); break;
    case TrackingEventKind::batch_ready:
    case TrackingEventKind::candidate: require(is(TrackingEventPhase::batch, TrackingEventStage::f04b), "wrong batch phase/stage"); break;
    case TrackingEventKind::associate:
    case TrackingEventKind::lost: require(is(TrackingEventPhase::update, TrackingEventStage::f08b), "wrong update phase/stage"); break;
    case TrackingEventKind::birth: require(is(TrackingEventPhase::birth, TrackingEventStage::f08b), "wrong birth phase/stage"); break;
    case TrackingEventKind::confirm: require((e.phase == TrackingEventPhase::update || e.phase == TrackingEventPhase::birth) && e.stage == TrackingEventStage::f08b, "wrong confirm phase/stage"); break;
    case TrackingEventKind::remove: require(is(TrackingEventPhase::predict, TrackingEventStage::f08a) || is(TrackingEventPhase::update, TrackingEventStage::f08b), "wrong remove phase/stage"); break;
    case TrackingEventKind::relink: require(is(TrackingEventPhase::bridge, TrackingEventStage::f08b), "wrong relink phase/stage"); break;
    case TrackingEventKind::emit: require(is(TrackingEventPhase::emit, TrackingEventStage::f09), "wrong emit phase/stage"); break;
    case TrackingEventKind::frame_commit:
    case TrackingEventKind::frame_fail:
    case TrackingEventKind::frame_skipped: require(is(TrackingEventPhase::commit, TrackingEventStage::control), "wrong terminal phase/stage"); break;
    }
}
void before_matches(const Session& s, const Track& before) {
    const auto it = s.working.find(before.uid);
    require(it != s.working.end() && same_track(it->second, before), "before state differs from current track");
}
void ready_for_phase(const Session& s, TrackingEventPhase phase) {
    if (phase > TrackingEventPhase::predict) {
        require(s.predicted == s.original, "missing or duplicate required predict");
        for (const auto& [uid, t] : s.working) {
            (void)uid;
            require(!bad_geometry(t), "missing geometry removal");
        }
    }
    if (phase > TrackingEventPhase::batch) {
        require(s.batch_ready && s.candidates.size() == s.batch_count, "missing batch/candidate events");
    }
}
void validate_event(Session& s, const TrackingEvent& e, const TrackerConfig& c) {
    validate_stage(e);
    if (e.kind == TrackingEventKind::frame_begin) {
        require(!s.active && e.frame == s.next_frame, "missing terminal or nonconsecutive frame");
        require(e.committed_frame == s.committed_frame, "wrong source committed frame");
        s.active = true; s.frame = e.frame; s.phase = e.phase;
        s.working = s.committed; s.working_next_uid = s.next_uid;
        s.original.clear(); for (const auto& [uid, t] : s.committed) { (void)t; s.original.insert(uid); }
        s.predicted.clear(); s.updated.clear(); s.matched.clear(); s.emitted.clear(); s.candidates.clear(); s.used.clear();
        s.batch_ready = false; s.batch_count = 0; s.pending_confirm = 0; s.last_emit_uid = 0;
        s.last_predict_uid = s.last_update_uid = s.last_donor_uid = 0;
        return;
    }
    require(s.active && e.frame == s.frame, "event outside active frame");
    if (e.kind == TrackingEventKind::frame_fail || e.kind == TrackingEventKind::frame_skipped) {
        require(e.committed_frame == s.committed_frame, "failure changed committed frame");
        if (e.kind == TrackingEventKind::frame_fail) {
            require(s.failure == Error::none, "duplicate first failure"); s.failure = e.error;
        } else require(s.failure != Error::none && s.failure == e.error && s.phase == TrackingEventPhase::frame, "invalid sticky skipped frame");
        s.active = false; ++s.next_frame; return;
    }
    require(s.failure == Error::none, "behavior after sticky failure");
    require(e.phase >= s.phase, "event phase order regressed");
    require(s.pending_confirm == 0 || (e.kind == TrackingEventKind::confirm && e.before.uid == s.pending_confirm), "missing required confirm");
    ready_for_phase(s, e.phase);
    // Predict removals must occur before any later phase, including expiration.
    if (e.phase > TrackingEventPhase::predict)
        for (const auto& [uid, t] : s.working) { (void)uid; require(t.age <= c.life.track_buffer, "missing expired removal"); }
    s.phase = e.phase;
    switch (e.kind) {
    case TrackingEventKind::predict: {
        before_matches(s, e.before);
        require(s.original.contains(e.before.uid) && s.predicted.insert(e.before.uid).second && e.before.uid > s.last_predict_uid, "duplicate/out-of-order predict");
        s.last_predict_uid = e.before.uid;
        require(e.before.age < std::numeric_limits<std::uint32_t>::max(), "predict age overflow");
        Track expected = e.before; ++expected.age; expected.x = e.after.x; expected.P = e.after.P;
        require(same_track(expected, e.after), "invalid predict bookkeeping"); s.working[e.after.uid] = e.after; break;
    }
    case TrackingEventKind::batch_ready: require(!s.batch_ready && e.count <= kBatchCapacity, "duplicate/oversized batch"); s.batch_ready = true; s.batch_count = e.count; break;
    case TrackingEventKind::candidate: {
        const auto& d = e.detection;
        require(s.batch_ready && s.candidates.size() < s.batch_count, "candidate without declared capacity");
        require(d.can_continue, "invalid candidate continuation eligibility");
        require(d.observed_score >= 0 && d.observed_score <= 1 && d.association_score >= 0 && d.association_score <= 1, "invalid candidate score");
        require(s.candidates.empty() || s.candidates.rbegin()->first < d.candidate_id, "candidate order/identity duplicate");
        if (d.can_birth) require(d.association_score == d.observed_score, "birth candidate score changed");
        else require(d.association_score == tracking_math::min(d.observed_score, c.life.new_track_thresh), "continuation association score mismatch");
        s.candidates.emplace(d.candidate_id, d); break;
    }
    case TrackingEventKind::associate: {
        before_matches(s, e.before);
        require(s.updated.insert(e.before.uid).second && e.before.uid > s.last_update_uid, "duplicate/out-of-order track update");
        s.last_update_uid = e.before.uid;
        const auto d = s.candidates.find(e.detection.candidate_id);
        require(d != s.candidates.end() && same_detection(d->second, e.detection), "association candidate differs from batch");
        require(s.used.insert(e.detection.candidate_id).second, "candidate used twice");
        const auto stage = e.association_stage; const float score = e.detection.association_score;
        const bool tentative = e.before.lifecycle == TrackLifecycle::tentative;
        require(stage >= 1 && stage <= 4 && tentative == (stage == 3), "illegal association track stage");
        const bool range = stage == 1 ? score >= c.assoc.high_thresh : stage == 2 ? score >= c.assoc.mid_thresh && score < c.assoc.high_thresh : stage == 3 ? score >= c.assoc.mid_thresh : score >= c.batch.track_thresh && score < c.assoc.mid_thresh;
        require(range, "illegal association score stage");
        Track expected = e.before; expected.x = e.after.x; expected.P = e.after.P;
        expected.age = 0; ++expected.hit_streak; expected.score_sum += e.detection.observed_score;
        expected.observed_score = e.detection.observed_score; expected.association_score = e.detection.association_score;
        expected.class_id = e.detection.class_id; expected.last_observed_frame = e.frame;
        push_history(expected, e.detection); expected.h_ema = tracking_math::height_ema(expected.h_ema, e.detection.xyxy[3] - e.detection.xyxy[1]);
        require(same_track(expected, e.after), "invalid associate state transition");
        s.matched.insert(e.after.uid); s.working[e.after.uid] = e.after;
        if (confirmable(e.after, c)) s.pending_confirm = e.after.uid;
        break;
    }
    case TrackingEventKind::birth: {
        const auto d = s.candidates.find(e.detection.candidate_id);
        require(d != s.candidates.end() && same_detection(d->second, e.detection), "birth candidate differs from batch");
        require(e.detection.can_birth && e.detection.observed_score >= c.life.new_track_thresh, "ineligible continuation-only/low-score birth");
        require(s.used.insert(e.detection.candidate_id).second, "candidate used twice");
        require(s.working.size() < kTrackCapacity && e.after.uid == s.working_next_uid && e.after.uid != 0 && !s.working.contains(e.after.uid), "birth identity/capacity violation");
        // Birth order is the batch candidate-ID order among eligible unused rows.
        for (const auto& [id, candidate] : s.candidates)
            if (id < e.detection.candidate_id && !s.used.contains(id) && candidate.can_birth && candidate.observed_score >= c.life.new_track_thresh)
                require(false, "birth candidate order regressed");
        Track expected; expected.uid = expected.output_id = s.working_next_uid++;
        expected.x = e.after.x; expected.P = e.after.P; expected.hit_streak = 1;
        expected.score_sum = expected.observed_score = e.detection.observed_score;
        expected.association_score = e.detection.association_score; expected.class_id = e.detection.class_id;
        expected.birth_frame = expected.last_observed_frame = e.frame; push_history(expected, e.detection);
        expected.h_ema = e.detection.xyxy[3] - e.detection.xyxy[1];
        require(same_track(expected, e.after), "invalid birth state transition");
        s.updated.insert(e.after.uid); s.matched.insert(e.after.uid); s.working.emplace(e.after.uid, e.after);
        if (confirmable(e.after, c)) s.pending_confirm = e.after.uid;
        break;
    }
    case TrackingEventKind::confirm: {
        before_matches(s, e.before);
        require(s.pending_confirm == e.before.uid && confirmable(e.before, c), "ineligible/duplicate confirmation");
        Track expected = e.before; expected.lifecycle = TrackLifecycle::confirmed;
        require(same_track(expected, e.after), "confirmation changed unrelated state");
        s.working[e.after.uid] = e.after; s.pending_confirm = 0; break;
    }
    case TrackingEventKind::lost: {
        before_matches(s, e.before);
        require(e.before.lifecycle == TrackLifecycle::confirmed && e.before.age > 0 && s.updated.insert(e.before.uid).second && e.before.uid > s.last_update_uid, "illegal/out-of-order lost transition");
        s.last_update_uid = e.before.uid;
        Track expected = e.before; expected.hit_streak = 0; expected.score_sum = 0;
        require(same_track(expected, e.after), "lost changed unrelated state"); s.working[e.after.uid] = e.after; break;
    }
    case TrackingEventKind::remove: {
        before_matches(s, e.before);
        if (e.phase == TrackingEventPhase::predict) {
            require(s.predicted.contains(e.before.uid), "remove before predict");
            if (e.reason == TrackingEventReason::expired) require(e.before.age > c.life.track_buffer, "premature expiry");
            else require(e.reason == TrackingEventReason::invalid_geometry && bad_geometry(e.before), "invalid geometry removal");
        } else {
            require(e.reason == TrackingEventReason::unmatched && e.before.lifecycle == TrackLifecycle::tentative && s.updated.insert(e.before.uid).second && e.before.uid > s.last_update_uid, "illegal/out-of-order unmatched removal");
            s.last_update_uid = e.before.uid;
        }
        s.working.erase(e.before.uid); break;
    }
    case TrackingEventKind::relink: {
        before_matches(s, e.before);
        require(c.bridge.enabled && s.matched.contains(e.before.uid) && e.related_uid > s.last_donor_uid, "bridge disabled, receiver unmatched or donor order regressed");
        s.last_donor_uid = e.related_uid;
        const auto donor = s.working.find(e.related_uid);
        require(donor != s.working.end() && e.related_uid != e.before.uid && !s.matched.contains(e.related_uid) && s.updated.contains(e.related_uid), "invalid Bridge donor");
        const auto& lost = donor->second; const auto& receiver = e.before;
        require(receiver.output_id == receiver.uid && receiver.hit_streak == c.bridge.trigger_streak && e.frame - receiver.birth_frame == c.bridge.trigger_streak - 1, "ineligible Bridge receiver");
        require(lost.lifecycle == TrackLifecycle::confirmed && lost.age >= c.bridge.min_lost && lost.age <= c.life.track_buffer && lost.last_observed_frame < receiver.birth_frame && lost.history_count == kBridgeHistory, "ineligible Bridge donor");
        Track expected = receiver; expected.output_id = lost.output_id;
        require(same_track(expected, e.after), "Bridge changed UID/state or lost output identity");
        s.working.erase(e.related_uid); s.working[e.after.uid] = e.after; break;
    }
    case TrackingEventKind::emit: {
        const auto it = s.working.find(e.row.uid);
        require(it != s.working.end() && it->second.lifecycle == TrackLifecycle::confirmed && it->second.age == 0 && s.updated.contains(e.row.uid), "ineligible emitted track");
        require(e.row.uid > s.last_emit_uid && s.emitted.insert(e.row.uid).second, "emit order/identity duplicate");
        s.last_emit_uid = e.row.uid; const auto& t = it->second;
        TrackRow expected; expected.uid = t.uid; expected.output_id = t.output_id; expected.class_id = t.class_id; expected.flags = 1;
        tracking_math::state_to_box(t.x.data(), expected.xyxy.data()); expected.observed_score = t.observed_score; expected.association_score = t.association_score;
        require(same_row(expected, e.row), "emit row differs from committed decision"); break;
    }
    case TrackingEventKind::frame_commit: {
        require(e.committed_frame == e.frame && e.count == s.emitted.size() && e.count <= packet_capacity, "invalid successful commit/count");
        for (const auto& [uid, t] : s.working) {
            require(s.updated.contains(uid), "missing track update/lost event");
            require(!confirmable(t, c), "missing confirmation");
            const bool expected_emit = t.lifecycle == TrackLifecycle::confirmed && t.age == 0;
            require(s.emitted.contains(uid) == expected_emit, "missing or unexpected emit");
        }
        for (const auto& [id, d] : s.candidates)
            require(s.used.contains(id) || !d.can_birth || d.observed_score < c.life.new_track_thresh, "missing eligible birth");
        std::set<std::uint64_t> output_ids;
        for (const auto& [uid, t] : s.working) { (void)uid; require(output_ids.insert(t.output_id).second, "duplicate live output identity"); }
        s.committed = s.working; s.next_uid = s.working_next_uid; s.committed_frame = e.frame;
        s.active = false; ++s.next_frame; break;
    }
    case TrackingEventKind::frame_begin:
    case TrackingEventKind::frame_fail:
    case TrackingEventKind::frame_skipped: break;
    }
}
using Fields = std::vector<std::pair<std::string, std::string>>;
void number(Fields& f, const std::string& key, std::uint64_t value) { f.emplace_back(key, std::to_string(value)); }
void floating(Fields& f, const std::string& key, float value) { number(f, key + ".fp32_bits", std::bit_cast<std::uint32_t>(value)); }
void track_fields(Fields& f, const std::string& p, const Track& t) {
    number(f,p+"uid",t.uid); number(f,p+"output_id",t.output_id); number(f,p+"lifecycle",static_cast<std::uint32_t>(t.lifecycle));
    for (std::size_t i=0;i<t.x.size();++i) floating(f,p+"x["+std::to_string(i)+"]",t.x[i]);
    for (std::size_t i=0;i<t.P.size();++i) floating(f,p+"P["+std::to_string(i)+"]",t.P[i]);
    number(f,p+"age",t.age); number(f,p+"hit_streak",t.hit_streak); floating(f,p+"score_sum",t.score_sum); number(f,p+"class_id",t.class_id);
    floating(f,p+"observed_score",t.observed_score); floating(f,p+"association_score",t.association_score);
    number(f,p+"birth_frame",static_cast<std::uint64_t>(t.birth_frame)); number(f,p+"last_observed_frame",static_cast<std::uint64_t>(t.last_observed_frame)); number(f,p+"history_count",t.history_count);
    for (std::uint32_t i=0;i<t.history_count;++i) { const auto h=p+"history["+std::to_string(i)+"]."; floating(f,h+"cx",t.history[i].cx); floating(f,h+"cy",t.history[i].cy); floating(f,h+"h",t.history[i].h); }
    floating(f,p+"h_ema",t.h_ema);
}
void detection_fields(Fields& f, const BatchDetection& d) {
    for (std::size_t i=0;i<d.xyxy.size();++i) floating(f,"detection.xyxy["+std::to_string(i)+"]",d.xyxy[i]);
    floating(f,"detection.observed_score",d.observed_score); floating(f,"detection.association_score",d.association_score);
    number(f,"detection.class_id",d.class_id); number(f,"detection.candidate_id",d.candidate_id); number(f,"detection.can_continue",d.can_continue); number(f,"detection.can_birth",d.can_birth);
}
Fields fields(const TrackingEvent& e) {
    Fields f; number(f,"source",e.source); number(f,"epoch",e.epoch); number(f,"frame",static_cast<std::uint64_t>(e.frame));
    number(f,"kind",static_cast<std::uint32_t>(e.kind)); number(f,"phase",static_cast<std::uint32_t>(e.phase)); number(f,"stage",static_cast<std::uint32_t>(e.stage));
    switch(e.kind) {
    case TrackingEventKind::frame_begin: number(f,"committed_frame",static_cast<std::uint64_t>(e.committed_frame)); break;
    case TrackingEventKind::batch_ready: number(f,"count",e.count); break;
    case TrackingEventKind::candidate: detection_fields(f,e.detection); break;
    case TrackingEventKind::predict: case TrackingEventKind::confirm: case TrackingEventKind::lost: track_fields(f,"before.",e.before); track_fields(f,"after.",e.after); break;
    case TrackingEventKind::associate: track_fields(f,"before.",e.before); track_fields(f,"after.",e.after); detection_fields(f,e.detection); number(f,"association_stage",e.association_stage); break;
    case TrackingEventKind::birth: track_fields(f,"after.",e.after); detection_fields(f,e.detection); break;
    case TrackingEventKind::remove: track_fields(f,"before.",e.before); number(f,"reason",static_cast<std::uint32_t>(e.reason)); break;
    case TrackingEventKind::relink: track_fields(f,"before.",e.before); track_fields(f,"after.",e.after); number(f,"related_uid",e.related_uid); break;
    case TrackingEventKind::emit:
        number(f,"row.uid",e.row.uid); number(f,"row.output_id",e.row.output_id); number(f,"row.class_id",e.row.class_id); number(f,"row.flags",e.row.flags);
        for(std::size_t i=0;i<4;++i) floating(f,"row.xyxy["+std::to_string(i)+"]",e.row.xyxy[i]);
        floating(f,"row.observed_score",e.row.observed_score); floating(f,"row.association_score",e.row.association_score); break;
    case TrackingEventKind::frame_commit: number(f,"committed_frame",static_cast<std::uint64_t>(e.committed_frame)); number(f,"count",e.count); break;
    case TrackingEventKind::frame_fail: case TrackingEventKind::frame_skipped: number(f,"committed_frame",static_cast<std::uint64_t>(e.committed_frame)); number(f,"error",static_cast<std::uint32_t>(e.error)); break;
    }
    return f;
}
} // namespace

std::vector<std::uint8_t> encode_tracking_events(std::span<const TrackingEvent> events) {
    Bytes out{'S','C','E','V','T',1,0,0}; u64(out, events.size());
    for (const auto& e : events) {
        auto encoded = encode_event(e); require(encoded.size() <= std::numeric_limits<std::uint32_t>::max(), "oversized tracking event");
        u32(out, static_cast<std::uint32_t>(encoded.size())); out.insert(out.end(), encoded.begin(), encoded.end());
    }
    return out;
}
std::vector<TrackingEvent> decode_tracking_events(std::span<const std::uint8_t> bytes) {
    Reader in{bytes}; const Bytes magic{'S','C','E','V','T',1,0,0};
    const auto header = in.take(magic.size()); require(std::equal(header.begin(), header.end(), magic.begin()), "unknown tracking-events schema");
    const auto count = in.u64(); require(count <= (bytes.size()-in.at)/40, "impossible event count");
    std::vector<TrackingEvent> events; events.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i=0;i<count;++i) {
        const auto n = in.u32(); Reader event{in.take(n)}; auto decoded = decode_event(event);
        require(event.at == event.bytes.size(), "trailing event payload");
        const auto canonical = encode_event(decoded); require(canonical.size() == event.bytes.size() && std::equal(canonical.begin(), canonical.end(), event.bytes.begin()), "noncanonical event payload");
        events.push_back(std::move(decoded));
    }
    require(in.at == bytes.size(), "trailing tracking-events bytes"); return events;
}
std::string tracking_events_hash(std::span<const TrackingEvent> events) {
    const auto bytes = encode_tracking_events(events);
    return sha256_hex(std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}
TrackingEventComparison compare_tracking_events(std::span<const TrackingEvent> left, std::span<const TrackingEvent> right) {
    for (std::size_t i=0;i<std::min(left.size(),right.size());++i) {
        (void)encode_event(left[i]); (void)encode_event(right[i]);
        const auto a=fields(left[i]), b=fields(right[i]);
        for(std::size_t j=0;j<std::min(a.size(),b.size());++j)
            if(a[j]!=b[j]) return {false,i,a[j].first == b[j].first ? a[j].first : "payload",a[j].first+"="+a[j].second,b[j].first+"="+b[j].second};
        if(a.size()!=b.size()) return {false,i,"payload.length",std::to_string(a.size()),std::to_string(b.size())};
    }
    if(left.size()!=right.size()) return {false,std::min(left.size(),right.size()),"events.length",std::to_string(left.size()),std::to_string(right.size())};
    return {};
}
TrackingEventValidation validate_tracking_events(std::span<const TrackingEvent> events, const TrackerConfig& config) {
    TrackingEventValidation result;
    try { validate_tracker_config(config); } catch(const std::exception& e) { return {false,0,0,e.what()}; }
    if(events.empty()) return {false,0,0,"empty event trace"};
    std::map<std::pair<std::uint64_t,Epoch>,Session> sessions;
    for(std::size_t i=0;i<events.size();++i) {
        result.event_index=i;
        try {
            (void)encode_event(events[i]);
            auto& session=sessions[{events[i].source,events[i].epoch}]; validate_event(session,events[i],config);
            if(events[i].kind==TrackingEventKind::frame_commit || events[i].kind==TrackingEventKind::frame_fail || events[i].kind==TrackingEventKind::frame_skipped) ++result.frames;
        } catch(const std::exception& e) { result.valid=false; result.message=e.what(); return result; }
    }
    for(const auto& [identity,session]:sessions) {
        (void)identity;
        if(session.active) return {false,events.size(),result.frames,"missing terminal event at end of trace"};
    }
    result.event_index=events.size(); return result;
}

TrackingEventValidation validate_tracking_event_packets(std::span<const TrackingEvent> events,
                                                        std::span<const std::uint8_t> packets) {
    TrackingEventValidation result;
    if(events.empty()) return {false,0,0,"empty event trace"};
    if(packets.empty() || packets.size()%packet_bytes != 0)
        return {false,0,0,"empty or partial ResultPacket stream"};
    const auto source=events.front().source;
    const auto epoch=events.front().epoch;
    bool active=false;
    FrameId frame=no_frame, committed=no_frame, first_failure=no_frame;
    Error fail_code=Error::none;
    ResultPacket packet;
    std::uint64_t last_emit_uid=0;
    for(std::size_t i=0;i<events.size();++i) {
        result.event_index=i;
        const auto& e=events[i];
        try {
            (void)encode_event(e); validate_stage(e);
            require(e.source==source && e.epoch==epoch,"packet binding requires one source/epoch");
            if(e.kind==TrackingEventKind::frame_begin) {
                require(!active && e.frame==static_cast<FrameId>(result.frames),"missing terminal or nonconsecutive packet frame");
                require(e.committed_frame==committed,"packet source committed frame mismatch");
                active=true; frame=e.frame; last_emit_uid=0; packet=ResultPacket{};
                packet.epoch=epoch; packet.frame=frame; packet.committed_frame=committed;
                continue;
            }
            require(active && e.frame==frame,"packet event outside active frame");
            if(first_failure!=no_frame)
                require(e.kind==TrackingEventKind::frame_skipped,"packet behavior after first failure");
            if(e.kind==TrackingEventKind::emit) {
                require(e.row.uid>last_emit_uid,"packet emit UID order/duplicate");
                last_emit_uid=e.row.uid; packet.tracks.push_back(e.row); continue;
            }
            const bool terminal=e.kind==TrackingEventKind::frame_commit || e.kind==TrackingEventKind::frame_fail || e.kind==TrackingEventKind::frame_skipped;
            if(!terminal) continue;
            if(e.kind==TrackingEventKind::frame_commit) {
                require(first_failure==no_frame && e.committed_frame==frame && e.count==packet.tracks.size(),"packet successful commit/count mismatch");
                committed=frame; packet.committed_frame=frame;
            } else {
                require(e.committed_frame==committed,"packet failure changed committed frame");
                packet.tracks.clear();
                if(e.kind==TrackingEventKind::frame_fail) {
                    require(first_failure==no_frame,"packet duplicate first failure");
                    first_failure=frame; fail_code=e.error; packet.status=ResultStatus::failed; packet.frame_error=e.error;
                } else {
                    require(first_failure!=no_frame && e.error==fail_code,"packet sticky failure mismatch");
                    packet.status=ResultStatus::skipped_after_failure;
                }
                packet.fail_frame=first_failure; packet.fail_code=fail_code;
            }
            const auto expected=encode_packet(packet);
            require(result.frames<packets.size()/packet_bytes,"more event frames than packets");
            const auto actual=packets.subspan(result.frames*packet_bytes,packet_bytes);
            for(std::size_t offset=0;offset<packet_bytes;++offset)
                if(expected[offset]!=actual[offset]) {
                    result.valid=false;
                    result.message="packet byte mismatch at frame "+std::to_string(frame)+" offset "+std::to_string(offset)+": event="+std::to_string(expected[offset])+" packet="+std::to_string(actual[offset]);
                    return result;
                }
            ++result.frames; active=false;
        } catch(const std::exception& error) { result.valid=false; result.message=error.what(); return result; }
    }
    if(active) return {false,events.size(),result.frames,"missing terminal event at end of packet trace"};
    if(result.frames!=packets.size()/packet_bytes)
        return {false,events.size(),result.frames,"more packets than event frames"};
    result.event_index=events.size(); return result;
}

} // namespace saccade

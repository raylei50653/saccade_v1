#include "saccade/tracking.hpp"
#include "saccade/tracking_math.hpp"
#include "saccade/tracking_events.hpp"

#include <algorithm>
#include <utility>

namespace saccade {
namespace {
namespace math = tracking_math;
TrackingEvent event(std::uint64_t source, Epoch epoch, FrameId frame, TrackingEventKind kind,
                    TrackingEventPhase phase, TrackingEventStage stage) {
    TrackingEvent e;
    e.source = source; e.epoch = epoch; e.frame = frame;
    e.kind = kind; e.phase = phase; e.stage = stage;
    return e;
}
void transition(TrackingEventObserver* observer, std::uint64_t source, Epoch epoch, FrameId frame,
                TrackingEventKind kind, TrackingEventPhase phase, const Track& before, const Track& after,
                const BatchDetection* detection = nullptr, std::uint32_t association_stage = 0,
                TrackingEventReason reason = TrackingEventReason::none, std::uint64_t related_uid = 0) {
    if (!observer) return;
    auto e = event(source, epoch, frame, kind, phase,
                   phase == TrackingEventPhase::predict ? TrackingEventStage::f08a : TrackingEventStage::f08b);
    e.before = before; e.after = after; e.reason = reason; e.related_uid = related_uid;
    if (detection) e.detection = *detection;
    e.association_stage = association_stage;
    observer->record(e);
}
bool finite_box(const std::array<float, 4>& box) {
    for (float v : box) if (!math::finite(v)) return false;
    return true;
}
bool finite_track(const Track& t) {
    for (float v : t.x) if (!math::finite(v)) return false;
    for (float v : t.P) if (!math::finite(v)) return false;
    if (!math::finite(t.score_sum) || !math::finite(t.h_ema) ||
        !math::finite(t.observed_score) || !math::finite(t.association_score)) return false;
    for (std::uint32_t i = 0; i < t.history_count; ++i)
        if (!math::finite(t.history[i].cx) || !math::finite(t.history[i].cy) ||
            !math::finite(t.history[i].h)) return false;
    return true;
}
bool finite_detection(const BatchDetection& d) {
    float z[4];
    math::box_to_state(d.xyxy.data(), z);
    for (float v : z) if (!math::finite(v)) return false;
    return finite_box(d.xyxy) && math::finite(d.observed_score) && math::finite(d.association_score);
}
void push_history(Track& t, const std::array<float, 4>& box) {
    float z[4];
    math::box_to_state(box.data(), z);
    if (t.history_count == kBridgeHistory) {
        for (std::uint32_t i = 1; i < kBridgeHistory; ++i) t.history[i - 1] = t.history[i];
        --t.history_count;
    }
    t.history[t.history_count++] = {z[0], z[1], z[3]};
}
void confirm(Track& t, const TrackerConfig& c) {
    if (t.lifecycle == TrackLifecycle::tentative && t.hit_streak >= c.life.confirm_streak &&
        t.score_sum / static_cast<float>(t.hit_streak) >= c.life.confirm_score)
        t.lifecycle = TrackLifecycle::confirmed;
}
float velocity_x(const Track& t) {
    return math::regression_velocity(t.history[0].cx, t.history[1].cx, t.history[2].cx, t.history[3].cx);
}
float velocity_y(const Track& t) {
    return math::regression_velocity(t.history[0].cy, t.history[1].cy, t.history[2].cy, t.history[3].cy);
}
} // namespace

TrackState initial_track_state(Epoch epoch) {
    TrackState state;
    state.epoch = epoch;
    return state;
}

PredictedState predict_tracks(const TrackState& committed, const MotionContext& motion, FrameId frame,
                              const TrackerConfig& c, TrackingEventObserver* observer, std::uint64_t source) {
    PredictedState out;
    out.epoch = committed.epoch;
    out.source_committed_frame = committed.committed_frame;
    out.frame = frame;
    out.motion = motion;
    out.next_uid = committed.next_uid;
    for (auto t : committed.tracks) {
        const auto before = observer ? t : Track{};
        if (!finite_track(t)) { out.error = Error::numeric; return out; }
        if (motion.status == MotionStatus::valid) {
            t.x[0] += motion.tx;
            t.x[1] += motion.ty;
        }
        math::kalman_predict(t.x.data(), t.P.data(), c.kalman.std_position, c.kalman.std_velocity);
        ++t.age;
        if (!finite_track(t)) { out.error = Error::numeric; return out; }
        transition(observer, source, committed.epoch, frame, TrackingEventKind::predict,
                   TrackingEventPhase::predict, before, t);
        if (t.age > c.life.track_buffer || t.x[3] < 1 || t.x[2] <= 0) {
            transition(observer, source, committed.epoch, frame, TrackingEventKind::remove,
                       TrackingEventPhase::predict, t, {}, nullptr, 0,
                       t.age > c.life.track_buffer ? TrackingEventReason::expired : TrackingEventReason::invalid_geometry);
            continue;
        }
        std::array<float, 4> box;
        math::state_to_box(t.x.data(), box.data());
        if (!finite_box(box)) { out.error = Error::numeric; return out; }
        if (t.lifecycle == TrackLifecycle::confirmed) out.priors.push_back({t.uid, box, t.class_id});
        out.tracks.push_back(t);
    }
    return out;
}

DetectionBatch assemble_detection_batch(const DetectionCandidates& candidates, const PredictedState& predicted,
                                        const TrackerConfig& c) {
    DetectionBatch out;
    out.epoch = candidates.header.epoch;
    out.frame = static_cast<FrameId>(candidates.header.frame);
    const auto& h = candidates.header;
    if (h.epoch != predicted.epoch || out.frame != predicted.frame || h.done != kPostprocessDone ||
        h.count > kPostprocessCapacity) // count bounds the rows read below
        out.error = out.error | Error::protocol;
    if (h.flags & kPostprocessOverflow) out.error = out.error | Error::overflow;
    if (h.flags & ~kPostprocessOverflow) out.error = out.error | Error::producer;
    if (out.error != Error::none) return out;
    const auto convert = [](const DetectionCandidate& d, bool birth, float association) {
        return BatchDetection{{d.x1, d.y1, d.x2, d.y2}, d.score, association,
                              d.class_id, d.candidate_id, true, birth};
    };
    std::vector<const DetectionCandidate*> kept, suppressed;
    for (std::uint32_t i = 0; i < h.count; ++i) {
        const auto& d = candidates.rows[i];
        if (!math::finite(d.score) || !math::finite(d.x1) || !math::finite(d.y1) ||
            !math::finite(d.x2) || !math::finite(d.y2)) {
            out.error = Error::numeric;
            out.rows.clear();
            return out;
        }
        if (d.flags & kDetectionKept) {
            kept.push_back(&d);
            if (d.score >= c.batch.track_thresh) out.rows.push_back(convert(d, true, d.score));
        } else if (d.flags & kDetectionSuppressed) suppressed.push_back(&d);
    }
    if (c.continuation.enabled && !predicted.priors.empty()) {
        std::sort(suppressed.begin(), suppressed.end(), [](const auto* a, const auto* b) {
            return a->score != b->score ? a->score > b->score : a->candidate_id < b->candidate_id;
        });
        std::vector<BatchDetection> accepted;
        for (const auto* d : suppressed) {
            if (accepted.size() == c.continuation.max_candidates) break;
            if (d->score < c.continuation.min_score) continue;
            const std::array<float, 4> box{d->x1, d->y1, d->x2, d->y2};
            bool prior_match = false, dedup = true;
            for (const auto& p : predicted.priors) {
                const float overlap = math::iou(box.data(), p.xyxy.data());
                if (!math::finite(overlap)) { out.error = Error::numeric; break; }
                if (overlap >= c.continuation.prior_iou) prior_match = true;
            }
            if (!prior_match || out.error != Error::none) continue;
            for (const auto* k : kept) {
                const float kb[4] = {k->x1, k->y1, k->x2, k->y2};
                const float overlap = math::iou(box.data(), kb);
                if (!math::finite(overlap)) out.error = Error::numeric;
                if (overlap > c.continuation.dedup_iou) dedup = false;
            }
            for (const auto& a : accepted) {
                const float overlap = math::iou(box.data(), a.xyxy.data());
                if (!math::finite(overlap)) out.error = Error::numeric;
                if (overlap > c.continuation.dedup_iou) dedup = false;
            }
            if (dedup) accepted.push_back(convert(*d, false, math::min(d->score, c.life.new_track_thresh)));
        }
        out.rows.insert(out.rows.end(), accepted.begin(), accepted.end());
    }
    std::sort(out.rows.begin(), out.rows.end(), [](const auto& a, const auto& b) {
        return a.candidate_id < b.candidate_id;
    });
    if (out.error != Error::none) out.rows.clear();
    return out;
}

UpdateResult associate_and_update(const PredictedState& predicted, const DetectionBatch& batch,
                                  const MotionContext& motion, const TrackerConfig& c,
                                  TrackingEventObserver* observer, std::uint64_t source) {
    UpdateResult out;
    out.error = predicted.error | batch.error;
    if (batch.epoch != predicted.epoch || motion.epoch != predicted.epoch ||
        predicted.motion.epoch != predicted.epoch || batch.frame != predicted.frame ||
        motion.frame != predicted.frame || predicted.motion.frame != predicted.frame ||
        predicted.source_committed_frame != predicted.frame - 1)
        out.error = out.error | Error::protocol;
    if (out.error != Error::none) return out;
    out.next = {predicted.epoch, predicted.frame, predicted.next_uid, predicted.tracks};
    auto& tracks = out.next.tracks;
    for (const auto& t : tracks) if (!finite_track(t)) out.error = Error::numeric;
    for (const auto& d : batch.rows) if (!finite_detection(d)) out.error = Error::numeric;
    if (out.error != Error::none) return out;
    std::vector<int> matched(tracks.size(), -1);
    std::vector<std::uint32_t> matched_stage(observer ? tracks.size() : 0, 0);
    std::vector<bool> used(batch.rows.size(), false), removed(tracks.size(), false);
    struct Proposal { int detection = -1; float p = 0, bid = 0; };
    struct Option { int detection; float p; };
    for (int stage = 0; stage < 4; ++stage) {
        std::vector<Proposal> proposals(tracks.size());
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            const auto& t = tracks[i];
            if (matched[i] >= 0 || (t.lifecycle == TrackLifecycle::tentative) != (stage == 2)) continue;
            float box[4];
            math::state_to_box(t.x.data(), box);
            std::vector<Option> options;
            for (std::size_t j = 0; j < batch.rows.size(); ++j) {
                const auto& d = batch.rows[j];
                const float score = d.association_score;
                const bool in_range = stage == 0 ? score >= c.assoc.high_thresh :
                    stage == 1 ? score >= c.assoc.mid_thresh && score < c.assoc.high_thresh :
                    stage == 2 ? score >= c.assoc.mid_thresh :
                                 score >= c.batch.track_thresh && score < c.assoc.mid_thresh;
                if (used[j] || !in_range) continue;
                const float overlap = math::iou(box, d.xyxy.data());
                if (!math::finite(overlap)) { out.error = Error::numeric; return out; }
                if (overlap <= 0) continue;
                const float h = d.xyxy[3] - d.xyxy[1], w = d.xyxy[2] - d.xyxy[0];
                const float cost = math::association_cost(overlap, t.x[3], h, c.assoc.lambda, c.assoc.stability_cost_w);
                const float p = math::association_value(cost, w, h, c.assoc.lambda);
                if (!math::finite(cost) || !math::finite(p)) { out.error = Error::numeric; return out; }
                if (cost <= (stage == 3 ? c.assoc.stage2_match_thresh : c.assoc.match_thresh))
                    options.push_back({static_cast<int>(j), p});
            }
            std::sort(options.begin(), options.end(), [&](const auto& a, const auto& b) {
                return a.p != b.p ? a.p > b.p : batch.rows[a.detection].candidate_id < batch.rows[b.detection].candidate_id;
            });
            if (options.size() > c.assoc.top_k) options.resize(c.assoc.top_k);
            if (options.empty()) continue;
            const auto& best = options.front();
            const auto& d = batch.rows[best.detection];
            const float bid = math::auction_bid(best.p, options.size() > 1 ? options[1].p : 0.0f,
                options.size() > 1, c.assoc.auction_eps, c.assoc.stability_bid_w, t.x[3], d.xyxy[3] - d.xyxy[1]);
            if (!math::finite(bid)) { out.error = Error::numeric; return out; }
            proposals[i] = {best.detection, best.p, bid};
        }
        std::vector<int> winners(batch.rows.size(), -1);
        for (std::size_t i = 0; i < proposals.size(); ++i) {
            const auto& p = proposals[i];
            if (p.detection < 0) continue;
            int& winner = winners[p.detection];
            if (winner < 0 || p.bid > proposals[winner].bid ||
                (p.bid == proposals[winner].bid && (p.p > proposals[winner].p ||
                 (p.p == proposals[winner].p && tracks[i].uid < tracks[winner].uid))))
                winner = static_cast<int>(i);
        }
        for (std::size_t j = 0; j < winners.size(); ++j)
            if (winners[j] >= 0) {
                matched[winners[j]] = static_cast<int>(j); used[j] = true;
                if (observer) matched_stage[winners[j]] = static_cast<std::uint32_t>(stage + 1);
            }
    }
    for (std::size_t i = 0; i < tracks.size(); ++i) {
        auto& t = tracks[i];
        const auto before = observer ? t : Track{};
        if (matched[i] < 0) {
            if (t.lifecycle == TrackLifecycle::tentative) removed[i] = true;
            else { t.hit_streak = 0; t.score_sum = 0; }
            transition(observer, source, predicted.epoch, predicted.frame,
                       removed[i] ? TrackingEventKind::remove : TrackingEventKind::lost,
                       TrackingEventPhase::update, before, removed[i] ? Track{} : t, nullptr, 0,
                       TrackingEventReason::unmatched);
            continue;
        }
        const auto& d = batch.rows[matched[i]];
        math::kalman_update(t.x.data(), t.P.data(), d.xyxy.data(), c.kalman.std_position, c.kalman.r_scale);
        t.age = 0;
        ++t.hit_streak;
        t.score_sum += d.observed_score;
        t.observed_score = d.observed_score;
        t.association_score = d.association_score;
        t.class_id = d.class_id;
        t.last_observed_frame = predicted.frame;
        push_history(t, d.xyxy);
        t.h_ema = math::height_ema(t.h_ema, d.xyxy[3] - d.xyxy[1]);
        confirm(t, c);
        if (!finite_track(t)) { out.error = Error::numeric; return out; }
        if (observer) {
            auto associated = t;
            associated.lifecycle = before.lifecycle;
            transition(observer, source, predicted.epoch, predicted.frame, TrackingEventKind::associate,
                       TrackingEventPhase::update, before, associated, &d, matched_stage[i]);
            if (t.lifecycle != before.lifecycle)
                transition(observer, source, predicted.epoch, predicted.frame, TrackingEventKind::confirm,
                           TrackingEventPhase::update, associated, t);
        }
    }
    if (c.bridge.enabled) {
        struct Claim { int lost = -1; };
        std::vector<Claim> claims(tracks.size());
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            const auto& t = tracks[i];
            if (matched[i] < 0 || t.hit_streak != c.bridge.trigger_streak ||
                predicted.frame - t.birth_frame != c.bridge.trigger_streak - 1 || t.output_id != t.uid) continue;
            std::vector<std::pair<float, int>> candidates;
            for (std::size_t j = 0; j < tracks.size(); ++j) {
                const auto& lost = tracks[j];
                if (matched[j] >= 0 || lost.lifecycle != TrackLifecycle::confirmed ||
                    lost.age < c.bridge.min_lost || lost.age > c.life.track_buffer ||
                    lost.last_observed_frame >= t.birth_frame || lost.history_count < kBridgeHistory) continue;
                const auto& l = lost.history[lost.history_count - 1];
                const auto& first = t.history[0];
                const float distance = math::bridge_distance(l.cx, l.cy, first.cx, first.cy,
                    velocity_x(lost), velocity_y(lost), velocity_x(t), velocity_y(t),
                    static_cast<float>(t.birth_frame - lost.last_observed_frame), lost.h_ema, t.h_ema, c.bridge.speed_norm);
                const float ratio = lost.h_ema / t.h_ema;
                if (!math::finite(distance) || !math::finite(ratio)) { out.error = Error::numeric; return out; }
                if (distance <= c.bridge.max_distance && ratio >= c.bridge.height_ratio_lo && ratio <= c.bridge.height_ratio_hi)
                    candidates.emplace_back(distance, static_cast<int>(j));
            }
            std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
                return a.first != b.first ? a.first < b.first : tracks[a.second].uid < tracks[b.second].uid;
            });
            if (candidates.empty() || (candidates.size() >= 2 &&
                candidates[1].first - candidates[0].first < c.bridge.margin)) continue;
            claims[i].lost = candidates[0].second;
        }
        std::vector<int> winners(tracks.size(), -1);
        for (std::size_t i = 0; i < tracks.size(); ++i) {
            if (claims[i].lost < 0) continue;
            int& winner = winners[claims[i].lost];
            if (winner < 0 || tracks[i].observed_score > tracks[winner].observed_score ||
                (tracks[i].observed_score == tracks[winner].observed_score && tracks[i].uid < tracks[winner].uid))
                winner = static_cast<int>(i);
        }
        for (std::size_t j = 0; j < winners.size(); ++j)
            if (winners[j] >= 0) {
                auto& receiver = tracks[winners[j]];
                const auto before = observer ? receiver : Track{};
                receiver.output_id = tracks[j].output_id; removed[j] = true;
                transition(observer, source, predicted.epoch, predicted.frame, TrackingEventKind::relink,
                           TrackingEventPhase::bridge, before, receiver, nullptr, 0,
                           TrackingEventReason::none, tracks[j].uid);
            }
    }
    std::vector<Track> survivors;
    for (std::size_t i = 0; i < tracks.size(); ++i) if (!removed[i]) survivors.push_back(tracks[i]);
    tracks = std::move(survivors);
    std::vector<std::size_t> birth_order;
    for (std::size_t j = 0; j < batch.rows.size(); ++j) if (!used[j]) birth_order.push_back(j);
    std::sort(birth_order.begin(), birth_order.end(), [&](auto a, auto b) {
        return batch.rows[a].candidate_id < batch.rows[b].candidate_id;
    });
    for (auto j : birth_order) {
        const auto& d = batch.rows[j];
        if (!d.can_birth || d.observed_score < c.life.new_track_thresh) continue;
        if (tracks.size() == kTrackCapacity) { out.error = Error::overflow; return out; }
        Track t;
        t.uid = out.next.next_uid++;
        t.output_id = t.uid;
        math::kalman_init(d.xyxy.data(), t.x.data(), t.P.data(), c.kalman.std_position, c.kalman.std_velocity);
        t.hit_streak = 1;
        t.score_sum = t.observed_score = d.observed_score;
        t.association_score = d.association_score;
        t.class_id = d.class_id;
        t.birth_frame = t.last_observed_frame = predicted.frame;
        push_history(t, d.xyxy);
        t.h_ema = d.xyxy[3] - d.xyxy[1];
        confirm(t, c);
        if (!finite_track(t)) { out.error = Error::numeric; return out; }
        if (observer) {
            auto born = t;
            born.lifecycle = TrackLifecycle::tentative;
            transition(observer, source, predicted.epoch, predicted.frame, TrackingEventKind::birth,
                       TrackingEventPhase::birth, {}, born, &d);
            if (t.lifecycle != born.lifecycle)
                transition(observer, source, predicted.epoch, predicted.frame, TrackingEventKind::confirm,
                           TrackingEventPhase::birth, born, t);
        }
        tracks.push_back(t);
    }
    for (const auto& t : tracks) {
        if (t.lifecycle != TrackLifecycle::confirmed || t.age != 0) continue;
        TrackRow row;
        row.uid = t.uid;
        row.output_id = t.output_id;
        row.class_id = t.class_id;
        row.flags = 1;
        math::state_to_box(t.x.data(), row.xyxy.data());
        row.observed_score = t.observed_score;
        row.association_score = t.association_score;
        if (!finite_box(row.xyxy)) { out.error = Error::numeric; out.rows.clear(); return out; }
        out.rows.push_back(row);
        if (observer) {
            auto e = event(source, predicted.epoch, predicted.frame, TrackingEventKind::emit,
                           TrackingEventPhase::emit, TrackingEventStage::f09);
            e.row = row;
            observer->record(e);
        }
    }
    if (out.rows.size() > packet_capacity) { out.error = Error::overflow; out.rows.clear(); }
    return out;
}

TrackerReference::TrackerReference(Epoch epoch, const TrackerConfig& config,
                                   TrackingEventObserver* observer, std::uint64_t source)
    : config_(config), committed_(initial_track_state(epoch)), observer_(observer), source_(source) {
    validate_tracker_config(config_);
}
ResultPacket TrackerReference::step(const DetectionCandidates& candidates, const MotionContext& motion) {
    ResultPacket packet;
    packet.epoch = committed_.epoch;
    packet.frame = next_frame_++;
    packet.committed_frame = committed_.committed_frame;
    const auto terminal = [&](TrackingEventKind kind) {
        if (!observer_) return;
        auto e = event(source_, packet.epoch, packet.frame, kind, TrackingEventPhase::commit, TrackingEventStage::control);
        e.committed_frame = packet.committed_frame;
        e.count = static_cast<std::uint32_t>(packet.tracks.size());
        e.error = packet.fail_code;
        observer_->record(e);
    };
    if (observer_) {
        auto e = event(source_, packet.epoch, packet.frame, TrackingEventKind::frame_begin,
                       TrackingEventPhase::frame, TrackingEventStage::control);
        e.committed_frame = packet.committed_frame;
        observer_->record(e);
    }
    if (failed()) {
        packet.status = ResultStatus::skipped_after_failure;
        packet.fail_frame = fail_frame_;
        packet.fail_code = fail_code_;
        terminal(TrackingEventKind::frame_skipped);
        return packet;
    }
    const auto predicted = predict_tracks(committed_, motion, packet.frame, config_, observer_, source_);
    const auto batch = assemble_detection_batch(candidates, predicted, config_);
    if (observer_ && batch.error == Error::none && predicted.error == Error::none) {
        auto e = event(source_, packet.epoch, packet.frame, TrackingEventKind::batch_ready,
                       TrackingEventPhase::batch, TrackingEventStage::f04b);
        e.count = static_cast<std::uint32_t>(batch.rows.size());
        observer_->record(e);
        for (const auto& d : batch.rows) {
            e.kind = TrackingEventKind::candidate; e.detection = d;
            observer_->record(e);
        }
    }
    auto update = associate_and_update(predicted, batch, motion, config_, observer_, source_);
    if (update.error != Error::none) {
        fail_frame_ = packet.frame;
        fail_code_ = update.error;
        packet.status = ResultStatus::failed;
        packet.fail_frame = fail_frame_;
        packet.frame_error = packet.fail_code = fail_code_;
        terminal(TrackingEventKind::frame_fail);
        return packet;
    }
    committed_ = std::move(update.next);
    packet.committed_frame = committed_.committed_frame;
    packet.tracks = std::move(update.rows);
    terminal(TrackingEventKind::frame_commit);
    return packet;
}
} // namespace saccade

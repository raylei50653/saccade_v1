#include "saccade/tracking_cuda.hpp"
#include "saccade/tracking_math.hpp"

#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace saccade {
namespace {
namespace math = tracking_math;
constexpr int threads = 256;
constexpr int sort_items = 2; // 512 blocked items cover the complete 300-row pool.
using SuppressedSort = cub::BlockRadixSort<std::uint64_t, threads, sort_items, int>;
using CandidateIdSort = cub::BlockRadixSort<std::uint32_t, threads, sort_items, int>;
using SuppressedScan = cub::BlockScan<unsigned, threads>;
union BatchSortStorage {
    SuppressedSort::TempStorage suppressed;
    CandidateIdSort::TempStorage ids;
    SuppressedScan::TempStorage compact;
};
// All finite FP32 scores are valid inputs here, including negative values.
// Canonicalize signed zero because the reference compares -0 == +0.
__device__ std::uint64_t suppressed_key(const DetectionCandidate& candidate) {
    const auto bits = __float_as_uint(candidate.score == 0.0f ? 0.0f : candidate.score);
    const auto ascending = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    return (std::uint64_t(~ascending) << 32) | candidate.candidate_id;
}
__device__ bool finite_box(const float* box) {
    for (int i = 0; i < 4; ++i) if (!math::finite(box[i])) return false;
    return true;
}
__device__ bool finite_track(const DeviceTrack& t) {
    for (float v : t.x) if (!math::finite(v)) return false;
    for (float v : t.P) if (!math::finite(v)) return false;
    if (!math::finite(t.score_sum) || !math::finite(t.h_ema) ||
        !math::finite(t.observed_score) || !math::finite(t.association_score)) return false;
    for (std::uint32_t i = 0; i < t.history_count; ++i)
        if (!math::finite(t.history[i].cx) || !math::finite(t.history[i].cy) ||
            !math::finite(t.history[i].h)) return false;
    return true;
}
__device__ bool finite_detection(const DeviceBatchDetection& d) {
    float z[4];
    math::box_to_state(d.xyxy, z);
    for (float v : z) if (!math::finite(v)) return false;
    return finite_box(d.xyxy) && math::finite(d.observed_score) && math::finite(d.association_score);
}
__device__ void push_history(DeviceTrack& t, const float* box) {
    float z[4];
    math::box_to_state(box, z);
    if (t.history_count == kBridgeHistory) {
        for (std::uint32_t i = 1; i < kBridgeHistory; ++i) t.history[i - 1] = t.history[i];
        --t.history_count;
    }
    t.history[t.history_count++] = {z[0], z[1], z[3]};
}
__device__ void confirm(DeviceTrack& t, const TrackerConfig& c) {
    if (t.lifecycle == TrackLifecycle::tentative && t.hit_streak >= c.life.confirm_streak &&
        t.score_sum / static_cast<float>(t.hit_streak) >= c.life.confirm_score)
        t.lifecycle = TrackLifecycle::confirmed;
}
__device__ float velocity_x(const DeviceTrack& t) {
    return math::regression_velocity(t.history[0].cx, t.history[1].cx, t.history[2].cx, t.history[3].cx);
}
__device__ float velocity_y(const DeviceTrack& t) {
    return math::regression_velocity(t.history[0].cy, t.history[1].cy, t.history[2].cy, t.history[3].cy);
}
__device__ void numeric(TrackingCudaScratch* s) {
    atomicOr(reinterpret_cast<unsigned int*>(&s->error), static_cast<unsigned int>(Error::numeric));
}
__global__ void predict_kernel(const DeviceTrackState* in, const DeviceMotionContext* motion,
                               const FrameId* frame, const TrackerConfig* config, DevicePredicted* out,
                               TrackingCudaScratch* s, const bool* run_stage) {
    if (run_stage && !*run_stage) return;
    const auto& c = *config;
    for (unsigned i = threadIdx.x; i < in->count; i += blockDim.x) {
        auto& t = s->tracks[i];
        t = in->tracks[i];
        s->status[i] = -1;
        if (!finite_track(t)) continue;
        if (motion->status == MotionStatus::valid) { t.x[0] += motion->tx; t.x[1] += motion->ty; }
        math::kalman_predict(t.x, t.P, c.kalman.std_position, c.kalman.std_velocity);
        ++t.age;
        if (!finite_track(t)) continue;
        s->status[i] = 0;
        if (t.age > c.life.track_buffer || t.x[3] < 1 || t.x[2] <= 0) continue;
        float box[4]; math::state_to_box(t.x, box);
        s->status[i] = finite_box(box) ? 1 : -1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        out->epoch = in->epoch; out->source_committed_frame = in->committed_frame;
        out->frame = *frame; out->motion = *motion; out->next_uid = in->next_uid;
        out->count = out->prior_count = 0; out->error = Error::none;
        for (unsigned i = 0; i < in->count; ++i) {
            if (s->status[i] < 0) { out->error = Error::numeric; break; }
            if (!s->status[i]) continue;
            const auto& t = s->tracks[i];
            out->tracks[out->count++] = t;
            if (t.lifecycle == TrackLifecycle::confirmed) {
                auto& p = out->priors[out->prior_count++];
                p.uid = t.uid; p.class_id = t.class_id; math::state_to_box(t.x, p.xyxy);
            }
        }
    }
}
__device__ DeviceBatchDetection convert(const DetectionCandidate& d, bool birth, float association) {
    return {{d.x1, d.y1, d.x2, d.y2}, d.score, association, d.class_id, d.candidate_id, true, birth};
}
__global__ void batch_kernel(const DetectionCandidates* input, const DevicePredicted* predicted,
                             const TrackerConfig* config, DeviceBatch* out, TrackingCudaScratch* s, const bool* run_stage) {
    if (run_stage && !*run_stage) return;
    __shared__ BatchSortStorage sort_storage;
    __shared__ unsigned input_unsorted;
    const auto& h = input->header;
    const auto* candidate_rows = reinterpret_cast<const DetectionCandidate*>(&input->rows);
    const auto& c = *config;
    if (threadIdx.x == 0) {
        input_unsorted = 0;
        out->epoch = h.epoch; out->frame = static_cast<FrameId>(h.frame); out->count = 0;
        unsigned error = 0;
        if (h.epoch != predicted->epoch || out->frame != predicted->frame || h.done != kPostprocessDone ||
            h.count > kPostprocessCapacity) error |= static_cast<unsigned>(Error::protocol);
        if (h.flags & kPostprocessOverflow) error |= static_cast<unsigned>(Error::overflow);
        if (h.flags & ~kPostprocessOverflow) error |= static_cast<unsigned>(Error::producer);
        out->error = static_cast<Error>(error);
    }
    __syncthreads();
    if (out->error != Error::none) return;
    for (unsigned i = threadIdx.x; i < h.count; i += blockDim.x) {
        const auto& d = candidate_rows[i];
        s->status[i] = math::finite(d.score) && math::finite(d.x1) && math::finite(d.y1) &&
                       math::finite(d.x2) && math::finite(d.y2);
        if (i && suppressed_key(candidate_rows[i - 1]) > suppressed_key(d)) atomicOr(&input_unsorted, 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) for (unsigned i = 0; i < h.count; ++i)
        if (!s->status[i]) out->error = Error::numeric;
    __syncthreads();
    if (out->error != Error::none) return;
    if (c.continuation.enabled && predicted->prior_count && c.continuation.max_candidates) {
        if (!input_unsorted) {
            unsigned selected[sort_items], ranks[sort_items];
            for (unsigned item = 0; item < sort_items; ++item) {
                const unsigned i = threadIdx.x * sort_items + item;
                selected[item] = i < h.count && !(candidate_rows[i].flags & kDetectionKept) &&
                                 (candidate_rows[i].flags & kDetectionSuppressed);
            }
            // Native F04 already delivers score/ID order. Verify that order
            // on device, then compact its suppressed subsequence without
            // sorting it again. General callers still use the radix path.
            SuppressedScan(sort_storage.compact).ExclusiveSum(selected, ranks);
            for (unsigned item = 0; item < sort_items; ++item)
                if (selected[item]) s->order[ranks[item]] = static_cast<int>(threadIdx.x * sort_items + item);
        } else {
            std::uint64_t keys[sort_items];
            int indices[sort_items];
            for (unsigned item = 0; item < sort_items; ++item) {
                const unsigned i = threadIdx.x * sort_items + item;
                keys[item] = ~std::uint64_t{0}; indices[item] = -1;
                if (i < h.count) {
                    const auto& d = candidate_rows[i];
                    if (!(d.flags & kDetectionKept) && (d.flags & kDetectionSuppressed)) {
                        keys[item] = suppressed_key(d); indices[item] = static_cast<int>(i);
                    }
                }
            }
            // Stable blocked input preserves original row order for equal
            // score/ID keys. Striped output makes the order stores coalesced.
            SuppressedSort(sort_storage.suppressed).SortBlockedToStriped(keys, indices);
            for (unsigned item = 0; item < sort_items; ++item) {
                const unsigned rank = threadIdx.x + item * threads;
                if (rank < h.count) s->order[rank] = indices[item];
            }
        }
        for (unsigned i = threadIdx.x; i < h.count; i += blockDim.x) {
            const auto& d = candidate_rows[i];
            if ((d.flags & kDetectionKept) || !(d.flags & kDetectionSuppressed)) continue;
            s->matched[i] = 0; s->claims[i] = 1;
            if (d.score < c.continuation.min_score) continue;
            const float box[4] = {d.x1, d.y1, d.x2, d.y2};
            for (unsigned j = 0; j < predicted->prior_count; ++j) {
                const float overlap = math::iou(box, predicted->priors[j].xyxy);
                if (!math::finite(overlap)) { s->matched[i] = -1; break; }
                if (overlap >= c.continuation.prior_iou) s->matched[i] = 1;
            }
            if (s->matched[i] != 1) continue;
            for (unsigned j = 0; j < h.count; ++j) {
                const auto& k = candidate_rows[j];
                if (!(k.flags & kDetectionKept)) continue;
                const float kb[4] = {k.x1, k.y1, k.x2, k.y2};
                const float overlap = math::iou(box, kb);
                if (!math::finite(overlap)) { s->claims[i] = -1; break; }
                if (overlap > c.continuation.dedup_iou) s->claims[i] = 0;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned suppressed = 0;
        for (unsigned i = 0; i < h.count; ++i) {
            const auto& d = candidate_rows[i];
            if (d.flags & kDetectionKept) {
                if (d.score >= c.batch.track_thresh) out->rows[out->count++] = convert(d, true, d.score);
            } else if (d.flags & kDetectionSuppressed) ++suppressed;
        }
        const unsigned kept_count = out->count;
        unsigned accepted = 0;
        // Apply precomputed gates only when the reference visits this candidate.
        if (c.continuation.enabled && predicted->prior_count) {
            for (unsigned i = 0; i < suppressed; ++i) {
                if (accepted == c.continuation.max_candidates) break;
                const int index = s->order[i]; const auto& d = candidate_rows[index];
                if (d.score < c.continuation.min_score) continue;
                if (s->matched[index] < 0) out->error = Error::numeric;
                if (s->matched[index] != 1 || out->error != Error::none) continue;
                if (s->claims[index] < 0) out->error = Error::numeric;
                bool dedup = s->claims[index] != 0;
                const float box[4] = {d.x1, d.y1, d.x2, d.y2};
                for (unsigned j = kept_count; j < out->count; ++j) {
                    const float overlap = math::iou(box, out->rows[j].xyxy);
                    if (!math::finite(overlap)) out->error = Error::numeric;
                    if (overlap > c.continuation.dedup_iou) dedup = false;
                }
                if (dedup) {
                    out->rows[out->count++] = convert(d, false, math::min(d.score, c.life.new_track_thresh)); ++accepted;
                }
            }
        }
        if (out->error != Error::none) out->count = 0;
    }
    __syncthreads();
    if (out->count <= 1) return;
    // Small batches avoid the fixed 512-item radix cost. At most 32 rows
    // participate here; larger batches use the stable block radix below.
    if (out->count <= 32) {
        if (threadIdx.x < out->count) {
            const unsigned i = threadIdx.x;
            const auto d = out->rows[i];
            unsigned rank = 0;
            for (unsigned j = 0; j < out->count; ++j)
                if (out->rows[j].candidate_id < d.candidate_id ||
                    (out->rows[j].candidate_id == d.candidate_id && j < i)) ++rank;
            s->batch_rows[rank] = d;
        }
        __syncthreads();
        if (threadIdx.x < out->count) out->rows[threadIdx.x] = s->batch_rows[threadIdx.x];
        return;
    }
    std::uint32_t ids[sort_items];
    int indices[sort_items];
    for (unsigned item = 0; item < sort_items; ++item) {
        const unsigned i = threadIdx.x * sort_items + item;
        ids[item] = i < out->count ? out->rows[i].candidate_id : ~std::uint32_t{0};
        indices[item] = static_cast<int>(i);
    }
    // A valid UINT32_MAX ID precedes padding because the radix sort is stable
    // and every valid row precedes padding in the blocked input sequence.
    CandidateIdSort(sort_storage.ids).SortBlockedToStriped(ids, indices);
    for (unsigned item = 0; item < sort_items; ++item) {
        const unsigned rank = threadIdx.x + item * threads;
        if (rank < out->count) s->batch_rows[rank] = out->rows[indices[item]];
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < out->count; i += blockDim.x) out->rows[i] = s->batch_rows[i];
}

__global__ void update_kernel(const DevicePredicted* predicted, const DeviceBatch* batch,
                              const DeviceMotionContext* motion, const TrackerConfig* config,
                              DeviceTrackState* next, DeviceTrackingRows* out, TrackingCudaScratch* s, const bool* run_stage) {
    if (run_stage && !*run_stage) return;
    __shared__ unsigned birth_unsorted;
    const auto& c = *config;
    const unsigned n = predicted->count, m = batch->count;
    if (threadIdx.x == 0) {
        birth_unsorted = 0;
        unsigned error = static_cast<unsigned>(predicted->error) | static_cast<unsigned>(batch->error);
        if (batch->epoch != predicted->epoch || motion->epoch != predicted->epoch ||
            predicted->motion.epoch != predicted->epoch || batch->frame != predicted->frame ||
            motion->frame != predicted->frame || predicted->motion.frame != predicted->frame ||
            predicted->source_committed_frame != predicted->frame - 1) error |= static_cast<unsigned>(Error::protocol);
        s->error = static_cast<Error>(error); out->count = 0; out->error = s->error;
    }
    __syncthreads();
    if (s->error != Error::none) return;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        s->tracks[i] = predicted->tracks[i]; s->matched[i] = -1; s->removed[i] = false;
        if (!finite_track(s->tracks[i])) numeric(s);
    }
    for (unsigned j = threadIdx.x; j < m; j += blockDim.x) {
        s->used[j] = false;
        if (j && batch->rows[j - 1].candidate_id > batch->rows[j].candidate_id)
            atomicOr(&birth_unsorted, 1u);
        if (!finite_detection(batch->rows[j])) numeric(s);
    }
    __syncthreads();
    if (s->error != Error::none) { if (threadIdx.x == 0) out->error = s->error; return; }
    for (int stage = 0; stage < 4; ++stage) {
        for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
            auto& proposal = s->proposals[i]; proposal = {-1, 0, 0};
            const auto& t = s->tracks[i];
            if (s->matched[i] >= 0 || (t.lifecycle == TrackLifecycle::tentative) != (stage == 2)) continue;
            float box[4]; math::state_to_box(t.x, box);
            int best = -1, second = -1; float p1 = 0, p2 = 0;
            for (unsigned j = 0; j < m; ++j) {
                const auto& d = batch->rows[j]; const float score = d.association_score;
                const bool in_range = stage == 0 ? score >= c.assoc.high_thresh :
                    stage == 1 ? score >= c.assoc.mid_thresh && score < c.assoc.high_thresh :
                    stage == 2 ? score >= c.assoc.mid_thresh : score >= c.batch.track_thresh && score < c.assoc.mid_thresh;
                if (s->used[j] || !in_range) continue;
                const float overlap = math::iou(box, d.xyxy);
                if (!math::finite(overlap)) { numeric(s); continue; }
                if (overlap <= 0) continue;
                const float h = d.xyxy[3] - d.xyxy[1], w = d.xyxy[2] - d.xyxy[0];
                const float cost = math::association_cost(overlap, t.x[3], h, c.assoc.lambda, c.assoc.stability_cost_w);
                const float p = math::association_value(cost, w, h, c.assoc.lambda);
                if (!math::finite(cost) || !math::finite(p)) { numeric(s); continue; }
                if (cost > (stage == 3 ? c.assoc.stage2_match_thresh : c.assoc.match_thresh)) continue;
                if (best < 0 || p > p1 || (p == p1 && d.candidate_id < batch->rows[best].candidate_id)) {
                    second = best; p2 = p1; best = j; p1 = p;
                } else if (second < 0 || p > p2 || (p == p2 && d.candidate_id < batch->rows[second].candidate_id)) {
                    second = j; p2 = p;
                }
            }
            if (best < 0) continue;
            const auto& d = batch->rows[best];
            const bool has_second = c.assoc.top_k > 1 && second >= 0;
            const float bid = math::auction_bid(p1, has_second ? p2 : 0.0f, has_second,
                c.assoc.auction_eps, c.assoc.stability_bid_w, t.x[3], d.xyxy[3] - d.xyxy[1]);
            if (!math::finite(bid)) numeric(s);
            proposal = {best, p1, bid};
        }
        __syncthreads();
        if (s->error != Error::none) { if (threadIdx.x == 0) out->error = s->error; return; }
        for (unsigned j = threadIdx.x; j < m; j += blockDim.x) {
            int winner = -1;
            for (unsigned i = 0; i < n; ++i) {
                const auto& p = s->proposals[i];
                if (p.detection != static_cast<int>(j)) continue;
                if (winner < 0 || p.bid > s->proposals[winner].bid ||
                    (p.bid == s->proposals[winner].bid && (p.p > s->proposals[winner].p ||
                     (p.p == s->proposals[winner].p && s->tracks[i].uid < s->tracks[winner].uid)))) winner = i;
            }
            if (winner >= 0) { s->matched[winner] = j; s->used[j] = true; }
        }
        __syncthreads();
    }
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        auto& t = s->tracks[i];
        if (s->matched[i] < 0) {
            if (t.lifecycle == TrackLifecycle::tentative) s->removed[i] = true;
            else { t.hit_streak = 0; t.score_sum = 0; }
            continue;
        }
        const auto& d = batch->rows[s->matched[i]];
        math::kalman_update(t.x, t.P, d.xyxy, c.kalman.std_position, c.kalman.r_scale);
        t.age = 0; ++t.hit_streak; t.score_sum += d.observed_score;
        t.observed_score = d.observed_score; t.association_score = d.association_score;
        t.class_id = d.class_id; t.last_observed_frame = predicted->frame;
        push_history(t, d.xyxy); t.h_ema = math::height_ema(t.h_ema, d.xyxy[3] - d.xyxy[1]);
        confirm(t, c);
        if (!finite_track(t)) numeric(s);
    }
    __syncthreads();
    if (s->error != Error::none) { if (threadIdx.x == 0) out->error = s->error; return; }
    if (c.bridge.enabled) {
        for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
            s->claims[i] = -1;
            const auto& t = s->tracks[i];
            if (s->matched[i] < 0 || t.hit_streak != c.bridge.trigger_streak ||
                predicted->frame - t.birth_frame != c.bridge.trigger_streak - 1 || t.output_id != t.uid) continue;
            int best = -1, second = -1; float d1 = 0, d2 = 0;
            for (unsigned j = 0; j < n; ++j) {
                const auto& lost = s->tracks[j];
                if (s->matched[j] >= 0 || lost.lifecycle != TrackLifecycle::confirmed ||
                    lost.age < c.bridge.min_lost || lost.age > c.life.track_buffer ||
                    lost.last_observed_frame >= t.birth_frame || lost.history_count < kBridgeHistory) continue;
                const auto& l = lost.history[lost.history_count - 1]; const auto& first = t.history[0];
                const float distance = math::bridge_distance(l.cx, l.cy, first.cx, first.cy,
                    velocity_x(lost), velocity_y(lost), velocity_x(t), velocity_y(t),
                    static_cast<float>(t.birth_frame - lost.last_observed_frame), lost.h_ema, t.h_ema, c.bridge.speed_norm);
                const float ratio = lost.h_ema / t.h_ema;
                if (!math::finite(distance) || !math::finite(ratio)) { numeric(s); continue; }
                if (distance > c.bridge.max_distance || ratio < c.bridge.height_ratio_lo || ratio > c.bridge.height_ratio_hi) continue;
                if (best < 0 || distance < d1 || (distance == d1 && lost.uid < s->tracks[best].uid)) {
                    second = best; d2 = d1; best = j; d1 = distance;
                } else if (second < 0 || distance < d2 || (distance == d2 && lost.uid < s->tracks[second].uid)) {
                    second = j; d2 = distance;
                }
            }
            if (best >= 0 && !(second >= 0 && d2 - d1 < c.bridge.margin)) s->claims[i] = best;
        }
        __syncthreads();
        if (s->error != Error::none) { if (threadIdx.x == 0) out->error = s->error; return; }
        // Commit claims in lost-track order, matching the reference.
        if (threadIdx.x == 0) for (unsigned j = 0; j < n; ++j) {
            int winner = -1;
            for (unsigned i = 0; i < n; ++i) {
                if (s->claims[i] != static_cast<int>(j)) continue;
                if (winner < 0 || s->tracks[i].observed_score > s->tracks[winner].observed_score ||
                    (s->tracks[i].observed_score == s->tracks[winner].observed_score &&
                     s->tracks[i].uid < s->tracks[winner].uid)) winner = i;
            }
            if (winner >= 0) { s->tracks[winner].output_id = s->tracks[j].output_id; s->removed[j] = true; }
        }
        __syncthreads();
    }
    // Stable compaction and birth UID assignment use one fixed traversal.
    if (threadIdx.x != 0) return;
    next->epoch = predicted->epoch; next->committed_frame = predicted->frame;
    next->next_uid = predicted->next_uid; next->count = 0;
    for (unsigned i = 0; i < n; ++i) if (!s->removed[i]) next->tracks[next->count++] = s->tracks[i];
    // The session batch is already sorted. Keep the original stable insertion
    // fallback for arbitrary public inputs, without reserving radix shared
    // storage on every hot-path update merely for this uncommon branch.
    if (birth_unsorted) {
        for (unsigned j = 0; j < m; ++j) s->order[j] = j;
        for (unsigned j = 1; j < m; ++j) {
            const int v = s->order[j]; unsigned k = j;
            while (k > 0 && batch->rows[v].candidate_id < batch->rows[s->order[k - 1]].candidate_id) {
                s->order[k] = s->order[k - 1]; --k;
            }
            s->order[k] = v;
        }
    }
    for (unsigned k = 0; k < m; ++k) {
        const int j = birth_unsorted ? s->order[k] : static_cast<int>(k); const auto& d = batch->rows[j];
        if (s->used[j] || !d.can_birth || d.observed_score < c.life.new_track_thresh) continue;
        if (next->count == kTrackCapacity) { out->error = Error::overflow; return; }
        DeviceTrack t{}; t.lifecycle = TrackLifecycle::tentative;
        t.uid = next->next_uid++; t.output_id = t.uid;
        math::kalman_init(d.xyxy, t.x, t.P, c.kalman.std_position, c.kalman.std_velocity);
        t.hit_streak = 1; t.score_sum = t.observed_score = d.observed_score;
        t.association_score = d.association_score; t.class_id = d.class_id;
        t.birth_frame = t.last_observed_frame = predicted->frame;
        push_history(t, d.xyxy); t.h_ema = d.xyxy[3] - d.xyxy[1]; confirm(t, c);
        if (!finite_track(t)) { out->error = Error::numeric; return; }
        next->tracks[next->count++] = t;
    }
    for (unsigned i = 0; i < next->count; ++i) {
        const auto& t = next->tracks[i];
        if (t.lifecycle != TrackLifecycle::confirmed || t.age != 0) continue;
        DeviceTrackRow row{}; row.uid = t.uid; row.output_id = t.output_id;
        row.class_id = t.class_id; row.flags = 1; math::state_to_box(t.x, row.xyxy);
        row.observed_score = t.observed_score; row.association_score = t.association_score;
        if (!finite_box(row.xyxy)) { out->error = Error::numeric; out->count = 0; return; }
        // Keep checking boxes after capacity is reached: NUMERIC takes precedence.
        if (out->count < packet_capacity) out->rows[out->count] = row;
        ++out->count;
    }
    if (out->count > packet_capacity) { out->error = Error::overflow; out->count = 0; }
}
void explicit_stream(cudaStream_t stream) {
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread)
        throw std::invalid_argument("tracking requires an explicit non-default stream");
}
void launch_check() {
    const auto result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(std::string("tracking launch: ") + cudaGetErrorString(result));
}
} // namespace
void enqueue_predict_tracks(const DeviceTrackState* committed, const DeviceMotionContext* motion,
                            const FrameId* frame, const TrackerConfig* config, DevicePredicted* output,
                            TrackingCudaScratch* scratch, cudaStream_t stream, const bool* run_stage) {
    explicit_stream(stream); predict_kernel<<<1, threads, 0, stream>>>(committed, motion, frame, config, output, scratch, run_stage); launch_check();
}
void enqueue_assemble_detection_batch(const DetectionCandidates* candidates, const DevicePredicted* predicted,
                                     const TrackerConfig* config, DeviceBatch* output,
                                     TrackingCudaScratch* scratch, cudaStream_t stream, const bool* run_stage) {
    explicit_stream(stream); batch_kernel<<<1, threads, 0, stream>>>(candidates, predicted, config, output, scratch, run_stage); launch_check();
}
void enqueue_associate_and_update(const DevicePredicted* predicted, const DeviceBatch* batch,
                                 const DeviceMotionContext* motion, const TrackerConfig* config,
                                 DeviceTrackState* next, DeviceTrackingRows* rows,
                                 TrackingCudaScratch* scratch, cudaStream_t stream, const bool* run_stage) {
    explicit_stream(stream); update_kernel<<<1, threads, 0, stream>>>(predicted, batch, motion, config, next, rows, scratch, run_stage); launch_check();
}

DeviceTrack to_device(const Track& v) {
    DeviceTrack out{};
    out.uid = v.uid;
    out.output_id = v.output_id;
    out.lifecycle = v.lifecycle;
    out.age = v.age;
    out.hit_streak = v.hit_streak;
    out.score_sum = v.score_sum;
    out.class_id = v.class_id;
    out.observed_score = v.observed_score;
    out.association_score = v.association_score;
    out.birth_frame = v.birth_frame;
    out.last_observed_frame = v.last_observed_frame;
    out.history_count = v.history_count;
    out.h_ema = v.h_ema;
    std::copy_n(std::begin(v.x), 8, std::begin(out.x));
    std::copy_n(std::begin(v.P), 64, std::begin(out.P));
    for (unsigned i = 0; i < kBridgeHistory; ++i) out.history[i] = {v.history[i].cx, v.history[i].cy, v.history[i].h};
    return out;
}

Track to_host(const DeviceTrack& v) {
    Track out{};
    out.uid = v.uid;
    out.output_id = v.output_id;
    out.lifecycle = v.lifecycle;
    out.age = v.age;
    out.hit_streak = v.hit_streak;
    out.score_sum = v.score_sum;
    out.class_id = v.class_id;
    out.observed_score = v.observed_score;
    out.association_score = v.association_score;
    out.birth_frame = v.birth_frame;
    out.last_observed_frame = v.last_observed_frame;
    out.history_count = v.history_count;
    out.h_ema = v.h_ema;
    std::copy_n(std::begin(v.x), 8, std::begin(out.x));
    std::copy_n(std::begin(v.P), 64, std::begin(out.P));
    for (unsigned i = 0; i < kBridgeHistory; ++i) out.history[i] = {v.history[i].cx, v.history[i].cy, v.history[i].h};
    return out;
}

DeviceMotionContext to_device(const MotionContext& v) {
    return {v.epoch, v.frame, v.reference_frame, v.status, v.peak_x, v.peak_y, v.pcr, v.confidence, v.tx, v.ty};
}

MotionContext to_host(const DeviceMotionContext& v) {
    return {v.epoch, v.frame, v.reference_frame, v.status, v.peak_x, v.peak_y, v.pcr, v.confidence, v.tx, v.ty};
}

DeviceTrackState to_device(const TrackState& v) {
    DeviceTrackState out{};
    out.epoch = v.epoch; out.committed_frame = v.committed_frame; out.next_uid = v.next_uid;
    out.count = static_cast<unsigned>(v.tracks.size());
    for (unsigned i = 0; i < out.count; ++i) out.tracks[i] = to_device(v.tracks[i]);
    return out;
}
TrackState to_host(const DeviceTrackState& v) {
    TrackState out{v.epoch, v.committed_frame, v.next_uid, {}};
    for (unsigned i = 0; i < v.count; ++i) out.tracks.push_back(to_host(v.tracks[i]));
    return out;
}
DevicePredicted to_device(const PredictedState& v) {
    DevicePredicted out{};
    out.epoch = v.epoch; out.source_committed_frame = v.source_committed_frame; out.frame = v.frame;
    out.motion = to_device(v.motion); out.next_uid = v.next_uid; out.error = v.error;
    out.count = static_cast<unsigned>(v.tracks.size()); out.prior_count = static_cast<unsigned>(v.priors.size());
    for (unsigned i = 0; i < out.count; ++i) out.tracks[i] = to_device(v.tracks[i]);
    for (unsigned i = 0; i < out.prior_count; ++i) {
        out.priors[i].uid = v.priors[i].uid; out.priors[i].class_id = v.priors[i].class_id;
        std::copy_n(v.priors[i].xyxy.begin(), 4, out.priors[i].xyxy);
    }
    return out;
}
PredictedState to_host(const DevicePredicted& v) {
    PredictedState out;
    out.epoch = v.epoch; out.source_committed_frame = v.source_committed_frame; out.frame = v.frame;
    out.motion = to_host(v.motion); out.next_uid = v.next_uid; out.error = v.error;
    for (unsigned i = 0; i < v.count; ++i) out.tracks.push_back(to_host(v.tracks[i]));
    for (unsigned i = 0; i < v.prior_count; ++i) {
        PredictedPrior p; p.uid = v.priors[i].uid; p.class_id = v.priors[i].class_id;
        std::copy_n(v.priors[i].xyxy, 4, p.xyxy.begin()); out.priors.push_back(p);
    }
    return out;
}
DeviceBatch to_device(const DetectionBatch& v) {
    DeviceBatch out{};
    out.epoch = v.epoch; out.frame = v.frame; out.error = v.error; out.count = static_cast<unsigned>(v.rows.size());
    for (unsigned i = 0; i < out.count; ++i) {
        const auto& d = v.rows[i]; auto& r = out.rows[i];
        std::copy_n(d.xyxy.begin(), 4, r.xyxy); r.observed_score = d.observed_score; r.association_score = d.association_score;
        r.class_id = d.class_id; r.candidate_id = d.candidate_id; r.can_continue = d.can_continue; r.can_birth = d.can_birth;
    }
    return out;
}
DetectionBatch to_host(const DeviceBatch& v) {
    DetectionBatch out; out.epoch = v.epoch; out.frame = v.frame; out.error = v.error;
    for (unsigned i = 0; i < v.count; ++i) {
        const auto& d = v.rows[i]; BatchDetection r;
        std::copy_n(d.xyxy, 4, r.xyxy.begin()); r.observed_score = d.observed_score; r.association_score = d.association_score;
        r.class_id = d.class_id; r.candidate_id = d.candidate_id; r.can_continue = d.can_continue; r.can_birth = d.can_birth;
        out.rows.push_back(r);
    }
    return out;
}
UpdateResult to_host(const DeviceTrackState& next, const DeviceTrackingRows& rows) {
    UpdateResult out; out.error = rows.error;
    if (out.error == Error::none) out.next = to_host(next);
    for (unsigned i = 0; i < rows.count; ++i) {
        const auto& d = rows.rows[i]; TrackRow r;
        r.uid = d.uid; r.output_id = d.output_id; r.class_id = d.class_id; r.flags = d.flags;
        std::copy_n(d.xyxy, 4, r.xyxy.begin()); r.observed_score = d.observed_score; r.association_score = d.association_score;
        out.rows.push_back(r);
    }
    return out;
}
} // namespace saccade

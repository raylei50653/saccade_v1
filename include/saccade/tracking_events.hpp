#pragma once

#include "saccade/tracking.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace saccade {

inline constexpr const char* kTrackingEventsSchema = "tracking-events/1";
inline constexpr std::uint32_t kNoEventCandidate = 0xffffffffU;

enum class TrackingEventKind : std::uint32_t {
    frame_begin = 1, predict, batch_ready, candidate, associate, birth, confirm,
    lost, remove, relink, emit, frame_commit, frame_fail, frame_skipped,
};
enum class TrackingEventPhase : std::uint32_t {
    frame = 0, predict = 1, batch = 2, update = 3, bridge = 4, birth = 5,
    emit = 6, commit = 7,
};
enum class TrackingEventStage : std::uint32_t {
    control = 0, f08a = 1, f04b = 2, f08b = 3, f09 = 4,
};
enum class TrackingEventReason : std::uint32_t {
    none = 0, unmatched = 1, expired = 2, invalid_geometry = 3,
};

// Logical values. Only the payload selected by kind is encoded; struct padding,
// unused Track history, addresses, timing and diagnostics are never encoded.
// before/after are actual decision-point snapshots, never inferred from packets.
struct TrackingEvent {
    std::uint64_t source = 0;
    Epoch epoch = 0;
    FrameId frame = no_frame;
    TrackingEventKind kind = TrackingEventKind::frame_begin;
    TrackingEventPhase phase = TrackingEventPhase::frame;
    TrackingEventStage stage = TrackingEventStage::control;
    Track before{}, after{};
    BatchDetection detection{};
    TrackRow row{};
    std::uint64_t related_uid = 0; // relink donor; receiver is before/after.uid
    std::uint32_t association_stage = 0; // 1..4 for associate, otherwise unused
    TrackingEventReason reason = TrackingEventReason::none;
    FrameId committed_frame = no_frame; // frame_begin/terminal
    std::uint32_t count = 0; // batch_ready: candidates; frame_commit: emit rows
    Error error = Error::none; // frame_fail/frame_skipped; sticky first error
};

class TrackingEventObserver {
public:
    virtual ~TrackingEventObserver() = default;
    virtual void record(const TrackingEvent& event) = 0;
};
class TrackingEventRecorder final : public TrackingEventObserver {
public:
    void record(const TrackingEvent& event) override { events.push_back(event); }
    std::vector<TrackingEvent> events;
};

// Stable little-endian schema-tagged encoding with per-event payload lengths.
// Decode rejects unknown enums, malformed lengths, noncanonical bools,
// nonfinite values, invalid history bounds and trailing bytes.
std::vector<std::uint8_t> encode_tracking_events(std::span<const TrackingEvent> events);
std::vector<TrackingEvent> decode_tracking_events(std::span<const std::uint8_t> bytes);
std::string tracking_events_hash(std::span<const TrackingEvent> events);

struct TrackingEventComparison {
    bool equal = true;
    std::size_t event_index = 0;
    std::string field;
    std::string left, right;
};
TrackingEventComparison compare_tracking_events(std::span<const TrackingEvent> left,
                                                std::span<const TrackingEvent> right);
struct TrackingEventValidation {
    bool valid = true;
    std::size_t event_index = 0;
    std::size_t frames = 0;
    std::string message;
};
// Complete trace from initially empty source/epoch sessions, consecutive frames
// starting at zero. Enforces lifecycle, candidate eligibility/unique ownership,
// identity transfer, required predict/confirm/emit/terminal events, atomic
// commit, and sticky failure. Checks declared association stage score ranges;
// Numeric x/P snapshots are chained bitwise but Kalman/GMC arithmetic is not
// recomputed; numerical auction and Bridge geometry/ranking are not re-solved.
TrackingEventValidation validate_tracking_events(std::span<const TrackingEvent> events,
                                                 const TrackerConfig& config = {});
// Independently bind one source/epoch's complete event prefix to every byte of
// its ResultPacket stream. Reconstructs OK/FAILED/SKIPPED packets, including
// sticky first-failure identity and zero reserved/unused bytes. This does not
// replace validate_tracking_events(events, matching_config) lifecycle checks.
TrackingEventValidation validate_tracking_event_packets(
    std::span<const TrackingEvent> events, std::span<const std::uint8_t> packets);

} // namespace saccade

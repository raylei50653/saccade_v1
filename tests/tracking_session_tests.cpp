#include "saccade/tracking_session.hpp"
#include "saccade/file_io.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/sha256.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace saccade {
struct TrackingSessionTestAccess {
    static void enqueue(TrackingSession& s, std::span<const std::uint8_t> jpeg, FrameId k,
                        std::uint32_t flags = 0, bool tail = false) { s.enqueue_for_test(jpeg, k, flags, tail); }
    static detail::TrackingSessionDiagnostic diagnostic(const TrackingSession& s) { return s.diagnostic_for_test(); }
};
}
namespace {
using namespace saccade;
std::uint64_t checks = 0;
std::string context;
void check(bool value, const std::string& what) {
    ++checks;
    if (!value) throw std::runtime_error(context + ": " + what);
}
std::string jpeg(const std::filesystem::path& sequence, FrameId frame) {
    std::ostringstream name;
    name << std::setfill('0') << std::setw(6) << frame + 1 << ".jpg";
    return read_text_file(sequence / "img1" / name.str());
}
std::span<const std::uint8_t> span(const std::string& s) {
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}
using Stream = std::vector<PacketBytes>;
Stream run(const std::string& engine, const std::filesystem::path& sequence, TrackingSessionConfig cfg,
           std::size_t count, detail::TrackingSessionDiagnostic* diagnostic = nullptr) {
    TrackingSession session(engine, cfg);
    check(session.identity().find("tracker_config_sha256=" + sha256_hex(canonical_tracker_config(cfg.tracker))) != std::string::npos,
          "tracker identity hash");
    check(session.identity().find("motion_profile=v1-motion/2\ntracker_profile=v1-tracker/1\npacket_profile=v1") != std::string::npos,
          "identity profiles");
    Stream out;
    const auto receive = [&] {
        auto r = session.receive();
        check(r.packet.status == ResultStatus::ok && r.packet.frame == static_cast<FrameId>(out.size()), "OK identity");
        check(r.bytes == encode_packet(r.packet), "raw packet canonical bytes");
        out.push_back(r.bytes);
    };
    for (std::size_t k = 0; k < count; ++k) {
        const auto bytes = jpeg(sequence, k);
        if (session.pending() == cfg.result_slots) receive();
        session.enqueue(span(bytes), k);
    }
    session.finish();
    check(session.state() == TrackingSessionState::Finished, "EOF finished");
    while (session.pending()) receive();
    if (diagnostic) *diagnostic = TrackingSessionTestAccess::diagnostic(session);
    session.finish(); session.close(); session.close();
    check(session.state() == TrackingSessionState::Closed, "repeatable close");
    return out;
}
std::size_t different_byte(const PacketBytes& a, const PacketBytes& b) {
    return std::mismatch(a.begin(), a.end(), b.begin()).first - a.begin();
}
void compare(const Stream& a, const Stream& b, const char* label) {
    check(a.size() == b.size(), std::string(label) + " frame count");
    for (std::size_t k = 0; k < a.size(); ++k) {
        const auto byte = different_byte(a[k], b[k]);
        check(byte == packet_bytes, std::string(label) + " first differing frame=" + std::to_string(k) +
              " byte=" + std::to_string(byte));
    }
}
void diagnose(const std::string& engine, const std::filesystem::path& sequence, TrackingSessionConfig cfg,
              std::size_t frame, const DetectionRecords& detections, const MotionRecords& motion) {
    detail::TrackingSessionDiagnostic d;
    (void)run(engine, sequence, cfg, frame + 1, &d);
    const auto* actual = reinterpret_cast<const unsigned char*>(&d.candidates);
    const auto* expected = reinterpret_cast<const unsigned char*>(&detections.frames[frame]);
    const auto byte = static_cast<std::size_t>(std::mismatch(actual, actual + sizeof(d.candidates), expected).first - actual);
    if (byte != sizeof(d.candidates)) {
        std::cerr << "input differs: F04 candidates frame=" << frame << " byte=" << byte
                  << " live=" << unsigned(actual[byte]) << " recorded=" << unsigned(expected[byte]) << '\n';
    } else std::cerr << "F04 candidates match at differing packet frame=" << frame << '\n';
    const auto& a = d.motion; const auto& b = motion.frames[frame];
    if (a.epoch != b.epoch || a.frame != b.frame || a.reference_frame != b.reference_frame || a.status != b.status ||
        a.peak_x != b.peak_x || a.peak_y != b.peak_y)
        std::cerr << "input differs: MotionContext identity/status/peak frame=" << frame << '\n';
    const auto field = [&](float x, float y, const char* name) {
        if (std::bit_cast<std::uint32_t>(x) != std::bit_cast<std::uint32_t>(y))
            std::cerr << "input differs: MotionContext frame=" << frame << " field=" << name
                      << " live=" << std::hexfloat << x << " recorded=" << y << std::defaultfloat << '\n';
    };
    field(a.pcr, b.pcr, "pcr"); field(a.confidence, b.confidence, "confidence");
    field(a.tx, b.tx, "tx"); field(a.ty, b.ty, "ty");
}
void sequence_test(const std::string& engine, const std::filesystem::path& sequence,
                   const DetectionRecords& detections, const MotionRecords& motion, std::size_t count) {
    TrackingSessionConfig cfg;
    cfg.epoch = detections.epoch;
    cfg.original_width = motion.key.original_width; cfg.original_height = motion.key.original_height;
    cfg.mode = TrackingSessionMode::Serial;
    cfg.cuda_graph = false;
    Stream expected;
    TrackerReference reference(cfg.epoch);
    for (std::size_t k = 0; k < count; ++k) expected.push_back(encode_packet(reference.step(detections.frames[k], motion.frames[k])));
    const auto serial = run(engine, sequence, cfg, count);
    for (std::size_t k = 0; k < count; ++k) {
        const auto byte = different_byte(serial[k], expected[k]);
        if (byte != packet_bytes) {
            std::cerr << "CPU comparison first differing frame=" << k << " byte=" << byte << '\n';
            diagnose(engine, sequence, cfg, k, detections, motion);
            check(false, "CPU packet comparison; input diagnosis above, no relaxed comparison");
        }
        check(true, "CPU serial packet bytes");
    }
    cfg.mode = TrackingSessionMode::Pipelined;
    cfg.cuda_graph = true;
    const auto pipelined = run(engine, sequence, cfg, count);
    compare(expected, pipelined, "CPU/graph pipelined"); compare(serial, pipelined, "ordinary serial/graph pipelined");
    // Unequal ring periods expose captures that accidentally retain another
    // slot's address or the first captured frame identity. Keep these controls
    // bounded independently of the full-sequence comparison above.
    const std::size_t prefix_count = std::min<std::size_t>(count, 30);
    const Stream prefix(serial.begin(), serial.begin() + prefix_count);
    for (const unsigned slots : {1U, 3U}) {
        cfg.candidate_slots = slots; cfg.motion_slots = slots;
        cfg.result_slots = slots == 1 ? 8 : 5;
        compare(prefix, run(engine, sequence, cfg, prefix_count), slots == 1 ? "graph rings 1/1/8" : "graph rings 3/3/5");
    }
    {
        for (const auto seed : {0x12345678ULL, 0xbadc0ffeeULL}) {
            cfg.perturb_seed = seed;
            cfg.motion_slots = 1; cfg.candidate_slots = 1; cfg.result_slots = 8;
            compare(serial, run(engine, sequence, cfg, count), "perturbed pipelined");
        }
    }
    SequenceOutput cpu, gpu;
    for (std::size_t k = 0; k < count; ++k) { cpu.add(decode_packet(expected[k])); gpu.add(decode_packet(pipelined[k])); }
    check(format_mot_rows(cpu.finish()) == format_mot_rows(gpu.finish()), "F09 MOT bytes");
    std::cout << sequence.filename().string() << " frames=" << count
              << " CPU/ordinary serial/graph pipelined packet bytes identical; graph ring controls=" << prefix_count << " frames\n";
}
void lifecycle_test(const std::string& engine, const std::filesystem::path& sequence,
                    const DetectionRecords& detections, const MotionRecords& motion) {
    TrackingSessionConfig cfg;
    cfg.epoch = detections.epoch; cfg.original_width = motion.key.original_width; cfg.original_height = motion.key.original_height;
    cfg.motion_slots = 1; cfg.candidate_slots = 1;
    Stream previous;
    // Perturbation intentionally exercises the ordinary boundary schedule.
    // The graph branch must run without it to test actual graph execution,
    // including sticky failure, frozen committed state and resource cleanup.
    for (const bool use_graph : {false, true}) {
        cfg.cuda_graph = use_graph;
        cfg.result_slots = 8; cfg.perturb_seed = use_graph ? 0 : 123;
        for (const auto mode : {TrackingSessionMode::Serial, TrackingSessionMode::Pipelined}) {
            cfg.mode = mode;
            TrackingSession session(engine, cfg);
            detail::TrackingSessionDiagnostic baseline;
            for (FrameId k = 0; k < 8; ++k) {
                if (k == 3) baseline = TrackingSessionTestAccess::diagnostic(session);
                const auto bytes = jpeg(sequence, k);
                TrackingSessionTestAccess::enqueue(session, span(bytes), k, k == 3 ? static_cast<std::uint32_t>(kPostprocessUpstream) : k == 6 ? static_cast<std::uint32_t>(kPostprocessOverflow) : 0U);
            }
            bool full = false;
            try { session.enqueue({}, 8); } catch (const std::length_error&) { full = true; }
            check(full && session.pending() == 8, "ring backpressure before submission");
            session.finish();
            check(session.state() == TrackingSessionState::Failed, "failure observed on finish");
            Stream current;
            for (FrameId k = 0; k < 8; ++k) {
                const auto r = session.receive(); current.push_back(r.bytes);
                check(r.packet.status == (k < 3 ? ResultStatus::ok : k == 3 ? ResultStatus::failed : ResultStatus::skipped_after_failure), "failure status");
                check(r.packet.committed_frame == (k < 3 ? k : 2), "frozen committed frame");
                if (k >= 3) {
                    check(r.packet.fail_frame == 3 && r.packet.fail_code == Error::producer, "first failure sticky");
                    check(r.packet.tracks.empty() && r.packet.frame_error == (k == 3 ? Error::producer : Error::none), "failed/skipped rows and frame_error");
                }
            }
            const auto d = TrackingSessionTestAccess::diagnostic(session);
            check(d.control.committed_slot == baseline.control.committed_slot && d.control.committed_frame == 2, "CTL version frozen");
            const auto* actual = reinterpret_cast<const unsigned char*>(&d.tracks[d.control.committed_slot]);
            const auto* before = reinterpret_cast<const unsigned char*>(&baseline.tracks[baseline.control.committed_slot]);
            const auto byte = static_cast<std::size_t>(std::mismatch(actual, actual + sizeof(DeviceTrackState), before).first - actual);
            if (byte != sizeof(DeviceTrackState))
                std::cerr << "TS snapshot first byte=" << byte << " live=" << unsigned(actual[byte]) << " baseline=" << unsigned(before[byte])
                          << " track_base=" << offsetof(DeviceTrackState, tracks) << " track_size=" << sizeof(DeviceTrack)
                          << " h_ema_offset=" << offsetof(DeviceTrack, h_ema) << '\n';
            check(byte == sizeof(DeviceTrackState), "complete committed TS frozen");
            if (!previous.empty()) compare(previous, current, "failure ordinary/graph serial/pipelined");
            previous = current;
            session.cancel(); session.close(); session.close();
        }
        cfg.mode = TrackingSessionMode::Pipelined; cfg.perturb_seed = use_graph ? 0 : 456;
        {
            cfg.result_slots = 1;
            TrackingSession session(engine, cfg);
            for (FrameId k = 0; k < 3; ++k) {
                const auto bytes = jpeg(sequence, k); session.enqueue(span(bytes), k);
                bool full = false;
                try { session.enqueue(span(bytes), k + 1); } catch (const std::length_error&) { full = true; }
                check(full, "recoverable full ring"); check(session.receive().packet.status == ResultStatus::ok, "receive releases HP");
            }
            session.cancel();
            check(session.state() == TrackingSessionState::Cancelled, "cancel distinct from EOF");
            session.cancel(); session.close(); session.close();
        }
        {
            cfg.result_slots = 4;
            TrackingSession session(engine, cfg);
            for (FrameId k = 0; k < 3; ++k) { const auto bytes = jpeg(sequence, k); session.enqueue(span(bytes), k); }
            session.cancel();
            check(session.state() == TrackingSessionState::Cancelled, "cancel with pending packets");
            while (session.pending()) check(session.receive().packet.status == ResultStatus::ok, "cancel drains pending packets");
            session.close();
        }
        {
            TrackingSession session(engine, cfg);
            for (FrameId k = 0; k < 3; ++k) { const auto bytes = jpeg(sequence, k); session.enqueue(span(bytes), k); }
            const auto bytes = jpeg(sequence, 3);
            bool threw = false;
            try { TrackingSessionTestAccess::enqueue(session, span(bytes), 3, 0, true); } catch (const std::runtime_error&) { threw = true; }
            check(threw && session.state() == TrackingSessionState::Failed, "D-only host failure");
            session.cancel();
            while (session.pending()) check(session.receive().packet.status == ResultStatus::ok, "D-only tail retains prior packets");
            session.close(); session.close();
        }
        {
            TrackingSession session(engine, cfg);
            const auto bytes = jpeg(sequence, 0); session.enqueue(span(bytes), 0);
            // Destructor drains an unread packet and both stream tails.
        }
        check(true, "destructor drain");
    }
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6 && argc != 7) throw std::invalid_argument("usage: tracking_session_tests ENGINE SEQUENCES RECORDS SEQUENCE [--limit N|--lifecycle]");
        context = argv[4];
        const std::filesystem::path records = argv[3], sequence = std::filesystem::path(argv[2]) / argv[4];
        const auto detections = read_detection_records(records / (context + ".scdr"));
        const auto motion = read_motion_records(records / (context + ".scmo"));
        check(detections.frames.size() == motion.frames.size(), "record frame counts");
        if (argc == 6 && std::string(argv[5]) == "--lifecycle") lifecycle_test(argv[1], sequence, detections, motion);
        else {
            std::size_t count = detections.frames.size();
            if (argc == 7 && std::string(argv[5]) == "--limit") count = std::min(count, static_cast<std::size_t>(std::stoull(argv[6])));
            sequence_test(argv[1], sequence, detections, motion, count);
        }
        std::cout << "tracking_session_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "tracking_session_tests: " << e.what() << '\n'; return 1; }
}

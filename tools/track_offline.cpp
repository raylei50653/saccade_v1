// Fixed recorded inputs -> the serial CPU tracker and offline MOT writer.
#include "saccade/config_cli.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/motion.hpp"
#include "saccade/tracking.hpp"
#include "saccade/tracking_events.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: track_offline DETECTIONS.scdr MOTION.scmo|- OUT.txt [--config FILE] "
                     "[--set KEY=VALUE] [--resolved-config FILE] [--events OUT.events] [--packets OUT.bin]\n";
        return 2;
    }
    try {
        using namespace saccade;
        std::filesystem::path config_path, receipt_path, events_path, packets_path;
        ConfigAssignments assignments;
        std::set<std::string> flags;
        for (int i = 4; i < argc; ++i) {
            const std::string flag = argv[i];
            if (i + 1 == argc) throw std::invalid_argument("incomplete offline option: " + flag);
            const std::string value = argv[++i];
            if (flag == "--set") { assignments.push_back(tracking_cli_assignment(value)); continue; }
            if (!flags.insert(flag).second) throw std::invalid_argument("duplicate offline option: " + flag);
            if (flag == "--config") config_path = value;
            else if (flag == "--resolved-config") receipt_path = value;
            else if (flag == "--events") {
                events_path = value; assignments.emplace_back("observer.event_dump", "true");
            } else if (flag == "--packets") packets_path = value;
            else throw std::invalid_argument("unknown offline option: " + flag);
        }
        const auto normalized = [](const auto& p) {
            return std::filesystem::weakly_canonical(std::filesystem::absolute(p));
        };
        std::set<std::filesystem::path> inputs{normalized(argv[1])};
        if (std::string_view(argv[2]) != "-") inputs.insert(normalized(argv[2]));
        if (!config_path.empty()) inputs.insert(normalized(config_path));
        std::set<std::filesystem::path> outputs;
        const auto mot_scratch = std::filesystem::path(std::string(argv[3]) + ".tmp");
        const auto scratch_normalized = normalized(mot_scratch);
        if (inputs.contains(scratch_normalized) || std::filesystem::symlink_status(mot_scratch).type() !=
            std::filesystem::file_type::not_found)
            throw std::invalid_argument("MOT temporary output collides with an input or already exists");
        for (const auto& p : {std::filesystem::path(argv[3]), receipt_path, events_path, packets_path}) {
            if (p.empty()) continue;
            const auto path = normalized(p);
            if (inputs.contains(path) || path == scratch_normalized || !outputs.insert(path).second)
                throw std::invalid_argument("offline outputs must differ from inputs and each other");
            if (p != std::filesystem::path(argv[3]) && std::filesystem::symlink_status(p).type() !=
                std::filesystem::file_type::not_found)
                throw std::invalid_argument("diagnostic output already exists; choose a new path");
        }
        const auto detection_bytes = read_text_file(argv[1]);
        const auto detections = decode_detection_records(detection_bytes);
        std::string motion_bytes;
        MotionRecords motion;
        const bool disabled = std::string_view(argv[2]) == "-";
        if (disabled) {
            for (std::size_t i = 0; i < detections.frames.size(); ++i) {
                MotionContext m;
                m.epoch = detections.epoch; m.frame = static_cast<FrameId>(i);
                m.status = MotionStatus::disabled; motion.frames.push_back(m);
            }
        } else {
            motion_bytes = read_text_file(argv[2]);
            motion = decode_motion_records(motion_bytes);
        }
        if (detections.frames.size() != motion.frames.size())
            throw std::runtime_error("detection and motion frame counts differ");
        ResolvedTrackingConfig defaults;
        defaults.source.epoch = detections.epoch;
        defaults.implementation.motion_profile = disabled ? "disabled" : "recorded";
        defaults.execution.mode = "serial"; defaults.execution.cuda_graph = false;
        defaults.execution.candidate_slots = defaults.execution.motion_slots = defaults.execution.result_slots = 1;
        if (!disabled) {
            defaults.source.width = static_cast<std::uint32_t>(motion.key.original_width);
            defaults.source.height = static_cast<std::uint32_t>(motion.key.original_height);
            defaults.preprocess.max_width = std::max(1920U, defaults.source.width);
            defaults.preprocess.max_height = std::max(1080U, defaults.source.height);
        }
        const auto resolved = resolve_tracking_config(config_path.empty() ? "" : read_text_file(config_path),
            declared_tracking_environment(), assignments, {}, defaults, "track_offline-records");
        if (resolved.source.epoch != detections.epoch || resolved.source.width != defaults.source.width ||
            resolved.source.height != defaults.source.height || resolved.implementation.motion_profile != defaults.implementation.motion_profile)
            throw std::invalid_argument("resolved source or motion differs from the supplied records");
        if (resolved.output_profile != "v1-mot/1")
            throw std::invalid_argument("offline records require v1-mot/1 output");
        if (resolved.observer.event_dump != !events_path.empty())
            throw std::invalid_argument("observer.event_dump requires --events and cannot disable that flag");
        auto identity = tracking_build_identity();
        identity += "input.frames=" + std::to_string(detections.frames.size()) + '\n';
        identity += "build.detections_sha256=" + sha256_hex(detection_bytes) + '\n';
        if (!disabled) identity += "build.motion_sha256=" + sha256_hex(motion_bytes) + '\n';
        const auto receipt = resolved_config_record(resolved, identity);
        if (!receipt_path.empty()) write_resolved_config_file(receipt_path, receipt);
        std::cout << receipt;
        const auto start = std::chrono::steady_clock::now();
        TrackingEventRecorder recorder;
        TrackerReference tracker(detections.epoch, resolved.tracker,
            resolved.observer.event_dump ? &recorder : nullptr, resolved.source.id);
        SequenceOutput output(resolved.mot);
        std::vector<std::uint8_t> packets;
        for (std::size_t i = 0; i < detections.frames.size(); ++i) {
            const auto packet = tracker.step(detections.frames[i], motion.frames[i]);
            if (!packets_path.empty()) {
                const auto bytes = encode_packet(packet);
                packets.insert(packets.end(), bytes.begin(), bytes.end());
            }
            output.add(packet);
            if (output.failed()) {
                if (!events_path.empty()) write_exclusive_binary_file(events_path, encode_tracking_events(recorder.events));
                if (!packets_path.empty()) write_exclusive_binary_file(packets_path, packets);
                throw std::runtime_error("frame " + std::to_string(i) + " failed: status " +
                    std::to_string(static_cast<unsigned>(packet.status)) + " error " +
                    std::to_string(static_cast<unsigned>(packet.frame_error)));
            }
        }
        const auto rows = output.finish();
        if (!events_path.empty()) {
            const auto validation = validate_tracking_events(recorder.events, resolved.tracker);
            if (!validation.valid) throw std::runtime_error("recorded event " + std::to_string(validation.event_index) + ": " + validation.message);
            write_exclusive_binary_file(events_path, encode_tracking_events(recorder.events));
            std::cout << "events_sha256=" << tracking_events_hash(recorder.events) << '\n';
        }
        if (!packets_path.empty()) write_exclusive_binary_file(packets_path, packets);
        write_mot_file(argv[3], rows);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::set<std::uint64_t> ids;
        std::size_t interpolated = 0;
        for (const auto& row : rows) { ids.insert(row.id); interpolated += row.interpolated ? 1 : 0; }
        std::cout << "track_offline frames=" << detections.frames.size() << " rows=" << rows.size()
                  << " interpolated=" << interpolated << " ids=" << ids.size() << " tracker_ms=" << ms << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "track_offline: " << e.what() << '\n'; return 1;
    }
}

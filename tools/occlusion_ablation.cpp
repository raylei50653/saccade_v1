// CPU-only research replay. See evidence/occlusion/2026-11-05/PLAN.md.
#include "saccade/file_io.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/occlusion_experiment.hpp"
#include "saccade/protocol.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c);
        else out << c;
    }
    out << '"';
    return out.str();
}
void require_ok(saccade::Error error, std::size_t frame, const char* stage) {
    if (error != saccade::Error::none)
        throw std::runtime_error(std::string(stage) + " failed at frame " + std::to_string(frame) +
                                 " error=" + std::to_string(static_cast<unsigned>(error)));
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: occlusion_ablation DET.scdr MOTION.scmo OUT.txt baseline|oao|front|both DIAG.json\n";
        return 2;
    }
    try {
        using namespace saccade;
        const std::string arm = argv[4];
        if (arm != "baseline" && arm != "oao" && arm != "front" && arm != "both")
            throw std::invalid_argument("unknown experiment arm");
        if (std::filesystem::weakly_canonical(std::filesystem::absolute(argv[3])) ==
            std::filesystem::weakly_canonical(std::filesystem::absolute(argv[5])))
            throw std::invalid_argument("experiment outputs must name distinct files");
        if (std::filesystem::exists(argv[3]) || std::filesystem::exists(argv[5]))
            throw std::invalid_argument("experiment outputs must be new files");
        const auto detections = read_detection_records(argv[1]);
        const auto motion = read_motion_records(argv[2]);
        if (detections.frames.empty() || detections.frames.size() != motion.frames.size() ||
            detections.epoch != motion.epoch)
            throw std::runtime_error("empty records or mismatched detection/motion identity");
        const TrackerConfig config;
        OcclusionExperimentConfig experiment;
        experiment.oao_enabled = arm == "oao" || arm == "both";
        experiment.front_enabled = arm == "front" || arm == "both";
        OcclusionExperiment occlusion(config, experiment);
        TrackState committed = initial_track_state(detections.epoch);
        TrackerReference oracle(detections.epoch, config);
        SequenceOutput output;
        const AssociationCostHook hook = [&](const Track& track, const BatchDetection& detection, float overlap) {
            return occlusion.cost(track, detection, overlap);
        };
        std::size_t oracle_packets = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < detections.frames.size(); ++i) {
            const auto predicted = predict_tracks(committed, motion.frames[i], static_cast<FrameId>(i), config);
            require_ok(predicted.error, i, "predict");
            require_ok(occlusion.prepare(predicted), i, "occlusion");
            const auto batch = assemble_detection_batch(detections.frames[i], predicted, config);
            require_ok(batch.error, i, "batch");
            auto update = associate_and_update(predicted, batch, motion.frames[i], config, hook);
            require_ok(update.error, i, "association");
            ResultPacket packet;
            packet.epoch = committed.epoch;
            packet.frame = static_cast<FrameId>(i);
            committed = std::move(update.next);
            packet.committed_frame = committed.committed_frame;
            packet.tracks = std::move(update.rows);
            if (arm == "baseline") {
                const auto expected = oracle.step(detections.frames[i], motion.frames[i]);
                if (encode_packet(packet) != encode_packet(expected))
                    throw std::runtime_error("baseline packet differs from stock reference at frame " + std::to_string(i));
                ++oracle_packets;
            }
            output.add(packet);
            if (output.failed()) throw std::runtime_error("MOT output rejected packet");
        }
        const auto rows = output.finish();
        const double replay_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        write_mot_file(argv[3], rows);
        std::set<std::uint64_t> ids;
        std::size_t interpolated = 0;
        for (const auto& row : rows) {
            ids.insert(row.id);
            interpolated += row.interpolated ? 1 : 0;
        }
        const auto& d = occlusion.diagnostics();
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::setprecision(9) << "{\n"
            << "  \"arm\": " << json_string(arm) << ",\n"
            << "  \"scope\": \"CPU fixed-input development replay; no GPU or end-to-end timing\",\n"
            << "  \"formal_acceptance\": false,\n"
            << "  \"tracker_config\": " << json_string(canonical_tracker_config(config)) << ",\n"
            << "  \"occlusion_config\": {\"oao_enabled\": " << (experiment.oao_enabled ? "true" : "false")
            << ", \"front_enabled\": " << (experiment.front_enabled ? "true" : "false")
            << ", \"oao_tau\": " << experiment.oao_tau << ", \"oao_ramp_frames\": " << experiment.oao_ramp_frames
            << ", \"front_iou_thresh\": " << experiment.front_iou_thresh
            << ", \"front_foot_gap\": " << experiment.front_foot_gap << ", \"front_ttl\": " << experiment.front_ttl
            << ", \"front_cost_weight\": " << experiment.front_cost_weight << "},\n"
            << "  \"frames\": " << detections.frames.size() << ", \"mot_rows\": " << rows.size()
            << ", \"interpolated_rows\": " << interpolated << ", \"ids\": " << ids.size() << ",\n"
            << "  \"baseline_packet_comparisons\": " << oracle_packets << ",\n"
            << "  \"cpu_replay_ms_including_baseline_oracle\": " << replay_ms << ",\n"
            << "  \"diagnostics\": {\"frames\": " << d.frames
            << ", \"overlapped_track_frames\": " << d.overlapped_track_frames
            << ", \"front_latched_track_frames\": " << d.front_latched_track_frames
            << ", \"cost_calls\": " << d.cost_calls
            << ", \"oao_positive_calls\": " << d.oao_positive_calls
            << ", \"front_positive_calls\": " << d.front_positive_calls << "}\n}\n";
        write_text_file(argv[5], out.str());
        std::cout << "arm=" << arm << " frames=" << detections.frames.size() << " rows=" << rows.size()
                  << " ids=" << ids.size() << " baseline_packet_comparisons=" << oracle_packets << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "occlusion_ablation: " << e.what() << '\n';
        return 1;
    }
}

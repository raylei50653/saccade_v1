// M1 offline harness (docs/TRACKING.md §12): recorded F01-F04 detections
// (SCDR) and F05/F06 motion (SCMO) -> CPU TrackerReference -> F09 MOT file.
// MOTION "-" runs with GMC explicitly disabled (diagnostic ablation only).
#include "saccade/mot_output.hpp"
#include "saccade/motion.hpp"
#include "saccade/tracking.hpp"

#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: track_offline DETECTIONS.scdr MOTION.scmo|- OUT.txt\n";
        return 2;
    }
    try {
        using namespace saccade;
        const auto detections = read_detection_records(argv[1]);
        MotionRecords motion;
        if (std::string(argv[2]) == "-") {
            for (std::size_t i = 0; i < detections.frames.size(); ++i) {
                MotionContext m;
                m.epoch = detections.epoch;
                m.frame = static_cast<std::int64_t>(i);
                m.status = MotionStatus::disabled;
                motion.frames.push_back(m);
            }
        } else {
            motion = read_motion_records(argv[2]);
        }
        if (detections.frames.size() != motion.frames.size())
            throw std::runtime_error("detection and motion frame counts differ");
        const auto start = std::chrono::steady_clock::now();
        TrackerReference tracker(detections.epoch);
        SequenceOutput output;
        for (std::size_t i = 0; i < detections.frames.size(); ++i) {
            const auto packet = tracker.step(detections.frames[i], motion.frames[i]);
            output.add(packet);
            if (output.failed())
                throw std::runtime_error("frame " + std::to_string(i) + " failed: status " +
                                         std::to_string(static_cast<unsigned>(packet.status)) + " error " +
                                         std::to_string(static_cast<unsigned>(packet.frame_error)));
        }
        const auto rows = output.finish();
        write_mot_file(argv[3], rows);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::set<std::uint64_t> ids;
        std::size_t interpolated = 0;
        for (const auto& row : rows) {
            ids.insert(row.id);
            interpolated += row.interpolated ? 1 : 0;
        }
        std::cout << "track_offline frames=" << detections.frames.size() << " rows=" << rows.size()
                  << " interpolated=" << interpolated << " ids=" << ids.size() << " tracker_ms=" << ms << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "track_offline: " << e.what() << '\n';
        return 1;
    }
}

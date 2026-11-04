#pragma once

#include "saccade/postprocess.hpp"
#include "saccade/protocol.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

// F09 contract, profile v1-mot/1 (docs/TRACKING.md §10), plus the detection
// record stream used by tools/detect_sequence. Line C owns the implementation
// and may extend this header within its own scope.
namespace saccade {

inline constexpr const char* kMotOutputProfile = "v1-mot/1";

struct MotOutputConfig {
    std::uint32_t interpolate_max_gap = 35; // maximum number of missing frames filled
    std::uint32_t interpolate_min_rows = 5; // observations an ID needs before filling
};

struct MotRow {
    std::int64_t frame = 0; // MOT frame number = processing frame + 1
    std::uint64_t id = 0;   // per-sequence first-appearance ID, from 1
    double x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    double conf = 0;        // observed_score; -1 for interpolated rows
    bool interpolated = false;
};

// Consumes decoded ResultPackets in processing order from frame 0. A non-OK
// packet, a frame gap or an epoch change marks the sequence failed; failure is
// sticky. Rows are only produced after finish() on a complete sequence.
class SequenceOutput {
public:
    explicit SequenceOutput(const MotOutputConfig& config = {});
    void add(const ResultPacket& packet);
    bool failed() const { return failed_; }
    std::int64_t frames() const { return frames_; }
    // Interpolated, sorted by (frame, id). logic_error if failed.
    std::vector<MotRow> finish() const;

private:
    MotOutputConfig config_;
    bool failed_ = false;
    std::int64_t frames_ = 0;
    std::uint64_t epoch_ = 0;
    std::vector<MotRow> observed_;
    std::vector<std::uint64_t> output_ids_; // index + 1 is the MOT ID
};

// "frame,id,left,top,width,height,conf,-1,-1,-1\n" lines, C locale %.3f;
// interpolated rows write conf as -1.
std::string format_mot_rows(std::span<const MotRow> rows);
// Writes <path>.tmp, closes it checked and renames it to path. On any failure
// the temporary file is removed and the error is rethrown; path is untouched.
void write_mot_file(const std::filesystem::path& path, std::span<const MotRow> rows);

// Per-sequence detection stream written by tools/detect_sequence: "SCDR",
// u32 version=1, u32 sizeof(DetectionCandidates), u64 epoch, u64 count, then
// count raw DetectionCandidates. Internal tool format on one machine, not a
// wire ABI. The reader rejects other magic/version/size, truncation and
// trailing bytes.
struct DetectionRecords {
    std::uint64_t epoch = 0;
    std::vector<DetectionCandidates> frames;
};
void write_detection_records(const std::filesystem::path& path, const DetectionRecords& records);
DetectionRecords read_detection_records(const std::filesystem::path& path);

} // namespace saccade

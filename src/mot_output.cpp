#include "saccade/mot_output.hpp"
#include "saccade/file_io.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace saccade {

SequenceOutput::SequenceOutput(const MotOutputConfig& config) : config_(config) {}

void SequenceOutput::add(const ResultPacket& packet) {
    if (failed_) return;
    if (packet.status != ResultStatus::ok || packet.frame != frames_ ||
        (frames_ != 0 && packet.epoch != epoch_)) {
        failed_ = true;
        return;
    }
    if (frames_ == 0) epoch_ = packet.epoch;
    for (const auto& row : packet.tracks) {
        auto position = std::find(output_ids_.begin(), output_ids_.end(), row.output_id);
        if (position == output_ids_.end()) {
            output_ids_.push_back(row.output_id);
            position = output_ids_.end() - 1;
        }
        const auto id = static_cast<std::uint64_t>(position - output_ids_.begin()) + 1;
        observed_.push_back({packet.frame + 1, id, row.xyxy[0], row.xyxy[1],
                             row.xyxy[2], row.xyxy[3], row.observed_score, false});
    }
    ++frames_;
}

std::vector<MotRow> SequenceOutput::finish() const {
    if (failed_) throw std::logic_error("cannot finish a failed sequence");
    auto rows = observed_;
    std::vector<std::vector<const MotRow*>> by_id(output_ids_.size());
    for (const auto& row : observed_) by_id[row.id - 1].push_back(&row);
    for (const auto& observations : by_id) {
        if (observations.size() < config_.interpolate_min_rows) continue;
        for (std::size_t i = 1; i < observations.size(); ++i) {
            const auto& a = *observations[i - 1];
            const auto& b = *observations[i];
            const auto distance = b.frame - a.frame;
            if (distance <= 1 || distance > static_cast<std::int64_t>(config_.interpolate_max_gap) + 1)
                continue;
            for (auto frame = a.frame + 1; frame < b.frame; ++frame) {
                const double t = static_cast<double>(frame - a.frame) / static_cast<double>(distance);
                rows.push_back({frame, a.id,
                                a.x1 + t * (b.x1 - a.x1), a.y1 + t * (b.y1 - a.y1),
                                a.x2 + t * (b.x2 - a.x2), a.y2 + t * (b.y2 - a.y2), -1, true});
            }
        }
    }
    std::sort(rows.begin(), rows.end(), [](const MotRow& a, const MotRow& b) {
        return std::tie(a.frame, a.id) < std::tie(b.frame, b.id);
    });
    return rows;
}

std::string format_mot_rows(std::span<const MotRow> rows) {
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::fixed << std::setprecision(3);
    for (const auto& row : rows) {
        output << row.frame << ',' << row.id << ',' << row.x1 << ',' << row.y1 << ','
               << row.x2 - row.x1 << ',' << row.y2 - row.y1 << ',';
        if (row.interpolated) output << "-1";
        else output << row.conf;
        output << ",-1,-1,-1\n";
    }
    return output.str();
}

void write_mot_file(const std::filesystem::path& path, std::span<const MotRow> rows) {
    const std::filesystem::path temporary = path.string() + ".tmp";
    try {
        write_text_file(temporary, format_mot_rows(rows));
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

void write_detection_records(const std::filesystem::path& path, const DetectionRecords& records) {
    std::string bytes = "SCDR";
    const auto append = [&](const auto& value) {
        bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    const std::uint32_t version = 1, size = sizeof(DetectionCandidates);
    const std::uint64_t count = records.frames.size();
    append(version);
    append(size);
    append(records.epoch);
    append(count);
    if (!records.frames.empty())
        bytes.append(reinterpret_cast<const char*>(records.frames.data()),
                     records.frames.size() * sizeof(DetectionCandidates));
    write_text_file(path, bytes);
}

DetectionRecords read_detection_records(const std::filesystem::path& path) {
    const auto bytes = read_text_file(path);
    constexpr std::size_t header_size = 28;
    if (bytes.size() < header_size) throw std::runtime_error("SCDR: truncated header");
    if (bytes.compare(0, 4, "SCDR") != 0) throw std::runtime_error("SCDR: invalid magic");
    std::uint32_t version = 0, size = 0;
    std::uint64_t count = 0;
    DetectionRecords records;
    std::memcpy(&version, bytes.data() + 4, sizeof(version));
    std::memcpy(&size, bytes.data() + 8, sizeof(size));
    std::memcpy(&records.epoch, bytes.data() + 12, sizeof(records.epoch));
    std::memcpy(&count, bytes.data() + 20, sizeof(count));
    if (version != 1) throw std::runtime_error("SCDR: unsupported version");
    if (size != sizeof(DetectionCandidates)) throw std::runtime_error("SCDR: incompatible struct size");
    const auto payload_size = bytes.size() - header_size;
    if (count > payload_size / size) throw std::runtime_error("SCDR: truncated records");
    if (count * size != payload_size) throw std::runtime_error("SCDR: trailing bytes");
    records.frames.resize(static_cast<std::size_t>(count));
    if (payload_size != 0) std::memcpy(records.frames.data(), bytes.data() + header_size, payload_size);
    return records;
}

} // namespace saccade

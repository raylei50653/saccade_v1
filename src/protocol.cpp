#include "saccade/protocol.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace saccade {
namespace {
constexpr std::uint32_t known_errors = 31;
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void put(std::span<std::uint8_t> out, std::size_t offset, std::uint64_t value, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) out[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
}
std::uint64_t get(std::span<const std::uint8_t> in, std::size_t offset, std::size_t size) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < size; ++i) value |= std::uint64_t{in[offset + i]} << (8 * i);
    return value;
}
void put_float(std::span<std::uint8_t> out, std::size_t offset, float value) {
    require(std::isfinite(value), "non-finite packet float");
    put(out, offset, std::bit_cast<std::uint32_t>(value == 0 ? 0.0F : value), 4);
}
float get_float(std::span<const std::uint8_t> in, std::size_t offset) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(get(in, offset, 4)));
}
void validate(const ResultPacket& p) {
    require(p.frame >= 0, "packet frame must be nonnegative");
    require(p.committed_frame >= no_frame && p.committed_frame <= p.frame, "invalid committed frame");
    require((static_cast<std::uint32_t>(p.frame_error) & ~known_errors) == 0 &&
            (static_cast<std::uint32_t>(p.fail_code) & ~known_errors) == 0, "unknown error bits");
    require(p.tracks.size() <= packet_capacity, "packet capacity exceeded");
    switch (p.status) {
    case ResultStatus::ok:
        require(p.committed_frame == p.frame && p.fail_frame == no_frame &&
                p.fail_code == Error::none && p.frame_error == Error::none, "invalid OK packet");
        break;
    case ResultStatus::failed:
        require(p.fail_frame == p.frame && p.committed_frame < p.frame &&
                p.fail_code != Error::none && p.frame_error == p.fail_code &&
                p.tracks.empty(), "invalid FAILED packet");
        break;
    case ResultStatus::skipped_after_failure:
        require(p.fail_frame >= 0 && p.fail_frame < p.frame && p.committed_frame < p.fail_frame &&
                p.fail_code != Error::none && p.frame_error == Error::none &&
                p.tracks.empty(), "invalid SKIPPED packet");
        break;
    default: throw std::invalid_argument("unknown result status");
    }
    for (const auto& row : p.tracks) {
        require(row.uid != 0 && row.output_id != 0, "track IDs must be nonzero");
        require(row.flags == 1 || row.flags == 2, "invalid observation flags");
        require(std::isfinite(row.observed_score) && row.observed_score >= 0 && row.observed_score <= 1 &&
                std::isfinite(row.association_score),
                "invalid score");
        require(row.flags != 2 || (row.observed_score == 0 && row.association_score == 0),
                "predicted row must have zero observation scores");
    }
}
} // namespace

PacketBytes encode_packet(const ResultPacket& p) {
    validate(p);
    PacketBytes bytes{}; // Includes all reserved bytes and all unused rows.
    bytes[0] = 'S'; bytes[1] = 'C'; bytes[2] = 'R'; bytes[3] = 'P';
    put(bytes, 4, 1, 2);
    put(bytes, 6, packet_header_bytes, 2);
    put(bytes, 8, packet_bytes, 4);
    put(bytes, 12, packet_capacity, 4);
    put(bytes, 16, p.epoch, 8);
    put(bytes, 24, static_cast<std::uint64_t>(p.frame), 8);
    put(bytes, 32, static_cast<std::uint64_t>(p.committed_frame), 8);
    put(bytes, 40, static_cast<std::uint64_t>(p.fail_frame), 8);
    put(bytes, 48, static_cast<std::uint32_t>(p.status), 4);
    put(bytes, 52, p.tracks.size(), 4);
    put(bytes, 56, static_cast<std::uint32_t>(p.frame_error), 4);
    put(bytes, 60, static_cast<std::uint32_t>(p.fail_code), 4);
    for (std::size_t i = 0; i < p.tracks.size(); ++i) {
        const auto offset = packet_header_bytes + packet_row_bytes * i;
        const auto& row = p.tracks[i];
        put(bytes, offset, row.uid, 8);
        put(bytes, offset + 8, row.output_id, 8);
        put(bytes, offset + 16, row.class_id, 4);
        put(bytes, offset + 20, row.flags, 4);
        for (std::size_t j = 0; j < 4; ++j) put_float(bytes, offset + 24 + 4 * j, row.xyxy[j]);
        put_float(bytes, offset + 40, row.observed_score);
        put_float(bytes, offset + 44, row.association_score);
    }
    return bytes;
}

ResultPacket decode_packet(std::span<const std::uint8_t> bytes) {
    require(bytes.size() == packet_bytes, "invalid packet byte length");
    require(bytes[0] == 'S' && bytes[1] == 'C' && bytes[2] == 'R' && bytes[3] == 'P', "invalid packet magic");
    require(get(bytes, 4, 2) == 1 && get(bytes, 6, 2) == packet_header_bytes &&
            get(bytes, 8, 4) == packet_bytes && get(bytes, 12, 4) == packet_capacity,
            "unsupported packet profile");
    ResultPacket p;
    p.epoch = get(bytes, 16, 8);
    p.frame = std::bit_cast<FrameId>(get(bytes, 24, 8));
    p.committed_frame = std::bit_cast<FrameId>(get(bytes, 32, 8));
    p.fail_frame = std::bit_cast<FrameId>(get(bytes, 40, 8));
    p.status = static_cast<ResultStatus>(get(bytes, 48, 4));
    const auto count = get(bytes, 52, 4);
    require(count <= packet_capacity, "packet count exceeds capacity");
    p.frame_error = static_cast<Error>(get(bytes, 56, 4));
    p.fail_code = static_cast<Error>(get(bytes, 60, 4));
    for (std::size_t i = 0; i < count; ++i) {
        const auto offset = packet_header_bytes + packet_row_bytes * i;
        TrackRow row;
        row.uid = get(bytes, offset, 8);
        row.output_id = get(bytes, offset + 8, 8);
        row.class_id = static_cast<std::uint32_t>(get(bytes, offset + 16, 4));
        row.flags = static_cast<std::uint32_t>(get(bytes, offset + 20, 4));
        for (std::size_t j = 0; j < 4; ++j) row.xyxy[j] = get_float(bytes, offset + 24 + 4 * j);
        row.observed_score = get_float(bytes, offset + 40);
        row.association_score = get_float(bytes, offset + 44);
        p.tracks.push_back(row);
    }
    const auto canonical = encode_packet(p);
    require(std::equal(bytes.begin(), bytes.end(), canonical.begin()), "noncanonical packet bytes");
    return p;
}

ByteComparison compare_packet_bytes(std::span<const std::uint8_t> left,
                                   std::span<const std::uint8_t> right) {
    require(!left.empty() && left.size() % packet_bytes == 0 &&
            !right.empty() && right.size() % packet_bytes == 0, "truncated or empty packet stream");
    const auto common = std::min(left.size(), right.size());
    for (std::size_t i = 0; i < common; ++i) {
        if (left[i] != right[i]) return {false, i, left[i], right[i]};
    }
    if (left.size() != right.size()) return {false, common, 0, 0};
    return {};
}

std::vector<std::uint8_t> read_packet_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open packet file: " + path.string());
    const auto size = file.tellg();
    if (size <= 0 || static_cast<std::uint64_t>(size) % packet_bytes != 0)
        throw std::invalid_argument("truncated or empty packet stream: " + path.string());
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("cannot read packet file: " + path.string());
    return bytes;
}

FailureModel::FailureModel(Epoch epoch) { control_.epoch = epoch; }
void FailureModel::require_active() const {
    if (!active_) throw std::logic_error("no active frame");
}
void FailureModel::begin(FrameId frame) {
    if (active_) throw std::logic_error("frame already active");
    require(frame >= 0, "frame must be nonnegative");
    require(frame > control_.current_frame, "submitted frame IDs must increase");
    control_.current_frame = frame;
    control_.working_slot = 1 - control_.committed_slot;
    control_.frame_status = Error::none;
    control_.run_stage = false;
    tracks_ready_ = spectrum_ready_ = updated_ = false;
    active_ = true;
    if (control_.fail_code == Error::none &&
        (control_.committed_frame == std::numeric_limits<FrameId>::max() ||
         frame != control_.committed_frame + 1)) add_error(Error::protocol);
}
void FailureModel::prepare_stage() {
    require_active();
    control_.run_stage = control_.fail_code == Error::none && control_.frame_status == Error::none;
}
void FailureModel::add_error(Error error) {
    require_active();
    control_.frame_status = control_.frame_status | error;
}
std::size_t FailureModel::read_slot() const {
    return control_.committed_slot;
}
std::size_t FailureModel::write_slot() const { return control_.working_slot; }
const ModelBuffer& FailureModel::committed_tracks() const { return tracks_[read_slot()]; }
const ModelBuffer* FailureModel::committed_spectrum() const {
    return control_.committed_frame == no_frame ? nullptr : &spectra_[read_slot()];
}
void FailureModel::fft() {
    require_active();
    fft_scratch_.frame = control_.current_frame;
    for (std::size_t i = 0; i < fft_scratch_.words.size(); ++i)
        fft_scratch_.words[i] = (static_cast<std::uint64_t>(control_.current_frame) + 1) * 17 + i;
}
void FailureModel::publish_spectrum() {
    require_active();
    if (!control_.run_stage) return;
    if (fft_scratch_.frame != control_.current_frame) { add_error(Error::protocol); return; }
    spectra_[write_slot()] = fft_scratch_;
    spectrum_ready_ = true;
}
void FailureModel::predict() {
    require_active();
    if (!control_.run_stage) return;
    if (tracks_ready_ || !spectrum_ready_) { add_error(Error::protocol); return; }
    tracks_[write_slot()] = tracks_[read_slot()];
    tracks_[write_slot()].frame = control_.current_frame;
    ++tracks_[write_slot()].words[0];
    tracks_ready_ = true;
}
void FailureModel::update() {
    require_active();
    if (!control_.run_stage) return;
    if (!tracks_ready_) { add_error(Error::protocol); return; }
    if (updated_) { add_error(Error::protocol); return; }
    ++tracks_[write_slot()].words[1];
    updated_ = true;
}
void FailureModel::commit() {
    require_active();
    if (control_.fail_code == Error::none) {
        if (control_.frame_status == Error::none && (!tracks_ready_ || !spectrum_ready_ || !updated_))
            add_error(Error::protocol);
        if (control_.frame_status != Error::none) {
            control_.fail_frame = control_.current_frame;
            control_.fail_code = control_.frame_status;
        } else {
            control_.committed_frame = control_.current_frame;
            control_.committed_slot = control_.working_slot;
        }
    }
    control_.run_stage = false;
    active_ = false;
}
ResultPacket FailureModel::packet() const {
    if (active_ || control_.current_frame == no_frame) throw std::logic_error("packet requires commit");
    ResultPacket p;
    p.epoch = control_.epoch;
    p.frame = control_.current_frame;
    p.committed_frame = control_.committed_frame;
    p.fail_frame = control_.fail_frame;
    p.fail_code = control_.fail_code;
    if (control_.fail_code != Error::none) {
        p.status = control_.fail_frame == control_.current_frame ? ResultStatus::failed : ResultStatus::skipped_after_failure;
        p.frame_error = p.status == ResultStatus::failed ? control_.fail_code : Error::none;
    }
    return p;
}

} // namespace saccade

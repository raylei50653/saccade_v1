#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace saccade {

using FrameId = std::int64_t; // Processing order, separate from source frame IDs.
using Epoch = std::uint64_t;
inline constexpr FrameId no_frame = -1;
inline constexpr std::size_t packet_capacity = 256;
inline constexpr std::size_t packet_header_bytes = 80;
inline constexpr std::size_t packet_row_bytes = 48;
inline constexpr std::size_t packet_bytes = packet_header_bytes + packet_capacity * packet_row_bytes;

enum class Error : std::uint32_t {
    none = 0, protocol = 1U << 0, producer = 1U << 1,
    overflow = 1U << 2, numeric = 1U << 3, compute = 1U << 4,
};
constexpr Error operator|(Error a, Error b) {
    return static_cast<Error>(static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b));
}
enum class ResultStatus : std::uint32_t { ok = 0, failed = 1, skipped_after_failure = 2 };

// Logical values only: sizeof/offsetof are deliberately not a wire ABI.
struct TrackRow {
    std::uint64_t uid = 0;
    std::uint64_t output_id = 0;
    std::uint32_t class_id = 0;
    std::uint32_t flags = 0; // bit0 observed; bit1 predicted; exactly one for valid rows.
    std::array<float, 4> xyxy{};
    float observed_score = 0; // Original observation probability; 0 for predicted rows.
    float association_score = 0; // Tracker-defined finite association score; 0 for predicted rows.
};
struct ResultPacket {
    Epoch epoch = 0;
    FrameId frame = no_frame;
    FrameId committed_frame = no_frame;
    FrameId fail_frame = no_frame;
    ResultStatus status = ResultStatus::ok;
    Error frame_error = Error::none;
    Error fail_code = Error::none;
    std::vector<TrackRow> tracks;
};
using PacketBytes = std::array<std::uint8_t, packet_bytes>;
PacketBytes encode_packet(const ResultPacket& packet);
// Strictly validates the canonical v1 encoding, including reserved/unused bytes.
ResultPacket decode_packet(std::span<const std::uint8_t> bytes);

struct ByteComparison {
    bool equal = true;
    std::size_t offset = 0;
    std::uint8_t left = 0, right = 0;
};
// Compares every byte, including reserved and unused rows. Length must be a
// positive integral number of fixed v1 packets. Decoding is a separate action.
ByteComparison compare_packet_bytes(std::span<const std::uint8_t> left,
                                   std::span<const std::uint8_t> right);
std::vector<std::uint8_t> read_packet_file(const std::filesystem::path& path);

struct Control {
    Epoch epoch = 0;
    FrameId committed_frame = no_frame;
    FrameId current_frame = no_frame;
    std::uint32_t committed_slot = 1;
    std::uint32_t working_slot = 0;
    Error frame_status = Error::none;
    FrameId fail_frame = no_frame;
    Error fail_code = Error::none;
    bool run_stage = false;
};
struct ModelBuffer {
    FrameId frame = no_frame;
    std::array<std::uint64_t, 4> words{};
    bool operator==(const ModelBuffer&) const = default;
};

// CPU semantic model of ordered M work, not an implementation of CUDA kernels.
// A prepare_stage call represents a preceding one-thread control kernel.
class FailureModel {
public:
    // Epoch is immutable for the lifetime of this session model.
    explicit FailureModel(Epoch epoch);
    void begin(FrameId frame);
    void prepare_stage();
    void add_error(Error error);
    void fft(); // Unconditional library work; writes only isolated scratch.
    void publish_spectrum(); // Computation reads immutable run_stage only.
    void predict();
    void update();
    void commit(); // Control operation; never subject to run_stage.
    ResultPacket packet() const; // Also unconditional; current submitted identity.

    const Control& control() const { return control_; }
    const ModelBuffer& committed_tracks() const;
    const ModelBuffer* committed_spectrum() const;
    const ModelBuffer& scratch() const { return fft_scratch_; }
    std::size_t read_slot() const;
    std::size_t write_slot() const;
private:
    void require_active() const;
    Control control_{};
    std::array<ModelBuffer, 2> tracks_{};
    std::array<ModelBuffer, 2> spectra_{};
    ModelBuffer fft_scratch_{};
    bool active_ = false;
    bool spectrum_ready_ = false, tracks_ready_ = false, updated_ = false;
};

} // namespace saccade

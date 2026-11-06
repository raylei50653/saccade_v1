#include "saccade/mot_output.hpp"
#include "saccade/file_io.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <tuple>

namespace {
using namespace saccade;
std::size_t checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class Exception, class F> void rejects(F&& action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const Exception&) { rejected = true; }
    check(rejected, message);
}

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("saccade-mot-output-" + std::to_string(nonce));
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("cannot create temporary directory");
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::permissions(path / "unwritable", std::filesystem::perms::owner_all, ignored);
        std::filesystem::remove_all(path, ignored);
    }
};

TrackRow track(std::uint64_t id, std::array<float, 4> box = {1, 2, 4, 6}, float score = 0.75f) {
    TrackRow row;
    row.uid = id + 100;
    row.output_id = id;
    row.flags = 1;
    row.xyxy = box;
    row.observed_score = score;
    row.association_score = 0.25f;
    return row;
}

ResultPacket packet(FrameId frame, std::vector<TrackRow> rows = {}, Epoch epoch = 9) {
    ResultPacket result;
    result.frame = result.committed_frame = frame;
    result.epoch = epoch;
    result.tracks = std::move(rows);
    return result;
}

void mapping_tests() {
    SequenceOutput empty;
    check(!empty.failed() && empty.frames() == 0 && empty.finish().empty(), "empty sequence");
    SequenceOutput sequence;
    sequence.add(packet(0, {track(90), track(7)}));
    sequence.add(packet(1, {track(42), track(7)}));
    sequence.add(packet(2));
    auto reappearing = track(90, {-2, 3, 8, 12}, 0.625f);
    reappearing.uid = 1000;
    sequence.add(packet(3, {reappearing, track(42)}));
    check(!sequence.failed() && sequence.frames() == 4, "empty packet advances frame count");
    const auto rows = sequence.finish();
    check(rows.size() == 6, "only observations below interpolation threshold");
    constexpr std::array<std::pair<FrameId, std::uint64_t>, 6> expected{{{1, 1}, {1, 2}, {2, 2}, {2, 3}, {4, 1}, {4, 3}}};
    for (std::size_t i = 0; i < rows.size(); ++i) {
        check(rows[i].frame == expected[i].first && rows[i].id == expected[i].second,
              "first appearance mapping and sorted rows");
        check(!rows[i].interpolated, "observed row flag");
    }
    check(rows[4].x1 == -2 && rows[4].y1 == 3 && rows[4].x2 == 8 && rows[4].y2 == 12,
          "reappearing output identity retains box and MOT ID despite a new UID");
    check(rows[4].conf == 0.625 && rows[0].conf == 0.75, "observed score used for confidence");
    check(format_mot_rows(rows) == format_mot_rows(sequence.finish()), "finish is repeatable");
    SequenceOutput independent;
    independent.add(packet(0, {track(42)}, 17));
    check(independent.finish().front().id == 1, "MOT IDs restart per sequence");
}

std::vector<MotRow> interpolation_case(const std::vector<FrameId>& observations) {
    SequenceOutput sequence;
    for (FrameId frame = 0; frame < 46; ++frame) {
        auto input = packet(frame);
        if (std::find(observations.begin(), observations.end(), frame) != observations.end()) {
            const auto f = static_cast<float>(frame);
            input.tracks.push_back(track(4, {f, 2 * f, 3 * f + 1, 4 * f + 2}, 0.5f));
        }
        sequence.add(input);
    }
    check(sequence.frames() == 46, "all input frames retained including trailing empty frames");
    return sequence.finish();
}

void interpolation_tests() {
    const auto filled = interpolation_case({2, 3, 4, 5, 41});
    check(filled.size() == 40, "five observations fill exactly 35 missing frames");
    check(filled.front().frame == 3 && filled.back().frame == 42, "no leading or trailing extrapolation");
    for (std::size_t i = 0; i < filled.size(); ++i) {
        const auto& row = filled[i];
        const auto frame = static_cast<FrameId>(i) + 2;
        check(row.frame == frame + 1 && row.id == 1, "interpolation frame and ID");
        check(row.x1 == frame && row.y1 == 2 * frame && row.x2 == 3 * frame + 1 && row.y2 == 4 * frame + 2,
              "linear interpolation coordinates");
        const bool interpolated = frame > 5 && frame < 41;
        check(row.interpolated == interpolated && row.conf == (interpolated ? -1.0 : 0.5),
              "interpolation flag and confidence");
    }
    const auto too_wide = interpolation_case({2, 3, 4, 5, 42});
    check(too_wide.size() == 5, "36 missing frames are not filled");
    const auto too_short = interpolation_case({2, 3, 4, 40});
    check(too_short.size() == 4, "four observations are not interpolated");

    SequenceOutput fractions({2, 2});
    fractions.add(packet(0, {track(7, {0, 1, 2, 3}), track(2, {6, 9, 12, 15})}));
    fractions.add(packet(1));
    fractions.add(packet(2));
    fractions.add(packet(3, {track(2, {9, 12, 15, 18}), track(7, {1, 3, 5, 7})}));
    const auto rows = fractions.finish();
    check(rows.size() == 8, "configured minimum and gap applied for multiple IDs");
    check(rows[2].x1 == 1.0 / 3.0 && rows[2].y1 == 1.0 + (1.0 / 3.0) * 2.0 &&
          rows[2].x2 == 3.0 && rows[2].y2 == 3.0 + (1.0 / 3.0) * 4.0,
          "interpolation uses binary64 fractions");
    check(rows[4].x1 == 2.0 / 3.0 && rows[4].conf == -1 && rows[4].interpolated,
          "second fractional interpolation");
    for (std::size_t i = 1; i < rows.size(); ++i)
        check(std::tie(rows[i - 1].frame, rows[i - 1].id) < std::tie(rows[i].frame, rows[i].id),
              "all observed and interpolated rows sorted together");
}

void failure_tests() {
    for (int scenario = 0; scenario < 4; ++scenario) {
        SequenceOutput sequence;
        sequence.add(packet(0, {track(3)}));
        auto bad = packet(1);
        if (scenario == 0) {
            bad.status = ResultStatus::failed;
            bad.committed_frame = 0;
            bad.fail_frame = 1;
            bad.frame_error = bad.fail_code = Error::compute;
        } else if (scenario == 1) {
            bad.status = ResultStatus::skipped_after_failure;
            bad.frame = 2;
            bad.committed_frame = 0;
            bad.fail_frame = 1;
            bad.fail_code = Error::compute;
        } else if (scenario == 2) {
            bad.frame = bad.committed_frame = 2;
        } else {
            ++bad.epoch;
        }
        sequence.add(bad);
        check(sequence.failed() && sequence.frames() == 1, "bad packet marks sequence failed");
        sequence.add(packet(1, {track(7)}));
        sequence.add(packet(2, {track(8)}));
        check(sequence.failed() && sequence.frames() == 1, "failure sticky and later packets ignored");
        rejects<std::logic_error>([&] { sequence.finish(); }, "failed finish throws logic_error");
    }
    SequenceOutput skipped;
    auto first = packet(0);
    first.status = ResultStatus::skipped_after_failure;
    skipped.add(first);
    check(skipped.failed() && skipped.frames() == 0, "SKIPPED also rejected at expected frame");
    SequenceOutput gap;
    gap.add(packet(1));
    check(gap.failed() && gap.frames() == 0, "first frame must be zero");
    SequenceOutput epoch;
    epoch.add(packet(0, {}, 0));
    epoch.add(packet(1, {}, 1));
    check(epoch.failed(), "epoch zero is a valid first epoch and remains fixed");
}

struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '_'; }
    std::string do_grouping() const override { return "\3"; }
};

void formatting_and_file_tests(const TemporaryDirectory& temp) {
    const std::array<MotRow, 2> rows{{
        {1, 1, -1.25, -2.5, 2.25, 1.5, 0.125, false},
        {std::numeric_limits<std::int64_t>::max(), std::numeric_limits<std::uint64_t>::max(),
         1000000000000.125, -0.0, 1000000000000.625, 1.25, 0.75, true}}};
    const std::string expected =
        "1,1,-1.250,-2.500,3.500,4.000,0.125,-1,-1,-1\n"
        "9223372036854775807,18446744073709551615,1000000000000.125,-0.000,0.500,1.250,-1,-1,-1,-1\n";
    const auto previous = std::locale();
    std::locale::global(std::locale(previous, new CommaDecimal));
    const auto actual = format_mot_rows(rows);
    std::locale::global(previous);
    check(actual == expected, "exact C-locale fixed-three text for negative and large values");
    check(format_mot_rows({}).empty(), "empty MOT text");
    const auto file = temp.path / "sequence.txt";
    const auto temporary = std::filesystem::path(file.string() + ".tmp");
    write_mot_file(file, rows);
    check(read_text_file(file) == expected, "MOT file has exactly formatted bytes");
    check(!std::filesystem::exists(temporary), "successful MOT write leaves no temporary file");
    write_mot_file(file, {});
    check(read_text_file(file).empty(), "successful MOT replacement may be empty");

    const auto unwritable = temp.path / "unwritable";
    std::filesystem::create_directory(unwritable);
    std::filesystem::permissions(unwritable, std::filesystem::perms::owner_read | std::filesystem::perms::owner_exec);
    const auto blocked = unwritable / "sequence.txt";
    rejects<std::runtime_error>([&] { write_mot_file(blocked, rows); }, "unwritable directory rejects MOT write");
    check(!std::filesystem::exists(blocked) && !std::filesystem::exists(blocked.string() + ".tmp"),
          "failed open leaves neither output nor temporary");
    std::filesystem::permissions(unwritable, std::filesystem::perms::owner_all);

    const auto directory = temp.path / "rename-target";
    std::filesystem::create_directory(directory);
    write_text_file(directory / "marker", "preserved");
    rejects<std::runtime_error>([&] { write_mot_file(directory, rows); }, "rename failure rejected");
    check(read_text_file(directory / "marker") == "preserved" &&
          !std::filesystem::exists(directory.string() + ".tmp"), "rename failure preserves destination and removes temporary");
    if (std::filesystem::exists("/dev/full")) {
        write_text_file(file, "previous");
        std::filesystem::create_symlink("/dev/full", temporary);
        rejects<std::runtime_error>([&] { write_mot_file(file, rows); }, "checked close reports buffered write failure");
        check(read_text_file(file) == "previous" && !std::filesystem::exists(temporary),
              "write failure preserves destination and removes temporary");
    }
}

void record_tests(const TemporaryDirectory& temp) {
    DetectionRecords records{0x123456789abcdef0ULL, std::vector<DetectionCandidates>(2)};
    for (std::size_t i = 0; i < records.frames.size(); ++i) {
        auto& frame = records.frames[i];
        frame.header = {records.epoch, i, kPostprocessSchema, 2, kPostprocessCapacity, 0, kPostprocessDone,
                        1, 1, 3, 4, 5, 6, 0};
        frame.rows[0] = {-1.5f, 2.5f, 8.0f, 20.0f, 0.75f, 0, 23, kDetectionKept};
        frame.rows[1] = {3.0f, 4.0f, 9.0f, 15.0f, 0.5f, 0, 27, kDetectionSuppressed};
        frame.rows.back().x1 = -0.0f;
    }
    const auto file = temp.path / "records.scdr";
    const auto rewrite = temp.path / "rewrite.scdr";
    write_detection_records(file, records);
    const auto bytes = read_text_file(file);
    check(bytes.size() == 28 + 2 * sizeof(DetectionCandidates) && bytes.substr(0, 4) == "SCDR",
          "SCDR header and total byte size");
    const auto decoded = read_detection_records(file);
    const auto memory_decoded = decode_detection_records(bytes);
    check(memory_decoded.epoch == decoded.epoch && memory_decoded.frames.size() == decoded.frames.size() &&
          std::memcmp(memory_decoded.frames.data(), decoded.frames.data(), bytes.size() - 28) == 0,
          "SCDR decode uses exactly the immutable receipt bytes");
    check(decoded.epoch == records.epoch && decoded.frames.size() == 2, "SCDR epoch and frame count roundtrip");
    check(std::memcmp(decoded.frames.data(), records.frames.data(), 2 * sizeof(DetectionCandidates)) == 0,
          "SCDR preserves all raw candidate bytes");
    write_detection_records(rewrite, decoded);
    check(read_text_file(rewrite) == bytes, "SCDR rewrite byte identical");
    write_detection_records(rewrite, {7, {}});
    const auto empty = read_detection_records(rewrite);
    check(empty.epoch == 7 && empty.frames.empty() && read_text_file(rewrite).size() == 28, "empty SCDR stream");

    const auto malformed = temp.path / "bad.scdr";
    const auto reject_bytes = [&](const std::string& contents, const char* message) {
        write_text_file(malformed, contents);
        rejects<std::runtime_error>([&] { read_detection_records(malformed); }, message);
        rejects<std::runtime_error>([&] { decode_detection_records(contents); }, message);
    };
    auto bad = bytes;
    bad[0] = 'X';
    reject_bytes(bad, "reader rejects magic");
    bad = bytes;
    const std::uint32_t version = 2;
    std::memcpy(bad.data() + 4, &version, sizeof(version));
    reject_bytes(bad, "reader rejects version");
    bad = bytes;
    const std::uint32_t size = sizeof(DetectionCandidates) + 1;
    std::memcpy(bad.data() + 8, &size, sizeof(size));
    reject_bytes(bad, "reader rejects struct size");
    for (std::size_t length = 0; length < 28; ++length)
        reject_bytes(bytes.substr(0, length), "reader rejects truncated header");
    reject_bytes(bytes.substr(0, 28), "reader rejects missing records");
    reject_bytes(bytes.substr(0, bytes.size() - 1), "reader rejects truncated record");
    reject_bytes(bytes + "x", "reader rejects trailing byte");
    reject_bytes(bytes + std::string(sizeof(DetectionCandidates), '\0'), "reader rejects trailing record");
}
} // namespace

int main() {
    try {
        TemporaryDirectory temp;
        mapping_tests();
        interpolation_tests();
        failure_tests();
        formatting_and_file_tests(temp);
        record_tests(temp);
        std::cout << "mot_output_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mot_output_tests: " << error.what() << '\n';
        return 1;
    }
}

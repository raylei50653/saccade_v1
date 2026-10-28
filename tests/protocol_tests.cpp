#include "saccade/protocol.hpp"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace saccade;
std::size_t checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template <typename F> void rejected(F&& action, const char* message) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    check(threw, message);
}
void run_frame(FailureModel& model, FrameId frame, int failure_phase = -1) {
    model.begin(frame);
    if (failure_phase == 0) model.add_error(Error::producer);
    model.fft();
    if (failure_phase == 1) model.add_error(Error::numeric);
    model.prepare_stage();
    model.publish_spectrum();
    if (failure_phase == 2) model.add_error(Error::overflow);
    model.prepare_stage();
    model.predict();
    if (failure_phase == 3) model.add_error(Error::compute);
    model.prepare_stage();
    model.update();
    if (failure_phase == 4) model.add_error(Error::protocol);
    model.commit();
}
void test_failures() {
    FailureModel first(42);
    check(first.read_slot() == 1 && first.write_slot() == 0, "bootstrap slots");
    check(first.committed_tracks() == ModelBuffer{}, "bootstrap empty state initialized");
    check(first.committed_spectrum() == nullptr, "no uninitialized first spectrum read");
    check(first.control().current_frame == no_frame && first.control().committed_frame == no_frame &&
          first.control().fail_frame == no_frame && first.control().fail_code == Error::none &&
          first.control().frame_status == Error::none && !first.control().run_stage, "control initialization");
    rejected([&] { first.packet(); }, "packet before commit");
    run_frame(first, 0);
    check(first.read_slot() == 0 && first.committed_tracks().words[0] == 1, "first frame predicts once");
    check(first.packet().status == ResultStatus::ok, "first frame commits");
    rejected([&] { first.begin(0); }, "completed frame cannot be resubmitted");
    run_frame(first, 1);
    check(first.read_slot() == 1 && first.committed_tracks().words[0] == 2, "second frame predicts once");

    for (FrameId fail_at : {FrameId{0}, FrameId{1}, FrameId{2}, FrameId{3}}) {
        for (int phase = 0; phase < 5; ++phase) {
            FailureModel model(77);
            for (FrameId k = 0; k < fail_at; ++k) run_frame(model, k);
            const auto tracks = model.committed_tracks();
            const auto* spectrum = model.committed_spectrum();
            const ModelBuffer saved_spectrum = spectrum ? *spectrum : ModelBuffer{};
            run_frame(model, fail_at, phase);
            const auto failure = model.packet();
            check(failure.status == ResultStatus::failed && failure.frame == fail_at, "failing frame packet");
            check(failure.fail_frame == fail_at && failure.fail_code != Error::none, "first failure sticky");
            check(failure.committed_frame == fail_at - 1, "failing frame does not commit");
            check(model.committed_tracks() == tracks, "failure preserves committed tracks");
            check(!spectrum || *model.committed_spectrum() == saved_spectrum, "failure preserves committed spectrum");
            decode_packet(encode_packet(failure));
            for (FrameId k = fail_at + 1; k < fail_at + 5; ++k) {
                model.begin(k);
                model.add_error(Error::numeric | Error::overflow); // Secondary errors cannot replace first.
                model.fft(); // Already enqueued cuFFT still completes.
                model.prepare_stage();
                model.publish_spectrum(); model.predict(); model.update();
                model.commit();
                const auto skipped = model.packet();
                check(skipped.status == ResultStatus::skipped_after_failure && skipped.frame == k,
                      "later submitted frame remains skipped");
                check(skipped.fail_frame == fail_at && skipped.fail_code == failure.fail_code,
                      "first failure cannot be replaced");
                check(skipped.committed_frame == fail_at - 1 && skipped.tracks.empty(), "skipped no commit or rows");
                check(model.committed_tracks() == tracks, "skipped preserves tracks");
                check(!spectrum || *model.committed_spectrum() == saved_spectrum, "skipped FFT cannot overwrite spectrum");
                check(model.scratch().frame == k, "unconditional FFT is restricted to scratch");
                decode_packet(encode_packet(skipped));
            }
        }
    }

    FailureModel frozen(0);
    frozen.begin(0); frozen.fft(); frozen.prepare_stage(); frozen.publish_spectrum(); frozen.predict();
    const bool block0 = frozen.control().run_stage;
    frozen.add_error(Error::numeric); // One block observes an error during this grid.
    const bool block1 = frozen.control().run_stage;
    check(block0 && block1, "same grid blocks read frozen execution decision");
    frozen.update(); // Current grid finishes its scratch writes despite the error.
    frozen.prepare_stage();
    check(!frozen.control().run_stage, "next grid observes preceding error");
    frozen.commit();
    check(frozen.packet().status == ResultStatus::failed, "unguarded commit must latch failed stage");

    FailureModel aggregated(0);
    aggregated.begin(0); aggregated.add_error(Error::producer); aggregated.add_error(Error::overflow);
    aggregated.commit();
    check(aggregated.packet().fail_code == (Error::producer | Error::overflow), "first frame errors OR together");
    decode_packet(encode_packet(aggregated.packet()));

    FailureModel missing(1);
    missing.begin(0); missing.commit();
    check(missing.packet().fail_code == Error::protocol, "missing state delivery cannot commit");
    FailureModel wrong(1);
    run_frame(wrong, 2);
    check(wrong.packet().fail_code == Error::protocol && wrong.control().committed_frame == no_frame,
          "out-of-order first frame cannot commit");
    FailureModel repeated(1);
    repeated.begin(0); repeated.fft(); repeated.prepare_stage(); repeated.publish_spectrum(); repeated.predict(); repeated.predict(); repeated.commit();
    check(repeated.packet().status == ResultStatus::failed, "duplicate predict rejected");
    FailureModel no_update(1);
    no_update.begin(0); no_update.fft(); no_update.prepare_stage(); no_update.publish_spectrum(); no_update.predict(); no_update.commit();
    check(no_update.packet().fail_code == Error::protocol, "missing update cannot commit");
    FailureModel twice_update(1);
    twice_update.begin(0); twice_update.fft(); twice_update.prepare_stage(); twice_update.publish_spectrum(); twice_update.predict(); twice_update.update(); twice_update.update(); twice_update.commit();
    check(twice_update.packet().fail_code == Error::protocol, "duplicate update rejected");
    FailureModel wrong_order(1);
    wrong_order.begin(0); wrong_order.prepare_stage(); wrong_order.predict(); wrong_order.commit();
    check(wrong_order.packet().fail_code == Error::protocol, "predict before spectrum rejected");
    FailureModel wrong_update(1);
    wrong_update.begin(0); wrong_update.prepare_stage(); wrong_update.update(); wrong_update.commit();
    check(wrong_update.packet().fail_code == Error::protocol, "update before predict rejected");
}
ResultPacket example() {
    ResultPacket p;
    p.epoch = 0x0102030405060708ULL;
    p.frame = p.committed_frame = 0;
    TrackRow row;
    row.uid = 0x1122334455667788ULL; row.output_id = 9;
    row.class_id = 3; row.flags = 1;
    row.xyxy = {-0.0F, 1.0F, 2.5F, 4.0F};
    row.observed_score = 0.5F; row.association_score = 0.25F;
    p.tracks.push_back(row);
    return p;
}
void write_file(const std::filesystem::path& path, std::span<const std::uint8_t> bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("cannot write fixture");
}
void test_packets() {
    const auto p = example();
    const auto bytes = encode_packet(p);
    const std::array<std::uint8_t, 16> prefix{'S','C','R','P',1,0,80,0,0x50,0x30,0,0,0,1,0,0};
    check(std::equal(prefix.begin(), prefix.end(), bytes.begin()), "golden packet profile");
    check(bytes[16] == 8 && bytes[23] == 1, "epoch little endian");
    check(std::all_of(bytes.begin() + 40, bytes.begin() + 48, [](auto b){return b == 255;}), "signed -1 sentinel");
    check(bytes[80] == 0x88 && bytes[87] == 0x11 && bytes[88] == 9, "row UID and output ID");
    check(bytes[104] == 0 && bytes[107] == 0, "negative zero canonicalized");
    check(bytes[110] == 0x80 && bytes[111] == 0x3f && bytes[114] == 0x20 && bytes[115] == 0x40,
          "golden IEEE754 float encoding");
    check(bytes[122] == 0 && bytes[123] == 0x3f && bytes[126] == 0x80 && bytes[127] == 0x3e,
          "explicit score encoding");
    check(std::all_of(bytes.begin() + 64, bytes.begin() + 80, [](auto b){return b == 0;}), "reserved bytes initialized");
    check(std::all_of(bytes.begin() + 128, bytes.end(), [](auto b){return b == 0;}), "unused rows initialized");
    check(encode_packet(decode_packet(bytes)) == bytes, "complete packet roundtrip");
    auto positive_zero = p; positive_zero.tracks[0].xyxy[0] = 0;
    check(encode_packet(positive_zero) == bytes, "zero normalization deterministic");
    check(compare_packet_bytes(bytes, bytes).equal, "identical bytes compare");
    for (const auto offset : {std::size_t{0}, std::size_t{64}, std::size_t{128}, packet_bytes - 1}) {
        auto changed = bytes; changed[offset] ^= 1;
        const auto result = compare_packet_bytes(bytes, changed);
        check(!result.equal && result.offset == offset, "all bytes including padding compared");
        rejected([&] { decode_packet(changed); }, "noncanonical packet rejected");
    }
    rejected([&] { compare_packet_bytes(std::span<const std::uint8_t>{}, bytes); }, "empty comparison rejected");
    rejected([&] { compare_packet_bytes(std::span<const std::uint8_t>(bytes).first(packet_bytes - 1), bytes); }, "truncated comparison rejected");
    std::vector<std::uint8_t> twice(bytes.begin(), bytes.end());
    twice.insert(twice.end(), bytes.begin(), bytes.end());
    check(!compare_packet_bytes(bytes, twice).equal, "complete stream count difference");
    auto invalid = p; invalid.tracks[0].xyxy[0] = std::numeric_limits<float>::infinity();
    rejected([&] { encode_packet(invalid); }, "nonfinite coordinate rejected");
    invalid = p; invalid.tracks.resize(packet_capacity + 1);
    rejected([&] { encode_packet(invalid); }, "capacity overflow rejected");
    invalid = p; invalid.tracks[0].flags = 2;
    rejected([&] { encode_packet(invalid); }, "predicted row cannot claim observation scores");
    invalid = p; invalid.status = ResultStatus::failed;
    rejected([&] { encode_packet(invalid); }, "failed status cannot carry success rows");
    invalid = p; invalid.frame_error = static_cast<Error>(128);
    rejected([&] { encode_packet(invalid); }, "unknown error bits rejected");
    invalid = p; invalid.tracks[0].xyxy[0] = std::numeric_limits<float>::quiet_NaN();
    rejected([&] { encode_packet(invalid); }, "NaN coordinate rejected");
    auto association = p; association.tracks[0].association_score = 123.0F;
    check(decode_packet(encode_packet(association)).tracks[0].association_score == 123.0F,
          "association score policy remains module-defined");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--fixtures") {
            const std::filesystem::path dir(argv[2]);
            std::filesystem::create_directories(dir);
            const auto bytes = saccade::encode_packet(example());
            write_file(dir / "same.bin", bytes);
            auto changed = bytes; changed.back() ^= 1;
            write_file(dir / "different-unused.bin", changed);
            changed = bytes; changed[64] ^= 1;
            write_file(dir / "different-reserved.bin", changed);
            write_file(dir / "truncated.bin", std::span<const std::uint8_t>(bytes).first(bytes.size() - 1));
            write_file(dir / "empty.bin", {});
            return 0;
        }
        if (argc != 1) throw std::invalid_argument("usage: protocol_tests [--fixtures DIRECTORY]");
        test_failures(); test_packets();
        std::cout << "protocol_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "protocol_tests: " << error.what() << '\n';
        return 1;
    }
}

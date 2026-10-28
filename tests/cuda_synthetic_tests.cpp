#include "saccade/synthetic_runtime.hpp"
#include "saccade/file_io.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace saccade;
std::size_t checks = 0, sessions = 0;
std::filesystem::path evidence_directory;
void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
Error expected_failure(const SyntheticConfig& config) {
    if (config.fail_stage == "before_spectrum") return Error::numeric;
    if (config.fail_stage == "after_spectrum") return Error::overflow;
    if (config.fail_stage == "after_predict") return Error::compute;
    if (config.fail_stage == "after_update") return Error::protocol;
    return Error::none;
}
// The fixture oracle uses its own closed-form answers, not a second execution
// of the GPU's state machine or copies of its intermediate device buffers.
ResultPacket expected_packet(const SyntheticConfig& config, FrameId frame) {
    ResultPacket packet;
    packet.epoch = config.epoch;
    packet.frame = frame;
    packet.committed_frame = frame;
    if (config.fail_frame >= 0 && frame >= config.fail_frame) {
        packet.committed_frame = config.fail_frame - 1;
        packet.fail_frame = config.fail_frame;
        packet.fail_code = expected_failure(config);
        packet.status = frame == config.fail_frame ? ResultStatus::failed : ResultStatus::skipped_after_failure;
        packet.frame_error = frame == config.fail_frame ? packet.fail_code : Error::none;
        return packet;
    }
    if (frame % 2 == 0) return packet; // Explicit successful empty fixture.
    for (std::uint32_t index = 0; index < 2; ++index) {
        TrackRow row;
        row.uid = static_cast<std::uint64_t>(2 * frame) + index + 1;
        row.output_id = 100 + row.uid;
        row.class_id = index;
        row.flags = 1;
        const auto base = static_cast<float>(config.seed % 7 + 2 * frame + 1 + 10 * index);
        row.xyxy = {base, base + 1.0F, base + 2.0F, base + 3.0F};
        row.observed_score = 0.5F;
        row.association_score = 0.25F;
        packet.tracks.push_back(row);
    }
    return packet;
}
std::vector<std::uint8_t> expected_bytes(const SyntheticConfig& config, int count) {
    std::vector<std::uint8_t> result;
    for (int frame = 0; frame < count; ++frame) {
        const auto bytes = encode_packet(expected_packet(config, frame));
        result.insert(result.end(), bytes.begin(), bytes.end());
    }
    return result;
}
void check_packets(const SyntheticConfig& config, const SyntheticRun& run, int count) {
    check(run.packets.size() == static_cast<std::size_t>(count) * packet_bytes, "full raw packet count");
    check(run.packet_copies == static_cast<std::size_t>(count), "one full D2H per completed frame");
    if (count == 0) return;
    const auto oracle = expected_bytes(config, count);
    const auto comparison = compare_packet_bytes(run.packets, oracle);
    check(comparison.equal, "raw GPU bytes differ from closed-form oracle at byte " + std::to_string(comparison.offset));
    for (int frame = 0; frame < count; ++frame) {
        const auto packet = decode_packet(std::span<const std::uint8_t>(run.packets).subspan(
            static_cast<std::size_t>(frame) * packet_bytes, packet_bytes));
        check(packet.frame == frame && packet.epoch == config.epoch, "decoded packet identity");
        if (config.fail_frame < 0 || frame < config.fail_frame)
            check(packet.tracks.size() == static_cast<std::size_t>((frame % 2) * 2), "independent empty/nonempty counts");
    }
}
void check_state(const SyntheticConfig& config, const SyntheticRun& run, FrameId committed,
                 FrameId current, FrameId scratch, std::uint64_t fft_count) {
    const auto& state = run.snapshot;
    check(state.committed_frame == committed, "terminal committed version");
    check(state.current_frame == current, "terminal current frame");
    check(state.tracks_frame == committed && state.spectrum_frame == committed, "TS/SPEC share committed version");
    check(state.committed_slot == static_cast<std::uint32_t>((committed + 2) % 2), "committed parity including empty slot");
    check(state.predict_count == static_cast<std::uint64_t>(committed + 1) &&
          state.update_count == static_cast<std::uint64_t>(committed + 1), "committed TS counts closed form");
    const auto count = static_cast<std::uint64_t>(committed + 1);
    const auto pixels = static_cast<std::uint64_t>(config.width * config.height);
    const auto expected_sum = pixels * (count * (config.seed % 13 + 1) + count * (count == 0 ? 0 : count - 1) / 2);
    check(state.input_sum == expected_sum, "committed TS FFT-derived input sum closed form");
    check(state.scratch_frame == scratch && state.fft_completed == fft_count, "all queued FFTs completed in isolated scratch");
    const bool failed = config.fail_frame >= 0 && current >= config.fail_frame;
    check(state.fail_frame == (failed ? config.fail_frame : -1), "sticky first failure frame");
    check(state.fail_code == (failed ? static_cast<std::uint32_t>(expected_failure(config)) : 0), "sticky first failure code");
    check(state.spectrum_bits.size() == static_cast<std::size_t>(config.height * (config.width / 2 + 1) * 2),
          "terminal-only full committed R2C spectrum present");
    check(!state.tracks_bytes.empty(), "terminal-only complete committed TS payload present");
    const float dc = committed < 0 ? 0.0F : static_cast<float>(pixels * (config.seed % 13 + committed + 1));
    for (std::size_t index = 0; index < state.spectrum_bits.size(); ++index) {
        const auto expected = index == 0 ? std::bit_cast<std::uint32_t>(dc) : 0U;
        check(state.spectrum_bits[index] == expected, "committed FFT spectrum closed-form word " + std::to_string(index));
    }
}
void check_ledger(const SyntheticConfig& config, const SyntheticRun& run) {
    using Key = std::pair<std::string, int>;
    std::map<Key, long long> records, waits, host_waits;
    std::set<long long> packets_recorded, packets_consumed, stage_recorded;
    bool d_terminal = false, m_terminal = false, snapshot_seen = false, finalized = false;
    for (const auto& line : run.ledger) {
        if (finalized) check(false, "no ledger operation follows successful resource finalization");
        std::istringstream input(line);
        std::string operation, resource, slot_text, generation_text;
        input >> operation >> resource;
        if (operation == "initialize" || operation == "cancel") continue;
        if (operation == "secondary_error") {
            std::string bits;
            input >> bits;
            check(resource.starts_with("frame=") && bits == "bits=10", "secondary diagnostics recorded");
            check(config.fail_frame >= 0 && std::stoll(resource.substr(6)) > config.fail_frame,
                  "secondary diagnostic cannot masquerade as first failure");
            continue;
        }
        if (operation == "terminal_snapshot") {
            check(d_terminal && m_terminal, "terminal diagnostics follow both stream drains");
            check(config.terminal_snapshot && !snapshot_seen, "terminal diagnostics follow resolved observation setting exactly once");
            snapshot_seen = true;
            continue;
        }
        if (operation == "finalize") {
            check(resource == "resources_released" && d_terminal && m_terminal,
                  "successful resource release follows both terminal drains");
            check(snapshot_seen == config.terminal_snapshot,
                  "resource release follows optional terminal diagnostics");
            finalized = true;
            continue;
        }
        input >> slot_text >> generation_text;
        check(slot_text.starts_with("slot=") && generation_text.starts_with("gen="), "well-formed event ledger");
        const int slot = std::stoi(slot_text.substr(5));
        const auto generation = std::stoll(generation_text.substr(4));
        const Key key{resource, slot};
        if (operation == "record") {
            const auto previous = records.find(key);
            check(previous == records.end() || previous->second < generation, "event records strictly increase generation");
            if (resource == "staged") {
                check(generation == 0 || packets_recorded.contains(generation - 1), "D(k) starts only after M(k-1) submission");
                if (generation >= config.n_h)
                    check(host_waits.contains(key) && host_waits.at(key) == generation - config.n_h,
                          "pinned host staging reused only after exact generation completes");
                stage_recorded.insert(generation);
            } else if (resource == "mi_ready" || resource == "cs_ready") {
                check(stage_recorded.contains(generation), "producer stages this frame before slot publication");
                const bool motion = resource == "mi_ready";
                const int size = motion ? config.n_mi : config.n_cs;
                if (generation >= size) {
                    const Key consumed{motion ? "mi_consumed" : "cs_consumed", slot};
                    check(waits.contains(consumed) && waits.at(consumed) == generation - size,
                          "each device slot has its own explicit consumed wait before reuse");
                }
            } else if (resource == "mi_consumed" || resource == "cs_consumed") {
                const Key ready{resource == "mi_consumed" ? "mi_ready" : "cs_ready", slot};
                check(waits.contains(ready) && waits.at(ready) == generation,
                      "slot returned only after consumer waits for exact ready generation");
            } else if (resource == "pkt") {
                check(!packets_recorded.contains(generation), "one packet record per frame");
                if (generation >= config.n_rp)
                    check(packets_consumed.contains(generation - config.n_rp), "host packet slot consumed before reuse");
                packets_recorded.insert(generation);
            } else if (resource == "d_terminal" || resource == "m_terminal") {
                check(packets_recorded == packets_consumed, "all complete packets consumed before terminal release");
            } else check(false, "unknown event ledger resource");
            records[key] = generation;
        } else if (operation == "wait" || operation == "host_wait" || operation == "consume") {
            check(records.contains(key) && records.at(key) == generation,
                  "every wait/consume binds latest successfully recorded event generation");
            if (operation == "wait") {
                waits[key] = generation;
                if (resource == "mi_ready") {
                    const Key cs_ready{"cs_ready", static_cast<int>(generation % config.n_cs)};
                    check(records.contains(cs_ready) && records.at(cs_ready) == generation,
                          "M(k) wait enqueued after full D(k) publication");
                }
            } else if (operation == "host_wait") {
                host_waits[key] = generation;
                if (resource == "d_terminal") d_terminal = true;
                if (resource == "m_terminal") m_terminal = true;
            } else {
                check(resource == "pkt" && host_waits.contains(key) && host_waits.at(key) == generation,
                      "host consumes only a completed packet generation");
                if (config.fixed_submit && config.n_rp >= config.frames && generation == config.fail_frame)
                    check(packets_recorded.size() == static_cast<std::size_t>(config.frames),
                          "fixed failure tail completely enqueued before host observes FAILED");
                check(packets_consumed.insert(generation).second, "packet generation consumed exactly once");
            }
        } else check(false, "unknown event ledger operation");
    }
    check(d_terminal && m_terminal, "both terminal records and waits appear in ledger");
    check(finalized, "successful return records checked resource finalization");
    check(packets_recorded.size() == run.packet_copies && packets_recorded == packets_consumed,
          "ledger packet generations account for all original D2H copies");
}
void save_evidence(const SyntheticConfig& config, const SyntheticRun& run, const std::string& label) {
    if (evidence_directory.empty()) return;
    std::ostringstream number;
    number << "case" << std::setw(4) << std::setfill('0') << sessions;
    const auto directory = evidence_directory / number.str();
    std::filesystem::create_directories(directory);
    // This is the untouched GPU D2H stream. It is saved before either decode or
    // oracle comparison, and is never replaced with the CPU's expected bytes.
    write_binary_file(directory / "packets.bin", run.packets);
    write_text_file(directory / "resolved.cfg", canonical_config(config));
    write_text_file(directory / "behavior.cfg", behavior_config(config));
    write_text_file(directory / "config.sha256", config_hash(config) + "\n");
    write_text_file(directory / "behavior.sha256", behavior_hash(config) + "\n");
    write_text_file(directory / "identity.txt", run.identity);
    std::ostringstream audit;
    audit << "label=" << label << '\n'
          << "d_frames=" << run.d_frames << '\n'
          << "m_frames=" << run.m_frames << '\n'
          << "packet_copies=" << run.packet_copies << '\n'
          << "d_drained=" << std::boolalpha << run.d_drained << '\n'
          << "m_drained=" << run.m_drained << '\n'
          << "cancelled=" << run.cancelled << '\n'
          << "stopped_on_failure=" << run.stopped_on_failure << '\n';
    for (const auto& line : run.ledger) audit << line << '\n';
    write_text_file(directory / "audit.txt", audit.str());
    if (config.terminal_snapshot) {
        const auto& state = run.snapshot;
        std::ostringstream snapshot;
        snapshot << "current_frame=" << state.current_frame << '\n'
                 << "committed_frame=" << state.committed_frame << '\n'
                 << "committed_slot=" << state.committed_slot << '\n'
                 << "fail_frame=" << state.fail_frame << '\n'
                 << "fail_code=" << state.fail_code << '\n'
                 << "tracks_frame=" << state.tracks_frame << '\n'
                 << "spectrum_frame=" << state.spectrum_frame << '\n'
                 << "scratch_frame=" << state.scratch_frame << '\n'
                 << "predict_count=" << state.predict_count << '\n'
                 << "update_count=" << state.update_count << '\n'
                 << "input_sum=" << state.input_sum << '\n'
                 << "fft_completed=" << state.fft_completed << '\n'
                 << "spectrum_bits=";
        for (std::size_t index = 0; index < state.spectrum_bits.size(); ++index) {
            if (index) snapshot << ',';
            snapshot << std::hex << std::setw(8) << std::setfill('0') << state.spectrum_bits[index];
        }
        snapshot << '\n' << "tracks_bytes=";
        for (const auto byte : state.tracks_bytes)
            snapshot << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
        snapshot << '\n';
        write_text_file(directory / "terminal-snapshot.txt", snapshot.str());
    }
    std::ostringstream index;
    index << number.str() << '\t' << label << '\t' << config.mode << '\t'
          << config_hash(config) << '\t' << behavior_hash(config) << '\t'
          << run.packets.size() << '\n';
    append_text_file(evidence_directory / "index.tsv", index.str());
}
SyntheticRun execute(const SyntheticConfig& config, const std::string& label) {
    auto run = run_synthetic(config);
    ++sessions;
    save_evidence(config, run, label);
    check(run.d_drained && run.m_drained, "both terminal events completed before release");
    check(!run.identity.empty(), "runtime records build/compiler/CUDA/driver/GPU identity");
    check(!run.ledger.empty(), "generation ledger recorded");
    check_ledger(config, run);
    std::cout << "case=" << label << " mode=" << config.mode << " rings="
              << config.n_mi << ',' << config.n_cs << ',' << config.n_h << ',' << config.n_rp
              << " delay_seed=" << config.delay_seed << " packets=" << run.packet_copies
              << " config_sha256=" << config_hash(config) << '\n';
    return run;
}
void success_matrix(bool sanitizer) {
    const std::array<std::array<int, 4>, 4> rings{{{1,1,1,1}, {2,2,2,2}, {1,2,3,4}, {2,1,3,4}}};
    const std::array<std::uint64_t, 3> seeds{11,29,97};
    for (std::size_t r = 0; r < rings.size(); ++r) {
        if (sanitizer && r != 2) continue;
        for (auto seed : seeds) {
            if (sanitizer && seed != seeds.front()) continue;
            SyntheticConfig config;
            config.n_mi = rings[r][0]; config.n_cs = rings[r][1];
            config.n_h = rings[r][2]; config.n_rp = rings[r][3];
            config.frames = 16;
            config.delay_seed = seed;
            config.terminal_snapshot = true;
            config.mode = "serial";
            const auto first = execute(config, "serial-repeat-1");
            const auto second = execute(config, "serial-repeat-2");
            check_packets(config, first, config.frames);
            check_packets(config, second, config.frames);
            check(compare_packet_bytes(first.packets, second.packets).equal, "serial rerun complete raw bytes identical");
            check_state(config, first, config.frames - 1, config.frames - 1, config.frames - 1, config.frames);
            config.mode = "pipelined";
            const auto pipeline = execute(config, "pipelined");
            check_packets(config, pipeline, config.frames);
            check_state(config, pipeline, config.frames - 1, config.frames - 1, config.frames - 1, config.frames);
            check(compare_packet_bytes(first.packets, pipeline.packets).equal, "serial/pipelined complete raw bytes identical");
            check(pipeline.d_frames == config.frames && pipeline.m_frames == config.frames, "complete scheduled range");
            check(first.snapshot.spectrum_bits == pipeline.snapshot.spectrum_bits, "serial/pipelined terminal spectrum identical");
        }
    }
}
void failures(bool sanitizer) {
    SyntheticConfig baseline;
    baseline.frames = 3;
    baseline.terminal_snapshot = true;
    baseline.n_mi = 1; baseline.n_cs = 2; baseline.n_h = 8; baseline.n_rp = 8;
    const auto success = execute(baseline, "isolation-baseline-frames-0-through-2");
    check_packets(baseline, success, baseline.frames);
    check_state(baseline, success, 2, 2, 2, 3);
    for (const std::string stage : {"before_spectrum", "after_spectrum", "after_predict", "after_update"}) {
        if (sanitizer && stage != "after_update") continue;
        SyntheticConfig config = baseline;
        config.fail_frame = 3;
        config.fail_stage = stage;
        config.frames = 8;
        config.fixed_submit = true;
        config.mode = "serial";
        const auto serial = execute(config, "fixed-failure-" + stage);
        const auto repeated_serial = execute(config, "fixed-failure-repeat-" + stage);
        check(compare_packet_bytes(serial.packets, repeated_serial.packets).equal,
              "same fixed failure behavior serial rerun complete raw bytes identical");
        config.mode = "pipelined";
        const auto pipeline = execute(config, "fixed-failure-" + stage);
        for (const auto* run : {&serial, &repeated_serial, &pipeline}) {
            check_packets(config, *run, config.frames);
            check_state(config, *run, 2, 7, 7, 8);
            check(run->snapshot.spectrum_bits == success.snapshot.spectrum_bits, "failure and queued FFT tails preserve full committed SPEC payload");
            check(run->snapshot.tracks_bytes == success.snapshot.tracks_bytes,
                  "failure and queued FFT tails preserve every committed TS payload byte");
            check(run->snapshot.predict_count == success.snapshot.predict_count &&
                  run->snapshot.update_count == success.snapshot.update_count &&
                  run->snapshot.input_sum == success.snapshot.input_sum,
                  "failure and queued FFT tails preserve committed TS payload");
        }
        check(compare_packet_bytes(serial.packets, pipeline.packets).equal, "fixed identical failure submission range raw byte comparison");
        if (!sanitizer) {
            config.frames = 4; config.fail_frame = 0;
            const auto bootstrap_failure = execute(config, "first-frame-failure-" + stage);
            check_packets(config, bootstrap_failure, config.frames);
            check_state(config, bootstrap_failure, -1, 3, 3, 4);
        }
    }
    if (!sanitizer) {
        SyntheticConfig config = baseline;
        config.frames = 32;
        config.n_h = 3;
        config.n_rp = 4;
        config.fail_frame = 3;
        config.fail_stage = "after_predict";
        config.fixed_submit = false;
        const auto normal = execute(config, "ordinary-stop-on-observed-FAILED");
        check(normal.stopped_on_failure, "ordinary executor observes FAILED and stops submission");
        check(normal.d_frames >= 4 && normal.d_frames <= config.fail_frame + config.n_rp,
              "ordinary executor stops by next failed packet observation");
        check(normal.d_frames == normal.m_frames && normal.d_frames < config.frames, "ordinary submitted range ends at observed failure");
        check_packets(config, normal, normal.m_frames);
        check_state(config, normal, 2, normal.m_frames - 1, normal.m_frames - 1, normal.m_frames);
    }
}
void cancellations(bool sanitizer) {
    const std::array<std::string, 8> cuts{"after_h2d", "after_mi", "after_cs", "after_begin", "after_fft", "after_predict", "after_update", "after_packet"};
    for (std::size_t index = 0; index < cuts.size(); ++index) {
        if (sanitizer && index != 0 && index != 4) continue;
        for (const std::string mode : {"serial", "pipelined"}) {
            SyntheticConfig config;
            config.n_mi = 1; config.n_cs = 2; config.n_h = 3; config.n_rp = 4;
            config.frames = 16;
            config.cancel_frame = 7;
            config.cancel_cut = cuts[index];
            config.mode = mode;
            config.terminal_snapshot = true;
            const auto run = execute(config, "cancel-" + cuts[index]);
            check(run.cancelled && !run.stopped_on_failure, "cancellation distinct from failure and EOF");
            const int packet_count = index == 7 ? 8 : 7;
            check_packets(config, run, packet_count);
            check(run.d_frames == 8, "D-only or partial-M cancellation frame retained in ledger");
            check(run.m_frames == (index >= 3 ? 8 : 7), "only successfully enqueued begin counted");
            check_state(config, run, packet_count - 1, index >= 3 ? 7 : 6,
                        index >= 4 ? 7 : 6, index >= 4 ? 8 : 7);
        }
    }
}
void extra_shapes() {
    for (const auto shape : {std::array<int, 2>{16,8}, std::array<int, 2>{32,32}}) {
        SyntheticConfig config;
        config.width = shape[0]; config.height = shape[1];
        config.frames = 8;
        config.seed = 0xffffffffULL;
        config.terminal_snapshot = true;
        const auto run = execute(config, "shape-and-maximum-input-seed");
        check_packets(config, run, config.frames);
        check_state(config, run, 7, 7, 7, 8);
    }
}
void observation_invariance() {
    // Preserve the original 72-case regression matrix above, then vary only
    // the allowlisted terminal observation field for identical submitted work.
    for (const bool failed : {false, true}) {
        for (const std::string mode : {"serial", "pipelined"}) {
            SyntheticConfig hidden;
            hidden.mode = mode;
            hidden.n_mi = 1; hidden.n_cs = 2; hidden.n_h = 8; hidden.n_rp = 8;
            hidden.frames = 8;
            hidden.terminal_snapshot = false;
            if (failed) {
                hidden.fail_frame = 3;
                hidden.fail_stage = "after_update";
                hidden.fixed_submit = true;
            }
            auto observed = hidden;
            observed.terminal_snapshot = true;
            check(behavior_hash(hidden) == behavior_hash(observed),
                  "terminal observation does not change resolved behavior hash");
            check(config_hash(hidden) != config_hash(observed),
                  "terminal observation remains visible in full resolved config hash");
            const std::string label = failed ? "observation-failure-after_update" : "observation-success";
            const auto without_snapshot = execute(hidden, label + "-off");
            const auto with_snapshot = execute(observed, label + "-on");
            check_packets(hidden, without_snapshot, hidden.frames);
            check_packets(observed, with_snapshot, observed.frames);
            check(compare_packet_bytes(without_snapshot.packets, with_snapshot.packets).equal,
                  "normal-path raw GPU packets equal terminal-observation packets byte for byte");
            check(without_snapshot.snapshot.tracks_bytes.empty() && without_snapshot.snapshot.spectrum_bits.empty(),
                  "disabled terminal observation returns no TS/SPEC readback buffers");
            check(without_snapshot.snapshot.current_frame == no_frame && without_snapshot.snapshot.fft_completed == 0,
                  "disabled terminal observation leaves diagnostic fields unpopulated");
            check(without_snapshot.d_frames == hidden.frames && without_snapshot.m_frames == hidden.frames &&
                  with_snapshot.d_frames == observed.frames && with_snapshot.m_frames == observed.frames,
                  "observation pair retains identical complete submitted frame range");
            check_state(observed, with_snapshot, failed ? 2 : 7, 7, 7, 8);
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        bool sanitizer = false;
        for (int argument = 1; argument < argc; ++argument) {
            const std::string value(argv[argument]);
            if (value == "--sanitizer" && !sanitizer) sanitizer = true;
            else if (value != "--sanitizer" && evidence_directory.empty()) evidence_directory = value;
            else throw std::invalid_argument("usage: cuda_synthetic_tests [--sanitizer] [OUTPUT_DIRECTORY]");
        }
        if (!evidence_directory.empty()) {
            std::filesystem::create_directories(evidence_directory);
            write_text_file(evidence_directory / "index.tsv", "case\tlabel\tmode\tconfig_sha256\tbehavior_sha256\tpacket_bytes\n");
        }
        success_matrix(sanitizer);
        failures(sanitizer);
        cancellations(sanitizer);
        if (!sanitizer) extra_shapes();
        observation_invariance();
        std::cout << "cuda_synthetic_tests: " << checks << " checks, " << sessions << " GPU sessions passed"
                  << (sanitizer ? " (representative sanitizer subset)" : " (full finite acceptance matrix)") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_synthetic_tests: " << error.what() << '\n';
        return 1;
    }
}

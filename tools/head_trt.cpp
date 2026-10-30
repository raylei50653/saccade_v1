// S04 complete-head engine lifecycle (docs/HEAD_TRT.md §6):
//   build   BUNDLE BUNDLE_SHA256 ENGINE BUILD_REPORT [--debug]
//   run     ENGINE ENGINE_SHA256 BUNDLE BUNDLE_SHA256 REPORT [--features FILE FILE_SHA256] [--dump DIR]
//   run-unregistered ENGINE
// `build` writes the plan with checked I/O, reads it back and records the
// builder options, identity and per-layer tactics. `run` is a new process:
// plugin registration, deserialization, frame-session initialization, the
// policy comparison of all six outputs for the fixed inputs, determinism,
// graph replay, a failure sequence, checked close and a normal exit.
// `--features` adds real P3/P4/P5 frames (FP32 little-endian NCHW, frames
// back to back) after the fixed inputs, under the same policy and checks;
// `--dump` writes each input's six committed outputs to DIR/<label>.f32.
#include "saccade/file_io.hpp"
#include "saccade/head_compare.hpp"
#include "saccade/head_reference.hpp"
#include "saccade/head_trt_network.hpp"
#include "saccade/scan_trt_frame.hpp"
#include "saccade/scan_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/sha256.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
using namespace nvinfer1;
using namespace saccade;

class Logger final : public ILogger {
public:
    std::atomic<unsigned> errors{0};
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kERROR) ++errors;
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

std::span<const std::uint8_t> bytes(const std::string& value) { return {reinterpret_cast<const std::uint8_t*>(value.data()), value.size()}; }

std::string checked_read(const std::string& path, const std::string& sha, const char* what) {
    auto data = read_text_file(path);
    const auto got = sha256_hex(data);
    require(got == sha, std::string(what) + " SHA-256 " + got + " differs from the expected " + sha);
    return data;
}

double seconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void build(const std::string& bundle_path, const std::string& bundle_sha, const std::string& engine_path, const std::string& report_path, bool debug,
           Logger& logger) {
    const auto weights = decode_head_bundle(checked_read(bundle_path, bundle_sha, "bundle"));
    const auto t0 = seconds();
    HeadBuildOptions options;
    options.debug_intermediates = debug;
    const auto engine = build_head_engine(logger, weights, options);
    const auto build_seconds = seconds() - t0;
    write_binary_file(engine_path, engine.plan);
    const auto readback = read_text_file(engine_path);
    require(readback.size() == engine.plan.size() && std::memcmp(readback.data(), engine.plan.data(), readback.size()) == 0,
            "engine readback differs from the serialized plan");
    register_scan_trt_plugins(*getPluginRegistry());
    std::ostringstream report;
    {
        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        require(runtime != nullptr, "createInferRuntime failed");
        std::unique_ptr<ICudaEngine> deserialized(runtime->deserializeCudaEngine(readback.data(), readback.size()));
        require(deserialized != nullptr, "deserializeCudaEngine of the written plan failed");
        report << "bundle_sha256=" << bundle_sha << "\nengine_bytes=" << readback.size() << "\nengine_sha256=" << sha256_hex(readback)
               << "\nbuild_seconds=" << build_seconds << '\n' << engine.build_options << scan_trt_plugin_identity() << describe_scan_engine(*deserialized);
        for (const auto& s : engine.statuses) report << "status_output=" << s << '\n';
        for (const auto& d : engine.debug_outputs) report << "debug_output=" << d << '\n';
        report << "layers_json=" << head_engine_layers(*deserialized) << '\n';
    }
    write_text_file(report_path, report.str());
    std::cout << "engine_sha256=" << sha256_hex(readback) << "\nengine_bytes=" << readback.size() << '\n';
}

// Whole P3/P4/P5 frames read from a checked file (HEAD_TRT §10).
std::vector<std::array<std::vector<float>, 3>> read_feature_frames(const std::string& path, const std::string& sha) {
    static_assert(std::endian::native == std::endian::little, "feature files are little-endian FP32");
    const auto data = checked_read(path, sha, "features");
    std::array<std::size_t, 3> counts{};
    std::size_t frame_bytes = 0;
    for (std::size_t level = 0; level < 3; ++level) {
        counts[level] = 1;
        for (const auto d : head_input_shape(static_cast<int>(level))) counts[level] *= static_cast<std::size_t>(d);
        frame_bytes += counts[level] * sizeof(float);
    }
    require(!data.empty() && data.size() % frame_bytes == 0, "features: " + std::to_string(data.size()) + " bytes is not a whole number of frames");
    std::vector<std::array<std::vector<float>, 3>> frames(data.size() / frame_bytes);
    std::size_t offset = 0;
    for (auto& frame : frames) {
        for (std::size_t level = 0; level < 3; ++level) {
            frame[level].resize(counts[level]);
            std::memcpy(frame[level].data(), data.data() + offset, counts[level] * sizeof(float));
            offset += counts[level] * sizeof(float);
        }
    }
    return frames;
}

std::vector<std::vector<float>> frame_inputs(const std::array<std::vector<float>, 3>& f) { return {f[0], f[1], f[2]}; }

std::array<std::span<const float>, 6> split(const ScanFrameSession& session, const std::vector<float>& arena) {
    std::array<std::span<const float>, 6> out;
    for (std::size_t k = 0; k < 6; ++k) {
        const auto& o = session.outputs().at(k);
        out[k] = std::span<const float>(arena).subspan(o.offset, o.count);
    }
    return out;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct RunOptions {
    std::string features, features_sha, dump_dir;
};

void run(const std::string& engine_path, const std::string& engine_sha, const std::string& bundle_path, const std::string& bundle_sha,
         const std::string& report_path, const RunOptions& options, Logger& logger) {
    const auto plan = checked_read(engine_path, engine_sha, "engine");
    const auto weights = decode_head_bundle(checked_read(bundle_path, bundle_sha, "bundle"));
    const auto spec = head_io_spec();
    std::ostringstream report;
    report << "engine_sha256=" << engine_sha << "\nbundle_sha256=" << bundle_sha << "\npolicy=" << kHeadPolicyVersion
           << "\ncuda_module_loading=" << (std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "default") << '\n';
    std::vector<std::string> labels;
    std::vector<std::array<std::vector<float>, 3>> features;
    for (const auto& [kind, seed] : std::vector<std::pair<std::string, std::uint64_t>>{{"asymmetric", 1}, {"random", 1}, {"random", 2}}) {
        labels.push_back(kind + "/" + std::to_string(seed));
        features.push_back(head_test_features(kind, seed));
    }
    if (!options.features.empty()) {
        auto frames = read_feature_frames(options.features, options.features_sha);
        report << "features_file=" << options.features << "\nfeatures_sha256=" << options.features_sha << "\nfeatures_frames=" << frames.size() << '\n';
        for (std::size_t i = 0; i < frames.size(); ++i) {
            labels.push_back("features/" + std::to_string(i));
            features.push_back(std::move(frames[i]));
        }
    }
    for (std::size_t i = 0; i < features.size(); ++i) {
        for (std::size_t level = 0; level < 3; ++level) {
            const auto& v = features[i][level];
            report << "input\t" << labels[i] << '\t' << kHeadInputNames[level] << "\tsha256="
                   << sha256_hex(std::string_view(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float))) << '\n';
        }
    }
    bool pass = true;
    std::vector<std::vector<float>> committed;
    try {
    {
        ScanFrameConfig cfg;
        cfg.slots = 2;
        ScanFrameSession session(bytes(plan), cfg, logger, nullptr, &spec);
        report << session.identity();
        for (std::size_t i = 0; i < features.size(); ++i) {
            const auto& label = labels[i];
            const auto result = session.run({frame_inputs(features[i])});
            require(result.published == std::vector<std::int64_t>{0} && result.committed_payload_frame == 0, label + ": frame not accepted");
            const auto reference = head_reference(weights, {features[i][0], features[i][1], features[i][2]});
            const auto comparison = compare_head_outputs(split(session, result.committed_payload), reference);
            report << comparison.report(label);
            pass = pass && comparison.pass;
            committed.push_back(result.committed_payload);
            // Same engine, same context, same input: bit-identical rerun.
            const auto again = session.run({frame_inputs(features[i])});
            require(same_bits(again.committed_payload, result.committed_payload), label + ": rerun differs bitwise");
            report << "determinism\t" << label << "\trerun_bit_identical\tPASS\n";
            if (!options.dump_dir.empty()) {
                auto name = label;
                std::replace(name.begin(), name.end(), '/', '-');
                const auto path = options.dump_dir + "/" + name + ".f32";
                const auto outputs = split(session, result.committed_payload);
                std::string blob;
                for (const auto& o : outputs) blob.append(reinterpret_cast<const char*>(o.data()), o.size() * sizeof(float));
                write_binary_file(path, bytes(blob));
                report << "dump\t" << label << '\t' << path << "\tsha256=" << sha256_hex(blob) << '\n';
            }
        }
        // Failure sequence: a NaN in P4 fails the frame; the later valid frame is skipped.
        auto bad = features[0];
        bad[1][12345] = std::numeric_limits<float>::quiet_NaN();
        ScanFrameConfig fixed;
        fixed.slots = 1;
        fixed.fixed_submit = true;
        ScanFrameSession failing(bytes(plan), fixed, logger, nullptr, &spec);
        const auto r = failing.run({frame_inputs(features[0]), frame_inputs(bad), frame_inputs(features[0])});
        require(r.outcomes.size() == 3 && r.outcomes[0].status == ResultStatus::ok && r.outcomes[1].status == ResultStatus::failed &&
                    r.outcomes[2].status == ResultStatus::skipped_after_failure && r.published == std::vector<std::int64_t>{0},
                "failure sequence is not OK / FAILED / SKIPPED");
        require(r.control.committed == 0 && same_bits(r.committed_payload, committed[0]), "failure moved the committed result");
        report << "failure_sequence\tOK,FAILED(" << static_cast<std::uint32_t>(r.outcomes[1].frame_error) << "),SKIPPED\tcommitted=0\tPASS\n";
        failing.close();
        session.close();
    }
    {
        ScanFrameConfig cfg;
        cfg.graphs = true;
        cfg.slots = 2;
        ScanFrameSession graph(bytes(plan), cfg, logger, nullptr, &spec);
        for (std::size_t i = 0; i < features.size(); ++i) {
            const auto result = graph.run({frame_inputs(features[i])});
            require(result.published == std::vector<std::int64_t>{0} && same_bits(result.committed_payload, committed[i]),
                    "graph replay differs bitwise from enqueueV3");
        }
        report << "graph_replay\t" << features.size() << "_inputs\tbit_identical_to_enqueueV3\tPASS\n";
        graph.close();
    }
    require(logger.errors == 0, "TensorRT reported errors");
    } catch (const std::exception& error) {
        // Keep what was measured before the failure.
        report << "error\t" << error.what() << "\nresult\tFAIL\n";
        write_text_file(report_path, report.str());
        std::cout << report.str();
        throw;
    }
    report << "result\t" << (pass ? "PASS" : "FAIL") << '\n';
    write_text_file(report_path, report.str());
    std::cout << report.str();
    require(pass, "policy comparison failed");
}

void run_unregistered(const std::string& engine_path, Logger& logger) {
    const auto plan = read_text_file(engine_path);
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    require(runtime != nullptr, "createInferRuntime failed");
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(), plan.size()));
    // TensorRT 10.16 defers plugin materialization to context creation (SCAN_TRT §3).
    const bool deserialized = engine != nullptr;
    bool context = false;
    if (engine) {
        std::unique_ptr<IExecutionContext> c(engine->createExecutionContext());
        context = c != nullptr;
    }
    std::cout << "unregistered_deserialization=" << (deserialized ? "deferred" : "rejected") << "\nunregistered_context=" << (context ? "created" : "rejected")
              << '\n';
    require(!context, "an engine with unregistered plugins produced an execution context");
}

} // namespace

int main(int argc, char** argv) {
    Logger logger;
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "build" && (argc == 6 || (argc == 7 && std::string(argv[6]) == "--debug"))) build(argv[2], argv[3], argv[4], argv[5], argc == 7, logger);
        else if (mode == "run" && argc >= 7) {
            RunOptions options;
            for (int i = 7; i < argc; ++i) {
                const std::string flag = argv[i];
                if (flag == "--features" && i + 2 < argc && options.features.empty()) {
                    options.features = argv[++i];
                    options.features_sha = argv[++i];
                } else if (flag == "--dump" && i + 1 < argc && options.dump_dir.empty()) {
                    options.dump_dir = argv[++i];
                } else {
                    throw std::invalid_argument("run: unexpected argument " + flag);
                }
            }
            run(argv[2], argv[3], argv[4], argv[5], argv[6], options, logger);
        }
        else if (mode == "run-unregistered" && argc == 3) run_unregistered(argv[2], logger);
        else throw std::invalid_argument("usage: head_trt build BUNDLE SHA ENGINE REPORT [--debug] | run ENGINE SHA BUNDLE SHA REPORT "
                                         "[--features FILE SHA] [--dump DIR] | run-unregistered ENGINE");
        std::cout.flush();
        require(static_cast<bool>(std::cout), "cannot write standard output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "head_trt: " << error.what() << '\n';
        return 1;
    }
}

// F03 complete-model engine lifecycle (docs/BACKBONE_POLICY.md §7):
//   build ONNX ONNX_SHA256 BUNDLE BUNDLE_SHA256 ENGINE BUILD_REPORT
//   run   ENGINE ENGINE_SHA256 BUNDLE BUNDLE_SHA256 IMAGES IMAGES_SHA256 BACKBONE_REF BACKBONE_REF_SHA256 REPORT
//         [--labels L0,L1,...] [--inputs N] [--dump DIR]
// `build` parses the backbone ONNX into the head's network and builds the
// S04 head on its outputs (one strongly typed FP32 engine), then records the
// plan like head_trt build. `run` is a new process: plugin registration,
// deserialization, frame sessions of HEAD_TRT §5 with model_io_spec(), the
// head policy comparison of all six outputs for every input of IMAGES against
// head_reference_from_refs on the binary64 backbone reference (per input, per
// level p3/p4/p5: value, bound, sigma, float64 little-endian), determinism,
// graph replay, a failure sequence, checked close and a normal exit.
// `--inputs N` keeps the first N inputs (sanitizer runs); `--dump` writes each
// input's six committed outputs to DIR/input-<i>.f32.
#include "saccade/file_io.hpp"
#include "saccade/head_compare.hpp"
#include "saccade/head_reference.hpp"
#include "saccade/head_trt_network.hpp"
#include "saccade/model_trt_network.hpp"
#include "saccade/scan_trt_frame.hpp"
#include "saccade/scan_trt_network.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/sha256.hpp"

#include <NvInfer.h>

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

std::size_t count(const std::vector<std::int64_t>& shape) {
    std::size_t n = 1;
    for (const auto d : shape) n *= static_cast<std::size_t>(d);
    return n;
}

double seconds() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void build(const std::string& onnx_path, const std::string& onnx_sha, const std::string& bundle_path, const std::string& bundle_sha,
           const std::string& engine_path, const std::string& report_path, Logger& logger) {
    const auto onnx = checked_read(onnx_path, onnx_sha, "ONNX");
    const auto weights = decode_head_bundle(checked_read(bundle_path, bundle_sha, "bundle"));
    const auto t0 = seconds();
    const auto engine = build_model_engine(logger, bytes(onnx), weights);
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
        report << "onnx_sha256=" << onnx_sha << "\nbundle_sha256=" << bundle_sha << "\nengine_bytes=" << readback.size()
               << "\nengine_sha256=" << sha256_hex(readback) << "\nbuild_seconds=" << build_seconds << '\n' << engine.build_options
               << onnx_parser_identity() << scan_trt_plugin_identity() << describe_scan_engine(*deserialized);
        for (const auto& s : engine.statuses) report << "status_output=" << s << '\n';
        report << "layers_json=" << head_engine_layers(*deserialized) << '\n';
    }
    require(logger.errors == 0, "TensorRT reported errors");
    write_text_file(report_path, report.str());
    std::cout << "engine_sha256=" << sha256_hex(readback) << "\nengine_bytes=" << readback.size() << '\n';
}

// The backbone reference of one input as the head reference's inputs.
std::array<HeadTensorRef, 3> backbone_refs(const std::string& data, std::size_t input) {
    static_assert(std::endian::native == std::endian::little, "reference files are little-endian float64");
    std::size_t per_input = 0;
    for (int level = 0; level < 3; ++level) per_input += 3 * count(head_input_shape(level));
    require(data.size() % (per_input * sizeof(double)) == 0 && (input + 1) * per_input * sizeof(double) <= data.size(),
            "backbone reference: input " + std::to_string(input) + " missing");
    std::array<HeadTensorRef, 3> out;
    auto at = data.data() + input * per_input * sizeof(double);
    for (int level = 0; level < 3; ++level) {
        auto& t = out[static_cast<std::size_t>(level)];
        t.shape = head_input_shape(level);
        const auto n = count(t.shape);
        for (auto* v : {&t.value, &t.bound, &t.sigma}) {
            v->resize(n);
            std::memcpy(v->data(), at, n * sizeof(double));
            at += n * sizeof(double);
        }
    }
    return out;
}

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
    std::vector<std::string> labels;
    std::size_t inputs = 0; // 0: all
    std::string dump_dir;
};

void run(const std::vector<std::string>& a, const RunOptions& options, Logger& logger) {
    const auto& [engine_path, engine_sha, bundle_path, bundle_sha, images_path, images_sha, ref_path, ref_sha, report_path] =
        std::array<std::string, 9>{a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]};
    const auto plan = checked_read(engine_path, engine_sha, "engine");
    const auto weights = decode_head_bundle(checked_read(bundle_path, bundle_sha, "bundle"));
    const auto images = checked_read(images_path, images_sha, "images");
    const auto refs = checked_read(ref_path, ref_sha, "backbone reference");
    const auto frame_values = count(model_input_shape());
    require(!images.empty() && images.size() % (frame_values * sizeof(float)) == 0, "images: not a whole number of [1,3,640,640] frames");
    auto frames = images.size() / (frame_values * sizeof(float));
    require(options.labels.empty() || options.labels.size() == frames, "--labels must name every input");
    if (options.inputs) {
        require(options.inputs <= frames, "--inputs exceeds the inputs in IMAGES");
        frames = options.inputs;
    }
    const auto spec = model_io_spec();
    std::ostringstream report;
    report << "engine_sha256=" << engine_sha << "\nbundle_sha256=" << bundle_sha << "\nimages_file=" << images_path << "\nimages_sha256=" << images_sha
           << "\nbackbone_reference_sha256=" << ref_sha << "\ninputs_compared=" << frames << "\npolicy=t01-f03-fp32/1 (" << kHeadPolicyVersion
           << " comparator)\ncuda_module_loading=" << (std::getenv("CUDA_MODULE_LOADING") ? std::getenv("CUDA_MODULE_LOADING") : "default") << '\n'
           << onnx_parser_identity();
    std::vector<std::string> labels;
    std::vector<std::vector<float>> inputs;
    for (std::size_t i = 0; i < frames; ++i) {
        labels.push_back(options.labels.empty() ? "input/" + std::to_string(i) : options.labels[i]);
        const auto* p = reinterpret_cast<const float*>(images.data() + i * frame_values * sizeof(float));
        inputs.emplace_back(p, p + frame_values);
        report << "input\t" << i << '\t' << labels[i] << "\tsha256="
               << sha256_hex(std::string_view(images.data() + i * frame_values * sizeof(float), frame_values * sizeof(float))) << '\n';
    }
    bool pass = true;
    std::vector<std::vector<float>> committed;
    try {
        {
            ScanFrameConfig cfg;
            cfg.slots = 2;
            ScanFrameSession session(bytes(plan), cfg, logger, nullptr, &spec);
            report << session.identity();
            for (std::size_t i = 0; i < frames; ++i) {
                const auto& label = labels[i];
                const auto result = session.run({{inputs[i]}});
                require(result.published == std::vector<std::int64_t>{0} && result.committed_payload_frame == 0, label + ": frame not accepted");
                const auto t0 = seconds();
                const auto reference = head_reference_from_refs(weights, backbone_refs(refs, i));
                report << "reference_seconds\t" << label << '\t' << seconds() - t0 << '\n';
                const auto comparison = compare_head_outputs(split(session, result.committed_payload), reference);
                report << comparison.report(label);
                pass = pass && comparison.pass;
                committed.push_back(result.committed_payload);
                const auto again = session.run({{inputs[i]}});
                require(same_bits(again.committed_payload, result.committed_payload), label + ": rerun differs bitwise");
                report << "determinism\t" << label << "\trerun_bit_identical\tPASS\n";
                if (!options.dump_dir.empty()) {
                    const auto path = options.dump_dir + "/input-" + std::to_string(i) + ".f32";
                    std::string blob;
                    for (const auto& o : split(session, result.committed_payload)) blob.append(reinterpret_cast<const char*>(o.data()), o.size() * sizeof(float));
                    write_binary_file(path, bytes(blob));
                    report << "dump\t" << label << '\t' << path << "\tsha256=" << sha256_hex(blob) << '\n';
                }
                std::cout << "input " << label << (comparison.pass ? " PASS" : " FAIL") << std::endl;
            }
            // Failure sequence: a NaN in images fails the frame; the later valid frame is skipped.
            auto bad = inputs[0];
            bad[123457] = std::numeric_limits<float>::quiet_NaN();
            ScanFrameConfig fixed;
            fixed.slots = 1;
            fixed.fixed_submit = true;
            ScanFrameSession failing(bytes(plan), fixed, logger, nullptr, &spec);
            const auto r = failing.run({{inputs[0]}, {bad}, {inputs[0]}});
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
            for (std::size_t i = 0; i < frames; ++i) {
                const auto result = graph.run({{inputs[i]}});
                require(result.published == std::vector<std::int64_t>{0} && same_bits(result.committed_payload, committed[i]),
                        "graph replay differs bitwise from enqueueV3");
            }
            report << "graph_replay\t" << frames << "_inputs\tbit_identical_to_enqueueV3\tPASS\n";
            graph.close();
        }
        require(logger.errors == 0, "TensorRT reported errors");
    } catch (const std::exception& error) {
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

std::vector<std::string> split_list(const std::string& text) {
    std::vector<std::string> out;
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const auto end = std::min(text.find(',', begin), text.size());
        out.push_back(text.substr(begin, end - begin));
        begin = end + 1;
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    Logger logger;
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "build" && argc == 8) build(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], logger);
        else if (mode == "run" && argc >= 11) {
            RunOptions options;
            for (int i = 11; i < argc; ++i) {
                const std::string flag = argv[i];
                if (flag == "--labels" && i + 1 < argc && options.labels.empty()) options.labels = split_list(argv[++i]);
                else if (flag == "--inputs" && i + 1 < argc && options.inputs == 0) options.inputs = std::stoul(argv[++i]);
                else if (flag == "--dump" && i + 1 < argc && options.dump_dir.empty()) options.dump_dir = argv[++i];
                else throw std::invalid_argument("run: unexpected argument " + flag);
            }
            run(std::vector<std::string>(argv + 2, argv + 11), options, logger);
        } else {
            throw std::invalid_argument("usage: model_trt build ONNX SHA BUNDLE SHA ENGINE REPORT | run ENGINE SHA BUNDLE SHA IMAGES SHA "
                                        "BACKBONE_REF SHA REPORT [--labels L,...] [--inputs N] [--dump DIR]");
        }
        std::cout.flush();
        require(static_cast<bool>(std::cout), "cannot write standard output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_trt: " << error.what() << '\n';
        return 1;
    }
}

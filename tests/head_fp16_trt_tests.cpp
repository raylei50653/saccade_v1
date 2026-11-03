// FP16 build mode (docs/TRACKING.md §16): builder readback, FP32 I/O,
// accepted frames, rerun and graph replay bit identity, and a loose error
// bound against the binary64 reference that only catches wiring mistakes
// (a wrong weight or layer gives a relative L2 of order 1).
#include "saccade/head_reference.hpp"
#include "saccade/head_trt_network.hpp"
#include "saccade/scan_trt_frame.hpp"

#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using namespace saccade;
unsigned checks = 0;
void check(bool ok, const std::string& why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << message << '\n';
    }
};
bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
}

int main() {
    try {
        check(parse_model_precision("fp32") == ModelPrecision::Fp32, "parse fp32");
        check(parse_model_precision("fp16") == ModelPrecision::Fp16, "parse fp16");
        bool rejected = false;
        try { (void)parse_model_precision("bf16"); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "bf16 is not a model precision");

        check(cudaSetDevice(0) == cudaSuccess, "select device");
        Logger logger;
        const auto weights = synthetic_head_weights(3);
        HeadBuildOptions options;
        options.precision = ModelPrecision::Fp16;
        const auto engine = build_head_engine(logger, weights, options);
        for (const char* line : {"network=weakly_typed\n", "precision=fp16\n", "tf32=off\n", "fp16=on\n", "bf16=off\n", "int8=off\n",
                                 "optimization_level=3\n"})
            check(engine.build_options.find(line) != std::string::npos, std::string("build report lacks ") + line);

        const auto spec = head_io_spec();
        const auto features = head_test_features("asymmetric", 1);
        const auto reference = head_reference(weights, {features[0], features[1], features[2]});
        std::vector<float> first;
        for (bool graphs : {false, true}) {
            ScanFrameConfig config;
            config.slots = 1;
            config.graphs = graphs;
            // The session checks the engine's I/O names, shapes and FP32 types against the spec.
            ScanFrameSession session(engine.plan, config, logger, nullptr, &spec);
            const auto result = session.run({{{features[0], features[1], features[2]}}});
            check(result.published == std::vector<std::int64_t>{0}, "FP16 frame accepted (finite outputs, PASS statuses)");
            const auto again = session.run({{{features[0], features[1], features[2]}}});
            check(same_bits(again.committed_payload, result.committed_payload), "FP16 rerun bits");
            if (graphs) check(same_bits(result.committed_payload, first), "FP16 graph replay bits");
            else first = result.committed_payload;
            if (!graphs) {
                for (unsigned i = 0; i < 6; ++i) {
                    const auto& output = session.outputs().at(i);
                    const auto& want = reference.outputs[i].value;
                    check(output.count == want.size(), "output size");
                    double diff = 0, norm = 0;
                    for (std::size_t j = 0; j < want.size(); ++j) {
                        const double d = static_cast<double>(result.committed_payload[output.offset + j]) - want[j];
                        diff += d * d;
                        norm += want[j] * want[j];
                    }
                    const double relative_l2 = std::sqrt(diff / norm);
                    std::cout << kHeadOutputNames[i] << " relative_l2=" << relative_l2 << '\n';
                    check(relative_l2 <= 0.05, std::string(kHeadOutputNames[i]) + ": relative L2 above the wiring bound");
                }
            }
            session.close();
        }
        std::cout << "head_fp16_trt_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "head_fp16_trt_tests: " << error.what() << '\n';
        return 1;
    }
}

#include "saccade/head_compare.hpp"
#include "saccade/head_trt_network.hpp"
#include "saccade/scan_trt_frame.hpp"

#include <cuda_runtime_api.h>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using namespace saccade;
unsigned checks = 0;
void check(bool ok, const char* why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << message << '\n';
    }
};
}

int main() {
    try {
        check(cudaSetDevice(0) == cudaSuccess, "select device");
        Logger logger;
        constexpr auto profile = ModelProfile::Hr1280x768;
        const auto weights = synthetic_head_weights(3);
        HeadBuildOptions options;
        options.profile = profile;
        const auto engine = build_head_engine(logger, weights, options);
        const auto spec = head_io_spec(profile);
        check(spec.inputs[0].shape == std::vector<std::int64_t>({1, 256, 96, 160}), "HR P3 shape");
        check(spec.inputs[1].shape == std::vector<std::int64_t>({1, 512, 48, 80}), "HR P4 shape");
        check(spec.inputs[2].shape == std::vector<std::int64_t>({1, 512, 24, 40}), "HR P5 shape");
        const auto features = head_test_features("asymmetric", 1, profile);
        const auto reference = head_reference(weights, {features[0], features[1], features[2]}, {}, false, profile);
        std::vector<float> committed;
        for (bool graphs : {false, true}) {
            ScanFrameConfig config;
            config.slots = 1;
            config.graphs = graphs;
            ScanFrameSession session(engine.plan, config, logger, nullptr, &spec);
            const auto result = session.run({{{features[0], features[1], features[2]}}});
            check(result.published == std::vector<std::int64_t>{0}, "HR head frame accepted");
            std::array<std::span<const float>, 6> outputs;
            for (unsigned i = 0; i < 6; ++i) {
                const auto& output = session.outputs().at(i);
                outputs[i] = std::span<const float>(result.committed_payload).subspan(output.offset, output.count);
            }
            const auto comparison = compare_head_outputs(outputs, reference);
            std::cout << comparison.report("hr-full-shape");
            for (const auto& output : comparison.outputs) check(output.failures == 0 && output.nonfinite == 0, "HR HEAD_POLICY");
            if (graphs) {
                check(result.committed_payload.size() == committed.size() &&
                      std::memcmp(result.committed_payload.data(), committed.data(), committed.size() * sizeof(float)) == 0,
                      "HR graph output bits");
            } else committed = result.committed_payload;
            session.close();
        }
        bool rejected = false;
        try {
            const auto wrong = head_io_spec();
            ScanFrameSession session(engine.plan, {}, logger, nullptr, &wrong);
            session.close();
        } catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "HR engine rejects T01 I/O specification");
        std::cout << "head_profile_trt_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "head_profile_trt_tests: " << error.what() << '\n';
        return 1;
    }
}

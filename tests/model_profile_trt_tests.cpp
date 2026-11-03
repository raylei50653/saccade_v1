#include "saccade/detection_session.hpp"
#include "saccade/file_io.hpp"
#include "saccade/head_compare.hpp"
#include "saccade/model_trt_network.hpp"
#include "saccade/postprocess.hpp"

#include <cuda_runtime_api.h>
#include <cstring>
#include <filesystem>
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

// Minimal static ONNX: three stride-8/16/32 1x1 convolutions copy RGB channels.
std::string varint(std::uint64_t value) {
    std::string bytes;
    do {
        bytes += static_cast<char>((value & 127) | (value >= 128 ? 128 : 0));
        value >>= 7;
    } while (value);
    return bytes;
}
std::string integer(unsigned field, std::uint64_t value) { return varint(field << 3) + varint(value); }
std::string blob(unsigned field, const std::string& value) {
    return varint((field << 3) | 2) + varint(value.size()) + value;
}
std::string value_info(const std::string& name, const std::vector<std::int64_t>& shape) {
    std::string dimensions;
    for (auto size : shape) dimensions += blob(1, integer(1, static_cast<std::uint64_t>(size)));
    return blob(1, name) + blob(2, blob(1, integer(1, 1) + blob(2, dimensions)));
}
std::string synthetic_onnx(ModelProfile profile) {
    std::string graph = blob(2, "profile-test") + blob(11, value_info(kModelInputName, model_input_shape(profile)));
    for (int level = 0; level < 3; ++level) {
        const auto shape = head_input_shape(level, profile);
        const auto channels = shape[1];
        std::vector<float> weights(static_cast<std::size_t>(channels * 3));
        for (std::int64_t c = 0; c < channels; ++c) weights[static_cast<std::size_t>(c * 3 + c % 3)] = 1;
        const auto name = "w" + std::to_string(level);
        const auto tensor = integer(1, channels) + integer(1, 3) + integer(1, 1) + integer(1, 1) + integer(2, 1) + blob(8, name) +
            blob(9, std::string(reinterpret_cast<const char*>(weights.data()), weights.size() * sizeof(float)));
        const auto stride = ModelProfileSpec::strides[level];
        const auto strides = blob(1, "strides") + integer(8, stride) + integer(8, stride) + integer(20, 7);
        graph += blob(1, blob(1, kModelInputName) + blob(1, name) + blob(2, kBackboneOutputNames[level]) +
                         blob(3, "copy" + std::to_string(level)) + blob(4, "Conv") + blob(5, strides));
        graph += blob(5, tensor) + blob(12, value_info(kBackboneOutputNames[level], shape));
    }
    return integer(1, 8) + blob(2, "saccade-profile-test") + blob(7, graph) + blob(8, integer(2, 17));
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 3, "usage: model_profile_trt_tests WORK FIXTURES");
        check(cudaSetDevice(0) == cudaSuccess, "select device");
        constexpr auto profile = ModelProfile::Hr1280x768;
        Logger logger;
        const auto onnx = synthetic_onnx(profile);
        const std::span<const std::uint8_t> onnx_bytes{reinterpret_cast<const std::uint8_t*>(onnx.data()), onnx.size()};
        const auto weights = synthetic_head_weights(3);
        bool rejected = false;
        try { (void)build_model_engine(logger, onnx_bytes, weights); }
        catch (const std::runtime_error&) { rejected = true; }
        check(rejected, "HR ONNX rejected for default profile");
        const auto engine = build_model_engine(logger, onnx_bytes, weights, profile);
        const auto spec = model_io_spec(profile);
        check(spec.inputs[0].shape == std::vector<std::int64_t>({1, 3, 768, 1280}), "HR model input shape");
        std::vector<float> input(kHrModelProfile.input_elements());
        for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 127) / 128.0f;
        std::array<std::vector<float>, 3> features;
        for (unsigned level = 0; level < 3; ++level) {
            const auto h = kHrModelProfile.level_height(level), w = kHrModelProfile.level_width(level);
            const auto stride = ModelProfileSpec::strides[level];
            auto& feature = features[level];
            feature.resize(static_cast<std::size_t>(kHeadInChannels[level] * h * w));
            for (std::int64_t c = 0; c < kHeadInChannels[level]; ++c)
                for (std::int64_t y = 0; y < h; ++y)
                    for (std::int64_t x = 0; x < w; ++x)
                        feature[static_cast<std::size_t>((c * h + y) * w + x)] =
                            input[static_cast<std::size_t>(((c % 3) * 768 + y * stride) * 1280 + x * stride)];
        }
        const auto reference = head_reference(weights, {features[0], features[1], features[2]}, {}, false, profile);
        ScanFrameSession model(engine.plan, {}, logger, nullptr, &spec);
        const auto result = model.run({{{input}}});
        check(result.published == std::vector<std::int64_t>{0}, "HR F03 frame accepted");
        std::array<std::span<const float>, 6> outputs;
        for (unsigned i = 0; i < 6; ++i) {
            const auto& output = model.outputs().at(i);
            outputs[i] = std::span<const float>(result.committed_payload).subspan(output.offset, output.count);
        }
        const auto comparison = compare_head_outputs(outputs, reference);
        std::cout << comparison.report("hr-synthetic-model");
        for (const auto& output : comparison.outputs) check(output.failures == 0 && output.nonfinite == 0, "HR F03 HEAD_POLICY");
        model.close();

        const std::filesystem::path work = argv[1];
        std::filesystem::create_directories(work);
        const auto engine_path = work / "synthetic-hr.engine";
        write_binary_file(engine_path, engine.plan);
        const auto jpeg = read_text_file(std::filesystem::path(argv[2]) / "baseline.jpg");
        DetectionSessionConfig config;
        config.profile = profile;
        config.preprocess.max_width = 48;
        config.preprocess.max_height = 32;
        config.result_slots = 3;
        DetectionSession session(engine_path, config);
        check(session.identity().find("output_width=1280") != std::string::npos &&
              session.identity().find("output_height=768") != std::string::npos, "session selects HR F02 size");
        const std::span<const std::uint8_t> jpeg_bytes{reinterpret_cast<const std::uint8_t*>(jpeg.data()), jpeg.size()};
        for (unsigned frame = 0; frame < 3; ++frame) session.enqueue(jpeg_bytes, frame, 35, 27);
        session.finish();
        DetectionCandidates first{};
        for (unsigned frame = 0; frame < 3; ++frame) {
            const auto detection = session.receive();
            check(detection.status == DetectionSessionStatus::Ok && detection.frame == frame &&
                  detection.candidates.header.flags == 0 && detection.candidates.header.done == kPostprocessDone,
                  "HR F01-F04 session frame accepted");
            auto normalized = detection.candidates;
            normalized.header.frame = 0;
            if (frame == 0) first = normalized;
            else check(std::memcmp(&first, &normalized, sizeof(first)) == 0, "HR session repeated frame bits");
        }
        session.close();
        rejected = false;
        try { DetectionSession wrong(engine_path); wrong.close(); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "HR model rejected by default session profile");
        std::cout << "model_profile_trt_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_profile_trt_tests: " << error.what() << '\n';
        return 1;
    }
}

#pragma once

#include "saccade/head_trt_network.hpp"
#include "saccade/head_weights.hpp"
#include "saccade/scan_trt_frame.hpp"

#include <NvInfer.h>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace saccade {

// T01 backbone/neck and F03 complete model on TensorRT (docs/BACKBONE_POLICY.md):
// the backbone ONNX is imported with the pinned SDK's nvonnxparser into a
// strongly typed FP32 network with the head's builder settings (TF32 off,
// optimization level 0, kHeadBuilderWorkspace, no auxiliary streams, no timing
// cache) and the same tactic check (head_tactic_violations).
inline constexpr const char* kModelInputName = "images";
inline constexpr std::array<const char*, 3> kBackboneOutputNames{"p3_feat", "p4_feat", "p5_feat"};
std::vector<std::int64_t> model_input_shape(); // {1, 3, 640, 640}

// The parser's identity: library version, and the header's API version.
std::string onnx_parser_identity();

// Backbone engine: the ONNX's own I/O, which must be exactly the input
// `images` and the outputs p3_feat, p4_feat, p5_feat in order, FP32 with the
// head's input shapes. Throws std::runtime_error with the parser's errors,
// on any other I/O, or when the built engine runs a tactic outside the policy.
HeadEngine build_backbone_engine(nvinfer1::ILogger& logger, std::span<const std::uint8_t> onnx);

// F03 model engine: the backbone parsed into the head's network
// (HeadBuildOptions::features), its three outputs unmarked and read by the S04
// head layers; I/O is model_io_spec().
HeadEngine build_model_engine(nvinfer1::ILogger& logger, std::span<const std::uint8_t> onnx, const HeadWeights& weights);

// head_io_spec() with the single input `images` instead of p3/p4/p5.
FrameIoSpec model_io_spec();

} // namespace saccade

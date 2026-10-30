#pragma once

#include "saccade/head_weights.hpp"
#include "saccade/scan_trt_frame.hpp"

#include <NvInfer.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace saccade {

// S04 complete T=1 head as a native TensorRT network built from the weight
// bundle (docs/HEAD_TRT.md §3): no ONNX import, every layer added through the
// TensorRT network API, Cross-Scan and SelectiveScan as the S03 plugins.
// Strongly typed FP32 network, TF32 off, no auxiliary streams, no timing cache.
inline constexpr std::size_t kHeadBuilderWorkspace = std::size_t{256} << 20;
inline constexpr int kHeadBuilderOptimizationLevel = 0;

// The head's engine I/O: inputs p3, p4, p5 (FP32 NCHW), the six raw outputs in
// kHeadOutputNames order, the nine plugin statuses in creation order (per level:
// expand, scan, merge) with their producers, and the harness's finite checks of
// every input and output element.
FrameIoSpec head_io_spec();

struct HeadBuildOptions {
    // Also mark the reference's intermediate tensors ("p3.tokens", ...; see
    // head_reference) as outputs, to locate a first divergence. Diagnostic only:
    // such an engine is never the accepted one.
    bool debug_intermediates = false;
    // F03 (docs/BACKBONE_POLICY.md §7): when set, called once with the head's
    // network before any head layer is added; the head then reads P3/P4/P5
    // from the returned FP32 tensors (head_input_shape) instead of creating
    // network inputs. The caller keeps whatever owns those tensors' weights
    // (the ONNX parser) alive until build_head_engine returns.
    std::function<std::array<nvinfer1::ITensor*, 3>(nvinfer1::INetworkDefinition&)> features;
};

struct HeadEngine {
    std::vector<std::uint8_t> plan;
    std::vector<std::string> statuses;      // status outputs in creation order
    std::vector<std::string> debug_outputs; // marked intermediates, forward order
    std::string build_options;              // key=value lines of the builder configuration
};

// Throws std::runtime_error when TensorRT rejects a layer or the build, when
// the statuses created differ from head_io_spec(), or when the built engine
// runs a tactic outside the comparison policy's operation model
// (head_tactic_violations).
HeadEngine build_head_engine(nvinfer1::ILogger& logger, const HeadWeights& weights, const HeadBuildOptions& options = {});

// Per-layer engine information (IEngineInspector, JSON, detailed verbosity):
// layer types, fused layers and the tactic/kernel each one runs.
std::string head_engine_layers(const nvinfer1::ICudaEngine& engine);

// Tactic names in the layer information that leave docs/HEAD_POLICY.md §2:
// transform-based convolution (Winograd "wngd"/"winograd", "fft") or reduced
// precision ("f16", "bf16", "tf32", "hmma", "i8", "int8"). Empty when the
// engine runs direct-summation FP32 kernels only.
std::vector<std::string> head_tactic_violations(const std::string& layers_json);

} // namespace saccade

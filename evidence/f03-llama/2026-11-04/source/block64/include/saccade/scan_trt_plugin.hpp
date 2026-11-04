#pragma once

#include <NvInfer.h>

#include <cstdint>
#include <memory>
#include <string>

namespace saccade {

// S03 TensorRT plugins around the S02 CUDA FP32 scan (docs/SCAN_TRT.md). Each
// plugin is an IPluginV3 (V3One core/build/runtime capabilities) that calls the
// S02 forward API on TensorRT's stream with TensorRT's buffers and returns two
// outputs: the FP32 result and a fixed INT32[2] status execution tensor holding
// the S02 ScanStatus words {flags, done}. A result is valid only when its status
// reads flags == 0 and done == kScanDone; NaN in the payload is a failure marker,
// never the validity test.
//
//   SelectiveScan    v2  inputs u, delta, A, B, C[, D]  -> y (B,L,Dc), status
//   CrossScanExpand  v1  input  x (B,C,H,W)             -> tokens (4B,H*W,C), status
//   CrossScanMerge   v1  input  y (4B,H*W,C)            -> out (B,C,H,W), status
//
// Serialized fields are the semantic attributes only: SelectiveScan `has_d`,
// CrossScanMerge `height` and `width` (INT32, one value each). Device pointers,
// workspace and the loaded-kernel binding are never serialized.
inline constexpr const char* kScanTrtNamespace = "saccade";
inline constexpr const char* kSelectiveScanPluginName = "SelectiveScan";
inline constexpr const char* kSelectiveScanPluginVersion = "2";
inline constexpr const char* kCrossScanExpandPluginName = "CrossScanExpand";
inline constexpr const char* kCrossScanExpandPluginVersion = "1";
inline constexpr const char* kCrossScanMergePluginName = "CrossScanMerge";
inline constexpr const char* kCrossScanMergePluginVersion = "1";
inline constexpr std::int32_t kScanStatusWords = 2;

// Registers the three creators in `registry`; a creator already registered
// there is left as is. Throws std::runtime_error when TensorRT refuses one.
// Call during initialization, before building or deserializing an engine.
void register_scan_trt_plugins(nvinfer1::IPluginRegistry& registry);

// Build-phase plugins created through the registered creators, i.e. with the
// same field parsing that deserialization uses. Throw std::runtime_error when
// the creator is missing or rejects the fields.
std::unique_ptr<nvinfer1::IPluginV3> create_selective_scan_plugin(bool has_d);
std::unique_ptr<nvinfer1::IPluginV3> create_cross_scan_expand_plugin();
std::unique_ptr<nvinfer1::IPluginV3> create_cross_scan_merge_plugin(std::int32_t height, std::int32_t width);

// Plugin build identity: TensorRT headers and library, plugin API, and the
// S02 kernel identity (commit, dirty flag, nvcc, CUDA flags, architectures).
std::string scan_trt_plugin_identity();

} // namespace saccade

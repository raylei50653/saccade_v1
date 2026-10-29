#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace saccade {

// Scalar reference for the original YOLO26m Mamba-FPN head's scan operator,
// ONNX `saccade::SelectiveScan` (custom opset `saccade` 1). Contract, sources and
// tolerances: docs/SCAN_CONTRACT.md. All tensors are dense, row-major FP32:
//   u, delta: (batch, length, channels)   delta is raw, before softplus
//   a:        (a_rows, state)             a_rows = 1 (shared) or channels
//   b:        (batch, length, state)
//   c:        (batch, length, c_width)    c_width = 1 (broadcast over state) or state
//   d:        (channels), only when has_d
// Every (batch, channel) lane starts from a zero state for each call; there is
// no initial-state input, no final-state output and no state between calls.
struct SelectiveScanShape {
    std::size_t batch = 0;
    std::size_t length = 0;    // 0 is valid and yields an empty output
    std::size_t channels = 0;
    std::size_t state = 0;     // power of two in [1, 32]
    std::size_t a_rows = 1;
    std::size_t c_width = 1;
    bool has_d = true;
};

struct SelectiveScanInputs {
    std::span<const float> u, delta, a, b, c, d;
};

// Computes in binary64 and returns y as (batch, length, channels). Throws
// std::invalid_argument for shape, capacity or non-finite input violations and
// std::range_error when a decay, drive, state, term or output magnitude leaves
// the FP32 finite range of the operator's output ABI. Nothing is returned on
// failure.
std::vector<double> selective_scan_reference(const SelectiveScanShape& shape,
                                             const SelectiveScanInputs& inputs);

// softplus with the source threshold: x for x > 20, otherwise log1p(exp(x)).
double selective_scan_softplus(double x);

// Four-direction Cross-Scan around each level's Mamba block. Direction k reads
// token t = row-major position of the k-flipped map: 0 identity, 1 flip H and W,
// 2 flip W, 3 flip H. Returns the source (h, w) as h * width + w.
std::size_t cross_scan_source(std::size_t direction, std::size_t token,
                              std::size_t height, std::size_t width);

// x: (batch, channels, height, width) -> tokens (4 * batch, height * width,
// channels); sequence row = direction * batch + image, as the head stacks
// directions before folding them into the batch.
std::vector<float> cross_scan_expand(std::span<const float> x, std::size_t batch,
                                     std::size_t channels, std::size_t height,
                                     std::size_t width);

// y: (4 * batch, height * width, channels), each row in its own direction's
// token order -> (batch, channels, height, width): the four directions are
// mapped back to (h, w) and averaged as ((y0 + y1) + y2) + y3) / 4.
std::vector<double> cross_scan_merge(std::span<const double> y, std::size_t batch,
                                     std::size_t channels, std::size_t height,
                                     std::size_t width);

} // namespace saccade

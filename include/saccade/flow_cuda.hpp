#pragma once

#include "saccade/flow_records.hpp"
#include <cuda_runtime_api.h>
#include <memory>
#include <span>
#include <string>

namespace saccade::flow {
struct Timing { float grayscale_ms = 0, execute_ms = 0, cell_ms = 0; };
// API 2.0 only; primary context, one owned non-default stream, grid 4,
// grayscale8, no external/temporal hints, output cost or ROI.
class Engine {
public:
    Engine(int width, int height, Perf perf);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    cudaStream_t stream() const;
    std::string library_path() const;
    // The borrowed RGB input stays live until push_rgb returns. Frame zero
    // initializes the previous image; subsequent frames execute previous->current.
    Frame push_rgb(const std::uint8_t* planar_rgb, std::int64_t frame, Timing& timing);
    // Synthetic test input follows the identical Execute and cell kernels.
    Frame push_gray(std::span<const std::uint8_t> gray, std::int64_t frame, Timing& timing);
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace saccade::flow

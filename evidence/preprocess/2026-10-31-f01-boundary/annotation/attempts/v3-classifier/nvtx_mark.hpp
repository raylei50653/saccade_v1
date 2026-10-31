// Diagnostic annotation only. Does not initialize bytes or validate ordering.
// Caller must identify and bound the LIVE SDK allocation and supply its stream.
#pragma once

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExtMemCudaRt.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace f01_boundary {

inline void mark_ranges(nvtxDomainHandle_t domain, cudaStream_t stream,
                        const std::vector<nvtxMemVirtualRangeDesc_t>& ranges) {
    nvtxMemMarkInitializedBatch_t batch{};
    batch.extCompatID = NVTX_EXT_COMPATID_MEM;
    batch.structSize = sizeof(batch);
    batch.regionType = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
    batch.regionDescCount = ranges.size();
    batch.regionDescElementSize = sizeof(nvtxMemVirtualRangeDesc_t);
    batch.regionDescElements = ranges.data();
    // false: stream is explicit, not a per-thread default-stream handle.
    // The NVTX API returns void; invocation is not an acknowledgement.
    nvtxMemCudaMarkInitialized(domain, stream, false, &batch);
}

inline void mark_valid_rgb(nvtxDomainHandle_t domain, cudaStream_t stream,
                           const void* surface, std::size_t width,
                           std::size_t height, std::size_t pitch) {
    // SDK-private layout is a measured fact for this binary, not an API ABI.
    if (!surface || !width || !height || width > SIZE_MAX / 4 ||
        pitch < 4 * width || height > SIZE_MAX / pitch ||
        width > SIZE_MAX / height)
        throw std::invalid_argument("invalid bounded RGBA surface");
    std::vector<nvtxMemVirtualRangeDesc_t> ranges;
    ranges.reserve(width * height);
    const auto* bytes = static_cast<const unsigned char*>(surface);
    for (std::size_t y = 0; y < height; ++y)
        for (std::size_t x = 0; x < width; ++x)
            ranges.push_back({3, bytes + y * pitch + 4 * x});
    mark_ranges(domain, stream, ranges);
}

} // namespace f01_boundary

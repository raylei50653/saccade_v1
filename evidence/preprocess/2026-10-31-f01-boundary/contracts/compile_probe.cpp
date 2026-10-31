#include "nvtx_mark.hpp"

void compile_only(nvtxDomainHandle_t domain, cudaStream_t stream,
                  const void* surface) {
    f01_boundary::mark_valid_rgb(domain, stream, surface, 35, 27, 512);
}

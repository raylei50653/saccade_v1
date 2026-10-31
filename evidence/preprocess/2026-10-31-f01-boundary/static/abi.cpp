// Host-only ABI observation. Does not initialize CUDA or execute GPU work.
#include <nvjpeg.h>
#include <nppdefs.h>
#include <cstddef>
#include <iostream>

int main() {
    std::cout << "{\"pointer_bytes\":" << sizeof(void*)
              << ",\"size_t_bytes\":" << sizeof(std::size_t)
              << ",\"nvjpegImage_t\":{\"size\":" << sizeof(nvjpegImage_t)
              << ",\"align\":" << alignof(nvjpegImage_t)
              << ",\"channel\":" << offsetof(nvjpegImage_t, channel)
              << ",\"pitch\":" << offsetof(nvjpegImage_t, pitch)
              << ",\"components\":" << NVJPEG_MAX_COMPONENT
              << "},\"NppiSize\":{\"size\":" << sizeof(NppiSize)
              << ",\"align\":" << alignof(NppiSize)
              << ",\"width\":" << offsetof(NppiSize, width)
              << ",\"height\":" << offsetof(NppiSize, height)
              << "},\"NVJPEG_OUTPUT_RGB\":" << NVJPEG_OUTPUT_RGB << "}\n";
}

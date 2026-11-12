#pragma once
#include "saccade/model_producer_cuda.hpp"
namespace saccade {
inline constexpr const char* kYoloProducerPolicy = "research-yolo-finite-producer/1";
std::size_t yolo_producer_workspace_bytes();
void yolo_producer_cuda(const float* images, std::size_t image_elements,
                       const PostprocessRawView& raw, PostprocessProducerStatus* producer,
                       void* workspace, std::size_t workspace_bytes, cudaStream_t stream,
                       ModelProfile profile = kDefaultModelProfile);
std::string yolo_producer_cuda_load_kernels();
std::string yolo_producer_cuda_identity();
}

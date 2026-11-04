#pragma once
#include "saccade/flow_config.hpp"
#include "saccade/flow_records.hpp"
#include "saccade/postprocess.hpp"
#include "saccade/protocol.hpp"
#include <cuda_runtime_api.h>
#include <memory>
#include <string>

namespace saccade {
void flow_gray_cuda(const std::uint8_t* rgb,std::uint8_t* gray,std::size_t pitch,const MotionImageKey&,cudaStream_t);
void flow_mask_cuda(const DetectionCandidates*,std::uint8_t* masks,const MotionImageKey&,const Control*,cudaStream_t);
void flow_input_cuda(const std::uint8_t* gray,std::uint8_t* input,std::size_t pitch,const MotionImageKey&,const Control*,cudaStream_t);
void flow_output_cuda(const std::uint8_t* image,std::size_t pitch,std::uint8_t* gray,const MotionImageKey&,const Control*,cudaStream_t);
void flow_cells_cuda(const std::uint8_t* raw,std::size_t pitch,flow::Vector*,int cols,int rows,const Control*,cudaStream_t);
std::size_t flow_median_workspace_bytes(int cells);
void flow_median_cuda(const flow::Vector*,const std::uint8_t* masks,int cells,int scale,const Control*,MotionContext*,
                      void* scratch,std::size_t scratch_bytes,cudaStream_t);

// D owns MI production, M owns OFA, scratch and the two committed gray/mask
// slots. CTL publishes motion and tracker state with the same committed_slot.
class FlowMotionCuda {
public:
    FlowMotionCuda(const MotionImageKey&,Epoch,unsigned image_slots,cudaStream_t main);
    ~FlowMotionCuda();
    FlowMotionCuda(const FlowMotionCuda&)=delete;
    FlowMotionCuda& operator=(const FlowMotionCuda&)=delete;
    void enqueue_image(cudaStream_t detect,const std::uint8_t* rgb,FrameId);
    void enqueue_estimate(cudaStream_t main,FrameId,MotionContext*,const Control*);
    void enqueue_mask(cudaStream_t main,const DetectionCandidates*,const Control*);
    std::string library_path() const;
    void close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

#include "saccade/flow_motion_cuda.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/detail/flow_gray.cuh"
#include "saccade/detail/flow_cells.cuh"
#include "nvOpticalFlowCuda.h"
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <climits>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace saccade {
namespace {
void ck(cudaError_t s,const char* op) {
    if(s!=cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(s));
}
__global__ void flow_gray(const std::uint8_t* rgb,std::uint8_t* dst,std::size_t pitch,MotionImageKey key) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<key.width*key.height) dst[(i/key.width)*pitch+i%key.width]=
        flow::detail::gray_pixel(rgb,key.original_width,key.original_height,key.downscale,i%key.width,i/key.width);
}
__global__ void flow_mask(const DetectionCandidates* source,std::uint8_t* masks,MotionImageKey key,const Control* ctl) {
    if(!ctl->run_stage) return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x, cols=key.width/8, n=cols*(key.height/8);
    if(i>=n) return;
    const float x=float(key.downscale*(8*(i%cols)+4)),y=float(key.downscale*(8*(i/cols)+4));
    std::uint8_t masked=0;
    for(unsigned j=0;j<source->header.count;++j) {
        const auto& d=reinterpret_cast<const DetectionCandidate*>(&source->rows)[j];
        if((d.flags&kDetectionKept) && d.class_id==0 && d.score>=FlowConfig::mask_score &&
           x>=d.x1 && x<d.x2 && y>=d.y1 && y<d.y2) { masked=1; break; }
    }
    masks[ctl->working_slot*n+i]=masked;
}
__global__ void flow_input(const std::uint8_t* gray,std::uint8_t* dst,std::size_t pitch,MotionImageKey key,const Control* ctl) {
    if(!ctl->run_stage || ctl->committed_frame==no_frame) return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x,n=key.width*key.height;
    if(i<n) dst[(i/key.width)*pitch+i%key.width]=gray[ctl->committed_slot*n+i];
}
__global__ void flow_output(const std::uint8_t* src,std::size_t pitch,std::uint8_t* gray,MotionImageKey key,const Control* ctl) {
    if(!ctl->run_stage) return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x,n=key.width*key.height;
    if(i<n) gray[ctl->working_slot*n+i]=src[(i/key.width)*pitch+i%key.width];
}
__global__ void flow_cells(const std::uint8_t* raw,std::size_t pitch,flow::Vector* cells,int cols,int rows,const Control* ctl) {
    if(!ctl->run_stage || ctl->committed_frame==no_frame) return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<cols*rows) cells[i]=flow::detail::reduce_cell(raw,pitch,i%cols,i/cols);
}
__global__ void flow_count(int* count,const Control* ctl) { if(ctl->run_stage) *count=0; }
__global__ void flow_keys(const flow::Vector* cells,const std::uint8_t* masks,int n,const Control* ctl,int* keys,int* count) {
    if(!ctl->run_stage) return;
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    const bool keep=!masks[ctl->committed_slot*n+i];
    keys[i]=keep?int(cells[i].dx*64.f):INT_MAX;
    keys[n+i]=keep?int(cells[i].dy*64.f):INT_MAX;
    if(keep) atomicAdd(count,1);
}
__global__ void flow_context(const int* sorted,const int* count,int n,int scale,const Control* ctl,MotionContext* out) {
    if(!ctl->run_stage) return;
    *out={}; out->epoch=ctl->epoch; out->frame=ctl->current_frame; out->reference_frame=ctl->committed_frame;
    if(ctl->committed_frame==no_frame) return;
    if(*count<FlowConfig::min_cells) { out->status=MotionStatus::rejected; return; }
    const int lo=(*count-1)/2,hi=*count/2;
    out->status=MotionStatus::valid; out->confidence=1;
    out->tx=float(scale)*(float(sorted[lo]+sorted[hi])/128.f);
    out->ty=float(scale)*(float(sorted[n+lo]+sorted[n+hi])/128.f);
}
std::size_t prefix_bytes(int n) { return (std::size_t(4*n+1)*sizeof(int)+255)&~std::size_t(255); }
}
void flow_gray_cuda(const std::uint8_t* rgb,std::uint8_t* gray,std::size_t pitch,const MotionImageKey& key,cudaStream_t s) {
    flow_gray<<<(key.width*key.height+255)/256,256,0,s>>>(rgb,gray,pitch,key); ck(cudaGetLastError(),"F05 gray");
}
void flow_mask_cuda(const DetectionCandidates* d,std::uint8_t* masks,const MotionImageKey& key,const Control* ctl,cudaStream_t s) {
    flow_mask<<<((key.width/8)*(key.height/8)+255)/256,256,0,s>>>(d,masks,key,ctl); ck(cudaGetLastError(),"flow mask");
}
void flow_input_cuda(const std::uint8_t* gray,std::uint8_t* dst,std::size_t pitch,const MotionImageKey& key,const Control* ctl,cudaStream_t s) {
    flow_input<<<(key.width*key.height+255)/256,256,0,s>>>(gray,dst,pitch,key,ctl); ck(cudaGetLastError(),"flow input copy");
}
void flow_output_cuda(const std::uint8_t* src,std::size_t pitch,std::uint8_t* gray,const MotionImageKey& key,const Control* ctl,cudaStream_t s) {
    flow_output<<<(key.width*key.height+255)/256,256,0,s>>>(src,pitch,gray,key,ctl); ck(cudaGetLastError(),"flow output copy");
}
void flow_cells_cuda(const std::uint8_t* raw,std::size_t pitch,flow::Vector* cells,int cols,int rows,const Control* ctl,cudaStream_t s) {
    flow_cells<<<(cols*rows+255)/256,256,0,s>>>(raw,pitch,cells,cols,rows,ctl); ck(cudaGetLastError(),"flow cells");
}
std::size_t flow_median_workspace_bytes(int n) {
    std::size_t bytes=0;
    ck(cub::DeviceRadixSort::SortKeys(nullptr,bytes,static_cast<int*>(nullptr),static_cast<int*>(nullptr),n),"size flow median sort");
    return prefix_bytes(n)+bytes;
}
void flow_median_cuda(const flow::Vector* cells,const std::uint8_t* masks,int n,int scale,const Control* ctl,MotionContext* out,
                      void* scratch,std::size_t bytes,cudaStream_t s) {
    auto* keys=static_cast<int*>(scratch); auto* sorted=keys+2*n; auto* count=keys+4*n;
    auto* temp=static_cast<std::uint8_t*>(scratch)+prefix_bytes(n); bytes-=prefix_bytes(n);
    flow_count<<<1,1,0,s>>>(count,ctl);
    flow_keys<<<(n+255)/256,256,0,s>>>(cells,masks,n,ctl,keys,count);
    ck(cudaGetLastError(),"flow median keys");
    // CUB touches only initialized scratch; state reads and publication are guarded.
    ck(cub::DeviceRadixSort::SortKeys(temp,bytes,keys,sorted,n,0,32,s),"sort flow x");
    ck(cub::DeviceRadixSort::SortKeys(temp,bytes,keys+n,sorted+n,n,0,32,s),"sort flow y");
    flow_context<<<1,1,0,s>>>(sorted,count,n,scale,ctl,out); ck(cudaGetLastError(),"flow median context");
}
struct FlowMotionCuda::Impl {
    struct Buffer { NvOFGPUBufferHandle handle=nullptr; std::uint8_t* ptr=nullptr; std::size_t pitch=0; };
    struct Image { Buffer buffer; cudaEvent_t ready=nullptr,consumed=nullptr; bool recorded=false; };
    MotionImageKey key;
    cudaStream_t main;
    void* library=nullptr;
    std::string path;
    NV_OF_CUDA_API_FUNCTION_LIST api{};
    NvOFHandle handle=nullptr;
    Buffer input,output;
    std::vector<Image> images;
    std::uint8_t *gray=nullptr,*masks=nullptr;
    flow::Vector* cells=nullptr;
    void* scratch=nullptr;
    std::size_t scratch_bytes=0;
    void of(NV_OF_STATUS status,const char* op) {
        if(status==NV_OF_SUCCESS) return;
        char error[1024]{}; std::uint32_t n=sizeof(error);
        if(handle && api.nvOFGetLastError) api.nvOFGetLastError(handle,error,&n);
        throw std::runtime_error(std::string(op)+": NV_OF_STATUS="+std::to_string(status)+" "+error);
    }
    void allocate(Buffer& b,int w,int h,NV_OF_BUFFER_USAGE usage,NV_OF_BUFFER_FORMAT format) {
        const NV_OF_BUFFER_DESCRIPTOR desc{std::uint32_t(w),std::uint32_t(h),usage,format};
        of(api.nvOFCreateGPUBufferCuda(handle,&desc,NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR,&b.handle),"create OFA buffer");
        b.ptr=reinterpret_cast<std::uint8_t*>(api.nvOFGPUBufferGetCUdeviceptr(b.handle));
        NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
        of(api.nvOFGPUBufferGetStrideInfo(b.handle,&stride),"get OFA pitch"); b.pitch=stride.strideInfo[0].strideXInBytes;
    }
    Impl(const MotionImageKey& k,Epoch,unsigned slots,cudaStream_t stream):key(k),main(stream),images(slots) {
        try {
            CUcontext ctx=nullptr;
            if(cuCtxGetCurrent(&ctx)!=CUDA_SUCCESS || !ctx) throw std::runtime_error("get session CUDA context");
            library=dlopen("libnvidia-opticalflow.so.1",RTLD_NOW|RTLD_LOCAL);
            if(!library) throw std::runtime_error(dlerror());
            auto create=reinterpret_cast<decltype(&NvOFAPICreateInstanceCuda)>(dlsym(library,"NvOFAPICreateInstanceCuda"));
            if(!create) throw std::runtime_error("missing NvOFAPICreateInstanceCuda");
            Dl_info info{};
            if(!dladdr(reinterpret_cast<void*>(create),&info)) throw std::runtime_error("dladdr OFA library");
            path=std::filesystem::canonical(info.dli_fname).string();
            of(create(NV_OF_API_VERSION,&api),"create OFA API 2.0");
            of(api.nvCreateOpticalFlowCuda(ctx,&handle),"create session OFA");
            of(api.nvOFSetIOCudaStreams(handle,main,main),"set OFA M IO streams");
            NV_OF_INIT_PARAMS init{};
            init.width=k.width; init.height=k.height; init.outGridSize=NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
            init.mode=NV_OF_MODE_OPTICALFLOW; init.perfLevel=NV_OF_PERF_LEVEL_SLOW;
            of(api.nvOFInit(handle,&init),"initialize session OFA");
            allocate(input,k.width,k.height,NV_OF_BUFFER_USAGE_INPUT,NV_OF_BUFFER_FORMAT_GRAYSCALE8);
            allocate(output,(k.width+3)/4,(k.height+3)/4,NV_OF_BUFFER_USAGE_OUTPUT,NV_OF_BUFFER_FORMAT_SHORT2);
            ck(cudaMemset2DAsync(input.ptr,input.pitch,0,k.width,k.height,main),"initialize fixed OFA input");
            ck(cudaMemset2DAsync(output.ptr,output.pitch,0,((k.width+3)/4)*sizeof(short2),(k.height+3)/4,main),"initialize OFA output");
            for(auto& image:images) {
                allocate(image.buffer,k.width,k.height,NV_OF_BUFFER_USAGE_INPUT,NV_OF_BUFFER_FORMAT_GRAYSCALE8);
                ck(cudaEventCreateWithFlags(&image.ready,cudaEventDisableTiming),"create MI ready");
                ck(cudaEventCreateWithFlags(&image.consumed,cudaEventDisableTiming),"create MI consumed");
            }
            const int n=(k.width/8)*(k.height/8);
            scratch_bytes=flow_median_workspace_bytes(n);
            ck(cudaMalloc(&gray,2*std::size_t(k.width)*k.height),"allocate committed gray");
            ck(cudaMalloc(&masks,2*n),"allocate committed masks");
            ck(cudaMalloc(&cells,n*sizeof(*cells)),"allocate cells");
            ck(cudaMalloc(&scratch,scratch_bytes),"allocate median scratch");
            ck(cudaMemsetAsync(gray,0,2*std::size_t(k.width)*k.height,main),"initialize gray");
            ck(cudaMemsetAsync(masks,0,2*n,main),"initialize masks");
            ck(cudaMemsetAsync(cells,0,n*sizeof(*cells),main),"initialize cells");
            ck(cudaMemsetAsync(scratch,0,scratch_bytes,main),"initialize median scratch");
            cudaFuncAttributes a{};
            ck(cudaFuncGetAttributes(&a,flow_gray),"load gray"); ck(cudaFuncGetAttributes(&a,flow_mask),"load mask");
            ck(cudaFuncGetAttributes(&a,flow_input),"load input copy"); ck(cudaFuncGetAttributes(&a,flow_output),"load output copy");
            ck(cudaFuncGetAttributes(&a,flow_cells),"load cells"); ck(cudaFuncGetAttributes(&a,flow_count),"load count");
            ck(cudaFuncGetAttributes(&a,flow_keys),"load keys"); ck(cudaFuncGetAttributes(&a,flow_context),"load context");
        } catch(...) { release(); throw; }
    }
    CleanupReport release() {
        CleanupReport r;
        if(main) { int token=1; r.release(token,[&](int){ck(cudaStreamSynchronize(main),"drain flow M");}); }
        for(auto& image:images) {
            r.release(image.ready,[](auto e){ck(cudaEventDestroy(e),"destroy MI ready");});
            r.release(image.consumed,[](auto e){ck(cudaEventDestroy(e),"destroy MI consumed");});
            r.release(image.buffer.handle,[&](auto h){of(api.nvOFDestroyGPUBufferCuda(h),"destroy MI buffer");});
        }
        for(auto* b:{&input,&output}) r.release(b->handle,[&](auto h){of(api.nvOFDestroyGPUBufferCuda(h),"destroy OFA buffer");});
        r.release(gray,[](auto p){ck(cudaFree(p),"free gray");}); r.release(masks,[](auto p){ck(cudaFree(p),"free masks");});
        r.release(cells,[](auto p){ck(cudaFree(p),"free cells");}); r.release(scratch,[](auto p){ck(cudaFree(p),"free median");});
        r.release(handle,[&](auto h){of(api.nvOFDestroy(h),"destroy OFA");});
        r.release(library,[](auto p){if(dlclose(p)) throw std::runtime_error("close OFA library");});
        return r;
    }
    ~Impl() { release(); }
};
FlowMotionCuda::FlowMotionCuda(const MotionImageKey& k,Epoch e,unsigned slots,cudaStream_t m):impl_(std::make_unique<Impl>(k,e,slots,m)) {}
FlowMotionCuda::~FlowMotionCuda()=default;
void FlowMotionCuda::enqueue_image(cudaStream_t d,const std::uint8_t* rgb,FrameId frame) {
    auto& s=*impl_; auto& image=s.images[frame%s.images.size()];
    if(image.recorded) ck(cudaStreamWaitEvent(d,image.consumed,0),"D wait MI consumed");
    flow_gray_cuda(rgb,image.buffer.ptr,image.buffer.pitch,s.key,d);
    ck(cudaEventRecord(image.ready,d),"publish MI ready");
}
void FlowMotionCuda::enqueue_estimate(cudaStream_t m,FrameId frame,MotionContext* out,const Control* ctl) {
    auto& s=*impl_; auto& image=s.images[frame%s.images.size()];
    ck(cudaStreamWaitEvent(m,image.ready,0),"M wait MI ready");
    flow_input_cuda(s.gray,s.input.ptr,s.input.pitch,s.key,ctl,m);
    if(frame!=0) {
        NV_OF_EXECUTE_INPUT_PARAMS in{};
        in.inputFrame=s.input.handle; in.referenceFrame=image.buffer.handle; in.disableTemporalHints=NV_OF_TRUE;
        NV_OF_EXECUTE_OUTPUT_PARAMS output{}; output.outputBuffer=s.output.handle;
        s.of(s.api.nvOFExecute(s.handle,&in,&output),"session NvOFExecute");
    }
    const int cols=s.key.width/8,rows=s.key.height/8;
    flow_cells_cuda(s.output.ptr,s.output.pitch,s.cells,cols,rows,ctl,m);
    flow_median_cuda(s.cells,s.masks,cols*rows,s.key.downscale,ctl,out,s.scratch,s.scratch_bytes,m);
    flow_output_cuda(image.buffer.ptr,image.buffer.pitch,s.gray,s.key,ctl,m);
    ck(cudaEventRecord(image.consumed,m),"release MI consumed"); image.recorded=true;
}
void FlowMotionCuda::enqueue_mask(cudaStream_t m,const DetectionCandidates* d,const Control* ctl) {
    auto& s=*impl_; flow_mask_cuda(d,s.masks,s.key,ctl,m);
}
std::string FlowMotionCuda::library_path() const { return impl_->path; }
void FlowMotionCuda::close() { impl_->release().throw_if_failed(); }
}

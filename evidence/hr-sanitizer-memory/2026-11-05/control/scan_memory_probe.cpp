// Diagnostic-only exact deployed SelectiveScan shape probe; no TensorRT, JPEG or tracker.
#include "saccade/selective_scan_cuda.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <bit>
#include <charconv>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
void cu(cudaError_t code,const char* call) { if(code!=cudaSuccess) throw std::runtime_error(std::string(call)+": "+cudaGetErrorString(code)); }
std::size_t number(std::string_view text) { std::size_t n=0; const auto [end,e]=std::from_chars(text.data(),text.data()+text.size(),n); if(e!=std::errc{}||end!=text.data()+text.size()||!n) throw std::invalid_argument("expected positive integer"); return n; }
struct Buffer {
    std::size_t length, count;
    std::vector<void*> allocations;
    saccade::SelectiveScanDeviceInputs input;
    float* output=nullptr;
    saccade::ScanStatus* status=nullptr;
    float* zero(std::size_t n,cudaStream_t stream) {
        void* p=nullptr; cu(cudaMalloc(&p,n*sizeof(float)),"cudaMalloc tensor"); allocations.push_back(p);
        cu(cudaMemsetAsync(p,0,n*sizeof(float),stream),"zero tensor"); return static_cast<float*>(p);
    }
    Buffer(std::size_t l,cudaStream_t stream):length(l),count(4*l*256) {
        input.u=zero(count,stream);input.delta=zero(count,stream);input.a=zero(16,stream);
        input.b=zero(4*l*16,stream);input.c=zero(4*l,stream);input.d=zero(256,stream);
        output=zero(count,stream);void* p=nullptr;cu(cudaMalloc(&p,sizeof(*status)),"cudaMalloc status");allocations.push_back(p);status=static_cast<saccade::ScanStatus*>(p);
    }
    ~Buffer() { for(auto* p:allocations) cudaFree(p); }
    void enqueue(cudaStream_t stream) { saccade::selective_scan_cuda({4,length,256,16,1,1,true},input,output,status,stream); }
    void verify() {
        saccade::ScanStatus s{};cu(cudaMemcpy(&s,status,sizeof(s),cudaMemcpyDeviceToHost),"read status");
        if(s.flags!=0||s.done!=saccade::kScanDone) throw std::runtime_error("bad ScanStatus");
        std::vector<float> values(count);cu(cudaMemcpy(values.data(),output,count*sizeof(float),cudaMemcpyDeviceToHost),"read output");
        if(!std::all_of(values.begin(),values.end(),[](float f){return std::bit_cast<std::uint32_t>(f)==0;})) throw std::runtime_error("zero-input output not exact +0 bits");
        std::cout<<"VERIFY length="<<length<<" elements="<<count<<" flags="<<s.flags<<" done="<<s.done<<" zero_bits=PASS\n"<<std::flush;
    }
};
}
int main(int argc,char** argv) {
    cudaStream_t stream=nullptr;
    try {
        std::size_t length=60,repeats=1;std::string sync="each",sequence="single";
        for(int i=1;i<argc;++i) {
            const std::string_view k=argv[i];if(i+1>=argc) throw std::invalid_argument("usage: scan_memory_probe [--length N] [--repeats N] [--sync each|end] [--sequence single|hr]");
            const std::string_view v=argv[++i];
            if(k=="--length")length=number(v);else if(k=="--repeats")repeats=number(v);else if(k=="--sync")sync=v;else if(k=="--sequence")sequence=v;else throw std::invalid_argument("unknown flag");
        }
        if(length>100000||repeats>10000||(sync!="each"&&sync!="end")||(sequence!="single"&&sequence!="hr")) throw std::invalid_argument("invalid bounds or mode");
        cu(cudaSetDevice(0),"set device");cu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");
        std::cout<<"probe=selective-scan-only/1 batch=4 channels=256 state=16 a_rows=1 c_width=1 has_d=true repeats="<<repeats<<" sync="<<sync<<" sequence="<<sequence<<"\n"<<saccade::selective_scan_cuda_identity()<<std::flush;
        std::vector<std::unique_ptr<Buffer>> buffers;
        if(sequence=="hr") for(const auto l:{60U,240U,960U})buffers.push_back(std::make_unique<Buffer>(l,stream));
        else buffers.push_back(std::make_unique<Buffer>(length,stream));
        cu(cudaStreamSynchronize(stream),"initialization drain");
        std::cout<<"INITIALIZED\n"<<std::flush;
        std::cout<<saccade::selective_scan_cuda_load_kernels()<<std::flush;
        std::cout<<"KERNELS_LOADED\n"<<std::flush;
        for(std::size_t n=0;n<repeats;++n) for(auto& b:buffers) {
            std::cout<<"ENQUEUE_BEGIN repeat="<<n<<" length="<<b->length<<"\n"<<std::flush;b->enqueue(stream);
            std::cout<<"ENQUEUE_RETURN repeat="<<n<<" length="<<b->length<<"\n"<<std::flush;
            if(sync=="each") { std::cout<<"SYNC_BEGIN\n"<<std::flush;cu(cudaStreamSynchronize(stream),"per-call drain");std::cout<<"SYNC_DONE\n"<<std::flush; }
        }
        std::cout<<"FINAL_SYNC_BEGIN\n"<<std::flush;cu(cudaStreamSynchronize(stream),"final drain");std::cout<<"FINAL_SYNC_DONE\n"<<std::flush;
        for(auto& b:buffers) { b->verify(); }
        buffers.clear();cu(cudaStreamDestroy(stream),"destroy stream");stream=nullptr;
        std::cout<<"scan_memory_probe PASS\n"<<std::flush;return 0;
    } catch(const std::exception& e) { if(stream){cudaStreamSynchronize(stream);cudaStreamDestroy(stream);} std::cerr<<"scan_memory_probe: "<<e.what()<<'\n';return 1; }
}

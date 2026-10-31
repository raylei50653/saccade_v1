// Independent nvJPEG HW batch-one repro: no saccade library, no F02 kernels.
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static int initialization = -1;
static std::size_t allocations = 0, releases = 0;
static bool decoding = false;
static int allocate(void** p, std::size_t bytes) {
    const auto rc = cudaMalloc(p, bytes);
    ++allocations;
    std::cerr << "allocator=custom ptr=" << *p << " bytes=" << bytes
              << " fill=" << initialization << " phase=" << (decoding ? "decode" : "init") << '\n';
    if (rc != cudaSuccess || initialization < 0) return static_cast<int>(rc);
    const auto initialized = cudaMemset(*p, initialization, bytes);
    return static_cast<int>(initialized);
}
static int release(void* p) { ++releases; return static_cast<int>(cudaFree(p)); }
static int allocate_v2(void*, void** p, std::size_t bytes, cudaStream_t stream) {
    const auto rc=cudaMalloc(p,bytes);
    ++allocations;
    std::cerr << "allocator=v2 ptr=" << *p << " bytes=" << bytes << " fill=" << initialization
              << " phase=" << (decoding ? "decode" : "init") << " stream=" << stream << '\n';
    if(rc!=cudaSuccess || initialization<0)return static_cast<int>(rc);
    return static_cast<int>(cudaMemsetAsync(*p,initialization,bytes,stream));
}
static int release_v2(void*,void* p,std::size_t,cudaStream_t) { return release(p); }
static void ck(cudaError_t rc, const char* op) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(rc));
}
static void nj(nvjpegStatus_t rc, const char* op) {
    if (rc != NVJPEG_STATUS_SUCCESS) throw std::runtime_error(std::string(op)+": "+std::to_string(rc));
}
struct Owned {
    nvjpegHandle_t handle{};
    nvjpegJpegState_t state{};
    cudaStream_t stream{};
    unsigned char *input{}, *rgb{}, *download{};
    ~Owned() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (state) (void)nvjpegJpegStateDestroy(state);
        if (handle) (void)nvjpegDestroy(handle);
        if (rgb) (void)cudaFree(rgb);
        if (input) (void)cudaFreeHost(input);
        if (download) (void)cudaFreeHost(download);
        if (stream) (void)cudaStreamDestroy(stream);
    }
};
static std::vector<unsigned char> read(const char* name) {
    std::ifstream in(name, std::ios::binary);
    if (!in) throw std::runtime_error(std::string("open ")+name);
    return {std::istreambuf_iterator<char>(in), {}};
}
int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: repro default|custom-dirty|custom-zero|custom-a5 JPEG EXPECTED_RGB OUTPUT_RGB");
        const std::string mode=argv[1];
        const bool v2=mode.starts_with("v2-");
        const bool hybrid=mode=="hybrid";
        const bool custom=mode!="default" && !hybrid;
        if (mode=="custom-zero" || mode=="v2-zero") initialization=0;
        else if(mode=="custom-a5" || mode=="v2-a5") initialization=0xa5;
        else if(mode!="default" && mode!="custom-dirty" && mode!="v2-dirty" && !hybrid) throw std::runtime_error("bad allocator mode");
        const auto encoded=read(argv[2]);
        const auto expected=read(argv[3]);
        Owned o;
        ck(cudaStreamCreateWithFlags(&o.stream,cudaStreamNonBlocking),"stream create");
        nvjpegDevAllocator_t allocator{allocate,release};
        nvjpegDevAllocatorV2_t allocator_v2{allocate_v2,release_v2,nullptr};
        if(v2)nj(nvjpegCreateExV2(NVJPEG_BACKEND_HARDWARE,&allocator_v2,nullptr,NVJPEG_FLAGS_DEFAULT,&o.handle),"nvjpegCreateExV2 HARDWARE");
        else nj(nvjpegCreateEx(hybrid?NVJPEG_BACKEND_HYBRID:NVJPEG_BACKEND_HARDWARE,custom?&allocator:nullptr,nullptr,NVJPEG_FLAGS_DEFAULT,&o.handle),"nvjpegCreateEx");
        nj(nvjpegJpegStateCreate(o.handle,&o.state),"state create");
        int major=0,minor=0,patch=0;
        nj(nvjpegGetProperty(MAJOR_VERSION,&major),"major");
        nj(nvjpegGetProperty(MINOR_VERSION,&minor),"minor");
        nj(nvjpegGetProperty(PATCH_LEVEL,&patch),"patch");
        int components=0,w[NVJPEG_MAX_COMPONENT]{},h[NVJPEG_MAX_COMPONENT]{};
        nvjpegChromaSubsampling_t sampling{};
        nj(nvjpegGetImageInfo(o.handle,encoded.data(),encoded.size(),&components,&sampling,w,h),"image info");
        if (components!=3||sampling!=NVJPEG_CSS_420||w[0]<=0||h[0]<=0) throw std::runtime_error("unexpected fixture format");
        nj(nvjpegDecodeBatchedInitialize(o.handle,o.state,1,1,NVJPEG_OUTPUT_RGB),"batch init");
        nj(nvjpegDecodeBatchedPreAllocate(o.handle,o.state,1,w[0],h[0],NVJPEG_CSS_420,NVJPEG_OUTPUT_RGB),"batch preallocate");
        const std::size_t plane=std::size_t(w[0])*h[0], count=3*plane;
        if (expected.size()!=count) throw std::runtime_error("wrong expected size");
        ck(cudaMallocHost(&o.input,encoded.size()),"pinned input");
        ck(cudaMallocHost(&o.download,count),"pinned output");
        ck(cudaMalloc(&o.rgb,count),"RGB allocation");
        std::memcpy(o.input,encoded.data(),encoded.size());
        nvjpegImage_t destination{};
        for(int c=0;c<3;++c){destination.channel[c]=o.rgb+c*plane;destination.pitch[c]=w[0];}
        const unsigned char* input=o.input;
        const std::size_t length=encoded.size();
        std::vector<unsigned char> first;
        for(int repeat=0;repeat<2;++repeat){
            decoding=true;
            nj(nvjpegDecodeBatched(o.handle,o.state,&input,&length,&destination,o.stream),"decode HW");
            ck(cudaStreamSynchronize(o.stream),"decode completion");
            ck(cudaMemcpyAsync(o.download,o.rgb,count,cudaMemcpyDeviceToHost,o.stream),"D2H");
            ck(cudaStreamSynchronize(o.stream),"D2H completion");
            decoding=false;
            if(!std::equal(expected.begin(),expected.end(),o.download)) throw std::runtime_error("RGB differs from djpeg nosmooth reference");
            if(repeat==0)first.assign(o.download,o.download+count);
            else if(!std::equal(first.begin(),first.end(),o.download))throw std::runtime_error("repeat differs");
        }
        std::ofstream out(argv[4],std::ios::binary);
        out.write(reinterpret_cast<const char*>(o.download),count);out.close();
        if(!out)throw std::runtime_error("output write/close failed");
        std::cout << "nvjpeg="<<major<<'.'<<minor<<'.'<<patch<<" mode="<<mode
                  <<" size="<<w[0]<<'x'<<h[0]<<" rgb_exact=1 repeat_exact=1 allocations="<<allocations<<'\n';
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}

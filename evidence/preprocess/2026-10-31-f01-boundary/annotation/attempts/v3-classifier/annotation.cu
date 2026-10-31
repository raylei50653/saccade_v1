// Diagnostic only. Derived ownership/checking pattern from standalone repro.cu.
// Current-process Initcheck discovery; NVTX alters checks, never device bytes.
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvjpeg.h>
#include "nvtx_mark.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
static void cu(cudaError_t s, const char* op) {
    std::cerr << "api=" << op << " cuda_status=" << static_cast<int>(s) << '\n';
    if (s != cudaSuccess) throw std::runtime_error(op);
}
static void drv(CUresult s, const char* op) {
    std::cerr << "api=" << op << " driver_status=" << static_cast<int>(s) << '\n';
    if (s != CUDA_SUCCESS) throw std::runtime_error(op);
}
static void nj(nvjpegStatus_t s, const char* op) {
    std::cerr << "api=" << op << " nvjpeg_status=" << static_cast<int>(s) << '\n';
    if (s != NVJPEG_STATUS_SUCCESS) throw std::runtime_error(op);
}
static std::vector<unsigned char> read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open failed: " + path);
    std::vector<unsigned char> v{std::istreambuf_iterator<char>(f), {}};
    if (f.bad()) throw std::runtime_error("read failed: " + path);
    return v;
}
static void write_bytes(const std::string& path, const std::vector<unsigned char>& v) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("create failed: " + path);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size()));
    f.close();
    if (!f) throw std::runtime_error("write failed: " + path);
}

struct ReadRecord { unsigned x, y; std::uint64_t address; };
struct Parsed { std::vector<ReadRecord> reads; std::size_t uninitialized_headers{}, bytes{}; };
static Parsed parse_log(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::string text{std::istreambuf_iterator<char>(f), {}};
    Parsed p; p.bytes = text.size();
    std::istringstream lines(text); std::string line;
    bool target = false, have_thread = false, size_three = false;
    unsigned tx=0,ty=0,tz=0,bx=0,by=0,bz=0;
    while (std::getline(lines, line)) {
        if (line.find("Uninitialized ") != std::string::npos) {
            ++p.uninitialized_headers; target=false; have_thread=false;
            size_three=line.find("read of size 3 bytes")!=std::string::npos;
        }
        if (line.find("nvjpeg::rgba2rgb<(nvjpegOutputFormat_t)3,") != std::string::npos &&
            line.find("+0x170") != std::string::npos && size_three) target=true;
        if (target && std::sscanf(line.c_str(), "=========     by thread (%u,%u,%u) in block (%u,%u,%u)",
                                  &tx,&ty,&tz,&bx,&by,&bz)==6) {
            if (tx>=32 || ty>=8 || tz || bz) throw std::runtime_error("unexpected converter thread dimensions");
            have_thread=true;
        }
        unsigned long long address=0;
        if (target && have_thread && std::sscanf(line.c_str(), "=========     Address 0x%llx", &address)==1) {
            p.reads.push_back({32*bx+tx,8*by+ty,address}); target=false; have_thread=false;
        }
    }
    return p;
}
struct Surface { CUdeviceptr source{}, base{}; std::size_t bytes{}, pitch{}; unsigned long long id{}; };
static Surface identity(CUdeviceptr source, std::size_t pitch, const char* point, int repeat) {
    Surface s; s.source=source;s.pitch=pitch;
    drv(cuMemGetAddressRange(&s.base,&s.bytes,source), "surface_range");
    drv(cuPointerGetAttribute(&s.id,CU_POINTER_ATTRIBUTE_BUFFER_ID,source), "surface_id");
    std::cout << "surface point=" << point << " repeat=" << repeat << " source=0x" << std::hex << source
              << " base=0x" << s.base << std::dec << " bytes=" << s.bytes << " pitch=" << pitch << " id=" << s.id << std::endl;
    return s;
}
static Surface discover(const std::string& path, unsigned width, unsigned height) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    Parsed p;
    do {
        p=parse_log(path);
        if (p.reads.size()>=static_cast<std::size_t>(width)*height) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    } while (std::chrono::steady_clock::now()<deadline);
    if (p.reads.size()!=static_cast<std::size_t>(width)*height || p.uninitialized_headers!=p.reads.size())
        throw std::runtime_error("discovery unavailable: expected exactly complete warmup valid-pixel reports");
    std::map<std::pair<unsigned,unsigned>,std::uint64_t> address;
    for (const auto& r:p.reads) {
        if (r.x>=width || r.y>=height || !address.emplace(std::make_pair(r.x,r.y),r.address).second)
            throw std::runtime_error("discovery duplicate/out-of-range pixel");
    }
    const auto base=address.at({0,0});
    const auto stride=address.at({1,0})-base;
    const auto pitch=address.at({0,1})-base;
    if (stride!=4 || pitch<4*width || pitch>1U<<20) throw std::runtime_error("unrecognized private surface layout");
    for (unsigned y=0;y<height;++y) for(unsigned x=0;x<width;++x)
        if(address.at({x,y})!=base+pitch*y+4*x) throw std::runtime_error("private surface affine address mismatch");
    auto s=identity(base,pitch,"discovered",0);
    const auto footprint=pitch*(height-1)+4*width;
    if (base<s.base || base-s.base>s.bytes || footprint>s.bytes-(base-s.base))
        throw std::runtime_error("valid pixel footprint exceeds live allocation");
    std::cout << "discovery current_process=1 complete_rgb_records=" << p.reads.size() << " headers=" << p.uninitialized_headers
              << " valid_coverage=" << address.size() << " pixel_stride=" << stride << " pitch=" << pitch << " tool_bytes=" << p.bytes << std::endl;
    return s;
}
static void stable(const Surface& expected, const char* point, int repeat) {
    const auto got=identity(expected.source,expected.pitch,point,repeat);
    if(got.base!=expected.base || got.bytes!=expected.bytes || got.id!=expected.id)
        throw std::runtime_error("private allocation generation changed; annotation association invalid");
}

struct Owned {
    cudaStream_t stream{}; nvjpegHandle_t handle{}; nvjpegJpegState_t state{};
    unsigned char *input{},*output{},*download{}; nvtxDomainHandle_t domain{}; bool closed=false;
    void close(bool checked) {
        if(closed)return;closed=true;std::string failures;
        auto c=[&](cudaError_t rc,const char* op){std::cerr<<"cleanup="<<op<<" cuda_status="<<static_cast<int>(rc)<<'\n';if(rc!=cudaSuccess)failures+=std::string(op)+";";};
        auto n=[&](nvjpegStatus_t rc,const char* op){std::cerr<<"cleanup="<<op<<" nvjpeg_status="<<static_cast<int>(rc)<<'\n';if(rc!=NVJPEG_STATUS_SUCCESS)failures+=std::string(op)+";";};
        if(stream)c(cudaStreamSynchronize(stream),"stream_drain");
        if(domain)nvtxDomainDestroy(domain);
        if(state)n(nvjpegJpegStateDestroy(state),"state_destroy");
        if(handle)n(nvjpegDestroy(handle),"handle_destroy");
        if(output)c(cudaFree(output),"output_free");
        if(download)c(cudaFreeHost(download),"download_free");
        if(input)c(cudaFreeHost(input),"input_free");
        if(stream)c(cudaStreamDestroy(stream),"stream_destroy");
        if(checked && !failures.empty())throw std::runtime_error(failures);
    }
    ~Owned(){try{close(false);}catch(...){}}
};

int main(int argc,char** argv) {
    Owned o;
    try {
        if(argc!=9)throw std::runtime_error("usage: annotation none|rgb|alpha|caller TOOL_LOG A.jpg A.rgb B.jpg B.rgb 35x27 OUTPUT_DIR");
        const std::string mode=argv[1],tool_log=argv[2],out=argv[8];
        if(mode!="none" && mode!="rgb" && mode!="alpha" && mode!="caller")throw std::runtime_error("invalid mode");
        if(!fs::is_directory(out))throw std::runtime_error("output directory absent");
        const std::array<std::vector<unsigned char>,2> encoded{read_bytes(argv[3]),read_bytes(argv[5])};
        const std::array<std::vector<unsigned char>,2> expected{read_bytes(argv[4]),read_bytes(argv[6])};
        // argv[7] is an explicit fixed dimensions contract, not an inferred layout.
        if(std::string(argv[7])!="35x27")throw std::runtime_error("only frozen 35x27 inputs are supported");
        if(encoded[0]==encoded[1] || expected[0]==expected[1])throw std::runtime_error("controls must have distinct JPEG and RGB bytes");
        constexpr unsigned width=35,height=27;constexpr std::size_t pixels=width*height,bytes=3*pixels;
        for(const auto& e:expected)if(e.size()!=bytes)throw std::runtime_error("reference size mismatch");
        cu(cudaStreamCreateWithFlags(&o.stream,cudaStreamNonBlocking),"stream_create");
        o.domain=nvtxDomainCreateA("F01 boundary diagnostic");
        nj(nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE,nullptr,nullptr,NVJPEG_FLAGS_DEFAULT,&o.handle),"create_hw");
        nj(nvjpegJpegStateCreate(o.handle,&o.state),"state_create");
        for(const auto& input:encoded) {
            int components=0;nvjpegChromaSubsampling_t sampling{};std::array<int,4>w{},h{};
            nj(nvjpegGetImageInfo(o.handle,input.data(),input.size(),&components,&sampling,w.data(),h.data()),"get_info");
            if(components!=3 || sampling!=NVJPEG_CSS_420 || w[0]!=width || h[0]!=height)throw std::runtime_error("input metadata mismatch");
        }
        nj(nvjpegDecodeBatchedInitialize(o.handle,o.state,1,1,NVJPEG_OUTPUT_RGB),"batch_initialize");
        nj(nvjpegDecodeBatchedPreAllocate(o.handle,o.state,1,width,height,NVJPEG_CSS_420,NVJPEG_OUTPUT_RGB),"preallocate");
        const auto input_capacity=std::max(encoded[0].size(),encoded[1].size());
        cu(cudaMallocHost(&o.input,input_capacity),"input_allocate");
        cu(cudaMallocHost(&o.download,bytes),"download_allocate");
        cu(cudaMalloc(&o.output,bytes),"output_allocate");
        nvjpegImage_t dst{};for(unsigned c=0;c<3;++c){dst.channel[c]=o.output+c*pixels;dst.pitch[c]=width;}
        std::cout << "diagnostic mode=" << mode << " dimensions=35x27 explicit_stream=" << o.stream << " caller_output=" << static_cast<void*>(o.output) << " harness_device_writes=none" << std::endl;
        Surface surface{};
        for(int repeat=0;repeat<3;++repeat) {
            const int which=repeat==1?1:0;
            if(repeat) {
                stable(surface,"before_annotation",repeat);
                std::vector<nvtxMemVirtualRangeDesc_t> ranges;
                const auto* b=reinterpret_cast<const unsigned char*>(surface.source);
                if(mode=="rgb")for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)ranges.push_back({3,b+y*surface.pitch+4*x});
                if(mode=="alpha")for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)ranges.push_back({1,b+y*surface.pitch+4*x+3});
                if(mode=="caller")ranges.push_back({bytes,o.output});
                if(!ranges.empty())f01_boundary::mark_ranges(o.domain,o.stream,ranges);
                // Converter may use an SDK-private stream. Finish the caller-
                // stream annotation before invoking decode, in every control.
                // This extra synchronization is a diagnostic intervention.
                cu(cudaStreamSynchronize(o.stream),"annotation_diagnostic_sync");
                std::size_t marked_bytes=0;for(const auto& r:ranges)marked_bytes+=r.size;
                std::cout<<"annotation repeat="<<repeat<<" mode="<<mode<<" ranges="<<ranges.size()<<" bytes="<<marked_bytes<<" changes_device_bytes=0 acknowledged=unknown"<<std::endl;
            }
            std::copy(encoded[which].begin(),encoded[which].end(),o.input);
            const unsigned char* data=o.input;const std::size_t length=encoded[which].size();
            std::cout<<"phase=decode repeat="<<repeat<<" input="<<(which?"B":"A")<<std::endl;
            nj(nvjpegDecodeBatched(o.handle,o.state,&data,&length,&dst,o.stream),"decode");
            cu(cudaStreamSynchronize(o.stream),"decode_sync");
            if(!repeat)surface=discover(tool_log,width,height);else stable(surface,"after_decode",repeat);
            const auto snapshot=parse_log(tool_log);
            std::cout<<"tool_checkpoint repeat="<<repeat<<" complete_records="<<snapshot.reads.size()<<" uninitialized_headers="<<snapshot.uninitialized_headers<<" bytes="<<snapshot.bytes<<" completeness_after_warmup=not_assumed"<<std::endl;
            cu(cudaMemcpyAsync(o.download,o.output,bytes,cudaMemcpyDeviceToHost,o.stream),"download");
            cu(cudaStreamSynchronize(o.stream),"download_sync");
            std::vector<unsigned char> actual(o.download,o.download+bytes);
            write_bytes(out+"/repeat-"+std::to_string(repeat)+".rgb",actual);
            std::size_t differences=0;for(std::size_t i=0;i<bytes;++i)differences+=actual[i]!=expected[which][i];
            std::cout<<"output repeat="<<repeat<<" input="<<(which?"B":"A")<<" rgb_exact="<<(differences==0)<<" differences="<<differences<<std::endl;
            if(differences)throw std::runtime_error("RGB differs from fixed independent reference");
        }
        std::ifstream maps("/proc/self/maps");std::ofstream saved(out+"/loaded.maps");saved<<maps.rdbuf();saved.close();
        if(!maps || !saved)throw std::runtime_error("failed to retain process maps");
        o.close(true);
        std::cout<<"result=complete warmup_fail_preserved=1 discovery=current_process repeats=3 sequence=A,B,A rgb_exact=1 diagnostic_only=1"<<std::endl;
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"result=failed error="<<e.what()<<std::endl;
        try{o.close(true);}catch(const std::exception& c){std::cerr<<"cleanup_failed="<<c.what()<<std::endl;}
        return 1;
    }
}

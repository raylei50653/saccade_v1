// Standalone diagnostic: links CUDA/nvJPEG only; no Saccade runtime or F02.
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

static void cu(cudaError_t status, const char* op) {
    std::cerr << "api=" << op << " cuda_status=" << static_cast<int>(status) << '\n';
    if (status != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(status));
}
static void nj(nvjpegStatus_t status, const char* op) {
    std::cerr << "api=" << op << " nvjpeg_status=" << static_cast<int>(status) << '\n';
    if (status != NVJPEG_STATUS_SUCCESS) throw std::runtime_error(std::string(op) + ": status " + std::to_string(status));
}
static std::vector<unsigned char> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("open failed: " + path);
    std::vector<unsigned char> result{std::istreambuf_iterator<char>(in), {}};
    if (in.bad()) throw std::runtime_error("read failed: " + path);
    return result;
}
static void write_bytes(const std::string& path, const std::vector<unsigned char>& value) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("create failed: " + path);
    out.write(reinterpret_cast<const char*>(value.data()), static_cast<std::streamsize>(value.size()));
    out.close();
    if (!out) throw std::runtime_error("write/close failed: " + path);
}

// This deliberately reads the caller-owned decode destination. It separates
// initialization of public output from the SDK's private rgba2rgb input surface.
__global__ void inspect_caller_output(const unsigned char* decoded, unsigned char* observed, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) observed[i] = decoded[i];
}

struct Owned {
    cudaStream_t stream{};
    nvjpegHandle_t handle{};
    nvjpegJpegState_t state{};
    unsigned char *input{}, *output{}, *observed{}, *download{};
    bool closed = false;
    // Explicit close checks every result and attempts every independent release.
    void close(bool checked) {
        if (closed) return;
        closed = true;
        std::string failures;
        auto record_cu = [&](cudaError_t rc, const char* op) {
            std::cerr << "cleanup=" << op << " cuda_status=" << static_cast<int>(rc) << '\n';
            if (rc != cudaSuccess) failures += std::string(op) + ": " + cudaGetErrorString(rc) + "; ";
        };
        auto record_nj = [&](nvjpegStatus_t rc, const char* op) {
            std::cerr << "cleanup=" << op << " nvjpeg_status=" << static_cast<int>(rc) << '\n';
            if (rc != NVJPEG_STATUS_SUCCESS) failures += std::string(op) + ": " + std::to_string(rc) + "; ";
        };
        if (stream) record_cu(cudaStreamSynchronize(stream), "stream_drain");
        if (state) record_nj(nvjpegJpegStateDestroy(state), "state_destroy");
        if (handle) record_nj(nvjpegDestroy(handle), "handle_destroy");
        if (observed) record_cu(cudaFree(observed), "observed_free");
        if (output) record_cu(cudaFree(output), "output_free");
        if (download) record_cu(cudaFreeHost(download), "download_free");
        if (input) record_cu(cudaFreeHost(input), "input_free");
        if (stream) record_cu(cudaStreamDestroy(stream), "stream_destroy");
        if (checked && !failures.empty()) throw std::runtime_error(failures);
    }
    ~Owned() { try { close(false); } catch (...) {} }
};

int main(int argc, char** argv) {
    Owned owned;
    try {
        if (argc != 10) throw std::runtime_error("usage: repro hw|hybrid|skip rgb|rgbi|yuv|y|nv12 default|no-pipeline preallocate|no-preallocate uninit|zero|a5 REPEATS JPEG EXPECTED_PLANAR_RGB OUTPUT");
        const std::string backend = argv[1], format = argv[2], flags_name = argv[3], preallocation = argv[4], fill = argv[5];
        if (backend != "hw" && backend != "hybrid" && backend != "skip") throw std::runtime_error("invalid backend");
        const nvjpegBackend_t backend_value = backend == "hybrid" ? NVJPEG_BACKEND_HYBRID : NVJPEG_BACKEND_HARDWARE;
        nvjpegOutputFormat_t output_format;
        if (format == "rgb") output_format = NVJPEG_OUTPUT_RGB;
        else if (format == "rgbi") output_format = NVJPEG_OUTPUT_RGBI;
        else if (format == "yuv") output_format = NVJPEG_OUTPUT_YUV;
        else if (format == "y") output_format = NVJPEG_OUTPUT_Y;
        else if (format == "nv12") output_format = NVJPEG_OUTPUT_NV12;
        else throw std::runtime_error("invalid format");
        if (flags_name != "default" && flags_name != "no-pipeline") throw std::runtime_error("invalid flags");
        const unsigned flags = flags_name == "default" ? NVJPEG_FLAGS_DEFAULT : NVJPEG_FLAGS_HW_DECODE_NO_PIPELINE;
        if (preallocation != "preallocate" && preallocation != "no-preallocate") throw std::runtime_error("invalid preallocation");
        if (fill != "uninit" && fill != "zero" && fill != "a5") throw std::runtime_error("invalid fill");
        std::size_t consumed = 0;
        const int repeats = std::stoi(argv[6], &consumed);
        if (consumed != std::string(argv[6]).size() || repeats < 1 || repeats > 100) throw std::runtime_error("repeats must be 1..100");
        const auto encoded = read_bytes(argv[7]);
        const auto expected = read_bytes(argv[8]);
        if (encoded.empty()) throw std::runtime_error("JPEG empty");
        cu(cudaStreamCreateWithFlags(&owned.stream, cudaStreamNonBlocking), "stream_create");
        nj(nvjpegCreateEx(backend_value, nullptr, nullptr, flags, &owned.handle), "nvjpegCreateEx");
        nj(nvjpegJpegStateCreate(owned.handle, &owned.state), "nvjpegJpegStateCreate");
        int major=0, minor=0, patch=0;
        nj(nvjpegGetProperty(MAJOR_VERSION, &major), "version_major");
        nj(nvjpegGetProperty(MINOR_VERSION, &minor), "version_minor");
        nj(nvjpegGetProperty(PATCH_LEVEL, &patch), "version_patch");
        int components=0;
        std::array<int, NVJPEG_MAX_COMPONENT> widths{}, heights{};
        nvjpegChromaSubsampling_t sampling{};
        nj(nvjpegGetImageInfo(owned.handle, encoded.data(), encoded.size(), &components, &sampling, widths.data(), heights.data()), "nvjpegGetImageInfo");
        if (components != 3 || sampling != NVJPEG_CSS_420 || widths[0] <= 0 || heights[0] <= 0) throw std::runtime_error("control requires three-component 4:2:0 JPEG");
        const std::size_t width=static_cast<std::size_t>(widths[0]), height=static_cast<std::size_t>(heights[0]), pixels=width*height;
        if (expected.size() != 3*pixels) throw std::runtime_error("expected planar RGB length differs from dimensions");
        nj(nvjpegDecodeBatchedInitialize(owned.handle, owned.state, 1, 1, output_format), "nvjpegDecodeBatchedInitialize");
        if (preallocation == "preallocate") nj(nvjpegDecodeBatchedPreAllocate(owned.handle, owned.state, 1, widths[0], heights[0], sampling, output_format), "nvjpegDecodeBatchedPreAllocate");

        int planes = 0;
        std::array<std::size_t, 3> pitches{}, rows{}, sizes{};
        if (format == "rgb") {
            planes=3; for (int c=0;c<3;++c) {pitches[c]=width;rows[c]=height;}
        } else if (format == "rgbi") {planes=1;pitches[0]=3*width;rows[0]=height;}
        else if (format == "yuv") {
            planes=3;for(int c=0;c<3;++c){pitches[c]=static_cast<std::size_t>(widths[c]);rows[c]=static_cast<std::size_t>(heights[c]);}
        } else if (format == "y") {planes=1;pitches[0]=width;rows[0]=height;}
        else {planes=2;pitches[0]=width;rows[0]=height;pitches[1]=2*static_cast<std::size_t>(widths[1]);rows[1]=static_cast<std::size_t>(heights[1]);}
        std::size_t bytes=0;
        for(int c=0;c<planes;++c){sizes[c]=pitches[c]*rows[c];bytes+=sizes[c];}
        cu(cudaMallocHost(&owned.input, encoded.size()), "input_allocate");
        cu(cudaMallocHost(&owned.download, bytes), "download_allocate");
        cu(cudaMalloc(&owned.output, bytes), "output_allocate");
        cu(cudaMalloc(&owned.observed, bytes), "observed_allocate");
        std::copy(encoded.begin(), encoded.end(), owned.input);
        nvjpegImage_t destination{};
        std::size_t offset=0;
        for(int c=0;c<planes;++c){destination.channel[c]=owned.output+offset;destination.pitch[c]=pitches[c];offset+=sizes[c];std::cout<<"plane="<<c<<" pitch="<<pitches[c]<<" rows="<<rows[c]<<" bytes="<<sizes[c]<<'\n';}
        std::cout << "nvjpeg=" << major << '.' << minor << '.' << patch << " backend=" << backend << " format=" << format << " flags=" << flags_name << " flags_value=" << flags << " preallocation=" << preallocation << " fill=" << fill << " dimensions=" << width << 'x' << height << " repeats=" << repeats << " raw_bytes=" << bytes << " caller_output=" << static_cast<void*>(owned.output) << " observed=" << static_cast<void*>(owned.observed) << std::endl;
        const unsigned char* input=owned.input;
        const std::size_t length=encoded.size();
        std::vector<unsigned char> first, current, canonical;
        bool repeat_exact=true, rgb_exact=true;
        std::size_t rgb_differences=0;
        int rgb_max_difference=0;
        for(int repeat=0;repeat<repeats;++repeat){
            std::cerr<<"phase=decode repeat="<<repeat<<std::endl;
            if(fill != "uninit") cu(cudaMemsetAsync(owned.output, fill=="zero"?0:0xa5, bytes, owned.stream), "output_initialize");
            if(backend!="skip")nj(nvjpegDecodeBatched(owned.handle, owned.state, &input, &length, &destination, owned.stream), "nvjpegDecodeBatched");
            else std::cerr<<"api=nvjpegDecodeBatched status=SKIPPED positive-control=1"<<std::endl;
            cu(cudaStreamSynchronize(owned.stream), "decode_sync");
            std::cerr<<"phase=inspect_caller_output repeat="<<repeat<<std::endl;
            inspect_caller_output<<<static_cast<unsigned>((bytes+255)/256),256,0,owned.stream>>>(owned.output,owned.observed,bytes);
            cu(cudaGetLastError(), "inspect_launch");
            cu(cudaStreamSynchronize(owned.stream), "inspect_sync");
            cu(cudaMemcpyAsync(owned.download, owned.observed, bytes, cudaMemcpyDeviceToHost, owned.stream), "download");
            cu(cudaStreamSynchronize(owned.stream), "download_sync");
            current.assign(owned.download, owned.download+bytes);
            if(repeat==0)first=current;else repeat_exact=repeat_exact && current==first;
            canonical=current;
            if(format=="rgbi")for(std::size_t i=0;i<pixels;++i)for(std::size_t c=0;c<3;++c)canonical[c*pixels+i]=current[3*i+c];
            if(format=="rgb" || format=="rgbi"){
                std::size_t differences=0;int max_difference=0;
                for(std::size_t i=0;i<canonical.size();++i){int difference=static_cast<int>(canonical[i])-static_cast<int>(expected[i]);if(difference<0)difference=-difference;if(difference){++differences;max_difference=std::max(max_difference,difference);}}
                rgb_exact=rgb_exact && differences==0;rgb_differences+=differences;rgb_max_difference=std::max(rgb_max_difference,max_difference);
                std::cout<<"repeat="<<repeat<<" rgb_exact="<<(differences==0)<<" rgb_differences="<<differences<<" max_difference="<<max_difference<<std::endl;
            }
        }
        // RGB and RGBI files are canonical contiguous planar RGB. Others retain
        // the described packed valid planes; no RGB-equivalence claim is made.
        write_bytes(argv[9], canonical);
        owned.close(true);
        std::cout<<"result=complete repeat_exact="<<repeat_exact<<" rgb_equivalence="<<((format=="rgb" || format=="rgbi")?(rgb_exact?"exact":"different"):"not-compared")<<" rgb_differences_across_repeats="<<rgb_differences<<" rgb_max_difference="<<rgb_max_difference<<" output_layout="<<((format=="rgb" || format=="rgbi")?"planar-rgb":"raw-planes")<<std::endl;
        if(!repeat_exact) return 2;
        // Hybrid and skipped-decode mismatches are explicitly diagnostic. Exit
        // zero means the harness completed; rgb_equivalence still records the
        // mismatch. Hardware RGB mismatch remains a nonzero contract failure.
        if(backend=="hw" && (format=="rgb" || format=="rgbi") && !rgb_exact)return 3;
        return 0;
    }catch(const std::exception& error){
        std::cerr<<"result=failed error="<<error.what()<<std::endl;
        try{owned.close(true);}catch(const std::exception& cleanup){std::cerr<<"cleanup_failed="<<cleanup.what()<<std::endl;}
        return 1;
    }
}

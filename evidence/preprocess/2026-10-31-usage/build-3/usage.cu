// Normal public NativePreprocessor use; no private SDK memory or instrumentation.
#include "saccade/preprocess.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace saccade;
constexpr unsigned long long delay_cycles = 10000000;
constexpr int cycles = 4;
constexpr std::array<int, 13> sequence{0,4,1,4,2,4,3,4,5,4,6,4,0};
constexpr std::size_t rgb_capacity = 3u*1920u*1080u;
constexpr std::size_t f32_elements = 3u*640u*640u;
constexpr std::size_t f32_bytes = f32_elements*sizeof(float);

__global__ void consumer_delay(unsigned long long n) {
    const unsigned long long start = clock64();
    while (clock64()-start < n) { }
}

void require(bool value, const std::string& message) { if (!value) throw std::runtime_error(message); }
void ck(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(status));
}
std::string hash(const void* p, std::size_t n) {
    return sha256_hex(std::string_view(static_cast<const char*>(p), n));
}
struct Case {
    std::string label, jpeg, rgb, images;
    int width{}, height{};
    std::string rgb_hash, f32_hash;
};
std::vector<Case> load_cases(const fs::path& refs) {
    std::istringstream in(read_text_file(refs/"cases.tsv"));
    std::vector<Case> result;
    std::string line;
    while (std::getline(in,line) && result.size()<7) {
        if (line.empty() || line[0]=='#') continue;
        std::vector<std::string> fields;
        std::istringstream row(line);
        std::string field;
        while (std::getline(row,field,'\t')) fields.push_back(field);
        require(fields.size()==8,"cases.tsv columns");
        require(fields[5]!="-" && fields[3]=="640" && fields[4]=="640","seven real fixed model inputs required");
        Case c;
        c.label=fields[0]; c.width=std::stoi(fields[1]); c.height=std::stoi(fields[2]);
        c.jpeg=read_text_file(refs/fields[5]); c.rgb=read_text_file(refs/fields[6]); c.images=read_text_file(refs/fields[7]);
        require(c.rgb.size()==3u*static_cast<std::size_t>(c.width)*c.height && c.images.size()==f32_bytes,"reference capacities");
        c.rgb_hash=hash(c.rgb.data(),c.rgb.size()); c.f32_hash=hash(c.images.data(),c.images.size());
        result.push_back(std::move(c));
    }
    require(result.size()==7 && result[4].width==640 && result[4].height==480,"fixed seven cases/dimension switch absent");
    return result;
}

struct Slot {
    std::uint8_t* rgb=nullptr;
    float* images=nullptr;
    cudaEvent_t ready=nullptr;
    std::uint64_t frame{};
    int case_index{};
    std::size_t bytes{};
    ImageMapping mapping{};
};
struct Session {
    std::string name,phase;
    int window;
    const std::vector<Case>& cases;
    fs::path output;
    std::ostream& report;
    std::ostream& submissions;
    std::ostream& summaries;
    std::map<int,std::pair<std::string,std::string>>& global_first;
    std::map<int,std::pair<std::string,std::string>> first;
    cudaStream_t stream=nullptr;
    std::unique_ptr<NativePreprocessor> processor;
    std::array<Slot,3> slots;
    std::deque<int> pending;
    std::uint64_t next_frame=0,verified=0,not_ready=0,ready_at_enqueue=0;
    std::size_t high_water=0;
    bool closed=false;

    Session(std::string n, std::string p, int w, const std::vector<Case>& c,
            const fs::path& out, std::ostream& r, std::ostream& sub, std::ostream& sum,
            std::map<int,std::pair<std::string,std::string>>& global)
        : name(std::move(n)),phase(std::move(p)),window(w),cases(c),output(out),
          report(r),submissions(sub),summaries(sum),global_first(global) {
        try {
            ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create independent non-default stream");
            for (auto& slot:slots) {
                ck(cudaMallocHost(&slot.rgb,rgb_capacity),"allocate independent pinned RGB readback");
                ck(cudaMallocHost(&slot.images,f32_bytes),"allocate independent pinned FP32 readback");
                ck(cudaEventCreateWithFlags(&slot.ready,cudaEventDisableTiming),"create consumer completion event");
            }
            processor=std::make_unique<NativePreprocessor>(stream);
            write_text_file(output/(name+"-identity.txt"),processor->identity());
            write_text_file(output/(name+"-loaded-maps.txt"),read_text_file("/proc/self/maps"));
            std::cout<<"session="<<name<<" phase="<<phase<<" window="<<window<<" stream="<<stream<<" ring_slots=3 delay_cycles="<<delay_cycles<<std::endl;
        } catch (...) { cleanup(false); throw; }
    }
    Session(const Session&)=delete;
    Session& operator=(const Session&)=delete;
    ~Session() { cleanup(false); }

    void complete_front(const char* reason) {
        require(!pending.empty(),"drain empty queue");
        const int index=pending.front();
        Slot& slot=slots[index];
        ck(cudaEventSynchronize(slot.ready),"readback slot reuse/drain");
        const Case& c=cases[slot.case_index];
        const bool rgb_exact=std::memcmp(slot.rgb,c.rgb.data(),slot.bytes)==0;
        const bool f32_exact=std::memcmp(slot.images,c.images.data(),f32_bytes)==0;
        bool finite=true;
        for (std::size_t i=0;i<f32_elements;++i) finite=finite && std::isfinite(slot.images[i]);
        const auto rh=hash(slot.rgb,slot.bytes), fh=hash(slot.images,f32_bytes);
        const auto current=std::make_pair(rh,fh);
        const bool repeat=first.emplace(slot.case_index,current).first->second==current;
        const bool cross=global_first.emplace(slot.case_index,current).first->second==current;
        const bool mapping=slot.mapping.original_width==c.width && slot.mapping.original_height==c.height
            && slot.mapping.model_width==640 && slot.mapping.model_height==640
            && slot.mapping.scale_x==640.0/c.width && slot.mapping.scale_y==640.0/c.height;
        report<<name<<'\t'<<phase<<'\t'<<window<<'\t'<<slot.frame<<'\t'<<slot.case_index<<'\t'<<c.label<<'\t'
              <<c.width<<'\t'<<c.height<<'\t'<<slot.bytes<<'\t'<<f32_elements<<'\t'
              <<rgb_exact<<'\t'<<f32_exact<<'\t'<<finite<<'\t'<<repeat<<'\t'<<cross<<'\t'<<mapping<<'\t'
              <<rh<<'\t'<<fh<<'\t'<<c.rgb_hash<<'\t'<<c.f32_hash<<'\t'<<reason<<'\n'<<std::flush;
        require(static_cast<bool>(report),"report flush failed");
        pending.pop_front();
        ++verified;
        if (!(rgb_exact && f32_exact && finite && repeat && cross && mapping)) {
            const auto stem=name+"-frame-"+std::to_string(slot.frame);
            write_binary_file(output/(stem+"-actual.rgb"),{slot.rgb,slot.bytes});
            write_binary_file(output/(stem+"-actual.f32"),{reinterpret_cast<const std::uint8_t*>(slot.images),f32_bytes});
            throw std::runtime_error("usage output mismatch: "+stem+" case="+std::to_string(slot.case_index));
        }
    }

    void submit(int case_index) {
        require(!closed,"closed session");
        if (pending.size()==static_cast<std::size_t>(window)) complete_front("slot-reuse");
        const auto& c=cases.at(static_cast<std::size_t>(case_index));
        std::string caller_jpeg=c.jpeg;
        int query=-1,query_last_error=-1;
        if (!pending.empty()) {
            const auto status=cudaEventQuery(slots[pending.back()].ready);
            query=static_cast<int>(status);
            if (status==cudaErrorNotReady) {
                ++not_ready;
                const auto last=cudaGetLastError();
                query_last_error=static_cast<int>(last);
                if (last!=cudaErrorNotReady) ck(last,"acknowledge expected event NotReady without hiding other CUDA errors");
            }
            else { ck(status,"query pending downstream consumer event"); ++ready_at_enqueue; }
        }
        // There is no host output wait here. enqueue may wait for its own prior
        // F02 event; our delayed consumer event was recorded AFTER that event.
        const auto frame=processor->enqueue({reinterpret_cast<const std::uint8_t*>(caller_jpeg.data()),caller_jpeg.size()},
                                             next_frame,c.width,c.height);
        require(frame.frame==next_frame,"returned frame identity mismatch");
        const int index=static_cast<int>(next_frame%slots.size());
        require(std::find(pending.begin(),pending.end(),index)==pending.end(),"pinned readback slot reused while pending");
        Slot& slot=slots[index];
        slot.frame=frame.frame; slot.case_index=case_index; slot.bytes=c.rgb.size(); slot.mapping=frame.mapping;
        // This caller-owned GPU consumer deliberately waits on the SAME detect
        // stream before reading borrowed RAW/images. Queue both D2H copies and
        // their completion event before any following enqueue for this session.
        consumer_delay<<<1,1,0,stream>>>(delay_cycles);
        ck(cudaGetLastError(),"launch bounded downstream consumer delay");
        ck(cudaMemcpyAsync(slot.rgb,frame.rgb,slot.bytes,cudaMemcpyDeviceToHost,stream),"queue pinned RGB consumer");
        ck(cudaMemcpyAsync(slot.images,frame.images,f32_bytes,cudaMemcpyDeviceToHost,stream),"queue pinned FP32 consumer");
        ck(cudaEventRecord(slot.ready,stream),"record downstream readback completion");
        std::fill(caller_jpeg.begin(),caller_jpeg.end(),'\x3c');
        pending.push_back(index);
        high_water=std::max(high_water,pending.size());
        submissions<<name<<'\t'<<phase<<'\t'<<window<<'\t'<<next_frame<<'\t'<<case_index<<'\t'<<query<<'\t'<<query_last_error<<'\t'
                   <<pending.size()<<'\t'<<index<<'\t'<<delay_cycles<<"\t1\n"<<std::flush;
        require(static_cast<bool>(submissions),"submission log flush failed");
        ++next_frame;
    }

    void finish() {
        while (!pending.empty()) complete_front("final-drain");
        require(next_frame==cycles*sequence.size() && verified==next_frame,"incomplete session frame count");
        require(window==1 || not_ready>0,"queued coverage unmet: no pending consumer observed at next enqueue");
        cleanup(true);
        summaries<<name<<'\t'<<phase<<'\t'<<window<<'\t'<<next_frame<<'\t'<<verified<<'\t'
                 <<not_ready<<'\t'<<ready_at_enqueue<<'\t'<<high_water<<"\t1\n"<<std::flush;
        require(static_cast<bool>(summaries),"summary flush failed");
    }

    void cleanup(bool checked) {
        if (closed) return;
        closed=true;
        CleanupReport r;
        int drain=stream?1:0;
        r.release(drain,[&](int){ck(cudaStreamSynchronize(stream),"checked consumer stream drain");});
        int close=processor?1:0;
        r.release(close,[&](int){processor->close();});
        processor.reset();
        for (auto& slot:slots) {
            r.release(slot.ready,[](auto p){ck(cudaEventDestroy(p),"checked consumer event release");});
            r.release(slot.images,[](auto p){ck(cudaFreeHost(p),"checked FP32 pinned release");});
            r.release(slot.rgb,[](auto p){ck(cudaFreeHost(p),"checked RGB pinned release");});
        }
        r.release(stream,[](auto p){ck(cudaStreamDestroy(p),"checked independent stream release");});
        std::cerr<<"cleanup_session="<<name<<" attempts="<<r.attempts()<<" failures="<<r.failures()<<std::endl;
        if (checked) r.throw_if_failed();
    }
};

int main(int argc,char** argv) {
    try {
        require(argc==3,"usage: usage REFS NEW_OUTPUT");
        require(std::endian::native==std::endian::little,"reference bit layout requires little endian");
        const fs::path output=argv[2];
        require(!fs::exists(output),"output must be new");
        fs::create_directories(output);
        const auto cases=load_cases(argv[1]);
        cudaFuncAttributes attributes{};
        ck(cudaFuncGetAttributes(&attributes,consumer_delay),"preload caller delay kernel before tested phases");
        std::cout<<"delay_preloaded=1 delay_cycles="<<delay_cycles<<" binary_version="<<attributes.binaryVersion<<std::endl;
        std::ofstream report(output/"report.tsv"), submissions(output/"submissions.tsv"), summaries(output/"summary.tsv");
        require(report && submissions && summaries,"create evidence output files");
        report<<"session\tphase\twindow\tframe\tcase\tlabel\twidth\theight\trgb_bytes\tf32_elements\trgb_exact\tf32_bitwise\tfinite\trepeat_exact\tcross_session_exact\tmapping\trgb_sha256\tf32_sha256\treference_rgb_sha256\treference_f32_sha256\tcompletion_reason\n";
        submissions<<"session\tphase\twindow\tframe\tcase\tprior_consumer_query\tquery_last_error\tpending_after_submit\tslot\tdelay_cycles\tcaller_jpeg_overwritten\n";
        summaries<<"session\tphase\twindow\tsubmitted\tverified\tprior_consumer_not_ready\tprior_consumer_ready\tmax_pending\tchecked_close\n";
        std::map<int,std::pair<std::string,std::string>> global_first;
        {
            Session s("serial","serial",1,cases,output,report,submissions,summaries,global_first);
            for (int cycle=0;cycle<cycles;++cycle) for (int index:sequence) s.submit(index);
            s.finish();
        }
        {
            Session s("queued","queued",3,cases,output,report,submissions,summaries,global_first);
            for (int cycle=0;cycle<cycles;++cycle) for (int index:sequence) s.submit(index);
            s.finish();
        }
        {
            Session a("dual-a","dual",3,cases,output,report,submissions,summaries,global_first);
            Session b("dual-b","dual",3,cases,output,report,submissions,summaries,global_first);
            require(a.stream!=b.stream,"dual sessions must have distinct streams");
            for (int cycle=0;cycle<cycles;++cycle) for (int index:sequence) { a.submit(index); b.submit((index+1)%7); }
            a.finish(); b.finish();
        }
        report.close(); submissions.close(); summaries.close();
        require(report && submissions && summaries,"checked evidence output close");
        std::cout<<"result=complete sessions=4 frames=208 rgb_exact=1 f32_bitwise=1 finite=1 repeat_exact=1 mapping=1 checked_close=1\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<"result=failed error="<<e.what()<<std::endl;
        return 1;
    }
}

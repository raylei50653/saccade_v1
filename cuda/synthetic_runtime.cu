#include "saccade/synthetic_runtime.hpp"
#include "saccade/cleanup.hpp"

#include <cuda_runtime.h>
#include <cufft.h>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef SACCADE_BUILD_COMMIT
#define SACCADE_BUILD_COMMIT "unknown"
#endif
#ifndef SACCADE_BUILD_DIRTY
#define SACCADE_BUILD_DIRTY "unknown"
#endif

namespace saccade {
namespace {
void ck(cudaError_t result, const char* call) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(result));
}
void fk(cufftResult result, const char* call) {
    if (result != CUFFT_SUCCESS) throw std::runtime_error(std::string(call) + ": cuFFT code " + std::to_string(result));
}
#define CUDA(call) ck((call), #call)
#define FFT(call) fk((call), #call)
// These internal structs are not wire ABIs. Only packet_kernel defines wire bytes.
struct Meta { unsigned long long epoch, seed; long long frame; };
struct Header { unsigned long long epoch; long long frame; unsigned status, count; };
struct Row {
    unsigned long long uid, output_id;
    unsigned class_id, flags;
    float xyxy[4], observed, association;
};
struct Candidates { Header header; Row rows[2]; };
struct Tracks {
    long long frame;
    unsigned long long predict_count, update_count, input_sum;
    unsigned count;
    Row rows[2];
};
struct DeviceControl {
    unsigned long long epoch;
    long long current, committed, fail_frame;
    unsigned committed_slot, working_slot, frame_status, fail_code, run_stage;
    unsigned spectrum_ready, predict_ready, update_ready;
    long long spec_frame[2], scratch_frame;
    unsigned long long fft_completed;
};
__global__ void initialize(DeviceControl* ctl, Tracks* tracks, unsigned long long epoch) {
    if (threadIdx.x || blockIdx.x) return;
    ctl->epoch = epoch;
    ctl->current = ctl->committed = ctl->fail_frame = -1;
    ctl->committed_slot = 1; ctl->working_slot = 0;
    ctl->spec_frame[0] = ctl->spec_frame[1] = ctl->scratch_frame = -1;
    tracks[0].frame = tracks[1].frame = -1;
}
__global__ void invalidate_header(Header* header) {
    if (!threadIdx.x && !blockIdx.x) header->frame = -1;
}
__global__ void delay_kernel(unsigned cycles) {
    const unsigned long long start = clock64();
    while (clock64() - start < cycles) { }
}
__global__ void make_motion(const Meta* meta, Header* header, float* image, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) image[i] = static_cast<float>(meta->seed % 13 + meta->frame + 1);
    if (i == 0) *header = Header{meta->epoch, meta->frame, 0, static_cast<unsigned>(n)};
}
__global__ void make_candidates(const Meta* meta, Candidates* cs) {
    const int i = threadIdx.x;
    if (i == 0) cs->header = Header{meta->epoch, meta->frame, 0, (meta->frame & 1) ? 2U : 0U};
    if (i < 2) {
        Row row{};
        row.uid = 2 * meta->frame + i + 1; row.output_id = 100 + row.uid;
        row.class_id = i; row.flags = 1;
        const float base = static_cast<float>(meta->seed % 7 + meta->frame + 10 * i);
        for (int j = 0; j < 4; ++j) row.xyxy[j] = base + j;
        row.observed = .5F; row.association = .25F;
        cs->rows[i] = row;
    }
}
__global__ void begin_frame(DeviceControl* ctl, unsigned long long epoch, long long frame) {
    if (threadIdx.x || blockIdx.x) return;
    ctl->current = frame; ctl->working_slot = 1 - ctl->committed_slot;
    ctl->frame_status = 0; ctl->run_stage = 0;
    ctl->spectrum_ready = ctl->predict_ready = ctl->update_ready = 0;
    if (epoch != ctl->epoch || (!ctl->fail_code && frame != ctl->committed + 1)) ctl->frame_status |= 1;
}
// Always-run, single-thread phase boundary. Computation never consults the mutable error word.
__global__ void prepare(DeviceControl* ctl, const Header* header, unsigned expected_count, bool check_count) {
    if (threadIdx.x || blockIdx.x) return;
    if (header) {
        if (header->epoch != ctl->epoch || header->frame != ctl->current) ctl->frame_status |= 1;
        ctl->frame_status |= header->status;
        if (check_count && header->count != expected_count) ctl->frame_status |= 1;
        if (!check_count && header->count > 2) ctl->frame_status |= 4;
    }
    ctl->run_stage = !ctl->fail_code && !ctl->frame_status;
}
__global__ void fft_finished(DeviceControl* ctl) {
    if (threadIdx.x || blockIdx.x) return;
    ctl->scratch_frame = ctl->current;
    ++ctl->fft_completed;
}
__global__ void inject(DeviceControl* ctl, unsigned code) {
    if (!threadIdx.x && !blockIdx.x) atomicOr(&ctl->frame_status, code);
}
__global__ void publish(DeviceControl* ctl, const cufftComplex* scratch, cufftComplex* spectra,
                        int n, unsigned injection) {
    if (!ctl->run_stage) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) spectra[ctl->working_slot * n + i] = scratch[i];
    if (i == 0) {
        ctl->spec_frame[ctl->working_slot] = ctl->current;
        ctl->spectrum_ready = 1;
        if (injection) atomicOr(&ctl->frame_status, injection);
    }
}
__global__ void predict(DeviceControl* ctl, Tracks* tracks, unsigned injection) {
    if (!ctl->run_stage) return;
    // Multiple blocks read the same frozen flag; only thread 0 writes scalar synthetic state.
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0) {
        if (!ctl->spectrum_ready || ctl->predict_ready) atomicOr(&ctl->frame_status, 1);
        tracks[ctl->working_slot] = tracks[ctl->committed_slot];
        Tracks* work = tracks + ctl->working_slot;
        work->frame = ctl->current; ++work->predict_count;
        ctl->predict_ready = 1;
        if (injection) atomicOr(&ctl->frame_status, injection);
    }
}
__global__ void compact(DeviceControl* ctl, const Candidates* cs, Candidates* db) {
    if (!ctl->run_stage) return;
    const int i = threadIdx.x;
    if (i == 0) db->header = cs->header;
    if (i < 2) db->rows[i] = cs->rows[i];
}
__global__ void update(DeviceControl* ctl, Tracks* tracks, const Candidates* db,
                       const cufftComplex* spectra, int n, unsigned injection) {
    if (!ctl->run_stage) return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    Tracks* work = tracks + ctl->working_slot;
    if (i == 0) {
        if (!ctl->predict_ready || ctl->update_ready || ctl->spec_frame[ctl->working_slot] != ctl->current)
            atomicOr(&ctl->frame_status, 1);
        ++work->update_count; work->count = db->header.count;
        work->input_sum += static_cast<unsigned long long>(spectra[ctl->working_slot * n].x);
        ctl->update_ready = 1;
        if (injection) atomicOr(&ctl->frame_status, injection);
    }
    if (i < 2) {
        Row row = db->rows[i];
        for (int j = 0; j < 4; ++j) row.xyxy[j] += static_cast<float>(work->predict_count);
        work->rows[i] = row;
    }
}
__global__ void commit(DeviceControl* ctl) {
    if (threadIdx.x || blockIdx.x) return;
    if (!ctl->fail_code) {
        if (!ctl->frame_status && (!ctl->spectrum_ready || !ctl->predict_ready || !ctl->update_ready))
            ctl->frame_status |= 1;
        if (ctl->frame_status) { ctl->fail_frame = ctl->current; ctl->fail_code = ctl->frame_status; }
        else { ctl->committed = ctl->current; ctl->committed_slot = ctl->working_slot; }
    }
    ctl->run_stage = 0;
}
__device__ void put(unsigned char* p, unsigned long long v, int n) {
    for (int i = 0; i < n; ++i) p[i] = static_cast<unsigned char>(v >> (8 * i));
}
__device__ void put_float(unsigned char* p, float v) { put(p, v == 0.F ? 0U : __float_as_uint(v), 4); }
__global__ void packet_kernel(const DeviceControl* ctl, const Tracks* tracks, unsigned char* out) {
    // One block: the clear and field writes are separated by a real block barrier.
    for (int i = threadIdx.x; i < static_cast<int>(packet_bytes); i += blockDim.x) out[i] = 0;
    __syncthreads();
    if (threadIdx.x) return;
    out[0]='S'; out[1]='C'; out[2]='R'; out[3]='P';
    put(out+4,1,2); put(out+6,80,2); put(out+8,packet_bytes,4); put(out+12,256,4);
    put(out+16,ctl->epoch,8); put(out+24,ctl->current,8); put(out+32,ctl->committed,8);
    put(out+40,ctl->fail_frame,8);
    const unsigned status = !ctl->fail_code ? 0U : (ctl->current == ctl->fail_frame ? 1U : 2U);
    put(out+48,status,4);
    put(out+56,status == 1 ? ctl->fail_code : 0,4); put(out+60,ctl->fail_code,4);
    if (status) return;
    const Tracks& ts = tracks[ctl->committed_slot];
    put(out+52,ts.count,4);
    for (unsigned i = 0; i < ts.count; ++i) {
        const Row& row = ts.rows[i]; unsigned char* dest = out + 80 + 48*i;
        put(dest,row.uid,8); put(dest+8,row.output_id,8); put(dest+16,row.class_id,4); put(dest+20,row.flags,4);
        for (int j=0;j<4;++j) put_float(dest+24+4*j,row.xyxy[j]);
        put_float(dest+40,row.observed); put_float(dest+44,row.association);
    }
}

// Every event's current generation is recorded on successful API return only.
struct Event { cudaEvent_t handle{}; long long generation = -1; };
struct Lease { Event ready, consumed; };
class Session {
public:
    explicit Session(const SyntheticConfig& config) : cfg(config) {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session() noexcept {
        // On a recoverable host/API exception, finish both streams before any free/destroy.
        // Fatal context errors cannot yield a completed run; checked normal drain throws.
        if (!drained) {
            if (d) cudaStreamSynchronize(d);
            if (m) cudaStreamSynchronize(m);
        }
        release_resources(); // No throw; only handles never attempted by close().
    }
    void release_resources() noexcept {
        cleanup.release(plan, [](auto handle) { FFT(cufftDestroy(handle)); });
        for (auto& p : device_allocations) cleanup.release(p, [](auto handle) { CUDA(cudaFree(handle)); });
        for (auto& p : host_allocations) cleanup.release(p, [](auto handle) { CUDA(cudaFreeHost(handle)); });
        for (auto& e : events) cleanup.release(e, [](auto handle) { CUDA(cudaEventDestroy(handle)); });
        cleanup.release(d, [](auto handle) { CUDA(cudaStreamDestroy(handle)); });
        cleanup.release(m, [](auto handle) { CUDA(cudaStreamDestroy(handle)); });
    }
    void close() {
        if (!drained) throw std::logic_error("close requires both terminal drains and completed diagnostics");
        if (closed) return;
        release_resources();
        cleanup.throw_if_failed(); // Never return a successful run on a release error.
        result.ledger.push_back("finalize resources_released");
        closed=true;
    }
    template <typename T> T* allocate(std::size_t count) {
        void* p{}; CUDA(cudaMalloc(&p, count * sizeof(T)));
        try { device_allocations.push_back(p); }
        catch (...) { cudaFree(p); throw; } // Acquired but not registered/used yet.
        CUDA(cudaMemsetAsync(p, 0, count * sizeof(T), m));
        return static_cast<T*>(p);
    }
    template <typename T> T* pinned(std::size_t count) {
        void* p{}; CUDA(cudaMallocHost(&p, count * sizeof(T)));
        try { host_allocations.push_back(p); }
        catch (...) { cudaFreeHost(p); throw; }
        std::memset(p, 0, count * sizeof(T)); return static_cast<T*>(p);
    }
    Event event() {
        Event e; CUDA(cudaEventCreateWithFlags(&e.handle,cudaEventDisableTiming));
        try { events.push_back(e.handle); }
        catch (...) { cudaEventDestroy(e.handle); throw; }
        return e;
    }
    void log(const char* op, const std::string& resource, int slot, long long gen) {
        result.ledger.push_back(std::string(op)+" "+resource+" slot="+std::to_string(slot)+" gen="+std::to_string(gen));
    }
    void record(Event& e, cudaStream_t s, long long gen, const std::string& name, int slot) {
        CUDA(cudaEventRecord(e.handle,s)); e.generation=gen; log("record",name,slot,gen);
    }
    void wait(Event& e, cudaStream_t s, long long expected, const std::string& name, int slot) {
        if (e.generation != expected) throw std::logic_error("event generation mismatch: "+name);
        CUDA(cudaStreamWaitEvent(s,e.handle,0)); log("wait",name,slot,expected);
    }
    void host_wait(Event& e, long long expected, const std::string& name, int slot) {
        if (e.generation != expected || expected < 0) throw std::logic_error("host event generation mismatch: "+name);
        CUDA(cudaEventSynchronize(e.handle)); log("host_wait",name,slot,expected);
    }
    void boundary(cudaStream_t s) { CUDA(cudaGetLastError()); if (cfg.mode == "serial") CUDA(cudaStreamSynchronize(s)); }
    unsigned delay(long long frame, unsigned stage) const {
        // Fixed integer mixer, independent of behavioral seed; clock counts are scheduling only.
        unsigned long long x = cfg.delay_seed ^ (static_cast<unsigned long long>(frame+1)*0x9e3779b97f4a7c15ULL) ^ stage;
        x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL; x ^= x >> 27;
        return cfg.delay_cycles ? static_cast<unsigned>(x % (static_cast<unsigned long long>(cfg.delay_cycles)+1)) : 0;
    }
    void perturb(cudaStream_t s, long long frame, unsigned stage) {
        if (cfg.delay_cycles) { delay_kernel<<<1,1,0,s>>>(delay(frame,stage)); boundary(s); }
    }
    void init() {
        CUDA(cudaSetDevice(0));
        CUDA(cudaStreamCreateWithFlags(&d,cudaStreamNonBlocking));
        CUDA(cudaStreamCreateWithFlags(&m,cudaStreamNonBlocking));
        n = cfg.width*cfg.height; nf = cfg.height*(cfg.width/2+1);
        ctl = allocate<DeviceControl>(1); tracks=allocate<Tracks>(2);
        spectra=allocate<cufftComplex>(2*nf); scratch=allocate<cufftComplex>(nf);
        desc=allocate<Meta>(1); db=allocate<Candidates>(1); rp=allocate<unsigned char>(packet_bytes);
        stage=pinned<Meta>(cfg.n_h); hp=pinned<unsigned char>(cfg.n_rp*packet_bytes);
        // Terminal diagnostics have separate pinned allocations, never used by normal frame traffic.
        if (cfg.terminal_snapshot) {
            snap_ctl=pinned<DeviceControl>(1); snap_tracks=pinned<Tracks>(2);
            snap_spec=pinned<cufftComplex>(2*nf);
        }
        for(int i=0;i<cfg.n_h;++i) staged.push_back(event());
        for(int i=0;i<cfg.n_rp;++i) pkt.push_back(event());
        for(int i=0;i<cfg.n_mi;++i) {
            mi.push_back({event(),event()}); images.push_back(allocate<float>(n)); headers.push_back(allocate<Header>(1));
        }
        for(int i=0;i<cfg.n_cs;++i) { cs.push_back({event(),event()}); candidates.push_back(allocate<Candidates>(1)); }
        d_end=event(); m_end=event();
        FFT(cufftCreate(&plan)); FFT(cufftSetAutoAllocation(plan,0));
        std::size_t workspace_bytes=0;
        FFT(cufftMakePlan2d(plan,cfg.height,cfg.width,CUFFT_R2C,&workspace_bytes));
        workspace=allocate<unsigned char>(std::max<std::size_t>(workspace_bytes,1));
        FFT(cufftSetWorkArea(plan,workspace)); FFT(cufftSetStream(plan,m));
        initialize<<<1,1,0,m>>>(ctl,tracks,cfg.epoch); CUDA(cudaGetLastError());
        for (auto* header : headers) invalidate_header<<<1,1,0,m>>>(header);
        for (auto* candidate : candidates) invalidate_header<<<1,1,0,m>>>(&candidate->header);
        CUDA(cudaGetLastError());
        // Warm the exact plan during initialization. This also pays lazy module/JIT costs before A06.
        FFT(cufftExecR2C(plan,images[0],scratch));
        CUDA(cudaMemsetAsync(scratch,0,nf*sizeof(cufftComplex),m));
        CUDA(cudaDeviceSynchronize());
        result.ledger.push_back("initialize device_synchronized");
        std::ostringstream id;
        cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop,0));
        int driver=0,runtime=0,cufft_version=0;
        CUDA(cudaDriverGetVersion(&driver)); CUDA(cudaRuntimeGetVersion(&runtime)); FFT(cufftGetVersion(&cufft_version));
        id << "build_commit=" << SACCADE_BUILD_COMMIT << "\nbuild_dirty=" << SACCADE_BUILD_DIRTY
           << "\nhost_compiler=" << __VERSION__ << "\nnvcc=" << __CUDACC_VER_MAJOR__ << '.' << __CUDACC_VER_MINOR__ << '.' << __CUDACC_VER_BUILD__
           << "\ncuda_runtime=" << runtime << "\ncuda_driver_api=" << driver << "\ncufft=" << cufft_version
           << "\ngpu=" << prop.name << "\ncompute_capability=" << prop.major << '.' << prop.minor
           << "\npci=" << prop.pciDomainID << ':' << prop.pciBusID << ':' << prop.pciDeviceID << "\nuuid=";
        for (unsigned char b : prop.uuid.bytes) id << std::hex << std::setw(2) << std::setfill('0') << unsigned(b);
        id << std::dec << "\ncufft_workspace_bytes=" << workspace_bytes << "\n";
        result.identity=id.str();
        pending.assign(cfg.n_rp,-1);
        result.packets.reserve(static_cast<std::size_t>(cfg.frames)*packet_bytes);
        result.ledger.reserve(static_cast<std::size_t>(cfg.frames)*16+4);
    }
    bool consume(int r) {
        if (pending[r] < 0) return false;
        host_wait(pkt[r],pending[r],"pkt",r);
        const auto* bytes=hp+static_cast<std::size_t>(r)*packet_bytes;
        result.packets.insert(result.packets.end(),bytes,bytes+packet_bytes);
        // Observe only the wire status for stop policy. CLI/tests first persist the
        // original stream, then run strict decode_packet, retaining even malformed
        // output as evidence rather than losing it in a runtime exception.
        unsigned status=0;
        for (int i=0;i<4;++i) status |= static_cast<unsigned>(bytes[48+i]) << (8*i);
        log("consume","pkt",r,pending[r]); pending[r]=-1;
        return status == static_cast<unsigned>(ResultStatus::failed);
    }
    bool cut(int k,const char* name) {
        if (cfg.cancel_frame==k && cfg.cancel_cut==name) { result.cancelled=true; result.ledger.push_back(std::string("cancel frame=")+std::to_string(k)+" cut="+name); return true; }
        return false;
    }
    unsigned injection(int k,const char* stage_name,unsigned code) const {
        if (cfg.fail_stage != stage_name || k < cfg.fail_frame || cfg.fail_frame < 0) return 0;
        return k==cfg.fail_frame ? code : (2U|8U); // Secondary diagnostics cannot replace first failure.
    }
    void execute() {
        init();
        for(int k=0;k<cfg.frames;++k) {
            const int h=k%cfg.n_h, r=k%cfg.n_rp, a=k%cfg.n_mi, b=k%cfg.n_cs;
            // The two host leases are independent; neither substitutes for device slot consumed waits.
            if(k>=cfg.n_h) host_wait(staged[h],k-cfg.n_h,"staged",h);
            if(consume(r) && !cfg.fixed_submit) { result.stopped_on_failure=true; break; }
            stage[h]=Meta{cfg.epoch,cfg.seed,k};
            CUDA(cudaMemcpyAsync(desc,stage+h,sizeof(Meta),cudaMemcpyHostToDevice,d));
            record(staged[h],d,k,"staged",h); ++result.d_frames;
            boundary(d); if(cut(k,"after_h2d")) break;
            if(k>=cfg.n_mi) wait(mi[a].consumed,d,k-cfg.n_mi,"mi_consumed",a);
            perturb(d,k,1);
            make_motion<<<(n+127)/128,128,0,d>>>(desc,headers[a],images[a],n); boundary(d);
            record(mi[a].ready,d,k,"mi_ready",a); if(cut(k,"after_mi")) break;
            if(k>=cfg.n_cs) wait(cs[b].consumed,d,k-cfg.n_cs,"cs_consumed",b);
            perturb(d,k,2);
            make_candidates<<<1,32,0,d>>>(desc,candidates[b]); boundary(d);
            record(cs[b].ready,d,k,"cs_ready",b); if(cut(k,"after_cs")) break;
            begin_frame<<<1,1,0,m>>>(ctl,cfg.epoch,k); boundary(m); ++result.m_frames;
            if(cut(k,"after_begin")) break;
            if (cfg.fail_frame >= 0 && k > cfg.fail_frame) {
                inject<<<1,1,0,m>>>(ctl,2U|8U); boundary(m);
                result.ledger.push_back("secondary_error frame="+std::to_string(k)+" bits=10");
            }
            wait(mi[a].ready,m,k,"mi_ready",a);
            prepare<<<1,1,0,m>>>(ctl,headers[a],n,true); boundary(m);
            perturb(m,k,3);
            FFT(cufftExecR2C(plan,images[a],scratch)); boundary(m);
            fft_finished<<<1,1,0,m>>>(ctl); boundary(m);
            if(cut(k,"after_fft")) break;
            inject<<<1,1,0,m>>>(ctl,injection(k,"before_spectrum",8)); boundary(m);
            prepare<<<1,1,0,m>>>(ctl,headers[a],n,true); boundary(m);
            publish<<<(nf+127)/128,128,0,m>>>(ctl,scratch,spectra,nf,injection(k,"after_spectrum",4)); boundary(m);
            // Diagnostic header access is before the final-reader consumed event.
            prepare<<<1,1,0,m>>>(ctl,headers[a],n,true); boundary(m);
            record(mi[a].consumed,m,k,"mi_consumed",a);
            perturb(m,k,4);
            predict<<<2,128,0,m>>>(ctl,tracks,injection(k,"after_predict",16)); boundary(m);
            if(cut(k,"after_predict")) break;
            wait(cs[b].ready,m,k,"cs_ready",b);
            prepare<<<1,1,0,m>>>(ctl,&candidates[b]->header,0,false); boundary(m);
            compact<<<1,32,0,m>>>(ctl,candidates[b],db); boundary(m);
            prepare<<<1,1,0,m>>>(ctl,&candidates[b]->header,0,false); boundary(m);
            record(cs[b].consumed,m,k,"cs_consumed",b);
            perturb(m,k,5);
            update<<<2,128,0,m>>>(ctl,tracks,db,spectra,nf,injection(k,"after_update",1)); boundary(m);
            if(cut(k,"after_update")) break;
            commit<<<1,1,0,m>>>(ctl); boundary(m);
            packet_kernel<<<1,256,0,m>>>(ctl,tracks,rp); boundary(m);
            CUDA(cudaMemcpyAsync(hp+static_cast<std::size_t>(r)*packet_bytes,rp,packet_bytes,cudaMemcpyDeviceToHost,m));
            record(pkt[r],m,k,"pkt",r); pending[r]=k; ++result.packet_copies;
            boundary(m); if(cut(k,"after_packet")) break;
        }
        // Consume all queued packets in their frame order, including the complete fixed failure tail.
        while(true) {
            const auto pos=std::min_element(pending.begin(),pending.end(),[](auto a,auto b){return a>=0 && (b<0 || a<b);});
            if(pos==pending.end() || *pos<0) break;
            if(consume(static_cast<int>(pos-pending.begin())) && !cfg.fixed_submit) result.stopped_on_failure=true;
        }
        record(d_end,d,result.d_frames,"d_terminal",0);
        record(m_end,m,result.m_frames,"m_terminal",0);
        host_wait(d_end,result.d_frames,"d_terminal",0); result.d_drained=true;
        host_wait(m_end,result.m_frames,"m_terminal",0); result.m_drained=true; drained=true;
        if(cfg.terminal_snapshot) {
            drained=false;
            snapshot();
            drained=true;
        }
    }
    void snapshot() {
        // Explicit test-only observation after both terminal drains; no normal-frame D2H additions.
        CUDA(cudaMemcpyAsync(snap_ctl,ctl,sizeof(DeviceControl),cudaMemcpyDeviceToHost,m));
        CUDA(cudaMemcpyAsync(snap_tracks,tracks,2*sizeof(Tracks),cudaMemcpyDeviceToHost,m));
        CUDA(cudaMemcpyAsync(snap_spec,spectra,2*nf*sizeof(cufftComplex),cudaMemcpyDeviceToHost,m));
        CUDA(cudaStreamSynchronize(m));
        auto& out=result.snapshot; const auto& c=*snap_ctl; const auto& t=snap_tracks[c.committed_slot];
        out.current_frame=c.current; out.committed_frame=c.committed; out.fail_frame=c.fail_frame;
        out.fail_code=c.fail_code; out.committed_slot=c.committed_slot;
        out.tracks_frame=t.frame; out.spectrum_frame=c.spec_frame[c.committed_slot];
        out.predict_count=t.predict_count; out.update_count=t.update_count; out.input_sum=t.input_sum;
        out.scratch_frame=c.scratch_frame; out.fft_completed=c.fft_completed;
        out.spectrum_bits.resize(2*nf);
        std::memcpy(out.spectrum_bits.data(),snap_spec+c.committed_slot*nf,nf*sizeof(cufftComplex));
        out.tracks_bytes.resize(sizeof(Tracks));
        std::memcpy(out.tracks_bytes.data(),&t,sizeof(Tracks));
        result.ledger.push_back("terminal_snapshot after_both_drains");
    }
    const SyntheticConfig& cfg;
    SyntheticRun result;
    cudaStream_t d{},m{}; cufftHandle plan{}; bool drained=false;
    CleanupReport cleanup;
    bool closed=false;
    std::vector<void*> device_allocations,host_allocations;
    std::vector<cudaEvent_t> events;
    std::vector<Event> staged,pkt; std::vector<Lease> mi,cs;
    std::vector<float*> images; std::vector<Header*> headers; std::vector<Candidates*> candidates;
    std::vector<long long> pending;
    Event d_end,m_end; int n=0,nf=0;
    DeviceControl *ctl{},*snap_ctl{}; Tracks *tracks{},*snap_tracks{};
    cufftComplex *spectra{},*scratch{},*snap_spec{};
    Meta *stage{},*desc{}; Candidates* db{}; unsigned char *hp{},*rp{},*workspace{};
};
} // namespace
SyntheticRun run_synthetic(const SyntheticConfig& config) {
    validate_config(config); // Must complete before any CUDA session resource exists.
    Session session(config);
    session.execute();
    session.close();
    return std::move(session.result);
}
} // namespace saccade

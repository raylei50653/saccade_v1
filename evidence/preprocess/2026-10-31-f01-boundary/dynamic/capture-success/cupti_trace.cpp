// Standalone diagnostic preload, not Saccade runtime. No CUDA call in callbacks.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <nvjpeg.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <unistd.h>
#include <sys/syscall.h>

static FILE* output;
static CUpti_SubscriberHandle subscriber;
static bool subscribed;
static unsigned long long sequence, decode_number, launch_number;
static nvjpegImage_t last_source, last_destination;
static int last_width, last_height;
static CUstream last_stream;

static unsigned long long stamp() {
    timespec ts{}; clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<unsigned long long>(ts.tv_sec)*1000000000ULL+ts.tv_nsec;
}
static void prefix(const char* kind) {
    std::fprintf(output,"{\"kind\":\"%s\",\"seq\":%llu,\"monotonic_ns\":%llu,\"tid\":%ld",
        kind,++sequence,stamp(),syscall(SYS_gettid));
}
static void pointer(const char* key, const void* p) {
    std::fprintf(output,",\"%s\":\"0x%llx\"",key,
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(p)));
}
static void frames() {
    void* addresses[24]; const int count=backtrace(addresses,24);
    std::fprintf(output,",\"frames\":[");
    for(int i=0;i<count;++i) {
        Dl_info info{}; dladdr(addresses[i],&info);
        std::fprintf(output,"%s{\"module\":\"%s\",\"offset\":\"0x%llx\"}",i?",":"",
            info.dli_fname?info.dli_fname:"?",
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(addresses[i])-
                reinterpret_cast<uintptr_t>(info.dli_fbase)));
    }
    std::fprintf(output,"]");
}
static void image(const char* key,const nvjpegImage_t& value) {
    std::fprintf(output,",\"%s\":{\"channels\":[",key);
    for(int i=0;i<4;++i) std::fprintf(output,"%s\"0x%llx\"",i?",":"",
        static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(value.channel[i])));
    std::fprintf(output,"],\"pitches\":[%zu,%zu,%zu,%zu]}",
        value.pitch[0],value.pitch[1],value.pitch[2],value.pitch[3]);
}
static void kernel(const char* symbol,void** args,CUstream stream) {
    if(!symbol || !args || !std::strstr(symbol,"8rgba2rgbIL20nvjpegOutputFormat_t3")) return;
    std::memcpy(&last_source,args[0],sizeof(last_source));
    int dimensions[2]; std::memcpy(dimensions,args[1],sizeof(dimensions));
    last_width=dimensions[0]; last_height=dimensions[1];
    std::memcpy(&last_destination,args[2],sizeof(last_destination));
    last_stream=stream; ++launch_number;
    std::fprintf(output,",\"converter_launch\":%llu,\"width\":%d,\"height\":%d",
        launch_number,last_width,last_height);
    image("source",last_source); image("destination",last_destination);
}
static void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* raw) {
    if(!output || (domain!=CUPTI_CB_DOMAIN_DRIVER_API && domain!=CUPTI_CB_DOMAIN_RUNTIME_API)) return;
    const auto* data=static_cast<const CUpti_CallbackData*>(raw);
    const char* fn=data->functionName?data->functionName:"?";
    const bool enter=data->callbackSite==CUPTI_API_ENTER;
    flockfile(output); prefix("api");
    std::fprintf(output,",\"domain\":\"%s\",\"cbid\":%u,\"site\":\"%s\",\"api\":\"%s\",\"correlation\":%u",
        domain==CUPTI_CB_DOMAIN_DRIVER_API?"driver":"runtime",id,enter?"enter":"exit",fn,data->correlationId);
    if(!enter && data->functionReturnValue) std::fprintf(output,",\"status\":%d",*static_cast<const int*>(data->functionReturnValue));
    if(data->symbolName) std::fprintf(output,",\"symbol\":\"%s\"",data->symbolName);
    if(domain==CUPTI_CB_DOMAIN_DRIVER_API) {
        if(!std::strcmp(fn,"cuLaunchKernel") || !std::strcmp(fn,"cuLaunchKernel_ptsz")) {
            const auto* p=static_cast<const cuLaunchKernel_params*>(data->functionParams);
            pointer("stream",p->hStream);
            std::fprintf(output,",\"kernel_params_present\":%s,\"extra_present\":%s",p->kernelParams?"true":"false",p->extra?"true":"false");
            if(enter) kernel(data->symbolName,p->kernelParams,p->hStream);
        } else if(!std::strcmp(fn,"cuLaunchKernelEx") || !std::strcmp(fn,"cuLaunchKernelEx_ptsz")) {
            const auto* p=static_cast<const cuLaunchKernelEx_params*>(data->functionParams);
            pointer("stream",p->config->hStream);
            std::fprintf(output,",\"kernel_params_present\":%s,\"extra_present\":%s",p->kernelParams?"true":"false",p->extra?"true":"false");
            if(enter) kernel(data->symbolName,p->kernelParams,p->config->hStream);
        } else if(!std::strcmp(fn,"cuEventRecord") || !std::strcmp(fn,"cuEventRecord_ptsz")) {
            const auto* p=static_cast<const cuEventRecord_params*>(data->functionParams);
            pointer("event",p->hEvent);pointer("stream",p->hStream);
        } else if(!std::strcmp(fn,"cuStreamWaitEvent") || !std::strcmp(fn,"cuStreamWaitEvent_ptsz")) {
            const auto* p=static_cast<const cuStreamWaitEvent_params*>(data->functionParams);
            pointer("event",p->hEvent);pointer("stream",p->hStream);
        } else if(!std::strcmp(fn,"cuMemAlloc_v2")) {
            const auto* p=static_cast<const cuMemAlloc_v2_params*>(data->functionParams);
            std::fprintf(output,",\"bytes\":%zu",p->bytesize);
            if(!enter && data->functionReturnValue && *static_cast<const int*>(data->functionReturnValue)==0)
                pointer("allocation",reinterpret_cast<void*>(*p->dptr));
        } else if(!std::strcmp(fn,"cuMemFree_v2")) {
            const auto* p=static_cast<const cuMemFree_v2_params*>(data->functionParams);
            pointer("allocation",reinterpret_cast<void*>(p->dptr));
        }
    } else {
        if(!std::strcmp(fn,"cudaLaunchKernel") || !std::strcmp(fn,"cudaLaunchKernel_ptsz")) {
            const auto* p=static_cast<const cudaLaunchKernel_v7000_params*>(data->functionParams);
            pointer("stream",p->stream);
            // Driver callbacks are the authoritative argument snapshot.
        } else if(!std::strcmp(fn,"cudaEventRecord")) {
            const auto* p=static_cast<const cudaEventRecord_v3020_params*>(data->functionParams);
            pointer("event",p->event);pointer("stream",p->stream);
        } else if(!std::strcmp(fn,"cudaStreamWaitEvent")) {
            const auto* p=static_cast<const cudaStreamWaitEvent_v3020_params*>(data->functionParams);
            pointer("event",p->event);pointer("stream",p->stream);
        }
    }
    if(enter && domain==CUPTI_CB_DOMAIN_DRIVER_API &&
        (std::strstr(fn,"LaunchKernel") || std::strstr(fn,"StreamWait") || std::strstr(fn,"EventRecord"))) frames();
    std::fprintf(output,"}\n");funlockfile(output);
}
static void fail(CUptiResult result,const char* op) {
    if(result==CUPTI_SUCCESS) return;
    const char* message=nullptr;cuptiGetResultString(result,&message);
    std::fprintf(stderr,"f01_cupti %s failed=%d %s\n",op,static_cast<int>(result),message?message:"?");
    _exit(90);
}
__attribute__((constructor)) static void initialize() {
    const char* path=std::getenv("F01_TRACE_LOG"); if(!path) return;
    output=std::fopen(path,"wx");if(!output) {_exit(91);}
    setvbuf(output,nullptr,_IOLBF,0);
    CUpti_SubscriberParams params{};params.structSize=CUpti_SubscriberParams_STRUCT_SIZE;
    params.subscriberName="F01 boundary diagnostic";
    fail(cuptiSubscribe_v2(&subscriber,callback,nullptr,&params),"subscribe"); subscribed=true;
    fail(cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_DRIVER_API),"enable driver");
    fail(cuptiEnableDomain(1,subscriber,CUPTI_CB_DOMAIN_RUNTIME_API),"enable runtime");
    prefix("tracer_ready");std::fprintf(output,",\"cuda_calls_inside_callback\":false}\n");
}
__attribute__((destructor)) static void finalize() {
    if(subscribed) cuptiUnsubscribe(subscriber);
    if(output) {prefix("tracer_end");std::fprintf(output,"}\n");std::fclose(output);output=nullptr;}
}
extern "C" nvjpegStatus_t nvjpegDecodeBatched(nvjpegHandle_t handle,nvjpegJpegState_t state,
    const unsigned char* const* data,const size_t* lengths,nvjpegImage_t* destinations,cudaStream_t stream) {
    using Fn=nvjpegStatus_t(*)(nvjpegHandle_t,nvjpegJpegState_t,const unsigned char* const*,const size_t*,nvjpegImage_t*,cudaStream_t);
    static Fn next=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"nvjpegDecodeBatched"));
    if(!next) _exit(92);
    const unsigned long long number=++decode_number, before=launch_number;
    if(output) {flockfile(output);prefix("decode_enter");std::fprintf(output,",\"decode\":%llu",number);pointer("caller_stream",stream);std::fprintf(output,"}\n");funlockfile(output);}
    const auto status=next(handle,state,data,lengths,destinations,stream);
    if(output) {
        flockfile(output);prefix("decode_exit");std::fprintf(output,",\"decode\":%llu,\"status\":%d,\"converter_snapshots\":%llu}\n",number,status,launch_number-before);funlockfile(output);
        if(status==NVJPEG_STATUS_SUCCESS && launch_number>before) {
            // All CUDA driver queries are OUTSIDE CUPTI callbacks, after public API returns.
            CUdeviceptr base=0;size_t size=0;unsigned long long buffer_id=0;
            const auto address=reinterpret_cast<CUdeviceptr>(last_source.channel[0]);
            const auto range=cuMemGetAddressRange(&base,&size,address);
            const auto identity=cuPointerGetAttribute(&buffer_id,CU_POINTER_ATTRIBUTE_BUFFER_ID,address);
            flockfile(output);prefix("surface_identity");
            std::fprintf(output,",\"decode\":%llu,\"range_status\":%d,\"id_status\":%d,\"buffer_id\":%llu,\"allocation_bytes\":%zu",number,range,identity,buffer_id,size);
            pointer("base",reinterpret_cast<void*>(base));pointer("converter_stream",last_stream);image("source",last_source);std::fprintf(output,"}\n");funlockfile(output);
        }
        if(number==1) {
            const char* path=std::getenv("F01_MAPS_OUT");
            if(path) {FILE* in=std::fopen("/proc/self/maps","r");FILE* out=std::fopen(path,"wx");
                if(!in||!out) _exit(93);char bytes[4096];size_t count;
                while((count=std::fread(bytes,1,sizeof(bytes),in)))std::fwrite(bytes,1,count,out);
                std::fclose(in);std::fclose(out);}
        }
    }
    return status;
}

// Research model-only benchmark. No JPEG I/O, preprocessing, tracker or output copy
// is included in the measured regions. Raw6 and original end-to-end work scopes
// are explicitly different and are labelled in the output.
#include "saccade/file_io.hpp"
#include "saccade/scan_trt_plugin.hpp"
#include "saccade/selective_scan_cuda.hpp"
#include "saccade/sha256.hpp"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace nvinfer1;
using Clock=std::chrono::steady_clock;
constexpr std::size_t frame_bytes=3*768*1280*sizeof(float);
constexpr const char* usage="model_compare ENGINE FOUR_IMAGES.f32 NEW_OUT_DIR [--layout raw6|default] [--iterations 40] [--warmup 12] [--rounds 1] [--queue 64] [--verify-only]";
void need(bool value,const std::string& msg){if(!value)throw std::runtime_error(msg);}
void cc(cudaError_t x,const char* s){if(x!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(x));}
#define CUDA(x) cc((x),#x)
unsigned number(const char* text){unsigned n{};const std::string s=text;auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),n);need(e==std::errc{}&&p==s.data()+s.size(),"invalid integer");return n;}
struct Logger:ILogger{unsigned errors=0;void log(Severity s,const char* m)noexcept override{if(s<=Severity::kERROR)++errors;if(s<=Severity::kWARNING)std::cerr<<m<<'\n';}};
struct Resources{
 cudaStream_t stream{};cudaEvent_t begin{},end{};cudaGraph_t graph{};cudaGraphExec_t executable{};std::vector<void*> buffers;
 ~Resources(){if(stream)cudaStreamSynchronize(stream);if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);for(auto p:buffers)cudaFree(p);if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);if(stream)cudaStreamDestroy(stream);}
 void* alloc(std::size_t n){void* p{};CUDA(cudaMalloc(&p,n));buffers.push_back(p);return p;}
};
std::size_t count(Dims d){std::size_t n=1;need(d.nbDims>0,"scalar or dynamic IO is unsupported");for(int k=0;k<d.nbDims;++k){need(d.d[k]>0,"dynamic IO is unsupported");n*=std::size_t(d.d[k]);}return n;}
bool shape(Dims d,std::initializer_list<std::int64_t> dims){if(d.nbDims!=int(dims.size()))return false;unsigned i=0;for(auto x:dims)if(d.d[i++]!=x)return false;return true;}
std::string ds(Dims d){std::ostringstream s;s<<'[';for(int i=0;i<d.nbDims;++i)s<<(i?",":"")<<d.d[i];return s.str()+']';}
double mean(const std::vector<double>& x){need(!x.empty(),"empty statistic");return std::accumulate(x.begin(),x.end(),0.0)/x.size();}
double pct(std::vector<double> x,double q){need(!x.empty(),"empty percentile");std::sort(x.begin(),x.end());auto n=std::size_t(std::ceil(q*x.size()));return x[std::max(std::size_t(1),n)-1];}
struct Tensor{std::string name;std::size_t elements;void* p;Dims dims;};
struct Snapshot{std::string payload,statuses;};
void equal(const Snapshot& a,const Snapshot& b){need(a.payload==b.payload&&a.statuses==b.statuses,"ordinary/repeat/Graph output bits differ");}
}

int main(int argc,char** argv){
 std::filesystem::path out;
 try{
  if(argc==2&&std::string(argv[1])=="--help"){std::cout<<usage<<'\n';return 0;}
  need(argc>=4,usage);unsigned iterations=40,warmup=12,rounds=1,queue=64;bool verify_only=false;std::string layout="raw6";
  for(int i=4;i<argc;++i){std::string k=argv[i];if(k=="--verify-only"){verify_only=true;continue;}need(i+1<argc,usage);auto v=argv[++i];if(k=="--layout")layout=v;else if(k=="--iterations")iterations=number(v);else if(k=="--warmup")warmup=number(v);else if(k=="--rounds")rounds=number(v);else if(k=="--queue")queue=number(v);else throw std::runtime_error("unknown option "+k);}
  need(layout=="raw6"||layout=="default","unknown layout");need(iterations>0&&iterations<=10000&&warmup>0&&warmup<=1000&&rounds>0&&rounds<=10&&queue>0&&queue<=1000,"invalid measurement bounds");
  out=argv[3];need(!std::filesystem::exists(out)&&std::filesystem::create_directory(out),"new output directory required");
  const auto plan=saccade::read_text_file(argv[1]),images=saccade::read_text_file(argv[2]);need(images.size()==4*frame_bytes,"exactly four fixed HR FP32 frames required");
  CUDA(cudaSetDevice(0));CUDA(cudaFree(nullptr));std::size_t free_before_runtime=0,total_device=0,free_after_engine=0,free_after_context=0,free_after_io=0,free_after_warmup=0,free_after_measure=0;CUDA(cudaMemGetInfo(&free_before_runtime,&total_device));Logger logger;saccade::register_scan_trt_plugins(*getPluginRegistry());
  auto init=Clock::now();std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));need(bool(runtime),"runtime failed");std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(plan.data(),plan.size()));need(bool(engine)&&engine->getNbAuxStreams()==0,"engine failed or aux streams enabled");CUDA(cudaMemGetInfo(&free_after_engine,&total_device));const auto context_device_bytes=engine->getDeviceMemorySizeV2();std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());need(bool(context),"context failed");CUDA(cudaMemGetInfo(&free_after_context,&total_device));
  context->setPersistentCacheLimit(0);need(context->getPersistentCacheLimit()==0,"persistent cache is not zero");need(context->setNvtxVerbosity(ProfilingVerbosity::kNONE),"cannot disable NVTX");
  Resources r;CUDA(cudaStreamCreateWithFlags(&r.stream,cudaStreamNonBlocking));CUDA(cudaEventCreate(&r.begin));CUDA(cudaEventCreate(&r.end));
  std::vector<Tensor> outputs,statuses;Tensor input;std::map<std::string,Dims> floats;unsigned inputs=0;
  for(int i=0;i<engine->getNbIOTensors();++i){const std::string name=engine->getIOTensorName(i);auto mode=engine->getTensorIOMode(name.c_str());auto type=engine->getTensorDataType(name.c_str());auto d=engine->getTensorShape(name.c_str());need(engine->getTensorLocation(name.c_str())==TensorLocation::kDEVICE&&engine->getTensorFormat(name.c_str())==TensorFormat::kLINEAR,"nonlinear/nondevice tensor "+name);auto n=count(d);need(type==DataType::kFLOAT||type==DataType::kINT32,"unsupported tensor dtype "+name);Tensor t{name,n,r.alloc(n*4),d};need(context->setTensorAddress(name.c_str(),t.p),"cannot bind "+name);
   if(mode==TensorIOMode::kINPUT){++inputs;need(name=="images"&&type==DataType::kFLOAT&&shape(d,{1,3,768,1280}),"unexpected input");input=t;}
   else if(type==DataType::kFLOAT){outputs.push_back(t);floats[name]=d;}
   else{need(layout=="raw6"&&shape(d,{2}),"unexpected INT32 output");statuses.push_back(t);}
  }
  need(inputs==1,"one input required");
  // Canonical raw-output order is stable even if TensorRT enumerates IO differently.
  if(layout=="raw6"){
   need(outputs.size()==6&&(statuses.empty()||statuses.size()==9),"raw6 output/status count differs");
   const std::array<std::string,6> names={"cls_p3","cls_p4","cls_p5","reg_p3","reg_p4","reg_p5"};
   std::vector<Tensor> ordered;
   for(unsigned i=0;i<6;++i){auto it=std::find_if(outputs.begin(),outputs.end(),[&](auto& t){return t.name==names[i];});need(it!=outputs.end(),"missing raw output "+names[i]);const int k=i%3;need(shape(it->dims,{1,i<3?80:4,96>>k,160>>k}),"raw output shape mismatch "+names[i]);ordered.push_back(*it);}outputs=ordered;
   std::sort(statuses.begin(),statuses.end(),[](auto& a,auto& b){return a.name<b.name;});
  }else need(outputs.size()==1&&outputs[0].name=="output0"&&shape(outputs[0].dims,{1,300,6})&&statuses.empty(),"official default decoded output differs");
  CUDA(cudaMemGetInfo(&free_after_io,&total_device));std::size_t explicit_io_bytes=input.elements*4;for(auto& t:outputs)explicit_io_bytes+=t.elements*4;for(auto& t:statuses)explicit_io_bytes+=t.elements*4;const auto init_ms=std::chrono::duration<double,std::milli>(Clock::now()-init).count();
  const auto upload=[&](unsigned f){CUDA(cudaMemcpyAsync(input.p,images.data()+f*frame_bytes,frame_bytes,cudaMemcpyHostToDevice,r.stream));CUDA(cudaStreamSynchronize(r.stream));};
  const auto launch=[&](bool graph){if(graph)CUDA(cudaGraphLaunch(r.executable,r.stream));else need(context->enqueueV3(r.stream),"enqueueV3 failed");};
  const auto poison=[&]{for(auto& t:outputs)CUDA(cudaMemsetAsync(t.p,0xff,t.elements*4,r.stream));for(auto& t:statuses)CUDA(cudaMemsetAsync(t.p,0xff,t.elements*4,r.stream));};
  const auto snapshot=[&]{Snapshot s;for(auto& t:outputs){std::vector<float> h(t.elements);CUDA(cudaMemcpyAsync(h.data(),t.p,h.size()*4,cudaMemcpyDeviceToHost,r.stream));CUDA(cudaStreamSynchronize(r.stream));for(auto v:h)need(std::isfinite(v),"nonfinite output "+t.name);s.payload.append(reinterpret_cast<const char*>(h.data()),h.size()*4);}for(auto& t:statuses){saccade::ScanStatus h{};CUDA(cudaMemcpyAsync(&h,t.p,sizeof(h),cudaMemcpyDeviceToHost,r.stream));CUDA(cudaStreamSynchronize(r.stream));need(h.flags==0&&h.done==saccade::kScanDone,"scan status failed "+t.name);s.statuses.append(reinterpret_cast<const char*>(&h),sizeof(h));}return s;};
  upload(0);for(unsigned i=0;i<warmup;++i)launch(false);CUDA(cudaStreamSynchronize(r.stream));CUDA(cudaStreamBeginCapture(r.stream,cudaStreamCaptureModeGlobal));need(context->enqueueV3(r.stream),"capture failed");CUDA(cudaStreamEndCapture(r.stream,&r.graph));CUDA(cudaGraphInstantiate(&r.executable,r.graph,0));CUDA(cudaGraphUpload(r.executable,r.stream));CUDA(cudaStreamSynchronize(r.stream));
  CUDA(cudaMemGetInfo(&free_after_warmup,&total_device));std::array<Snapshot,4> refs;std::ostringstream v,io;v<<"input,payload_sha256,status_sha256,ordinary_repeat_bits,graph_bits,finite,status\n";io<<"kind,name,shape,elements,offset_elements\n";std::size_t offset=0;for(auto& t:outputs){io<<"payload,"<<t.name<<",\""<<ds(t.dims)<<"\","<<t.elements<<','<<offset<<'\n';offset+=t.elements;}offset=0;for(auto& t:statuses){io<<"status,"<<t.name<<",\""<<ds(t.dims)<<"\","<<t.elements<<','<<offset<<'\n';offset+=t.elements;}
  for(unsigned f=0;f<4;++f){upload(f);poison();launch(false);refs[f]=snapshot();poison();launch(false);equal(refs[f],snapshot());poison();launch(true);equal(refs[f],snapshot());saccade::write_text_file(out/("input-"+std::to_string(f)+".f32"),refs[f].payload);saccade::write_text_file(out/("input-"+std::to_string(f)+".status"),refs[f].statuses);v<<f<<','<<saccade::sha256_hex(refs[f].payload)<<','<<saccade::sha256_hex(refs[f].statuses)<<",PASS,PASS,PASS,PASS\n";}
  saccade::write_text_file(out/"verification.csv",v.str());saccade::write_text_file(out/"tensor-layout.csv",io.str());
  std::ostringstream samples,batches,groups;samples<<"round,input,ordinal,mode,iteration,gpu_event_ms,cpu_wall_ms\n"<<std::fixed<<std::setprecision(9);batches<<"round,input,ordinal,mode,replays,gpu_event_total_ms,cpu_wall_total_ms,gpu_interval_ms,cpu_interval_ms\n"<<std::fixed<<std::setprecision(9);groups<<"mode,gpu_mean_ms,gpu_p50_ms,gpu_p95_ms,cpu_mean_ms,cpu_p50_ms,cpu_p95_ms,queued_gpu_interval_ms,queued_cpu_interval_ms,queued_gpu_fps,queued_cpu_fps\n"<<std::fixed<<std::setprecision(9);
  std::array<std::vector<double>,2> gpu,wall,qgpu,qwall;
  if(!verify_only)for(unsigned round=0;round<rounds;++round)for(unsigned f=0;f<4;++f){upload(f);for(unsigned ordinal=0;ordinal<4;++ordinal){const bool graph=ordinal==1||ordinal==2;const unsigned m=graph?1:0;const char* label=graph?"graph":"ordinary";for(unsigned w=0;w<warmup;++w)launch(graph);CUDA(cudaStreamSynchronize(r.stream));for(unsigned i=0;i<iterations;++i){auto start=Clock::now();CUDA(cudaEventRecord(r.begin,r.stream));launch(graph);CUDA(cudaEventRecord(r.end,r.stream));CUDA(cudaEventSynchronize(r.end));float ms{};CUDA(cudaEventElapsedTime(&ms,r.begin,r.end));double cpu=std::chrono::duration<double,std::milli>(Clock::now()-start).count();gpu[m].push_back(ms);wall[m].push_back(cpu);samples<<round<<','<<f<<','<<ordinal<<','<<label<<','<<i<<','<<ms<<','<<cpu<<'\n';}equal(refs[f],snapshot());
    auto started=Clock::now();CUDA(cudaEventRecord(r.begin,r.stream));for(unsigned i=0;i<queue;++i)launch(graph);CUDA(cudaEventRecord(r.end,r.stream));CUDA(cudaEventSynchronize(r.end));float ms{};CUDA(cudaEventElapsedTime(&ms,r.begin,r.end));double cpu=std::chrono::duration<double,std::milli>(Clock::now()-started).count();qgpu[m].push_back(ms/queue);qwall[m].push_back(cpu/queue);batches<<round<<','<<f<<','<<ordinal<<','<<label<<','<<queue<<','<<ms<<','<<cpu<<','<<ms/queue<<','<<cpu/queue<<'\n';equal(refs[f],snapshot());std::cout<<"round="<<round<<" input="<<f<<" ordinal="<<ordinal<<" mode="<<label<<" complete\n"<<std::flush;
  }}
  if(!verify_only)for(unsigned m=0;m<2;++m)groups<<(m?"graph":"ordinary")<<','<<mean(gpu[m])<<','<<pct(gpu[m],.5)<<','<<pct(gpu[m],.95)<<','<<mean(wall[m])<<','<<pct(wall[m],.5)<<','<<pct(wall[m],.95)<<','<<mean(qgpu[m])<<','<<mean(qwall[m])<<','<<1000/mean(qgpu[m])<<','<<1000/mean(qwall[m])<<'\n';
  CUDA(cudaMemGetInfo(&free_after_measure,&total_device));need(logger.errors==0,"TensorRT logged errors");saccade::write_text_file(out/"samples.csv",samples.str());saccade::write_text_file(out/"queued.csv",batches.str());saccade::write_text_file(out/"groups.csv",groups.str());
  std::ostringstream memory;memory<<"{\n\"engine_plan_bytes\":"<<plan.size()<<",\n\"engine_getDeviceMemorySizeV2_bytes\":"<<context_device_bytes<<",\n\"explicit_IO_and_status_allocations_bytes\":"<<explicit_io_bytes<<",\n\"device_total_bytes\":"<<total_device<<",\n\"device_free_before_runtime_bytes\":"<<free_before_runtime<<",\n\"device_free_after_engine_bytes\":"<<free_after_engine<<",\n\"device_free_after_context_bytes\":"<<free_after_context<<",\n\"device_free_after_IO_bytes\":"<<free_after_io<<",\n\"device_free_after_graph_warmup_bytes\":"<<free_after_warmup<<",\n\"device_free_after_measure_bytes\":"<<free_after_measure<<",\n\"device_free_delta_scope\":\"cudaMemGetInfo whole-device free bytes at this fresh-process stages, not private allocation attribution or NVML peak\"\n}\n";saccade::write_text_file(out/"memory.json",memory.str());
  std::ostringstream id;id<<"{\n\"status\":\"PASS\",\n\"layout\":\""<<layout<<"\",\n\"verify_only\":"<<(verify_only?"true":"false")<<",\n\"engine_sha256\":\""<<saccade::sha256_hex(plan)<<"\",\n\"images_sha256\":\""<<saccade::sha256_hex(images)<<"\",\n\"frames_verified\":4,\n\"input_timing_frames\":[0,1,2,3],\n\"iterations_per_group\":"<<iterations<<",\n\"warmup_per_group\":"<<warmup<<",\n\"rounds\":"<<rounds<<",\n\"queued_replays_per_group\":"<<queue<<",\n\"initialization_ms\":"<<std::setprecision(9)<<init_ms<<",\n\"timing_excludes\":\"H2D,D2H,allocation,initialization,warmup,verification,file_IO\",\n\"single_scope\":\"one enqueue/Graph launch through event synchronization; GPUevent span and CPUwall reported\",\n\"queued_scope\":\"back-to-back same-input replays per batch; interval/throughput, not individual frame latency\",\n\"ordinary_repeat_and_graph_bits_equal\":true,\n\"persistent_cache_bytes\":0,\n\"nvtx\":\"none\",\n\"aux_streams\":0,\n\"scan_status_count\":"<<statuses.size()<<",\n\"formal_acceptance\":false\n}\n";saccade::write_text_file(out/"summary.json",id.str());std::cout<<id.str();return 0;
 }catch(const std::exception& e){std::cerr<<"model_compare: "<<e.what()<<'\n';if(!out.empty()&&std::filesystem::exists(out))try{saccade::write_text_file(out/"FAILURE.txt",e.what());}catch(...){}return 1;}
}

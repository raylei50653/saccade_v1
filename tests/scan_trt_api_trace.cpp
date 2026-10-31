#include "scan_trt_api_trace.hpp"

#include <cuda.h>
#include <cupti.h>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {

std::mutex trace_mutex;
std::vector<TracedCall> trace_calls;
std::atomic_bool armed{false};
CUpti_SubscriberHandle subscriber{};

void cupti(CUptiResult result, const char* call) {
    if (result == CUPTI_SUCCESS) return;
    const char* text = nullptr;
    cuptiGetResultString(result, &text);
    throw std::runtime_error(std::string(call) + ": " + (text ? text : "unknown CUPTI error"));
}

void decode_runtime(CUpti_CallbackId id, const void* p, TracedCall& c) {
    const auto stream = [&](cudaStream_t s) { c.stream_bearing = c.stream_known = true; c.stream = s; };
    switch (id) {
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000: stream(static_cast<const cudaLaunchKernel_v7000_params*>(p)->stream); break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
        stream(static_cast<const cudaLaunchKernelExC_v11060_params*>(p)->config->stream);
        break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemsetAsync_v3020: stream(static_cast<const cudaMemsetAsync_v3020_params*>(p)->stream); break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020: {
        const auto* q = static_cast<const cudaMemcpyAsync_v3020_params*>(p);
        stream(q->stream);
        c.device_to_host = q->kind == cudaMemcpyDeviceToHost || q->kind == cudaMemcpyDefault;
        break;
    }
    case CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020: stream(static_cast<const cudaEventRecord_v3020_params*>(p)->stream); break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020: stream(static_cast<const cudaStreamWaitEvent_v3020_params*>(p)->stream); break;
    case CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000: stream(static_cast<const cudaGraphLaunch_v10000_params*>(p)->stream); break;
    default: break;
    }
}

void decode_driver(CUpti_CallbackId id, const void* p, TracedCall& c) {
    const auto stream = [&](CUstream s) { c.stream_bearing = c.stream_known = true; c.stream = reinterpret_cast<cudaStream_t>(s); };
    switch (id) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel: stream(static_cast<const cuLaunchKernel_params*>(p)->hStream); break;
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx: stream(static_cast<const cuLaunchKernelEx_params*>(p)->config->hStream); break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async: stream(static_cast<const cuMemsetD8Async_params*>(p)->hStream); break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async: stream(static_cast<const cuMemsetD32Async_params*>(p)->hStream); break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2: stream(static_cast<const cuMemcpyDtoDAsync_v2_params*>(p)->hStream); break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2: stream(static_cast<const cuMemcpyHtoDAsync_v2_params*>(p)->hStream); break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
        stream(static_cast<const cuMemcpyDtoHAsync_v2_params*>(p)->hStream);
        c.device_to_host = true;
        break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
        stream(static_cast<const cuMemcpyAsync_params*>(p)->hStream);
        c.device_to_host = true; // unified addressing: direction unknown, treated as a possible D2H
        break;
    case CUPTI_DRIVER_TRACE_CBID_cuEventRecord: stream(static_cast<const cuEventRecord_params*>(p)->hStream); break;
    case CUPTI_DRIVER_TRACE_CBID_cuStreamWaitEvent: stream(static_cast<const cuStreamWaitEvent_params*>(p)->hStream); break;
    case CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch: stream(static_cast<const cuGraphLaunch_params*>(p)->hStream); break;
    default: break;
    }
}

void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data) {
    if (!armed.load()) return;
    const auto* info = static_cast<const CUpti_CallbackData*>(data);
    if (info->callbackSite != CUPTI_API_ENTER) return;
    try {
        TracedCall c;
        c.name = info->functionName ? info->functionName : "unknown";
        c.driver = domain == CUPTI_CB_DOMAIN_DRIVER_API;
        if (c.driver) decode_driver(id, info->functionParams, c);
        else decode_runtime(id, info->functionParams, c);
        // A call named like stream work whose stream was not decoded stays
        // flagged, so a new entry point cannot slip through unchecked.
        const auto has = [&](const char* s) { return c.name.find(s) != std::string::npos; };
        if (!c.stream_known && (has("Launch") || has("Async") || has("EventRecord") || has("WaitEvent"))) c.stream_bearing = true;
        std::lock_guard lock(trace_mutex);
        trace_calls.push_back(std::move(c));
    } catch (...) {
    }
}

} // namespace

CudaApiTrace::CudaApiTrace() {
    cupti(cuptiSubscribe(&subscriber, reinterpret_cast<CUpti_CallbackFunc>(callback), nullptr), "cuptiSubscribe");
    cupti(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API), "cuptiEnableDomain(runtime)");
    cupti(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API), "cuptiEnableDomain(driver)");
}

CudaApiTrace::~CudaApiTrace() {
    armed = false;
    cuptiUnsubscribe(subscriber);
}

void CudaApiTrace::arm() {
    std::lock_guard lock(trace_mutex);
    trace_calls.clear();
    armed = true;
}

std::vector<TracedCall> CudaApiTrace::disarm() {
    armed = false;
    std::lock_guard lock(trace_mutex);
    return std::move(trace_calls);
}

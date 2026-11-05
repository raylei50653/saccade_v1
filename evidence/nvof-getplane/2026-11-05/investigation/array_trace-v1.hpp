#pragma once

// Optional diagnostic observer. It never issues CUDA calls from a callback.
// Array descriptors are copied from successful driver API creation callbacks.
#include <cupti.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>

#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace nvof_array_trace_detail {
struct Descriptor {
    std::size_t width = 0, height = 0, depth = 0;
    CUarray_format format = CU_AD_FORMAT_UNSIGNED_INT8;
    unsigned channels = 0, flags = 0;
    const char* source = nullptr;
};
struct State {
    CUpti_SubscriberHandle subscriber = nullptr;
    std::mutex mutex;
    std::unordered_map<CUarray, Descriptor> arrays;
    unsigned long long get_plane_calls = 0, unknown_arrays = 0;
};
inline State state;

inline const char* format_name(CUarray_format f) noexcept {
    switch (f) {
    case CU_AD_FORMAT_UNSIGNED_INT8: return "UNSIGNED_INT8";
    case CU_AD_FORMAT_UNSIGNED_INT16: return "UNSIGNED_INT16";
    case CU_AD_FORMAT_UNSIGNED_INT32: return "UNSIGNED_INT32";
    case CU_AD_FORMAT_SIGNED_INT8: return "SIGNED_INT8";
    case CU_AD_FORMAT_SIGNED_INT16: return "SIGNED_INT16";
    case CU_AD_FORMAT_SIGNED_INT32: return "SIGNED_INT32";
    case CU_AD_FORMAT_HALF: return "HALF";
    case CU_AD_FORMAT_FLOAT: return "FLOAT";
    case CU_AD_FORMAT_NV12: return "NV12";
    default: return "OTHER";
    }
}
inline void check(CUptiResult result, const char* operation) {
    if (result == CUPTI_SUCCESS) return;
    const char* message = nullptr;
    (void)cuptiGetResultString(result, &message);
    throw std::runtime_error(std::string(operation) + ": CUPTI=" +
                             std::to_string(static_cast<int>(result)) + " " +
                             (message ? message : "unknown error"));
}

inline void CUPTIAPI callback(void*, CUpti_CallbackDomain domain,
                              CUpti_CallbackId id, const void* payload) noexcept {
    if (domain != CUPTI_CB_DOMAIN_DRIVER_API || !payload) return;
    const auto* cb = static_cast<const CUpti_CallbackData*>(payload);
    if (cb->callbackSite != CUPTI_API_EXIT || !cb->functionParams || !cb->functionReturnValue) return;
    const auto result = *static_cast<const CUresult*>(cb->functionReturnValue);
    try {
        std::lock_guard<std::mutex> lock(state.mutex);
        if (id == CUPTI_DRIVER_TRACE_CBID_cuArrayCreate_v2 && result == CUDA_SUCCESS) {
            const auto* p = static_cast<const cuArrayCreate_v2_params*>(cb->functionParams);
            if (p->pHandle && p->pAllocateArray) {
                const auto& d = *p->pAllocateArray;
                state.arrays[*p->pHandle] = {d.Width, d.Height, 0, d.Format, d.NumChannels, 0, "cuArrayCreate_v2"};
            }
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate_v2 && result == CUDA_SUCCESS) {
            const auto* p = static_cast<const cuArray3DCreate_v2_params*>(cb->functionParams);
            if (p->pHandle && p->pAllocateArray) {
                const auto& d = *p->pAllocateArray;
                state.arrays[*p->pHandle] = {d.Width, d.Height, d.Depth, d.Format, d.NumChannels, d.Flags, "cuArray3DCreate_v2"};
            }
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuArrayDestroy && result == CUDA_SUCCESS) {
            const auto* p = static_cast<const cuArrayDestroy_params*>(cb->functionParams);
            state.arrays.erase(p->hArray);
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuArrayGetPlane) {
            const auto* p = static_cast<const cuArrayGetPlane_params*>(cb->functionParams);
            const auto entry = state.arrays.find(p->hArray);
            const auto sequence = ++state.get_plane_calls;
            if (entry == state.arrays.end()) {
                ++state.unknown_arrays;
                std::fprintf(stderr,
                    "ARRAY_TRACE GetPlane seq=%llu context=%p hArray=%p planeIdx=%u CUresult=%d descriptor=UNKNOWN\n",
                    sequence, static_cast<void*>(cb->context), static_cast<void*>(p->hArray), p->planeIdx,
                    static_cast<int>(result));
            } else {
                const auto& d = entry->second;
                std::fprintf(stderr,
                    "ARRAY_TRACE GetPlane seq=%llu context=%p hArray=%p planeIdx=%u CUresult=%d "
                    "format=%s(0x%x) channels=%u width=%zu height=%zu depth=%zu flags=0x%x source=%s\n",
                    sequence, static_cast<void*>(cb->context), static_cast<void*>(p->hArray), p->planeIdx,
                    static_cast<int>(result), format_name(d.format), static_cast<unsigned>(d.format),
                    d.channels, d.width, d.height, d.depth, d.flags, d.source);
            }
            std::fflush(stderr);
        }
    } catch (...) {
        // CUPTI invokes this through a C ABI; exceptions must never escape.
        std::fprintf(stderr, "ARRAY_TRACE callback failed; descriptor coverage is incomplete\n");
        std::fflush(stderr);
    }
}
} // namespace nvof_array_trace_detail

inline void start_array_trace() {
    using namespace nvof_array_trace_detail;
    if (state.subscriber) throw std::runtime_error("array trace is already subscribed");
    state.arrays.clear();
    state.get_plane_calls = state.unknown_arrays = 0;
    check(cuptiSubscribe(&state.subscriber, callback, nullptr), "cuptiSubscribe array trace");
    try {
        for (const auto id : {CUPTI_DRIVER_TRACE_CBID_cuArrayCreate_v2,
                              CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate_v2,
                              CUPTI_DRIVER_TRACE_CBID_cuArrayDestroy,
                              CUPTI_DRIVER_TRACE_CBID_cuArrayGetPlane})
            check(cuptiEnableCallback(1, state.subscriber, CUPTI_CB_DOMAIN_DRIVER_API, id), "cuptiEnableCallback array trace");
    } catch (...) {
        (void)cuptiUnsubscribe(state.subscriber);
        state.subscriber = nullptr;
        throw;
    }
    std::fprintf(stderr, "ARRAY_TRACE subscribed; driver GetPlane and v2 array create/destroy callbacks enabled\n");
}

inline void stop_array_trace() {
    using namespace nvof_array_trace_detail;
    if (!state.subscriber) return;
    check(cuptiUnsubscribe(state.subscriber), "cuptiUnsubscribe array trace");
    state.subscriber = nullptr;
    std::lock_guard<std::mutex> lock(state.mutex);
    std::fprintf(stderr, "ARRAY_TRACE stopped GetPlane_calls=%llu unknown_arrays=%llu tracked_live_arrays=%zu\n",
                 state.get_plane_calls, state.unknown_arrays, state.arrays.size());
    std::fflush(stderr);
}

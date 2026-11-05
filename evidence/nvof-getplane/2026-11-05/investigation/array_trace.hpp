#pragma once

// Optional diagnostic observer. It never issues CUDA calls from a callback.
// Array descriptors are copied from successful driver API creation callbacks.
#include <cupti.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
    std::unordered_map<CUmipmappedArray, Descriptor> mipmaps;
    std::unordered_map<CUarray, CUmipmappedArray> parents;
    std::unordered_map<CUarray, CUcontext> observed;
    unsigned long long get_plane_calls = 0, unknown_arrays = 0;
    unsigned long long query_boundaries = 0;
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

inline Descriptor descriptor(const CUDA_ARRAY3D_DESCRIPTOR& d, const char* source) {
    return {d.Width, d.Height, d.Depth, d.Format, d.NumChannels, d.Flags, source};
}
inline void erase_array(CUarray array) {
    state.arrays.erase(array);
    state.observed.erase(array);
    state.parents.erase(array);
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
            erase_array(p->hArray);
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuArrayGetDescriptor_v2 && result == CUDA_SUCCESS) {
            const auto* p = static_cast<const cuArrayGetDescriptor_v2_params*>(cb->functionParams);
            if (p->pArrayDescriptor) {
                const auto& d = *p->pArrayDescriptor;
                state.arrays[p->hArray] = {d.Width, d.Height, 0, d.Format, d.NumChannels, 0, "cuArrayGetDescriptor_v2"};
            }
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuArray3DGetDescriptor_v2 && result == CUDA_SUCCESS) {
            const auto* p = static_cast<const cuArray3DGetDescriptor_v2_params*>(cb->functionParams);
            if (p->pArrayDescriptor) state.arrays[p->hArray] = descriptor(*p->pArrayDescriptor, "cuArray3DGetDescriptor_v2");
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayCreate && result == CUDA_SUCCESS) {
            const auto* p = static_cast<const cuMipmappedArrayCreate_params*>(cb->functionParams);
            if (p->pHandle && p->pMipmappedArrayDesc)
                state.mipmaps[*p->pHandle] = descriptor(*p->pMipmappedArrayDesc, "cuMipmappedArrayCreate");
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuExternalMemoryGetMappedMipmappedArray && result == CUDA_SUCCESS) {
            const auto* p = static_cast<const cuExternalMemoryGetMappedMipmappedArray_params*>(cb->functionParams);
            if (p->mipmap && p->mipmapDesc)
                state.mipmaps[*p->mipmap] = descriptor(p->mipmapDesc->arrayDesc, "cuExternalMemoryGetMappedMipmappedArray");
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetLevel && result == CUDA_SUCCESS) {
            const auto* p = static_cast<const cuMipmappedArrayGetLevel_params*>(cb->functionParams);
            if (p->pLevelArray) {
                state.parents[*p->pLevelArray] = p->hMipmappedArray;
                const auto entry = state.mipmaps.find(p->hMipmappedArray);
                if (entry != state.mipmaps.end()) {
                    auto d = entry->second;
                    const auto shrink = [level=p->level](std::size_t v) {
                        if (!v) return v;
                        return level >= 8 * sizeof(v) ? std::size_t(1) : std::max(std::size_t(1), v >> level);
                    };
                    d.width = shrink(d.width);
                    if (!(d.flags & CUDA_ARRAY3D_LAYERED) || d.depth) d.height = shrink(d.height);
                    if (!(d.flags & (CUDA_ARRAY3D_LAYERED | CUDA_ARRAY3D_CUBEMAP))) d.depth = shrink(d.depth);
                    d.source = "cuMipmappedArrayGetLevel-inferred-from-base";
                    state.arrays[*p->pLevelArray] = d;
                }
            }
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayDestroy && result == CUDA_SUCCESS) {
            const auto* p = static_cast<const cuMipmappedArrayDestroy_params*>(cb->functionParams);
            state.mipmaps.erase(p->hMipmappedArray);
            for (auto it = state.parents.begin(); it != state.parents.end();) {
                if (it->second == p->hMipmappedArray) {
                    state.arrays.erase(it->first);
                    state.observed.erase(it->first);
                    it = state.parents.erase(it);
                } else ++it;
            }
        } else if (id == CUPTI_DRIVER_TRACE_CBID_cuArrayGetPlane) {
            const auto* p = static_cast<const cuArrayGetPlane_params*>(cb->functionParams);
            state.observed[p->hArray] = cb->context;
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
    state.mipmaps.clear(); state.parents.clear(); state.observed.clear();
    state.get_plane_calls = state.unknown_arrays = 0;
    state.query_boundaries = 0;
    check(cuptiSubscribe(&state.subscriber, callback, nullptr), "cuptiSubscribe array trace");
    try {
        for (const auto id : {CUPTI_DRIVER_TRACE_CBID_cuArrayCreate_v2,
                              CUPTI_DRIVER_TRACE_CBID_cuArray3DCreate_v2,
                              CUPTI_DRIVER_TRACE_CBID_cuArrayGetDescriptor_v2,
                              CUPTI_DRIVER_TRACE_CBID_cuArray3DGetDescriptor_v2,
                              CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayCreate,
                              CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayGetLevel,
                              CUPTI_DRIVER_TRACE_CBID_cuMipmappedArrayDestroy,
                              CUPTI_DRIVER_TRACE_CBID_cuExternalMemoryGetMappedMipmappedArray,
                              CUPTI_DRIVER_TRACE_CBID_cuArrayDestroy,
                              CUPTI_DRIVER_TRACE_CBID_cuArrayGetPlane})
            check(cuptiEnableCallback(1, state.subscriber, CUPTI_CB_DOMAIN_DRIVER_API, id), "cuptiEnableCallback array trace");
    } catch (...) {
        (void)cuptiUnsubscribe(state.subscriber);
        state.subscriber = nullptr;
        throw;
    }
    std::fprintf(stderr, "ARRAY_TRACE subscribed v2; GetPlane, array/mipmap create/destroy/getter callbacks enabled\n");
}

// Call only at ordinary host API boundaries, never from a CUPTI callback.
// The mutex is deliberately released before any CUDA call because those calls
// may synchronously invoke the observer again.
inline void query_traced_arrays() {
    using namespace nvof_array_trace_detail;
    std::vector<std::pair<CUarray, CUcontext>> live;
    unsigned long long boundary;
    {
        std::lock_guard<std::mutex> lock(state.mutex);
        boundary = ++state.query_boundaries;
        live.assign(state.observed.begin(), state.observed.end());
    }
    CUcontext current = nullptr;
    const auto current_result = cuCtxGetCurrent(&current);
    std::fprintf(stderr, "ARRAY_QUERY BEGIN boundary=%llu current_context=%p context_CUresult=%d observed_live=%zu\n",
                 boundary, static_cast<void*>(current), static_cast<int>(current_result), live.size());
    for (const auto& entry : live) {
        const auto array = entry.first;
        const auto context = entry.second;
        bool pushed = false;
        if (context && context != current) {
            const auto result = cuCtxPushCurrent(context);
            if (result != CUDA_SUCCESS) {
                std::fprintf(stderr, "ARRAY_QUERY boundary=%llu hArray=%p observed_context=%p push_CUresult=%d\n",
                             boundary, static_cast<void*>(array), static_cast<void*>(context), static_cast<int>(result));
                continue;
            }
            pushed = true;
        }
        CUDA_ARRAY3D_DESCRIPTOR d{};
        const auto result = cuArray3DGetDescriptor(&d, array);
        if (result == CUDA_SUCCESS) {
            std::fprintf(stderr, "ARRAY_QUERY boundary=%llu hArray=%p observed_context=%p CUresult=0 "
                         "format=%s(0x%x) channels=%u width=%zu height=%zu depth=%zu flags=0x%x\n",
                         boundary, static_cast<void*>(array), static_cast<void*>(context), format_name(d.Format),
                         static_cast<unsigned>(d.Format), d.NumChannels, d.Width, d.Height, d.Depth, d.Flags);
        } else {
            std::fprintf(stderr, "ARRAY_QUERY boundary=%llu hArray=%p observed_context=%p CUresult=%d descriptor=UNAVAILABLE\n",
                         boundary, static_cast<void*>(array), static_cast<void*>(context), static_cast<int>(result));
        }
        if (pushed) {
            CUcontext popped = nullptr;
            const auto pop_result = cuCtxPopCurrent(&popped);
            if (pop_result != CUDA_SUCCESS)
                throw std::runtime_error("array query failed to restore CUDA context: CUresult=" + std::to_string(static_cast<int>(pop_result)));
        }
    }
    std::fprintf(stderr, "ARRAY_QUERY END boundary=%llu\n", boundary);
    std::fflush(stderr);
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

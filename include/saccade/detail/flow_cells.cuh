#pragma once
#include "saccade/flow_records.hpp"
#include <cuda_runtime.h>
namespace saccade::flow::detail {
__device__ inline float median4(int a,int b,int c,int d) {
    if(a>b) { const int t=a; a=b; b=t; }
    if(c>d) { const int t=c; c=d; d=t; }
    const int lo=a>c?a:c, hi=b<d?b:d;
    return float(lo+hi)/64.f;
}
__device__ inline Vector reduce_cell(const std::uint8_t* raw,std::size_t pitch,int x,int y) {
    const auto* a=reinterpret_cast<const short2*>(raw+2*y*pitch)+2*x;
    const auto* b=reinterpret_cast<const short2*>(raw+(2*y+1)*pitch)+2*x;
    return {median4(a[0].x,a[1].x,b[0].x,b[1].x),median4(a[0].y,a[1].y,b[0].y,b[1].y)};
}
}

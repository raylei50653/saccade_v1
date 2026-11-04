#include "saccade/gmc_flow.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace saccade::gmc_flow {
const char* name(Arm a) {
    switch(a) {
    case Arm::off: return "OFF"; case Arm::g0: return "G0"; case Arm::g1: return "G1";
    case Arm::g1q: return "G1q"; case Arm::l1: return "L1"; case Arm::l2: return "L2";
    }
    throw std::invalid_argument("unknown GMC arm");
}
Arm parse_arm(std::string_view text) {
    for(auto arm:arms) if(text==name(arm)) return arm;
    throw std::invalid_argument("unknown GMC arm");
}
bool contains(const Box& b,float x,float y) { return x>=b[0] && x<b[2] && y>=b[1] && y<b[3]; }
flow::Vector median(std::vector<flow::Vector> values) {
    if(values.empty()) throw std::invalid_argument("median of empty cells");
    const auto n=values.size();
    auto component=[&](auto member) {
        std::sort(values.begin(),values.end(),[&](const auto& a,const auto& b){return a.*member < b.*member;});
        return n%2 ? values[n/2].*member : (values[n/2-1].*member + values[n/2].*member)*0.5f;
    };
    return {component(&flow::Vector::dx),component(&flow::Vector::dy)};
}
FrameControl::FrameControl(const flow::Frame& f,int cols,int rows,const DetectionCandidates* previous,const MotionContext& pcr)
    : frame_(&f),cols_(cols),rows_(rows),pcr_{pcr.status==MotionStatus::valid?pcr.tx:0.f,pcr.status==MotionStatus::valid?pcr.ty:0.f},
      mask_(static_cast<std::size_t>(cols)*rows,0) {
    if(previous) {
        for(std::uint32_t i=0;i<previous->header.count;++i) {
            const auto& d=previous->rows[i];
            if(!(d.flags & kDetectionKept) || d.class_id!=0 || d.score<0.45f) continue;
            const Box box{d.x1,d.y1,d.x2,d.y2};
            const int c0=std::clamp(static_cast<int>(std::ceil((box[0]-4.f)/8.f)),0,cols_);
            const int c1=std::clamp(static_cast<int>(std::ceil((box[2]-4.f)/8.f)),0,cols_);
            const int r0=std::clamp(static_cast<int>(std::ceil((box[1]-4.f)/8.f)),0,rows_);
            const int r1=std::clamp(static_cast<int>(std::ceil((box[3]-4.f)/8.f)),0,rows_);
            for(int r=r0;r<r1;++r) for(int c=c0;c<c1;++c) mask_[static_cast<std::size_t>(r)*cols_+c]=1;
        }
    }
    std::vector<flow::Vector> background;
    for(std::size_t i=0;i<mask_.size();++i) if(!mask_[i]) background.push_back(f.cells[i]);
    available_=background.size();
    if(valid()) global_=median(std::move(background));
}
Control FrameControl::evaluate(Arm a,const Box& own) const {
    if(a==Arm::off) return {};
    if(a==Arm::g0) return {pcr_,false,0};
    if(a==Arm::g1) return {global_,false,available_};
    if(a==Arm::g1q) return {{4.f*std::round(global_.dx/4.f),4.f*std::round(global_.dy/4.f)},false,available_};
    const float w=own[2]-own[0], h=own[3]-own[1], cx=(own[0]+own[2])*0.5f, cy=(own[1]+own[3])*0.5f;
    const Box region = a==Arm::l1 ? Box{cx-1.5f*w,cy-0.75f*h,cx+1.5f*w,cy+0.75f*h} :
                                   Box{cx-1.5f*w,own[3],cx+1.5f*w,own[3]+0.3f*h};
    const int c0=std::clamp(static_cast<int>(std::ceil((region[0]-4.f)/8.f)),0,cols_);
    const int c1=std::clamp(static_cast<int>(std::ceil((region[2]-4.f)/8.f)),0,cols_);
    const int r0=std::clamp(static_cast<int>(std::ceil((region[1]-4.f)/8.f)),0,rows_);
    const int r1=std::clamp(static_cast<int>(std::ceil((region[3]-4.f)/8.f)),0,rows_);
    std::vector<flow::Vector> local;
    for(int r=r0;r<r1;++r) for(int c=c0;c<c1;++c) {
        const auto i=static_cast<std::size_t>(r)*cols_+c;
        if(!mask_[i] && !contains(own,float(8*c+4),float(8*r+4))) local.push_back(frame_->cells[i]);
    }
    const auto n=local.size();
    if(n<8) return {global_,true,n};
    return {median(std::move(local)),false,n};
}
} // namespace saccade::gmc_flow

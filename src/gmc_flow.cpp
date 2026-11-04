#include "saccade/gmc_flow.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace saccade::gmc_flow {
const char* name(Arm a) {
    switch(a) {
    case Arm::off: return "OFF"; case Arm::g0: return "G0"; case Arm::g1: return "G1";
    case Arm::g1q: return "G1q"; case Arm::l1: return "L1"; case Arm::l2: return "L2";
    case Arm::ld: return "LD"; case Arm::r: return "R"; case Arm::rc: return "RC";
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
namespace {
double scalar_median(std::vector<double> v) {
    std::sort(v.begin(),v.end());
    const auto n=v.size();
    return n%2 ? v[n/2] : (v[n/2-1]+v[n/2])*0.5;
}
double kernel(double x,double y) { return std::exp(-(x*x+y*y)/2.0); }
}
Line tukey_line(const std::vector<double>& u,const std::vector<double>& d) {
    Line line{scalar_median(d),0};
    for(int iteration=0;iteration<10;++iteration) {
        std::vector<double> residuals;
        for(std::size_t i=0;i<d.size();++i) residuals.push_back(d[i]-line.a-line.b*u[i]);
        const double center=scalar_median(residuals);
        std::vector<double> deviations;
        for(double r:residuals) deviations.push_back(std::abs(r-center));
        const double scale=std::max(1.4826*scalar_median(deviations),0.25);
        double sw=0,su=0,suu=0,sd=0,sud=0;
        for(std::size_t i=0;i<d.size();++i) {
            const double t=residuals[i]/(4.685*scale);
            if(std::abs(t)>=1) continue;
            const double w=(1-t*t)*(1-t*t);
            sw+=w; su+=w*u[i]; suu+=w*u[i]*u[i]; sd+=w*d[i]; sud+=w*u[i]*d[i];
        }
        const double determinant=sw*suu-su*su;
        if(sw<2 || determinant==0) break;
        line={(sd*suu-su*sud)/determinant,(sw*sud-su*sd)/determinant};
    }
    return line;
}
double concentration_alpha(double d) { return std::clamp((d-0.2)/0.4,0.0,1.0); }
std::size_t density_start(const std::vector<WeightedCell>& cells) {
    std::size_t start=0; double best=-1;
    for(std::size_t i=0;i<cells.size();++i) {
        double density=0;
        for(const auto& c:cells) density+=c.weight*kernel(cells[i].dx-c.dx,cells[i].dy-c.dy);
        if(density>best) { best=density; start=i; }
    }
    return start;
}
std::array<double,2> mean_shift(const std::vector<WeightedCell>& cells,std::array<double,2> mu) {
    for(int iteration=0;iteration<50;++iteration) {
        double sum=0,x=0,y=0;
        for(const auto& c:cells) {
            const double w=c.weight*kernel(c.dx-mu[0],c.dy-mu[1]);
            sum+=w; x+=w*c.dx; y+=w*c.dy;
        }
        if(sum<1e-12) break;
        const std::array next{x/sum,y/sum};
        const double step=std::hypot(next[0]-mu[0],next[1]-mu[1]);
        mu=next;
        if(step<0.01) break;
    }
    return mu;
}
FrameControl::FrameControl(const flow::Frame& f,int cols,int rows,int width,const DetectionCandidates* previous,const MotionContext& pcr,int scale)
    : frame_(&f),cols_(cols),rows_(rows),width_(width),pcr_{pcr.status==MotionStatus::valid?pcr.tx:0.f,pcr.status==MotionStatus::valid?pcr.ty:0.f},
      mask_(static_cast<std::size_t>(cols)*rows,0) {
    if(previous) {
        for(std::uint32_t i=0;i<previous->header.count;++i) {
            const auto& d=previous->rows[i];
            if(!(d.flags & kDetectionKept) || d.class_id!=0 || d.score<0.45f) continue;
            const Box box{d.x1,d.y1,d.x2,d.y2};
            const auto bound=[&](float edge,int size) {
                int lo=0,hi=size;
                while(lo<hi) { const int mid=lo+(hi-lo)/2; if(float(scale*(8*mid+4))<edge) lo=mid+1; else hi=mid; }
                return lo;
            };
            const int c0=bound(box[0],cols_), c1=bound(box[2],cols_);
            const int r0=bound(box[1],rows_), r1=bound(box[3],rows_);
            for(int r=r0;r<r1;++r) for(int c=c0;c<c1;++c) mask_[static_cast<std::size_t>(r)*cols_+c]=1;
        }
    }
    std::vector<flow::Vector> background;
    for(std::size_t i=0;i<mask_.size();++i) if(!mask_[i]) background.push_back(f.cells[i]);
    available_=background.size();
    if(valid()) global_=median(std::move(background));
}
void FrameControl::prepare_rows() const {
    if(!ground_rows_.empty()) return;
    ground_rows_.resize(rows_);
    for(int row=0;row<rows_;++row) {
        std::vector<double> u,dx,dy;
        for(int r=std::max(0,row-1);r<=std::min(rows_-1,row+1);++r) for(int c=0;c<cols_;++c) {
            if(masked(c,r)) continue;
            const auto v=frame_->cells[static_cast<std::size_t>(r)*cols_+c];
            u.push_back(8*c+4-width_/2.0); dx.push_back(v.dx); dy.push_back(v.dy);
        }
        if(u.size()>=16) ground_rows_[row]={true,tukey_line(u,dx),tukey_line(u,dy)};
    }
}
flow::Vector FrameControl::row_value(int row,double x) const {
    const auto& m=ground_rows_[row];
    const double u=std::clamp(x,0.0,double(width_-1))-width_/2.0;
    return {static_cast<float>(m.dx.a+m.dx.b*u),static_cast<float>(m.dy.a+m.dy.b*u)};
}
Control FrameControl::ground(const Box& box) const {
    prepare_rows();
    const int foot=std::clamp(static_cast<int>(std::floor(box[3]/8)),0,rows_-1);
    for(int distance=0;distance<=4;++distance) for(int sign:{-1,1}) {
        if(distance==0 && sign==1) continue;
        const int row=foot+sign*distance;
        if(row<0 || row>=rows_ || !ground_rows_[row].available) continue;
        Control result{row_value(row,(double(box[0])+box[2])*0.5)};
        result.row_offset=row-foot;
        return result;
    }
    Control result{global_,true}; result.r_fallback=true; return result;
}
Control FrameControl::evaluate(Arm a,const Box& own) const {
    if(a==Arm::off) return {};
    if(a==Arm::g0) return {pcr_,false,0};
    if(a==Arm::g1) return {global_,false,available_};
    if(a==Arm::g1q) return {{4.f*std::round(global_.dx/4.f),4.f*std::round(global_.dy/4.f)},false,available_};
    if(a==Arm::r) return ground(own);
    Control prior;
    if(a==Arm::rc) prior=ground(own);
    const float w=own[2]-own[0], h=own[3]-own[1], cx=(own[0]+own[2])*0.5f, cy=(own[1]+own[3])*0.5f;
    const Box region = a==Arm::l1 ? Box{cx-1.5f*w,cy-0.75f*h,cx+1.5f*w,cy+0.75f*h} :
                                   Box{cx-1.5f*w,own[3],cx+1.5f*w,own[3]+0.3f*h};
    const int c0=std::clamp(static_cast<int>(std::ceil((region[0]-4.f)/8.f)),0,cols_);
    const int c1=std::clamp(static_cast<int>(std::ceil((region[2]-4.f)/8.f)),0,cols_);
    const int r0=std::clamp(static_cast<int>(std::ceil((region[1]-4.f)/8.f)),0,rows_);
    const int r1=std::clamp(static_cast<int>(std::ceil((region[3]-4.f)/8.f)),0,rows_);
    std::vector<flow::Vector> local;
    std::vector<WeightedCell> weighted;
    for(int r=r0;r<r1;++r) for(int c=c0;c<c1;++c) {
        const auto i=static_cast<std::size_t>(r)*cols_+c;
        if(!mask_[i] && !contains(own,float(8*c+4),float(8*r+4))) {
            local.push_back(frame_->cells[i]);
            if(a==Arm::ld || (a==Arm::rc && ground_rows_[r].available)) {
                const auto value=frame_->cells[i];
                const auto base=a==Arm::rc?row_value(r,8*c+4):flow::Vector{};
                const double z=(double(8*c+4)-cx)/(0.75*double(w));
                weighted.push_back({double(value.dx)-base.dx,double(value.dy)-base.dy,std::exp(-z*z/2)});
            }
        }
    }
    const auto n=local.size();
    if(a==Arm::rc) {
        prior.local_cells=weighted.size();
        if(weighted.size()<8) { prior.fallback=true; return prior; }
        const auto mu=mean_shift(weighted,{0,0});
        double numerator=0,denominator=0;
        for(const auto& c:weighted) { numerator+=c.weight*kernel(c.dx-mu[0],c.dy-mu[1]); denominator+=c.weight; }
        prior.density=numerator/denominator;
        prior.alpha=concentration_alpha(prior.density);
        prior.value={static_cast<float>(prior.value.dx+prior.alpha*mu[0]),static_cast<float>(prior.value.dy+prior.alpha*mu[1])};
        return prior;
    }
    if(n<8) return {global_,true,n};
    if(a==Arm::ld) {
        const auto start=density_start(weighted);
        const auto mu=mean_shift(weighted,{weighted[start].dx,weighted[start].dy});
        return {{static_cast<float>(mu[0]),static_cast<float>(mu[1])},false,n};
    }
    return {median(std::move(local)),false,n};
}
} // namespace saccade::gmc_flow

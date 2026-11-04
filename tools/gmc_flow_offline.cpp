// Fixed PLAN §4 controls and §5.3 residuals share FrameControl::evaluate.
#include "saccade/gmc_flow.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/tracking.hpp"
#include "saccade/tracking_math.hpp"
#include "saccade/file_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace saccade;
using namespace saccade::gmc_flow;
struct Inputs {
    DetectionRecords detections;
    MotionRecords motion;
    flow::Records flow;
    Inputs(char** argv): detections(read_detection_records(argv[2])), motion(read_motion_records(argv[3])), flow(flow::read_records(argv[4])) {
        if(detections.frames.size()!=motion.frames.size() || detections.frames.size()!=flow.frames.size())
            throw std::runtime_error("input frame counts differ");
    }
    FrameControl control(std::size_t k) const {
        return {flow.frames[k],flow.cols,flow.rows,flow.width,k?&detections.frames[k-1]:nullptr,motion.frames[k]};
    }
};
void track(const Inputs& in,Arm arm,const char* mot_path,const char* diag_path,const char* equivalent) {
    TrackerReference tracker(in.detections.epoch);
    SequenceOutput output;
    MotionRecords scm{in.motion.key,in.motion.epoch,{}};
    std::ofstream diag(diag_path);
    if(!diag) throw std::runtime_error("cannot open tracking diagnostics");
    diag << "frame,uid,x1,y1,x2,y2,dx,dy,local_cells,fallback,global_cells,r_fallback,row_offset,alpha,density,delta_l2,delta_r\n" << std::setprecision(9);
    std::ostringstream motion;
    motion << "frame,g0_dx,g0_dy,g0_rejected,g1_dx,g1_dy,g1_valid,background_cells,delta_norm\n" << std::setprecision(9);
    for(std::size_t k=0;k<in.detections.frames.size();++k) {
        const auto control=in.control(k);
        auto m=in.motion.frames[k];
        if(arm==Arm::off) { m.status=MotionStatus::disabled; m.tx=m.ty=0; }
        else if(arm!=Arm::g0) {
            m.tx=m.ty=m.pcr=m.confidence=0; m.peak_x=m.peak_y=0;
            m.status=k==0?MotionStatus::uninitialized:(control.valid()?MotionStatus::valid:MotionStatus::rejected);
            m.reference_frame=k==0?-1:static_cast<std::int64_t>(k)-1;
        }
        tracker.set_control_hook([&](const Track& t) {
            Box box; tracking_math::state_to_box(t.x.data(),box.data());
            const auto c=control.evaluate(arm,box);
            diag << k << ',' << t.uid;
            for(auto v:box) diag << ',' << v;
            diag << ',' << c.value.dx << ',' << c.value.dy << ',' << c.local_cells << ',' << c.fallback << ',' << control.available()
                 << ',' << c.r_fallback << ',' << c.row_offset << ',' << c.alpha << ',' << c.density;
            double delta_l2=0,delta_r=0;
            if(arm==Arm::rc) {
                const auto l2=control.evaluate(Arm::l2,box).value, r=control.evaluate(Arm::r,box).value;
                delta_l2=std::hypot(double(c.value.dx)-l2.dx,double(c.value.dy)-l2.dy);
                delta_r=std::hypot(double(c.value.dx)-r.dx,double(c.value.dy)-r.dy);
            }
            diag << ',' << delta_l2 << ',' << delta_r << '\n';
            return std::array{c.value.dx,c.value.dy};
        });
        const auto packet=tracker.step(in.detections.frames[k],m);
        output.add(packet);
        if(output.failed()) throw std::runtime_error("tracker failed at frame "+std::to_string(k)+" error="+std::to_string(static_cast<unsigned>(packet.frame_error)));
        if(equivalent) {
            const auto v=control.evaluate(arm,{}).value;
            MotionContext eq; eq.epoch=in.motion.epoch; eq.frame=static_cast<std::int64_t>(k);
            eq.reference_frame=eq.frame-1; eq.status=MotionStatus::valid; eq.confidence=1; eq.tx=v.dx; eq.ty=v.dy;
            scm.frames.push_back(eq);
        }
        if(arm==Arm::g1) {
            const auto g0=control.evaluate(Arm::g0,{}).value, g1=control.global();
            motion << k << ',' << g0.dx << ',' << g0.dy << ',' << (in.motion.frames[k].status==MotionStatus::rejected)
                   << ',' << g1.dx << ',' << g1.dy << ',' << control.valid() << ',' << control.available()
                   << ',' << std::hypot(double(g1.dx)-g0.dx,double(g1.dy)-g0.dy) << '\n';
        }
    }
    diag.close(); if(!diag) throw std::runtime_error("cannot write/close tracking diagnostics");
    write_mot_file(mot_path,output.finish());
    if(equivalent) write_motion_records(equivalent,scm);
    if(arm==Arm::g1) write_text_file(std::string(diag_path)+".motion.csv",motion.str());
    std::cout << "gmc_flow_offline arm=" << name(arm) << " frames=" << in.detections.frames.size() << '\n';
}
double quantile(std::vector<double> v,double p) {
    if(v.empty()) throw std::runtime_error("empty GT residual sample");
    std::sort(v.begin(),v.end()); const double pos=p*double(v.size()-1);
    const auto lo=static_cast<std::size_t>(std::floor(pos)), hi=static_cast<std::size_t>(std::ceil(pos));
    return v[lo]+(v[hi]-v[lo])*(pos-double(lo));
}
void stratify(const Inputs& in,const char* path) {
    std::ofstream csv(path);
    csv << "frame,m_k\n" << std::setprecision(17);
    std::vector<double> scores;
    for(std::size_t k=1;k<in.flow.frames.size();++k) {
        const auto control=in.control(k);
        std::vector<double> magnitudes;
        for(int r=0;r<in.flow.rows;++r) for(int c=0;c<in.flow.cols;++c) if(!control.masked(c,r)) {
            const auto v=in.flow.frames[k].cells[static_cast<std::size_t>(r)*in.flow.cols+c];
            magnitudes.push_back(std::hypot(double(v.dx),double(v.dy)));
        }
        const double m=quantile(std::move(magnitudes),0.5)*1920.0/in.flow.width;
        scores.push_back(m); csv << k << ',' << m << '\n';
    }
    csv.close();
    if(!csv) throw std::runtime_error("cannot write stratification CSV");
    std::cout << std::setprecision(17) << "camera_motion_score=" << quantile(std::move(scores),0.5) << '\n';
}
void residual(const Inputs& in,const char* gt_path,const char* output) {
    // The only GT reader in this experiment, implementing PLAN §5.3.
    std::ifstream gt(gt_path); if(!gt) throw std::runtime_error("cannot open residual GT");
    std::map<std::int64_t,std::map<int,Box>> frames;
    std::string line;
    while(std::getline(gt,line)) {
        std::replace(line.begin(),line.end(),',',' '); std::istringstream row(line);
        std::int64_t f; int id,consider,cls; float x,y,w,h,visibility;
        if(!(row>>f>>id>>x>>y>>w>>h>>consider>>cls>>visibility)) throw std::runtime_error("malformed residual GT row");
        if(consider==1 && cls==1 && visibility>=0.5f) frames[f-1][id]={x,y,x+w,y+h};
    }
    if(gt.bad()) throw std::runtime_error("cannot read residual GT");
    std::vector<FrameControl> controls; controls.reserve(in.flow.frames.size());
    for(std::size_t k=0;k<in.flow.frames.size();++k) controls.push_back(in.control(k));
    struct Samples { std::vector<double> e1,e2; std::size_t fallback=0; };
    std::array<Samples,arms.size()> samples;
    auto center=[](const Box& b) {return std::array<double,2>{(double(b[0])+b[2])*0.5,(double(b[1])+b[3])*0.5};};
    for(std::size_t k=2;k<in.flow.frames.size();++k) {
        const auto f=static_cast<std::int64_t>(k);
        const auto current=frames.find(f), previous=frames.find(f-1), older=frames.find(f-2);
        if(current==frames.end() || previous==frames.end() || older==frames.end()) continue;
        for(const auto& [id,box]:current->second) {
            const auto p=previous->second.find(id), o=older->second.find(id);
            if(p==previous->second.end() || o==older->second.end()) continue;
            const auto c=center(box), b=center(p->second), a=center(o->second);
            const double height=double(p->second[3])-p->second[1];
            for(std::size_t i=0;i<arms.size();++i) {
                const auto t=controls[k].evaluate(arms[i],p->second), old=controls[k-1].evaluate(arms[i],o->second);
                const double ex=(c[0]-b[0])-t.value.dx, ey=(c[1]-b[1])-t.value.dy;
                const double px=(b[0]-a[0])-old.value.dx, py=(b[1]-a[1])-old.value.dy;
                samples[i].e1.push_back(std::hypot(ex,ey)/height);
                samples[i].e2.push_back(std::hypot(ex-px,ey-py)/height);
                samples[i].fallback+=t.fallback;
            }
        }
    }
    std::ostringstream csv;
    csv << "arm,samples,e2_median,e2_p90,e1_median,e1_p90,fallback_count,fallback_fraction\n" << std::setprecision(12);
    for(std::size_t i=0;i<arms.size();++i) {
        const auto& s=samples[i];
        csv << name(arms[i]) << ',' << s.e1.size() << ',' << quantile(s.e2,0.5) << ',' << quantile(s.e2,0.9)
            << ',' << quantile(s.e1,0.5) << ',' << quantile(s.e1,0.9) << ',' << s.fallback << ','
            << double(s.fallback)/double(s.e1.size()) << '\n';
    }
    write_text_file(output,csv.str());
    std::cout << "GT residuals written to " << output << '\n';
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("expected track or residual subcommand");
        const std::string mode=argv[1];
        if(mode=="track" && (argc==8 || argc==9)) {
            const Inputs in(argv); track(in,parse_arm(argv[5]),argv[6],argv[7],argc==9?argv[8]:nullptr);
        } else if(mode=="stratify" && argc==6) {
            const Inputs in(argv); stratify(in,argv[5]);
        } else if(mode=="residual" && argc==7) {
            const Inputs in(argv); residual(in,argv[5],argv[6]);
        } else throw std::runtime_error("usage: gmc_flow_offline track SCDR SCMO SCFL ARM OUT.txt DIAG.csv [EQUIV.scmo]\n"
                                         "       gmc_flow_offline stratify SCDR SCMO SCFL OUT.csv\n"
                                         "       gmc_flow_offline residual SCDR SCMO SCFL GT.txt OUT.csv");
        return 0;
    } catch(const std::exception& e) { std::cerr << "gmc_flow_offline: " << e.what() << '\n'; return 1; }
}

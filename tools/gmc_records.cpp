#include "saccade/flow_config.hpp"
#include "saccade/gmc_flow.hpp"
#include "saccade/mot_output.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::invalid_argument("usage: gmc_records SCDR SCFL OUT.scmo");
        const auto detections=saccade::read_detection_records(argv[1]);
        const auto flow=saccade::flow::read_records(argv[2]);
        const auto key=saccade::flow_image_key(flow.width,flow.height);
        if(detections.frames.size()!=flow.frames.size()) throw std::runtime_error("SCDR/SCFL frame counts differ");
        saccade::MotionRecords records{key,detections.epoch,{}};
        for(std::size_t k=0;k<flow.frames.size();++k) {
            const saccade::gmc_flow::FrameControl control(flow.frames[k],flow.cols,flow.rows,key.width,
                k?&detections.frames[k-1]:nullptr,{},key.downscale);
            saccade::MotionContext m;
            m.epoch=records.epoch; m.frame=static_cast<std::int64_t>(k); m.reference_frame=m.frame-1;
            m.status=k==0?saccade::MotionStatus::uninitialized:
                (control.valid()?saccade::MotionStatus::valid:saccade::MotionStatus::rejected);
            if(m.status==saccade::MotionStatus::valid) {
                m.confidence=1; m.tx=float(key.downscale)*control.global().dx; m.ty=float(key.downscale)*control.global().dy;
            }
            records.frames.push_back(m);
        }
        saccade::write_motion_records(argv[3],records);
        std::cout << "gmc_records profile=" << saccade::kFlowMotionProfile << " frames=" << records.frames.size()
                  << " s=" << key.downscale << '\n';
        return 0;
    } catch(const std::exception& e) { std::cerr << "gmc_records: " << e.what() << '\n'; return 1; }
}

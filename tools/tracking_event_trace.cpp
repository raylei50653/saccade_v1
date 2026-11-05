// CPU-only, observational trace. Every packet is checked against the stock reference.
#include "saccade/file_io.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/tracking.hpp"
#include "saccade/tracking_math.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <stdexcept>

namespace {
using namespace saccade;
template<class T> void box(std::ostream& o, const T& b) {
    o << '[' << b[0] << ',' << b[1] << ',' << b[2] << ',' << b[3] << ']';
}
void tracks(std::ostream& o, const std::vector<Track>& ts) {
    o << '[';
    bool first = true;
    for (const auto& t : ts) {
        if (!first) o << ',';
        first = false;
        float b[4]; tracking_math::state_to_box(t.x.data(), b);
        o << "{\"uid\":" << t.uid << ",\"output_id\":" << t.output_id << ",\"box\":";
        box(o,b);
        o << ",\"age\":" << t.age << ",\"streak\":" << t.hit_streak
          << ",\"lifecycle\":" << unsigned(t.lifecycle) << ",\"birth_frame\":" << t.birth_frame + 1
          << ",\"last_observed_frame\":" << t.last_observed_frame + 1 << '}';
    }
    o << ']';
}
void ok(Error e) { if (e != Error::none) throw std::runtime_error("tracker stage failed"); }
}
int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: tracking_event_trace DET.scdr MOTION.scmo NEW_OUTPUT_DIR");
        const std::filesystem::path out(argv[3]);
        if (std::filesystem::exists(out)) throw std::runtime_error("output already exists");
        const auto ds = saccade::read_detection_records(argv[1]);
        const auto ms = saccade::read_motion_records(argv[2]);
        if (ds.frames.empty() || ds.frames.size()!=ms.frames.size() || ds.epoch!=ms.epoch)
            throw std::runtime_error("input identity mismatch");
        if (!out.parent_path().empty()) std::filesystem::create_directories(out.parent_path());
        if (!std::filesystem::create_directory(out)) throw std::runtime_error("output already exists");
        std::ofstream f(out/"trace.jsonl");
        if (!f) throw std::runtime_error("cannot open trace output");
        f.imbue(std::locale::classic()); f << std::setprecision(9);
        const saccade::TrackerConfig config;
        auto state=saccade::initial_track_state(ds.epoch);
        saccade::TrackerReference oracle(ds.epoch,config);
        saccade::SequenceOutput mot;
        for (std::size_t k=0;k<ds.frames.size();++k) {
            const auto& m=ms.frames[k];
            auto pred=saccade::predict_tracks(state,m,static_cast<saccade::FrameId>(k),config); ok(pred.error);
            auto batch=saccade::assemble_detection_batch(ds.frames[k],pred,config); ok(batch.error);
            f << "{\"frame\":" << k+1 << ",\"motion\":{\"tx\":" << m.tx << ",\"ty\":" << m.ty
              << ",\"status\":" << unsigned(m.status) << "},\"before\":";
            tracks(f,state.tracks); f << ",\"predicted\":"; tracks(f,pred.tracks);
            f << ",\"candidates\":[";
            for (std::size_t j=0;j<ds.frames[k].header.count;++j) {
                const auto& d=ds.frames[k].rows[j]; if(j) f << ',';
                f << "{\"id\":" << d.candidate_id << ",\"box\":[" << d.x1 << ',' << d.y1 << ',' << d.x2 << ',' << d.y2
                  << "],\"score\":" << d.score << ",\"flags\":" << d.flags << '}';
            }
            f << "],\"batch\":[";
            for(std::size_t j=0;j<batch.rows.size();++j) {
                const auto& d=batch.rows[j]; if(j) f << ',';
                f << "{\"id\":" << d.candidate_id << ",\"box\":"; box(f,d.xyxy);
                f << ",\"score\":" << d.observed_score << ",\"association_score\":" << d.association_score
                  << ",\"can_birth\":" << d.can_birth << ",\"can_continue\":" << d.can_continue << '}';
            }
            f << "],\"cost_calls\":["; bool first=true;
            const saccade::AssociationCostHook hook=[&](const auto& t,const auto& d,float overlap) {
                const float cost=saccade::tracking_math::association_cost(overlap,t.x[3],d.xyxy[3]-d.xyxy[1],
                    config.assoc.lambda,config.assoc.stability_cost_w);
                if(!first) f << ',';
                first=false;
                f << "{\"uid\":" << t.uid << ",\"candidate\":" << d.candidate_id
                  << ",\"iou\":" << overlap << ",\"cost\":" << cost << '}';
                return cost;
            };
            auto u=saccade::associate_and_update(pred,batch,m,config,hook); ok(u.error);
            saccade::ResultPacket packet;
            packet.epoch=ds.epoch; packet.frame=static_cast<saccade::FrameId>(k);
            packet.committed_frame=u.next.committed_frame; packet.tracks=std::move(u.rows);
            if(saccade::encode_packet(packet)!=saccade::encode_packet(oracle.step(ds.frames[k],m)))
                throw std::runtime_error("instrumented packet differs at frame "+std::to_string(k));
            mot.add(packet); if(mot.failed()) throw std::runtime_error("MOT output failed");
            f << "],\"packet_rows\":[";
            for(std::size_t j=0;j<packet.tracks.size();++j) {
                const auto& r=packet.tracks[j]; if(j) f << ',';
                f << "{\"uid\":" << r.uid << ",\"output_id\":" << r.output_id << ",\"box\":";
                box(f,r.xyxy); f << '}';
            }
            f << "],\"after\":"; tracks(f,u.next.tracks); f << "}\n";
            state=std::move(u.next);
        }
        f.close(); if(!f) throw std::runtime_error("trace write failed");
        saccade::write_mot_file(out/"MOT17-10-SDP.txt",mot.finish());
        saccade::write_text_file(out/"tracker-config.txt",saccade::canonical_tracker_config(config));
        std::cout << "packet comparisons=" << ds.frames.size() << " PASS\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
}

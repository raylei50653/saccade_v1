#include "saccade/tracking_events.hpp"
#include "saccade/tracking_math.hpp"

#include <algorithm>
#include <bit>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace saccade;
int checks = 0;
void check(bool value, const std::string& message) { ++checks; if (!value) throw std::runtime_error(message); }
void rejected(const std::function<void()>& f, const std::string& message) {
    bool bad=false; try { f(); } catch(const std::invalid_argument&) { bad=true; } check(bad,message);
}
constexpr Epoch epoch=17;
DetectionCandidate candidate(std::uint32_t id=5,float x=0,float score=.8f,std::uint32_t flags=kDetectionKept) {
    return {x,0,x+10,20,score,0,id,flags};
}
DetectionCandidates candidates(FrameId frame,const std::vector<DetectionCandidate>& rows={}) {
    DetectionCandidates c{}; c.header.epoch=epoch; c.header.frame=static_cast<std::uint64_t>(frame);
    c.header.schema=kPostprocessSchema; c.header.capacity=kPostprocessCapacity; c.header.count=static_cast<std::uint32_t>(rows.size()); c.header.done=kPostprocessDone;
    std::copy(rows.begin(),rows.end(),c.rows.begin()); return c;
}
MotionContext motion(FrameId frame,float camera=0) {
    MotionContext m; m.epoch=epoch; m.frame=frame; m.reference_frame=frame-1;
    m.status=camera==0?MotionStatus::uninitialized:MotionStatus::valid; m.tx=camera; return m;
}
bool same_float(float a,float b) { return std::bit_cast<std::uint32_t>(a)==std::bit_cast<std::uint32_t>(b); }
bool same_track(const Track& a,const Track& b) {
    if(a.uid!=b.uid || a.output_id!=b.output_id || a.lifecycle!=b.lifecycle || a.age!=b.age || a.hit_streak!=b.hit_streak || a.class_id!=b.class_id || a.birth_frame!=b.birth_frame || a.last_observed_frame!=b.last_observed_frame || a.history_count!=b.history_count) return false;
    for(std::size_t i=0;i<a.x.size();++i) if(!same_float(a.x[i],b.x[i])) return false;
    for(std::size_t i=0;i<a.P.size();++i) if(!same_float(a.P[i],b.P[i])) return false;
    for(std::size_t i=0;i<a.history.size();++i) if(!same_float(a.history[i].cx,b.history[i].cx)||!same_float(a.history[i].cy,b.history[i].cy)||!same_float(a.history[i].h,b.history[i].h)) return false;
    return same_float(a.score_sum,b.score_sum)&&same_float(a.observed_score,b.observed_score)&&same_float(a.association_score,b.association_score)&&same_float(a.h_ema,b.h_ema);
}
void same_state(const TrackState& a,const TrackState& b) {
    check(a.epoch==b.epoch&&a.committed_frame==b.committed_frame&&a.next_uid==b.next_uid&&a.tracks.size()==b.tracks.size(),"observer full-state identity/count");
    for(std::size_t i=0;i<a.tracks.size();++i) check(same_track(a.tracks[i],b.tracks[i]),"observer every state field bits including unused history");
}
struct Run {
    TrackerConfig config;
    TrackingEventRecorder recorder;
    TrackerReference control,observed;
    std::vector<std::uint8_t> packets;
    explicit Run(const TrackerConfig& c={}):config(c),control(epoch,c),observed(epoch,c,&recorder,9) {}
    ResultPacket step(const DetectionCandidates& input,const MotionContext& m) {
        const auto a=control.step(input,m),b=observed.step(input,m);
        check(encode_packet(a)==encode_packet(b),"observer ON/OFF full ResultPacket bytes");
        same_state(control.committed(),observed.committed());
        const auto bytes=encode_packet(b); packets.insert(packets.end(),bytes.begin(),bytes.end()); return b;
    }
    void valid() {
        const auto v=validate_tracking_events(recorder.events,config);
        check(v.valid,"valid actual decision trace at "+std::to_string(v.event_index)+": "+v.message);
        const auto binding=validate_tracking_event_packets(recorder.events,packets);
        check(binding.valid,"actual events bind every ResultPacket byte at "+std::to_string(binding.event_index)+": "+binding.message);
    }
};
std::size_t index_of(const std::vector<TrackingEvent>& events,TrackingEventKind kind) {
    for(std::size_t i=0;i<events.size();++i) if(events[i].kind==kind) return i;
    throw std::runtime_error("fixture missing event");
}
void mutation_rejected(const std::vector<TrackingEvent>& input,const TrackerConfig& c,const std::function<void(std::vector<TrackingEvent>&)>& mutation,const std::string& message) {
    auto events=input; mutation(events); const auto v=validate_tracking_events(events,c); check(!v.valid,message);
}
std::vector<TrackingEvent> lifecycle_trace() {
    TrackerConfig config; config.life.track_buffer=2; Run run(config);
    for(FrameId f=0;f<3;++f) run.step(candidates(f,{candidate()}),motion(f));
    for(FrameId f=3;f<6;++f) run.step(candidates(f),motion(f));
    run.step(candidates(6,{candidate(8)}),motion(6)); run.step(candidates(7),motion(7));
    run.valid();
    check(run.control.committed().tracks.empty(),"expired confirmed and unmatched tentative removed");
    for(auto kind:{TrackingEventKind::predict,TrackingEventKind::associate,TrackingEventKind::confirm,TrackingEventKind::birth,TrackingEventKind::lost,TrackingEventKind::remove,TrackingEventKind::emit}) (void)index_of(run.recorder.events,kind);
    return run.recorder.events;
}
void model_tests() {
    auto events=lifecycle_trace(); TrackerConfig config; config.life.track_buffer=2;
    mutation_rejected(events,config,[](auto& e){ e.erase(e.begin()+static_cast<std::ptrdiff_t>(index_of(e,TrackingEventKind::predict))); },"missing predict rejected");
    mutation_rejected(events,config,[](auto& e){ e.insert(e.begin()+1,e.front()); },"duplicate begin rejected");
    mutation_rejected(events,config,[](auto& e){ e.pop_back(); },"missing terminal rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::frame_commit)].count=1; },"commit count rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::birth)].detection.can_birth=false; },"continuation-only birth rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::birth)].after.uid=99; },"nonmonotonic UID rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::associate)].association_stage=1; },"tentative association wrong stage rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::associate)].after.hit_streak=99; },"wrong associate state rejected");
    mutation_rejected(events,config,[](auto& e){ e.erase(e.begin()+static_cast<std::ptrdiff_t>(index_of(e,TrackingEventKind::confirm))); },"missing confirm rejected");
    mutation_rejected(events,config,[](auto& e){ auto i=index_of(e,TrackingEventKind::confirm); e.insert(e.begin()+static_cast<std::ptrdiff_t>(i),e[i]); },"duplicate confirmation rejected");
    mutation_rejected(events,config,[](auto& e){ e.erase(e.begin()+static_cast<std::ptrdiff_t>(index_of(e,TrackingEventKind::emit))); },"missing emit rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::emit)].row.output_id=7; },"wrong output identity rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::lost)].after.hit_streak=1; },"illegal lost transition rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::remove)].reason=TrackingEventReason::unmatched; },"wrong expiry removal reason rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::birth)].phase=TrackingEventPhase::update; },"wrong event phase rejected");
    mutation_rejected(events,config,[](auto& e){ e[index_of(e,TrackingEventKind::candidate)].count=30; e[index_of(e,TrackingEventKind::batch_ready)].count=2; },"missing candidate rejected");
    mutation_rejected(events,config,[](auto& e){ e[1].frame=3; },"wrong event frame rejected");
    check(!validate_tracking_events({},config).valid,"empty trace rejected");
    TrackerConfig immediate; immediate.life.confirm_streak=1; Run birth(immediate);
    birth.step(candidates(0,{candidate()}),motion(0)); birth.valid();
    check(index_of(birth.recorder.events,TrackingEventKind::confirm)==index_of(birth.recorder.events,TrackingEventKind::birth)+1,"same-frame birth confirmation observed");
    Run empty; empty.step(candidates(0),motion(0)); empty.valid();
    check(empty.recorder.events.size()==3,"successful empty frame has begin/batch/commit");
    auto combined=empty.recorder.events; auto other=combined; for(auto& e:other) ++e.source;
    combined.insert(combined.end(),other.begin(),other.end()); check(validate_tracking_events(combined).valid,"independent source sessions");
    for(auto& e:other) { --e.source; ++e.epoch; }
    combined=empty.recorder.events; combined.insert(combined.end(),other.begin(),other.end()); check(validate_tracking_events(combined).valid,"new epoch resets lifecycle");
}
void bridge_tests() {
    Run run;
    for(FrameId f=0;f<10;++f) {
        std::vector<DetectionCandidate> rows;
        if(f<=3||f>=6) rows.push_back(candidate(5));
        const float camera=f>=4?100.0f:0.0f;
        rows.push_back(candidate(2,200+camera+static_cast<float>(f)));
        rows.push_back(candidate(9,400+camera-static_cast<float>(f)));
        run.step(candidates(f,rows),motion(f,f==4?100.0f:0.0f));
    }
    run.valid(); const auto i=index_of(run.recorder.events,TrackingEventKind::relink); const auto& relink=run.recorder.events[i];
    check(relink.after.uid==4&&relink.after.output_id==2&&relink.related_uid==2,"Bridge keeps receiver UID and transfers donor output identity");
    mutation_rejected(run.recorder.events,{},[&](auto& e){ e[i].related_uid=e[i].before.uid; },"Bridge self donor rejected");
    mutation_rejected(run.recorder.events,{},[&](auto& e){ e[i].after.output_id=e[i].before.uid; },"Bridge missing identity transfer rejected");
    auto disabled=TrackerConfig{}; disabled.bridge.enabled=false;
    check(!validate_tracking_events(run.recorder.events,disabled).valid,"disabled Bridge trace rejected");
}
void failure_tests() {
    for(int mode=0;mode<3;++mode) {
        Run run; run.step(candidates(0,{candidate()}),motion(0));
        auto input=candidates(1,{candidate()}); auto m=motion(1);
        if(mode==0) ++input.header.epoch;
        if(mode==1) input.rows[0].score=std::numeric_limits<float>::quiet_NaN();
        if(mode==2) { m.status=MotionStatus::valid; m.tx=std::numeric_limits<float>::infinity(); }
        const auto packet=run.step(input,m); check(packet.status==ResultStatus::failed,"real protocol/numeric failure");
        run.step(candidates(2,{candidate()}),motion(2)); run.valid();
        const auto i=index_of(run.recorder.events,TrackingEventKind::frame_skipped);
        mutation_rejected(run.recorder.events,{},[&](auto& e){ e[i].kind=TrackingEventKind::frame_commit; },"commit after sticky failure rejected");
        mutation_rejected(run.recorder.events,{},[&](auto& e){ e[i].error=Error::compute; },"wrong sticky failure code rejected");
    }
    TrackerConfig config; config.life.confirm_streak=1; Run overflow(config);
    std::vector<DetectionCandidate> rows; for(std::uint32_t id=0;id<257;++id) rows.push_back(candidate(id,static_cast<float>(id)*30));
    const auto packet=overflow.step(candidates(0,rows),motion(0));
    check(packet.status==ResultStatus::failed&&packet.fail_code==Error::overflow,"mid-update packet overflow failure");
    check(overflow.control.committed().tracks.empty()&&overflow.control.committed().next_uid==1,"failed frame rolls back births and next UID");
    overflow.step(candidates(1),motion(1)); overflow.valid();
    check(index_of(overflow.recorder.events,TrackingEventKind::birth)<index_of(overflow.recorder.events,TrackingEventKind::frame_fail),"failure retains actual uncommitted intermediate events");
}
void packet_binding_tests() {
    TrackerConfig config; config.life.confirm_streak=1;
    Run run(config); run.step(candidates(0,{candidate()}),motion(0));
    auto broken=candidates(1,{candidate()}); ++broken.header.epoch;
    run.step(broken,motion(1)); run.step(candidates(2),motion(2)); run.valid();
    auto packets=run.packets; packets[64]=1;
    check(!validate_tracking_event_packets(run.recorder.events,packets).valid,"reserved packet byte corruption rejected");
    packets=run.packets; packets[packet_header_bytes+packet_row_bytes]=1;
    check(!validate_tracking_event_packets(run.recorder.events,packets).valid,"unused packet row corruption rejected");
    packets=run.packets; packets[packet_header_bytes+8]=7;
    check(!validate_tracking_event_packets(run.recorder.events,packets).valid,"packet output identity corruption rejected");
    packets=run.packets; packets[packet_bytes+40]=99;
    check(!validate_tracking_event_packets(run.recorder.events,packets).valid,"FAILED first failure frame mismatch rejected");
    packets=run.packets; packets[2*packet_bytes+60]=16;
    check(!validate_tracking_event_packets(run.recorder.events,packets).valid,"SKIPPED sticky error mismatch rejected");
    packets=run.packets; packets.pop_back();
    check(!validate_tracking_event_packets(run.recorder.events,packets).valid,"partial packet rejected");
    check(!validate_tracking_event_packets(run.recorder.events,{}).valid,"empty packet stream rejected");
    check(!validate_tracking_event_packets({},run.packets).valid,"empty event stream packet binding rejected");
    check(!validate_tracking_event_packets(run.recorder.events,std::span(run.packets).first(packet_bytes)).valid,"fewer packets than event frames rejected");
    auto events=run.recorder.events; events.resize(index_of(events,TrackingEventKind::frame_fail)+1);
    check(!validate_tracking_event_packets(events,run.packets).valid,"more packets than event frames rejected");
    events=run.recorder.events; events.pop_back();
    check(!validate_tracking_event_packets(events,run.packets).valid,"missing event terminal rejected by packet binding");
    events=run.recorder.events; ++events.back().source;
    check(!validate_tracking_event_packets(events,run.packets).valid,"multiple source packet binding rejected");
    events=run.recorder.events; ++events.back().epoch;
    check(!validate_tracking_event_packets(events,run.packets).valid,"multiple epoch packet binding rejected");
    events=run.recorder.events; events.back().error=Error::compute;
    check(!validate_tracking_event_packets(events,run.packets).valid,"event sticky failure mismatch rejected by packet binding");
}
void codec_tests() {
    const auto events=lifecycle_trace(); const auto bytes=encode_tracking_events(events); const auto decoded=decode_tracking_events(bytes);
    check(compare_tracking_events(events,decoded).equal,"stable codec round trip"); check(encode_tracking_events(decoded)==bytes,"canonical encoding round trip");
    check(tracking_events_hash(events)==tracking_events_hash(decoded)&&tracking_events_hash(events).size()==64,"stable SHA256");
    auto unused=events; const auto i=index_of(unused,TrackingEventKind::birth);
    unused[i].after.history[3].cx=std::numeric_limits<float>::quiet_NaN(); unused[i].before.x[0]=123; unused[i].row.uid=999; unused[i].count=77;
    check(encode_tracking_events(unused)==bytes,"inactive fields/history excluded from behavior encoding");
    auto changed=events; changed[i].after.score_sum=0.1f; const auto diff=compare_tracking_events(events,changed);
    check(!diff.equal&&diff.event_index==i&&diff.field=="after.score_sum.fp32_bits","first semantic field divergence without reorder");
    check(tracking_events_hash(changed)!=tracking_events_hash(events),"changed effective payload changes hash");
    changed=events; changed[i].after.x[4]=-0.0f;
    check(!compare_tracking_events(events,changed).equal,"signed zero behavior divergence preserved");
    changed=events; changed.pop_back(); check(!compare_tracking_events(events,changed).equal,"event count divergence preserved");
    for(std::size_t n:{std::size_t{0},std::size_t{7},std::size_t{15},bytes.size()-1}) rejected([&]{ (void)decode_tracking_events(std::span(bytes).first(n)); },"truncated event codec rejected");
    // Corrupt a candidate's canonical bool and an enum at exact wire offsets.
    std::size_t candidate_record=16;
    for(std::size_t event_index=0;event_index<index_of(events,TrackingEventKind::candidate);++event_index) {
        const auto length=static_cast<std::uint32_t>(bytes[candidate_record]) | (static_cast<std::uint32_t>(bytes[candidate_record+1])<<8) | (static_cast<std::uint32_t>(bytes[candidate_record+2])<<16) | (static_cast<std::uint32_t>(bytes[candidate_record+3])<<24);
        candidate_record+=4+length;
    }
    auto invalid_bool=bytes; invalid_bool[candidate_record+4+36+32]=2;
    rejected([&]{ (void)decode_tracking_events(invalid_bool); },"noncanonical bool rejected");
    auto invalid_kind=bytes; invalid_kind[16+4+24]=255;
    rejected([&]{ (void)decode_tracking_events(invalid_kind); },"unknown wire kind rejected");
    auto bad=bytes; bad.push_back(0); rejected([&]{ (void)decode_tracking_events(bad); },"trailing bytes rejected");
    bad=bytes; bad[5]=2; rejected([&]{ (void)decode_tracking_events(bad); },"unknown schema rejected");
    changed=events; changed[i].after.history_count=5; rejected([&]{ (void)encode_tracking_events(changed); },"history overflow rejected");
    changed=events; changed[i].after.score_sum=std::numeric_limits<float>::infinity(); rejected([&]{ (void)encode_tracking_events(changed); },"nonfinite payload rejected");
    changed=events; changed[0].kind=static_cast<TrackingEventKind>(77); rejected([&]{ (void)encode_tracking_events(changed); },"unknown kind rejected");
}
} // namespace
int main() {
    try { model_tests(); bridge_tests(); failure_tests(); packet_binding_tests(); codec_tests(); std::cout<<checks<<" checks passed\n"; return 0; }
    catch(const std::exception& e) { std::cerr<<"tracking events failure after "<<checks<<" checks: "<<e.what()<<'\n'; return 1; }
}

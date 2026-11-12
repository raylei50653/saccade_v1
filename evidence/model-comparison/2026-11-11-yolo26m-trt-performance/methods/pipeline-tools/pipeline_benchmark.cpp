#include "saccade/detection_session.hpp"
#include "saccade/tracking_session.hpp"
#include "saccade/file_io.hpp"
#include "saccade/mot_output.hpp"
#include "saccade/sha256.hpp"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::duration value) { return std::chrono::duration<double, std::milli>(value).count(); }
template<class T> T number(std::string_view text) {
    T value{}; const auto [end,error] = std::from_chars(text.data(),text.data()+text.size(),value);
    if (error!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("invalid number");
    return value;
}
std::string trim(std::string_view text) {
    const auto a=text.find_first_not_of(" \t\r\n"),b=text.find_last_not_of(" \t\r\n");
    return a==std::string_view::npos ? "" : std::string(text.substr(a,b-a+1));
}
std::string quote(const std::string& text) {
    std::string out="\"";
    for (const char c:text) { if (c=='\\' || c=='\"') out+='\\'; if (c=='\n') out+="\\n"; else out+=c; }
    return out+'\"';
}
double percentile(std::vector<double> values,double q) {
    std::sort(values.begin(),values.end());
    return values[std::max<std::size_t>(1,static_cast<std::size_t>(std::ceil(q*values.size())))-1];
}
}

int main(int argc,char** argv) {
    try {
        if (argc<5) throw std::invalid_argument("pipeline_benchmark detect|tracking ENGINE SEQ_DIR NEW_OUT_DIR [--frames140 --warmup20 --mode single|queued]");
        const std::string path=argv[1],engine=argv[2];
        const std::filesystem::path sequence=argv[3],out=argv[4];
        if (path!="detect" && path!="tracking") throw std::invalid_argument("path differs");
        std::size_t frames=140,warmup=20;std::string mode="single";
        for (int i=5;i<argc;i+=2) {
            if (i+1==argc) throw std::invalid_argument("incomplete option");
            const std::string flag=argv[i],value=argv[i+1];
            if (flag=="--frames") frames=number<std::size_t>(value);
            else if (flag=="--warmup") warmup=number<std::size_t>(value);
            else if (flag=="--mode") mode=value;
            else throw std::invalid_argument("unknown option");
        }
        if (frames<=warmup || warmup<1 || frames>100000 || (mode!="single" && mode!="queued"))
            throw std::invalid_argument("invalid frame/mode configuration");
        const bool queued=mode=="queued";
        std::map<std::string,std::string> info;
        std::istringstream metadata(saccade::read_text_file(sequence/"seqinfo.ini"));
        std::string line;
        while (std::getline(metadata,line)) {
            const auto equal=line.find('=');
            if (equal!=std::string::npos) info[trim(std::string_view(line).substr(0,equal))]=trim(std::string_view(line).substr(equal+1));
        }
        const int width=number<int>(info.at("imWidth")),height=number<int>(info.at("imHeight"));
        if (frames>number<std::size_t>(info.at("seqLength"))) throw std::invalid_argument("sequence too short");
        std::vector<std::string> jpegs;
        for (std::size_t frame=0;frame<frames;++frame) {
            std::ostringstream name;name<<std::setfill('0')<<std::setw(6)<<frame+1<<info.at("imExt");
            jpegs.push_back(saccade::read_text_file(sequence/info.at("imDir")/name.str()));
        }
        if (!std::filesystem::create_directory(out)) throw std::invalid_argument("output must be new");
        std::unique_ptr<saccade::DetectionSession> detection;
        std::unique_ptr<saccade::TrackingSession> tracking;
        const auto init_start=Clock::now();
        if (path=="detect") {
            saccade::DetectionSessionConfig cfg;cfg.profile=saccade::ModelProfile::Hr1280x768;
            cfg.candidate_slots=2;cfg.result_slots=4;
            cfg.preprocess.max_width=std::max(1920,width);cfg.preprocess.max_height=std::max(1080,height);
            detection=std::make_unique<saccade::DetectionSession>(engine,cfg);
        } else {
            saccade::TrackingSessionConfig cfg;cfg.profile=saccade::ModelProfile::Hr1280x768;
            cfg.original_width=width;cfg.original_height=height;
            cfg.candidate_slots=cfg.motion_slots=2;cfg.result_slots=4;
            cfg.cuda_graph=true;cfg.mode=saccade::TrackingSessionMode::Pipelined;
            tracking=std::make_unique<saccade::TrackingSession>(engine,cfg);
        }
        const double init_ms=ms(Clock::now()-init_start);
        std::vector<Clock::time_point> starts(frames),ends(frames);
        std::vector<saccade::DetectionSessionResult> detections(path=="detect"?frames:0);
        std::vector<saccade::TrackingSessionResult> packets(path=="tracking"?frames:0);
        std::size_t received=0;
        const auto pending=[&] {return detection?detection->pending():tracking->pending();};
        const auto receive=[&] {
            if (detection) {
                const auto result=detection->receive();
                const auto delivered=Clock::now();
                ends[received]=delivered;detections[received]=result;
            } else {
                const auto result=tracking->receive();
                const auto delivered=Clock::now();
                ends[received]=delivered;packets[received]=result;
            }
            ++received;
        };
        const auto enqueue=[&](std::size_t frame) {
            const auto& jpeg=jpegs[frame];
            starts[frame]=Clock::now();
            const std::span<const std::uint8_t> bytes{reinterpret_cast<const std::uint8_t*>(jpeg.data()),jpeg.size()};
            if (detection) detection->enqueue(bytes,frame,width,height);else tracking->enqueue(bytes,frame);
        };
        // Fresh tracker history and the same warmup prefix in both scheduling modes.
        for (std::size_t frame=0;frame<warmup;++frame) {enqueue(frame);receive();}
        const auto measured_start=Clock::now();
        for (std::size_t frame=warmup;frame<frames;++frame) {
            if (queued && pending()==4) receive();
            enqueue(frame);
            if (!queued) receive();
        }
        while (pending()) receive();
        const auto measured_end=ends.back();
        if (received!=frames) throw std::runtime_error("delivery coverage differs");
        std::string identity;
        if (detection) {
            detection->finish();identity=detection->identity();detection->close();
        } else {
            tracking->finish();identity=tracking->identity();tracking->close();
        }
        saccade::write_text_file(out/"identity.txt",identity);
        std::string result_hash;
        if (path=="detect") {
            saccade::DetectionRecords records{1,{}};
            for (std::size_t frame=0;frame<frames;++frame) {
                const auto& value=detections[frame];
                if (value.status!=saccade::DetectionSessionStatus::Ok || value.frame!=frame || value.epoch!=1)
                    throw std::runtime_error("non-OK detection or identity mismatch");
                records.frames.push_back(value.candidates);
            }
            saccade::write_detection_records(out/"detections.scdr",records);
            result_hash=saccade::sha256_hex(saccade::read_text_file(out/"detections.scdr"));
        } else {
            std::vector<std::uint8_t> bytes;
            for (std::size_t frame=0;frame<frames;++frame) {
                const auto& value=packets[frame];
                const auto decoded=saccade::decode_packet(value.bytes);
                if (decoded.status!=saccade::ResultStatus::ok || decoded.frame!=static_cast<saccade::FrameId>(frame) || decoded.epoch!=1)
                    throw std::runtime_error("non-OK packet or identity mismatch");
                bytes.insert(bytes.end(),value.bytes.begin(),value.bytes.end());
            }
            saccade::write_binary_file(out/"packets.bin",bytes);
            result_hash=saccade::sha256_hex({reinterpret_cast<const char*>(bytes.data()),bytes.size()});
        }
        std::ostringstream csv;csv<<"frame,warmup,delivery_ms\n"<<std::fixed<<std::setprecision(9);
        std::vector<double> samples;
        for (std::size_t frame=0;frame<frames;++frame) {
            const double elapsed=ms(ends[frame]-starts[frame]);
            csv<<frame<<','<<(frame<warmup?1:0)<<','<<elapsed<<'\n';
            if (frame>=warmup) samples.push_back(elapsed);
        }
        saccade::write_text_file(out/"delivery.csv",csv.str());
        const double interval=ms(measured_end-measured_start)/samples.size();
        std::ostringstream summary;summary<<std::fixed<<std::setprecision(9)
            <<"{\n  \"status\": \"PASS\",\n  \"path\": "<<quote(path)<<",\n  \"mode\": "<<quote(mode)
            <<",\n  \"frames\": "<<frames<<",\n  \"warmup\": "<<warmup<<",\n  \"samples\": "<<samples.size()
            <<",\n  \"ring_slots\": 4,\n  \"candidate_slots\": 2,\n  \"motion_slots\": "<<(tracking?2:0)
            <<",\n  \"result_slots\": 4,\n  \"tracking_graph\": "<<(tracking?"true":"false")
            <<",\n  \"init_ms\": "<<init_ms<<",\n  \"mean_delivery_ms\": "<<std::accumulate(samples.begin(),samples.end(),0.0)/samples.size()
            <<",\n  \"p50_delivery_ms\": "<<percentile(samples,.5)<<",\n  \"p95_delivery_ms\": "<<percentile(samples,.95)
            <<",\n  \"p99_delivery_ms\": "<<percentile(samples,.99)<<",\n  \"interval_ms_per_frame\": "<<interval
            <<",\n  \"throughput_fps\": "<<1000.0/interval<<",\n  \"result_sha256\": "<<quote(result_hash)
            <<",\n  \"engine_sha256\": "<<quote(saccade::sha256_hex(saccade::read_text_file(engine)))
            <<",\n  \"timing\": \"JPEGs preloaded; start before enqueue; end immediately after receive before caller copy/check; initialization/warmup/finalization/output excluded\"\n}\n";
        saccade::write_text_file(out/"summary.json",summary.str());
        std::cout<<summary.str();return 0;
    } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}

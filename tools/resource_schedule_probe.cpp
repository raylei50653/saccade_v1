#include "saccade/tracking_session.hpp"
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"
#include "saccade/mot_output.hpp"
#include <nvtx3/nvToolsExt.h>
#include <sys/resource.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Range {
    explicit Range(const std::string& name) { nvtxRangePushA(name.c_str()); }
    ~Range() { nvtxRangePop(); }
};
double us(Clock::duration d) { return std::chrono::duration<double, std::micro>(d).count(); }
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
}

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::invalid_argument("usage: resource_schedule_probe ENGINE SEQ_DIR NEW_OUT_DIR [--frames 120] [--warmup 20] [--result-slots 4] [--no-cuda-graph] [--only single|queued|serial]");
        unsigned frames = 120, warmup = 20, result_slots = 4;
        bool graph = true;
        std::string only;
        for (int i = 4; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--no-cuda-graph") graph = false;
            else if (i+1 < argc && a == "--frames") frames = std::stoul(argv[++i]);
            else if (i+1 < argc && a == "--warmup") warmup = std::stoul(argv[++i]);
            else if (i+1 < argc && a == "--result-slots") result_slots = std::stoul(argv[++i]);
            else if (i+1 < argc && a == "--only") only = argv[++i];
            else throw std::invalid_argument("invalid option");
        }
        if (!frames || !warmup || frames > 100000 || warmup > 100000 || !result_slots || result_slots > 64 ||
            (!only.empty() && only != "single" && only != "queued" && only != "serial"))
            throw std::invalid_argument("positive bounded frames/warmup and valid mode required");
        const std::filesystem::path seq = argv[2], out = argv[3];
        std::istringstream info(saccade::read_text_file(seq / "seqinfo.ini"));
        std::map<std::string, std::string> fields;
        std::string line;
        while (std::getline(info, line)) {
            const auto eq = line.find('=');
            if (eq != std::string::npos) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                fields[line.substr(0, eq)] = line.substr(eq+1);
            }
        }
        if (frames + warmup > std::stoul(fields.at("seqLength"))) throw std::invalid_argument("not enough images");
        if (!std::filesystem::create_directory(out)) throw std::invalid_argument("output directory must be new");
        std::vector<std::string> images;
        std::ostringstream hashes;
        hashes << saccade::sha256_hex(saccade::read_text_file(argv[1])) << "  " << argv[1] << '\n';
        std::ostringstream input_times;
        input_times << "frame,cached_file_read_us,jpeg_bytes\n";
        for (unsigned i=0; i<frames+warmup; ++i) {
            std::ostringstream name;
            name << std::setw(6) << std::setfill('0') << i+1 << fields.at("imExt");
            const auto p = seq / fields.at("imDir") / name.str();
            const auto before_read = Clock::now();
            images.push_back(saccade::read_text_file(p));
            input_times << i << ',' << us(Clock::now()-before_read) << ',' << images.back().size() << '\n';
            hashes << saccade::sha256_hex(images.back()) << "  " << p.string() << '\n';
        }
        saccade::write_text_file(out / "inputs.sha256", hashes.str());
        saccade::write_text_file(out / "input-io.csv", input_times.str());
        std::vector<std::uint8_t> reference;
        std::ostringstream samples, runs;
        samples << "mode,frame,enqueue_us,receive_us,enqueue_to_receive_us\n" << std::fixed << std::setprecision(6);
        runs << "mode,constructor_us,warmup_us,measured_wall_us,frames,device_used_before_B,device_used_constructor_B,device_used_warm_B,device_used_after_B,process_peak_rss_KiB,packet_sha256\n" << std::fixed << std::setprecision(6);
        for (const std::string mode : {"single", "queued", "serial"}) {
            if (!only.empty() && only != mode) continue;
            Range scope("mode:" + mode);
            saccade::TrackingSessionConfig cfg;
            cfg.profile = saccade::parse_model_profile("hr-1280x768");
            cfg.original_width = std::stoi(fields.at("imWidth"));
            cfg.original_height = std::stoi(fields.at("imHeight"));
            cfg.cuda_graph = graph;
            cfg.result_slots = result_slots;
            if (mode == "serial") cfg.mode = saccade::TrackingSessionMode::Serial;
            std::size_t free_before, free_constructor, free_warm, free_after, total;
            check(cudaFree(nullptr));
            check(cudaMemGetInfo(&free_before, &total));
            const auto start_init = Clock::now();
            auto session = std::make_unique<saccade::TrackingSession>(argv[1], cfg);
            const double init = us(Clock::now()-start_init);
            check(cudaMemGetInfo(&free_constructor, &total));
            std::vector<std::uint8_t> bytes;
            bytes.reserve(images.size()*saccade::packet_bytes);
            std::vector<Clock::time_point> submitted(images.size());
            std::vector<double> enqueue_times(images.size()), receive_times(images.size()), latency(images.size());
            unsigned received = 0;
            const auto receive = [&] {
                Range r("frame:"+std::to_string(received)+":receive");
                const auto start = Clock::now();
                const auto packet = session->receive();
                receive_times[received] = us(Clock::now()-start);
                latency[received] = us(Clock::now()-submitted[received]);
                if (packet.packet.status != saccade::ResultStatus::ok || packet.packet.frame != received)
                    throw std::runtime_error("non-OK or wrong frame");
                bytes.insert(bytes.end(), packet.bytes.begin(), packet.bytes.end());
                ++received;
            };
            const auto enqueue = [&](unsigned frame) {
                Range r("frame:"+std::to_string(frame)+":enqueue");
                const auto& jpeg = images[frame];
                submitted[frame] = Clock::now();
                session->enqueue({reinterpret_cast<const std::uint8_t*>(jpeg.data()), jpeg.size()}, frame);
                enqueue_times[frame] = us(Clock::now()-submitted[frame]);
            };
            const auto start_warm = Clock::now();
            for (unsigned i=0; i<warmup; ++i) { enqueue(i); receive(); }
            const double warm = us(Clock::now()-start_warm);
            check(cudaMemGetInfo(&free_warm, &total));
            const auto start = Clock::now();
            for (unsigned i=warmup; i<images.size(); ++i) {
                if (mode != "single" && session->pending() == cfg.result_slots) receive();
                enqueue(i);
                if (mode == "single") receive();
            }
            while (session->pending()) receive();
            const double wall = us(Clock::now()-start);
            session->finish();
            check(cudaMemGetInfo(&free_after, &total));
            saccade::write_text_file(out / (mode+".identity.txt"), session->identity());
            session->close();
            if (reference.empty()) reference = bytes;
            else if (reference != bytes) throw std::runtime_error("full packet bytes differ across modes");
            saccade::write_binary_file(out / (mode+".bin"), bytes);
            const auto hash = saccade::sha256_hex({reinterpret_cast<const char*>(bytes.data()),bytes.size()});
            rusage usage{};
            if (getrusage(RUSAGE_SELF,&usage)) throw std::runtime_error("getrusage failed");
            runs << mode << ',' << init << ',' << warm << ',' << wall << ',' << frames << ',' << total-free_before << ',' << total-free_constructor << ',' << total-free_warm << ',' << total-free_after << ',' << usage.ru_maxrss << ',' << hash << '\n';
            for (unsigned i=0; i<images.size(); ++i)
                samples << mode << ',' << i << ',' << enqueue_times[i] << ',' << receive_times[i] << ',' << latency[i] << '\n';
            std::cout << mode << " ms/frame=" << wall/frames/1000 << " packet=" << hash << " PASS\n" << std::flush;
        }
        saccade::write_text_file(out / "frames.csv", samples.str());
        saccade::write_text_file(out / "runs.csv", runs.str());
        // Replay owned packets outside GPU timing to measure the CPU output API.
        saccade::SequenceOutput output;
        std::ostringstream output_times;
        output_times << "frame,decode_and_add_us\n";
        for (unsigned frame=0; frame<images.size(); ++frame) {
            saccade::PacketBytes packet;
            std::copy_n(reference.begin()+frame*saccade::packet_bytes, saccade::packet_bytes, packet.begin());
            const auto before_add = Clock::now();
            output.add(saccade::decode_packet(packet));
            output_times << frame << ',' << us(Clock::now()-before_add) << '\n';
        }
        const auto before_finish = Clock::now();
        const auto mot = output.finish();
        const double finish_us = us(Clock::now()-before_finish);
        const auto before_write = Clock::now();
        saccade::write_mot_file(out / "output.txt", mot);
        const double write_us = us(Clock::now()-before_write);
        saccade::write_text_file(out / "output-frames.csv", output_times.str());
        std::ostringstream tail;
        tail << "{\"frames\":" << images.size() << ",\"finish_us\":" << finish_us << ",\"write_us\":" << write_us
             << ",\"write_scope\":\"format and write without fsync\",\"scope\":\"owned packets replayed outside GPU timing\"}\n";
        saccade::write_text_file(out / "output-tail.json", tail.str());
        saccade::write_text_file(out / "status.json", "{\"status\":\"PASS\",\"comparison\":\"full packet bytes across selected modes\"}\n");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

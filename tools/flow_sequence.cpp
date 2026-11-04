#include "saccade/flow_cuda.hpp"
#include "saccade/preprocess.hpp"
#include "saccade/file_io.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
namespace fs=std::filesystem;
std::string trim(const std::string& s) {
    const auto b=s.find_first_not_of(" \t\r\n");
    return b==std::string::npos ? "" : s.substr(b,s.find_last_not_of(" \t\r\n")-b+1);
}
struct Sequence { std::string dir, ext; int count=0,w=0,h=0; };
Sequence metadata(const fs::path& path) {
    std::ifstream in(path/"seqinfo.ini");
    if(!in) throw std::runtime_error("cannot open seqinfo.ini");
    Sequence s; std::string line; bool section=false;
    while(std::getline(in,line)) {
        line=trim(line); if(line.empty() || line[0]==';' || line[0]=='#') continue;
        if(line[0]=='[') { section=line=="[Sequence]"; continue; }
        if(!section) continue;
        const auto eq=line.find('='); if(eq==std::string::npos) continue;
        const auto key=trim(line.substr(0,eq)), value=trim(line.substr(eq+1));
        if(key=="imDir") s.dir=value;
        else if(key=="imExt") s.ext=value;
        else if(key=="seqLength") s.count=std::stoi(value);
        else if(key=="imWidth") s.w=std::stoi(value);
        else if(key=="imHeight") s.h=std::stoi(value);
    }
    if(s.dir.empty() || s.ext.empty() || s.count<=0 || s.w<=0 || s.h<=0) throw std::runtime_error("missing seqinfo field");
    return s;
}
}
int main(int argc,char** argv) {
    fs::path output;
    try {
        if(argc<3 || argc%2!=1) throw std::runtime_error("usage: flow_sequence SEQ_DIR OUT.scfl [--perf slow|medium|fast] [--timing OUT.csv]");
        output=argv[2]; fs::path timing_path;
        saccade::flow::Perf perf=saccade::flow::Perf::slow;
        for(int i=3;i<argc;i+=2) {
            const std::string opt=argv[i], value=argv[i+1];
            if(opt=="--perf") {
                if(value=="slow") perf=saccade::flow::Perf::slow;
                else if(value=="medium") perf=saccade::flow::Perf::medium;
                else if(value=="fast") perf=saccade::flow::Perf::fast;
                else throw std::runtime_error("invalid --perf");
            } else if(opt=="--timing") timing_path=value;
            else throw std::runtime_error("unknown option");
        }
        const fs::path dir=argv[1]; const auto seq=metadata(dir);
        saccade::flow::Engine engine(seq.w,seq.h,perf);
        saccade::PreprocessConfig config; config.max_width=seq.w; config.max_height=seq.h;
        saccade::NativePreprocessor preprocess(engine.stream(),config);
        saccade::flow::Records records{seq.w,seq.h,8,seq.w/8,seq.h/8,perf,{}};
        records.frames.reserve(seq.count);
        std::ostringstream timing;
        timing << "frame,grayscale_ms,execute_ms,cell_ms\n" << std::fixed << std::setprecision(6);
        for(int k=0;k<seq.count;++k) {
            std::ostringstream name; name << std::setfill('0') << std::setw(6) << k+1 << seq.ext;
            const auto jpeg=saccade::read_text_file(dir/seq.dir/name.str());
            const auto decoded=preprocess.enqueue({reinterpret_cast<const std::uint8_t*>(jpeg.data()),jpeg.size()},k,seq.w,seq.h);
            saccade::flow::Timing t;
            records.frames.push_back(engine.push_rgb(decoded.rgb,k,t));
            timing << k << ',' << t.grayscale_ms << ',' << t.execute_ms << ',' << t.cell_ms << '\n';
        }
        preprocess.close(); engine.close();
        saccade::flow::write_records(output.string()+".tmp",records);
        fs::rename(output.string()+".tmp",output);
        if(!timing_path.empty()) saccade::write_text_file(timing_path,timing.str());
        std::cout << "flow_sequence frames=" << records.frames.size() << " cells=" << records.cols << 'x' << records.rows
                  << " perf=" << static_cast<int>(perf) << '\n';
        return 0;
    } catch(const std::exception& e) {
        if(!output.empty()) { std::error_code ec; fs::remove(output.string()+".tmp",ec); }
        std::cerr << "flow_sequence: " << e.what() << '\n'; return 1;
    }
}

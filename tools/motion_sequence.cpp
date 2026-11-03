#include "saccade/motion_cuda.hpp"
#include "saccade/preprocess.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/file_io.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace saccade;
namespace fs = std::filesystem;
void ck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    return begin == std::string::npos ? "" : s.substr(begin, s.find_last_not_of(" \t\r\n") - begin + 1);
}
int positive(const std::string& s) {
    std::size_t end = 0;
    const int value = std::stoi(s, &end);
    if (end != s.size() || value <= 0) throw std::invalid_argument("invalid seqinfo integer");
    return value;
}
struct Sequence {
    std::string directory, extension;
    int length = 0, width = 0, height = 0;
};
Sequence metadata(const fs::path& dir) {
    std::ifstream input(dir / "seqinfo.ini");
    if (!input) throw std::runtime_error("cannot open seqinfo.ini");
    Sequence seq;
    std::string line;
    bool section = false;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line[0] == ';' || line[0] == '#') continue;
        if (line[0] == '[') { section = line == "[Sequence]"; continue; }
        if (!section) continue;
        const auto equal = line.find('=');
        if (equal == std::string::npos) continue;
        const auto key = trim(line.substr(0, equal)), value = trim(line.substr(equal + 1));
        if (key == "imDir") seq.directory = value;
        else if (key == "imExt") seq.extension = value;
        else if (key == "seqLength") seq.length = positive(value);
        else if (key == "imWidth") seq.width = positive(value);
        else if (key == "imHeight") seq.height = positive(value);
    }
    if (input.bad()) throw std::runtime_error("cannot read seqinfo.ini");
    if (seq.directory.empty() || seq.extension.empty() || seq.length == 0 || seq.width == 0 || seq.height == 0)
        throw std::invalid_argument("missing seqinfo field");
    return seq;
}
struct Resources {
    cudaStream_t detect = nullptr, main = nullptr;
    MotionContext *device = nullptr, *host = nullptr;
    Resources() {
        try {
            ck(cudaStreamCreateWithFlags(&detect, cudaStreamNonBlocking), "create detect stream");
            ck(cudaStreamCreateWithFlags(&main, cudaStreamNonBlocking), "create main stream");
            ck(cudaMalloc(&device, sizeof(MotionContext)), "allocate MotionContext");
            ck(cudaMallocHost(&host, sizeof(MotionContext)), "allocate pinned MotionContext");
        } catch (...) { release(); throw; }
    }
    CleanupReport release() {
        CleanupReport report;
        for (auto stream : {detect, main}) {
            if (stream) { int pending = 1; report.release(pending, [&](int) { ck(cudaStreamSynchronize(stream), "drain stream"); }); }
        }
        report.release(device, [](auto p) { ck(cudaFree(p), "free MotionContext"); });
        report.release(host, [](auto p) { ck(cudaFreeHost(p), "free pinned MotionContext"); });
        report.release(detect, [](auto p) { ck(cudaStreamDestroy(p), "destroy detect stream"); });
        report.release(main, [](auto p) { ck(cudaStreamDestroy(p), "destroy main stream"); });
        return report;
    }
    ~Resources() { release(); }
};
void run(const fs::path& dir, const fs::path& output, std::uint64_t epoch) {
    const auto seq = metadata(dir);
    const MotionConfig config;
    const auto key = motion_image_key(seq.width, seq.height, config);
    Resources resources;
    PreprocessConfig preconfig;
    preconfig.max_width = seq.width; preconfig.max_height = seq.height;
    NativePreprocessor preprocess(resources.detect, preconfig);
    MotionCuda motion(key, config, epoch);
    MotionRecords records{key, epoch, {}};
    records.frames.reserve(seq.length);
    for (int index = 0; index < seq.length; ++index) {
        std::ostringstream name;
        name << std::setfill('0') << std::setw(6) << index + 1 << seq.extension;
        const auto jpeg = read_text_file(dir / seq.directory / name.str());
        const auto decoded = preprocess.enqueue({reinterpret_cast<const std::uint8_t*>(jpeg.data()), jpeg.size()},
                                                index, seq.width, seq.height);
        motion.enqueue_image(resources.detect, decoded.rgb, index);
        motion.enqueue_estimate(resources.main, index, resources.device);
        ck(cudaMemcpyAsync(resources.host, resources.device, sizeof(MotionContext), cudaMemcpyDeviceToHost, resources.main), "copy MotionContext");
        ck(cudaStreamSynchronize(resources.main), "wait MotionContext");
        if (!std::isfinite(resources.host->pcr) || !std::isfinite(resources.host->confidence) ||
            !std::isfinite(resources.host->tx) || !std::isfinite(resources.host->ty)) throw std::domain_error("motion NUMERIC");
        motion.enqueue_commit(resources.main, index);
        records.frames.push_back(*resources.host);
    }
    motion.close(); preprocess.close(); resources.release().throw_if_failed();
    const auto temporary = fs::path(output.string() + ".tmp");
    write_motion_records(temporary, records);
    fs::rename(temporary, output);
    std::size_t valid = 0, rejected = 0, uninitialized = 0;
    float min_tx = 0, max_tx = 0, min_ty = 0, max_ty = 0;
    for (const auto& f : records.frames) {
        valid += f.status == MotionStatus::valid; rejected += f.status == MotionStatus::rejected;
        uninitialized += f.status == MotionStatus::uninitialized;
        min_tx = std::min(min_tx, f.tx); max_tx = std::max(max_tx, f.tx);
        min_ty = std::min(min_ty, f.ty); max_ty = std::max(max_ty, f.ty);
    }
    std::cout << "motion_sequence: frames=" << records.frames.size() << " valid=" << valid << " rejected=" << rejected
              << " uninitialized=" << uninitialized << " tx=[" << min_tx << ',' << max_tx << "] ty=[" << min_ty << ',' << max_ty << "]\n";
}
}
int main(int argc, char** argv) {
    fs::path output;
    try {
        if (argc != 3 && argc != 5) throw std::invalid_argument("usage: motion_sequence SEQ_DIR OUT.scmo [--epoch N]");
        output = argv[2];
        std::uint64_t epoch = 1; // DetectionSessionConfig default, so records pair up
        if (argc == 5) {
            const std::string text = argv[4];
            if (std::string(argv[3]) != "--epoch" || text.empty() || text[0] == '-') throw std::invalid_argument("invalid epoch option");
            std::size_t end = 0;
            epoch = std::stoull(text, &end);
            if (end != text.size()) throw std::invalid_argument("invalid epoch");
        }
        run(argv[1], output, epoch);
        return 0;
    } catch (const std::exception& e) {
        if (!output.empty()) {
            std::error_code ec;
            fs::remove(fs::path(output.string() + ".tmp"), ec);
            fs::remove(output, ec);
        }
        std::cerr << "motion_sequence: " << e.what() << '\n'; return 1;
    }
}

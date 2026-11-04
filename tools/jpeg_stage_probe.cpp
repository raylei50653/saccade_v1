#include "saccade/cleanup.hpp"
#include "saccade/file_io.hpp"
#include "saccade/model_producer_cuda.hpp"
#include "saccade/postprocess_cuda.hpp"
#include "saccade/preprocess.hpp"
#include "saccade/resize_cuda.hpp"
#include "saccade/sha256.hpp"
#include "saccade/tracking_session.hpp"

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kJpegCapacity = 16 * 1024 * 1024;
constexpr int kMaxWidth = 1920, kMaxHeight = 1080;
constexpr int kOutputWidth = 1280, kOutputHeight = 768;
constexpr const char* usage = "usage: jpeg_stage_probe SEQ_DIR NEW_OUT_DIR [--frames 120] [--warmup 20]";

void ck(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc));
}
void jc(nvjpegStatus_t rc, const char* what) {
    if (rc != NVJPEG_STATUS_SUCCESS)
        throw std::runtime_error(std::string(what) + ": nvJPEG status " + std::to_string(rc));
}
void require(bool value, const char* what) { if (!value) throw std::invalid_argument(what); }
template<class T> T number(std::string_view text) {
    T value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    require(error == std::errc{} && end == text.data() + text.size(), "invalid integer");
    return value;
}
std::string trim(std::string_view text) {
    const auto a = text.find_first_not_of(" \t\r\n");
    return a == std::string_view::npos ? "" : std::string(text.substr(a, text.find_last_not_of(" \t\r\n") - a + 1));
}
std::string quote(std::string_view text) {
    std::ostringstream out;
    out << '"';
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else out << c;
    }
    return out.str() + '"';
}
double us(Clock::duration duration) { return std::chrono::duration<double, std::micro>(duration).count(); }
double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<std::size_t>(std::ceil(q * static_cast<double>(values.size())));
    return values[std::max<std::size_t>(1, rank) - 1];
}
struct Sample {
    std::size_t frame = 0, jpeg_bytes = 0;
    double f01_full_us = 0, staging_parse_us = 0, decode_call_us = 0, decode_to_rgb_ready_us = 0;
    double f02_gpu_event_envelope_us = 0, f02_enqueue_us = 0, f02_enqueue_to_done_us = 0;
};

// This deliberately reproduces NativePreprocessor's batch-1 hardware decoder,
// allocations, checks and resize. Each stage is drained before the next one.
// F01's host timer observes RGB delivery; it does not isolate decoder-engine busy time.
class Probe {
public:
    Probe() {
        try {
            ck(cudaSetDevice(0), "cudaSetDevice");
            ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create explicit detect stream");
            (void)saccade::resize_cuda_load_kernels();
            jc(nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, NVJPEG_FLAGS_DEFAULT, &handle_), "create HARDWARE decoder");
            jc(nvjpegJpegStateCreate(handle_, &state_), "create JPEG state");
            jc(nvjpegJpegStreamCreate(handle_, &parsed_), "create parsed JPEG");
            jc(nvjpegDecodeBatchedInitialize(handle_, state_, 1, 1, NVJPEG_OUTPUT_RGB), "initialize batch-1 RGB decode");
            jc(nvjpegDecodeBatchedPreAllocate(handle_, state_, 1, kMaxWidth, kMaxHeight, NVJPEG_CSS_420, NVJPEG_OUTPUT_RGB), "preallocate decoder");
            const saccade::ResizeShape capacity{kMaxWidth, kMaxHeight, kOutputWidth, kOutputHeight};
            raw_bytes_ = saccade::resize_input_bytes(capacity);
            output_elements_ = saccade::resize_output_elements(capacity);
            workspace_elements_ = saccade::resize_workspace_elements(capacity);
            ck(cudaMallocHost(&staging_, kJpegCapacity), "allocate pinned JPEG staging");
            ck(cudaMalloc(&raw_, raw_bytes_), "allocate RGB");
            ck(cudaMalloc(&images_, output_elements_ * sizeof(float)), "allocate HR images");
            ck(cudaMalloc(&workspace_, workspace_elements_ * sizeof(float)), "allocate horizontal intermediate");
            ck(cudaEventCreate(&start_), "create resize start event");
            ck(cudaEventCreate(&stop_), "create resize stop event");
            ck(cudaDeviceSynchronize(), "finish probe initialization");
        } catch (...) { release(); throw; }
    }
    ~Probe() { release(); }
    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;

    Sample measure(std::string_view jpeg, std::size_t frame, int expected_width, int expected_height) {
        // Excluded from stage timing. This is a same-process device-idle start,
        // not a reservation against unrelated processes; invoke under gpu.lock.
        ck(cudaDeviceSynchronize(), "idle before independent F01 sample");
        Sample result;
        result.frame = frame;
        result.jpeg_bytes = jpeg.size();
        const auto begin = Clock::now();
        require(jpeg.size() >= 4 && jpeg.size() <= kJpegCapacity, "JPEG size is truncated or over capacity");
        const auto* bytes = reinterpret_cast<const unsigned char*>(jpeg.data());
        require(bytes[0] == 0xff && bytes[1] == 0xd8 && bytes[jpeg.size() - 2] == 0xff && bytes[jpeg.size() - 1] == 0xd9,
                "JPEG must have complete SOI/EOI markers");
        std::memcpy(staging_, jpeg.data(), jpeg.size());
        jc(nvjpegJpegStreamParseHeader(handle_, staging_, jpeg.size(), parsed_), "parse JPEG header");
        nvjpegJpegEncoding_t encoding{};
        jc(nvjpegJpegStreamGetJpegEncoding(parsed_, &encoding), "get JPEG encoding");
        require(encoding == NVJPEG_ENCODING_BASELINE_DCT, "only baseline JPEG is supported");
        unsigned precision = 0;
        jc(nvjpegJpegStreamGetSamplePrecision(parsed_, &precision), "get sample precision");
        require(precision == 8, "only 8-bit JPEG is supported");
        int components = 0, widths[NVJPEG_MAX_COMPONENT]{}, heights[NVJPEG_MAX_COMPONENT]{};
        nvjpegChromaSubsampling_t subsampling{};
        jc(nvjpegGetImageInfo(handle_, staging_, jpeg.size(), &components, &subsampling, widths, heights), "get JPEG metadata");
        require(components == 3 && subsampling == NVJPEG_CSS_420, "only three-component YCbCr 4:2:0 JPEG is supported");
        require(widths[0] > 0 && widths[0] <= kMaxWidth && heights[0] > 0 && heights[0] <= kMaxHeight, "JPEG exceeds configured dimensions");
        require(widths[0] == expected_width && heights[0] == expected_height, "JPEG dimensions differ from sequence metadata");
        int unsupported = 1;
        jc(nvjpegDecodeBatchedSupported(handle_, parsed_, &unsupported), "check hardware support");
        require(unsupported == 0, "JPEG unsupported by HARDWARE backend");
        const auto plane = static_cast<std::size_t>(widths[0]) * heights[0];
        nvjpegImage_t destination{};
        for (unsigned c = 0; c < 3; ++c) {
            destination.channel[c] = raw_ + c * plane;
            destination.pitch[c] = static_cast<std::size_t>(widths[0]);
        }
        const unsigned char* input = staging_;
        const std::size_t length = jpeg.size();
        const auto decode_begin = Clock::now();
        jc(nvjpegDecodeBatched(handle_, state_, &input, &length, &destination, stream_), "decode HARDWARE batch-1 RGB");
        const auto decode_return = Clock::now();
        ck(cudaStreamSynchronize(stream_), "wait for RGB delivery");
        const auto rgb_ready = Clock::now();
        result.f01_full_us = us(rgb_ready - begin);
        result.staging_parse_us = us(decode_begin - begin);
        result.decode_call_us = us(decode_return - decode_begin);
        result.decode_to_rgb_ready_us = us(rgb_ready - decode_begin);

        const saccade::ResizeShape shape{widths[0], heights[0], kOutputWidth, kOutputHeight};
        const auto resize_begin = Clock::now();
        ck(cudaEventRecord(start_, stream_), "record F02 start");
        saccade::resize_rgb_chw_cuda(shape, raw_, raw_bytes_, images_, output_elements_, workspace_, workspace_elements_, stream_);
        ck(cudaEventRecord(stop_, stream_), "record F02 stop");
        const auto resize_enqueued = Clock::now();
        ck(cudaEventSynchronize(stop_), "wait for independent F02 delivery");
        const auto resize_done = Clock::now();
        float milliseconds = 0;
        ck(cudaEventElapsedTime(&milliseconds, start_, stop_), "F02 CUDA-event envelope");
        result.f02_gpu_event_envelope_us = static_cast<double>(milliseconds) * 1000;
        result.f02_enqueue_us = us(resize_enqueued - resize_begin);
        result.f02_enqueue_to_done_us = us(resize_done - resize_begin);
        return result;
    }
    void close() { release().throw_if_failed(); }
private:
    saccade::CleanupReport release() noexcept {
        saccade::CleanupReport r;
        if (stream_) {
            int drain = 1;
            r.release(drain, [&](int) { ck(cudaStreamSynchronize(stream_), "probe drain"); });
        }
        r.release(parsed_, [](auto p) { jc(nvjpegJpegStreamDestroy(p), "destroy parsed JPEG"); });
        r.release(state_, [](auto p) { jc(nvjpegJpegStateDestroy(p), "destroy JPEG state"); });
        r.release(handle_, [](auto p) { jc(nvjpegDestroy(p), "destroy JPEG decoder"); });
        r.release(start_, [](auto p) { ck(cudaEventDestroy(p), "destroy start event"); });
        r.release(stop_, [](auto p) { ck(cudaEventDestroy(p), "destroy stop event"); });
        r.release(workspace_, [](auto p) { ck(cudaFree(p), "free resize workspace"); });
        r.release(images_, [](auto p) { ck(cudaFree(p), "free images"); });
        r.release(raw_, [](auto p) { ck(cudaFree(p), "free RGB"); });
        r.release(staging_, [](auto p) { ck(cudaFreeHost(p), "free JPEG staging"); });
        r.release(stream_, [](auto p) { ck(cudaStreamDestroy(p), "destroy detect stream"); });
        return r;
    }
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_ = nullptr, stop_ = nullptr;
    nvjpegHandle_t handle_ = nullptr;
    nvjpegJpegState_t state_ = nullptr;
    nvjpegJpegStream_t parsed_ = nullptr;
    unsigned char *staging_ = nullptr, *raw_ = nullptr;
    float *images_ = nullptr, *workspace_ = nullptr;
    std::size_t raw_bytes_ = 0, output_elements_ = 0, workspace_elements_ = 0;
};

std::string csv(const std::vector<Sample>& samples) {
    std::ostringstream out;
    out << "frame,jpeg_bytes,f01_full_us,staging_parse_us,decode_call_us,decode_to_rgb_ready_us,f02_gpu_event_envelope_us,f02_enqueue_us,f02_enqueue_to_done_us\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& s : samples)
        out << s.frame << ',' << s.jpeg_bytes << ',' << s.f01_full_us << ',' << s.staging_parse_us << ',' << s.decode_call_us << ','
            << s.decode_to_rgb_ready_us << ',' << s.f02_gpu_event_envelope_us << ',' << s.f02_enqueue_us << ',' << s.f02_enqueue_to_done_us << '\n';
    return out.str();
}
std::string resources() {
    const auto& hr = saccade::kHrModelProfile;
    std::ostringstream out;
    out << "{\n  \"schema\":\"jpeg-stage-resources/1\",\n  \"abi_sizeof\":{\n"
        << "    \"DetectionCandidates\":" << sizeof(saccade::DetectionCandidates)
        << ",\n    \"DeviceTrack\":" << sizeof(saccade::DeviceTrack)
        << ",\n    \"DeviceTrackState\":" << sizeof(saccade::DeviceTrackState)
        << ",\n    \"DevicePredicted\":" << sizeof(saccade::DevicePredicted)
        << ",\n    \"DevicePredictedPrior\":" << sizeof(saccade::DevicePredictedPrior)
        << ",\n    \"DeviceBatchDetection\":" << sizeof(saccade::DeviceBatchDetection)
        << ",\n    \"DeviceBatch\":" << sizeof(saccade::DeviceBatch)
        << ",\n    \"DeviceTrackingRows\":" << sizeof(saccade::DeviceTrackingRows)
        << ",\n    \"TrackingCudaScratch\":" << sizeof(saccade::TrackingCudaScratch)
        << ",\n    \"TrackingSessionControl\":" << sizeof(saccade::detail::TrackingSessionControl)
        << ",\n    \"Control\":" << sizeof(saccade::Control)
        << ",\n    \"TrackerConfig\":" << sizeof(saccade::TrackerConfig)
        << ",\n    \"MotionContext\":" << sizeof(saccade::MotionContext)
        << ",\n    \"DeviceMotionContext\":" << sizeof(saccade::DeviceMotionContext)
        << ",\n    \"PostprocessProducerStatus\":" << sizeof(saccade::PostprocessProducerStatus)
        << ",\n    \"ScanStatus\":" << sizeof(saccade::ScanStatus)
        << ",\n    \"PacketBytes\":" << sizeof(saccade::PacketBytes) << "\n  },\n"
        << "  \"hr_positions\":" << hr.positions()
        << ",\n  \"hr_raw_bytes\":" << hr.raw_elements() * sizeof(float)
        << ",\n  \"hr_input_bytes\":" << hr.input_elements() * sizeof(float)
        << ",\n  \"postprocess_workspace_bytes\":" << saccade::postprocess_workspace_bytes()
        << ",\n  \"model_producer_workspace_bytes\":" << saccade::model_producer_workspace_bytes()
        << ",\n  \"rgb_allocation_bytes\":" << 3 * std::size_t(kMaxWidth) * kMaxHeight
        << ",\n  \"resize_workspace_bytes\":" << 3 * std::size_t(kOutputWidth) * kMaxHeight * sizeof(float)
        << ",\n  \"jpeg_pinned_allocation_bytes\":" << kJpegCapacity
        << ",\n  \"default_result_ring_pinned_bytes\":" << 4 * sizeof(saccade::PacketBytes)
        << ",\n  \"scope\":\"sizeof and explicit capacities; does not include library, context, decoder, FFT or graph private storage\"\n}\n";
    return out.str();
}
} // namespace

int main(int argc, char** argv) {
    std::filesystem::path output;
    try {
        if (argc < 3) throw std::invalid_argument(usage);
        std::size_t frames = 120, warmup = 20;
        for (int i = 3; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--frames" && i + 1 < argc) frames = number<std::size_t>(argv[++i]);
            else if (arg == "--warmup" && i + 1 < argc) warmup = number<std::size_t>(argv[++i]);
            else throw std::invalid_argument(usage);
        }
        require(frames > 0 && warmup > 0 && frames <= 1000000 && warmup <= 1000000, "positive frames/warmup required; each <=1000000");
        const std::filesystem::path sequence = argv[1], destination = argv[2];
        const auto info_bytes = saccade::read_text_file(sequence / "seqinfo.ini");
        std::map<std::string, std::string> fields;
        std::istringstream info(info_bytes);
        for (std::string line; std::getline(info, line);) {
            const auto equals = line.find('=');
            if (equals != std::string::npos) fields[trim(std::string_view(line).substr(0, equals))] = trim(std::string_view(line).substr(equals + 1));
        }
        const int width = number<int>(fields.at("imWidth")), height = number<int>(fields.at("imHeight"));
        require(width > 0 && width <= kMaxWidth && height > 0 && height <= kMaxHeight, "sequence dimensions exceed NativePreprocessor capacity");
        require(warmup + frames <= number<std::size_t>(fields.at("seqLength")), "sequence shorter than warmup+frames");
        require(std::filesystem::create_directory(destination), "output directory must be new");
        output = destination;
        saccade::write_text_file(output / "status.json", "{\"status\":\"RUNNING\"}\n");
        std::ostringstream manifest, command;
        manifest << saccade::sha256_hex(info_bytes) << "  " << (sequence / "seqinfo.ini").string() << '\n';
        for (int i = 0; i < argc; ++i) command << (i ? " " : "") << quote(argv[i]);
        saccade::write_text_file(output / "command.txt", command.str() + '\n');
        std::vector<std::string> images;
        images.reserve(warmup + frames);
        for (std::size_t frame = 0; frame < warmup + frames; ++frame) {
            std::ostringstream name;
            name << std::setfill('0') << std::setw(6) << frame + 1 << fields.at("imExt");
            const auto path = sequence / fields.at("imDir") / name.str();
            images.push_back(saccade::read_text_file(path));
            manifest << saccade::sha256_hex(images.back()) << "  " << path.string() << '\n';
        }
        saccade::write_text_file(output / "inputs.sha256", manifest.str());
        Probe probe;
        std::vector<Sample> warmup_samples, samples;
        warmup_samples.reserve(warmup);
        samples.reserve(frames);
        for (std::size_t frame = 0; frame < images.size(); ++frame) {
            auto sample = probe.measure(images[frame], frame, width, height);
            (frame < warmup ? warmup_samples : samples).push_back(sample);
        }
        probe.close();
        saccade::write_text_file(output / "warmup.csv", csv(warmup_samples));
        saccade::write_text_file(output / "samples.csv", csv(samples));
        saccade::write_text_file(output / "resources.json", resources());

        cudaDeviceProp gpu{};
        ck(cudaGetDeviceProperties(&gpu, 0), "query GPU identity");
        int runtime = 0, driver = 0, major = 0, minor = 0, patch = 0;
        ck(cudaRuntimeGetVersion(&runtime), "query CUDA runtime");
        ck(cudaDriverGetVersion(&driver), "query driver API");
        jc(nvjpegGetProperty(MAJOR_VERSION, &major), "query nvJPEG major");
        jc(nvjpegGetProperty(MINOR_VERSION, &minor), "query nvJPEG minor");
        jc(nvjpegGetProperty(PATCH_LEVEL, &patch), "query nvJPEG patch");
        std::ostringstream summary;
        summary << std::fixed << std::setprecision(6)
            << "{\n  \"schema\":\"jpeg-stage-probe/1\",\n  \"status\":\"PASS\",\n"
            << "  \"sequence\":" << quote(std::filesystem::absolute(sequence).string())
            << ",\n  \"frames\":" << frames << ",\n  \"warmup\":" << warmup
            << ",\n  \"first_measured_frame\":" << warmup << ",\n  \"last_measured_frame\":" << warmup + frames - 1
            << ",\n  \"gpu\":" << quote(gpu.name) << ",\n  \"cuda_runtime\":" << runtime << ",\n  \"cuda_driver_api\":" << driver
            << ",\n  \"nvjpeg\":" << quote(std::to_string(major) + '.' + std::to_string(minor) + '.' + std::to_string(patch))
            << ",\n  \"backend\":\"NVJPEG_BACKEND_HARDWARE\",\n  \"output\":\"NVJPEG_OUTPUT_RGB\",\n  \"batch\":1,\n"
            << "  \"input_io\":\"JPEGs preloaded; disk input and all output writes excluded\",\n"
            << "  \"f01_scope\":\"idle start; staging memcpy, metadata/support validation and decode through detect-stream synchronization to RGB delivery; not decoder-engine busy time\",\n"
            << "  \"f02_scope\":\"isolated after F01 completion; same RGB, NativePreprocessor horizontal/vertical FP32 resize; CUDA events enclose validation/submission gaps as well as kernels\",\n"
            << "  \"overlap\":\"none between F01 and F02 or frames; isolated measurements cannot be added to overlapping pipeline timings\",\n"
            << "  \"percentiles\":\"nearest rank of individual measured-frame samples; all outliers retained\",\n  \"metrics_us\":{\n";
        const std::pair<const char*, double Sample::*> metrics[] = {
            {"f01_full", &Sample::f01_full_us}, {"staging_parse", &Sample::staging_parse_us},
            {"decode_call", &Sample::decode_call_us}, {"decode_to_rgb_ready", &Sample::decode_to_rgb_ready_us},
            {"f02_gpu_event_envelope", &Sample::f02_gpu_event_envelope_us},
            {"f02_enqueue", &Sample::f02_enqueue_us}, {"f02_enqueue_to_done", &Sample::f02_enqueue_to_done_us}
        };
        bool first = true;
        for (const auto& [name, member] : metrics) {
            std::vector<double> values;
            values.reserve(samples.size());
            for (const auto& s : samples) values.push_back(s.*member);
            if (!first) summary << ",\n";
            first = false;
            summary << "    " << quote(name) << ":{\"samples\":" << values.size()
                << ",\"mean\":" << std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size())
                << ",\"p50\":" << percentile(values, .50) << ",\"p95\":" << percentile(values, .95)
                << ",\"p99\":" << percentile(values, .99) << ",\"min\":" << *std::min_element(values.begin(), values.end())
                << ",\"max\":" << *std::max_element(values.begin(), values.end()) << '}';
        }
        summary << "\n  }\n}\n";
        saccade::write_text_file(output / "summary.json", summary.str());
        saccade::write_text_file(output / "status.json", "{\"status\":\"PASS\"}\n");
        std::cout << summary.str();
        return 0;
    } catch (const std::exception& error) {
        if (!output.empty()) {
            try { saccade::write_text_file(output / "status.json", "{\"status\":\"FAIL\",\"error\":" + quote(error.what()) + "}\n"); }
            catch (const std::exception& write_error) { std::cerr << "record failure: " << write_error.what() << '\n'; }
        }
        std::cerr << "jpeg_stage_probe: " << error.what() << '\n';
        return 1;
    }
}

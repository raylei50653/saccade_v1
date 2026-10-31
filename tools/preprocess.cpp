#include "saccade/preprocess.hpp"
#include "saccade/resize_cuda.hpp"
#include "saccade/cleanup.hpp"
#include "saccade/file_io.hpp"
#include "saccade/sha256.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using namespace saccade;
namespace fs = std::filesystem;
void ck(cudaError_t rc, const char* what) {
    if (rc != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(rc));
}
void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}
std::span<const std::uint8_t> bytes(const std::string& data) {
    return {reinterpret_cast<const std::uint8_t*>(data.data()), data.size()};
}
template<class T> std::span<const std::uint8_t> bytes(const std::vector<T>& data) {
    return {reinterpret_cast<const std::uint8_t*>(data.data()), data.size()*sizeof(T)};
}
struct Resources {
    cudaStream_t stream = nullptr;
    std::uint8_t* rgb = nullptr;
    float *images = nullptr, *workspace = nullptr;
    Resources() { ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream"); }
    CleanupReport release() {
        CleanupReport r;
        if (stream) {
            int drain = 1;
            r.release(drain, [&](int) { ck(cudaStreamSynchronize(stream), "drain"); });
        }
        r.release(workspace, [](auto p) { ck(cudaFree(p), "free workspace"); });
        r.release(images, [](auto p) { ck(cudaFree(p), "free images"); });
        r.release(rgb, [](auto p) { ck(cudaFree(p), "free RGB"); });
        r.release(stream, [](auto s) { ck(cudaStreamDestroy(s), "destroy stream"); });
        return r;
    }
    ~Resources() { release(); }
};
struct Case {
    std::string label, jpeg, rgb, images;
    ResizeShape shape;
};
int positive(const std::string& text) {
    std::size_t end = 0;
    const int n = std::stoi(text, &end);
    require(end == text.size() && n > 0 && n <= 8192, "invalid case dimension");
    return n;
}
std::vector<Case> cases(const fs::path& dir) {
    std::istringstream input(read_text_file(dir / "cases.tsv"));
    std::string line;
    std::vector<Case> result;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> row;
        std::istringstream fields(line);
        std::string field;
        while (std::getline(fields, field, '\t')) row.push_back(field);
        require(row.size() == 8, "cases.tsv must have 8 fields per row");
        result.push_back({row[0], row[5], row[6], row[7], {positive(row[1]), positive(row[2]), positive(row[3]), positive(row[4])}});
    }
    require(!result.empty(), "empty reference list is not a successful run");
    return result;
}
std::string stem(std::size_t index) {
    std::ostringstream name;
    name << "case-" << std::setw(3) << std::setfill('0') << index;
    return name.str();
}
double tolerance(const ResizeShape& s) {
    const double nx = std::ceil(2*std::max(1., double(s.input_width)/s.output_width))+1;
    const double ny = std::ceil(2*std::max(1., double(s.input_height)/s.output_height))+1;
    const double nu = (4*(nx+ny)+24)*std::ldexp(1., -24);
    return 2*nu/(1-nu);
}
void dump(const fs::path& dir, const std::string& name, const std::vector<std::uint8_t>& rgb, const std::vector<float>& images) {
    write_binary_file(dir / (name+".rgb"), bytes(rgb));
    write_binary_file(dir / (name+".f32"), bytes(images));
}
void run(const fs::path& refs, const fs::path& out, int only) {
    require(std::endian::native == std::endian::little, "reference format requires little endian");
    fs::create_directories(out);
    const auto inputs = cases(refs);
    require(only < 0 || static_cast<std::size_t>(only) < inputs.size(), "case index out of range");
    Resources resources;
    std::ostringstream report;
    report << resize_cuda_identity() << "cases_sha256=" << sha256_hex(read_text_file(refs / "cases.tsv")) << '\n';
    report << "case\tlabel\trgb_exact\trepeat_exact\tmax_abs\tmean_abs\tatol\tviolations\tmin\tmax\n";
    bool pass = true;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        if (only >= 0 && index != static_cast<std::size_t>(only)) continue;
        const auto& item = inputs[index];
        const auto& s = item.shape;
        const auto ni = resize_input_bytes(s), no = resize_output_elements(s), nw = resize_workspace_elements(s);
        const auto rgb_ref = read_text_file(refs / item.rgb);
        const auto image_ref = read_text_file(refs / item.images);
        require(rgb_ref.size() == ni && image_ref.size() == no*sizeof(float), "reference capacity mismatch");
        std::vector<float> expected(no);
        std::memcpy(expected.data(), image_ref.data(), image_ref.size());
        std::vector<std::uint8_t> rgb(ni), again_rgb(ni);
        std::vector<float> images(no), again(no);
        std::string jpeg_data;
        std::unique_ptr<NativePreprocessor> decoder;
        if (item.jpeg != "-") {
            jpeg_data = read_text_file(refs / item.jpeg);
            PreprocessConfig config{s.input_width, s.input_height, s.output_width, s.output_height};
            decoder = std::make_unique<NativePreprocessor>(resources.stream, config);
            write_text_file(out / (stem(index)+"-identity.txt"), decoder->identity());
        } else {
            ck(cudaMalloc(&resources.rgb, ni), "allocate synthetic RGB");
            ck(cudaMalloc(&resources.images, no*sizeof(float)), "allocate synthetic output");
            ck(cudaMalloc(&resources.workspace, nw*sizeof(float)), "allocate synthetic workspace");
            ck(cudaMemcpyAsync(resources.rgb, rgb_ref.data(), ni, cudaMemcpyHostToDevice, resources.stream), "upload synthetic RGB");
        }
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            const std::uint8_t* raw = resources.rgb;
            float* output = resources.images;
            if (decoder) {
                const auto frame = decoder->enqueue(bytes(jpeg_data), repeat, s.input_width, s.input_height);
                raw = frame.rgb;
                output = frame.images;
                const auto& m = frame.mapping;
                require(m.scale_x == double(s.output_width)/s.input_width && m.scale_y == double(s.output_height)/s.input_height,
                        "coordinate mapping differs from stretch");
            } else {
                resize_rgb_chw_cuda(s, resources.rgb, ni, output, no, resources.workspace, nw, resources.stream);
            }
            ck(cudaMemcpyAsync((repeat ? again_rgb : rgb).data(), raw, ni, cudaMemcpyDeviceToHost, resources.stream), "download RGB");
            ck(cudaMemcpyAsync((repeat ? again : images).data(), output, no*sizeof(float), cudaMemcpyDeviceToHost, resources.stream), "download images");
            ck(cudaStreamSynchronize(resources.stream), "frame completion");
        }
        if (decoder) decoder->close();
        else {
            CleanupReport release;
            release.release(resources.workspace, [](auto p) { ck(cudaFree(p), "free synthetic workspace"); });
            release.release(resources.images, [](auto p) { ck(cudaFree(p), "free synthetic output"); });
            release.release(resources.rgb, [](auto p) { ck(cudaFree(p), "free synthetic RGB"); });
            release.throw_if_failed();
        }
        dump(out, stem(index), rgb, images);
        const bool exact = std::memcmp(rgb.data(), rgb_ref.data(), ni) == 0;
        const bool repeat = rgb == again_rgb && std::memcmp(images.data(), again.data(), no*sizeof(float)) == 0;
        const double atol = tolerance(s);
        double max_error = 0, sum_error = 0;
        std::size_t violations = 0;
        for (std::size_t j = 0; j < no; ++j) {
            const double error = std::abs(double(images[j])-expected[j]);
            if (!std::isfinite(images[j]) || !std::isfinite(expected[j]) || error > atol) ++violations;
            max_error = std::max(max_error, error); sum_error += error;
        }
        const auto [low, high] = std::minmax_element(images.begin(), images.end());
        report << std::setprecision(10) << index << '\t' << item.label << '\t' << exact << '\t' << repeat << '\t'
               << max_error << '\t' << sum_error/no << '\t' << atol << '\t' << violations << '\t' << *low << '\t' << *high << '\n';
        pass = pass && exact && repeat && violations == 0;
        write_text_file(out / "report.tsv", report.str()); // retain failures and completed prefix
        std::cout << item.label << " rgb_exact=" << exact << " repeat_exact=" << repeat << " max_abs=" << max_error << " violations=" << violations << '\n';
    }
    resources.release().throw_if_failed();
    require(pass, "native preprocessing comparison failed; see report.tsv");
}
void jpeg(const fs::path& input, const fs::path& out) {
    fs::create_directories(out);
    const auto compressed = read_text_file(input); // I/O errors are not EOF.
    Resources resources;
    NativePreprocessor decoder(resources.stream);
    const auto frame = decoder.enqueue(bytes(compressed), 0);
    const auto& m = frame.mapping;
    std::vector<std::uint8_t> rgb(std::size_t{3}*m.original_width*m.original_height);
    std::vector<float> images(std::size_t{3}*m.model_width*m.model_height);
    ck(cudaMemcpyAsync(rgb.data(), frame.rgb, rgb.size(), cudaMemcpyDeviceToHost, resources.stream), "download RGB");
    ck(cudaMemcpyAsync(images.data(), frame.images, images.size()*sizeof(float), cudaMemcpyDeviceToHost, resources.stream), "download images");
    ck(cudaStreamSynchronize(resources.stream), "frame completion");
    dump(out, "frame", rgb, images);
    std::ostringstream report;
    report << std::setprecision(17) << decoder.identity() << resize_cuda_identity() << "jpeg_sha256=" << sha256_hex(compressed)
           << "\noriginal_width=" << m.original_width << "\noriginal_height=" << m.original_height
           << "\nmodel_width=" << m.model_width << "\nmodel_height=" << m.model_height
           << "\nscale_x=" << m.scale_x << "\nscale_y=" << m.scale_y << '\n';
    decoder.close();
    resources.release().throw_if_failed();
    write_text_file(out / "mapping.txt", report.str());
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 4 || argc == 6, "usage: preprocess jpeg JPEG OUTPUT_DIR | run REFERENCES OUTPUT_DIR [--case N]");
        if (std::string(argv[1]) == "jpeg") {
            require(argc == 4, "jpeg mode takes JPEG OUTPUT_DIR");
            jpeg(argv[2], argv[3]);
        } else {
            require(std::string(argv[1]) == "run", "unknown command");
            int only = -1;
            if (argc == 6) {
                require(std::string(argv[4]) == "--case", "unknown run option");
                std::size_t end = 0; only = std::stoi(argv[5], &end);
                require(only >= 0 && end == std::strlen(argv[5]), "invalid case index");
            }
            run(argv[2], argv[3], only);
        }
        return 0;
    } catch (const std::exception& error) { std::cerr << "preprocess: " << error.what() << '\n'; return 1; }
}

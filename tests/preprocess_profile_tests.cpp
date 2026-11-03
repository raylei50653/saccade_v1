#include "saccade/file_io.hpp"
#include "saccade/model_profile.hpp"
#include "saccade/preprocess.hpp"
#include "saccade/resize_cuda.hpp"

#include <cuda_runtime.h>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
unsigned checks = 0;
void check(bool ok, const char* why) {
    ++checks;
    if (!ok) throw std::runtime_error(why);
}
void cu(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Resources {
    cudaStream_t stream = nullptr;
    unsigned char* rgb = nullptr;
    float *images = nullptr, *workspace = nullptr;
    Resources() { cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Resources() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (workspace) (void)cudaFree(workspace);
        if (images) (void)cudaFree(images);
        if (rgb) (void)cudaFree(rgb);
        if (stream) (void)cudaStreamDestroy(stream);
    }
};
}

int main(int argc, char** argv) {
    try {
        using namespace saccade;
        check(argc == 2, "usage: preprocess_profile_tests REFS");
        cu(cudaSetDevice(0));
        const std::filesystem::path directory = argv[1];
        std::istringstream cases(read_text_file(directory / "cases.tsv"));
        std::string line;
        unsigned count = 0;
        while (std::getline(cases, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> fields;
            std::istringstream row(line);
            for (std::string field; std::getline(row, field, '\t');) fields.push_back(field);
            check(fields.size() == 8, "reference row must have 8 fields");
            ResizeShape shape{std::stoi(fields[1]), std::stoi(fields[2]), std::stoi(fields[3]), std::stoi(fields[4])};
            check(shape.output_width == kHrModelProfile.width && shape.output_height == kHrModelProfile.height,
                  "reference must use HR dimensions");
            const auto rgb = read_text_file(directory / fields[6]);
            const auto images = read_text_file(directory / fields[7]);
            check(rgb.size() == resize_input_bytes(shape) && images.size() == resize_output_elements(shape) * sizeof(float),
                  "reference extents");
            Resources resource;
            std::vector<unsigned char> actual_rgb(rgb.size());
            std::vector<float> actual_images(resize_output_elements(shape));
            if (fields[5] != "-") {
                const auto jpeg = read_text_file(directory / fields[5]);
                PreprocessConfig config{shape.input_width, shape.input_height, shape.output_width, shape.output_height};
                NativePreprocessor preprocess(resource.stream, config);
                const auto frame = preprocess.enqueue({reinterpret_cast<const std::uint8_t*>(jpeg.data()), jpeg.size()}, 0,
                                                       shape.input_width, shape.input_height);
                check(frame.mapping.model_width == 1280 && frame.mapping.model_height == 768, "HR preprocessing mapping");
                cu(cudaMemcpyAsync(actual_rgb.data(), frame.rgb, rgb.size(), cudaMemcpyDeviceToHost, resource.stream));
                cu(cudaMemcpyAsync(actual_images.data(), frame.images, images.size(), cudaMemcpyDeviceToHost, resource.stream));
                preprocess.wait();
                cu(cudaStreamSynchronize(resource.stream));
                preprocess.close();
            } else {
                cu(cudaMalloc(&resource.rgb, rgb.size()));
                cu(cudaMalloc(&resource.images, images.size()));
                const auto workspace = resize_workspace_elements(shape);
                cu(cudaMalloc(&resource.workspace, workspace * sizeof(float)));
                cu(cudaMemcpyAsync(resource.rgb, rgb.data(), rgb.size(), cudaMemcpyHostToDevice, resource.stream));
                resize_rgb_chw_cuda(shape, resource.rgb, rgb.size(), resource.images, actual_images.size(),
                                    resource.workspace, workspace, resource.stream);
                cu(cudaMemcpyAsync(actual_rgb.data(), resource.rgb, rgb.size(), cudaMemcpyDeviceToHost, resource.stream));
                cu(cudaMemcpyAsync(actual_images.data(), resource.images, images.size(), cudaMemcpyDeviceToHost, resource.stream));
                cu(cudaStreamSynchronize(resource.stream));
            }
            check(std::memcmp(actual_rgb.data(), rgb.data(), rgb.size()) == 0, "HR RGB differs bitwise from Python");
            check(std::memcmp(actual_images.data(), images.data(), images.size()) == 0, "HR FP32 differs bitwise from Python");
            std::cout << "case=" << fields[0] << " RGB/FP32 bitwise PASS\n";
            ++count;
        }
        check(count != 0, "empty reference corpus");
        std::cout << "preprocess_profile_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "preprocess_profile_tests: " << error.what() << '\n';
        return 1;
    }
}

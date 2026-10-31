// Native weight delivery of the T=1 YOLO26m Mamba-FPN head (docs/HEAD_TRT.md
// §2). `convert` reads a torch.save checkpoint as zip + pickle data only, checks
// it and writes the canonical bundle and the key-by-key mapping report;
// `verify` decodes a bundle strictly and checks its SHA-256.
#include "saccade/file_io.hpp"
#include "saccade/head_reference.hpp"
#include "saccade/head_weights.hpp"
#include "saccade/sha256.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void distinct(const std::filesystem::path& input, const std::filesystem::path& output) {
    std::error_code error;
    if (std::filesystem::exists(output, error) && std::filesystem::equivalent(input, output, error))
        throw std::invalid_argument("output " + output.string() + " is the input file");
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "convert" && argc == 6) {
            distinct(argv[2], argv[4]);
            distinct(argv[2], argv[5]);
            auto result = saccade::convert_head_checkpoint(saccade::read_text_file(argv[2]), argv[3]);
            saccade::write_text_file(argv[4], result.bundle);
            saccade::write_text_file(argv[5], result.mapping);
            // Read back what was written: the bytes on disk are the deliverable.
            const auto written = saccade::read_text_file(argv[4]);
            if (written != result.bundle) throw std::runtime_error("bundle readback differs from the bytes written");
            (void)saccade::decode_head_bundle(written);
            std::cout << "checkpoint_sha256=" << result.weights.source_sha256 << "\nsource_archive=" << result.weights.source_archive
                      << "\nsource_epoch=" << result.weights.source_epoch << "\ntensors=" << result.weights.tensors.size()
                      << "\nbundle_bytes=" << written.size() << "\nbundle_sha256=" << saccade::sha256_hex(written) << '\n';
        } else if (mode == "verify" && argc == 4) {
            const auto bytes = saccade::read_text_file(argv[2]);
            const auto sha = saccade::sha256_hex(bytes);
            if (sha != argv[3]) throw std::runtime_error("bundle SHA-256 " + sha + " differs from the expected " + argv[3]);
            const auto weights = saccade::decode_head_bundle(bytes);
            std::cout << "bundle_sha256=" << sha << "\nschema=" << weights.schema << "\nsource_sha256=" << weights.source_sha256
                      << "\nsource_archive=" << weights.source_archive << "\nsource_epoch=" << weights.source_epoch
                      << "\ntensors=" << weights.tensors.size() << "\nfinite=true\n";
        } else if (mode == "synthetic" && argc == 4) {
            // Deterministic stand-in weights for structure tests without the checkpoint; never a model.
            const auto bundle = saccade::encode_head_bundle(saccade::synthetic_head_weights(std::stoull(argv[2])));
            saccade::write_text_file(argv[3], bundle);
            std::cout << "synthetic_seed=" << argv[2] << "\nbundle_sha256=" << saccade::sha256_hex(bundle) << '\n';
        } else {
            throw std::invalid_argument("usage: head_weights convert CHECKPOINT EXPECTED_SHA256 OUT.swb MAPPING.tsv | verify BUNDLE.swb EXPECTED_SHA256 | synthetic SEED OUT.swb");
        }
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("cannot write standard output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "head_weights: " << error.what() << '\n';
        return 2;
    }
}

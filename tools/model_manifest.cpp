#include "saccade/model_manifest.hpp"
#include "saccade/file_io.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 4 || (std::string(argv[1]) != "validate" && std::string(argv[1]) != "verify"))
            throw std::invalid_argument("usage: model_manifest validate|verify MANIFEST.json [CANONICAL.json]");
        const auto path = std::filesystem::absolute(argv[2]);
        const auto manifest = saccade::parse_model_manifest(saccade::read_text_file(path));
        const bool verify = std::string(argv[1]) == "verify";
        if (verify) saccade::verify_manifest_artifacts(manifest, path);
        if (argc == 4) {
            const auto output = std::filesystem::absolute(argv[3]).lexically_normal();
            // Reject overwriting any input, including a symlink/hardlink alias.
            std::error_code error;
            if (output == path.lexically_normal() || std::filesystem::equivalent(path, output, error))
                throw std::invalid_argument("canonical output aliases input manifest: " + output.string());
            for (const auto& artifact : manifest.artifacts) {
                const auto input = (path.parent_path() / artifact.path).lexically_normal();
                error.clear();
                if (output == input || std::filesystem::equivalent(input, output, error))
                    throw std::invalid_argument("canonical output aliases artifact: " + output.string());
            }
            saccade::write_text_file(output, manifest.canonical_json);
        }
        std::cout << "schema=" << manifest.schema << "\npurpose=" << manifest.purpose
                  << "\nmanifest_declaration=verified\nartifact_files=" << (verify ? "verified" : "not_checked")
                  << "\nmodel_content=not_checked\nreal_model_ready=false\ncanonical_sha256="
                  << saccade::model_manifest_hash(manifest) << '\n';
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("cannot write standard output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_manifest: " << error.what() << '\n';
        return 2;
    }
}

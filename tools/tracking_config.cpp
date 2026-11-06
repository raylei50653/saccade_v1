#include "saccade/config_cli.hpp"

#include <iostream>
#include <stdexcept>

namespace {
saccade::ConfigCapabilities capabilities() {
    saccade::ConfigCapabilities c;
#ifdef SACCADE_P08_HAS_CUDA
    c.backends.push_back("cuda"); c.inputs.push_back("jpeg");
#endif
#ifdef SACCADE_P08_HAS_STREAMING
    c.inputs.push_back("host-rgb"); c.inputs.push_back("device-video");
#endif
    return c;
}
saccade::ResolvedTrackingConfig read_config(const char* path) {
    const auto text = saccade::read_text_file(path);
    if (text.find("record.schema=") != std::string::npos)
        return saccade::parse_resolved_config_record(text, capabilities()).config;
    return saccade::resolve_tracking_config(text, {}, {}, capabilities());
}
}
int main(int argc, char** argv) {
    try {
        using namespace saccade;
        if (argc < 2) throw std::invalid_argument("usage: tracking_config schema | resolve [--config FILE] [--set KEY=VALUE] [--output CONFIG] [--record RECEIPT] | compare LEFT RIGHT [--allow KEY]");
        const std::string command = argv[1];
        if (command == "schema" && argc == 2) {
            std::cout << "schema=" << kResolvedConfigSchema << '\n';
            for (const auto& f : tracking_config_schema())
                std::cout << f.key << "\tcategory=" << static_cast<int>(f.category)
                          << "\ttype=" << static_cast<int>(f.type) << "\towner=" << f.owner
                          << "\tbehavior=" << f.behavior << "\tdefault=" << f.default_value
                          << "\tconstraint=" << f.constraint << "\tenv=" << f.environment
                          << "\tpurpose=" << f.purpose << '\n';
            return 0;
        }
        if (command == "resolve") {
            std::filesystem::path path, output, record;
            ConfigAssignments assignments;
            std::set<std::string> flags;
            for (int i = 2; i < argc; ++i) {
                const std::string flag = argv[i];
                if (++i == argc) throw std::invalid_argument("incomplete config option");
                if (flag == "--set") assignments.push_back(tracking_cli_assignment(argv[i]));
                else {
                    if (!flags.insert(flag).second) throw std::invalid_argument("duplicate config option");
                    if (flag == "--config") path = argv[i];
                    else if (flag == "--output") output = argv[i];
                    else if (flag == "--record") record = argv[i];
                    else throw std::invalid_argument("unknown config option");
                }
            }
            if (!output.empty() && !record.empty() && std::filesystem::weakly_canonical(output) == std::filesystem::weakly_canonical(record))
                throw std::invalid_argument("config and receipt outputs must differ");
            const auto config = resolve_tracking_config(path.empty() ? "" : read_text_file(path),
                declared_tracking_environment(), assignments, capabilities());
            const auto text = canonical_resolved_config(config);
            if (!output.empty()) write_resolved_config_file(output, text);
            if (!record.empty()) write_resolved_config_file(record, resolved_config_record(config, tracking_build_identity()));
            std::cout << text << "# config_sha256=" << resolved_config_hash(config) << '\n';
            return 0;
        }
        if (command == "compare" && argc >= 4) {
            std::vector<std::string> allowed;
            for (int i = 4; i < argc; ++i) {
                if (std::string_view(argv[i]) != "--allow" || ++i == argc) throw std::invalid_argument("expected --allow KEY");
                allowed.emplace_back(argv[i]);
            }
            const auto differences = compare_resolved_config(read_config(argv[2]), read_config(argv[3]), allowed);
            for (const auto& d : differences) std::cout << d.key << ": " << d.expected << " != " << d.actual << '\n';
            if (differences.empty()) std::cout << "configuration comparison PASS (build and input identity require separate checks)\n";
            return differences.empty() ? 0 : 1;
        }
        throw std::invalid_argument("unknown or incomplete tracking_config command");
    } catch (const std::exception& e) { std::cerr << "tracking_config: " << e.what() << '\n'; return 2; }
}

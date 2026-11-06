#include "saccade/config_cli.hpp"
#include "saccade/tracking_events.hpp"

#include <charconv>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace {
std::vector<saccade::TrackingEvent> read_events(const char* path) {
    const auto data = saccade::read_text_file(path);
    return saccade::decode_tracking_events({reinterpret_cast<const std::uint8_t*>(data.data()), data.size()});
}
struct Context {
    saccade::TrackerConfig tracker;
    std::optional<saccade::ResolvedTrackingConfig> resolved;
    std::optional<std::size_t> frames;
};
void valid(const std::vector<saccade::TrackingEvent>& events, const Context& context, const char* path) {
    const auto result = saccade::validate_tracking_events(events, context.tracker);
    if (!result.valid) throw std::runtime_error(std::string(path) + " event " + std::to_string(result.event_index) + ": " + result.message);
    if (context.resolved) {
        for (const auto& event : events)
            if (event.source != context.resolved->source.id || event.epoch != context.resolved->source.epoch)
                throw std::runtime_error("event source/epoch differs from resolved configuration");
    }
    if (context.frames && result.frames != *context.frames)
        throw std::runtime_error("event frame count differs from the recorded input count");
    std::cout << path << ": lifecycle/identity/commit PASS frames=" << result.frames
              << " events=" << events.size() << " sha256=" << saccade::tracking_events_hash(events)
              << " coverage=CPU-decision-events Kalman-GMC-auction-Bridge-ranking=not-recomputed\n";
}
}
int main(int argc, char** argv) {
    try {
        using namespace saccade;
        if (argc < 3) throw std::invalid_argument("usage: tracking_events validate TRACE [--config CONFIG_OR_RECEIPT] [--packets FILE] | compare LEFT RIGHT [--config CONFIG_OR_RECEIPT]");
        const std::string command = argv[1];
        const int first_option = command == "compare" ? 4 : 3;
        if ((command != "validate" && command != "compare") || argc < first_option)
            throw std::invalid_argument("unknown or incomplete events command");
        Context context;
        std::filesystem::path config_path, packet_path;
        std::set<std::string> flags;
        for (int i = first_option; i < argc; ++i) {
            const std::string flag = argv[i];
            if (++i == argc || !flags.insert(flag).second) throw std::invalid_argument("incomplete or duplicate events option");
            if (flag == "--config") config_path = argv[i];
            else if (flag == "--packets" && command == "validate") packet_path = argv[i];
            else throw std::invalid_argument("unknown events option");
        }
        if (!config_path.empty()) {
            const auto text = read_text_file(config_path);
            if (text.find("record.schema=") != std::string::npos) {
                const auto record = parse_resolved_config_record(text);
                context.resolved = record.config;
                constexpr std::string_view prefix = "input.frames=";
                std::istringstream input(record.build_identity);
                std::string line;
                while (std::getline(input, line)) {
                    if (!std::string_view(line).starts_with(prefix)) continue;
                    if (context.frames) throw std::runtime_error("duplicate input frame count in receipt");
                    std::size_t frames = 0;
                    const auto value = std::string_view(line).substr(prefix.size());
                    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), frames);
                    if (error != std::errc{} || end != value.data() + value.size()) throw std::runtime_error("invalid input frame count in receipt");
                    context.frames = frames;
                }
            } else context.resolved = resolve_tracking_config(text);
            context.tracker = context.resolved->tracker;
            std::cout << "validation_config_sha256=" << resolved_config_hash(*context.resolved) << '\n';
        }
        const auto left = read_events(argv[2]);
        valid(left, context, argv[2]);
        if (command == "validate") {
            if (!packet_path.empty()) {
                const auto bytes = read_text_file(packet_path);
                const auto checked = validate_tracking_event_packets(left,
                    {reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()});
                if (!checked.valid) throw std::runtime_error("packet binding: " + checked.message);
                std::cout << "full packet byte comparison PASS\n";
            }
            return 0;
        }
        const auto right = read_events(argv[3]);
        valid(right, context, argv[3]);
        const auto comparison = compare_tracking_events(left, right);
        if (!comparison.equal) {
            std::cout << "first divergence event=" << comparison.event_index << " field=" << comparison.field
                      << " left=" << comparison.left << " right=" << comparison.right << '\n';
            const auto show = [&](const auto& events, const char* side) {
                if (comparison.event_index >= events.size()) return;
                const auto& e = events[comparison.event_index];
                std::cout << side << " source=" << e.source << " epoch=" << e.epoch << " frame=" << e.frame
                          << " kind=" << static_cast<unsigned>(e.kind) << " before_uid=" << e.before.uid
                          << " after_uid=" << e.after.uid << " candidate_id=" << e.detection.candidate_id
                          << " row_uid=" << e.row.uid << " output_id=" << e.row.output_id << '\n';
            };
            show(left, "left"); show(right, "right");
            return 1;
        }
        std::cout << "event comparison PASS (MOT bytes and build/input identity require separate checks)\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "tracking_events: " << e.what() << '\n'; return 2; }
}

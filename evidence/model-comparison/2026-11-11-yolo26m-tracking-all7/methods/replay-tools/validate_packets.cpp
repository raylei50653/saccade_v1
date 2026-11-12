#include "saccade/protocol.hpp"
#include <cmath>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        const auto data = saccade::read_packet_file(argv[1]);
        const auto expected = std::stoull(argv[2]);
        if (data.size() / saccade::packet_bytes != expected) throw std::runtime_error("frame count differs");
        std::size_t observations = 0, predictions = 0;
        for (std::size_t i = 0; i < expected; ++i) {
            const auto p = saccade::decode_packet(std::span(data).subspan(i * saccade::packet_bytes, saccade::packet_bytes));
            if (p.epoch != 1 || p.frame != static_cast<saccade::FrameId>(i) ||
                p.committed_frame != p.frame || p.status != saccade::ResultStatus::ok)
                throw std::runtime_error("packet status or identity differs");
            std::set<std::uint64_t> uids, ids;
            for (const auto& row : p.tracks) {
                if (!uids.insert(row.uid).second || !ids.insert(row.output_id).second || row.class_id != 0)
                    throw std::runtime_error("duplicate identity or class differs");
                for (float value : row.xyxy) if (!std::isfinite(value)) throw std::runtime_error("nonfinite box");
                if (row.xyxy[2] <= row.xyxy[0] || row.xyxy[3] <= row.xyxy[1]) throw std::runtime_error("degenerate box");
                observations += row.flags == 1;
                predictions += row.flags == 2;
            }
        }
        std::cout << "PASS frames=" << expected << " observed_rows=" << observations
                  << " predicted_rows=" << predictions << " canonical=true all_status_ok=true\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

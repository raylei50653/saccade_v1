#include "saccade/protocol.hpp"

#include <algorithm>
#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: packet_compare LEFT.bin RIGHT.bin\n";
        return 2;
    }
    try {
        const auto left = saccade::read_packet_file(argv[1]);
        const auto right = saccade::read_packet_file(argv[2]);
        const auto comparison = saccade::compare_packet_bytes(left, right);
        if (comparison.equal) {
            std::cout << "identical: " << left.size() / saccade::packet_bytes
                      << " complete packets, " << left.size() << " bytes\n";
            return 0;
        }
        if (comparison.offset == std::min(left.size(), right.size())) {
            std::cout << "different lengths: " << left.size() << " versus " << right.size() << " bytes\n";
        } else {
            std::cout << "different at byte " << comparison.offset
                      << " (packet " << comparison.offset / saccade::packet_bytes
                      << ", offset " << comparison.offset % saccade::packet_bytes
                      << "): 0x" << std::hex << unsigned(comparison.left)
                      << " versus 0x" << unsigned(comparison.right) << '\n';
        }
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "packet_compare: " << error.what() << '\n';
        return 2;
    }
}

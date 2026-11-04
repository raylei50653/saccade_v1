#include "saccade/flow_records.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
}
int main(int argc, char** argv) {
    try {
        using namespace saccade::flow;
        if (argc != 2) throw std::runtime_error("expected SCFL test path");
        const auto v = reduce_cell({{{-320, 32}, {32, 160}, {96, -96}, {320, 96}}});
        check(v == Vector{2, 2}, "componentwise even median / S10.5");
        check(reduce_cell({{{-32768, 32767}, {-32768, 32767}, {-32768, 32767}, {-32768, 32767}}}) ==
              Vector{-1024, 1023.96875f}, "signed fixed point and sum range");
        Records r{17, 9, 8, 2, 1, Perf::slow, {{0, Status::uninitialized, {{0,0},{0,0}}},
                                             {1, Status::valid, {{2.5f,-1.5f},{-4,8}}}}};
        write_records(argv[1], r);
        check(std::filesystem::file_size(argv[1]) == 96, "explicit SCFL size");
        const auto back = read_records(argv[1]);
        check(back.width == 17 && back.height == 9 && back.cell == 8 && back.cols == 2 && back.rows == 1 &&
              back.perf == Perf::slow && back.frames.size() == 2, "SCFL header round trip");
        for (int i=0; i<2; ++i) check(back.frames[i].frame == r.frames[i].frame &&
            back.frames[i].status == r.frames[i].status && back.frames[i].cells == r.frames[i].cells, "SCFL frame round trip");
        // Wrong magic/version/size are the same reader boundary as SCMO.
        for (const int offset : {0, 4, 96}) {
            write_records(argv[1], r);
            std::fstream out(argv[1], std::ios::in | std::ios::out | std::ios::binary);
            out.seekp(offset); out.put('X'); out.close();
            bool caught = false;
            try { (void)read_records(argv[1]); } catch (const std::runtime_error&) { caught = true; }
            check(caught, "SCFL reader rejection");
        }
        std::filesystem::remove(argv[1]);
        std::cout << "cell reduction and SCFL I/O PASS\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

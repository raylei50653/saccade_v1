// Finite happens-before model of the bounded native detection session.
// Independent of CUDA: checks buffer conflicts, event generations and cleanup.
#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Access { int node; std::string resource; bool write; };
struct Graph {
    std::vector<std::vector<int>> edges;
    std::vector<Access> accesses;
    int node() { edges.emplace_back(); return static_cast<int>(edges.size()) - 1; }
    void edge(int a, int b) { if (a >= 0) edges.at(a).push_back(b); }
    void access(int n, std::string r, bool w) { accesses.push_back({n, std::move(r), w}); }
    bool valid() const {
        const auto n = edges.size();
        std::vector<std::vector<bool>> reach(n, std::vector<bool>(n));
        for (std::size_t a = 0; a < n; ++a)
            for (int b : edges[a]) reach[a][static_cast<std::size_t>(b)] = true;
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t a = 0; a < n; ++a) if (reach[a][k])
                for (std::size_t b = 0; b < n; ++b) reach[a][b] = reach[a][b] || reach[k][b];
        for (std::size_t a = 0; a < n; ++a) if (reach[a][a]) return false;
        for (const auto& a : accesses) for (const auto& b : accesses)
            if (a.node != b.node && a.resource == b.resource && (a.write || b.write) &&
                !reach[static_cast<std::size_t>(a.node)][static_cast<std::size_t>(b.node)] &&
                !reach[static_cast<std::size_t>(b.node)][static_cast<std::size_t>(a.node)]) return false;
        return true;
    }
};
enum Mutation { None, Ready, Consumed, Stage, HostPacket, DetectTail, WrongGeneration, Count };
constexpr std::array<const char*, Count> labels{
    "baseline", "remove-ready-wait", "remove-consumed-wait", "remove-staging-host-wait",
    "remove-result-host-wait", "remove-detect-terminal-wait", "bind-previous-ready-generation"};
bool check(unsigned slots, unsigned results, bool partial_tail, Mutation mutation) {
    Graph g;
    int h = -1, d = -1, m = -1;
    auto on = [&](int& stream) { int v = g.node(); g.edge(stream, v); stream = v; return v; };
    std::vector<int> consumed(slots, -1), ready(slots, -1), packets(results, -1);
    int staged = -1;
    unsigned received = 0;
    auto receive = [&](unsigned frame) {
        int host = on(h);
        if (mutation != HostPacket) g.edge(packets[frame % results], host);
        g.access(host, "HP" + std::to_string(frame % results), false);
    };
    constexpr unsigned frames = 10;
    for (unsigned f = 0; f < frames + static_cast<unsigned>(partial_tail); ++f) {
        if (f >= results && received < frames) receive(received++);
        int host = on(h);
        if (mutation != Stage) g.edge(staged, host);
        g.access(host, "STAGE", true);
        int decode = on(d); g.edge(host, decode);
        g.access(decode, "STAGE", false); g.access(decode, "RAW", true);
        int resize = on(d); g.access(resize, "RAW", false); g.access(resize, "INPUT", true);
        staged = resize;
        if (f == frames) break; // host/API failure after enqueueing D-only work
        int inference = on(d); g.access(inference, "INPUT", false); g.access(inference, "RAWP", true);
        int aggregate = on(d); g.access(aggregate, "RAWP", false); g.access(aggregate, "AGG", true);
        int output = on(d);
        if (mutation != Consumed) g.edge(consumed[f % slots], output);
        g.access(output, "RAWP", false); g.access(output, "AGG", false);
        g.access(output, "CS" + std::to_string(f % slots), true);
        int old_ready = ready[f % slots];
        ready[f % slots] = output; // record generation f is enqueued before M wait
        int finalize = on(m);
        if (mutation != Ready) {
            int generation = mutation == WrongGeneration && old_ready >= 0 ? old_ready : ready[f % slots];
            g.edge(generation, finalize);
        }
        g.access(finalize, "CS" + std::to_string(f % slots), false);
        consumed[f % slots] = finalize;
        int copy = on(m); g.access(copy, "HP" + std::to_string(f % results), true);
        packets[f % results] = copy;
    }
    while (received < frames) receive(received++);
    int release = on(h);
    if (mutation != DetectTail) g.edge(d, release);
    g.edge(m, release);
    for (const std::string resource : {"STAGE", "RAW", "INPUT", "RAWP", "AGG"})
        g.access(release, resource, true);
    for (unsigned s = 0; s < slots; ++s) g.access(release, "CS" + std::to_string(s), true);
    for (unsigned r = 0; r < results; ++r) g.access(release, "HP" + std::to_string(r), true);
    return g.valid();
}
}
int main() {
    try {
        unsigned configurations = 0;
        std::array<unsigned, Count> rejected{};
        for (unsigned slots = 1; slots <= 3; ++slots)
            for (unsigned results = slots + 1; results <= slots + 3; ++results)
                for (bool tail : {false, true}) {
                    ++configurations;
                    if (!check(slots, results, tail, None)) throw std::runtime_error("baseline conflict");
                    for (int mutation = 1; mutation < Count; ++mutation)
                        if (!check(slots, results, tail, static_cast<Mutation>(mutation))) ++rejected[mutation];
                }
        for (int mutation = 1; mutation < Count; ++mutation) {
            std::cout << labels[mutation] << " rejected=" << rejected[mutation] << '/' << configurations << '\n';
            if (rejected[mutation] == 0) throw std::runtime_error("mutation was never detected");
        }
        std::cout << "PASS configurations=" << configurations << " mutations=" << Count - 1 << '\n';
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}

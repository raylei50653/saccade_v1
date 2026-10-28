#include "saccade/cleanup.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::size_t checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void checked_failure(const saccade::CleanupReport& report, const std::string& expected) {
    bool rejected = false;
    try { report.throw_if_failed(); }
    catch (const std::runtime_error& error) { rejected = error.what() == expected; }
    check(rejected, "cleanup must preserve and propagate the first release error");
}
void tests() {
    // Simulate API failures at each independent kind of resource. These test the
    // exact ownership/error collector used by Session, not injected CUDA faults.
    const std::array<std::string, 6> kinds{"plan", "device", "pinned", "event", "detect", "main"};
    for (int failing = -1; failing < static_cast<int>(kinds.size()); ++failing) {
        std::array<int, 6> handles{11,22,33,44,55,66};
        saccade::CleanupReport report;
        std::vector<int> attempts;
        for (std::size_t i = 0; i < handles.size(); ++i) {
            report.release(handles[i], [&](int handle) {
                check(handles[i] == 0, "ownership consumed before release API call");
                attempts.push_back(handle);
                if (static_cast<int>(i) == failing) throw std::runtime_error(kinds[i]);
            });
        }
        check(attempts == std::vector<int>({11,22,33,44,55,66}), "remaining resources attempted after a release error");
        check(report.attempts() == 6 && report.failures() == (failing < 0 ? 0U : 1U), "release outcomes accounted");
        for (auto& handle : handles)
            report.release(handle, [](int) { throw std::logic_error("double release"); });
        check(report.attempts() == 6, "fallback never retries an attempted handle including failed releases");
        if (failing < 0) report.throw_if_failed();
        else {
            checked_failure(report, kinds[failing]);
            checked_failure(report, kinds[failing]); // A second close cannot turn failure into success.
        }
    }
    saccade::CleanupReport several;
    int a = 1, b = 2, c = 3;
    several.release(a, [](int) { throw std::runtime_error("first"); });
    several.release(b, [](int) { throw 9; }); // Also catch non-std exceptions during fallback.
    several.release(c, [](int) {});
    check(several.failures() == 2 && several.attempts() == 3, "multiple failures do not suppress later cleanup");
    checked_failure(several, "first");
    int marker = 0;
    void* pointer = &marker;
    several.release(pointer, [&](void* p) { check(p == &marker && pointer == nullptr, "pointer ownership consumed"); });
    check(pointer == nullptr, "pointer handle empty after release");
}
} // namespace
int main() {
    try {
        tests();
        std::cout << "cleanup_tests: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cleanup_tests: " << error.what() << '\n';
        return 1;
    }
}

#include "saccade/synthetic_runtime.hpp"
#include <cuda_runtime.h>
#include <cufft.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {
enum Category { plan, device, host, event, stream, category_count };
constexpr std::array<const char*, category_count> names{
    "cufftDestroy", "cudaFree", "cudaFreeHost", "cudaEventDestroy", "cudaStreamDestroy"};
struct Observation {
    int selected = -1, injected = 0;
    bool duplicate = false, real_failure = false;
    std::array<int, category_count> calls{};
    std::array<std::set<std::uintptr_t>, category_count> handles;
} observation;
std::size_t checks = 0;
void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
// Linker wrappers always release through the real API first. Only its returned
// status is then changed once: no memory fault or driver failure is fabricated.
bool released(Category category, std::uintptr_t handle, bool real_success) {
    ++observation.calls[category];
    if (!observation.handles[category].insert(handle).second) observation.duplicate = true;
    if (!real_success) observation.real_failure = true;
    if (real_success && observation.selected == category && observation.injected == 0) {
        ++observation.injected;
        return true;
    }
    return false;
}
} // namespace

extern "C" cufftResult __real_cufftDestroy(cufftHandle);
extern "C" cufftResult __wrap_cufftDestroy(cufftHandle handle) {
    const auto result = __real_cufftDestroy(handle);
    return released(plan, static_cast<std::uintptr_t>(handle), result == CUFFT_SUCCESS)
        ? CUFFT_INTERNAL_ERROR : result;
}
#define CUDA_RELEASE_WRAPPER(name, type, category) \
extern "C" cudaError_t __real_##name(type); \
extern "C" cudaError_t __wrap_##name(type handle) { \
    const auto result = __real_##name(handle); \
    return released(category, reinterpret_cast<std::uintptr_t>(handle), result == cudaSuccess) \
        ? cudaErrorUnknown : result; \
}
CUDA_RELEASE_WRAPPER(cudaFree, void*, device)
CUDA_RELEASE_WRAPPER(cudaFreeHost, void*, host)
CUDA_RELEASE_WRAPPER(cudaEventDestroy, cudaEvent_t, event)
CUDA_RELEASE_WRAPPER(cudaStreamDestroy, cudaStream_t, stream)
#undef CUDA_RELEASE_WRAPPER

namespace {
std::array<int, category_count> run_case(int selected, const char* label,
                                       const std::array<int, category_count>* expected = nullptr) {
    observation = {};
    observation.selected = selected;
    saccade::SyntheticConfig config;
    config.frames = 1;
    config.delay_cycles = 0;
    config.terminal_snapshot = false;
    bool threw = false;
    std::string message;
    try {
        const auto run = saccade::run_synthetic(config);
        check(selected < 0, "release error must escape checked Session::close");
        check(run.d_drained && run.m_drained, "successful session drains both streams");
        check(!run.ledger.empty() && run.ledger.back() == "finalize resources_released",
              "successful session returns only after checked resource release");
        check(run.packets.size() == saccade::packet_bytes, "normal one-frame output retained");
        const auto packet = saccade::decode_packet(run.packets);
        check(packet.frame == 0 && packet.status == saccade::ResultStatus::ok && packet.tracks.empty(),
              "normal session retains independent empty-result fixture");
    } catch (const std::exception& error) { threw = true; message = error.what(); }
    check(threw == (selected >= 0), "exactly injected cleanup runs throw");
    if (selected >= 0)
        check(message.starts_with(std::string(names[selected]) + "(handle): "),
              "cleanup exception retains exact failing API context: " + message);
    check(observation.injected == (selected >= 0 ? 1 : 0), "selected release error injected exactly once");
    check(!observation.real_failure, "all real releases succeeded before synthetic return-code injection");
    check(!observation.duplicate, "no resource handle retried during close or destructor fallback");
    for (const auto count : observation.calls) check(count > 0, "every independent release category attempted");
    check(observation.calls[plan] == 1 && observation.calls[host] == 2 && observation.calls[stream] == 2,
          "plan, pinned buffers and both distinct streams released");
    if (expected) check(observation.calls == *expected, "release error preserves all independent cleanup attempts");
    std::cout << "case=" << label << " injected=" << observation.injected << " releases=";
    for (int category = 0; category < category_count; ++category)
        std::cout << (category == 0 ? "" : "/") << observation.calls[category];
    if (threw) std::cout << " caught=" << message;
    std::cout << '\n';
    return observation.calls;
}
} // namespace

int main() {
    try {
        const auto baseline = run_case(-1, "normal-before");
        for (int category = 0; category < category_count; ++category)
            run_case(category, names[category], &baseline);
        run_case(-1, "normal-after", &baseline);
        std::cout << "cuda_cleanup_tests: " << checks << " checks, 7 GPU sessions passed"
                  << " (synthetic release return codes; no sanitizer or real driver fault injection)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_cleanup_tests: " << error.what() << '\n';
        return 1;
    }
}

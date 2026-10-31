#include "saccade/detail/detection_session_kernels.hpp"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using namespace saccade;
using namespace saccade::detail;
unsigned checks = 0;
void require(bool ok, const char* why) { ++checks; if (!ok) throw std::runtime_error(why); }
void cu(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
struct Resources {
    cudaStream_t stream = nullptr;
    DetectionCandidates* source = nullptr;
    DetectionSessionResult* result = nullptr;
    DetectionSessionControl* control = nullptr;
    Resources() {
        cu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        cu(cudaMalloc(&source, sizeof(*source))); cu(cudaMalloc(&result, sizeof(*result)));
        cu(cudaMalloc(&control, sizeof(*control)));
    }
    ~Resources() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (source) (void)cudaFree(source);
        if (result) (void)cudaFree(result);
        if (control) (void)cudaFree(control);
        if (stream) (void)cudaStreamDestroy(stream);
    }
    void close() {
        cu(cudaStreamSynchronize(stream));
        cu(cudaFree(source)); source = nullptr; cu(cudaFree(result)); result = nullptr;
        cu(cudaFree(control)); control = nullptr; cu(cudaStreamDestroy(stream)); stream = nullptr;
    }
};
constexpr std::uint64_t epoch = 0xF010F04;
DetectionCandidates candidate(std::uint64_t frame) {
    DetectionCandidates c{};
    c.header.epoch = epoch; c.header.frame = frame;
    c.header.schema = kPostprocessSchema; c.header.capacity = kPostprocessCapacity;
    c.header.done = kPostprocessDone; c.header.count = 2; c.header.kept = 1; c.header.suppressed = 1;
    c.rows[0] = {1,2,3,4,0.5f,0,83,kDetectionKept};
    c.rows[1] = {2,3,4,5,0.4f,0,79,kDetectionSuppressed};
    return c;
}
DetectionSessionResult run(Resources& r, DetectionCandidates c, std::uint64_t frame) {
    cu(cudaMemcpyAsync(r.source, &c, sizeof(c), cudaMemcpyHostToDevice, r.stream));
    cu(cudaMemsetAsync(r.result, 0xAB, sizeof(*r.result), r.stream));
    detection_session_finalize(r.source, r.result, r.control, epoch, frame, kPostprocessCapacity, r.stream);
    DetectionSessionResult out;
    cu(cudaMemcpyAsync(&out, r.result, sizeof(out), cudaMemcpyDeviceToHost, r.stream));
    cu(cudaStreamSynchronize(r.stream));
    require(out.epoch == epoch && out.frame == frame && out.reserved == 0, "result identity/reserved");
    return out;
}
void failure(const DetectionSessionResult& out, DetectionSessionStatus status, std::uint64_t first,
             std::uint64_t committed, std::uint32_t code) {
    require(out.status == status && out.fail_frame == first && out.committed_frame == committed && out.fail_code == code,
            "first failure and commit must remain latched");
    DetectionCandidates expected{};
    expected.header.epoch = epoch; expected.header.frame = out.frame;
    expected.header.schema = kPostprocessSchema; expected.header.capacity = kPostprocessCapacity;
    expected.header.done = kPostprocessDone; expected.header.flags = code;
    require(std::memcmp(&out.candidates, &expected, sizeof(expected)) == 0, "complete failed candidates must be initialized/zero");
}
void suite() {
    cu(cudaSetDevice(0)); detection_session_load_kernels(); Resources r;
    detection_session_initialize(r.control, r.stream);
    auto c = candidate(0); auto ok = run(r, c, 0);
    require(ok.status == DetectionSessionStatus::Ok && ok.fail_code == 0 && ok.fail_frame == kDetectionSessionNoFrame &&
            ok.committed_frame == 0, "initial success commits frame zero");
    require(std::memcmp(&ok.candidates, &c, sizeof(c)) == 0, "copy complete candidate bytes incl suppressed/padding");
    // Every source failure must lock, even after a prior successful commit;
    // subsequent different failure and success cannot change the first error.
    for (auto flag : {kPostprocessNumeric, kPostprocessUpstream, kPostprocessProtocol, kPostprocessOverflow}) {
        detection_session_initialize(r.control, r.stream); (void)run(r, candidate(0), 0);
        auto bad = candidate(1); bad.header.flags = flag;
        auto out = run(r, bad, 1); failure(out, DetectionSessionStatus::Failed, 1, 0, flag);
        bad = candidate(2); bad.header.flags = flag | kPostprocessProtocol;
        out = run(r, bad, 2); failure(out, DetectionSessionStatus::SkippedAfterFailure, 1, 0, flag);
        out = run(r, candidate(3), 3); failure(out, DetectionSessionStatus::SkippedAfterFailure, 1, 0, flag);
    }
    for (unsigned mutation = 0; mutation < 11; ++mutation) {
        detection_session_initialize(r.control, r.stream); auto bad = candidate(0);
        switch (mutation) {
        case 0: ++bad.header.epoch; break;
        case 1: ++bad.header.frame; break;
        case 2: ++bad.header.schema; break;
        case 3: --bad.header.capacity; break;
        case 4: bad.header.done = 0; break;
        case 5: bad.header.count = kPostprocessCapacity + 1; break;
        case 6: bad.header.kept = 0; break;
        case 7: bad.header.reserved = 1; break;
        case 8: bad.header.flags = 1u << 31; break;
        case 9: bad.header.kept = std::numeric_limits<std::uint32_t>::max(); bad.header.suppressed = 3; break;
        case 10: break;
        }
        auto out = run(r, bad, mutation == 10 ? 1 : 0);
        const auto expected_code = kPostprocessProtocol | bad.header.flags;
        failure(out, DetectionSessionStatus::Failed, mutation == 10 ? 1 : 0, kDetectionSessionNoFrame, expected_code);
    }
    // Reinitialization is only permitted at a new quiescent session boundary.
    detection_session_initialize(r.control, r.stream); ok = run(r, candidate(0), 0);
    require(ok.status == DetectionSessionStatus::Ok && ok.fail_code == 0 && ok.committed_frame == 0,
            "fresh control after failure must be isolated");
    r.close();
}
}
int main() {
    try { suite(); std::cout << "PASS checks=" << checks << " policy=f01-f04-session/1 scope=session-control-kernels\n"; return 0; }
    catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << '\n'; return 1; }
}

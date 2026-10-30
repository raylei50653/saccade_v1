#pragma once

#include "saccade/protocol.hpp"
#include "saccade/scan_frame_kernels.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace saccade {

// Frame harness (docs/SCAN_TRT.md §4, docs/HEAD_TRT.md §5): a deserialized
// engine on the D stream, its status outputs consumed by the frame-end sequence
// on the M stream, one ResultPacket v1 D2H per frame and host consumption in
// frame order. Engines must have static shapes and linear device FP32 inputs
// and outputs. Without a FrameIoSpec the engine is an S03 one: exactly one FP32
// output (the payload) and 1..4 INT32[2] status outputs, inferred from it.

// Explicit engine I/O of a frame session (S04): the engine must have exactly
// these tensors, no other and none twice. Outputs live in one slot arena in this
// order at 256-byte aligned offsets and are published together; statuses are
// read by the gate in this order, followed by the harness's own check statuses.
struct FrameTensorSpec {
    std::string name;
    std::vector<std::int64_t> shape;
};
struct FrameStatusSpec {
    std::string name, producer;
};
struct FrameIoSpec {
    std::vector<FrameTensorSpec> inputs, outputs;
    std::vector<FrameStatusSpec> statuses;
    // GPU finite checks on D after the engine: every input element, every
    // output element; each writes its own status (kFrameNonFiniteInput/Output).
    bool check_inputs = false, check_outputs = false;
};
inline constexpr const char* kFrameInputCheck = "frame_input_check";
inline constexpr const char* kFrameOutputCheck = "frame_output_check";

struct ScanFrameConfig {
    int slots = 2;                  // engine slots: execution context, I/O buffers, staging, ready/consumed events
    int packet_slots = 3;           // host ResultPacket ring (N_RP)
    bool graphs = false;            // launch the slot's CUDA graph, captured at initialization, instead of enqueueV3
    bool fixed_submit = false;      // keep submitting after a FAILED packet to observe the SKIPPED tail
    unsigned gate_delay_cycles = 0; // M busy-waits before reading a slot (ownership stress)
    std::int64_t max_frames = 256;  // frames per run and diagnostic record capacity
};

// One engine slot's device buffers: inputs in input order, the output arena
// (`payload`, all outputs in output order) and each output inside it, then the
// statuses in gate order (engine statuses, then harness check statuses).
struct ScanFrameSlotView {
    std::vector<float*> inputs;
    float* payload = nullptr;
    std::vector<float*> outputs;
    std::vector<ScanStatus*> statuses;
};

// Test hooks on D for `frame`: `hook` runs after the engine and before the
// harness checks, `after_checks` after the checks; both before the ready
// event. Directed tests overwrite outputs or statuses or plant a device fault.
using ScanFrameHook = std::function<void(std::int64_t frame, const ScanFrameSlotView& slot, cudaStream_t d)>;

struct ScanFrameOutcome {
    std::int64_t frame = -1;
    ResultStatus status = ResultStatus::failed;
    Error frame_error = Error::none, fail_code = Error::none;
    bool accepted = false; // OK packet of this frame: the consumer published its result
};

struct ScanFrameRun {
    std::uint64_t epoch = 0;
    std::vector<std::uint8_t> packets;    // raw D2H bytes in frame order
    std::vector<ScanFrameOutcome> outcomes;
    std::vector<std::int64_t> published;  // frames the consumer accepted, in order
    std::int64_t submitted = 0;
    bool stopped_on_failure = false;
    // Terminal snapshot after both drains; diagnostic, never frame traffic.
    ScanFrameControl control{};
    std::vector<float> committed_payload; // the committed result version
    std::int64_t committed_payload_frame = -1;
    std::vector<ScanFrameRecord> records; // one per submitted frame
    std::vector<std::vector<float>> slot_payloads;
    std::vector<std::vector<ScanStatus>> slot_statuses;
    std::vector<std::string> ledger;
};

class ScanFrameSession {
public:
    // Initialization stage: plugin registration, deserialization, one execution
    // context per slot with its own device memory, buffers, events, kernel
    // preload in this CUDA context, one warmup enqueue per slot, graph capture
    // when configured, then a full device synchronization.
    // With `spec`, the engine's I/O must match it exactly (names, modes, types,
    // shapes); otherwise construction throws before any frame work.
    ScanFrameSession(std::span<const std::uint8_t> engine, const ScanFrameConfig& config, nvinfer1::ILogger& logger,
                     nvinfer1::IErrorRecorder* recorder = nullptr, const FrameIoSpec* spec = nullptr);
    ~ScanFrameSession();
    ScanFrameSession(const ScanFrameSession&) = delete;
    ScanFrameSession& operator=(const ScanFrameSession&) = delete;

    const std::vector<std::string>& input_names() const;
    std::size_t input_elements(std::size_t input) const;
    std::size_t payload_elements() const; // the whole output arena
    // Outputs in arena order: name, element offset in the arena, element count.
    struct OutputSlot { std::string name; std::size_t offset, count; };
    const std::vector<OutputSlot>& outputs() const;
    // Status names in gate order, with their producers.
    const std::vector<FrameStatusSpec>& statuses() const;
    std::string identity() const;
    // Captured graph of a slot, one node per line in topological order, or "".
    std::string graph_nodes(int slot) const;
    // Engine context of a slot, for direct checks of the initialized state.
    nvinfer1::IExecutionContext& context(int slot);
    ScanFrameSlotView slot(int slot) const;
    cudaStream_t detect_stream() const;

    // Runs frames[k][i] = host values of engine input i for frame k, from a
    // quiescent reset: new epoch, initialized control, full synchronization.
    ScanFrameRun run(const std::vector<std::vector<std::vector<float>>>& frames, const ScanFrameHook& hook = {},
                     const ScanFrameHook& after_checks = {});
    // The last run as far as it got, also after run() threw: outcomes and
    // published frames up to the failure.
    const ScanFrameRun& last_run() const;
    // Checked release after the last run; throws the first release error.
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace saccade

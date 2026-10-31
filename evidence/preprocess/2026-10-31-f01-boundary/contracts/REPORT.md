# F01 contracts and diagnostic observability boundary

Scope: documentation and local SDK/header review only. No GPU commands were run
for this sub-bundle. The compile probe is syntax-only, not execution evidence.
F01 initcheck FAIL and P01 OPEN are unchanged.

## Stream contract versus implementation proof

The current [nvJPEG API documentation](https://docs.nvidia.com/cuda/nvjpeg/index.html#nvjpegdecodebatched)
specifies that `nvjpegDecodeBatched` is asynchronous to the host and submits its
GPU work to the caller's stream. This is an SDK contract, including the selected
HARDWARE backend; it does not disclose the internal hardware completion protocol.
It supports the caller's use of stream ordering for consumers and synchronization
before reusing state/resources. It does not prove the private driver's fence
implementation from independently observed events.

The caller boundary is therefore separate from the implementation boundary:
checking caller lifetime, stream, plane capacity and supported input establishes
compliance with a documented API under the SDK contract. A private wait primitive
need not be reverse engineered to justify that contract-based caller design.
The narrower remaining question is whether the observed initcheck diagnostic
signals a violation of that contract or lacks visibility into correct SDK work.

## Data-preserving initialization annotation exists

The official [NVTX Memory API guide](https://docs.nvidia.com/compute-sanitizer/SanitizerNvtxGuide/index.html#mark-initialized-api)
provides `nvtxMemCudaMarkInitialized`: a stream-scoped annotation excluding
specified ranges from Initcheck's uninitialized-memory checks. It is supported
through NVTX integration; the guide states CUDA initialization must precede NVTX
use. NVTX integration is enabled by default from CUDA 12.0; an explicit
`--nvtx yes` makes the capture intent clear.

This is not a pixel initialization operation and not an independent correctness
oracle. The call returns void. A successful executable exit does not show the
tool consumed the annotation. A fresh positive control with actual uninitialized
memory plus NVTX-disabled/annotation-absent negative controls must establish the
mechanism on the tested environment.

Local availability and exact API layout are pinned in
[local-sdk-identity.json](local-sdk-identity.json):

- `/opt/cuda/include/nvtx3/nvToolsExtMemCudaRt.h`, lines 172-194: batch fields,
  CUDA stream and per-thread-stream flag.
- `/opt/cuda/include/nvtx3/nvToolsExtMem.h`, lines 207-212: virtual range descriptor.
- `/opt/cuda/compute-sanitizer/docs/SanitizerNvtxGuide/index.html`, section
  `mark-initialized-api`: official local example uses virtual-address ranges
  directly; no heap registration is required by that example.

[nvtx_mark.hpp](nvtx_mark.hpp) implements one bounded diagnostic form: annotate
only the three RGB bytes of each measured RGBA pixel on an explicit stream.
It performs no CUDA API call, copy, memset, synchronization, allocation query or
device write. It does allocate host descriptor storage. The caller must validate
the current SDK allocation identity, pointer/offset/capacity and stream; the helper
cannot establish them. [compile-probe.json](compile-probe.json) records a clean
syntax-only compile with the installed headers.

## Minimal discriminating experiment

Keep the original unsuppressed binary/control intact. Build a separate diagnostic
with immutable source/binary identity, actual command/exit/log, SDK loaded-module
identity, fixed input/reference hashes and complete API cleanup checks.

1. Discover the converter's actual private input pointer from copied launch
   arguments or an observed allocation, rather than a hardcoded prior-process
   address. Match its allocation generation before every intervention. PreAllocate
   is only a hint: the nvJPEG documentation permits subsequent buffer resizing.
2. Hold the intervention point fixed. Compare annotation absent, exact valid RGB
   ranges, alpha-only ranges and caller-output ranges; optionally compare whole
   allocation. Use `--nvtx yes` and `--nvtx no` in distinct processes. Expected
   error-count differences are hypotheses to test, not acceptance requirements
   inferred from the desired result.
3. Use at least two distinct JPEGs with independently generated fixed RGB
   references, alternating them in the same live session. Check every output
   byte and preserve original errors, annotation logs and failures.
4. Contrast annotation with A5/5A poison only in separate diagnostics. Record
   every added synchronization: it can strengthen ordering and change what is
   being tested. Annotation alone leaves data untouched, but changes tool checks
   by design and adds host execution overhead.

If exact RGB marking removes the converter's reports without changing output,
while alpha-only/caller-output controls do not, this localizes the report to
tracked state for the consumed private RGB bytes. It does not independently show
that a producer ran, identify its internal engine or prove all schedules safe.
Two-image alternation and poison make stale or uninitialized output explanations
less plausible for the measured runs; neither gives a universal race proof.

## CUPTI capture restrictions

[CUPTI callback documentation](https://docs.nvidia.com/cupti/main/main.html#cupti-callback-api)
supports observing driver/runtime entry/exit and launch arguments. CUDA API calls
inside callbacks are generally unsupported except explicitly documented
synchronization calls. Thus do not call allocation queries or device copies from
a generic CUPTI callback. Copy launch argument bytes while they are valid, then
inspect allocation metadata at a supported application/instrumentation point.
An added synchronization inside an allowed callback remains a diagnostic change.

[CUpti_CallbackData](https://docs.nvidia.com/cupti/api/structCUpti__CallbackData.html)
provides `functionParams`, `symbolName`, entry/exit and correlation metadata.
Argument storage is valid during the callback; a shallow pointer copy cannot be
dereferenced afterwards. The local `generated_cuda_meta.h` defines ordinary and
extended launch parameter layouts. Handle both `kernelParams` and `extra` forms
according to the API actually observed, rather than assuming one ABI.

Compute Sanitizer's [known limitations](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#known-limitations)
state that its tools cannot interoperate with other CUDA developer tools.
CUPTI/Nsight tracing and Initcheck should therefore be captured separately unless
a specific supported combination is independently established. Cross-process
addresses or allocation generations cannot be copied from one capture to another.

## Remaining proof obligations

- Connect the observed converter argument and allocation generation to the
  static SASS/address model and the actual loaded SDK binary.
- Observe the userspace hardware submission/completion dependency, or identify
  exactly where it crosses an opaque interface. A public trace with no producer
  event does not prove that only NVJPG can be the producer.
- Distinguish untracked correct writes from an SDK/driver initialization or
  synchronization defect. Annotating a range deliberately excludes checks and
  cannot resolve this by itself.
- NVIDIA classification, other SDK/driver/GPU environments, and complete
  unprinted original session diagnostics remain separate obligations.

Finite runs establish the measured data and ordering. General synchronization
correctness additionally needs an applicable contract or a supported internal
ordering argument. This review supplies the public contract and a stronger
diagnostic intervention; it does not waive the original acceptance failure.

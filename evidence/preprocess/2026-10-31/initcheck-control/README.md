# Independent nvJPEG HW initcheck control

2026-10-31. Repository source stayed clean at
`2b39ddaf7cefcbdfb8a7fc143ecfd83466618cb3`; every control artifact is in this
ignored directory. This control links CUDA runtime and nvJPEG only: no Saccade
preprocessor, resize kernel, TensorRT, Python runtime or multistream pipeline.
It decodes the checked-in 35x27 baseline 4:2:0 JPEG twice on an explicit
nonblocking stream, synchronizes, copies to pinned host storage, verifies all
2,835 RGB bytes against independent `djpeg -dct int -nosmooth -rgb`, verifies
repeat equality and writes the raw planar RGB result.

## Result

| Decoder / allocator | Normal exit | Memcheck exit / errors | Initcheck exit / errors | Exact nosmooth RGB | Custom device allocation calls |
| --- | --- | --- | --- | --- | --- |
| HW, default | 0 | 0 / 0 | 99 / 1890 | yes | not installed |
| HW, V1 custom malloc, uninitialized | 0 | 0 / 0 | 99 / 1890 | yes | 0 |
| HW, V1 custom malloc + zero memset | 0 | 0 / 0 | 99 / 1890 | yes | 0 |
| HW, V1 custom malloc + 0xa5 memset | 0 | 0 / 0 | 99 / 1890 | yes | 0 |
| HW, V2 custom malloc, uninitialized | 0 | 0 / 0 | 99 / 1890 | yes | 0 |
| HW, V2 custom malloc + zero async memset | 0 | 0 / 0 | 99 / 1890 | yes | 0 |
| HW, V2 custom malloc + 0xa5 async memset | 0 | 0 / 0 | 99 / 1890 | yes | 0 |
| HYBRID, default, diagnostic only | 0 | 0 / 0 | 0 / 0 | no (93 bytes, max 2 LSB) | not installed |

All successful hardware-mode outputs, including outputs produced under both
sanitizers, have the same SHA-256 as independent CPU reference:
`761809bdcad84f79c1b01f0d2369bb4cf52f280cb2dff92be87c3ef9f11c7162`.
HYBRID output repeated exactly but differs from the required hardware reference;
it is a diagnostic control, not an approved fallback.

All hardware initcheck reports originate at the private
`nvjpeg::rgba2rgb<...NVJPEG_OUTPUT_RGB...>` kernel and its `nvjpegDecodeBatched`
call stack, the same location observed in the application capture. This
localizes the report to the SDK's hardware decode path independently of the
Saccade F02 kernel and its resources. It does not establish that the report is
a false positive. NVIDIA has not confirmed that classification here.

The initializer callbacks were **never invoked by hardware decode**, including
preallocation. Thus zero/a5 allocator scenarios did not initialize or poison the
private surface implicated by initcheck. They establish that neither V1 nor V2
user allocator provided access to that surface in this tested path; they do not
prove poison independence or provide an accepted initialization fix.

## Files and rerun

- `repro-v1.cpp` / `repro-v1`: first successful direct control; logs named
  `default-*`, `custom-dirty-*`, `custom-zero-*`, `custom-a5-*`.
- `repro.cpp` / `repro`: final control with valid V2 device **and pinned**
  allocators, plus HYBRID diagnostic; logs named `v2-*-try3-*` and
  `hybrid-try3-*`. `control3.sh` regenerates these logs after building.
- `control.sh` recorded the first valid control sequence; `control2.sh` recorded
  an intermediate experiment. Each `.command`, `.log`, `.exit`, `.time`
  preserves invocation, output, actual status and time/RSS.
- `versions*.log` captures GPU, CUDA compiler, sanitizer, library resolution and
  hashes. `output*.sha256` records the output bytes. `FILES.sha256` binds the
  final source, all scripts, binaries and raw artifacts.

Final direct reproduction (from repository root, on an idle GPU):

```sh
bash build-preprocess-initcheck-control/control3.sh
```

Tool identity: nvJPEG 13.2.3 (`libnvjpeg.so.13.2.3.58`), CUDA toolkit 13.4.92,
Compute Sanitizer 2026.3.0.0, RTX 5070 Ti Laptop GPU (SM 12.0), reported driver
617.42, WSL2. All GPU runs were sequential and coordinated with the root agent.

## Retained unsuccessful preparation attempts

`attempt1-compile-failure/` contains the first compile attempt (missing typed
CUDA allocation overloads when including `cuda_runtime_api.h` directly); no GPU
program ran in that attempt. Switching to `cuda_runtime.h` fixed compilation.

`repro-v2-unsupported.cpp` / binary retain an intermediate V2 call which passed
no pinned allocator: `nvjpegCreateExV2` returned status 2. Those `v2-*` logs are
not the final `*-try3-*` results. Providing both allocator objects made V2
creation valid, after which the hardware path still invoked no device callback.
The initial HYBRID check treated nosmooth mismatch as fatal (exit 1, initcheck
reported zero errors); the final diagnostic explicitly records that mismatch.

## Official documentation boundary

[NVIDIA Initcheck documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#what-is-initcheck)
says the tool recognizes initialization through device writes and CUDA
memcpy/memset calls. [nvJPEG documentation](https://docs.nvidia.com/cuda/nvjpeg/index.html)
describes the hardware backend and custom allocators. Searches of these documents,
[NVIDIA sanitizer release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html)
and [nvJPEG release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#nvjpeg-library)
did not locate a specific acknowledgement of this rgba2rgb/initcheck report.
A hardware surface written outside the tool's tracked operations remains a
plausible explanation, not a verified defect classification. Preserve the
unsuppressed initcheck failure and report this SDK/tool boundary as OPEN.

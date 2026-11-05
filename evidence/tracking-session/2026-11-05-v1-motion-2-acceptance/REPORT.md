# v1-motion/2 acceptance continuation — stopped at missing development input

Date: 2026-11-05 (Asia/Taipei). Worktree: `/home/ray/developer/ai/saccade_v1-worktrees/g1-ofa`, branch `g1-ofa-gmc`. Starting HEAD: `0cae6afa4121275605905de08ab684a394ac39ce`.

**Result: STOPPED. Development CTest failed while loading its frozen detection reference corpus. Configure and build passed; 41 tests passed, test 42 failed, and the remaining 23 tests were not run. No formal §20.8 acceptance was launched.** The requested stop-on-gate-failure rule applies; no input replacement, comparison relaxation, source workaround or subsequent GPU test was attempted.

This continues the [earlier failure report](../2026-11-04-v1-motion-2/REPORT.md), which is unchanged. The reviewer resolved that report's s=2 parity failure in **`0cae6af`**, “Order the flow parity test's setup before its non-blocking stream”: [complete commit and patch](development/harness-fix.patch), [current test source](../../../tests/flow_motion_cuda_tests.cpp). The reviewer recorded 20/20 passing runs at both sizes after the setup synchronization. That is prior reviewer evidence; this continuation did not reach or rerun the flow parity suite.

## Development gate and first failing input

The resumed development CMake cache retained:

`SACCADE_DETECTION_REFERENCES=/home/ray/developer/ai/saccade_v1/build-preprocess-evidence/refs`

That directory does not exist. I reused this cached path without first checking the required frozen manifest. The first failing stage is `Corpus::Corpus` in `tests/detection_session_tests.cpp`: it reads `manifest.json` before completing its first check. The failure is:

```text
detection_session=FAIL checks=0 error=cannot open input: /home/ray/developer/ai/saccade_v1/build-preprocess-evidence/refs/manifest.json
```

This is an unavailable test input, not an observed CUDA, OFA, packet or numerical mismatch. Neither `tests/detection_session_tests.cpp` nor production `src/detection_session.cpp` differs from main `23ade53` on this branch. No defect in the new flow implementation was established. No test or production source was changed. [Input diagnosis](development/first-failing-input.json), [CTest failure output](development/ctest.log), [complete executed test output](development/LastTest.log), [failed test identity](development/LastTestsFailed.log).

The GPU-lock command started at **08:55:28 +08:00**. Explicit configure preceded the build and CTest, all inside the same `flock ~/.cache/saccade-v1/gpu.lock timeout -k 10 3600` wrapper. CTest ran sequentially with `--output-on-failure --stop-on-failure`.

| Command | Exit | Result / time |
| --- | --- | --- |
| `cmake -S . -B build/development` | 0 | [Configure log](development/configure.log); TensorRT device probe under the lock |
| `cmake --build build/development --parallel 8` | 0 | [Build log](development/build.log) |
| `ctest --test-dir build/development --output-on-failure --stop-on-failure` | 8 | 229.70 seconds; 41 passed, 1 failed, 23 unexecuted |
| Enclosing locked timeout command | 8 | Stopped at CTest failure; no timeout or sanitizer termination |

[Exact command](development/command.txt), [exit and count record](development/exit.json), [CMake cache](development/CMakeCache.txt), [registered tests](development/CTestTestfile.cmake). Tests 1–41 passed, including the existing CUDA/TensorRT, resize, preprocessing, postprocessing and model-producer component tests. The seven T01 session packet comparisons, their two perturbation seeds, lifecycle, tracking suites, and all flow suites are among the unexecuted tests. A separate fresh CPU build was not started.

## Legacy-stream setup audit

Before launching development checks, all changed tests and tools since main `23ade53` were checked for legacy-stream memset/memcpy feeding nonblocking-stream work. [Complete file inventory and findings](development/new-test-tool-stream-audit.json).

- `tests/flow_motion_cuda_tests.cpp`: the affected `Device` constructor and `put` calls already synchronize after legacy-stream setup, as fixed by `0cae6af`. Downloads follow an explicit synchronization of the test stream.
- `tools/ofa_audit.cu`: copies and clears use the explicit nonblocking IO stream. Preparation and observation synchronize that stream; the order-test interval remains unchanged.
- `tools/flow_sequence.cpp` and `tests/flow_cuda_tests.cpp`: the Engine uses explicit-stream copies; sequence decoding uses the same stream.
- The other changed tests and tools contain no direct CUDA memset/memcpy setup calls. **No additional instance was found and no additional fix was applied.**

## §20.8 status

| Step | Result |
| --- | --- |
| 1. Clean source and identity | Formal step unexecuted. Starting source was clean at `0cae6af`; development source hashes were verified unchanged. Headers were rehashed and match the recorded specification. NVOF loading was not freshly exercised. |
| 2. Fresh builds, complete CTest, parity, audit and lifecycle | **Development prerequisite FAILED:** frozen reference directory missing at test 42. Fresh formal builds, seven packet comparisons, perturbation runs, lifecycle, flow parity and the repeated A1–A4 audit remain unexecuted. |
| 3. Seven MOT17 HR paths | Unexecuted. No fresh SCDR, MotionContext, packet or MOT comparisons. |
| 4. PP22 69-sequence regression and five 4K runs | Unexecuted. |
| 5. MOT17 and PP22 re-scoring; separate 4K scores | Unexecuted. |
| 6. Sanitizers and positive controls | Unexecuted. |
| 7. Stream placement and nsys OFA/F03 overlap | Unexecuted. **No overlap result.** |
| 8. Throughput against main `23ade53` | Unexecuted. Baseline worktree `../g1-ofa-base` was not created; no baseline or candidate timing and no serial latency result. The 3% threshold is unevaluated. |
| 9. Evidence report | This blocked-continuation report and raw development records are archived. It is not an acceptance certificate. |

The earlier report's §20.6 audit result remains **A1–A4 60/60 each; A5 recorded**, including the MOT17-10 SCFL comparison. Those are prior measurements, not a new audit from the resumed build. The raw prior audit is linked through the [earlier report](../2026-11-04-v1-motion-2/REPORT.md).

## Throughput

No unprofiled before/after runs were made after the failed development gate. Each sequence's prescribed three repetitions remains unexecuted.

| Sequence | Before ms/frame | After ms/frame | Change |
| --- | --- | --- | --- |
| MOT17-02-SDP | unmeasured | unmeasured | unavailable |
| MOT17-04-SDP | unmeasured | unmeasured | unavailable |
| MOT17-05-SDP | unmeasured | unmeasured | unavailable |
| MOT17-09-SDP | unmeasured | unmeasured | unavailable |
| MOT17-10-SDP | unmeasured | unmeasured | unavailable |
| MOT17-11-SDP | unmeasured | unmeasured | unavailable |
| MOT17-13-SDP | unmeasured | unmeasured | unavailable |
| Total | unmeasured | unmeasured | unavailable |

## 4K completion and scores

| Sequence | Completion | HOTA / DetA / AssA / IDF1 / MOTA / IDSW |
| --- | --- | --- |
| uid_vid_00179.mp4 | unexecuted | unmeasured |
| uid_vid_00189.mp4 | unexecuted | unmeasured |
| uid_vid_00218.mp4 | unexecuted | unmeasured |
| uid_vid_00226.mp4 | unexecuted | unmeasured |
| uid_vid_00230.mp4 | unexecuted | unmeasured |

## Custody and unchanged scope

Development output remains under `build/development/`. No retry build directories were created. No MOT, SCDR, SCMO, SCFL, raw gray or raw grid payload was copied to this evidence directory. [Binary hashes](development/binaries.sha256), [source hashes](development/source.sha256), [verified post-run source hashes](development/source-after.sha256), [header hashes](headers.sha256), [source identity](source.txt), [environment](environment.txt), [machine-readable status](status.json).

The acceptance runner update and its clean committed acceptance run were not attempted after the gate failure. `docs/TRACKING.md` has no new acceptance-result line. All existing docs/evidence, tracker/Kalman/F07/F08, packet wire format, PCR library, `motion_sequence`, and both evaluation scripts remain unchanged. No push or merge was performed. No GPU target remained running after the development command ended.

The blocker must be resolved with the exact frozen detection corpus before the full development gate can be rerun. This report does not authorize regenerating inputs, replacing the corpus, or skipping the detection test.

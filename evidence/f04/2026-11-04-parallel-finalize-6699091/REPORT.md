# F04 parallel finalize — blocked by the fixed workspace specification

Date: 2026-11-04. **Not accepted.** Implementation and tests are committed, but the literal workspace requirement in POSTPROCESS §5.2 is **FAIL** against the unchanged starting source. Work stopped after §5.7 step 1 when the contradiction was identified. Steps 2–6 are **UNEXECUTED**, not failed test runs or passing gates.

## Blocking specification contradiction

[POSTPROCESS.md §5.2](../../../docs/POSTPROCESS.md#52-不變的部分) requires both an unchanged workspace and `postprocess_workspace_bytes() == 295712`. The clean, unmodified starting commit's benchmark reports:

```text
source_commit=e14c94c49a7240efa428a62d299c33f78a0b72ec
source_dirty=false
workspace_bytes=698432
```

The final implementation also reports `workspace_bytes=698432`; `Workspace` and the function returning its size were not changed. See [baseline run](raw/runs/baseline-benchmark.log), [final development run](raw/runs/finalize-shared-kept-cuda.log), and [source excerpts](raw/blockers.txt).

The starting source sizes `Workspace` for the HR profile: 20,160 positions and 20 dense tiles. Its members occupy 698,432 bytes including alignment. The older 8,400-position, 9-tile layout totals 295,712 bytes. This explains the discrepancy, but does not authorize choosing which fixed requirement to override. Shrinking the current workspace would violate the unchanged-workspace requirement and the task's prohibition on changing the other kernels. The fixed specification must be clarified before acceptance continues.

This contradiction was noticed during the final identity review, after A/B and the CPU acceptance had completed. No acceptance runner was started after the discovery. This report preserves the completed work without claiming formal CUDA acceptance or performance acceptance.

## Source, scope and custody

| Item | Identity |
| --- | --- |
| Worktree | `/home/ray/developer/ai/saccade_v1-worktrees/f04-parallel` |
| Branch | `f04-parallel-finalize` |
| Starting source | `e14c94c49a7240efa428a62d299c33f78a0b72ec` |
| Commit A, tests only | `18eb1c4832129d0fe4d636dbfc7f646571da7de7` — Add ordered NMS failure and mask boundary cases |
| Commit B, implementation and mutations | `6699091add8787d7c21fe445d5581a189ca08692` — Parallelize finalize with ordered NMS bitmasks |
| Commit C | The commit containing this report and its small raw outputs |
| CPU formal source | Clean commit B; [source status](raw/acceptance-source-status.txt) and [stopped status](raw/stopped-source-status.txt) are empty |
| Source hashes | [policy/spec/reference/tests/implementation/scripts](raw/acceptance-source.sha256), [source scope](raw/source-scope.txt) |
| Replay identities | [six pinned dumps](raw/f03-dumps.sha256) |
| Uncommitted build artifacts | [paths and SHA-256](raw/build-artifacts.sha256); no binaries or build trees are committed |

The policy, CPU reference, public ABI/host validation, other four F04 kernels, tracking and session code, and existing documentation were left unchanged. No push was performed.

The baseline was built and timed before editing tracked source. Test-A runs are development evidence: the old CUDA library was configured at clean e14c94c, then linked to the newly changed test harness. Its embedded clean-source identity describes that library's original configuration, not a clean tree containing the new tests. The [test-run source hashes](raw/tests-current-source.sha256) identify the actual reference, CUDA source and tests used.

Final CUDA development runs embed `source_commit=18eb1c4...` and `source_dirty=true`; their final CUDA source is the source committed in B. They are separate from clean-B formal acceptance. The only §5.7 acceptance actually completed on clean B is the CPU suite.

## Implementation

Finalize now launches one fixed CTA of 256 threads. A blocked CUB exclusive scan preserves selected score/ID order while filtering class and degeneracy. Areas and all `j < i` pair results are computed in parallel, retaining the original `iou(i,j,area_i,area_j)` arithmetic and argument order.

Each pair contributes mutually exclusive failure/overlap bits to a 300 × 5-word mask of each kind. Thread 0 processes rows, words and the least decisive kept bit in the original order. Failures involving suppressed rows or pairs after an earlier suppression do not fail the frame. Mapping is parallel; the CTA completes its failure decision before output writes, reset and final done publication.

The final development identity reports **36,032 shared bytes, 0 local bytes, 48 registers** for finalize, within the 48 KiB requirement. The original finalize reports 1,200 local bytes and 0 shared bytes. The identity adds `finalize_strategy=cta-bitmask-greedy-v1`; `kernels_per_enqueue=5` remains unchanged. These resource figures are not timing measurements.

The mutation script retargets the three NMS variants and adds the two required variants. Three existing variants already had stale HR-era source needles at e14c94c: `wrong-p3-stride`, `dense-drop-last-tile` (both its bound and loop literals), and `dense-first512-only`. They now target the existing profile-aware stride, runtime tile count and runtime position bound, preserving the intended faulty behavior and diagnostics. This changes the disposable mutations, not those production kernels.

## Baseline timing before any code change

Commands and exits are archived under [raw/runs](raw/runs/). [run.sh](raw/run.sh) records the exact command, start/end timestamps, complete combined output and exit code. Every GPU command, including each CUDA development test, held `/home/ray/.cache/saccade-v1/gpu.lock` exclusively.

The [baseline build command](raw/runs/baseline-build.command) configured Release, CUDA architecture 120 and F04 at clean e14c94c, then built `postprocess_benchmark`. Exit **0**. The timed command was:

```sh
flock /home/ray/.cache/saccade-v1/gpu.lock \
  build/f04-base/postprocess_benchmark \
  /home/ray/developer/ai/saccade_v1/build/t01-f03-evidence/model-dump 32 7
```

Exit **0**; **12/12** sample/mode results PASS, each invocation has 5 kernel nodes and 0 other nodes. The GPU is an NVIDIA GeForce RTX 5070 Ti Laptop GPU, compute capability 12.0; nvcc 13.4.92, host compiler 16.2.1, CUDA runtime/driver API 13040, CUB 300500. Full identity and output hashes are in the [log](raw/runs/baseline-benchmark.log); [baseline binary hashes](raw/baseline-binaries.sha256) retain paths to the uncommitted artifacts.

GPU median µs per complete F04 invocation, across 7 round means of 32 iterations:

| Dump | Qualified | Before plain | Before graph batch | After plain / graph |
| --- | ---: | ---: | ---: | --- |
| F03-0 | 237 | 1059.63898 | 956.72899 | UNEXECUTED |
| F03-1 | 246 | 929.60697 | 917.56803 | UNEXECUTED |
| F03-2 | 285 | 1220.26706 | 1208.40096 | UNEXECUTED |
| F03-3 | 7 | 79.73000 | 40.34700 | UNEXECUTED |
| F03-4 | 5 | 71.51000 | 39.40000 | UNEXECUTED |
| F03-5 | 3 | 76.43000 | 37.15200 | UNEXECUTED |

[Raw CSV](raw/baseline-timing.csv) also includes round p95, wall medians and complete output hashes. Timing excludes H2D/D2H, F01–F03 and downstream tracking. No after benchmark or speedup is claimed.

## Five cases passed on the original finalize

[Test build](raw/runs/tests-current-build.command), [CPU command](raw/runs/tests-current-cpu.command), [CUDA command](raw/runs/tests-current-cuda.command), and [HR CUDA command](raw/runs/tests-current-cuda-hr.command) all exited **0**, before any finalize change and before commit A.

Let `W=2^62`, `H=2^63`, so one area unit is `2^125`. Near boxes start at x=0; far boxes start at x=8W and do not intersect near boxes. Each listed area is finite. A sum of 7.5 units is finite; 8.5 units overflows. Requests use identity mapping, so the mapping does not introduce overflow.

| Case | Independent construction and expected result | Original CPU / CUDA / HR |
| --- | --- | --- |
| `nms-uncompared-suppressed-pair` | Areas 3, 4, 4.5; A/B IoU=3/4, C/A=0; C/B overflows. A,C kept; B suppressed. | PASS / PASS / PASS |
| `nms-uncompared-after-decision` | Areas j1=2.5, j2=5, i=3.5; j1/j2 disjoint, i/j1 IoU=5/7, i/j2 overflows. i suppressed. | PASS / PASS / PASS |
| `nms-compared-failure-before-over` | Earlier disjoint kept rows have areas 4.5,3; i has area 4. First compared sum overflows before later IoU=3/4. Full numeric reset. | PASS / PASS / PASS |
| `nms-compared-over-before-failure` | Earlier rows swapped: i first sees IoU=3/4, so the later overflowing pair is unvisited. i suppressed. | PASS / PASS / PASS |
| `nms-word-boundaries` | Exactly 300 rows; 63→64→65 chain, 127→128, 191→192, 255→256, 257→258, 0→299. Suppressed IDs 64,128,192,256,258,299; 294 kept. | PASS / PASS / PASS |

The CPU tests hand-check counts, flags, IDs, scores, failure reset and zero tails. Each of the five CUDA cases additionally compares **all 9,664 output bytes** directly against the unchanged reference, alongside the existing repeat and graph byte comparisons. Scores in these fixtures are exactly 1 or 0.5, so no score tolerance is used for the new full-byte comparisons.

| Run | Result |
| --- | --- |
| [Original CPU log](raw/runs/tests-current-cpu.log) | 1,091 checks, five named cases PASS |
| [Original CUDA + F03 log](raw/runs/tests-current-cuda.log) | 54 cases; 37,382 checks; six replay hashes verified; held streams PASS; max score error 1.19209e-07 |
| [Original HR CUDA log](raw/runs/tests-current-cuda-hr.log) | 62 cases; 44,415 checks; held streams PASS; max score error 5.96046e-08 |
| [Final CUDA development log](raw/runs/finalize-shared-kept-cuda.log) | Same 54 cases, 37,382 checks and all six complete replay output hashes; held streams PASS |
| [Final HR CUDA development log](raw/runs/finalize-shared-kept-cuda-hr.log) | Same 62 cases and 44,415 checks; held streams PASS |

The final-source [literal-only mutation check](raw/runs/mutation-literals-final.command) exited **0**: **15/15 exact matches**. Its [results](raw/mutation-literals/results.tsv) explicitly say `MATCHES_ONLY_GPU_UNEXECUTED`. This is **not** 15/15 DETECTED, and no mutation was compiled or executed in that check.

## §5.7 acceptance status on commit B

| Step | Result | Evidence or reason |
| --- | --- | --- |
| 1. Full CPU CTest in `build/cpu` | **PASS** | Configure/build/CTest [command](raw/runs/step1-cpu.command), exit **0**, [log](raw/runs/step1-cpu.log), [LastTest.log](raw/cpu-LastTest.log): **16/16**, 47.39 s; postprocess reference 1,091 checks |
| 2. F04 acceptance runner | **UNEXECUTED** | Stopped for the §5.2 contradiction before starting the runner. Formal CUDA CTest, six formal replays, memcheck/initcheck/racecheck/synccheck and four positive controls remain unexecuted. |
| 3. Semantic mutations | **UNEXECUTED** | No 15/15 DETECTED claim. Only the earlier development literal check was run. |
| 4. Tracking session acceptance | **UNEXECUTED** | No new §14.5 output hashes or sanitizer results. Expected hashes were not changed. |
| 5. MOT17 evaluation + M4 hash comparison | **UNEXECUTED** | No new SCDR/MOT/offline outputs; `sha256sum -c` against M4's outputs was not run. |
| 6. Speed | **UNEXECUTED** | No new evaluation nsys trace, finalize average, F04 total, D-stream time, after benchmark or sequence timing. The required finalize ≤100 µs gate has no verdict. |
| 7. Evidence | Recorded here | This is a blocked-work report, not an acceptance certificate. |

The historical **1,539 µs** finalize average (grid 1, block 1), approximately **1,589 µs** F04 total and approximately **6.6 ms/frame** D-stream GPU work are the §5.1/M4 baseline, not new measurements from this task. All corresponding after values are UNEXECUTED. No new nsys report was produced or committed.

The seven historical M4 timings are retained for the requested comparison; the new side was not measured. Source: [M4 timing copy](raw/m4-timing.tsv), [original-file hashes](raw/m4-comparison-inputs.sha256).

| Sequence | Frames | M4 ms/frame | Commit B ms/frame |
| --- | ---: | ---: | --- |
| MOT17-02-SDP | 600 | 8.50913 | UNEXECUTED |
| MOT17-04-SDP | 1050 | 8.32949 | UNEXECUTED |
| MOT17-05-SDP | 837 | 7.3629 | UNEXECUTED |
| MOT17-09-SDP | 525 | 7.23846 | UNEXECUTED |
| MOT17-10-SDP | 654 | 8.45914 | UNEXECUTED |
| MOT17-11-SDP | 900 | 8.12514 | UNEXECUTED |
| MOT17-13-SDP | 750 | 8.34008 | UNEXECUTED |

The supplied environment is preserved in [acceptance-env.sh](raw/acceptance-env.sh). No scoring script was executed and no ground-truth contents were read. All builds and command outputs stayed under this worktree's `build/<name>/`; only the small evidence copies are committed. No build or acceptance retry failed, so there are no failed retry build directories to delete. Intermediate successful development builds are uncommitted and are not formal evidence.

## 範圍外

- The unchanged `run_postprocess_acceptance.sh` line 102 expects a sanitizer-log line matching `^checks=[0-9]+ max_score_abs_error=[0-9.e+-]+ PASS$`. The unchanged test summary format prints `max_score_abs_error=...` and `postprocess_cuda_tests: N checks passed` on separate lines. This is a pre-existing runner/test-summary mismatch, documented by [static source inspection](raw/blockers.txt). It was not repaired, bypassed or tested by invoking the runner; it is a separate obstacle to review after the workspace specification is resolved. No acceptance failure exit is attributed to this unexecuted runner.
- Existing documentation's 295,712-byte workspace value was not edited. The source's 698,432-byte workspace was not changed to fit it.

## Integrity

From this directory, `sha256sum -c SHA256SUMS` verifies this report and every archived raw output. `raw/build-artifacts.sha256` separately records paths and hashes for build artifacts retained only in this worktree; it does not imply those binaries are in this evidence directory. Commands, exits, identities, test logs and baseline timing remain available without committing build trees or large binaries.

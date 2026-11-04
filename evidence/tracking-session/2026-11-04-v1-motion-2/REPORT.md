# v1-motion/2 integration — stopped at kernel parity

Date: 2026-11-04 (Asia/Taipei). Worktree: `/home/ray/developer/ai/saccade_v1-worktrees/g1-ofa`, branch `g1-ofa-gmc`, starting commit `04a5547ec6899a10041f334795f1ede47643f1f5`.

**Result: incomplete. The §20.6 prerequisite OFA audit passed. Development kernel parity failed at s=2 cell reduction, so integration and acceptance work stopped under the requested gate rule. §20.8 formal acceptance was not launched.** This report preserves the failed comparison without changing its expected value or adding synchronization to make it pass.

## Source and scope

| Commit | Change |
| --- | --- |
| `d64cf05d89cd82dbe7c182f4628177031324616d` | Add the standalone OFA stream-order audit |
| `23e3ab9f61db17c89211f870c731d15dc9f621d2` | Generate G1 motion records and scale OFA grayscale inputs |
| `c6181dc33a8e94f1d17315714c2cf1abfcb0db2c` | Add the guarded OFA session path and parity gate |

The last commit is reviewable partial integration with a known failing development gate. It has no acceptance certificate. Builds ran during development; their identities are recorded in [artifacts.sha256](artifacts.sha256), [source.sha256](source.sha256), and the two CMake caches. The original audit binary was built from the audit changes before their source commit. The first-difference diagnostic binary includes the final test diagnostics at `c6181dc`. The session build embeds `source_commit=23e3ab9f61db17c89211f870c731d15dc9f621d2` and `source_dirty=true`, as recorded in [development/session-compile-flags.txt](development/session-compile-flags.txt); it was built before committing the integration changes.

Implemented source includes fixed flow configuration, shared grayscale and cell device functions, CPU `gmc_records`, scaled `flow_sequence`, CTL-selected gray/mask storage, MI leases, guarded input/output copies, OFA on M, cell masking and median, session identity changes, declared-size decode capacities, MotionContext export, kernel parity tests, all-sequence perturbation coverage, and updated stream-name rules. The session and stream rules have **not** passed their acceptance tests. The acceptance script update was not applied before the stop; `scripts/run_tracking_session_acceptance.sh` remains its original M3 version. Do not use that script as a v1-motion/2 acceptance runner.

TRACKING and all other existing documents are unchanged. No acceptance-result line was added. Existing evidence, tracker/Kalman/F07/F08 implementations, ResultPacket wire encoding, PCR library, `motion_sequence`, PCR tests, and both evaluation scripts are unchanged. No push or merge was performed.

## §20.6 audit

The standalone `ofa_audit` uses NVOF API 2.0, 1920×1080 grayscale, SLOW/grid4, a CUDA primary context, disabled temporal hints, no hints/cost/ROI, and one non-default nonblocking stream for both IO directions. The actual loaded library is `/usr/lib/wsl/lib/libnvidia-opticalflow.so.1`. Header hashes match the specification:

- `nvOpticalFlowCommon.h`: `a83f6045e5c470b35a6c50672f92e082b78d55752fe51cc20eb1a2738be05b9d`
- `nvOpticalFlowCuda.h`: `07defc79637fb9893f2a06204972eaa6bcf9e5bea400c88440439ebaaa39f115`

Input A is a deterministic integer-hash texture. Reference B is A translated (+8,+4), and replacement C is A translated (−16,+12), with edge clamping. Fully synchronized A→B and C→B outputs differ, ensuring the input-overwrite tests can distinguish the two cases. All comparisons use complete raw grid output bytes, excluding pitch padding. Preparation and final observation synchronize; the order-test interval has no host synchronization. Only the separately labeled controls synchronize the challenged dependency. There is no session workaround.

| Item | 0.2 ms spin, 20 trials | 20 ms spin, 20 trials | Synchronized control, 20 trials | Result |
| --- | --- | --- | --- | --- |
| A1 input-read order | 0 violations | 0 violations | 0 violations | PASS, 60/60 |
| A2 input write after Execute | 0 violations | 0 violations | 0 violations | PASS, 60/60 |
| A3 immediate output snapshot | 0 violations | 0 violations | 0 violations | PASS, 60/60 |
| A4 identical-pair determinism | 0 differences | 0 differences | 0 differences | PASS, 60/60 |
| A5 host blocking | 80 measured calls | 80 measured calls | 80 measured calls | 240 calls total |

A5 pooled host-call median **0.669904 ms**, p95 **0.902693 ms**. These are host API durations, not OFA engine latency or session throughput. Per-group median/p95 and every trial are in [audit/audit.log](audit/audit.log) and [audit/trials.csv](audit/trials.csv). Audit exit 0, elapsed 7.30 s: [timing](audit/audit.time).

A4 also reran the untouched `flow_sequence` on all 654 MOT17-10 frames. The replay and experiment SCFL were byte-identical, SHA-256:

`128b0498a844b5f00ac16a62ac44baf2729668404342811648b877cbe8a61112`

Sequence replay exit 0, elapsed 6.58 s: [log](audit/flow-sequence.log), [timing](audit/flow-sequence.time). After adding scaled grayscale support, the s=1 replay again had this exact SHA-256: [log](audit/flow-scaled-s1.log), [timing](audit/flow-scaled-s1.time). This latter comparison belongs to the records/scaling change at `23e3ab9`; the subsequent shared-cell/session change was stopped at the parity failure below.

## Records

Both prescribed CPU tests passed, 2/2: [CPU tests](audit/cpu-records-tests.log). The new tool generated seven T01 and seven HR SCMO files from the experiment SCFL inputs. Each T01 SCDR is a byte-preserving copy of `m1-data`. HR inputs are M4 SCDR. All seven HR `track_offline` MOT outputs using the new records are byte-identical to the experiment's G1 outputs: [generation and comparisons](records/generation.log), [record hashes](records/records.sha256).

Payload storage:

- T01 SCDR/SCMO: `build/records/t01/`.
- HR SCDR/SCMO and comparison MOT: `build/records/hr/`.
- Requested sibling `m1-data-v2` is a directory symlink to `g1-ofa/build/records/t01`, retaining the requested records entrypoint while keeping payload bytes in this worktree's `build/`.
- Audit raw gray/grid data and SCFL replays: `build/ofa-audit/`.

No payload files were copied into evidence. [artifacts.sha256](artifacts.sha256) identifies the payloads and executables.

## First differing stage and stop

The new `flow_motion_cuda_tests` executable was run under the GPU lock and timeout. The complete s=1 case (1920×1080) passed gray/cells/mask/median/copy byte comparisons. In the s=2 case, the grayscale and guarded-copy checks completed, then the first raw-grid cell comparison failed. The original run and a diagnostic rerun both exited 1. The diagnostic change only prints the first mismatch; it does not change arithmetic, ordering, or comparison criteria.

First failure at original **3840×2160**, OFA **1920×1080**, **s=2**, cell **(0,0)**:

| Four raw S10.5 input vectors, row-major | Value |
| --- | --- |
| (0,0) | (−16576, −7657) |
| (1,0) | (11235, 21357) |
| (0,1) | (−8040, 4248) |
| (1,1) | (16275, −15732) |
| GPU cell | **(0, 0)** |
| CPU `flow::reduce_cell` | **(49.921875, −53.265625)** |
| CPU hexadecimal float | `(0x1.8f6p+5, -0x1.aa2p+5)` |

The first differing stage is therefore the new suite's raw-grid-to-cell CUDA comparison. The failure was not relaxed or repaired. Its deeper cause has not been established; this evidence does not attribute it to OFA hardware or assert failure of a complete 4K session. See [original failure](development/flow-kernel-parity.log), [first-difference diagnosis](development/flow-cell-diagnostic.log), and [diagnostic timing](development/flow-cell-diagnostic.time).

Earlier compiler failures are preserved: the first CUDA compile rejected `std::array::operator[]` in device code, and the next C++ compile required an explicit `cudaMalloc` pointer cast. Both were corrected before the first kernel comparison. An initial records build attempted the new target before reconfiguring CMake; reconfiguration resolved it. No failed correctness comparison preceded the recorded kernel parity failure.

## §20.8 status

| Step | Result |
| --- | --- |
| 1 Clean source and identity | Formal capture unexecuted. Prerequisite NVOF path/header hashes checked; development source/binary identities archived. |
| 2 Fresh CPU/CUDA full CTest and parity | Formal capture unexecuted. Development build succeeded; CPU records tests 2/2 passed; the new CUDA parity suite failed at s=2 cell (0,0). Full CTest, seven session packet comparisons, perturbation runs, and lifecycle remain unexecuted. |
| 3 Seven HR deployment paths | Session/detector comparisons unexecuted. Only the seven CPU G1 records/MOT comparisons passed. |
| 4 PersonPath22 regression and five 4K sequences | Unexecuted. |
| 5 Re-scoring | Unexecuted. |
| 6 Sanitizers and controls | Unexecuted. |
| 7 Stream placement and OFA/F03 overlap | Unexecuted; no nsys capture or overlap result. |
| 8 Throughput versus main `23ade53` | Unexecuted; detached `g1-ofa-base` worktree was not created. Serial latency also unmeasured. |
| 9 Evidence report | This partial failure report is archived; acceptance remains incomplete. |

No before/after throughput measurements exist for this change:

| Sequence | Before ms/frame | After ms/frame |
| --- | --- | --- |
| MOT17-02-SDP | unmeasured | unmeasured |
| MOT17-04-SDP | unmeasured | unmeasured |
| MOT17-05-SDP | unmeasured | unmeasured |
| MOT17-09-SDP | unmeasured | unmeasured |
| MOT17-10-SDP | unmeasured | unmeasured |
| MOT17-11-SDP | unmeasured | unmeasured |
| MOT17-13-SDP | unmeasured | unmeasured |
| Total | unmeasured | unmeasured |

The 3% threshold has not been evaluated. The five excluded PP22 4K sequences have no new session outputs or scores. The s=2 primitive failure above is the only new 4K-sized test result.

## Execution custody and remaining work

[commands.tsv](commands.tsv) identifies audit/replay/parity commands, exits, and available timing files. Build logs and caches are archived in `audit/` and `development/`. Build output stayed under `build/ofa-audit/`, `build/records/`, and `build/development/`; no retry build directories were created. [environment-gpu.txt](environment-gpu.txt) records the device and driver.

**Execution deviation:** `cmake --build build/development ...` automatically regenerated CMake after the records-path change and reran the TensorRT GPU probe without the outer `flock`/`timeout` wrapper. This is visible at the beginning of [development/build-retry2.log](development/build-retry2.log). It violated the requested GPU-command wrapping rule. The audit, SCFL extractions, kernel comparisons, diagnostic rebuild/rerun, and environment GPU query were wrapped. No other GPU command ran concurrently in this task. The deviation prevents claiming that every GPU invocation followed the required execution discipline.

Remaining work was deliberately left unexecuted after the parity stop: diagnose/resolve the first discrepancy under a subsequent instruction, finish the §20.8 acceptance runner and OFA stream/overlap checks, then perform clean-source full acceptance, all MOT17/PP22/4K comparisons and scoring, sanitizers, same-run baseline throughput, and final acceptance documentation. No host-sync workaround, comparison relaxation, alternate flow profile, or tuning was added.

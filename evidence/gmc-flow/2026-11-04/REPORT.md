# Fixed GMC flow experiment — 2026-11-04

**Decision: GO; select L2 under PLAN §6.** G1, L1 and L2 each pass all three registered conditions. L2 has the highest MOT17-10 HOTA (53.205), exceeding L1 by 1.606378 points; the <0.2 preference for G1 does not apply. MOT17-02 is post-M4 and has zero decision weight. L1/L2 nevertheless lose about 5 HOTA there, as reported below.

The unedited [pre-registered plan](../PLAN.md) is the sole method and decision specification. All parameters and all six arms followed PLAN. The only fallback was the registered G1 fallback within L1/L2; no alternative API, operating system or software flow was used. All requested work completed. Integration into TrackingSession/F05/F06/CUDA tracking remains outside this experiment.

**Source and execution custody.** Started on clean `exp/gmc-flow` at `0e46d9ce04de4a8702ab46db65f1f3062ea54745`. The plan identifies main `23ade53` as its baseline. Extraction code and tests: `3d489b7`; CPU hook/shared control/harness: `b3874ba`. This is the requested fixed development experiment, not a new clean-source product acceptance certificate. Tools were built before their logical source commits; hashes identify the actual executables. B0 and B2 use the original stock `track_offline` binary, preserved as `build/gmc-flow/track_offline-b0`.

All writes stayed in this worktree; all build, SCFL and MOT outputs stayed under `build/gmc-flow/`. Sibling M4/m1 inputs were read only. No tracker configuration, Kalman arithmetic, CUDA tracker, TrackingSession, F05/F06, existing evidence or `docs/` files changed. No push or merge was performed.

**Gates and prescribed CPU tests.**

| Gate/test | Result | Evidence |
| --- | --- | --- |
| B0 | PASS: 7/7 MOT files byte-identical; all val and combined in-domain metrics exactly reproduce PLAN §2 | [comparison](B0.log), [val](B0-val.log), [in-domain](B0-in-domain.log) |
| Cell reduction / SCFL I/O | PASS: signed S10.5 reduction, round trip and format rejection | [CTest](cpu-flow-tests.log) |
| F1 | PASS: Init=0 and Execute=0 at 1920×1080 and 640×480, SLOW/grid4 | [CTest](F1.log) |
| F2 | PASS: all 6 prescribed resolution/shift cases; maximum component error 0.03125 px | [CTest](F2.log) |
| Mask / ring / band / median / fallback | PASS, including half-open boundaries, class/score selection, ±half quantization ties and N_min=8 | [CTest](cpu-control-tests.log) |
| B1 | PASS: 7/7 G0 hook MOT files byte-identical to B0 | [comparison](B1.log) |
| B2 | PASS: 14/14 G1/G1q hook MOT files byte-identical to stock plus equivalent SCMO | [comparison](B2.log) |

B0 ran before optical-flow work. F1 and F2 ran before extraction. B1 and B2 both passed before any flow-arm scoring or GT residual computation. The first flow-tool compile had a C++ auto-declaration error; it was corrected before any GPU gate (both [initial](build-flow.log) and [successful](build-flow-retry.log) build logs are retained). No correctness gate failed.

F2 used a fixed integer-hash texture, bilinear resampling with edge clamping, and every cell, including borders. Raw NVOF vectors already have the required previous→current positive right/down sign; no sign inversion was needed.

| Resolution | Truth (dx,dy) px | Median over all cells (dx,dy) px | Cells |
| --- | --- | --- | --- |
| 1920×1080 | (+8,+4) | (8,4) | 32400 |
| 1920×1080 | (−5,+3) | (−5,3) | 32400 |
| 1920×1080 | (+2.5,−1.5) | (2.5,−1.515625) | 32400 |
| 640×480 | (+8,+4) | (8,4) | 4800 |
| 640×480 | (−5,+3) | (−5,3) | 4800 |
| 640×480 | (+2.5,−1.5) | (2.5,−1.53125) | 4800 |

**Environment and extraction.** WSL2, RTX 5070 Ti Laptop GPU, driver 617.42, GCC 16.2.1, CUDA/nvcc 13.4.92, CMake 4.4.5. API 2.0 was loaded with `dlopen("libnvidia-opticalflow.so.1")`; the actual resolved path printed by every NVOF test/extraction invocation was `/usr/lib/wsl/lib/libnvidia-opticalflow.so.1`. Headers match SHA-256 `a83f6045e5c470b35a6c50672f92e082b78d55752fe51cc20eb1a2738be05b9d` and `07defc79637fb9893f2a06204972eaa6bcf9e5bea400c88440439ebaaa39f115`; provenance is NVIDIA commit `edb50da3cf849840d680249aa6dbef248ebce2ca` and the original BSD notices are retained.

Configured `build/gmc-flow` with Release, `SACCADE_ENABLE_CUDA=ON`, `SACCADE_ENABLE_PREPROCESS=ON`, `CMAKE_CUDA_ARCHITECTURES=120`, then enabled `SACCADE_ENABLE_FLOW=ON`. CUDA flow kernels use `--ftz=false --prec-div=true --prec-sqrt=true --fmad=false`. TensorRT was not needed. [Environment](environment.txt), [initial configure](configure.log), [flow configure](configure-flow.log), and [harness configure](configure-harness.log) record the execution identity.

Each GPU execution used `flock ~/.cache/saccade-v1/gpu.lock timeout ...`, one process at a time. API initialization used the CUDA primary context and the tool’s own non-default stream for both NVOF IO streams; grid4, optical-flow mode, no external or temporal hints, cost or ROI. `NativePreprocessor` used nvJPEG HARDWARE and its original-resolution planar RGB8. Full-resolution grayscale was computed with the registered float formula, then four S10.5 vectors were reduced componentwise to each 8 px cell. Frame zero is uninitialized with zero cells. The tool waits for its RGB/flow consumers before the next decode.

Quality flow was extracted exactly once per sequence using SLOW. The three additional MOT17-10 SLOW/MEDIUM/FAST passes are the registered timing runs; their SCFL files were not used for tracking quality.

| Sequence | Image | Frames | 8 px cells | Quality perf |
| --- | --- | --- | --- | --- |
| MOT17-02-SDP | 1920×1080 | 600 | 240×135 | SLOW |
| MOT17-04-SDP | 1920×1080 | 1050 | 240×135 | SLOW |
| MOT17-05-SDP | 640×480 | 837 | 80×60 | SLOW |
| MOT17-09-SDP | 1920×1080 | 525 | 240×135 | SLOW |
| MOT17-10-SDP | 1920×1080 | 654 | 240×135 | SLOW |
| MOT17-11-SDP | 1920×1080 | 900 | 240×135 | SLOW |
| MOT17-13-SDP | 1920×1080 | 750 | 240×135 | SLOW |

All 5,316 frames (5,309 pairs) were extracted successfully. [Extraction command logs](extraction/) retain the actual library path, Init status and completion for each sequence and timing run.

**Tracking scores.** Six arms × val, combined in-domain, five individually scored in-domain sequences, and MOT17-02 post-M4: 48 `scripts/eval_mot17.sh` invocations. Metrics are TrackEval percentages, except integer IDSW. Combined in-domain is TrackEval’s combined result, not a mean of five HOTA values. All MOT files remain in the build tree. [Rounded score CSV](scores.csv), [full precision score CSV](scores-precise.csv), and [all TrackEval summaries, detailed CSVs, logs and commands](trackeval/) are retained.

| Arm | Set | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OFF | val | 37.308 | 48.394 | 29.132 | 45.077 | 55.308 | 253 |
| OFF | in-domain | 67.424 | 73.184 | 62.414 | 74.948 | 82.794 | 378 |
| OFF | MOT17-04 | 74.802 | 83.170 | 67.376 | 80.578 | 92.464 | 71 |
| OFF | MOT17-05 | 56.351 | 56.773 | 56.569 | 69.353 | 62.325 | 113 |
| OFF | MOT17-09 | 57.735 | 70.060 | 47.734 | 66.057 | 81.840 | 24 |
| OFF | MOT17-11 | 60.528 | 66.165 | 55.583 | 67.889 | 74.205 | 65 |
| OFF | MOT17-13 | 50.961 | 54.123 | 48.242 | 63.870 | 62.850 | 105 |
| OFF | MOT17-02-post-M4 | 40.625 | 40.853 | 40.723 | 49.290 | 46.752 | 103 |
| G0 | val | 49.676 | 54.056 | 45.894 | 62.962 | 65.044 | 66 |
| G0 | in-domain | 68.423 | 74.454 | 63.148 | 76.136 | 84.284 | 231 |
| G0 | MOT17-04 | 74.802 | 83.170 | 67.376 | 80.578 | 92.464 | 71 |
| G0 | MOT17-05 | 57.999 | 58.981 | 57.559 | 69.842 | 66.329 | 48 |
| G0 | MOT17-09 | 57.724 | 69.782 | 47.909 | 65.467 | 81.822 | 24 |
| G0 | MOT17-11 | 65.821 | 70.249 | 61.768 | 74.569 | 78.317 | 19 |
| G0 | MOT17-13 | 54.361 | 59.051 | 50.267 | 67.848 | 67.497 | 69 |
| G0 | MOT17-02-post-M4 | 40.625 | 40.853 | 40.723 | 49.290 | 46.752 | 103 |
| G1 | val | 50.869 | 54.407 | 47.799 | 64.975 | 66.890 | 56 |
| G1 | in-domain | 68.930 | 74.705 | 63.848 | 76.905 | 84.458 | 227 |
| G1 | MOT17-04 | 74.802 | 83.169 | 67.376 | 80.578 | 92.464 | 71 |
| G1 | MOT17-05 | 60.827 | 60.851 | 61.278 | 74.030 | 68.961 | 51 |
| G1 | MOT17-09 | 57.734 | 70.060 | 47.734 | 66.057 | 81.840 | 24 |
| G1 | MOT17-11 | 65.608 | 69.971 | 61.616 | 74.424 | 77.904 | 19 |
| G1 | MOT17-13 | 56.306 | 59.078 | 53.887 | 70.335 | 67.471 | 62 |
| G1 | MOT17-02-post-M4 | 40.565 | 40.896 | 40.566 | 49.154 | 46.784 | 102 |
| G1q | val | 50.524 | 54.353 | 47.191 | 63.044 | 66.423 | 60 |
| G1q | in-domain | 68.838 | 74.486 | 63.880 | 76.954 | 84.344 | 254 |
| G1q | MOT17-04 | 74.802 | 83.170 | 67.376 | 80.578 | 92.464 | 71 |
| G1q | MOT17-05 | 60.531 | 60.232 | 61.355 | 73.947 | 67.761 | 61 |
| G1q | MOT17-09 | 57.735 | 70.060 | 47.734 | 66.057 | 81.840 | 24 |
| G1q | MOT17-11 | 65.354 | 69.892 | 61.212 | 73.824 | 77.851 | 19 |
| G1q | MOT17-13 | 56.221 | 58.680 | 54.112 | 71.267 | 67.437 | 79 |
| G1q | MOT17-02-post-M4 | 40.625 | 40.853 | 40.723 | 49.290 | 46.752 | 103 |
| L1 | val | 51.599 | 54.821 | 48.803 | 65.315 | 67.116 | 61 |
| L1 | in-domain | 69.745 | 74.892 | 65.191 | 78.094 | 84.988 | 236 |
| L1 | MOT17-04 | 75.492 | 82.632 | 69.063 | 82.039 | 92.321 | 69 |
| L1 | MOT17-05 | 60.502 | 60.556 | 60.928 | 72.468 | 68.296 | 59 |
| L1 | MOT17-09 | 57.736 | 70.081 | 47.724 | 66.057 | 81.840 | 24 |
| L1 | MOT17-11 | 65.648 | 70.350 | 61.359 | 73.891 | 78.190 | 20 |
| L1 | MOT17-13 | 58.925 | 61.390 | 56.801 | 73.701 | 71.903 | 64 |
| L1 | MOT17-02-post-M4 | 35.333 | 40.181 | 31.359 | 43.310 | 45.094 | 123 |
| L2 | val | 53.205 | 55.172 | 51.527 | 68.024 | 67.599 | 60 |
| L2 | in-domain | 69.081 | 74.366 | 64.426 | 77.107 | 84.145 | 295 |
| L2 | MOT17-04 | 74.885 | 82.562 | 68.023 | 80.809 | 92.100 | 73 |
| L2 | MOT17-05 | 59.622 | 59.323 | 60.445 | 72.144 | 66.011 | 60 |
| L2 | MOT17-09 | 57.736 | 70.055 | 47.742 | 66.057 | 81.840 | 24 |
| L2 | MOT17-11 | 65.366 | 70.318 | 60.872 | 73.444 | 78.116 | 23 |
| L2 | MOT17-13 | 57.895 | 59.661 | 56.468 | 72.593 | 68.365 | 115 |
| L2 | MOT17-02-post-M4 | 35.640 | 40.283 | 31.831 | 44.215 | 45.477 | 129 |

**Mechanical §6 verdict.** Calculations use the unrounded COMBINED rows in TrackEval’s detailed CSVs, multiplied by 100. Candidate requirements are val ΔHOTA≥+0.5 and ΔAssA≥0; combined in-domain ΔHOTA≥−0.2; every in-domain sequence ΔHOTA≥−0.5. [Machine-readable verdicts](verdicts.csv).

| Arm | Val ΔHOTA | Val ΔAssA | In-domain ΔHOTA | Worst in-domain ΔHOTA | Conditions 1/2/3 | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| G1 | 1.192993 | 1.905127 | 0.507070 | MOT17-11 -0.212691 | PASS / PASS / PASS | GO |
| L1 | 1.922747 | 2.909570 | 1.322424 | MOT17-11 -0.172204 | PASS / PASS / PASS | GO |
| L2 | 3.529125 | 5.633635 | 0.658411 | MOT17-11 -0.455042 | PASS / PASS / PASS | GO — SELECTED |

| Arm | Registered role / verdict |
| --- | --- |
| OFF | Reference; not an adoption candidate |
| G0 | Baseline; not an adoption candidate |
| G1 | GO |
| G1q | Diagnostic only; excluded from adoption selection |
| L1 | GO |
| L2 | GO; selected by highest val HOTA |

L2’s closest condition is MOT17-11: ΔHOTA=−0.455042, above the −0.5 bound by 0.044958. This was accepted mechanically without tuning. Both H1 and local-control candidates pass the registered operational rule; this is not the special “only local arms pass” outcome.

**GT residuals (§5.3).** The harness’s `residual` subcommand is the sole direct GT reader. It kept consider=1, class=1, visibility≥0.5 at all three consecutive frames of an ID. MOT frame f maps to SCDR frame f−1. Both tracking and residuals call the same `FrameControl::evaluate`; masks use the preceding SCDR frame and controls are evaluated on the preceding box, before prediction.

For each eligible triple, `e1_k=(c_k−c_{k−1})−t_k(box_{k−1})`; `e2_k=e1_k−e1_{k−1}`. Values below are Euclidean norms divided by `h_{k−1}`, with linear empirical quantiles (position `(n−1)p`). E1 and E2 use the same triple sample. Fallback count/fraction counts the current `t_k` once per triple; `t_{k−1}` is also evaluated for E2 but is not counted a second time. [Residual command log](gt-residual.log) and [per-sequence CSVs](gt-residual/).

| Sequence | Arm | N triples | E2 median | E2 p90 | E1 median | E1 p90 | Fallbacks | Fallback % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02 post-M4 | OFF | 7026 | 0.001543 | 0.010830 | 0.004711 | 0.020690 | 0 | 0.000% |
| MOT17-02 post-M4 | G0 | 7026 | 0.001543 | 0.010830 | 0.004711 | 0.020690 | 0 | 0.000% |
| MOT17-02 post-M4 | G1 | 7026 | 0.001543 | 0.010780 | 0.004702 | 0.020677 | 0 | 0.000% |
| MOT17-02 post-M4 | G1q | 7026 | 0.001543 | 0.010830 | 0.004711 | 0.020690 | 0 | 0.000% |
| MOT17-02 post-M4 | L1 | 7026 | 0.001753 | 0.010962 | 0.004746 | 0.020650 | 2 | 0.028% |
| MOT17-02 post-M4 | L2 | 7026 | 0.001642 | 0.011252 | 0.004717 | 0.021017 | 450 | 6.405% |
| MOT17-04 | OFF | 28632 | 0.002809 | 0.006803 | 0.005618 | 0.012500 | 0 | 0.000% |
| MOT17-04 | G0 | 28632 | 0.002809 | 0.006803 | 0.005618 | 0.012500 | 0 | 0.000% |
| MOT17-04 | G1 | 28632 | 0.002809 | 0.006803 | 0.005618 | 0.012500 | 0 | 0.000% |
| MOT17-04 | G1q | 28632 | 0.002809 | 0.006803 | 0.005618 | 0.012500 | 0 | 0.000% |
| MOT17-04 | L1 | 28632 | 0.002800 | 0.006849 | 0.005530 | 0.012459 | 0 | 0.000% |
| MOT17-04 | L2 | 28632 | 0.002778 | 0.006901 | 0.005525 | 0.012431 | 2387 | 8.337% |
| MOT17-05 | OFF | 3271 | 0.008194 | 0.043209 | 0.018490 | 0.067351 | 0 | 0.000% |
| MOT17-05 | G0 | 3271 | 0.018634 | 0.059450 | 0.017350 | 0.055375 | 0 | 0.000% |
| MOT17-05 | G1 | 3271 | 0.016700 | 0.053450 | 0.018007 | 0.053292 | 0 | 0.000% |
| MOT17-05 | G1q | 3271 | 0.018015 | 0.058140 | 0.019111 | 0.055375 | 0 | 0.000% |
| MOT17-05 | L1 | 3271 | 0.017176 | 0.054417 | 0.017991 | 0.050340 | 4 | 0.122% |
| MOT17-05 | L2 | 3271 | 0.018285 | 0.057137 | 0.023143 | 0.054237 | 1033 | 31.581% |
| MOT17-09 | OFF | 3024 | 0.002155 | 0.009342 | 0.013663 | 0.029091 | 0 | 0.000% |
| MOT17-09 | G0 | 3024 | 0.002243 | 0.010508 | 0.013656 | 0.029316 | 0 | 0.000% |
| MOT17-09 | G1 | 3024 | 0.002155 | 0.009342 | 0.013660 | 0.029070 | 0 | 0.000% |
| MOT17-09 | G1q | 3024 | 0.002155 | 0.009342 | 0.013663 | 0.029091 | 0 | 0.000% |
| MOT17-09 | L1 | 3024 | 0.002155 | 0.009365 | 0.013642 | 0.029076 | 0 | 0.000% |
| MOT17-09 | L2 | 3024 | 0.002155 | 0.009441 | 0.013642 | 0.029219 | 119 | 3.935% |
| MOT17-10 | OFF | 8765 | 0.007576 | 0.034493 | 0.027903 | 0.078459 | 0 | 0.000% |
| MOT17-10 | G0 | 8765 | 0.026430 | 0.086917 | 0.024691 | 0.063255 | 0 | 0.000% |
| MOT17-10 | G1 | 8765 | 0.020517 | 0.071757 | 0.023064 | 0.060002 | 0 | 0.000% |
| MOT17-10 | G1q | 8765 | 0.022727 | 0.081633 | 0.025266 | 0.065998 | 0 | 0.000% |
| MOT17-10 | L1 | 8765 | 0.022638 | 0.070919 | 0.020129 | 0.051769 | 2 | 0.023% |
| MOT17-10 | L2 | 8765 | 0.023369 | 0.073095 | 0.021217 | 0.053915 | 213 | 2.430% |
| MOT17-11 | OFF | 5910 | 0.001607 | 0.012050 | 0.011518 | 0.040444 | 0 | 0.000% |
| MOT17-11 | G0 | 5910 | 0.003583 | 0.021331 | 0.008960 | 0.024514 | 0 | 0.000% |
| MOT17-11 | G1 | 5910 | 0.003566 | 0.017173 | 0.008625 | 0.022758 | 0 | 0.000% |
| MOT17-11 | G1q | 5910 | 0.003459 | 0.019231 | 0.009442 | 0.025039 | 0 | 0.000% |
| MOT17-11 | L1 | 5910 | 0.003522 | 0.018082 | 0.007105 | 0.020318 | 0 | 0.000% |
| MOT17-11 | L2 | 5910 | 0.004254 | 0.019669 | 0.008163 | 0.022194 | 377 | 6.379% |
| MOT17-13 | OFF | 7475 | 0.013699 | 0.079290 | 0.062070 | 0.156898 | 0 | 0.000% |
| MOT17-13 | G0 | 7475 | 0.020012 | 0.086637 | 0.030906 | 0.069354 | 0 | 0.000% |
| MOT17-13 | G1 | 7475 | 0.020452 | 0.077265 | 0.029350 | 0.065720 | 0 | 0.000% |
| MOT17-13 | G1q | 7475 | 0.019919 | 0.087566 | 0.031427 | 0.071881 | 0 | 0.000% |
| MOT17-13 | L1 | 7475 | 0.023027 | 0.065221 | 0.019363 | 0.044669 | 10 | 0.134% |
| MOT17-13 | L2 | 7475 | 0.024474 | 0.067150 | 0.021028 | 0.052199 | 577 | 7.719% |

On MOT17-10, G1 reduces E2 median/p90 from G0’s 0.026430/0.086917 to 0.020517/0.071757. L1/L2 have larger E2 medians than G1 (0.022638/0.023369), while E1 median improves to 0.020129/0.021217 versus G1’s 0.023064. On MOT17-13, local E1 and E2 tails improve, although local E2 medians rise. MOT17-05 is mixed, especially L2 E1. Thus the residuals support a change in control error distribution, not a uniform residual improvement or the tracking-score ranking. OFF often has lower E2 despite much worse tracking; these GT finite differences are not the tracker’s actual learned Kalman state and are diagnostic only.

**Fallbacks in actual tracking.** Denominator is hook invocations on committed tracks while motion is valid. It differs from visible GT triples and from arm to arm because track histories differ. Every fallback is the registered <8 local-background-cell condition; G1 was never rejected on a paired frame. [Tracking fallback CSV](tracking-fallback.csv) includes counts, local-cell quantiles and box-height summaries.

| Sequence | L1 tracking fallback/calls | L1 tracking % | L2 tracking fallback/calls | L2 tracking % | L1 GT % | L2 GT % |
| --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-SDP | 2024/19516 | 10.371% | 5472/19781 | 27.663% | 0.028% | 6.405% |
| MOT17-04-SDP | 50/46618 | 0.107% | 4094/46693 | 8.768% | 0.000% | 8.337% |
| MOT17-05-SDP | 1753/9947 | 17.623% | 4559/10159 | 44.876% | 0.122% | 31.581% |
| MOT17-09-SDP | 151/5154 | 2.930% | 378/5154 | 7.334% | 0.000% | 3.935% |
| MOT17-10-SDP | 373/15247 | 2.446% | 1086/15301 | 7.098% | 0.023% | 2.430% |
| MOT17-11-SDP | 562/10672 | 5.266% | 1993/10709 | 18.611% | 0.000% | 6.379% |
| MOT17-13-SDP | 1393/16314 | 8.539% | 3039/17019 | 17.857% | 0.134% | 7.719% |

L2 uses G1 substantially more often than L1: the thin foot band supplies fewer eligible cells, and masking/image boundaries can exhaust it. MOT17-05 has the highest tracking fallback rates (L1 17.623%, L2 44.876%); MOT17-10 rates are 2.446%/7.098%. L1 fallback on visible GT boxes is near zero (0–0.134%) while actual tracked boxes fall back more often. These are distinct populations; fallback does not establish a per-track depth estimate.

**Motion statistics (§5.4).** Frames k≥1 only. Differences use Euclidean norm in original pixels; “>4” is strictly greater than 4. The static-camera mean is frame-weighted, not track-weighted. [Summary CSV](motion-statistics.csv) and [per-frame CSVs](motion/).

| Sequence | Pairs | &#124;G1−G0&#124; median px | p95 px | Frames >4 px | G0 rejected | G1 rejected | Static mean &#124;G1&#124; px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOT17-02-SDP | 599 | 0.000000 | 0.015625 | 0 | 0 | 0 | 0.003130 |
| MOT17-04-SDP | 1049 | 0.000000 | 0.015625 | 0 | 0 | 0 | 0.001986 |
| MOT17-05-SDP | 836 | 1.791065 | 8.578829 | 131 | 0 | 0 | — |
| MOT17-09-SDP | 524 | 0.000000 | 0.031250 | 6 | 0 | 0 | 0.003010 |
| MOT17-10-SDP | 653 | 1.594056 | 4.432056 | 61 | 0 | 0 | — |
| MOT17-11-SDP | 899 | 1.431032 | 3.930144 | 40 | 0 | 0 | — |
| MOT17-13-SDP | 749 | 1.401468 | 3.947965 | 36 | 0 | 0 | — |

G0 rejected no frames. Static G1 mean motion is 0.003130 px (02), 0.001986 px (04), and 0.003010 px (09). MOT17-09 has six >4 px G1−G0 differences despite near-zero G1, showing a small number of PCR excursions rather than a nonzero global-flow camera estimate.

**Mechanism reading.** G1q’s val HOTA is 50.524: +0.848 over G0 and −0.345 versus G1. Quantized flow retains about 71% of G1’s val gain (about 82% of its combined in-domain gain). Therefore fractional precision alone does not explain G1; the flow/median/mask estimator contributes most of the measured gain. The remaining G1−G1q difference supports a smaller additional subpixel contribution in this experiment.

Local control adds +0.730 val HOTA for L1 and +2.336 for L2 relative to G1. L1 has the strongest combined in-domain score (69.745); L2 wins the registered selection because highest val HOTA is the selection objective once all safeguards pass. Better HOTA is not a guarantee of fewer ID switches: L2 in-domain IDSW is 295 versus G0’s 231.

Static-camera behavior has a clear local-control cost on MOT17-02 post-M4: L1/L2 HOTA 35.333/35.640 versus G0 40.625. Their mean magnitudes over actual track controls are 0.224163/0.741653 px, versus G1’s 0.003221 px on that track-weighted basis. Local controls also inject more motion on 04/09, though those HOTA scores do not regress. This supports local motion contamination or noise as a mechanism; the experiment does not isolate which source cells cause it. The filtered visible-GT residuals show a smaller effect than complete tracking, so those medians do not explain the entire post-M4 loss. These results are reported without assigning MOT17-02 decision weight or changing the selected arm. [Static track-control summary](static-track-controls.csv).

**MOT17-10 timing (§3.2).** Full sequence, each perf once, 653 Execute pairs. CUDA events on the configured IO stream bracket `NvOFExecute`; grayscale and cell reduction have separate event pairs. Frame zero has no Execute and is excluded. There is no warmup trimming; maxima include the first pair. “Gray+cell” is the per-frame sum before taking quantiles. These are the registered event intervals, not end-to-end pipeline throughput, an overlap benchmark, or a measurement of only fixed-function OFA silicon. NVOF Execute is host-submitted; grayscale/cell kernels use CUDA.

| Perf | Execute median ms | Execute p95 ms | Execute max ms | Gray+cell median ms | Gray+cell p95 ms | Gray+cell max ms |
| --- | --- | --- | --- | --- | --- | --- |
| slow | 2.014624 | 2.277331 | 6.284608 | 0.047968 | 0.079955 | 0.110432 |
| medium | 1.116640 | 1.249958 | 5.892608 | 0.048768 | 0.060646 | 0.135840 |
| fast | 0.926816 | 1.058394 | 5.969408 | 0.048576 | 0.061786 | 0.101024 |

| Perf | Stage | Median ms | p95 ms | Max ms |
| --- | --- | --- | --- | --- |
| slow | grayscale_ms | 0.018368 | 0.018720 | 0.030336 |
| slow | cell_ms | 0.029504 | 0.061248 | 0.092000 |
| medium | grayscale_ms | 0.018368 | 0.018720 | 0.028832 |
| medium | cell_ms | 0.030400 | 0.042317 | 0.114592 |
| fast | grayscale_ms | 0.018304 | 0.018688 | 0.028416 |
| fast | cell_ms | 0.030208 | 0.042394 | 0.080256 |

[Raw timing CSVs](timing/) retain every frame. [Timing summary](timing-summary.csv) retains all stage quantiles.

**Replay and artifact custody.** [inputs.sha256](inputs.sha256) records all 7 M4 SCDR inputs, 7 original m1 SCMOs, 7 M4 reference MOTs, B0 MOTs, 7 quality SCFLs, 3 timing SCFLs, all 42 arm MOTs, 14 equivalent SCMOs, 14 stock B2 MOTs, diagnostics, both headers, executables, the unchanged plan and loaded NVOF library. SCFL and MOT payloads are only in `build/gmc-flow/`; none are archived in evidence. Input references in PLAN are resolved to the existing sibling `m4-formal` and `m1-data` directories under `/home/ray/developer/ai/saccade_v1-worktrees/`.

Archived text uses LF line endings with trailing whitespace removed; original build outputs are unchanged. Command logs preserve exact absolute inputs and output locations. Harness replay forms are `gmc_flow_offline track SCDR SCMO SCFL ARM MOT DIAG [EQUIV_SCMO]` and `gmc_flow_offline residual SCDR SCMO SCFL GT CSV`. The CPU hook has no configuration changes and is unset by default. The same control function serves both replay modes.

**Unexecuted:** none within PLAN. No product integration, CUDA-tracker counterpart, parameter sweep, extra variant, Windows/software/API fallback, pipeline-overlap experiment, push or merge was attempted.

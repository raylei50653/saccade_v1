# Fixed GMC flow experiment, round 2 — 2026-11-04

**Rule A: GO. Rule B: NO-GO for every local candidate.** All C0–C3 gates passed. The registered PP22 population is 9 moving and 60 static sequences; 5 sequences were excluded by PLAN-2 §2.4. These are operational decisions under the fixed rules, not general claims about unseen data.

The unedited [PLAN-2](../PLAN-2.md), its shared definitions in [PLAN](../PLAN.md), the [round-1 report](../2026-11-04/REPORT.md), and [REVIEW](../REVIEW.md) govern this experiment. No constants, arms, tracker/Kalman settings, CUDA tracker, TrackingSession, F05/F06, existing evidence, `docs/`, or `eval_mot17.sh` were changed. No parameter search, additional variant, product integration, push or merge was performed.

Execution began on clean `exp/gmc-flow` at `9b0df7214567c587b87f68adb3d9c7acaaa1e61b`, entirely in `/home/ray/developer/ai/saccade_v1-worktrees/gmc-flow`. Harness/test commit: `7bcc13fd8a0584ed2f3455f5e35ccb81db4ed777`; PP22 evaluation script: `caa87c2`; inputs, gates and camera strata committed **before scoring**: `0ef6b02`. Binaries were built during development; their actual SHA-256 identities are in [inputs.sha256](inputs.sha256). This is experiment evidence, not a separate clean-source product acceptance.

**Gates and execution order.**

| Gate | Result | Record |
| --- | --- | --- |
| C0 | PASS, 21/21: G0/G1/L2 × 7 MOT17 MOT files byte-identical to round 1 | [C0.log](C0.log) |
| C1 | PASS, 69/69 included PP22 G0 hook MOT files byte-identical to stock track_offline on the same SCDR+SCMO; 5 exclusions apply to every arm | [C1.log](C1.log), [commands](C1-commands/) |
| C2 | PASS, both CPU unit-test programs; row fitting, availability/search, mode convergence, alpha, LD start selection and retained round-1 controls/SCFL tests | [C2.log](C2.log) |
| C3 | PASS, GT-free table written and committed before any scoring; 9 moving / 60 static | [C3.log](C3.log), [table](stratification.csv) |

The build is `build/gmc-flow2`, Release/CUDA/preprocessing/flow, architecture 120, matching round 1. TensorRT/model-producer/postprocessing/detection-session targets were enabled there to build the existing `detect_sequence`. The TensorRT configure probes, as well as every GPU extraction invocation, ran under `flock ~/.cache/saccade-v1/gpu.lock` with `timeout`, one GPU command at a time. Included sequences ran exactly once per input tool; a first tool failure excluded that sequence without retry. Five failures is allowed; the >5 stop condition was never reached. [Build records](configure.log), [detector build](configure-detector.log), [extraction command/status table](extraction.csv), [extraction logs](extraction/).

**Inputs and exclusions.** The engine SHA-256 is `8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876`, verified before extraction. Detector invocation was the existing `detect_sequence ENGINE SEQ OUT.scdr --profile hr-1280x768`; PCR was the existing `motion_sequence` (`v1-motion/1`), and flow was `flow_sequence --perf slow`. The 1280×718 sequence succeeded. Every included sequence uses original frame rate and every frame for tracking. The 69 included sequences contain 48,452 frames. Their 69 SCFL files total 12,028,925,304 bytes (11.203 GiB); the five excluded 4K sequences have no flow files. Per-command free disk space is recorded in the extraction table.

| Excluded sequence | Failed tool / exit | Reason |
| --- | --- | --- |
| uid_vid_00179.mp4 | detect_sequence / 1 | detect_sequence: JPEG dimensions exceed configured capacity |
| uid_vid_00189.mp4 | detect_sequence / 1 | detect_sequence: JPEG dimensions exceed configured capacity |
| uid_vid_00218.mp4 | detect_sequence / 1 | detect_sequence: JPEG dimensions exceed configured capacity |
| uid_vid_00226.mp4 | detect_sequence / 1 | detect_sequence: JPEG dimensions exceed configured capacity |
| uid_vid_00230.mp4 | detect_sequence / 1 | detect_sequence: JPEG dimensions exceed configured capacity |

All five excluded images are 3840×2160; the prescribed detector invocation retains its default 1920×1080 source capacity. Motion/flow were not attempted after that sequence's detector failure. All arms omit the same five sequences. No source resizing, capacity adjustment or decoder fallback was added.

**GT-free camera-motion split.** For k≥1, m_k is the median magnitude of unmasked cells multiplied by 1920/W; sequence m is the median m_k. The mask is the preceding frame's kept class-0 detections with score≥0.45. m≥0.5 is moving. Neither the split tool nor the table generator read GT. The [74-row table](stratification.csv), [per-frame values and commands](stratification-frames/), and [sequence metadata inventory](pp22-sequences.csv) retain every included/excluded sequence.

| PP-moving sequence | m (1080p-width px) |
| --- | --- |
| uid_vid_00020.mp4 | 0.568758585 |
| uid_vid_00028.mp4 | 0.651488614 |
| uid_vid_00063.mp4 | 0.643476884 |
| uid_vid_00067.mp4 | 1.073131552 |
| uid_vid_00069.mp4 | 1.007103234 |
| uid_vid_00071.mp4 | 0.510150866 |
| uid_vid_00085.mp4 | 0.625780762 |
| uid_vid_00175.mp4 | 0.579073612 |
| uid_vid_00178.mp4 | 0.812500000 |

**Scoring.** `scripts/eval_pp22.sh` is the only PP22 GT entrypoint. It keeps prediction rows only on original frame numbers present anywhere in that sequence's GT file, preserving the original seqLength; all intervening frames have no GT or predictions. TrackEval HOTA/CLEAR/Identity settings and stdout line match unchanged `eval_mot17.sh`. No 5-fps tracking or frame-number compression was used. The archive contains the exact commands, keyframe row-count audits, logs, summaries and detailed CSVs for all 42 arm/set evaluations. Combined metrics are TrackEval combined results, not averages of per-sequence HOTA. MOT17 is secondary/post-diagnostic evidence.

Metrics below are percentages except integer IDSW. Decisions use full precision from detailed CSVs (HOTA/DetA/AssA AUC and Identity/CLEAR values), before display rounding. [Full precision scores](scores.csv), [TrackEval archive](trackeval/).

| Arm | Set | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OFF | PP-moving | 42.214 | 44.008 | 40.930 | 50.640 | 36.836 | 383 |
| OFF | PP-static | 41.576 | 44.126 | 39.769 | 47.627 | 39.406 | 4745 |
| OFF | PP-all | 41.618 | 44.118 | 39.845 | 47.821 | 39.247 | 5128 |
| OFF | MOT17-10 | 37.308 | 48.394 | 29.132 | 45.077 | 55.308 | 253 |
| OFF | in-domain | 67.424 | 73.184 | 62.414 | 74.948 | 82.794 | 378 |
| OFF | MOT17-02 | 40.625 | 40.853 | 40.723 | 49.290 | 46.752 | 103 |
| G0 | PP-moving | 41.990 | 44.017 | 40.519 | 50.307 | 36.644 | 390 |
| G0 | PP-static | 41.512 | 44.106 | 39.668 | 47.516 | 39.284 | 4778 |
| G0 | PP-all | 41.544 | 44.100 | 39.724 | 47.695 | 39.121 | 5168 |
| G0 | MOT17-10 | 49.676 | 54.056 | 45.894 | 62.962 | 65.044 | 66 |
| G0 | in-domain | 68.423 | 74.454 | 63.148 | 76.136 | 84.284 | 231 |
| G0 | MOT17-02 | 40.625 | 40.853 | 40.723 | 49.290 | 46.752 | 103 |
| G1 | PP-moving | 42.607 | 43.850 | 41.864 | 51.246 | 36.226 | 372 |
| G1 | PP-static | 41.581 | 44.121 | 39.780 | 47.721 | 39.330 | 4766 |
| G1 | PP-all | 41.649 | 44.103 | 39.914 | 47.947 | 39.138 | 5138 |
| G1 | MOT17-10 | 50.869 | 54.407 | 47.799 | 64.975 | 66.890 | 56 |
| G1 | in-domain | 68.930 | 74.705 | 63.848 | 76.905 | 84.458 | 227 |
| G1 | MOT17-02 | 40.565 | 40.896 | 40.566 | 49.154 | 46.784 | 102 |
| L2 | PP-moving | 41.987 | 43.494 | 40.974 | 50.179 | 35.751 | 384 |
| L2 | PP-static | 41.093 | 44.017 | 38.964 | 46.812 | 39.110 | 5098 |
| L2 | PP-all | 41.152 | 43.983 | 39.094 | 47.029 | 38.903 | 5482 |
| L2 | MOT17-10 | 53.205 | 55.172 | 51.527 | 68.024 | 67.599 | 60 |
| L2 | in-domain | 69.081 | 74.366 | 64.426 | 77.107 | 84.145 | 295 |
| L2 | MOT17-02 | 35.640 | 40.283 | 31.831 | 44.215 | 45.477 | 129 |
| LD | PP-moving | 41.860 | 43.054 | 41.166 | 50.159 | 34.265 | 397 |
| LD | PP-static | 41.126 | 44.027 | 39.017 | 46.810 | 39.144 | 5072 |
| LD | PP-all | 41.175 | 43.963 | 39.155 | 47.026 | 38.842 | 5469 |
| LD | MOT17-10 | 51.097 | 54.497 | 48.125 | 65.525 | 66.469 | 75 |
| LD | in-domain | 69.472 | 74.410 | 65.110 | 77.960 | 84.483 | 270 |
| LD | MOT17-02 | 36.077 | 40.382 | 32.528 | 45.029 | 45.735 | 133 |
| R | PP-moving | 42.371 | 43.789 | 41.475 | 50.674 | 35.968 | 374 |
| R | PP-static | 41.440 | 44.052 | 39.573 | 47.441 | 39.239 | 4903 |
| R | PP-all | 41.502 | 44.034 | 39.696 | 47.650 | 39.037 | 5277 |
| R | MOT17-10 | 51.673 | 54.593 | 49.135 | 64.828 | 65.885 | 60 |
| R | in-domain | 69.286 | 74.479 | 64.698 | 77.237 | 84.443 | 245 |
| R | MOT17-02 | 40.426 | 41.274 | 39.924 | 49.110 | 47.226 | 105 |
| RC | PP-moving | 42.011 | 43.457 | 41.086 | 50.421 | 35.367 | 381 |
| RC | PP-static | 41.402 | 44.088 | 39.471 | 47.321 | 39.301 | 4877 |
| RC | PP-all | 41.443 | 44.047 | 39.575 | 47.521 | 39.058 | 5258 |
| RC | MOT17-10 | 51.000 | 54.480 | 47.977 | 64.668 | 66.750 | 63 |
| RC | in-domain | 69.804 | 74.830 | 65.357 | 78.189 | 84.899 | 224 |
| RC | MOT17-02 | 40.802 | 40.931 | 40.998 | 49.816 | 47.296 | 109 |

**Mechanical rules A and B.** All deltas are percentage points. A compares G1 to G0; B compares each candidate to G1. LD is diagnostic only.

| Rule / arm | Moving ΔHOTA | Moving ΔAssA | Static ΔHOTA | MOT17-10 ΔHOTA | Conditions / verdict |
| --- | --- | --- | --- | --- | --- |
| A / G1 | 0.617266 | 1.344549 | 0.068864 | — | GO |
| B / L2 | -0.620181 | -0.889998 | -0.488410 | 2.336132 | FAIL / FAIL / PASS; NO-GO |
| B / R | -0.236401 | -0.388755 | -0.140965 | 0.803803 | FAIL / PASS / PASS; NO-GO |
| B / RC | -0.595877 | -0.777240 | -0.179043 | 0.130717 | FAIL / PASS / PASS; NO-GO |

A requires moving ΔHOTA≥0 and ΔAssA≥0, static ΔHOTA≥−0.2. B requires moving ΔHOTA≥+0.5 and ΔAssA≥0, static ΔHOTA≥−0.2, and MOT17-10 ΔHOTA≥0. Local selection uses highest moving HOTA among passing arms, preferring R, RC, L2 within a strict <0.2 gap. Selected local arm: **none**. [Machine-readable verdicts](verdicts.json). No arm was adjusted or augmented after scoring.

**PP-moving per sequence.** HOTA below; [CSV](pp-moving-per-sequence.csv) also retains all six metrics. Wins/losses use strict unrounded HOTA comparisons against G1; equality is a tie. These nine sequences were fixed before GT access.

| Sequence | OFF | G0 | G1 | L2 | LD | R | RC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uid_vid_00020.mp4 | 49.011 | 48.511 | 49.806 | 47.815 | 47.435 | 49.814 | 47.466 |
| uid_vid_00028.mp4 | 31.998 | 31.878 | 32.618 | 31.174 | 31.602 | 32.336 | 32.530 |
| uid_vid_00063.mp4 | 41.644 | 41.641 | 41.716 | 40.955 | 41.976 | 38.707 | 38.705 |
| uid_vid_00067.mp4 | 67.988 | 68.024 | 68.102 | 68.039 | 67.992 | 68.037 | 68.037 |
| uid_vid_00069.mp4 | 48.790 | 48.867 | 47.347 | 48.592 | 46.664 | 46.818 | 46.229 |
| uid_vid_00071.mp4 | 67.376 | 67.376 | 67.453 | 67.396 | 67.403 | 67.449 | 67.403 |
| uid_vid_00085.mp4 | 45.999 | 46.178 | 45.850 | 47.113 | 46.015 | 46.578 | 46.586 |
| uid_vid_00175.mp4 | 41.507 | 41.509 | 43.096 | 42.635 | 42.811 | 42.943 | 42.732 |
| uid_vid_00178.mp4 | 40.667 | 39.514 | 40.750 | 41.429 | 42.108 | 41.165 | 42.676 |

| Arm | Wins vs G1 | Losses vs G1 | Ties |
| --- | --- | --- | --- |
| OFF | 2 | 7 | 0 |
| G0 | 2 | 7 | 0 |
| G1 | 0 | 0 | 9 |
| L2 | 3 | 6 | 0 |
| LD | 3 | 6 | 0 |
| R | 3 | 6 | 0 |
| RC | 2 | 7 | 0 |

**R/RC/LD diagnostics.** Samples are actual hook calls on committed pre-prediction boxes while frame motion is valid. Denominators differ across arms because their tracking histories differ. R neighbor use is a nonzero selected-row offset (fallbacks have none); R fallback is the registered no-available-model-within-±4 condition. RC alpha includes zero for a <8-cell fallback to R; active-only alpha counts are retained separately in the CSV. Control differences compare RC against L2 and R evaluated on the **same RC box and frame**, so no cross-arm track matching is involved. Quantiles use linear interpolation at (n−1)p. [Complete per-sequence and pooled diagnostics](diagnostics.csv).

| Set / sequence | R calls | R→G1 % | R neighbor % |
| --- | --- | --- | --- |
| PP-moving | 79037 | 0.013% | 0.011% |
| PP-static | 1118450 | 0.044% | 0.030% |
| PP-all | 1197487 | 0.042% | 0.029% |
| MOT17-02-SDP | 19434 | 0.000% | 0.000% |
| MOT17-04-SDP | 46624 | 0.000% | 0.000% |
| MOT17-05-SDP | 10014 | 1.538% | 3.345% |
| MOT17-09-SDP | 5154 | 0.000% | 0.000% |
| MOT17-10-SDP | 15488 | 0.000% | 0.000% |
| MOT17-11-SDP | 10611 | 0.028% | 0.000% |
| MOT17-13-SDP | 16717 | 0.000% | 0.000% |

| Set / sequence | RC calls | α=0 | 0<α<1 | α=1 | RC→R (<8 cells) |
| --- | --- | --- | --- | --- | --- |
| PP-moving | 79449 | 20.165% | 3.679% | 76.156% | 19.283% |
| PP-static | 1118539 | 25.987% | 2.309% | 71.703% | 24.473% |
| PP-all | 1197988 | 25.601% | 2.400% | 71.999% | 24.129% |
| MOT17-02-SDP | 19366 | 28.075% | 1.611% | 70.314% | 27.347% |
| MOT17-04-SDP | 46619 | 8.745% | 0.112% | 91.143% | 8.743% |
| MOT17-09-SDP | 5154 | 7.354% | 0.155% | 92.491% | 7.315% |
| MOT17-10-SDP | 15254 | 12.502% | 25.154% | 62.344% | 7.408% |

| Set / sequence | Reference | Mean gap px | Median | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| PP-moving | L2 | 0.426163 | 0.086740 | 0.394765 | 6.874549 |
| PP-moving | R | 0.156647 | 0.074427 | 0.321221 | 1.523604 |
| PP-static | L2 | 0.653066 | 0.032671 | 0.188481 | 16.327645 |
| PP-static | R | 0.090731 | 0.021863 | 0.157983 | 1.187499 |
| PP-all | L2 | 0.638018 | 0.034724 | 0.208301 | 15.932309 |
| PP-all | R | 0.095102 | 0.023753 | 0.171715 | 1.215416 |
| MOT17-02-SDP | L2 | 0.341172 | 0.019623 | 0.078520 | 5.020593 |
| MOT17-02-SDP | R | 0.051441 | 0.010481 | 0.073956 | 0.712424 |
| MOT17-04-SDP | L2 | 0.048754 | 0.033757 | 0.100564 | 0.237432 |
| MOT17-04-SDP | R | 0.065102 | 0.037691 | 0.149841 | 0.431914 |
| MOT17-09-SDP | L2 | 0.046211 | 0.022925 | 0.055355 | 0.115374 |
| MOT17-09-SDP | R | 0.015022 | 0.008499 | 0.032148 | 0.101684 |
| MOT17-10-SDP | L2 | 1.139831 | 0.381088 | 2.899991 | 12.305438 |
| MOT17-10-SDP | R | 0.661397 | 0.440372 | 1.554092 | 3.557913 |

| Set / sequence | LD calls | LD→G1 % | Mean control magnitude px | p99 |
| --- | --- | --- | --- | --- |
| PP-moving | 79919 | 19.549% | 1.077758 | 7.891697 |
| PP-static | 1124883 | 24.621% | 0.778122 | 21.854985 |
| PP-all | 1204802 | 24.285% | 0.797998 | 20.210740 |
| MOT17-02-SDP | 19638 | 27.274% | 0.653141 | 35.758165 |
| MOT17-04-SDP | 46618 | 8.750% | 0.070619 | 0.444922 |
| MOT17-09-SDP | 5154 | 7.334% | 0.028547 | 0.154989 |
| MOT17-10-SDP | 15236 | 6.859% | 4.435756 | 23.909640 |

**Static-camera controls.** Means include the vector components and the mean magnitude; p99 is magnitude. OFF disables prediction control, so its value is identically zero and the hook has no calls. The zero OFF rows express that definition; other rows are track-call weighted. [CSV](static-controls.csv).

| Set / sequence | Arm | Calls | Mean dx px | Mean dy px | Mean magnitude px | p99 magnitude px |
| --- | --- | --- | --- | --- | --- | --- |
| PP-static | OFF | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| PP-static | G0 | 1115816 | 0.017938 | 0.031876 | 0.071554 | 4.000000 |
| PP-static | G1 | 1115590 | -0.014190 | 0.018583 | 0.093907 | 1.031723 |
| PP-static | L2 | 1126290 | 0.069642 | 0.010691 | 0.767419 | 22.246761 |
| PP-static | LD | 1124883 | 0.058618 | 0.009798 | 0.778122 | 21.854985 |
| PP-static | R | 1118450 | 0.028945 | 0.006788 | 0.513089 | 9.681838 |
| PP-static | RC | 1118539 | 0.027727 | 0.007383 | 0.552951 | 9.875219 |
| MOT17-02-SDP | OFF | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| MOT17-02-SDP | G0 | 19511 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| MOT17-02-SDP | G1 | 19455 | 0.002274 | 0.000053 | 0.003221 | 0.046875 |
| MOT17-02-SDP | L2 | 19781 | -0.657225 | -0.041009 | 0.741653 | 37.721427 |
| MOT17-02-SDP | LD | 19638 | -0.536541 | -0.026134 | 0.653141 | 35.758165 |
| MOT17-02-SDP | R | 19434 | -0.002315 | 0.005111 | 0.038101 | 0.171457 |
| MOT17-02-SDP | RC | 19366 | 0.003149 | 0.006256 | 0.074758 | 0.774733 |
| MOT17-04-SDP | OFF | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| MOT17-04-SDP | G0 | 46628 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| MOT17-04-SDP | G1 | 46628 | -0.000033 | 0.000232 | 0.001987 | 0.046875 |
| MOT17-04-SDP | L2 | 46693 | -0.000898 | 0.001037 | 0.035717 | 0.314932 |
| MOT17-04-SDP | LD | 46618 | -0.003693 | 0.000180 | 0.070619 | 0.444922 |
| MOT17-04-SDP | R | 46624 | -0.000657 | -0.001794 | 0.022331 | 0.086487 |
| MOT17-04-SDP | RC | 46619 | -0.004199 | 0.000314 | 0.073238 | 0.436767 |
| MOT17-09-SDP | OFF | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| MOT17-09-SDP | G0 | 5160 | -0.075194 | -0.006977 | 0.117911 | 0.000000 |
| MOT17-09-SDP | G1 | 5154 | 0.000540 | 0.001583 | 0.003034 | 0.046875 |
| MOT17-09-SDP | L2 | 5154 | 0.011156 | -0.000578 | 0.026451 | 0.118038 |
| MOT17-09-SDP | LD | 5154 | 0.013000 | 0.000126 | 0.028547 | 0.154989 |
| MOT17-09-SDP | R | 5154 | 0.011345 | -0.000510 | 0.028834 | 0.114459 |
| MOT17-09-SDP | RC | 5154 | 0.013353 | -0.000846 | 0.034560 | 0.164293 |

**Mechanism reading.**

G1 generalizes under rule A: moving HOTA/AssA improve by +0.617266/+1.344549 over G0, static HOTA by +0.068864. It wins 7/9 moving sequences against G0. This is compatible with a more useful global estimate on this registered population; the moving stratum contains only nine sequences, and five 4K sequences were excluded. HOTA/AssA gains do not imply a gain in every metric: moving MOTA falls from 36.643578 to 36.226321.

LD supports REVIEW §2.3's warning about using raw-flow concentration as a ground selector, but does not establish its cause on every sequence. Moving HOTA is 41.860128: −0.746804 vs G1 and −0.126623 vs L2, with only 3/9 wins against G1. On MOT17-10 LD is 51.096571 vs L2 53.205036. On static MOT17-02, LD improves over L2 by +0.436917 HOTA but still trails G1 by −4.488560; its control p99 remains 35.758165 px vs L2 37.721427. PP-static LD is only +0.033859 over L2 and −0.454551 vs G1, while its mean magnitude is 0.778122 vs L2 0.767419 px. Thus a slight static-score benefit does not make raw-density selection safe. These aggregate measurements are consistent with REVIEW's failure warning, rather than a new identification of which cells are occluders.

R replaces a foot-band statistic with the registered whole-three-row robust model evaluated at the foot point. It sharply reduces the previously diagnosed MOT17-02 excursion: mean/p99 magnitude 0.038101/0.171457 px vs L2 0.741653/37.721427; HOTA recovers to 40.426352. R retains +0.803803 HOTA over G1 on MOT17-10, but gives back part of L2's +2.336132. On PP-moving R improves over L2 by +0.383780, yet remains −0.236401 HOTA/−0.388755 AssA below G1. Its PP-static loss (−0.140965) is within the registered safeguard. This supports mitigation of the known local contamination event, without confirming a general local-control gain on PP22. R→G1 fallback is only 0.042172% of PP-all calls and neighbor-row use 0.028727%; the result predominantly reflects the row model itself, not frequent fallback.

RC adds a residual mode correction to R. On PP-moving its HOTA is −0.359476 below R and −0.595877 below G1; PP-static is −0.179043 vs G1, and MOT17-10 only +0.130717 vs G1. The residual correction therefore does not establish Q3 here. RC is +0.236391 over G1 on MOT17-02 and +5.161869 over L2, but that sequence is post-diagnostic and does not override rule B. On identical RC boxes the moving median/p99 RC−L2 control gap is 0.086740/6.874549 px, and RC−R is 0.074427/1.523604 px. In PP-static the corresponding p99 gaps are 16.327645 and 1.187499 px. Small median corrections can coexist with much larger tails and different track histories.

Across all 1,197,988 PP22 RC calls, α is 0 / (0,1) / 1 in 25.601091% / 2.400274% / 71.998634% of calls. Moving: 20.165137% / 3.679090% / 76.155773%; static: 25.987203% / 2.309441% / 71.703356%. The registered <8-cell RC→R condition accounts for 24.128789% of all calls. Among calls with at least eight eligible residual cells, α=1 in 94.895855% (moving 94.348891%, static 94.937376%); α is in the ramp in only 3.163617%. Thus this fixed rule is usually saturated, or falls back, rather than frequently applying a fractional blend. This is a diagnosis, not a request to retune thresholds.

PP-static means sequence median camera score <0.5; it does not assert that every frame or every background depth is stationary. Its nonzero control tails alone cannot be labeled noise. The report uses no additional GT residual or source-cell labeling experiment. The measured score/control changes do not isolate which scene geometries or occluders cause the remaining errors. Rules A and B are applied unchanged: G1 GO; L2/R/RC NO-GO; no local arm selected.

**Replay and custody.** SCDR, SCMO, SCFL and original/keyframe-filtered MOT payloads remain only under `build/`; evidence stores SHA-256, CSV diagnostics, text summaries/logs and replay scripts. [inputs.sha256](inputs.sha256) records all 69 PP22 input triplets, the seven reused round-1 MOT17 SLOW SCFL files, engine and executables. [outputs.sha256](outputs.sha256) identifies all 532 arm MOT outputs (483 PP22 + 49 MOT17), stock C1 MOT outputs and raw diagnostics. PP22 inputs are under `build/gmc-flow2/pp22/`; arm MOTs are in `pp22/arms/` and `build/gmc-flow2/mot17/`; raw diagnostics and TrackEval working files are also there. Round-1 MOT17 SCDR/SCMO inputs in sibling worktrees were read only. Execution scripts in this evidence folder document the commands; they use paths in the requested worktree. Archived text uses LF with trailing whitespace removed; original build outputs are unchanged.

**Unexecuted:** none within the registered plan for the 69 included PP22 sequences and seven MOT17 sequences. The five exclusions have no downstream tracking/scoring, as required. No GT residual experiment, extra validation suite, parameter tuning, product integration, push or merge was performed.

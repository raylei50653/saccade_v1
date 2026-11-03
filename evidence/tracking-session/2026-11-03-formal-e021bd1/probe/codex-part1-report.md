## 1. Summary

Implemented the three requested changes. Normal tests and initcheck/memcheck pass. **Acceptance remains incomplete: all three racecheck runs exit 1 on a packet mismatch, despite reporting zero hazards.**

## 2. Files changed

- [src/tracking_session.cpp](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/src/tracking_session.cpp): ordered config upload on `consumer`; conditional initialization spin.
- [cuda/tracking_session.cu](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/cuda/tracking_session.cu), [tracking_session.hpp](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/include/saccade/tracking_session.hpp): one fixed 20,000,000-cycle helper using `perturb`.
- [tracking_cuda.hpp](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/include/saccade/tracking_cuda.hpp): explicit `reserved` member and layout/408-byte assertions.
- [motion_cuda_tests.cpp](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/tests/motion_cuda_tests.cpp): optional `--synthetic`; invalid arguments produce usage errors.

## 3. Commands run and results

Configured GPU build with all supplied flags and paths; built both configurations with `--parallel 8`. All GPU commands used the lock.

Regression command: `tracking_session_tests ENGINE SEQUENCES RECORDS MOT17-10-SDP --limit 30`.

Before fix, with spin added: **exit 1**:

```text
tracking_session_tests: MOT17-10-SDP: perturbed pipelined first differing frame=0 byte=52
```

After fix: **exit 0**:

```text
MOT17-10-SDP frames=30 CPU/serial/pipelined packet bytes identical
tracking_session_tests: 412 checks passed
```

Other checks:

- GPU `ctest -R '^(tracking_|motion_)' --output-on-failure`: **14/14**, 39,018,353 checks. All seven sequences, totaling 5,316 frames, retained identical CPU/serial/pipelined packet bytes. [Detailed counts](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/build-m3-fix/Testing/Temporary/LastTest.log).
- Default CPU Release build and full CTest: **16/16**; 1,754,151 explicitly reported checks plus script tests.
- Motion invalid/extra arguments: expected usage errors, exit 1.
- `git diff --check`: passed.

Sanitizers used the prescribed flags, timeouts, frame limits and racecheck filter. Cleanup ran after every invocation; no target processes remain.

| Run | Checks / exit | Summary |
|---|---|---|
| tracking initcheck | 7,404,862 / 0 | `ERROR SUMMARY: 0 errors` |
| motion synthetic initcheck | 523 / 0 | `ERROR SUMMARY: 0 errors` |
| session memcheck | 412 / 0 | `ERROR SUMMARY: 0 errors` |
| session racecheck 1 | Not printed / 1 | `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |
| session racecheck 2 | Not printed / 1 | `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |
| session racecheck 3 | Not printed / 1 | `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)` |

## 4. Ambiguities and the choice made

Existing birth and host-conversion `{}` initialization already zeroes `reserved`, so conversions required no edits. Confirmed `cfg.tracker` remains alive through the constructor’s final `cudaDeviceSynchronize`.

## 5. Out-of-scope findings

All three racecheck runs reported:

```text
tracking_session_tests: MOT17-10-SDP: perturbed pipelined first differing frame=2 byte=104
```

Byte 104 belongs to the first track’s `xyxy[0]`. The source of this mismatch remains unresolved; no additional fix was applied. Logs: [run 1](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/build-m3-fix/racecheck-session-1.log), [run 2](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/build-m3-fix/racecheck-session-2.log), [run 3](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/build-m3-fix/racecheck-session-3.log).

## 6. Not done / known limitations

The requirement for three passing racecheck application runs is unmet. No commits, branch changes, rebases or pushes were performed.


**1. Summary**

Fixed RGB-buffer reuse by recording a timing-disabled event immediately after F05 and host-waiting before the next decode. Added the requested fixed spin; constructor spin remains 20,000,000 cycles. Part-1 changes are preserved.

**2. Files changed**

Only these three files were edited for part 2:

- [src/tracking_session.cpp](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/src/tracking_session.cpp:312)
- [cuda/tracking_session.cu](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/cuda/tracking_session.cu:140)
- [include/saccade/tracking_session.hpp](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/include/saccade/tracking_session.hpp:52)

**3. Commands and results**

Both builds succeeded using the existing GPU configuration and default Release CPU configuration. All GPU commands used the required lock. [Exact commands](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/build-m3-fix/part2/commands.json) and [part-2-only diff](/home/ray/developer/ai/saccade_v1-worktrees/m3-fix/build-m3-fix/part2/part2-only.diff) are saved.

The demonstration ran `tracking_session_tests ENGINE SEQUENCES RECORDS MOT17-10-SDP --limit 30` without a sanitizer.

Spin added, fix absent — **exit 1**:

```text
tracking_session_tests: MOT17-10-SDP: perturbed pipelined first differing frame=2 byte=104
```

With fix — **exit 0**:

```text
MOT17-10-SDP frames=30 CPU/serial/pipelined packet bytes identical
tracking_session_tests: 412 checks passed
```

`ctest --test-dir build-m3-fix -R '^(tracking_|motion_)' --output-on-failure`: **14/14 passed**, 233.55 seconds.

| Test | Checks | Seconds |
|---|---:|---:|
| motion_reference | 462 | 0.05 |
| motion_cuda | 537 | 0.98 |
| tracking_reference | 1,750,999 | 0.24 |
| tracking_cuda | 7,404,862 | 2.08 |
| tracking_cuda_mot17 | 29,814,341 | 4.01 |
| tracking_packet | 804 | 0.46 |
| tracking_session_mot17_02 | 4,212 | 23.19 |
| tracking_session_mot17_04 | 7,362 | 33.33 |
| tracking_session_mot17_05 | 10,903 | 53.24 |
| tracking_session_mot17_09 | 3,687 | 16.48 |
| tracking_session_mot17_10 | 8,524 | 43.98 |
| tracking_session_mot17_11 | 6,312 | 26.81 |
| tracking_session_mot17_13 | 5,262 | 23.60 |
| tracking_session_lifecycle | 86 | 5.08 |

Compared with the saved part-1 log: MOT17-05 increased **6.79 s (14.6%)**, MOT17-10 **4.89 s (12.5%)**, and lifecycle **0.40 s (8.5%)**. Total increased **19.02 s**.

`ctest --test-dir build-m3-fix-cpu --output-on-failure`: **16/16 passed**, 51.55 seconds.

All sanitizers used `timeout -k 10 900`, `--error-exitcode 99`, and the requested racecheck filter. **All five exited 0**:

```text
racecheck-1: ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
racecheck-2: ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
racecheck-3: ========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
memcheck-30: ========= ERROR SUMMARY: 0 errors
memcheck-lifecycle: ========= ERROR SUMMARY: 0 errors
```

Racecheck: 152 checks each; 76.13 / 74.12 / 73.26 seconds. Memcheck: 412 checks in 163.43 seconds; lifecycle 86 checks in 45.83 seconds. Required `pkill` cleanup ran after each; no matching processes remained.

**4. Ambiguities and choices**

Reused the existing perturb kernel and widened its cycle argument to match the requested helper signature. Timing comparisons use the existing part-1 log.

**5. Out-of-scope findings**

None found.

**6. Not done / known limitations**

No requested verification omitted. Racecheck coverage uses the specified kernel filter. Changes remain uncommitted; no branch, rebase, push, or writes to either excluded checkout were performed.


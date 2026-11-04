# llama.cpp source comparison for HR F03

Read-only inspection on 2026-11-04. No llama.cpp source changes or GPU jobs were run for this investigation. Followed documentation-first inspection; Graphify was not needed.

## Verified source identity

- llama.cpp root: `/home/ray/developer/src/llama.cpp`.
- HEAD: `aa588a3463357ee990b314f41dda242869130704`, `common : drop the Windows argv reconstruction (#31417)`.
- `git status --short` produced no entries when checked.
- Read local `AGENTS.md`, relevant documentation, `tests/fusion/README.md`, and implementation source.
- F03 context: `docs/MODEL_TRT.md`, `docs/SCAN_CUDA.md`, `src/head_trt_network.cpp`, `cuda/selective_scan.cu`, and `evidence/kernel-analysis/2026-11-04/REPORT.md`.

All llama.cpp references below are local paths under that root at the verified HEAD. These are source observations, not llama.cpp performance measurements on this machine.

## Highest-priority transferable idea: keep several states in registers

`ggml/src/ggml-cuda/ssm-scan.cu:33-124` implements a Mamba1 scan with one thread per channel. `regA[N]` and recurrent `regs0[N]` stay in registers (`:68-69`), while adjacent threads access adjacent channel values. Softplus is evaluated once per channel (`:108-113`); the states update in an unrolled loop (`:116-123`). B and C are staged in shared memory (`:71-72`, `:101-106`), with block barriers each timestep. The dispatch selects N=16 and 128-thread blocks (`:293-301`); token counts 1-8 get fixed-length specializations, while longer sequences retain a dynamic loop (`:302-356`).

F03's existing scan instead uses 16 adjacent threads per channel for N=16 (`cuda/selective_scan.cu:76-104`). One warp processes two channels, repeatedly computing identical softplus and u decomposition within each channel. Its recurrent h is already in registers: there is no global state-history tensor to remove.

A useful intermediate design is **four threads per channel, each retaining four states**:

- Let local thread state be n=0..3 and register state j=0..3 represent state `n + 4*j`.
- One warp processes eight channels; its unique u/delta channel payload is 32 B per timestep instead of 8 B under the original mapping. This is source-level payload reasoning, not measured DRAM traffic.
- Compute delta and u decomposition four times per channel instead of sixteen, retaining more parallelism than one thread/channel.
- For the 4x256-channel HR scan, 64-thread blocks retain 64 blocks: `4*256*4/64`. Register use, instruction count, occupancy, and full-model time still require measurement.
- Reproduce original XOR8 and XOR4 butterfly stages locally as `(term[n]+term[n+8]) + (term[n+4]+term[n+12])`, then use four-thread XOR2 and XOR1. Explicit FP32 round-to-nearest additions preserve the writer's original addition tree.

Do not copy llama's sequential `sumf += state*C` reduction: that changes F03 rounding. Also retain F03's scaled softplus/drive, explicit FMA, range categories, overflow fallback, finite-input pass, final poison, status lifecycle, and stream contract.

## Caution from llama's own source

`ggml/src/ggml-cuda/ssm-scan.cu:205-207` explicitly says warp-shared dt/dA/x_dt are intentionally recomputed because sharing via shuffles/shared memory proved slower. This is a statement about that implementation's earlier measurements. It is a reason to measure leader broadcast as a separate candidate, not evidence that it will speed up F03.

## Other mechanisms: applicability and exclusions

| llama mechanism and source | F03 assessment |
| --- | --- |
| Causal SSM convolution keeps a rolling input window and weights in registers and fuses bias/SiLU, `ggml/src/ggml-cuda/ssm-conv.cu:29-56`; long-token path uses 32-token shared-memory tiles, `:60-146` | F03 head has BLC-to-NCHW shuffle, depthwise convolution, causal slice, shuffle back and SiLU in `src/head_trt_network.cpp:195-210`. Inspect actual TRT fused tactics and timing before proposing a BLC convolution plugin; TRT may already eliminate part of that work. |
| Mamba2 SSD matmul reformulation, `ggml/src/ggml-cuda/ssm-scan.cu:365-374`, dispatch `:846-856` | Not directly applicable. Dispatch requires scalar A per head, aligned head dimension, and Mamba2 semantics. F03 has 16 state-dependent A values and per-channel delta, with rank-1 C. The path also uses `__expf`, FP16 intermediates and a different summation order. It does not inherit F03 bitwise/range contracts. |
| Scan writes recurrent cache directly, `ggml/src/ggml-cuda/ggml-cuda.cu:3147-3216`, kernel routing `ssm-scan.cu:252-254` | F03 has T=1 and no cross-frame scan state, so the avoided snapshot copy does not exist here. |
| VMM pool with aligned reusable suballocation, `ggml/src/ggml-cuda/ggml-cuda.cu:577-639` | Low priority. F03 already reuses context/buffers; plugin workspace is zero. No per-frame allocation bottleneck was demonstrated. |
| CUDA graph update/reinstantiate and graph chunks, `ggml/src/ggml-cuda/ggml-cuda.cu:2942-2991` | F03 shape is fixed and CUDA Graph is already enabled. Dynamic update/chunking is not an identified new acceleration. |
| Fusion memory overlap guards, `ggml/src/ggml-cuda/ggml-cuda.cu:3442-3503` | Transfer the reasoning: check live inputs, intermediate consumers, graph outputs, and allocation aliasing before fusing. TRT activation storage can alias/reuse allocations; logical tensor sizes alone do not establish fusion safety or traffic savings. |
| Shape specialization | N=16 compile-time specialization is already present in the working tree. FP16 deployment already uses builder optimization level 3; these are not new opportunities. HR lengths 960/240/60 do not justify fully unrolling time without compiler-resource evidence. |

## Independent review of the isolated register4 prototype

Reviewed `build/f03-llama-20261104/source/candidate/cuda/selective_scan.cu`, particularly `scan_kernel_register4` at lines 135-196 and its N=16-only dispatch at lines 366-371. No files in that source tree were changed by this reviewer.

No source-level semantic defect was found:

1. State mapping `n+4*j` loads the same A/B/C and updates each h using the same explicit multiplications, exponentiation, scaled drive and FMA as the original kernel.
2. The local tree at line 172 matches original XOR8 then XOR4 for original state lanes 0..3; four-thread XOR2/XOR1 then gives the original writer's tree. This avoids changing associativity.
3. Overflow fallback at lines 180-185 scales each term before the same tree, then combines/scales skip in the original order. It does not scale an already-overflowed sum.
4. The priority ternary chooses the same first failing arithmetic category for each state. OR across four register states and then four threads equals OR across the original sixteen threads at that timestep. The entire channel exits at the same first failing timestep.
5. Lane validity is channel-uniform. Block size 64 is divisible by four and warp size 32; even a partial last channel block contains complete four-thread groups. Each shuffle mask names only its valid group.
6. Shared/per-channel A, rank-1/full C and optional D use the original index rules. N other than 16 retains the previous kernel. L=0 skips the core as before.
7. The existing finite-input pass and finalizer are unchanged. Nonfinite input bits still accumulate separately; arithmetic flags retain their category union. Any failure still causes the finalizer to overwrite the complete output with canonical quiet NaNs and publish done.

This review is source reasoning only. Actual compiled bitwise comparisons, edge fixtures, graph/stream checks, sanitizer results, resource usage, and complete-model measurements must determine acceptance.

## Interpreting expected benefit

The earlier kernel-analysis report cites an existing pipeline capture with F03 around 4.87 ms and the three scan cores around 1.054 ms. Those are prior capture numbers, not measurements made during this source investigation. On that rough decomposition, halving scan time would improve full F03 by about 10.8%, before interactions. Report full-model graph timing separately from primitive scan speed and tracking throughput.

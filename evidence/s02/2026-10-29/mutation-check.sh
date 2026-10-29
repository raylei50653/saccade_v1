#!/usr/bin/env bash
# CUDA sensitivity check: each mutation changes one clause of a private copy of
# cuda/selective_scan.cu; the unmodified GPU tests must then fail. The test,
# S01 reference and file I/O objects are built once from the same source.
# Usage from the repository root: bash mutation-check.sh SOURCE_ROOT OUTPUT_DIR
set -euo pipefail
src=${1:?source root}; out=${2:?output dir}
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
cuda_home=$(cd "$(dirname "$nvcc")/.." && pwd)
flags=(-std=c++20 -O3 -arch=sm_120 --ftz=false --prec-div=true --prec-sqrt=true --fmad=true -lineinfo)
mkdir -p "$out"
for unit in tests/selective_scan_cuda_tests src/selective_scan src/file_io; do
  c++ -std=c++20 -O2 -ffp-contract=off -I "$src/include" -I "$cuda_home/include" -c "$src/$unit.cpp" -o "$out/${unit//\//_}.o"
done
build() { # mutated .cu -> test binary
  "$nvcc" "${flags[@]}" -I "$src/include" -c "$1" -o "$2.o"
  c++ "$2.o" "$out"/tests_selective_scan_cuda_tests.o "$out"/src_selective_scan.o "$out"/src_file_io.o \
      -L "$cuda_home/lib64" -lcudart -o "$2"
}
build "$src/cuda/selective_scan.cu" "$out/baseline"
timeout 600 "$out/baseline" > "$out/baseline.log" 2>&1 || { echo "baseline: unmutated tests FAILED"; exit 1; }
echo "baseline: $(tail -1 "$out/baseline.log")"
# Kernels loaded at initialization: the held-stream test passes under lazy loading.
CUDA_MODULE_LOADING=LAZY timeout 120 "$out/baseline" --group streams > "$out/baseline-lazy-streams.log" 2>&1 ||
  { echo "baseline: lazy-loading stream group FAILED"; exit 1; }
echo "baseline lazy-loading streams: $(tail -1 "$out/baseline-lazy-streams.log")"
mutations=(
  'legacy_softplus|s/if (x >= -80.0f) return split(log1pf(expf(x)));/if (x >= -80.0f) return split(x < -20.0f ? expf(x) : logf(1.0f + expf(x)));/'
  'no_scaled_underflow_branch|s/if (x >= -80.0f) return split(log1pf/if (true) return split(log1pf/'
  'plain_fp32_drive|s/const float drive = ldexpf(.*$/const float drive = __fmul_rn(__fmul_rn(ldexpf(delta.m, delta.e), bv), u);/'
  'no_partial_sum_rescale|s/if (!finite_fp32(y)) {/if (false) {/'
  'recurrence_order|s/h = __fmaf_rn(decay, h, drive);/h = __fmaf_rn(decay, drive, h);/'
  'drop_d_skip|s/y = p.has_d ? __fadd_rn(sum, skip) : sum;/y = sum;/'
  'per_channel_a_reads_row_0|s/p.a\[(p.a_shared ? 0 : d \* N) + n\]/p.a[n]/'
  'full_c_reads_state_0|s/p.c_full ? token \* N + n : token/p.c_full ? token * N : token/'
  'butterfly_missing_level|s/for (unsigned offset = width \/ 2; offset > 0;/for (unsigned offset = width \/ 4; offset > 0;/'
  'status_not_cleared|0,/cudaMemsetAsync(status/s/CUDA(cudaMemsetAsync(status, 0, sizeof(ScanStatus), stream));/(void)0;/'
  'failure_not_poisoned|s/^    if (status->flags)$/    if (false)/'
  'done_not_written|s/if (first == 0) status->done = kScanDone;/(void)first;/'
  'finite_check_skips_d|s/{in.d, dn, kScanNonFiniteD}/{in.d, 0, kScanNonFiniteD}/'
  'scan_on_default_stream|s/scan_kernel<<<blocks, kBlock, 0, stream>>>/scan_kernel<<<blocks, kBlock>>>/'
  'merge_without_mean|s/float mean = __fmul_rn(sum, 0.25f);/float mean = sum;/'
  'direction_2_flips_h|s/flip_w = direction == 1 || direction == 2/flip_w = direction == 1 || direction == 3/'
  'kernels_not_loaded|s/CUDA(cudaFuncGetAttributes(&attributes, kernel));/(void)kernel;/@@lazy-streams'
)
status=0
for entry in "${mutations[@]}"; do
  name=${entry%%|*}; expr=${entry#*|}; mode=
  if [[ $expr == *@@* ]]; then mode=${expr##*@@}; expr=${expr%@@*}; fi
  sed "$expr" "$src/cuda/selective_scan.cu" > "$out/$name.cu"
  if cmp -s "$out/$name.cu" "$src/cuda/selective_scan.cu"; then echo "$name: mutation did not apply"; status=1; continue; fi
  diff "$src/cuda/selective_scan.cu" "$out/$name.cu" > "$out/$name.diff" || true
  build "$out/$name.cu" "$out/$name" > "$out/$name.build.log" 2>&1
  # lazy-streams: the first enqueue in a fresh process happens while the stream is
  # held; without explicit loading it waits for the held stream (timeout 124).
  runs=(full)
  [[ $mode == lazy-streams ]] && runs=(lazy-streams)
  # The default-stream mutant must also fail each stream test on its own.
  [[ $name == scan_on_default_stream ]] && runs+=(held-stream capture)
  for run in "${runs[@]}"; do
    case $run in
      full) cmd=(timeout 600 "$out/$name") ;;
      lazy-streams) cmd=(env CUDA_MODULE_LOADING=LAZY timeout 120 "$out/$name" --group streams) ;;
      held-stream) cmd=(timeout 120 "$out/$name" --group streams) ;;
      capture) cmd=(timeout 120 "$out/$name" --no-held-stream --group streams) ;;
    esac
    code=0
    "${cmd[@]}" > "$out/$name.$run.log" 2>&1 || code=$?
    if [[ $code == 0 ]]; then echo "$name [$run]: tests PASSED on mutated kernel (insensitive)"; status=1
    else echo "$name [$run]: detected (exit $code; $(tail -1 "$out/$name.$run.log"))"; fi
  done
done
exit "$status"

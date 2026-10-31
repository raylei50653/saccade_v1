#!/usr/bin/env bash
# S03 sensitivity checks. Mutate one clause of a private source copy, rebuild
# incrementally, and require the unchanged targeted test to reject the mutant.
# Usage: mutation-check.sh SOURCE_ROOT OUTPUT_DIR
set -euo pipefail
src=$(cd "${1:?source root}" && pwd)
out=${2:?output directory}
mkdir -p "$out"
out=$(cd "$out" && pwd)
[[ ! -e "$out/source" ]] || { echo 'mutation source already exists' >&2; exit 2; }
mkdir "$out/source"
tar -C "$src" --exclude='./.git' --exclude='./build-*' -cf - . | tar -C "$out/source" -xf -
private=$out/source
build=$out/build
trt=${SACCADE_TENSORRT_ROOT:-/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk}
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
cmake -S "$private" -B "$build" -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_TENSORRT=ON \
  "-DSACCADE_TENSORRT_ROOT=$trt" "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 \
  > "$out/configure.log" 2>&1
cmake --build "$build" --parallel 4 --target scan_trt_tests scan_trt_engine > "$out/baseline-build.log" 2>&1
timeout 900 "$build/scan_trt_tests" > "$out/baseline.log" 2>&1
timeout 300 "$build/scan_trt_engine" build "$out/baseline.engine" > "$out/baseline-engine-build.log" 2>&1
CUDA_MODULE_LOADING=LAZY timeout 120 "$build/scan_trt_engine" run "$out/baseline.engine" "$out/baseline-engine-report.txt" \
  > "$out/baseline-engine-run.log" 2>&1
# Format: name|relative path|sed expression|group or lazy-engine.
mutations=(
  'ignore_flags|cuda/scan_frame.cu|s/if (flags) ctl->frame_status/if (false) ctl->frame_status/|frame'
  'ignore_done|cuda/scan_frame.cu|s/if (done != kScanDone) ctl->frame_status/if (false) ctl->frame_status/|frame'
  'ignore_unknown_flags|cuda/scan_frame.cu|s/if (flags \& ~kKnownScanFlags) ctl->frame_status/if (false) ctl->frame_status/|frame'
  'status_not_retired|cuda/scan_frame.cu|s/s->flags = 0;/\/\/ flags not retired/; s/s->done = 0;/\/\/ done not retired/|frame'
  'missing_consumed_wait|src/scan_trt_frame.cpp|s/if (k >= S) wait(slot.consumed, d, k - S, "consumed", s);/\/\/ no consumed wait/|ownership'
  'commit_advances_on_failure|cuda/scan_frame.cu|s/ctl->fail_code = ctl->frame_status;/ctl->fail_code = ctl->frame_status; ctl->committed = ctl->current; ctl->committed_slot = ctl->working_slot;/|frame'
  'publish_ignores_gate|cuda/scan_frame.cu|s/if (!ctl->run_stage) return;/if (false) return;/|frame'
  'kernels_not_loaded|src/scan_trt_plugin.cpp|s/(void)selective_scan_cuda_load_kernels();/(void)0;/|lazy-engine'
  'context_binding_ignored|src/scan_trt_plugin.cpp|s/require_bound(stream);/(void)stream;/|contract'
  'serialization_drops_has_d|src/scan_trt_plugin.cpp|s/if (op_ == Op::scan) fields_.emplace_back/if (false) fields_.emplace_back/|serialization'
  'enqueue_synchronizes|src/scan_trt_plugin.cpp|s/require_bound(stream);/require_bound(stream); cudaStreamSynchronize(stream);/|streams'
  'enqueue_uses_default_stream|src/scan_trt_plugin.cpp|s/status, stream);/status, cudaStream_t{});/|streams'
)
status=0
printf 'mutation\tgroup\texit\tresult\n' > "$out/results.tsv"
for entry in "${mutations[@]}"; do
  IFS='|' read -r name file expr group <<< "$entry"
  cp "$src/$file" "$private/$file"
  sed "$expr" "$src/$file" > "$private/$file"
  if cmp -s "$src/$file" "$private/$file"; then echo "$name: mutation did not apply" >&2; status=1; continue; fi
  diff -u "$src/$file" "$private/$file" > "$out/$name.diff" || true
  if ! cmake --build "$build" --parallel 4 --target scan_trt_tests scan_trt_engine > "$out/$name.build.log" 2>&1; then
    echo "$name: mutant did not compile (not a detection)" >&2
    printf '%s\t%s\tbuild\tINVALID\n' "$name" "$group" >> "$out/results.tsv"
    cp "$src/$file" "$private/$file"
    status=1
    continue
  fi
  case $group in
    lazy-engine) cmd=(env CUDA_MODULE_LOADING=LAZY timeout 45 "$build/scan_trt_engine" run "$out/baseline.engine" "$out/$name.report.txt") ;;
    serialization) cmd=(timeout 300 bash -c '"$1" build "$2" && "$1" run "$2" "$3"' _ "$build/scan_trt_engine" "$out/$name.engine" "$out/$name.report.txt") ;;
    *) cmd=(timeout 120 "$build/scan_trt_tests" --group "$group") ;;
  esac
  printf '%q ' "${cmd[@]}" > "$out/$name.command"; printf '\n' >> "$out/$name.command"
  code=0
  "${cmd[@]}" > "$out/$name.log" 2>&1 || code=$?
  cp "$src/$file" "$private/$file"
  if [[ $code == 0 ]]; then
    echo "$name [$group]: INSENSITIVE"
    printf '%s\t%s\t%s\tINSENSITIVE\n' "$name" "$group" "$code" >> "$out/results.tsv"
    status=1
  else
    echo "$name [$group]: detected (exit $code)"
    printf '%s\t%s\t%s\tDETECTED\n' "$name" "$group" "$code" >> "$out/results.tsv"
  fi
done
exit "$status"

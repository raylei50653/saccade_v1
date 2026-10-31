#!/usr/bin/env bash
# Capture the T01 native head (docs/HEAD_TRT.md §10) only from a clean tested
# source commit; raw logs are separate from the later evidence/docs commit.
# Run from repo root:
#   bash evidence/t01/2026-10-30/native-head/capture.sh build-NEW-OUTPUT [SOURCE_REV]
# Needs the T01 artifacts under ~/.cache/saccade-v1/t01 (selected T1 checkpoint,
# head-features.f32 + .json, venv-export), the pinned TensorRT SDK, CUDA,
# compute-sanitizer and a GPU.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=${1:?usage: capture.sh build-NEW-OUTPUT [REV]}
rev=$(git rev-parse "${2:-HEAD}")
s04_tested=3857e243e8fc32663aa688f073343002ad12cb30   # S04 tested source
s04_delivery=0b63f6fe4fc7738d4fbcacd0cc43365b5a5031fb # S04 delivery (evidence fixed)
policy=4d38bcc51caf829afe737959e334d0c85b1eba0b       # S04 policy commit
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
trt=${SACCADE_TENSORRT_ROOT:-/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk}
t01=${T01_ROOT:-$HOME/.cache/saccade-v1/t01}
checkpoint=$t01/runs/t1/attempt-01/run/epoch_0001.ckpt
checkpoint_sha=070580f9f5755e57acd9ee7ed9b4307672f4d9996332b40e6b61ee7cc6e6962b
bundle_sha=4fccfda6342b88d43fc2bf676228810b974b9e9956f45fa53352124098f046a3
features=$t01/export/head-features.f32
features_sha=f8224da5118155a1bb7d9ecb19a508719cbca9a3342706568ff11e528aba52e1
low_memory_mib=4096
[[ $out == build-* && ! -e $out ]] || { echo 'output must be a new build-* path' >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$rev" && -z $(git status --porcelain) ]] || { echo "HEAD must be $rev with a clean tree" >&2; exit 2; }
mkdir -p "$out/capture" "$out/source" "$out/stage" "$out/dump"
out=$(cd "$out" && pwd)
cap=$out/capture src=$out/source
# Memory samples every 10 s; below 4 GiB available the whole capture is stopped.
(
  while true; do
    date -Is; free -m
    avail=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
    if (( avail < low_memory_mib )); then
      echo "LOW MEMORY: ${avail} MiB available < ${low_memory_mib} MiB; stopping the capture"
      kill -TERM 0
    fi
    sleep 10
  done
) > "$cap/memory.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
clean_source() {
  if [[ $(git rev-parse HEAD) != "$rev" || -n $(git status --porcelain) ]]; then
    { date -Is; git rev-parse HEAD; git status --porcelain; } >> "$cap/source-drift.log"
    echo 'source HEAD or working tree changed during capture' >&2
    return 2
  fi
}
run() {
  local name=$1 code=0
  shift
  clean_source || return $?
  printf '%q ' "$@" > "$cap/$name.command"; printf '\n' >> "$cap/$name.command"
  "$@" > "$cap/$name.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$code"
  clean_source || return $?
  return "$code"
}
hashes=(-DSACCADE_HEAD_CHECKPOINT="$checkpoint" -DSACCADE_HEAD_CHECKPOINT_SHA256="$checkpoint_sha" -DSACCADE_HEAD_BUNDLE_SHA256="$bundle_sha")

run environment bash -c 'date -Is; uname -srmo; git --version; cmake --version | head -1; c++ --version | head -1; "$1" --version | tail -2; "$2" --version | head -3; nvidia-smi --query-gpu=name,driver_version,uuid,pci.bus_id,compute_cap --format=csv; ldd --version | head -1; nproc' _ "$nvcc" "$sanitizer"
run source git rev-parse HEAD "$rev" "$s04_tested" "$s04_delivery" "$policy"
run working-tree git status --porcelain=v1
run source-archive git archive --format=tar --output="$out/source.tar" "$rev"
run archive-hash sha256sum "$out/source.tar"
run source-extract tar -xf "$out/source.tar" -C "$src"
run native-changes-since-s04 git diff --stat "$s04_tested" "$rev" -- src include cuda tools tests cmake CMakeLists.txt
# Everything the GPU runs (kernels, plugins, network builder, frame session) and
# the reference are the S04 tested source; only the head_trt tool's file input
# and output dump, and the configurable candidate hashes, are new.
run native-gpu-and-reference-unchanged-since-s04 git diff --exit-code "$s04_tested" "$rev" -- src include cuda tests cmake \
  tools/head_weights.cpp tools/model_manifest.cpp
run policy-unchanged-since-policy-commit git diff --exit-code "$policy" "$rev" -- docs/HEAD_POLICY.md src/head_reference.cpp include/saccade/head_reference.hpp
run s04-and-earlier-evidence-unchanged git diff --exit-code "$s04_delivery" "$rev" -- evidence/s04 evidence/s03 evidence/s02 evidence/s01 \
  evidence/cuda-synthetic evidence/p01-a evidence/scan-contract manifests/s04
run checkpoint-identity bash -c 'stat -L -c "%s bytes" "$1" && sha256sum "$1" && [[ $(sha256sum < "$1" | cut -d" " -f1) == "$2" ]]' _ "$checkpoint" "$checkpoint_sha"
run checkpoint-selection-record python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print(json.dumps(d, indent=1)); sys.exit(d["sha256"] != sys.argv[2])' \
  "$t01/runs/t1/selection.json" "$checkpoint_sha"
run features-identity bash -c 'stat -c "%s bytes" "$1" && sha256sum "$1" && [[ $(sha256sum < "$1" | cut -d" " -f1) == "$2" ]] && cat "$3"' _ "$features" "$features_sha" "${features%.f32}.json"

run cpu-configure cmake -S "$src" -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release "${hashes[@]}"
run cpu-build cmake --build "$out/cpu" --parallel 8
run cpu-ctest ctest --test-dir "$out/cpu" --output-on-failure
run weights-convert "$out/cpu/head_weights" convert "$checkpoint" "$checkpoint_sha" "$out/stage/t1-head.swb" "$cap/t1-head-mapping.tsv"
run weights-verify "$out/cpu/head_weights" verify "$out/stage/t1-head.swb" "$bundle_sha"
run reference-report "$out/cpu/head_reference_tests" "$out/stage/t1-head.swb" "$bundle_sha"

run trt-configure cmake -S "$src" -B "$out/trt" -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_TENSORRT=ON \
  "-DSACCADE_TENSORRT_ROOT=$trt" "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 "${hashes[@]}"
run trt-options bash -c 'grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|SACCADE_ENABLE_CUDA|SACCADE_ENABLE_TENSORRT|SACCADE_TENSORRT_ROOT|SACCADE_HEAD_CHECKPOINT|SACCADE_HEAD_CHECKPOINT_SHA256|SACCADE_HEAD_BUNDLE_SHA256):" "$1/CMakeCache.txt"' _ "$out/trt"
run trt-build cmake --build "$out/trt" --parallel 8 --verbose
run trt-ctest ctest --test-dir "$out/trt" --output-on-failure --verbose
engine=$out/trt/head-trt/head.engine
engine_sha=$(sha256sum "$engine" | cut -d' ' -f1)
run engine-identity bash -c 'stat -c "%s bytes" "$1" && sha256sum "$1" "$(cat "$2")"' _ "$engine" "$out/trt/head-trt/bundle.path"
run engine-build-report cp "$out/trt/head-trt/build-report.txt" "$cap/head-build-report.txt"
run engine-run-report cp "$out/trt/head-trt/run-report.txt" "$cap/head-run-report.txt"
run head-suite-report cp "$out/trt/head-trt/suite-report.tsv" "$cap/head-suite-report.tsv"
run engine-rebuild-determinism bash -c '"$1" build "$2" "$3" "$4" "$5" && sha256sum "$4" "$6" && cmp "$4" "$6"' _ "$out/trt/head_trt" "$out/stage/t1-head.swb" "$bundle_sha" \
  "$out/stage/rebuild.engine" "$out/stage/rebuild-report.txt" "$engine"
# Real T01 features: the same policy, determinism, failure sequence and graph replay as the fixed inputs.
run head-run-real-features env CUDA_MODULE_LOADING=LAZY "$out/trt/head_trt" run "$engine" "$engine_sha" "$out/stage/t1-head.swb" "$bundle_sha" \
  "$cap/head-run-real-features-report.txt" --features "$features" "$features_sha" --dump "$out/dump"
run head-run-unregistered "$out/trt/head_trt" run-unregistered "$engine"
# Diagnostic beside the acceptance: the training-side PyTorch T1 head on the same features.
run pytorch-vs-native bash -c '"$1" "$2/training/t01/t01_export.py" head-compare "$3" "$4" && cp "$5/export/head-compare.json" "$6/pytorch-vs-native.json"' \
  _ "$t01/venv-export/bin/python" "$src" "$out/dump" "$out/stage/t1-head.swb" "$t01" "$cap"

# Head sanitizers as in S04: memcheck and initcheck with explicit API reporting; the
# debug-engine intermediates group is excluded, initcheck skips the held-stream groups.
# Head racecheck/synccheck stay unexecuted (S04 report); the S03 scan suite's four
# tools are not rerun because its code is unchanged since S04 (checked above).
suite=("$out/trt/head_trt_tests" "$engine" "$engine_sha" "$out/stage/t1-head.swb" "$bundle_sha" --skip intermediates)
run sanitizer-head-memcheck "$sanitizer" --tool memcheck --report-api-errors explicit --error-exitcode 99 "${suite[@]}"
run sanitizer-head-initcheck "$sanitizer" --tool initcheck --report-api-errors explicit --error-exitcode 99 "${suite[@]}" --no-held-stream
printf '%s\n' 'head intermediates: excluded from sanitizers (rebuilds a debug engine); native run in trt-ctest head_trt' \
  'initcheck: --no-held-stream skips the callback-held stream groups only' \
  'head racecheck/synccheck: UNEXECUTED, as in S04 (TensorRT instrumentation cost; WSL crash during S04 development, cause undetermined)' \
  'S03 scan suite sanitizers: not rerun; scan kernels, plugins and frame session unchanged since S04 tested source (native-gpu-and-reference-unchanged-since-s04)' > "$cap/sanitizer-skips.txt"

# Manifest: declaration and file integrity in a staging bundle directory.
cp "$src/manifests/t01/yolo26m-head-t1.json" "$src/docs/HEAD_POLICY.md" "$out/stage/"
ln -s "$checkpoint" "$out/stage/epoch_0001.ckpt"
run manifest-validate "$out/cpu/model_manifest" validate "$out/stage/yolo26m-head-t1.json" "$cap/yolo26m-head-t1.canonical.json"
run manifest-verify "$out/cpu/model_manifest" verify "$out/stage/yolo26m-head-t1.json"
run binary-hashes sha256sum "$out/cpu/head_weights" "$out/cpu/head_reference_tests" "$out/cpu/model_manifest" "$out/trt/head_trt" "$out/trt/head_trt_tests" \
  "$out/trt/libsaccade_head_trt.a" "$out/trt/libsaccade_scan_trt.a" "$out/trt/libsaccade_scan_trt_frame.a" "$engine" "$out/stage/t1-head.swb" "$out"/dump/*.f32
run build-source-identity bash -c '
  set -euo pipefail
  rev=$1; shift
  for report in "$@"; do
    grep -Eq "(^|[[:space:]])build_commit=$rev$" "$report"
    grep -Eq "(^|[[:space:]])build_dirty=false$" "$report"
    if grep -q "build_dirty=true" "$report"; then exit 1; fi
    echo "$report: tested commit and build_dirty=false verified"
  done
' _ "$rev" "$cap/head-build-report.txt" "$cap/head-run-report.txt" "$cap/head-run-real-features-report.txt" "$cap/head-suite-report.tsv"
run working-tree-final git status --porcelain=v1

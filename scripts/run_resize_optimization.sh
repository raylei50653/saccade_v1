#!/usr/bin/env bash
# bash -c positional parameters are intentionally expanded by the child shell.
# shellcheck disable=SC2016
# Development-only F02 baseline/candidate equivalence and isolated ABBA replay.
# Usage: bash scripts/run_resize_optimization.sh build/NEW-NAME [BASELINE_REVISION]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
usage() {
  echo 'usage: run_resize_optimization.sh build/NEW-NAME [BASELINE_REVISION]' >&2
  exit 2
}
[[ $# -ge 1 && $# -le 2 ]] || usage
out=$1
baseline_ref=${2:-f4865cdb27f0e624b8cc26219e78f4aa938690df}
[[ $out =~ ^build/[A-Za-z0-9][A-Za-z0-9._-]*$ && ! -e $out && ! -L $out ]] || {
  echo 'output must be a new build/NAME directory' >&2
  exit 2
}
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
cxx=${CXX:-c++}
arch=${SACCADE_CUDA_ARCHITECTURES:-120}
verify_timeout=${SACCADE_RESIZE_VERIFY_TIMEOUT:-300}
benchmark_timeout=${SACCADE_RESIZE_BENCHMARK_TIMEOUT:-300}
lock_timeout=${SACCADE_RESIZE_LOCK_TIMEOUT:-600}
[[ $arch =~ ^[0-9]+$ && $verify_timeout =~ ^[1-9][0-9]*$ &&
   $benchmark_timeout =~ ^[1-9][0-9]*$ && $lock_timeout =~ ^[1-9][0-9]*$ ]] || usage
for dependency in git sha256sum flock timeout nvidia-smi awk sort "$nvcc" "$cxx"; do
  command -v "$dependency" >/dev/null || {
    echo "missing dependency: $dependency" >&2
    exit 2
  }
done
baseline_revision=$(git rev-parse --verify "$baseline_ref^{commit}")
candidate_revision=$(git rev-parse HEAD)
candidate_dirty=false
[[ -z $(git status --porcelain=v1 --untracked-files=all) ]] || candidate_dirty=true
mkdir -p build
mkdir "$out"
out=$(cd "$out" && pwd)
cap=$out/capture
mkdir "$cap" "$out/objects"
mkdir -p "$out/source/baseline/cuda" "$out/source/baseline/include/saccade"
mkdir -p "$out/source/candidate/cuda" "$out/source/candidate/include/saccade" "$out/source/candidate/tools"

run() {
  local name=$1 status=0
  shift
  printf '%q ' "$@" > "$cap/$name.command"
  printf '\n' >> "$cap/$name.command"
  printf '%s: started\n' "$name"
  "$@" > "$cap/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$status"
  return "$status"
}
finish() {
  local status=$?
  trap - EXIT
  printf 'capture_kind=development-optimization\nformal_acceptance=false\nexit=%s\nfinished=%s\n' \
    "$status" "$(date -Is)" > "$cap/capture-status.txt"
  # Preserve a complete manifest on both success and failure, excluding itself.
  (cd "$out"; find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum) > "$out/SHA256SUMS"
  if [[ $status != 0 ]]; then
    echo "capture failed; commands/logs/exit statuses retained at $out" >&2
  fi
  exit "$status"
}
trap finish EXIT
printf 'capture_kind=development-optimization\nformal_acceptance=false\nbaseline_revision=%s\nbaseline_source_dirty=false\ncandidate_revision=%s\ncandidate_source_dirty=%s\narchitecture=%s\nstarted=%s\n' \
  "$baseline_revision" "$candidate_revision" "$candidate_dirty" "$arch" "$(date -Is)" > "$cap/source-identity.txt"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status.txt"
git diff --binary HEAD > "$cap/source.diff"
git diff --binary --cached > "$cap/source-staged.diff"
cp scripts/run_resize_optimization.sh "$cap/runner.sh"
cp cmake/Preprocess.cmake "$cap/Preprocess.cmake"
sha256sum cuda/resize.cu include/saccade/resize_cuda.hpp tools/resize_benchmark.cpp \
  scripts/run_resize_optimization.sh cmake/Preprocess.cmake > "$cap/live-source-before.sha256"

run baseline-kernel-source git show "$baseline_revision:cuda/resize.cu"
cp "$cap/baseline-kernel-source.log" "$out/source/baseline/cuda/resize.cu"
run baseline-header-source git show "$baseline_revision:include/saccade/resize_cuda.hpp"
cp "$cap/baseline-header-source.log" "$out/source/baseline/include/saccade/resize_cuda.hpp"
cp cuda/resize.cu "$out/source/candidate/cuda/resize.cu"
cp include/saccade/resize_cuda.hpp "$out/source/candidate/include/saccade/resize_cuda.hpp"
cp tools/resize_benchmark.cpp "$out/source/candidate/tools/resize_benchmark.cpp"
# Compile only the frozen copies. Check the candidate copies against the live
# source once; unrelated documentation edits during replay do not affect them.
cmp cuda/resize.cu "$out/source/candidate/cuda/resize.cu"
cmp include/saccade/resize_cuda.hpp "$out/source/candidate/include/saccade/resize_cuda.hpp"
cmp tools/resize_benchmark.cpp "$out/source/candidate/tools/resize_benchmark.cpp"
(cd "$out/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$cap/frozen-source.sha256"

run environment bash -c 'set -e
  date -Is
  uname -srmo
  git --version
  "$1" --version
  "$2" --version
  nvidia-smi --query-gpu=name,driver_version,uuid,pci.bus_id,compute_cap --format=csv
  nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,pstate,temperature.gpu,power.draw --format=csv
' _ "$nvcc" "$cxx"

# Match the Release CMake F02 arithmetic flags exactly; no fast-math flag.
flags=(-O3 -DNDEBUG -std=c++20 "-arch=sm_$arch" -ccbin "$cxx"
  --ftz=false --prec-div=true --prec-sqrt=true --fmad=true -lineinfo)
renames=()
for symbol in resize_input_bytes resize_output_elements resize_workspace_elements \
  resize_rgb_chw_cuda resize_cuda_load_kernels resize_cuda_identity; do
  renames+=("-D$symbol=${symbol}_baseline")
done
run compile-baseline "$nvcc" "${flags[@]}" -I"$out/source/baseline/include" \
  "-DSACCADE_BUILD_COMMIT=\"$baseline_revision\"" '-DSACCADE_BUILD_DIRTY="false"' \
  "${renames[@]}" -c "$out/source/baseline/cuda/resize.cu" -o "$out/objects/baseline.o"
run compile-candidate "$nvcc" "${flags[@]}" -I"$out/source/candidate/include" \
  "-DSACCADE_BUILD_COMMIT=\"$candidate_revision\"" "-DSACCADE_BUILD_DIRTY=\"$candidate_dirty\"" \
  -c "$out/source/candidate/cuda/resize.cu" -o "$out/objects/candidate.o"
run compile-harness "$nvcc" -O3 -DNDEBUG -std=c++20 "-arch=sm_$arch" -ccbin "$cxx" \
  -I"$out/source/candidate/include" -c "$out/source/candidate/tools/resize_benchmark.cpp" \
  -o "$out/objects/harness.o"
run link "$nvcc" -O3 -std=c++20 "-arch=sm_$arch" -ccbin "$cxx" \
  "$out/objects/harness.o" "$out/objects/baseline.o" "$out/objects/candidate.o" -o "$out/resize_benchmark"
run binary-identity bash -c 'set -e
  sha256sum "$1"
  ldd "$1"
' _ "$out/resize_benchmark"

gpu_lock=$HOME/.cache/saccade-v1/gpu.lock
mkdir -p "$(dirname "$gpu_lock")"
printf 'gpu_lock=%s\nlock_timeout_seconds=%s\nverify_timeout_seconds=%s\nbenchmark_timeout_seconds=%s\n' \
  "$gpu_lock" "$lock_timeout" "$verify_timeout" "$benchmark_timeout" > "$cap/replay-bounds.txt"
run verify flock -w "$lock_timeout" "$gpu_lock" \
  timeout --signal=TERM --kill-after=15s "${verify_timeout}s" \
  "$out/resize_benchmark" "$out/verify" --verify-only
run benchmark flock -w "$lock_timeout" "$gpu_lock" \
  timeout --signal=TERM --kill-after=15s "${benchmark_timeout}s" \
  "$out/resize_benchmark" "$out/benchmark" --benchmark-only --rounds 4 --iterations 200 --warmup 40
run frozen-source-check bash -c 'set -e; cd "$1/source"; sha256sum -c "$1/capture/frozen-source.sha256"' _ "$out"
sha256sum cuda/resize.cu include/saccade/resize_cuda.hpp tools/resize_benchmark.cpp \
  scripts/run_resize_optimization.sh cmake/Preprocess.cmake > "$cap/live-source-after.sha256"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status-after.txt"
printf 'Development optimization replay complete: %s\n' "$out"
printf 'Frozen source snapshots were tested; this is not clean-source formal acceptance.\n'

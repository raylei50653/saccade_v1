#!/usr/bin/env bash
# Development-only dual-linked baseline/candidate exact replay and ABBA timing.
# SACCADE_SCAN_BUILD_ONLY=1 builds the frozen harness without running GPU work.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
[[ $# -ge 1 && $# -le 2 ]] || { echo 'usage: run_scan_optimization.sh build/NEW-NAME [BASELINE_REVISION]' >&2; exit 2; }
out=$1
baseline_ref=${2:-HEAD}
[[ $out =~ ^build/[A-Za-z0-9][A-Za-z0-9._-]*$ && ! -e $out && ! -L $out ]] || {
  echo 'output must be a new build/NAME directory' >&2; exit 2;
}
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
cxx=${CXX:-c++}
arch=${SACCADE_CUDA_ARCHITECTURES:-120}
build_only=${SACCADE_SCAN_BUILD_ONLY:-0}
lock_timeout=${SACCADE_SCAN_LOCK_TIMEOUT:-600}
verify_timeout=${SACCADE_SCAN_VERIFY_TIMEOUT:-300}
benchmark_timeout=${SACCADE_SCAN_BENCHMARK_TIMEOUT:-300}
[[ $arch =~ ^[0-9]+$ && $build_only =~ ^[01]$ && $lock_timeout =~ ^[1-9][0-9]*$ &&
   $verify_timeout =~ ^[1-9][0-9]*$ && $benchmark_timeout =~ ^[1-9][0-9]*$ ]] || exit 2
for dependency in git sha256sum flock timeout nvidia-smi sort "$nvcc" "$cxx"; do
  command -v "$dependency" >/dev/null || { echo "missing dependency: $dependency" >&2; exit 2; }
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
  printf 'capture_kind=development-optimization\nformal_acceptance=false\nbuild_only=%s\nexit=%s\nfinished=%s\n' \
    "$build_only" "$status" "$(date -Is)" > "$cap/capture-status.txt"
  (cd "$out"; find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum) > "$out/SHA256SUMS"
  exit "$status"
}
trap finish EXIT
printf 'capture_kind=development-optimization\nformal_acceptance=false\nbaseline_revision=%s\ncandidate_revision=%s\ncandidate_source_dirty=%s\narchitecture=%s\nstarted=%s\n' \
  "$baseline_revision" "$candidate_revision" "$candidate_dirty" "$arch" "$(date -Is)" > "$cap/source-identity.txt"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status.txt"
git diff --binary HEAD > "$cap/source.diff"
git diff --binary --cached > "$cap/source-staged.diff"
cp scripts/run_scan_optimization.sh "$cap/runner.sh"
for path in cuda/selective_scan.cu include/saccade/selective_scan.hpp include/saccade/selective_scan_cuda.hpp; do
  git show "$baseline_revision:$path" > "$out/source/baseline/$path"
  cp "$path" "$out/source/candidate/$path"
done
cp tools/scan_benchmark.cpp "$out/source/candidate/tools/scan_benchmark.cpp"
(cd "$out/source"; find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$cap/frozen-source.sha256"
run environment bash -c 'set -e
  date -Is
  uname -srmo
  "$1" --version
  "$2" --version
  nvidia-smi --query-gpu=name,driver_version,uuid,pci.bus_id,compute_cap --format=csv
  nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,pstate,temperature.gpu,power.draw --format=csv
' _ "$nvcc" "$cxx"
flags=(-O3 -DNDEBUG -std=c++20 "-arch=sm_$arch" -ccbin "$cxx"
  --ftz=false --prec-div=true --prec-sqrt=true --fmad=true -lineinfo)
renames=()
for symbol in selective_scan_cuda selective_scan_cuda_load_kernels selective_scan_cuda_identity \
  cross_scan_expand_cuda cross_scan_merge_cuda; do
  renames+=("-D$symbol=${symbol}_baseline")
done
scan_flags='--ftz=false --prec-div=true --prec-sqrt=true --fmad=true -lineinfo'
run compile-baseline "$nvcc" "${flags[@]}" -I"$out/source/baseline/include" \
  "-DSACCADE_BUILD_COMMIT=\"$baseline_revision\"" '-DSACCADE_BUILD_DIRTY="false"' \
  "-DSACCADE_SCAN_CUDA_FLAGS=\"$scan_flags\"" "${renames[@]}" \
  -c "$out/source/baseline/cuda/selective_scan.cu" -o "$out/objects/baseline.o"
run compile-candidate "$nvcc" "${flags[@]}" -I"$out/source/candidate/include" \
  "-DSACCADE_BUILD_COMMIT=\"$candidate_revision\"" "-DSACCADE_BUILD_DIRTY=\"$candidate_dirty\"" \
  "-DSACCADE_SCAN_CUDA_FLAGS=\"$scan_flags\"" \
  -c "$out/source/candidate/cuda/selective_scan.cu" -o "$out/objects/candidate.o"
run compile-harness "$nvcc" -O3 -DNDEBUG -std=c++20 "-arch=sm_$arch" -ccbin "$cxx" \
  -Xcompiler=-Wall,-Wextra,-Wpedantic,-Werror -I"$out/source/candidate/include" \
  -c "$out/source/candidate/tools/scan_benchmark.cpp" -o "$out/objects/harness.o"
run link "$nvcc" -O3 -std=c++20 "-arch=sm_$arch" -ccbin "$cxx" \
  "$out/objects/harness.o" "$out/objects/baseline.o" "$out/objects/candidate.o" -o "$out/scan_benchmark"
run binary-identity bash -c 'set -e
  sha256sum "$1"
  ldd "$1"
' _ "$out/scan_benchmark"
if [[ $build_only == 0 ]]; then
  gpu_lock=$HOME/.cache/saccade-v1/gpu.lock
  mkdir -p "$(dirname "$gpu_lock")"
  run verify flock -w "$lock_timeout" "$gpu_lock" \
    timeout --signal=TERM --kill-after=15s "${verify_timeout}s" "$out/scan_benchmark" "$out/verify" --verify-only
  run benchmark flock -w "$lock_timeout" "$gpu_lock" \
    timeout --signal=TERM --kill-after=15s "${benchmark_timeout}s" "$out/scan_benchmark" "$out/benchmark" \
    --benchmark-only --rounds 4 --iterations 100 --warmup 20
fi
run frozen-source-check bash -c 'set -e
  cd "$1/source"
  sha256sum -c "$1/capture/frozen-source.sha256"
' _ "$out"
printf 'Development scan harness complete: %s (build_only=%s)\n' "$out" "$build_only"

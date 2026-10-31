#!/usr/bin/env bash
# Positional parameters in bash -c strings are intentionally expanded by that child.
# shellcheck disable=SC2016
# Reproducible F01/F02 capture. No source commits or Python runtime dependency.
# Usage: bash scripts/run_preprocess_acceptance.sh build-NEW-OUTPUT [--allow-dirty]
# Needs pinned T01 export venv/data, CUDA, a hardware JPEG GPU, djpeg and sanitizers.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
usage() { echo 'usage: run_preprocess_acceptance.sh build-NEW-OUTPUT [--allow-dirty]' >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || usage
out=$1
allow_dirty=false
if [[ $# == 2 ]]; then [[ $2 == --allow-dirty ]] || usage; allow_dirty=true; fi
[[ $out == build-* && $out != */* && ! -e $out && ! -L $out ]] || {
  echo 'output must be a new build-* directory at the repository root' >&2; exit 2;
}
if [[ $allow_dirty == false && -n $(git status --porcelain) ]]; then
  echo 'acceptance requires clean committed source; --allow-dirty records development evidence only' >&2
  exit 2
fi
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
t01=${SACCADE_T01_ROOT:-$HOME/.cache/saccade-v1/t01}
py=$t01/venv-export/bin/python
cxx=${CXX:-c++}
arch=${SACCADE_CUDA_ARCHITECTURES:-120}
sanitizer_timeout=${SACCADE_PREPROCESS_SANITIZER_TIMEOUT:-300}
[[ $sanitizer_timeout =~ ^[1-9][0-9]*$ && $arch =~ ^[0-9]+$ ]] || usage
for dependency in cmake ctest nvidia-smi timeout djpeg sha256sum awk "$cxx" "$nvcc" "$sanitizer" "$py" /usr/bin/time; do
  command -v "$dependency" >/dev/null || { echo "missing dependency: $dependency" >&2; exit 2; }
done
mkdir "$out"
out=$(cd "$out" && pwd)
mkdir "$out/capture"
cap=$out/capture
revision=$(git rev-parse HEAD)
# Hash both tracked and untracked nonignored source; build-* outputs are ignored.
# This guards development captures against edits while a command is running too.
source_digest() {
  {
    git rev-parse HEAD
    git status --porcelain=v1 --untracked-files=all
    git diff --binary HEAD
    git diff --binary --cached
    git ls-files -c -o --exclude-standard -z | sort -zu | while IFS= read -r -d '' path; do
      if [[ -L $path ]]; then printf 'symlink %s %s\n' "$path" "$(readlink -- "$path")"
      elif [[ -f $path ]]; then sha256sum -- "$path"
      else printf 'missing %s\n' "$path"
      fi
    done
  } | sha256sum | awk '{print $1}'
}
source_sha=$(source_digest)
check_source() {
  if [[ $(source_digest) != "$source_sha" ]]; then
    { date -Is; git rev-parse HEAD; git status --porcelain=v1 --untracked-files=all; } >> "$cap/source-drift.log"
    echo 'source changed during capture; retained output is incomplete' >&2
    return 2
  fi
}
run() {
  local name=$1 status=0
  shift
  check_source
  printf '%q ' "$@" > "$cap/$name.command"
  printf '\n' >> "$cap/$name.command"
  printf '%s: started\n' "$name"
  /usr/bin/time -v -o "$cap/$name.time" "$@" > "$cap/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$status"
  check_source
  return "$status"
}
finish() {
  local status=$?
  trap - EXIT
  printf 'exit=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$cap/capture-status.txt"
  if [[ $status != 0 ]]; then echo "capture failed; retained commands/logs/statuses: $cap" >&2; fi
  exit "$status"
}
trap finish EXIT
printf 'revision=%s\nsource_digest=%s\nallow_dirty=%s\nstarted=%s\n' \
  "$revision" "$source_sha" "$allow_dirty" "$(date -Is)" > "$cap/source.txt"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status.txt"
git diff --binary HEAD > "$cap/source.diff"
git diff --binary --cached > "$cap/source-staged.diff"
sha256sum "$cap/source.diff" "$cap/source-staged.diff" > "$cap/source-diffs.sha256"
git ls-files -c -o --exclude-standard -z | sort -zu | while IFS= read -r -d '' path; do
  [[ ! -f $path ]] || sha256sum -- "$path"
done > "$cap/source-files.sha256"
run environment bash -c 'set -e
  date -Is
  uname -srmo
  git --version
  cmake --version
  "$1" --version
  "$2" --version
  "$3" --version
  "$4" --version
  djpeg -version
  nvidia-smi --query-gpu=name,driver_version,uuid,pci.bus_id,compute_cap --format=csv
  free -m' _ "$cxx" "$nvcc" "$sanitizer" "$py"
run reference-generate "$py" training/t01/t01_preprocess_ref.py generate --out "$out/refs"
run reference-hashes bash -c 'set -e; cd "$1"; find . -type f -print0 | sort -z | xargs -0 sha256sum' _ "$out/refs"
run configure cmake -S . -B "$out/build" -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON \
  "-DCMAKE_CXX_COMPILER=$cxx" "-DCMAKE_CUDA_COMPILER=$nvcc" \
  "-DCMAKE_CUDA_ARCHITECTURES=$arch" "-DSACCADE_PREPROCESS_REFERENCES=$out/refs"
run build cmake --build "$out/build" --parallel 2
run build-identity bash -c 'set -e
  ldd "$1/preprocess"
  library=$(ldd "$1/preprocess" | awk '\''$1 ~ /^libnvjpeg/ && $2 == "=>" {print $3}'\'')
  [[ -n $library && -f $library ]]
  readlink -f "$library"
  sha256sum "$library" "$1/preprocess" "$1/preprocess_tests" "$1/resize_cuda_tests"
  cat "$1/CMakeCache.txt"' _ "$out/build"
# Serial CTest avoids overlapping GPU/sanitizer work; existing CPU and GPU
# regression tests also execute in this CUDA + preprocessing build.
run ctest ctest --test-dir "$out/build" --output-on-failure --parallel 1
run independent-compare "$py" training/t01/t01_preprocess_ref.py compare \
  --refs "$out/refs" --dump "$out/build/preprocess-reference"
run jpeg-cli "$out/build/preprocess" jpeg fixtures/preprocess/baseline.jpg "$out/jpeg-cli"

sanitize() {
  local name=$1 tool=$2 available
  shift 2
  available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  printf 'time=%s\navailable_kib=%s\nminimum_kib=4194304\n' "$(date -Is)" "$available" > "$cap/$name.memory"
  if [[ ! $available =~ ^[0-9]+$ ]] || (( available < 4194304 )); then
    echo "insufficient MemAvailable before $name; need at least 4 GiB" >&2
    return 2
  fi
  run "$name" timeout --signal=TERM --kill-after=15s "${sanitizer_timeout}s" \
    "$sanitizer" --tool "$tool" --error-exitcode 99 --print-limit 100 "$@"
}
for tool in memcheck initcheck racecheck synccheck; do
  resize_args=()
  if [[ $tool == initcheck ]]; then resize_args+=(--no-held-stream); fi
  sanitize "resize-$tool" "$tool" "$out/build/resize_cuda_tests" "${resize_args[@]}"
  sanitize "native-$tool" "$tool" "$out/build/preprocess_tests" fixtures/preprocess
done
# The first fixed case is a real JPEG: sanitizer coverage includes F01 and F02.
for tool in memcheck initcheck; do
  sanitize "reference-$tool" "$tool" "$out/build/preprocess" run \
    "$out/refs" "$out/reference-$tool" --case 0
done
check_source
printf 'capture complete: %s\n' "$out"
if [[ $allow_dirty == true ]]; then
  printf 'Development capture only: source was allowed dirty; clean-source acceptance remains separate.\n'
fi

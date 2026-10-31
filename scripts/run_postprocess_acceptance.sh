#!/usr/bin/env bash
# Standalone F04 capture. Dirty snapshots are development evidence, never a
# clean-source formal verdict. No source commits or publication are performed.
# bash -c positional parameters are expanded by the child shell.
# shellcheck disable=SC2016
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
usage() { echo 'usage: run_postprocess_acceptance.sh build-NEW-OUTPUT [--allow-dirty]' >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || usage
out=$1
allow_dirty=false
if [[ $# == 2 ]]; then [[ $2 == --allow-dirty ]] || usage; allow_dirty=true; fi
[[ $out == build-* && $out != */* && ! -e $out && ! -L $out ]] || usage
if [[ $allow_dirty == false && -n $(git status --porcelain) ]]; then
  echo 'clean committed source required; --allow-dirty records development evidence only' >&2
  exit 2
fi
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
arch=${SACCADE_CUDA_ARCHITECTURES:-120}
dumps=${SACCADE_F04_F03_DUMPS:-$PWD/build-t01-f03-evidence/model-dump}
limit=${SACCADE_F04_SANITIZER_TIMEOUT:-240}
[[ $arch =~ ^[0-9]+$ && $limit =~ ^[1-9][0-9]*$ ]] || usage
for dependency in cmake ctest timeout sha256sum rg nvidia-smi "$nvcc" "$sanitizer" /usr/bin/time; do
  command -v "$dependency" >/dev/null || { echo "missing dependency: $dependency" >&2; exit 2; }
done
for index in {0..5}; do [[ -f $dumps/input-$index.f32 ]] || { echo "missing pinned F03 dump: $dumps/input-$index.f32" >&2; exit 2; }; done
mkdir "$out"
out=$(cd "$out" && pwd)
cap=$out/capture
mkdir "$cap"
source_hashes() {
  git ls-files -c -o --exclude-standard -z -- CMakeLists.txt cmake include src cuda tests tools fixtures \
    docs/POSTPROCESS_POLICY.md scripts/run_postprocess_acceptance.sh scripts/check_postprocess_mutations.sh |
    sort -zu | while IFS= read -r -d '' path; do
      if [[ -f $path ]]; then sha256sum -- "$path"; else printf 'missing %s\n' "$path"; fi
    done
}
source_hashes > "$cap/source-files.sha256"
source_sha=$(sha256sum < "$cap/source-files.sha256" | cut -d' ' -f1)
revision=$(git rev-parse HEAD)
check_source() {
  [[ $(git rev-parse HEAD) == "$revision" && $(source_hashes | sha256sum | cut -d' ' -f1) == "$source_sha" ]] || {
    echo 'build inputs or frozen policy changed during capture' >&2; return 2;
  }
}
run() {
  local name=$1 status=0
  shift
  check_source
  printf '%q ' "$@" > "$cap/$name.command"; printf '\n' >> "$cap/$name.command"
  date -Is > "$cap/$name.started"
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
  exit "$status"
}
trap finish EXIT
printf 'revision=%s\nsource_digest=%s\nallow_dirty=%s\nstarted=%s\n' \
  "$revision" "$source_sha" "$allow_dirty" "$(date -Is)" > "$cap/source.txt"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status.txt"
git diff --binary HEAD > "$cap/source.diff"
sha256sum docs/POSTPROCESS_POLICY.md src/postprocess.cpp tests/postprocess_tests.cpp tests/postprocess_cuda_tests.cpp > "$cap/frozen-policy-reference-cases.sha256"
run environment bash -c 'set -e
  date -Is
  uname -srmo
  cmake --version
  c++ --version
  "$1" --version
  "$2" --version
  nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap,memory.total --format=csv
  free -m' _ "$nvcc" "$sanitizer"
run configure cmake -S . -B "$out/cuda" -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_POSTPROCESS=ON \
  -DCMAKE_CUDA_COMPILER="$nvcc" -DCMAKE_CUDA_ARCHITECTURES="$arch" \
  -DSACCADE_POSTPROCESS_F03_DUMPS="$dumps" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
run build cmake --build "$out/cuda" --parallel 3 --target postprocess_tests postprocess_cuda_tests postprocess_sanitizer_control
run binaries sha256sum "$out/cuda/postprocess_tests" "$out/cuda/postprocess_cuda_tests" "$out/cuda/postprocess_sanitizer_control"
# No GPU work has occurred in this new capture before the hashes and fixture
# names/invariants are recorded. --check-fixtures exits before cudaSetDevice.
run cases "$out/cuda/postprocess_cuda_tests" --list-cases
run fixtures "$out/cuda/postprocess_cuda_tests" --check-fixtures
run ctest ctest --test-dir "$out/cuda" -R '^postprocess_' --output-on-failure
run full-replay "$out/cuda/postprocess_cuda_tests" --f03-dumps "$dumps"
for tool in memcheck initcheck racecheck synccheck; do
  run "sanitizer-$tool" timeout "$limit" "$sanitizer" --tool "$tool" --error-exitcode 99 \
    "$out/cuda/postprocess_cuda_tests" --no-held-stream --f03-dumps "$dumps"
  if [[ $tool == racecheck ]]; then
    summary='========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
  else summary='========= ERROR SUMMARY: 0 errors'; fi
  [[ $(rg -Fxc -- "$summary" "$cap/sanitizer-$tool.log") == 1 ]] || {
    echo "missing or ambiguous zero-error $tool summary" >&2; exit 2;
  }
  rg -q '^checks=[0-9]+ max_score_abs_error=[0-9.e+-]+ PASS$' "$cap/sanitizer-$tool.log"
  case $tool in
    memcheck) diagnostic='Invalid __global__ write';;
    initcheck) diagnostic='Uninitialized __global__ memory read';;
    racecheck) diagnostic='Error: Race reported between';;
    synccheck) diagnostic='Barrier error detected';;
  esac
  status=0
  run "control-$tool" timeout "$limit" "$sanitizer" --tool "$tool" --error-exitcode 99 \
    "$out/cuda/postprocess_sanitizer_control" "$tool" || status=$?
  if [[ $status != 99 ]] || ! rg -Fq -- "$diagnostic" "$cap/control-$tool.log"; then
    echo "$tool positive control was not detected with its expected diagnostic and exit 99" >&2; exit 2;
  fi
done
cp "$out/cuda/compile_commands.json" "$cap/compile_commands.json"
check_source
printf 'PASS\n' > "$cap/standalone-checks.txt"

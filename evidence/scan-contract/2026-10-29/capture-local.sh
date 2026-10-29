#!/usr/bin/env bash
# Read-only source audit plus CPU baseline regression; no Python or GPU work.
set -euo pipefail
task_root=$(git rev-parse --show-toplevel)
cd "$task_root"
task_out=${1:?usage: bash capture-local.sh build-OUTPUT}
[[ $task_out == build-* && ! -e $task_out ]] || exit 2
mkdir -p "$task_out/capture" "$task_out/source"
task_rev=9f2c11a838a229c05ba7e9e17fa182bf0799746b
run() {
  local name=$1 code=0
  shift
  printf '%q ' "$@" > "$task_out/capture/$name.command"
  printf '\n' >> "$task_out/capture/$name.command"
  "$@" > "$task_out/capture/$name.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$task_out/capture/$name.exit"
  printf '%s: exit=%s\n' "$name" "$code"
  return "$code"
}
run environment bash -c 'date -Is; uname -srmo; git --version; cmake --version; c++ --version; node --version; jq --version'
run source git rev-parse HEAD origin/main
run working-tree git status --porcelain=v1
run tracked-tree git ls-tree -r --long "$task_rev"
run source-archive git archive --format=tar --output="$task_out/source.tar" "$task_rev"
run archive-hash sha256sum "$task_out/source.tar"
run source-extract tar -xf "$task_out/source.tar" -C "$task_out/source"
run source-file-hashes bash -o pipefail -c 'git ls-tree -r --name-only "$1" CMakeLists.txt cmake src include cuda tools tests fixtures scripts .github | while IFS= read -r path; do sha256sum "$2/$path"; done' _ "$task_rev" "$task_out/source"
run cpu-configure cmake -S "$task_out/source" -B "$task_out/cpu" -DCMAKE_BUILD_TYPE=Release
run cpu-build cmake --build "$task_out/cpu" --parallel 4
run cpu-ctest ctest --test-dir "$task_out/cpu" --verbose
run no-runtime-change git diff "$task_rev" -- CMakeLists.txt cmake src include cuda tools tests fixtures scripts .github
run historical-evidence-unchanged git diff "$task_rev" -- evidence/p01-a evidence/cuda-synthetic
run baseline-evidence-hashes sha256sum -c evidence/p01-a/2026-10-28/SHA256SUMS
run task-file-hash sha256sum '/tmp/Saccade v1 新 head 契約核實與 S01 C++ 參考實作.md'

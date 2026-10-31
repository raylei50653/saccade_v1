#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=$PWD/build-preprocess-evidence
cap=$out/supplement
mkdir "$cap"
git rev-parse HEAD > "$cap/source.txt"
git status --porcelain > "$cap/source-status.txt"
test ! -s "$cap/source-status.txt"
failed=0
run() {
  local name=$1 status=0
  shift
  test "$(git rev-parse HEAD)" = 2b39ddaf7cefcbdfb8a7fc143ecfd83466618cb3
  test -z "$(git status --porcelain)"
  printf '%q ' "$@" > "$cap/$name.command"
  printf '\n' >> "$cap/$name.command"
  /usr/bin/time -v -o "$cap/$name.time" "$@" > "$cap/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$cap/$name.exit"
  printf '%s exit=%s\n' "$name" "$status"
  if [[ $status != 0 ]]; then failed=$status; fi
}
sanitize() {
  local name=$1 tool=$2 available
  shift 2
  available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  printf 'available_kib=%s\nminimum_kib=4194304\n' "$available" > "$cap/$name.memory"
  (( available >= 4194304 ))
  run "$name" timeout --signal=TERM --kill-after=15s 300s /opt/cuda/bin/compute-sanitizer \
    --tool "$tool" --error-exitcode 99 --print-limit 100 "$@"
}
for tool in racecheck synccheck; do
  sanitize "resize-$tool" "$tool" "$out/build/resize_cuda_tests"
  sanitize "native-$tool" "$tool" "$out/build/preprocess_tests" fixtures/preprocess
done
for tool in memcheck initcheck; do
  sanitize "reference-$tool" "$tool" "$out/build/preprocess" run "$out/refs" "$out/reference-$tool" --case 0
done
test -z "$(git status --porcelain)"
printf 'exit=%s\nfinished=%s\n' "$failed" "$(date -Is)" > "$cap/capture-status.txt"
exit "$failed"

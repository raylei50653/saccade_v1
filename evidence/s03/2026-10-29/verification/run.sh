#!/usr/bin/env bash
# Document and script checks for this round; writes beside this script.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
here=evidence/s03/2026-10-29/verification
test -f evidence/s03/2026-10-29/REPORT.md
run() {
  local name=$1 code=0
  shift
  printf '%q ' "$@" > "$here/$name.command"; printf '\n' >> "$here/$name.command"
  "$@" > "$here/$name.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$here/$name.exit"
  printf '%s: exit=%s\n' "$name" "$code"
  return "$code"
}
run markdown-links node "$here/verify-links.mjs"
# Only authored paths are checked. Raw capture logs, traces, reports and diffs
# under capture/ retain the exact bytes produced by the tools.
run whitespace bash -c '
set -euo pipefail
authored=(
  README.md docs/SCAN_CUDA.md docs/SCAN_TRT.md CMakeLists.txt
  cmake/ScanTensorRT.cmake ":(glob)include/saccade/scan*" cuda/scan_frame.cu
  ":(glob)src/scan_trt*" ":(glob)tests/scan_trt*" tests/scan_cuda_suite.hpp
  tests/selective_scan_cuda_tests.cpp tools/scan_trt_engine.cpp
  evidence/s03/2026-10-29/REPORT.md ":(glob)evidence/s03/2026-10-29/*.sh"
  ":(glob)evidence/s03/2026-10-29/verification/*.mjs"
  ":(glob)evidence/s03/2026-10-29/verification/*.sh"
)
git diff --check e9dfa51 -- "${authored[@]}"
# git diff does not inspect untracked authored files before their commit.
while IFS= read -r -d "" file; do
  check_code=0
  check_output=$(git diff --no-index --check -- /dev/null "$file" 2>&1) || check_code=$?
  # --no-index may return 1 for the file difference alone. A clean check
  # emits no diagnostics; whitespace findings or command failures still fail.
  if [[ -n $check_output || $check_code -gt 1 ]]; then
    printf "%s\n" "$check_output"
    echo "$file: whitespace check failed (exit=$check_code)" >&2
    exit 1
  fi
done < <(git ls-files --others --exclude-standard -z -- "${authored[@]}")
echo "authored text: no whitespace errors"
'
run script-syntax bash -c '
set -euo pipefail
for file in evidence/s03/2026-10-29/*.sh evidence/s03/2026-10-29/verification/*.sh; do
  bash -n "$file"
  echo "$file: ok"
done
'

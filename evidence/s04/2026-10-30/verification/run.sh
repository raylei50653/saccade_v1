#!/usr/bin/env bash
# Document and script checks for this round; writes beside this script.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
here=evidence/s04/2026-10-30/verification
test -f evidence/s04/2026-10-30/REPORT.md
run() {
  local name=$1 code=0
  shift
  printf '%q ' "$@" > "$here/$name.command"; printf '\n' >> "$here/$name.command"
  "$@" > "$here/$name.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$here/$name.exit"
  printf '%s: exit=%s\n' "$name" "$code"
  return "$code"
}
run mutation-classification-fixtures bash -c 'node "$1" > "$2"' _ \
  "$here/classify-mutation-fixtures.mjs" "$here/mutation-classification-fixtures.json"
run default-memcheck-fixtures bash -c 'node "$1" > "$2"' _ \
  "$here/default-memcheck-fixtures.mjs" "$here/default-memcheck-fixtures.json"
run markdown-links node "$here/verify-links.mjs"
# Only authored paths are checked. Raw capture logs, reports and diffs under
# capture/ retain the exact bytes produced by the tools.
run whitespace bash -c '
set -euo pipefail
authored=(
  README.md docs/HEAD_TRT.md docs/HEAD_POLICY.md docs/MODEL_MANIFEST.md docs/SCAN_TRT.md docs/MODEL_ARTIFACT_READINESS.md
  docs/MODEL_FEASIBILITY.md docs/SCAN_CONTRACT.md CMakeLists.txt ":(glob)cmake/Head*" ":(glob)include/saccade/head*"
  include/saccade/torch_checkpoint.hpp include/saccade/scan_trt_frame.hpp include/saccade/scan_frame_kernels.hpp
  include/saccade/model_manifest.hpp ":(glob)src/head*" src/torch_checkpoint.cpp src/scan_trt_frame.cpp src/model_manifest.cpp
  cuda/scan_frame.cu ":(glob)tests/head*" tests/model_manifest_tests.cpp ":(glob)tools/head*" tools/model_manifest.cpp
  ":(glob)fixtures/model-manifest/*" ":(glob)manifests/s04/*" .github/workflows/protocol.yml
  evidence/s04/2026-10-30/REPORT.md ":(glob)evidence/s04/2026-10-30/*.sh"
  ":(glob)evidence/s04/2026-10-30/verification/*.mjs" ":(glob)evidence/s04/2026-10-30/verification/*.sh"
)
git diff --check f02dd7a -- "${authored[@]}"
# git diff does not inspect untracked authored files before their commit.
while IFS= read -r -d "" file; do
  check_code=0
  check_output=$(git diff --no-index --check -- /dev/null "$file" 2>&1) || check_code=$?
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
for file in evidence/s04/2026-10-30/*.sh evidence/s04/2026-10-30/verification/*.sh; do
  bash -n "$file"
  echo "$file: ok"
done
for file in evidence/s04/2026-10-30/verification/*.mjs; do
  node --check "$file"
  echo "$file: ok"
done
'

#!/usr/bin/env bash
# Document and script checks for this round; writes beside this script.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
here=evidence/s01/2026-10-29/verification
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
# Authored text only; captured command output keeps its original bytes.
run whitespace bash -c 'git diff --check 2aef5f8feaec6126f814cd1ed1cabf85f8994c9b -- README.md docs CMakeLists.txt .github include src tests fixtures/selective-scan && ! grep -nE "[[:blank:]]$" evidence/s01/2026-10-29/REPORT.md evidence/s01/2026-10-29/audit/*.sh evidence/s01/2026-10-29/audit/*.cpp evidence/s01/2026-10-29/reference/*.sh evidence/s01/2026-10-29/verification/*.mjs evidence/s01/2026-10-29/verification/*.sh && echo "authored text: no whitespace errors"'
run script-syntax bash -c 'for f in evidence/s01/2026-10-29/audit/*.sh evidence/s01/2026-10-29/reference/*.sh evidence/s01/2026-10-29/verification/run.sh fixtures/selective-scan/derive.sh; do bash -n "$f" && echo "$f: ok"; done'

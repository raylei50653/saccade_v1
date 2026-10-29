#!/usr/bin/env bash
# Document and script checks for this round; writes beside this script.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
here=evidence/s02/2026-10-29/verification
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
run whitespace bash -c 'git diff --check 60d722cfa45bb6d9763740cf6c8103c9defa9c0d -- README.md docs CMakeLists.txt include cuda tests && ! grep -nE "[[:blank:]]$" evidence/s02/2026-10-29/REPORT.md evidence/s02/2026-10-29/*.sh evidence/s02/2026-10-29/*.cu evidence/s02/2026-10-29/verification/*.mjs evidence/s02/2026-10-29/verification/*.sh && echo "authored text: no whitespace errors"'
run script-syntax bash -c 'for f in evidence/s02/2026-10-29/*.sh evidence/s02/2026-10-29/verification/run.sh; do bash -n "$f" && echo "$f: ok"; done'

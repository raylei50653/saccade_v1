#!/usr/bin/env bash
# Verify earlier evidence bundles without editing them. The 2026-10-29
# scan-contract SHA256SUMS also pins four repository documents that later
# rounds legitimately update; those entries are checked against the commit
# that delivered that bundle instead of the working tree.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for sums in evidence/p01-a/2026-10-28/SHA256SUMS evidence/cuda-synthetic/2026-10-28/SHA256SUMS \
            evidence/cuda-synthetic/2026-10-28-closeout/SHA256SUMS; do
  printf '== %s\n' "$sums"
  sha256sum --quiet -c "$sums" && echo "all entries OK"
done
sums=evidence/scan-contract/2026-10-29/SHA256SUMS delivered=88a5c9d518558ae20dd94089ae732ef3ff7b374e
printf '== %s (evidence entries)\n' "$sums"
grep '  evidence/' "$sums" | sha256sum --quiet -c - && echo "all $(grep -c '  evidence/' "$sums") evidence entries OK"
printf '== %s (repository documents, against %s)\n' "$sums" "$delivered"
grep -v '  evidence/' "$sums" | while read -r hash path; do
  actual=$(git show "$delivered:$path" | sha256sum | cut -d' ' -f1)
  [[ $actual == "$hash" ]] && echo "$path: OK at $delivered" || { echo "$path: MISMATCH at $delivered"; exit 1; }
done

#!/usr/bin/env bash
# For every ONNX initializer, list the checkpoint state_dict keys whose exact
# bytes hash equal (directly, or transposed for 2-D MatMul weights).
set -euo pipefail
onnx=$1; shift
for ckpt; do
  label=$(basename "$ckpt" .log)
  awk -F'\t' -v label="$label" '
    FNR == NR { if (match($2, /sha256=[0-9a-f]+/)) key[substr($2, RSTART + 7, 64)] = key[substr($2, RSTART + 7, 64)] $1 " "; next }
    $1 == "initializer" {
      direct = ""; trans = ""
      if (match($6, / sha256=[0-9a-f]+/)) direct = substr($6, RSTART + 8, 64)
      if (match($6, /sha256_transposed=[0-9a-f]+/)) trans = substr($6, RSTART + 18, 64)
      hit = (direct in key) ? "direct:" key[direct] : (trans != "" && trans in key) ? "transposed:" key[trans] : "none"
      total++; if (hit != "none") matched++
      printf "%s\t%s\t%s\t%s\n", label, $3, $5, hit
    }
    END { printf "%s\tmatched=%d/%d\n", label, matched, total }' "$ckpt" "$onnx"
done

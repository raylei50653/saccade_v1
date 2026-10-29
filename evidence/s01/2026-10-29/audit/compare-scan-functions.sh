#!/usr/bin/env bash
# Compare the scan-relevant top-level definitions of mamba_head.py between two
# legacy revisions, plus the spatial branch of MambaDetectionHead._forward_eager.
set -euo pipefail
legacy=$1 old=$2 new=$3
file=src/saccade/perception/temporal_yolo/mamba_head.py
top() {
  git -C "$legacy" show "$1:$file" |
    awk -v start="$2" 'index($0, start) == 1 { p = 1; print; next }
         p && /^(def |class |@torch|# -{5})/ { p = 0 }
         p { print }'
}
method() {
  git -C "$legacy" show "$1:$file" |
    awk -v start="$2" 'index($0, start) == 1 { p = 1; print; next }
         p && /^    def / { p = 0 }
         p { print }'
}
status=0
for def in "def _selective_scan(" "def _selective_scan_cuda(" "def _selective_scan_jit(" \
           "class MambaBlock(" "def _cross_scan_mamba("; do
  a=$(top "$old" "$def"); b=$(top "$new" "$def")
  [[ -n $a && -n $b ]] || { echo "missing $def" >&2; exit 2; }
  printf '== %s lines %s -> %s\n' "$def" "$(wc -l <<<"$a")" "$(wc -l <<<"$b")"
  diff <(printf '%s\n' "$a") <(printf '%s\n' "$b") || status=1
done
a=$(method "$old" "    def _forward_eager(" | sed -n '/for i, x in enumerate(feats)/,/x_cls = torch.cat/p')
b=$(method "$new" "    def _forward_eager(" | sed -n '/for i, x in enumerate(feats)/,/x_cls = torch.cat/p')
printf '== _forward_eager spatial branch lines %s -> %s\n' "$(wc -l <<<"$a")" "$(wc -l <<<"$b")"
diff <(printf '%s\n' "$a") <(printf '%s\n' "$b") || status=1
# Differences are reported, not hidden; the caller reviews them in the log.
echo "diff_status=$status"

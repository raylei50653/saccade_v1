#!/usr/bin/env bash
# M4 scoring (docs/TRACKING.md §17.2): the only step that reads MOT17 ground
# truth. Runs on the output of run_mot17_evaluation.sh after its gates passed,
# at the same source. MOT17-02 is scored last.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
usage() { echo 'usage: score_mot17_evaluation.sh build-OUTPUT [--allow-dirty]' >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || usage
allow_dirty=false
if [[ $# == 2 ]]; then [[ $2 == --allow-dirty ]] || usage; allow_dirty=true; fi
[[ -d $1 ]] || usage
out=$(cd "$1" && pwd)
cap=$out/capture
[[ $(cat "$cap/gates.txt") == PASS ]] || { echo 'evaluation gates did not pass' >&2; exit 2; }
[[ ! -e $cap/scoring.txt && ! -e $out/score ]] || { echo 'already scored' >&2; exit 2; }
recorded() { sed -n "s/^$1=//p" "$cap/source.txt"; }
[[ $allow_dirty == true || -z $(git status --porcelain) ]] || { echo 'clean source required; --allow-dirty records development evidence' >&2; exit 2; }
[[ $(recorded allow_dirty) == "$allow_dirty" ]] || { echo 'dirty mode differs from the evaluation run' >&2; exit 2; }
[[ $(git rev-parse HEAD) == $(recorded revision) ]] || { echo 'HEAD differs from the evaluation run' >&2; exit 2; }
check_source() { sha256sum --quiet -c "$cap/source-files.sha256" || { echo 'source changed since the evaluation run' >&2; return 2; }; }
check_source
sequences=$(recorded sequences)
records=$(recorded records)
legacy=${SACCADE_LEGACY_ROOT:-/home/ray/developer/ai/saccade}
[[ $(cd "$legacy/datasets/MOT17/train" && pwd -P) == "$sequences" ]] || { echo 'TrackEval GT root is not the evaluated sequence directory' >&2; exit 2; }
python=${SACCADE_EVAL_PYTHON:-$HOME/.cache/saccade-v1/t01/venv/bin/python}
run() {
  local name=$1 status=0
  shift
  check_source
  printf '%q ' "$@" > "$cap/$name.command"; printf '\n' >> "$cap/$name.command"
  date -Is > "$cap/$name.started"
  printf '%s: started\n' "$name"
  "$@" > "$cap/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$status"
  check_source
  return "$status"
}
ap50() { # NAME SCDR SEQUENCE
  run "$1" env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$python" training/hr_deploy/eval_scdr.py "$2" "$sequences/$3"
  cp "$cap/$1.log" "$cap/$1.json"
}
track() { # GROUP SEQUENCE...
  local group=$1 dir=$out/score/$1
  shift
  mkdir -p "$dir"
  for s in "$@"; do cp "$out/mot/$s.txt" "$dir/"; done
  run "trackeval-$group" env SACCADE_LEGACY_ROOT="$legacy" scripts/eval_mot17.sh "$dir" "$@"
  cp "$cap/trackeval-$group.log" "$cap/trackeval-$group.txt"
}
domain=(MOT17-04-SDP MOT17-05-SDP MOT17-09-SDP MOT17-11-SDP MOT17-13-SDP)
for s in "${domain[@]}" MOT17-10-SDP; do
  ap50 "ap50-hr-$s" "$out/scdr/$s.scdr" "$s"
  ap50 "ap50-t01-$s" "$records/$s.scdr" "$s"
done
track val MOT17-10-SDP
track in-domain "${domain[@]}"
for s in "${domain[@]}"; do track "$s" "$s"; done
# MOT17-02 (eval) last; opened here for the HR detector. T01 was scored in M1.
ap50 ap50-t01-MOT17-02-SDP "$records/MOT17-02-SDP.scdr" MOT17-02-SDP
ap50 ap50-hr-MOT17-02-SDP "$out/scdr/MOT17-02-SDP.scdr" MOT17-02-SDP
track eval MOT17-02-SDP

field() { sed -n "s/.*\\b$2=\\([^ ]*\\).*/\\1/p" "$cap/trackeval-$1.txt"; }
percent() { jq -er .ap50_percent "$cap/$1.json"; }
ms() { awk -F'\t' -v s="$1" '$1 == s { print $4 }' "$cap/timing.tsv"; }
{
  printf 'row\tHR_AP50\tT01_AP50\tHOTA\tDetA\tAssA\tIDF1\tMOTA\tIDSW\tms_per_frame\n'
  row() { # ROW GROUP SEQUENCE-or-empty
    local hr='' t01='' t=''
    [[ -n $3 ]] && { hr=$(percent "ap50-hr-$3"); t01=$(percent "ap50-t01-$3"); t=$(ms "$3"); }
    printf '%s\t%s\t%s' "$1" "$hr" "$t01"
    for k in HOTA DetA AssA IDF1 MOTA IDSW; do printf '\t%s' "$(field "$2" "$k")"; done
    printf '\t%s\n' "$t"
  }
  row val val MOT17-10-SDP
  row eval eval MOT17-02-SDP
  row in-domain in-domain ''
  for s in "${domain[@]}"; do row "$s" "$s" "$s"; done
} > "$cap/metrics.tsv"
cat "$cap/metrics.tsv"
check_source
printf 'PASS\n' > "$cap/scoring.txt"

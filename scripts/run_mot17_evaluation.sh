#!/usr/bin/env bash
# M4 evaluation run (docs/TRACKING.md §17.3 step 2): the deployed HR FP16 path
# on the 7 MOT17 SDP sequences with the binaries of a passed formal acceptance
# at the same commit, then every gate that does not read ground truth. Only
# score_mot17_evaluation.sh reads GT, after this script writes gates.txt.
# Run it under the GPU lock.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
usage() { echo 'usage: run_mot17_evaluation.sh ACCEPTANCE_DIR build/NEW-OUTPUT [--allow-dirty]' >&2; exit 2; }
[[ $# -ge 2 && $# -le 3 ]] || usage
acceptance=$1
out=$2
allow_dirty=false
if [[ $# == 3 ]]; then [[ $3 == --allow-dirty ]] || usage; allow_dirty=true; fi
[[ -d $acceptance && $out =~ ^build/[A-Za-z0-9][A-Za-z0-9._-]*$ && ! -e $out && ! -L $out ]] || usage
[[ $allow_dirty == true || -z $(git status --porcelain) ]] || { echo 'clean source required; --allow-dirty records development evidence' >&2; exit 2; }
acceptance=$(cd "$acceptance" && pwd)
sequences=$(cd "${SACCADE_TRACKING_SEQUENCES:?MOT17 train directory}" && pwd -P)
records=$(cd "${SACCADE_TRACKING_RECORDS:?M1 SCDR/SCMO records}" && pwd -P)
engine=${SACCADE_HR_ENGINE:-$HOME/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine}
deploy=${SACCADE_HR_DEPLOY:-$HOME/.cache/saccade-v1/hr-deploy}
nsys=${NSYS:-nsys}
# Fixed in §17.3 (values from §16).
engine_sha=8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876
scdr10_sha=98e2554f1b1db179bfd756fb1dbd37ee694e912d6debb2c7a11c3a610a433a6f
mot10_sha=1fb713f30444d2c6e3e9ae30d34621f3155b0967b3a52a9244f107e28547fba7
build_report=evidence/precision/2026-11-03-formal/build-report.txt
recorded_mot=evidence/precision/2026-11-03-formal/tracking-mot.sha256
sdp=(MOT17-02-SDP MOT17-04-SDP MOT17-05-SDP MOT17-09-SDP MOT17-10-SDP MOT17-11-SDP MOT17-13-SDP)
bin=$acceptance/cuda
[[ -f $engine && -f $deploy/manifest.json && -d $sequences && -d $records ]] || { echo 'missing engine, deploy manifest, sequences or records' >&2; exit 2; }
[[ $(cat "$acceptance/capture/acceptance.txt") == PASS ]] || { echo 'acceptance did not pass' >&2; exit 2; }
[[ $(sed -n 's/^revision=//p' "$acceptance/capture/source.txt") == $(git rev-parse HEAD) ]] || { echo 'acceptance revision is not HEAD' >&2; exit 2; }
binaries() { find "$bin" -maxdepth 1 -type f \( -perm -u+x -o -name '*.a' \) -print0 | sort -z | xargs -0 sha256sum; }
cmp <(binaries) "$acceptance/capture/binaries.sha256" || { echo 'acceptance binaries changed' >&2; exit 2; }
mkdir -p build && mkdir "$out"
out=$(cd "$out" && pwd)
cap=$out/capture
mkdir "$cap"
cp "$acceptance/capture/acceptance.txt" "$acceptance/capture/source.txt" "$acceptance/capture/binaries.sha256" "$cap/"
mv "$cap/acceptance.txt" "$cap/acceptance-acceptance.txt"
mv "$cap/source.txt" "$cap/acceptance-source.txt"
mv "$cap/binaries.sha256" "$cap/acceptance-binaries.sha256"
source_hashes() {
  git ls-files -c -o --exclude-standard -z -- CMakeLists.txt cmake include src cuda tests tools fixtures \
    docs/TRACKING.md scripts/run_tracking_session_acceptance.sh scripts/run_mot17_evaluation.sh \
    scripts/score_mot17_evaluation.sh scripts/check_stream_placement.py scripts/eval_mot17.sh \
    training/hr_deploy/eval_scdr.py training/t01/t01_eval.py "$build_report" "$recorded_mot" |
    sort -zu | while IFS= read -r -d '' path; do sha256sum -- "$path"; done
}
source_hashes > "$cap/source-files.sha256"
revision=$(git rev-parse HEAD)
source_sha=$(sha256sum < "$cap/source-files.sha256" | cut -d' ' -f1)
check_source() {
  [[ $(git rev-parse HEAD) == "$revision" && $(source_hashes | sha256sum | cut -d' ' -f1) == "$source_sha" ]] || { echo 'source changed during capture' >&2; return 2; }
}
# Engine, deploy artifacts, records and each sequence's seqinfo.ini and images
# (never gt/).
artifact_hashes() {
  sha256sum "$engine" "$deploy/manifest.json"
  (cd "$records"; find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum)
  for s in "${sdp[@]}"; do
    printf '%s  %s\n' "$(cd "$sequences/$s" && find seqinfo.ini img1/ -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)" "$s"
  done
}
artifact_hashes > "$cap/artifacts.sha256"
run() {
  local name=$1 status=0
  shift
  check_source
  printf '%q ' "$@" > "$cap/$name.command"; printf '\n' >> "$cap/$name.command"
  date -Is > "$cap/$name.started"
  printf '%s: started\n' "$name"
  /usr/bin/time -v -o "$cap/$name.time" "$@" > "$cap/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$status"
  check_source
  return "$status"
}
gate() { printf '%s\n' "$*" | tee -a "$cap/gates.log"; }
fail() { gate "FAIL: $*"; exit 1; }
finish() {
  local status=$?
  trap - EXIT
  printf 'exit=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$cap/capture-status.txt"
  exit "$status"
}
trap finish EXIT
printf 'revision=%s\nsource_digest=%s\nallow_dirty=%s\nacceptance=%s\nsequences=%s\nrecords=%s\nengine=%s\ndeploy=%s\n' \
  "$revision" "$source_sha" "$allow_dirty" "$acceptance" "$sequences" "$records" "$engine" "$deploy" > "$cap/source.txt"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status.txt"
git diff --binary HEAD > "$cap/source.diff"
run environment bash -c 'date -Is; uname -srmo; "$1" --version; nvidia-smi --query-gpu=name,uuid,driver_version,memory.free,clocks.max.sm --format=csv' _ "$nsys"

# 1. Identity chain: deploy manifest records and the D2 build report.
sha() { sha256sum < "$1" | cut -d' ' -f1; }
identity() {
  local path=$1 expected=$2 bytes=${3:-} actual
  actual=$(sha "$path")
  printf '%s  %s\n' "$actual" "$path" >> "$cap/identity.txt"
  [[ $actual == "$expected" ]] || fail "identity $path sha256 $actual, manifest $expected"
  [[ -z $bytes || $(stat -c %s "$path") == "$bytes" ]] || fail "identity $path size differs from manifest $bytes"
}
manifest=$deploy/manifest.json
cp "$manifest" "$cap/hr-deploy-manifest.json"
for key in student teacher base; do identity "$(jq -er ".$key" "$manifest")" "$(jq -er ".${key}_sha256" "$manifest")"; done
mapfile -t entries < <(jq -er '.artifacts | to_entries[] | "\(.key)\t\(.value.bytes)\t\(.value.sha256)"' "$manifest")
[[ ${#entries[@]} -gt 0 ]] || fail 'deploy manifest lists no artifacts'
for entry in "${entries[@]}"; do
  IFS=$'\t' read -r path bytes expected <<< "$entry"
  identity "$deploy/$path" "$expected" "$bytes"
done
report() { local v; v=$(sed -n "s/^$1=//p" "$build_report"); [[ $(wc -l <<< "$v") == 1 && -n $v ]] || return 1; printf '%s' "$v"; }
artifact() { jq -er --arg k "$1" '.artifacts[$k].sha256' "$manifest"; }
report_onnx=$(report onnx_sha256); report_bundle=$(report bundle_sha256); report_engine=$(report engine_sha256)
manifest_onnx=$(artifact yolo26m-backbone-hr.onnx); manifest_bundle=$(artifact hr-head.swb)
[[ $report_onnx == "$manifest_onnx" ]] || fail 'build report onnx_sha256 differs from the deploy manifest'
[[ $report_bundle == "$manifest_bundle" ]] || fail 'build report bundle_sha256 differs from the deploy manifest'
[[ $report_engine == "$engine_sha" ]] || fail 'build report engine_sha256 is not the deployed engine'
identity "$engine" "$engine_sha"
gate 'identity chain: PASS'

# 2. The deployed path on every sequence.
mkdir "$out/scdr" "$out/mot" "$out/offline"
printf 'sequence\tframes\twall_ms\tms_per_frame\n' > "$cap/timing.tsv"
for s in "${sdp[@]}"; do
  run "detect-$s" "$bin/detect_sequence" "$engine" "$sequences/$s" "$out/scdr/$s.scdr" --profile hr-1280x768
  run "track-$s" "$bin/track_sequence" "$engine" "$sequences/$s" "$out/mot/$s.txt" --profile hr-1280x768
  run "offline-$s" "$bin/track_offline" "$out/scdr/$s.scdr" "$records/$s.scmo" "$out/offline/$s.txt"
  line=$(rg '^track_sequence frames=[0-9]+ mode=pipelined wall_ms=[^ ]+ ms_per_frame=[^ ]+$' "$cap/track-$s.log")
  sed -E "s/^track_sequence frames=([0-9]+) mode=pipelined wall_ms=([^ ]+) ms_per_frame=([^ ]+)$/$s\t\1\t\2\t\3/" <<< "$line" >> "$cap/timing.tsv"
done
(cd "$out"; sha256sum scdr/*.scdr mot/*.txt offline/*.txt) > "$cap/outputs.sha256"

# 3. Gates.
for s in "${sdp[@]}"; do cmp "$out/offline/$s.txt" "$out/mot/$s.txt" || fail "$s track_offline and track_sequence MOT files differ"; done
gate 'CPU reference: PASS (track_offline == track_sequence on 7 sequences)'
[[ $(sha "$out/scdr/MOT17-10-SDP.scdr") == "$scdr10_sha" ]] || fail 'MOT17-10 SCDR differs from §16'
[[ $(sha "$out/mot/MOT17-10-SDP.txt") == "$mot10_sha" ]] || fail 'MOT17-10 MOT differs from §16'
recorded=0
while read -r s expected; do
  [[ $(sha "$out/mot/$s.txt") == "$expected" ]] || fail "$s MOT differs from §16"
  recorded=$((recorded + 1))
done < "$recorded_mot"
[[ $recorded == 6 ]] || fail "expected 6 recorded MOT hashes, found $recorded"
gate 'recorded hashes: PASS (MOT17-10 SCDR and MOT; 04/05/09/11/13 MOT)'
seq10=$sequences/MOT17-10-SDP
prefix=$out/prefix/MOT17-10-SDP
mkdir -p "$prefix" "$out/nsys"
sed 's/^seqLength=.*/seqLength=30/' "$seq10/seqinfo.ini" > "$prefix/seqinfo.ini"
ln -s "$seq10/img1" "$prefix/img1"
run nsys-profile "$nsys" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --output "$out/nsys/track" "$bin/track_sequence" "$engine" "$prefix" "$out/nsys/prefix.txt" --profile hr-1280x768
run nsys-export "$nsys" export --type sqlite --output "$out/nsys/track.sqlite" "$out/nsys/track.nsys-rep"
run stream-placement python3 -I scripts/check_stream_placement.py "$out/nsys/track.sqlite" "$cap/stream-placement.txt" ||
  fail 'stream placement'
gate 'stream placement: PASS'

binaries > "$cap/binaries-after.sha256"
cmp "$cap/acceptance-binaries.sha256" "$cap/binaries-after.sha256" || fail 'binaries changed during capture'
artifact_hashes > "$cap/artifacts-after.sha256"
cmp "$cap/artifacts.sha256" "$cap/artifacts-after.sha256" || fail 'inputs changed during capture'
check_source
printf 'PASS\n' > "$cap/gates.txt"

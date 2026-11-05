#!/usr/bin/env bash
# Fixed v1-motion/2 acceptance: docs/TRACKING.md §20.8 and §14.5.
# Invoke once from a clean commit under flock ~/.cache/saccade-v1/gpu.lock
# and timeout. All payloads and captures remain below the new build/ directory.
# bash -c snippets intentionally expand their own positional parameters.
# shellcheck disable=SC2016
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
usage() { echo 'usage: run_tracking_session_acceptance.sh build/NEW-OUTPUT' >&2; exit 2; }
[[ $# == 1 ]] || usage
out=$1
[[ $out =~ ^build/[A-Za-z0-9][A-Za-z0-9._-]*$ && ! -e $out && ! -L $out ]] || usage
[[ -z $(git status --porcelain) ]] || { echo 'clean committed source required' >&2; exit 2; }
root=$PWD
nvcc=/opt/cuda/bin/nvcc
sanitizer=/opt/cuda/bin/compute-sanitizer
sdk=/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk
t01_engine=/home/ray/developer/ai/saccade_v1/build/t01-f03-evidence/stage/model.engine
hr_engine=/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine
detection_refs=/home/ray/developer/ai/saccade_v1/build/preprocess-evidence/refs
sequences=/home/ray/developer/ai/saccade/datasets/MOT17/train
records=/home/ray/developer/ai/saccade_v1-worktrees/m1-data-v2
hr_records=$root/build/records/hr
m4=$root/../m4-formal/build-m4-evaluation/scdr
flow=$root/../gmc-flow/build/gmc-flow/flow
g1=$root/../gmc-flow/build/gmc-flow/arms/G1
pp_root=/home/ray/developer/ai/saccade/datasets/PersonPath22/mot_test_kf
pp_included=$root/../gmc-flow/build/gmc-flow2/pp22/included.txt
pp_g1=$root/../gmc-flow/build/gmc-flow2/pp22/arms/G1
strata=$root/evidence/gmc-flow/2026-11-04-round2/stratification.csv
base=$root/../g1-ofa-base
limit=1800
sdp=(MOT17-02-SDP MOT17-04-SDP MOT17-05-SDP MOT17-09-SDP MOT17-10-SDP MOT17-11-SDP MOT17-13-SDP)
domain=(MOT17-04-SDP MOT17-05-SDP MOT17-09-SDP MOT17-11-SDP MOT17-13-SDP)
fourk=(uid_vid_00179.mp4 uid_vid_00189.mp4 uid_vid_00218.mp4 uid_vid_00226.mp4 uid_vid_00230.mp4)
mapfile -t pp < "$pp_included"
v1_kernels='_tracking_session_cu_|_tracking_cu_|_flow_motion_cu_|void cub::'
mkdir -p build
mkdir "$out"
out=$(cd "$out" && pwd)
cap=$out/capture
mkdir "$cap"
last_command=initialization
step=1
baseline_source_sha=
baseline_revision=
sha() { sha256sum < "$1" | cut -d' ' -f1; }
source_hashes() {
  git ls-files -c -o --exclude-standard -z -- CMakeLists.txt cmake include src cuda tests tools fixtures third_party scripts training docs/TRACKING.md |
    sort -zu | while IFS= read -r -d '' path; do sha256sum -- "$path"; done
}
source_hashes > "$cap/source-files.sha256"
revision=$(git rev-parse HEAD)
source_sha=$(sha "$cap/source-files.sha256")
check_source() {
  [[ $(git rev-parse HEAD) == "$revision" && -z $(git status --porcelain) && $(source_hashes | sha256sum | cut -d' ' -f1) == "$source_sha" ]] ||
    { echo 'source changed during capture' >&2; return 2; }
  if [[ -n $baseline_source_sha ]]; then
    [[ $(git -C "$base" rev-parse HEAD) == "$baseline_revision" && -z $(git -C "$base" status --porcelain) &&
       $(cd "$base" && source_hashes | sha256sum | cut -d' ' -f1) == "$baseline_source_sha" ]] ||
      { echo 'baseline source changed during capture' >&2; return 2; }
  fi
}
artifact_hashes() {
  sha256sum "$t01_engine" "$hr_engine" "$pp_included" "$strata"
  for dir in "$detection_refs" "$records" "$hr_records" "$m4" "$flow" "$g1" "$pp_g1"; do
    find -L "$dir" -maxdepth 1 -type f -print0 | sort -z | xargs -0 -r sha256sum
  done
  for s in "${sdp[@]}"; do
    printf '%s  %s\n' "$(cd "$sequences/$s" && find -L seqinfo.ini img1 gt -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)" "$sequences/$s"
  done
  for s in "${pp[@]}" "${fourk[@]}"; do
    printf '%s  %s\n' "$(cd "$pp_root/$s" && find -L seqinfo.ini img1 gt -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)" "$pp_root/$s"
  done
}
binaries() {
  for dir in "$out/cpu" "$out/cuda"; do
    [[ ! -d $dir ]] || find "$dir" -maxdepth 1 -type f \( -perm -u+x -o -name '*.a' \) -print0 | sort -z | xargs -0 -r sha256sum
  done
}
kill_targets() { pkill -KILL -f "^$out/cuda/" || true; }
run() {
  local name=$1 status=0
  shift
  check_source || return $?
  last_command=$name
  printf '%q ' "$@" > "$cap/$name.command"
  printf '\n' >> "$cap/$name.command"
  date -Is > "$cap/$name.started"
  printf '%s: started\n' "$name"
  /usr/bin/time -v -o "$cap/$name.time" timeout -k 10 "$limit" "$@" > "$cap/$name.log" 2>&1 || status=$?
  # compute-sanitizer can outlive its own timeout; always reap its target too.
  kill_targets
  printf '%s\n' "$status" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$status"
  check_source || return $?
  return "$status"
}
finish() {
  local status=$?
  trap - EXIT TERM INT
  kill_targets
  printf 'exit=%s\nstep=%s\nlast_command=%s\nfinished=%s\n' "$status" "$step" "$last_command" "$(date -Is)" > "$cap/capture-status.txt"
  for dir in cpu cuda; do
    if [[ -f $out/$dir/CMakeCache.txt ]]; then cp "$out/$dir/CMakeCache.txt" "$cap/$dir-CMakeCache.txt"; fi
    if [[ -f $out/$dir/Testing/Temporary/LastTest.log ]]; then cp "$out/$dir/Testing/Temporary/LastTest.log" "$cap/$dir-LastTest.log"; fi
  done
  binaries > "$cap/binaries-after.sha256"
  if [[ -f $cap/artifacts.sha256 ]]; then artifact_hashes > "$cap/artifacts-after.sha256"; fi
  exit "$status"
}
trap finish EXIT
trap 'exit 124' TERM INT
pass() { printf '%s\tPASS\n' "$step" >> "$cap/steps.tsv"; }
printf 'step\tresult\n' > "$cap/steps.tsv"
printf 'revision=%s\nsource_digest=%s\n' "$revision" "$source_sha" > "$cap/source.txt"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status.txt"

# 1. Exact inputs, environment and identity. Never reuse a sibling CMake cache.
run t01-engine bash -c '[[ $(sha256sum < "$1" | cut -d" " -f1) == c6cff1936194e679ce11c3736cab552e4a73a2b1a9ae5508bc3e5d1b3fbab9c5 ]]' _ "$t01_engine"
run hr-engine bash -c '[[ $(sha256sum < "$1" | cut -d" " -f1) == 8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876 ]]' _ "$hr_engine"
run frozen-references bash -c 'cd "$1"; sha256sum -c "$2"' _ "$detection_refs" "$root/evidence/preprocess/2026-10-31/capture/reference-hashes.log"
printf '%s\n' \
 'a83f6045e5c470b35a6c50672f92e082b78d55752fe51cc20eb1a2738be05b9d  third_party/nvof/nvOpticalFlowCommon.h' \
 '07defc79637fb9893f2a06204972eaa6bcf9e5bea400c88440439ebaaa39f115  third_party/nvof/nvOpticalFlowCuda.h' > "$cap/headers.sha256"
run nvof-headers sha256sum -c "$cap/headers.sha256"
run nvof-library python3 -I - <<'PY'
import ctypes
from pathlib import Path
ctypes.CDLL('libnvidia-opticalflow.so.1')
paths = {Path(line.split()[-1]).resolve() for line in Path('/proc/self/maps').read_text().splitlines()
         if 'libnvidia-opticalflow.so' in line}
expected = Path('/usr/lib/wsl/lib/libnvidia-opticalflow.so.1')
print('NVOF loaded paths:', ', '.join(map(str, sorted(paths))))
if paths != {expected}:
    raise SystemExit('unexpected NVOF library path')
PY
run environment bash -c 'date -Is; uname -srmo; cmake --version; c++ --version; "$1" --version; "$2" --version; nsys --version; nvidia-smi --query-gpu=name,uuid,driver_version,memory.free --format=csv' _ "$nvcc" "$sanitizer"
artifact_hashes > "$cap/artifacts.sha256"
pass

# 2. New CPU/CUDA builds, complete CTest (including all seven packet/seed
# comparisons, flow kernel parity, A1-A4 and lifecycle), and A4 SCFL replay.
step=2
run cpu-configure cmake -S . -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release
run cpu-build cmake --build "$out/cpu" --parallel 8
run cpu-ctest ctest --test-dir "$out/cpu" --output-on-failure --stop-on-failure --parallel 1
run configure cmake -S . -B "$out/cuda" -DCMAKE_BUILD_TYPE=Release \
 -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON -DSACCADE_ENABLE_POSTPROCESS=ON \
 -DSACCADE_ENABLE_MODEL_PRODUCER=ON -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_ENABLE_MODEL_TRT=ON \
 -DSACCADE_ENABLE_DETECTION_SESSION=ON -DSACCADE_ENABLE_FLOW=ON "-DSACCADE_TENSORRT_ROOT=$sdk" \
 "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 "-DSACCADE_DETECTION_ENGINE=$t01_engine" "-DSACCADE_DETECTION_REFERENCES=$detection_refs" \
 "-DSACCADE_TRACKING_ENGINE=$t01_engine" "-DSACCADE_TRACKING_SEQUENCES=$sequences" "-DSACCADE_TRACKING_RECORDS=$records"
run build cmake --build "$out/cuda" --parallel 8
binaries > "$cap/binaries.sha256"
run ctest-list ctest --test-dir "$out/cuda" -N
run ctest ctest --test-dir "$out/cuda" --output-on-failure --stop-on-failure --parallel 1
cp "$out/cuda/Testing/Temporary/LastTest.log" "$cap/LastTest.log"
for s in 02 04 05 09 10 11 13; do
  rg -q "^MOT17-$s-SDP frames=[0-9]+ CPU/ordinary serial/graph pipelined packet bytes identical; graph ring controls=[0-9]+ frames$" "$cap/LastTest.log"
done
rg -Fq 'NVOF library=/usr/lib/wsl/lib/libnvidia-opticalflow.so.1 API=2.0' "$cap/LastTest.log"
run a4-replay "$out/cuda/flow_sequence" "$sequences/MOT17-10-SDP" "$out/MOT17-10-SDP.scfl"
run a4-compare cmp "$out/MOT17-10-SDP.scfl" "$flow/MOT17-10-SDP.scfl"
sha256sum "$out/MOT17-10-SDP.scfl" > "$cap/a4-scfl.sha256"
cp "$out/cuda/ofa-audit-raw/trials.csv" "$cap/ofa-trials.csv"
find "$out/cuda/ofa-audit-raw" -type f -print0 | sort -z | xargs -0 sha256sum > "$cap/ofa-artifacts.sha256"
pass

# 3. All seven HR paths. Each comparison is an immediate stop gate.
step=3
mkdir "$out/hr" "$out/hr/mot" "$out/hr/serial" "$out/hr/offline"
for s in "${sdp[@]}"; do
  run "hr-detect-$s" "$out/cuda/detect_sequence" "$hr_engine" "$sequences/$s" "$out/hr/$s.scdr" --profile hr-1280x768
  run "hr-scdr-compare-$s" cmp "$out/hr/$s.scdr" "$m4/$s.scdr"
  run "hr-records-$s" "$out/cuda/gmc_records" "$m4/$s.scdr" "$flow/$s.scfl" "$out/hr/$s.scmo"
  run "hr-records-compare-$s" cmp "$out/hr/$s.scmo" "$hr_records/$s.scmo"
  run "hr-pipelined-$s" "$out/cuda/track_sequence" "$hr_engine" "$sequences/$s" "$out/hr/mot/$s.txt" --profile hr-1280x768 --motion "$out/hr/$s-live.scmo" --packets "$out/hr/$s-pipelined.bin"
  run "hr-motion-compare-$s" cmp "$out/hr/$s-live.scmo" "$out/hr/$s.scmo"
  run "hr-offline-$s" "$out/cuda/track_offline" "$m4/$s.scdr" "$out/hr/$s.scmo" "$out/hr/offline/$s.txt"
  run "hr-offline-compare-$s" cmp "$out/hr/offline/$s.txt" "$out/hr/mot/$s.txt"
  run "hr-g1-compare-$s" cmp "$out/hr/mot/$s.txt" "$g1/$s.txt"
  run "hr-serial-$s" "$out/cuda/track_sequence" "$hr_engine" "$sequences/$s" "$out/hr/serial/$s.txt" --profile hr-1280x768 --serial --motion "$out/hr/$s-serial.scmo" --packets "$out/hr/$s-serial.bin"
  run "hr-serial-motion-compare-$s" cmp "$out/hr/$s-serial.scmo" "$out/hr/$s.scmo"
  run "hr-serial-mot-compare-$s" cmp "$out/hr/serial/$s.txt" "$out/hr/mot/$s.txt"
  run "hr-packets-compare-$s" cmp "$out/hr/$s-pipelined.bin" "$out/hr/$s-serial.bin"
done
find "$out/hr" -type f -print0 | sort -z | xargs -0 sha256sum > "$cap/hr-outputs.sha256"
pass

# 4. Fixed 69-sequence population and the five original 4K exclusions.
step=4
mkdir "$out/pp22" "$out/pp22-4k"
for s in "${pp[@]}"; do
  run "pp22-$s" "$out/cuda/track_sequence" "$hr_engine" "$pp_root/$s" "$out/pp22/$s.txt" --profile hr-1280x768
  run "pp22-compare-$s" cmp "$out/pp22/$s.txt" "$pp_g1/$s.txt"
done
for s in "${fourk[@]}"; do
  run "4k-$s" "$out/cuda/track_sequence" "$hr_engine" "$pp_root/$s" "$out/pp22-4k/$s.txt" --profile hr-1280x768
done
sha256sum "$out/pp22/"*.txt "$out/pp22-4k/"*.txt > "$cap/pp22-outputs.sha256"
pass

# 5. Same evaluation scripts and round-two strata; compare unrounded scores.
step=5
mkdir "$out/score" "$cap/scoring"
score() {
  local group=$1 method=$2 input=$3 reference=$4
  shift 4
  local dir=$out/score/$group
  mkdir "$dir"
  for s in "$@"; do cp "$input/$s.txt" "$dir/"; done
  run "score-$group" "scripts/eval_$method.sh" "$dir" "$@"
  local detail=$dir/trackeval/results/v1/pedestrian_detailed.csv
  run "score-compare-$group" python3 -I scripts/tracking_session_acceptance.py score "$group" "$detail" "$reference" "$cap/scores.csv"
  mkdir "$cap/scoring/$group"
  cp "$dir/trackeval/trackeval.log" "$dir/trackeval/results/v1/pedestrian_summary.txt" "$detail" "$cap/scoring/$group/"
}
ref1=$root/evidence/gmc-flow/2026-11-04/trackeval/G1
ref2=$root/evidence/gmc-flow/2026-11-04-round2/trackeval/G1
score val mot17 "$out/hr/mot" "$ref1/val/pedestrian_detailed.csv" MOT17-10-SDP
score in-domain mot17 "$out/hr/mot" "$ref1/in-domain/pedestrian_detailed.csv" "${domain[@]}"
score MOT17-02 mot17 "$out/hr/mot" "$ref1/MOT17-02-post-M4/pedestrian_detailed.csv" MOT17-02-SDP
for group in PP-moving PP-static; do
  mapfile -t group_seqs < <(python3 -I - "$strata" "$group" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as f:
    for row in csv.DictReader(f):
        if row['stratum'] == sys.argv[2] and not row['exclusion_reason']:
            print(row['sequence'])
PY
)
  score "$group" pp22 "$out/pp22" "$ref2/$group/pedestrian_detailed.csv" "${group_seqs[@]}"
done
score PP-all pp22 "$out/pp22" "$ref2/PP-all/pedestrian_detailed.csv" "${pp[@]}"
for s in "${fourk[@]}"; do score "$s" pp22 "$out/pp22-4k" - "$s"; done
pass

# 6. Four sanitizer suites and the original positive controls. The new flow
# suite replaces PCR motion kernels; OFA internals are not instrumented.
step=6
summary() {
  if [[ $1 == racecheck ]]; then echo '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
  else echo '========= ERROR SUMMARY: 0 errors'; fi
}
clean() { [[ $(rg -Fxc "$(summary "$1")" "$cap/$2.log") == 1 ]]; }
for tool in memcheck initcheck racecheck synccheck; do
  for test in tracking_packet_tests tracking_cuda_tests flow_motion_cuda_tests; do
    name=kernels-$tool-${test%_tests}
    args=(--tool "$tool" --error-exitcode 99)
    [[ $tool != initcheck ]] || args+=(--check-api-memory-access no)
    run "$name" "$sanitizer" "${args[@]}" "$out/cuda/$test"
    clean "$tool" "$name"
    if [[ $test == flow_motion_cuda_tests ]]; then rg -Fq 'flow kernels 3840x2160 s=2 gray/cells/mask/median/copies byte parity PASS' "$cap/$name.log"
    else rg -q 'checks passed' "$cap/$name.log"; fi
  done
  case $tool in
    memcheck) diagnostic='Invalid __global__ write';;
    initcheck) diagnostic='Uninitialized __global__ memory read';;
    racecheck) diagnostic='Error: Race reported between';;
    synccheck) diagnostic='Barrier error detected';;
  esac
  status=0
  run "control-$tool" "$sanitizer" --tool "$tool" --error-exitcode 99 "$out/cuda/postprocess_sanitizer_control" "$tool" || status=$?
  [[ $status == 99 ]] && rg -Fq "$diagnostic" "$cap/control-$tool.log"
done
seq10=$sequences/MOT17-10-SDP
prefix=$out/prefix/MOT17-10-SDP
mkdir -p "$prefix"
sed 's/^seqLength=.*/seqLength=30/' "$seq10/seqinfo.ini" > "$prefix/seqinfo.ini"
ln -s "$seq10/img1" "$prefix/img1"
run lifecycle-memcheck "$sanitizer" --tool memcheck --report-api-errors explicit --error-exitcode 99 "$out/cuda/tracking_session_tests" "$t01_engine" "$sequences" "$records" MOT17-10-SDP --lifecycle
clean memcheck lifecycle-memcheck
rg -q 'checks passed' "$cap/lifecycle-memcheck.log"
run hr-memcheck "$sanitizer" --tool memcheck --report-api-errors explicit --error-exitcode 99 "$out/cuda/track_sequence" "$hr_engine" "$prefix" "$out/hr/prefix.txt" --profile hr-1280x768
clean memcheck hr-memcheck
for tool in racecheck synccheck; do
  run "session-$tool" "$sanitizer" --tool "$tool" --error-exitcode 99 --kernel-name regex="$v1_kernels" "$out/cuda/tracking_session_tests" "$t01_engine" "$sequences" "$records" MOT17-10-SDP --limit 10
  clean "$tool" "session-$tool"
  rg -q 'checks passed' "$cap/session-$tool.log"
done
pass

# 7. Node/graph captures verify every v1 kernel's stream. The audited M
# ordering bounds OFA between input-copy completion and cell reduction. Require
# this entire device interval inside the same frame's F03 execution interval.
step=7
mkdir "$out/nsys"
run nsys-node nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --output "$out/nsys/track" "$out/cuda/track_sequence" "$hr_engine" "$prefix" "$out/nsys/node.txt" --profile hr-1280x768
run nsys-node-export nsys export --type sqlite --output "$out/nsys/track.sqlite" "$out/nsys/track.nsys-rep"
run nsys-graph nsys profile --trace=cuda,nvtx --cuda-graph-trace=graph --sample=none --cpuctxsw=none --output "$out/nsys/track-graph" "$out/cuda/track_sequence" "$hr_engine" "$prefix" "$out/nsys/graph.txt" --profile hr-1280x768
run nsys-graph-export nsys export --type sqlite --output "$out/nsys/track-graph.sqlite" "$out/nsys/track-graph.nsys-rep"
run nsys-output-compare cmp "$out/nsys/node.txt" "$out/nsys/graph.txt"
run stream-placement python3 -I scripts/check_stream_placement.py "$out/nsys/track.sqlite" "$out/nsys/track-graph.sqlite" "$cap/stream-placement.txt"
run ofa-overlap python3 -I scripts/tracking_session_acceptance.py overlap "$out/nsys/track.sqlite" "$cap/ofa-overlap.csv"
sha256sum "$out/nsys/"*.nsys-rep "$out/nsys/"*.sqlite > "$cap/nsys.sha256"
pass

# 8. Same-run main 23ade53 baseline, detached, with its own build/ output.
# Use interleaved before/after pairs; all measurements are unprofiled and HR.
step=8
run baseline-worktree git worktree add --detach "$base" 23ade53
base=$(cd "$base" && pwd)
base_build=$base/build/v1-motion-2-acceptance-2
baseline_revision=$(git -C "$base" rev-parse HEAD)
[[ $baseline_revision == $(git rev-parse 23ade53) ]]
(cd "$base" && source_hashes) > "$cap/baseline-source-files.sha256"
baseline_source_sha=$(sha "$cap/baseline-source-files.sha256")
run baseline-configure cmake -S "$base" -B "$base_build" -DCMAKE_BUILD_TYPE=Release \
 -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON -DSACCADE_ENABLE_POSTPROCESS=ON \
 -DSACCADE_ENABLE_MODEL_PRODUCER=ON -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_ENABLE_MODEL_TRT=ON \
 -DSACCADE_ENABLE_DETECTION_SESSION=ON "-DSACCADE_TENSORRT_ROOT=$sdk" \
 "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 "-DSACCADE_DETECTION_ENGINE=$t01_engine" "-DSACCADE_DETECTION_REFERENCES=$detection_refs" \
 "-DSACCADE_TRACKING_ENGINE=$t01_engine" "-DSACCADE_TRACKING_SEQUENCES=$sequences"
run baseline-build cmake --build "$base_build" --target track_sequence --parallel 8
cp "$base_build/CMakeCache.txt" "$cap/baseline-CMakeCache.txt"
git -C "$base" rev-parse HEAD > "$cap/baseline-source.txt"
[[ $(git -C "$base" rev-parse HEAD) == $(git rev-parse 23ade53) && -z $(git -C "$base" status --porcelain) ]]
sha256sum "$base_build/track_sequence" > "$cap/baseline-binary.sha256"
mkdir "$out/timing"
for s in "${sdp[@]}"; do
  for repeat in 1 2 3; do
    run "timing-before-$s-$repeat" "$base_build/track_sequence" "$hr_engine" "$sequences/$s" "$out/timing/before-$s-$repeat.txt" --profile hr-1280x768
    run "timing-after-$s-$repeat" "$out/cuda/track_sequence" "$hr_engine" "$sequences/$s" "$out/timing/after-$s-$repeat.txt" --profile hr-1280x768
  done
  run "timing-serial-$s" "$out/cuda/track_sequence" "$hr_engine" "$sequences/$s" "$out/timing/serial-$s.txt" --profile hr-1280x768 --serial
done
run throughput python3 -I scripts/tracking_session_acceptance.py throughput "$cap"
sha256sum -c "$cap/baseline-binary.sha256"
[[ -z $(git -C "$base" status --porcelain) ]]
pass

# 9. Capture custody. The report and doc result line are written only after
# this clean-source capture exits; payload bytes are represented by SHA-256.
step=9
binaries > "$cap/binaries-verified.sha256"
cmp "$cap/binaries.sha256" "$cap/binaries-verified.sha256"
artifact_hashes > "$cap/artifacts-verified.sha256"
cmp "$cap/artifacts.sha256" "$cap/artifacts-verified.sha256"
find "$out/hr" "$out/pp22" "$out/pp22-4k" "$out/timing" -type f -print0 | sort -z | xargs -0 sha256sum > "$cap/outputs.sha256"
cp "$out/cuda/compile_commands.json" "$cap/compile_commands.json"
check_source
pass
printf 'PASS\n' > "$cap/acceptance.txt"

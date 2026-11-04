#!/usr/bin/env bash
# M3 formal acceptance capture (docs/TRACKING.md §14.5): clean source, fresh
# CPU and CUDA builds, the full CTest suites, the deployed HR FP16 path on
# MOT17-10 and the four compute-sanitizer tools. Run it under the GPU lock.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
usage() { echo 'usage: run_tracking_session_acceptance.sh build/NEW-OUTPUT [--allow-dirty]' >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || usage
out=$1
allow_dirty=false
if [[ $# == 2 ]]; then [[ $2 == --allow-dirty ]] || usage; allow_dirty=true; fi
[[ $out =~ ^build/[A-Za-z0-9][A-Za-z0-9._-]*$ && ! -e $out && ! -L $out ]] || usage
[[ $allow_dirty == true || -z $(git status --porcelain) ]] || { echo 'clean source required; --allow-dirty records development evidence' >&2; exit 2; }
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
arch=${SACCADE_CUDA_ARCHITECTURES:-120}
sdk=${SACCADE_TENSORRT_ROOT:-$HOME/.cache/saccade-native-trt/10.16.1.11/sdk}
t01_engine=${SACCADE_TRACKING_ENGINE:?T01 F03 engine that recorded the M1 records}
hr_engine=${SACCADE_HR_ENGINE:-$HOME/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine}
detection_refs=${SACCADE_DETECTION_REFERENCES:?detection session reference corpus}
sequences=${SACCADE_TRACKING_SEQUENCES:?MOT17 train directory}
records=${SACCADE_TRACKING_RECORDS:?M1 SCDR/SCMO records}
limit=${SACCADE_ACCEPTANCE_TIMEOUT:-1800}
# Fixed in §14.5 before the capture (values from §16).
hr_engine_sha=8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876
hr_scdr_sha=98e2554f1b1db179bfd756fb1dbd37ee694e912d6debb2c7a11c3a610a433a6f
hr_mot_sha=1fb713f30444d2c6e3e9ae30d34621f3155b0967b3a52a9244f107e28547fba7
sdp=(MOT17-02-SDP MOT17-04-SDP MOT17-05-SDP MOT17-09-SDP MOT17-10-SDP MOT17-11-SDP MOT17-13-SDP)
v1_kernels='_tracking_session_cu_|_tracking_cu_|_motion_cu_'
[[ $arch =~ ^[0-9]+$ && $limit =~ ^[1-9][0-9]*$ ]] || usage
[[ -f $t01_engine && -f $hr_engine && -f $detection_refs/cases.tsv && -d $sdk && -d $sequences && -d $records ]] ||
  { echo 'missing engine, reference corpus, SDK, sequences or records' >&2; exit 2; }
[[ $(sha256sum < "$hr_engine" | cut -d' ' -f1) == "$hr_engine_sha" ]] || { echo 'HR engine is not the deployed FP16 engine' >&2; exit 2; }
mkdir -p build && mkdir "$out"
out=$(cd "$out" && pwd)
cap=$out/capture
mkdir "$cap"
source_hashes() {
  git ls-files -c -o --exclude-standard -z -- CMakeLists.txt cmake include src cuda tests tools fixtures \
    docs/TRACKING.md scripts/run_tracking_session_acceptance.sh |
    sort -zu | while IFS= read -r -d '' path; do sha256sum -- "$path"; done
}
source_hashes > "$cap/source-files.sha256"
revision=$(git rev-parse HEAD)
source_sha=$(sha256sum < "$cap/source-files.sha256" | cut -d' ' -f1)
check_source() {
  [[ $(git rev-parse HEAD) == "$revision" && $(source_hashes | sha256sum | cut -d' ' -f1) == "$source_sha" ]] || { echo 'build source changed during capture' >&2; return 2; }
}
# Engines, reference corpus and records per file; each MOT17 SDP sequence as
# one digest over its seqinfo.ini and image hashes.
artifact_hashes() {
  sha256sum "$t01_engine" "$hr_engine"
  (cd "$detection_refs"; find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum)
  (cd "$records"; find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum)
  for s in "${sdp[@]}"; do
    printf '%s  %s\n' "$(cd "$sequences/$s" && find seqinfo.ini img1 -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)" "$s"
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
  # Stopping compute-sanitizer does not stop its instrumented target.
  pkill -KILL -f "^$out/cuda/" || true
  printf '%s\n' "$status" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$status"
  check_source
  return "$status"
}
finish() {
  local status=$?
  trap - EXIT
  printf 'exit=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$cap/capture-status.txt"
  exit "$status"
}
trap finish EXIT
printf 'revision=%s\nsource_digest=%s\nallow_dirty=%s\n' "$revision" "$source_sha" "$allow_dirty" > "$cap/source.txt"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status.txt"
git diff --binary HEAD > "$cap/source.diff"
run environment bash -c 'date -Is; uname -srmo; cmake --version; c++ --version; "$1" --version; "$2" --version; nvidia-smi --query-gpu=name,uuid,driver_version,memory.free --format=csv' _ "$nvcc" "$sanitizer"

# 1. Fresh builds and the full CTest suites.
run cpu-configure cmake -S . -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release
run cpu-build cmake --build "$out/cpu" --parallel 8
run cpu-ctest ctest --test-dir "$out/cpu" --output-on-failure
cp "$out/cpu/Testing/Temporary/LastTest.log" "$cap/cpu-LastTest.log"
run configure cmake -S . -B "$out/cuda" -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON -DSACCADE_ENABLE_POSTPROCESS=ON \
  -DSACCADE_ENABLE_MODEL_PRODUCER=ON -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_ENABLE_MODEL_TRT=ON \
  -DSACCADE_ENABLE_DETECTION_SESSION=ON "-DSACCADE_TENSORRT_ROOT=$sdk" \
  "-DCMAKE_CUDA_COMPILER=$nvcc" "-DCMAKE_CUDA_ARCHITECTURES=$arch" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  "-DSACCADE_DETECTION_ENGINE=$t01_engine" "-DSACCADE_DETECTION_REFERENCES=$detection_refs" \
  "-DSACCADE_TRACKING_ENGINE=$t01_engine" "-DSACCADE_TRACKING_SEQUENCES=$sequences" "-DSACCADE_TRACKING_RECORDS=$records"
run build cmake --build "$out/cuda" --parallel 8
binaries() { find "$out/cuda" -maxdepth 1 -type f \( -perm -u+x -o -name '*.a' \) -print0 | sort -z | xargs -0 sha256sum; }
binaries > "$cap/binaries.sha256"
run ctest-list ctest --test-dir "$out/cuda" -N
run ctest ctest --test-dir "$out/cuda" --output-on-failure
cp "$out/cuda/Testing/Temporary/LastTest.log" "$cap/LastTest.log"
for s in 02 04 05 09 10 11 13; do
  rg -q "^MOT17-$s-SDP frames=[0-9]+ CPU/ordinary serial/graph pipelined packet bytes identical; graph ring controls=[0-9]+ frames$" "$cap/LastTest.log"
done

# 2. Deployed path: HR FP16 on MOT17-10, CPU reference vs serial vs pipelined.
hr=$out/hr
mkdir "$hr"
seq10=$sequences/MOT17-10-SDP
run hr-detect "$out/cuda/detect_sequence" "$hr_engine" "$seq10" "$hr/MOT17-10-SDP.scdr" --profile hr-1280x768
run hr-motion "$out/cuda/motion_sequence" "$seq10" "$hr/MOT17-10-SDP.scmo"
run hr-offline "$out/cuda/track_offline" "$hr/MOT17-10-SDP.scdr" "$hr/MOT17-10-SDP.scmo" "$hr/offline.txt"
run hr-pipelined "$out/cuda/track_sequence" "$hr_engine" "$seq10" "$hr/pipelined.txt" --packets "$hr/pipelined.bin" --profile hr-1280x768
run hr-serial "$out/cuda/track_sequence" "$hr_engine" "$seq10" "$hr/serial.txt" --serial --packets "$hr/serial.bin" --profile hr-1280x768
(cd "$hr"; sha256sum MOT17-10-SDP.scdr MOT17-10-SDP.scmo offline.txt pipelined.txt serial.txt pipelined.bin serial.bin) > "$cap/hr-outputs.sha256"
[[ $(sha256sum < "$hr/MOT17-10-SDP.scdr" | cut -d' ' -f1) == "$hr_scdr_sha" ]]
cmp "$hr/MOT17-10-SDP.scmo" "$records/MOT17-10-SDP.scmo"
cmp "$hr/offline.txt" "$hr/pipelined.txt"
cmp "$hr/serial.txt" "$hr/pipelined.txt"
cmp "$hr/serial.bin" "$hr/pipelined.bin"
[[ $(sha256sum < "$hr/pipelined.txt" | cut -d' ' -f1) == "$hr_mot_sha" ]]
printf 'PASS\n' > "$cap/hr-path.txt"

# 3. Sanitizers.
summary() {
  if [[ $1 == racecheck ]]; then echo '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
  else echo '========= ERROR SUMMARY: 0 errors'; fi
}
clean() { [[ $(rg -Fxc "$(summary "$1")" "$cap/$2.log") == 1 ]] && rg -q 'checks passed' "$cap/$2.log"; }
# 3a. Kernel suites. Initcheck leaves out host API copy checks (the harnesses
# download whole fixed-capacity structs whose tails past count are undefined
# by contract) and runs the motion suite without its two nvJPEG frames
# (PREPROCESS_USAGE_POLICY §3); kernel reads stay fully checked.
for tool in memcheck initcheck racecheck synccheck; do
  for test in tracking_packet_tests tracking_cuda_tests motion_cuda_tests; do
    name=kernels-$tool-${test%_tests}
    args=(--tool "$tool" --error-exitcode 99)
    [[ $tool == initcheck ]] && args+=(--check-api-memory-access no)
    target=("$out/cuda/$test")
    [[ $tool == initcheck && $test == motion_cuda_tests ]] && target+=(--synthetic)
    run "$name" timeout -k 10 "$limit" "$sanitizer" "${args[@]}" "${target[@]}"
    clean "$tool" "$name"
  done
  case $tool in
    memcheck) diagnostic='Invalid __global__ write';;
    initcheck) diagnostic='Uninitialized __global__ memory read';;
    racecheck) diagnostic='Error: Race reported between';;
    synccheck) diagnostic='Barrier error detected';;
  esac
  status=0
  run "control-$tool" timeout -k 10 "$limit" "$sanitizer" --tool "$tool" --error-exitcode 99 \
    "$out/cuda/postprocess_sanitizer_control" "$tool" || status=$?
  [[ $status == 99 ]] && rg -Fq "$diagnostic" "$cap/control-$tool.log"
done
# 3b. Full sessions under memcheck: every kernel, including nvJPEG, TensorRT and cuFFT.
# Explicit API reporting, as in S04 (evidence/s04/2026-10-30 REPORT, third
# capture): the default extended mode counts the driver note "Selective device
# code recompilation in progress", logged when TensorRT loads its kernel
# modules under the tool, as an error. Memory errors are reported either way.
prefix=$out/prefix/MOT17-10-SDP
mkdir -p "$prefix"
sed 's/^seqLength=.*/seqLength=30/' "$seq10/seqinfo.ini" > "$prefix/seqinfo.ini"
ln -s "$seq10/img1" "$prefix/img1"
run session-memcheck timeout -k 10 "$limit" "$sanitizer" --tool memcheck --report-api-errors explicit --error-exitcode 99 \
  "$out/cuda/tracking_session_tests" "$t01_engine" "$sequences" "$records" MOT17-10-SDP --limit 30
clean memcheck session-memcheck
run lifecycle-memcheck timeout -k 10 "$limit" "$sanitizer" --tool memcheck --report-api-errors explicit --error-exitcode 99 \
  "$out/cuda/tracking_session_tests" "$t01_engine" "$sequences" "$records" MOT17-10-SDP --lifecycle
clean memcheck lifecycle-memcheck
run hr-memcheck timeout -k 10 "$limit" "$sanitizer" --tool memcheck --report-api-errors explicit --error-exitcode 99 \
  "$out/cuda/track_sequence" "$hr_engine" "$prefix" "$hr/prefix.txt" --profile hr-1280x768
[[ $(rg -Fxc '========= ERROR SUMMARY: 0 errors' "$cap/hr-memcheck.log") == 1 ]]
# 3c. Full session, racecheck and synccheck on the tracking, motion and
# session-control kernels (TensorRT, nvJPEG and cuFFT kernels not instrumented).
for tool in racecheck synccheck; do
  run "session-$tool" timeout -k 10 "$limit" "$sanitizer" --tool "$tool" --error-exitcode 99 --kernel-name regex="$v1_kernels" \
    "$out/cuda/tracking_session_tests" "$t01_engine" "$sequences" "$records" MOT17-10-SDP --limit 10
  clean "$tool" "session-$tool"
done

binaries > "$cap/binaries-after.sha256"
cmp "$cap/binaries.sha256" "$cap/binaries-after.sha256"
artifact_hashes > "$cap/artifacts-after.sha256"
cmp "$cap/artifacts.sha256" "$cap/artifacts-after.sha256"
cp "$out/cuda/CMakeCache.txt" "$cap/CMakeCache.txt"
cp "$out/cuda/compile_commands.json" "$cap/compile_commands.json"
check_source
printf 'PASS\n' > "$cap/acceptance.txt"

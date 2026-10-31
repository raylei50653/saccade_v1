#!/usr/bin/env bash
# Native detection session capture. An uncommitted implementation is explicitly
# development evidence; the independent a9c3d9b F03 acceptance stays separate.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
usage() { echo 'usage: run_detection_session_acceptance.sh build-NEW-OUTPUT [--allow-dirty]' >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || usage
out=$1
allow_dirty=false
if [[ $# == 2 ]]; then [[ $2 == --allow-dirty ]] || usage; allow_dirty=true; fi
[[ $out == build-* && $out != */* && ! -e $out && ! -L $out ]] || usage
[[ $allow_dirty == true || -z $(git status --porcelain) ]] || { echo 'clean source required; --allow-dirty records development evidence' >&2; exit 2; }
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
arch=${SACCADE_CUDA_ARCHITECTURES:-120}
sdk=${SACCADE_TENSORRT_ROOT:-$HOME/.cache/saccade-native-trt/10.16.1.11/sdk}
engine=${SACCADE_DETECTION_ENGINE:-$PWD/build-t01-f03-evidence/stage/model.engine}
refs=${SACCADE_DETECTION_REFERENCES:-$PWD/build-preprocess-evidence/refs}
limit=${SACCADE_DETECTION_TIMEOUT:-300}
run_initcheck=${SACCADE_DETECTION_INITCHECK_DIAGNOSTIC:-0}
[[ $arch =~ ^[0-9]+$ && $limit =~ ^[1-9][0-9]*$ ]] || usage
[[ $run_initcheck == 0 || $run_initcheck == 1 ]] || usage
[[ -f $engine && -f $refs/cases.tsv && -d $sdk ]] || { echo 'missing engine/reference corpus/SDK' >&2; exit 2; }
mkdir "$out"
out=$(cd "$out" && pwd)
cap=$out/capture
mkdir "$cap"
source_hashes() {
  git ls-files -c -o --exclude-standard -z -- CMakeLists.txt cmake include src cuda tests tools fixtures \
    docs/DETECTION_SESSION.md scripts/run_detection_session_acceptance.sh |
    sort -zu | while IFS= read -r -d '' path; do sha256sum -- "$path"; done
}
source_hashes > "$cap/source-files.sha256"
revision=$(git rev-parse HEAD)
source_sha=$(sha256sum < "$cap/source-files.sha256" | cut -d' ' -f1)
check_source() {
  [[ $(git rev-parse HEAD) == "$revision" && $(source_hashes | sha256sum | cut -d' ' -f1) == "$source_sha" ]] || { echo 'build source changed during capture' >&2; return 2; }
}
artifact_hashes() {
  sha256sum "$engine"
  (cd "$refs"; find . -maxdepth 1 -type f -print0 | sort -z | xargs -0 sha256sum)
}
artifact_hashes > "$cap/artifacts.sha256"
check_artifacts() { artifact_hashes > "$cap/artifacts-after.sha256"; cmp "$cap/artifacts.sha256" "$cap/artifacts-after.sha256"; }
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
tar -czf "$cap/source-build-inputs.tar.gz" -T <(cut -c67- "$cap/source-files.sha256")
run environment bash -c 'date -Is; uname -srmo; cmake --version; c++ --version; "$1" --version; "$2" --version; nvidia-smi --query-gpu=name,uuid,driver_version,memory.free --format=csv' _ "$nvcc" "$sanitizer"
run cpu-configure cmake -S . -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release
run cpu-build cmake --build "$out/cpu" --parallel 3
run cpu-ctest ctest --test-dir "$out/cpu" --output-on-failure
run configure cmake -S . -B "$out/cuda" -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON -DSACCADE_ENABLE_POSTPROCESS=ON \
  -DSACCADE_ENABLE_MODEL_PRODUCER=ON -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_ENABLE_MODEL_TRT=ON \
  -DSACCADE_ENABLE_DETECTION_SESSION=ON "-DSACCADE_TENSORRT_ROOT=$sdk" \
  "-DCMAKE_CUDA_COMPILER=$nvcc" "-DCMAKE_CUDA_ARCHITECTURES=$arch" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  "-DSACCADE_DETECTION_ENGINE=$engine" "-DSACCADE_DETECTION_REFERENCES=$refs"
run build cmake --build "$out/cuda" --parallel 3 --target detection_session_tests detection_session_cuda_tests detection_session_model preprocess_tests resize_cuda_tests \
  model_producer_cuda_tests postprocess_cuda_tests postprocess_tests postprocess_sanitizer_control
binaries() {
  sha256sum "$out/cuda/detection_session_tests" "$out/cuda/detection_session_cuda_tests" "$out/cuda/libsaccade_detection_session.a" \
    "$out/cuda/libsaccade_preprocess.a" "$out/cuda/libsaccade_model_producer_cuda.a" \
    "$out/cuda/libsaccade_postprocess_cuda.a" "$out/cuda/postprocess_sanitizer_control"
}
binaries > "$cap/binaries.sha256"
run ctest ctest --test-dir "$out/cuda" -R '^(detection_session($|_)|native_preprocess$|resize_cuda$|model_producer_cuda|postprocess_)' --output-on-failure
cp "$out/cuda/Testing/Temporary/LastTest.log" "$cap/LastTest.log"
run session timeout "$limit" "$out/cuda/detection_session_tests" "$engine" "$refs" "$cap/session"
rg -q '^detection_session=PASS ' "$cap/session.log"
run memcheck timeout "$limit" "$sanitizer" --tool memcheck --error-exitcode 99 \
  "$out/cuda/detection_session_tests" "$engine" "$refs" "$cap/memcheck" --sanitizer
[[ $(rg -Fxc '========= ERROR SUMMARY: 0 errors' "$cap/memcheck.log") == 1 ]]
rg -q '^detection_session=PASS ' "$cap/memcheck.log"
for tool in memcheck initcheck racecheck synccheck; do
  run "kernels-$tool" timeout "$limit" "$sanitizer" --tool "$tool" --error-exitcode 99 \
    "$out/cuda/detection_session_cuda_tests"
  if [[ $tool == racecheck ]]; then summary='========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
  else summary='========= ERROR SUMMARY: 0 errors'; fi
  [[ $(rg -Fxc "$summary" "$cap/kernels-$tool.log") == 1 ]]
  rg -q '^PASS ' "$cap/kernels-$tool.log"
  case $tool in
    memcheck) diagnostic='Invalid __global__ write';;
    initcheck) diagnostic='Uninitialized __global__ memory read';;
    racecheck) diagnostic='Error: Race reported between';;
    synccheck) diagnostic='Barrier error detected';;
  esac
  status=0
  run "control-$tool" timeout "$limit" "$sanitizer" --tool "$tool" --error-exitcode 99 \
    "$out/cuda/postprocess_sanitizer_control" "$tool" || status=$?
  [[ $status == 99 ]] && rg -Fq "$diagnostic" "$cap/control-$tool.log"
done
# Unmodified nvJPEG path: preserve diagnostics under the existing usage policy.
# Exit 99 alone cannot establish that any report is the previously known issue.
# Optional only: the retained full-path attempt suffered an instrumented launch
# timeout. Never suppress or reclassify an application/API failure as PASS.
if [[ $run_initcheck == 1 ]]; then
status=0
run initcheck-diagnostic timeout "$limit" "$sanitizer" --tool initcheck --error-exitcode 99 \
  "$out/cuda/detection_session_tests" "$engine" "$refs" "$cap/initcheck" --sanitizer || status=$?
rg -q '^detection_session=PASS ' "$cap/initcheck-diagnostic.log"
if [[ $status == 0 ]]; then
  [[ $(rg -Fxc '========= ERROR SUMMARY: 0 errors' "$cap/initcheck-diagnostic.log") == 1 ]]
  printf 'Initcheck: PASS in this bounded run\n' > "$cap/initcheck-classification.txt"
elif [[ $status == 99 ]] && rg -Fq 'Uninitialized __global__ memory read' "$cap/initcheck-diagnostic.log"; then
  printf 'Initcheck: FAIL diagnostic; requires report attribution; not a usage gate per PREPROCESS_USAGE_POLICY.md\n' > "$cap/initcheck-classification.txt"
else
  echo 'unexpected initcheck failure' >&2; exit 2
fi
else
  printf 'UNEXECUTED in this capture: full-path initcheck is a separate diagnostic; retained prior instrumented execution failure remains unresolved. Session-control initcheck executed above.\n' > "$cap/initcheck-classification.txt"
fi
binaries > "$cap/binaries-after.sha256"
cmp "$cap/binaries.sha256" "$cap/binaries-after.sha256"
check_artifacts
cp "$out/cuda/compile_commands.json" "$cap/compile_commands.json"
cp "$out/cuda/CMakeCache.txt" "$cap/CMakeCache.txt"
find "$out/cuda" -maxdepth 1 -type f \( -name '*.a' -o -name 'detection_session*' -o -name 'postprocess_sanitizer_control' \) \
  -printf '%f\n' | sort > "$cap/local-bundle-files.txt"
tar -czf "$out/actual-binaries.tar.gz" -C "$out/cuda" -T "$cap/local-bundle-files.txt"
sha256sum "$out/actual-binaries.tar.gz" > "$cap/local-artifacts.sha256"
check_source
printf 'PASS (bounded normal usage and memcheck; initcheck classified separately)\n' > "$cap/bounded-checks.txt"

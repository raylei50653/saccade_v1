#!/usr/bin/env bash
# Bounded F03 producer capture; dirty work is development evidence only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
usage() { echo 'usage: run_model_producer_acceptance.sh build-NEW-OUTPUT [--allow-dirty]' >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || usage
out=$1
allow_dirty=false
if [[ $# == 2 ]]; then [[ $2 == --allow-dirty ]] || usage; allow_dirty=true; fi
[[ $out == build-* && $out != */* && ! -e $out && ! -L $out ]] || usage
if [[ $allow_dirty == false && -n $(git status --porcelain) ]]; then
  echo 'clean source required; --allow-dirty records development evidence only' >&2; exit 2
fi
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
arch=${SACCADE_CUDA_ARCHITECTURES:-120}
sdk=${SACCADE_TENSORRT_ROOT:-$HOME/.cache/saccade-native-trt/10.16.1.11/sdk}
engine=${SACCADE_MODEL_PRODUCER_ENGINE:-$PWD/build-t01-f03-evidence/stage/model.engine}
inputs=${SACCADE_MODEL_PRODUCER_INPUTS:-$PWD/build-t01-f03-evidence/t01root/export/f03-inputs.f32}
limit=${SACCADE_MODEL_PRODUCER_SANITIZER_TIMEOUT:-240}
[[ $arch =~ ^[0-9]+$ && $limit =~ ^[1-9][0-9]*$ ]] || usage
[[ -f $engine && -f $inputs && -d $sdk ]] || { echo 'missing pinned F03 engine/inputs or SDK' >&2; exit 2; }
mkdir "$out"
out=$(cd "$out" && pwd)
cap=$out/capture
mkdir "$cap"
source_hashes() {
  git ls-files -c -o --exclude-standard -z -- CMakeLists.txt cmake include src cuda tests tools fixtures \
    docs/MODEL_PRODUCER.md scripts/run_model_producer_acceptance.sh |
    sort -zu | while IFS= read -r -d '' path; do sha256sum -- "$path"; done
}
source_hashes > "$cap/source-files.sha256"
revision=$(git rev-parse HEAD)
source_sha=$(sha256sum < "$cap/source-files.sha256" | cut -d' ' -f1)
check_source() {
  [[ $(git rev-parse HEAD) == "$revision" && $(source_hashes | sha256sum | cut -d' ' -f1) == "$source_sha" ]] || {
    echo 'source changed during capture' >&2; return 2;
  }
}
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
run artifacts sha256sum "$engine" "$inputs"
run cpu-configure cmake -S . -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release
run cpu-build cmake --build "$out/cpu" --parallel 3
run cpu-ctest ctest --test-dir "$out/cpu" --output-on-failure
run configure cmake -S . -B "$out/cuda" -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_POSTPROCESS=ON -DSACCADE_ENABLE_MODEL_PRODUCER=ON \
  -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_ENABLE_MODEL_TRT=ON "-DSACCADE_TENSORRT_ROOT=$sdk" \
  "-DCMAKE_CUDA_COMPILER=$nvcc" "-DCMAKE_CUDA_ARCHITECTURES=$arch" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  "-DSACCADE_MODEL_PRODUCER_ENGINE=$engine" "-DSACCADE_MODEL_PRODUCER_INPUTS=$inputs"
run build cmake --build "$out/cuda" --parallel 3 --target \
  model_producer_cuda_tests model_producer_trt_tests postprocess_cuda_tests postprocess_tests postprocess_sanitizer_control
run binaries sha256sum "$out/cuda/model_producer_cuda_tests" "$out/cuda/model_producer_trt_tests" \
  "$out/cuda/libsaccade_model_producer_cuda.a" "$out/cuda/libsaccade_postprocess_cuda.a" "$out/cuda/postprocess_sanitizer_control"
run ctest ctest --test-dir "$out/cuda" -R '^(model_producer_|postprocess_)' --output-on-failure
cp "$out/cuda/Testing/Temporary/LastTest.log" "$cap/LastTest.log"
for tool in memcheck initcheck racecheck synccheck; do
  run "sanitizer-$tool" timeout "$limit" "$sanitizer" --tool "$tool" --error-exitcode 99 \
    "$out/cuda/model_producer_cuda_tests" --no-held-stream --no-api-trace
  if [[ $tool == racecheck ]]; then summary='========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
  else summary='========= ERROR SUMMARY: 0 errors'; fi
  [[ $(rg -Fxc -- "$summary" "$cap/sanitizer-$tool.log") == 1 ]]
  rg -q '^PASS checks=[0-9]+$' "$cap/sanitizer-$tool.log"
  case $tool in
    memcheck) diagnostic='Invalid __global__ write';;
    initcheck) diagnostic='Uninitialized __global__ memory read';;
    racecheck) diagnostic='Error: Race reported between';;
    synccheck) diagnostic='Barrier error detected';;
  esac
  status=0
  run "control-$tool" timeout "$limit" "$sanitizer" --tool "$tool" --error-exitcode 99 \
    "$out/cuda/postprocess_sanitizer_control" "$tool" || status=$?
  [[ $status == 99 ]] && rg -Fq -- "$diagnostic" "$cap/control-$tool.log"
done
# Preserve the historical head/F03 racecheck/synccheck boundary; only the new
# adapter/F04 standalone test above gets those tools. Bound live work to input 0.
for tool in memcheck initcheck; do
  run "live-$tool" timeout "$limit" "$sanitizer" --tool "$tool" --error-exitcode 99 \
    "$out/cuda/model_producer_trt_tests" "$engine" "$inputs" --inputs 1
  [[ $(rg -Fxc '========= ERROR SUMMARY: 0 errors' "$cap/live-$tool.log") == 1 ]]
  rg -q '^live_integration=PASS inputs=1 ' "$cap/live-$tool.log"
done
run binaries-after sha256sum "$out/cuda/model_producer_cuda_tests" "$out/cuda/model_producer_trt_tests" \
  "$out/cuda/libsaccade_model_producer_cuda.a" "$out/cuda/libsaccade_postprocess_cuda.a" "$out/cuda/postprocess_sanitizer_control"
cmp "$cap/binaries.log" "$cap/binaries-after.log"
cp "$out/cuda/compile_commands.json" "$cap/compile_commands.json"
cp "$out/cuda/CMakeCache.txt" "$cap/CMakeCache.txt"
check_source
printf 'PASS\n' > "$cap/bounded-checks.txt"

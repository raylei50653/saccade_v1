#!/usr/bin/env bash
# Run from repository root. All generated outputs stay in an ignored build-* tree
# so configure records the actual clean source commit. No Python is involved.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'usage: scripts/run_p01_acceptance.sh TENSORRT_SDK_ROOT build-OUTPUT' >&2
  exit 2
fi
sdk=$(realpath "$1")
out=$(realpath -m "$2")
root=$(git rev-parse --show-toplevel)
[[ $PWD == "$root" && $(basename "$out") == build-* && ! -e "$out" ]] || {
  echo 'run at repository root with a fresh build-* output directory' >&2; exit 2;
}
mkdir -p "$out"
run() {
  local name=$1
  shift
  printf '%q ' "$@" > "$out/$name.command"
  printf '\n' >> "$out/$name.command"
  local result=0
  "$@" > "$out/$name.log" 2>&1 || result=$?
  printf '%s\n' "$result" > "$out/$name.exit"
  echo "$name: exit=$result"
  if [[ $result != 0 ]]; then cat "$out/$name.log" >&2; return "$result"; fi
}
run source git rev-parse HEAD
run source-status git status --porcelain
run environment bash -c 'uname -a; cmake --version; c++ --version; /opt/cuda/bin/nvcc --version; nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv'
run cpu-configure cmake -S . -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release
run cpu-build cmake --build "$out/cpu" --parallel 4
run cpu-ctest ctest --test-dir "$out/cpu" --verbose
run manifest-sanitizer-build c++ -std=c++20 -O1 -g -Wall -Wextra -Wpedantic -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude \
  tests/model_manifest_tests.cpp src/model_manifest.cpp src/file_io.cpp src/sha256.cpp \
  -o "$out/manifest-sanitized"
run manifest-sanitizer "$out/manifest-sanitized" fixtures/model-manifest
run artifact-replay env OUTPUT_DIR="$out/artifact-replay" sh evidence/p01-a/2026-10-28/artifact/replay.sh
mkdir "$out/artifact-bundle"
run candidate-copy cp evidence/p01-a/2026-10-28/artifact/yolo26s.candidate.json "$out/artifact-bundle/"
run model-copy cp /home/ray/.cache/saccade-v1/p01-a/artifact/yolo26s.onnx "$out/artifact-bundle/"
run candidate-verify "$out/cpu/model_manifest" verify "$out/artifact-bundle/yolo26s.candidate.json" "$out/candidate.canonical.json"
run cuda-configure cmake -S . -B "$out/cuda" -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120
run cuda-build cmake --build "$out/cuda" --parallel 4
run cuda-ctest ctest --test-dir "$out/cuda" --verbose
run original-evidence sha256sum -c evidence/cuda-synthetic/2026-10-28-closeout/SHA256SUMS
mkdir "$out/baseline"
run original-extract tar -xzf evidence/cuda-synthetic/2026-10-28-closeout/raw-runs.tar.gz -C "$out/baseline"
run packet-regression bash -c '
  set -euo pipefail
  count=0
  for original in "$1"/baseline/gpu-evidence/case*/packets.bin; do
    relative=${original#"$1/baseline/"}
    "$1/cpu/packet_compare" "$original" "$1/cuda/$relative"
    cmp "${original%packets.bin}config.sha256" "$1/cuda/${relative%packets.bin}config.sha256"
    cmp "${original%packets.bin}behavior.sha256" "$1/cuda/${relative%packets.bin}behavior.sha256"
    count=$((count + 1))
  done
  [[ $count == 80 ]]
  echo "80 complete packet streams and config/behavior hashes match accepted baseline"
' _ "$out"
run trt-configure cmake -S . -B "$out/trt" -DCMAKE_BUILD_TYPE=Release \
  -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_TENSORRT_ROOT="$sdk" -DCUDAToolkit_ROOT=/opt/cuda
run trt-build cmake --build "$out/trt" --parallel 4
reject_configure() {
  local name=$1 expected=$2
  shift 2
  local result=0
  printf '%q ' "$@" > "$out/$name.command"
  printf '\n' >> "$out/$name.command"
  "$@" > "$out/$name.log" 2>&1 || result=$?
  printf '%s\n' "$result" > "$out/$name.exit"
  [[ $result != 0 ]] && rg -F "$expected" "$out/$name.log" > /dev/null || {
    cat "$out/$name.log" >&2; echo "expected rejection missing: $name" >&2; return 1;
  }
  echo "$name: expected exit=$result with targeted diagnostic"
}
reject_configure missing-sdk 'Set SACCADE_TENSORRT_ROOT' cmake -S . -B "$out/missing-sdk" \
  -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_TENSORRT_ROOT="$out/absent-sdk"
# A private header fixture tests the version guard; never alter the downloaded SDK.
trt_include=$(sed -n 's/^SACCADE_TENSORRT_INCLUDE:PATH=//p' "$out/trt/CMakeCache.txt")
trt_library=$(sed -n 's/^SACCADE_TENSORRT_LIBRARY:FILEPATH=//p' "$out/trt/CMakeCache.txt")
mkdir -p "$out/wrong-sdk/include"
cp -a "$trt_include/." "$out/wrong-sdk/include/"
ln -s "$(dirname "$trt_library")" "$out/wrong-sdk/lib"
sed -i -E 's/^#define NV_TENSORRT_MINOR .*/#define NV_TENSORRT_MINOR 15/' "$out/wrong-sdk/include/NvInferVersion.h"
rg -F '#define NV_TENSORRT_MINOR 15' "$out/wrong-sdk/include/NvInferVersion.h" > /dev/null
reject_configure wrong-sdk 'native TensorRT headers must be 10.16.1.11' cmake -S . -B "$out/wrong-sdk-build" \
  -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_TENSORRT_ROOT="$out/wrong-sdk" -DCUDAToolkit_ROOT=/opt/cuda
run trt-ctest ctest --test-dir "$out/trt" --verbose
mkdir "$out/native-run" "$out/native-memcheck"
run native-run "$out/trt/tensorrt_smoke" "$out/native-run/affine.engine" "$out/native-run/report.txt"
run native-memcheck /opt/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 \
  "$out/trt/tensorrt_smoke" "$out/native-memcheck/affine.engine" "$out/native-memcheck/report.txt"
run binaries sha256sum "$out/cpu/model_manifest" "$out/trt/tensorrt_smoke" "$out/cuda/cuda_synthetic_tests"
echo "Acceptance outputs: $out"

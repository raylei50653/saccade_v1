#!/usr/bin/env bash
# Capture S03 only from a clean tested source commit. Source + commands + raw
# logs are separate from the later evidence/docs commit. Run from repo root:
#   bash evidence/s03/2026-10-29/capture.sh build-s03-evidence [SOURCE_REV]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=${1:?usage: capture.sh build-NEW-OUTPUT [REV]}
rev=$(git rev-parse "${2:-HEAD}")
base=e9dfa51
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
trt=${SACCADE_TENSORRT_ROOT:-/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk}
[[ $out == build-* && ! -e $out ]] || { echo 'output must be a new build-* path' >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$rev" && -z $(git status --porcelain) ]] || { echo "HEAD must be $rev with a clean tree" >&2; exit 2; }
here=evidence/s03/2026-10-29
mkdir -p "$out/capture" "$out/source"
cap=$out/capture src=$out/source
run() {
  local name=$1 code=0
  shift
  printf '%q ' "$@" > "$cap/$name.command"; printf '\n' >> "$cap/$name.command"
  "$@" > "$cap/$name.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$code"
  return "$code"
}
run environment bash -c 'date -Is; uname -srmo; git --version; cmake --version | head -1; c++ --version | head -1; "$1" --version | tail -2; "$2" --version | head -3; nvidia-smi --query-gpu=name,driver_version,uuid,pci.bus_id,compute_cap --format=csv; ldd --version | head -1' _ "$nvcc" "$sanitizer"
run source git rev-parse HEAD origin/main "$rev"
run working-tree git status --porcelain=v1
run source-archive git archive --format=tar --output="$out/source.tar" "$rev"
run archive-hash sha256sum "$out/source.tar"
run source-extract tar -xf "$out/source.tar" -C "$src"
run source-file-hashes bash -c 'cd "$1" && sha256sum CMakeLists.txt cmake/ScanTensorRT.cmake include/saccade/scan* cuda/scan_frame.cu src/scan_trt* tests/scan_trt* tests/scan_cuda_suite.hpp tools/scan_trt_engine.cpp evidence/s03/2026-10-29/*.sh' _ "$src"
run changed-since-s02 git diff --stat "$base" "$rev"
run s02-kernels-and-reference-unchanged git diff --exit-code "$base" "$rev" -- cuda/selective_scan.cu src/selective_scan.cpp include/saccade/selective_scan.hpp include/saccade/selective_scan_cuda.hpp fixtures/selective-scan tests/selective_scan_tests.cpp
run historical-evidence-unchanged git diff --exit-code "$base" "$rev" -- evidence/s02 evidence/s01 evidence/cuda-synthetic evidence/p01-a evidence/scan-contract
run historical-evidence-hashes bash -c 'bash "$1/evidence/s02/2026-10-29/historical-hashes.sh" && sha256sum --quiet -c "$1/evidence/s02/2026-10-29/SHA256SUMS" && echo "S02 all entries OK"' _ "$src"
run cpu-configure cmake -S "$src" -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release
run cpu-build cmake --build "$out/cpu" --parallel 4
run cpu-ctest ctest --test-dir "$out/cpu" --output-on-failure
run trt-configure cmake -S "$src" -B "$out/trt" -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_TENSORRT=ON \
  "-DSACCADE_TENSORRT_ROOT=$trt" "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120
run trt-options bash -c 'rg "^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|SACCADE_ENABLE_CUDA|SACCADE_ENABLE_TENSORRT|SACCADE_TENSORRT_ROOT):" "$1/CMakeCache.txt"' _ "$out/trt"
run trt-build cmake --build "$out/trt" --parallel 4 --verbose
run trt-ctest ctest --test-dir "$out/trt" --output-on-failure --verbose
run engine-report cp "$out/trt/scan-trt/run-report.txt" "$cap/engine-run-report.txt"
run scan-trt-report "$out/trt/scan_trt_tests" "$cap/scan-trt-report.tsv"
run scan-trt-api-trace "$out/trt/scan_trt_tests" --group api_trace
run kernel-resources "$(dirname "$nvcc")/cuobjdump" --dump-resource-usage "$out/trt/libsaccade_selective_scan_cuda.a"
for tool in memcheck racecheck synccheck; do
  run "sanitizer-$tool" "$sanitizer" --tool "$tool" --error-exitcode 99 "$out/trt/scan_trt_tests"
done
run sanitizer-initcheck "$sanitizer" --tool initcheck --error-exitcode 99 "$out/trt/scan_trt_tests" --no-held-stream
printf '%s\n' 'api_trace: excluded from sanitizer; separate native CUPTI process' \
  'async_fault: excluded from sanitizer; separate fatal-context CTest process' \
  'initcheck: --no-held-stream skips the callback-held stream test only' > "$cap/sanitizer-skips.txt"
run control-build "$nvcc" -std=c++20 -arch=sm_120 -lineinfo -o "$out/sanitizer-control" "$src/evidence/s02/2026-10-29/sanitizer-control.cu"
for tool in memcheck initcheck racecheck synccheck; do
  run "control-$tool" bash -c '"$1" --tool "$2" --error-exitcode 99 "$3" "$2"; code=$?; echo "tool exit=$code"; [[ $code == 99 ]]' \
    _ "$sanitizer" "$tool" "$out/sanitizer-control"
done
run mutation-check env "NVCC=$nvcc" "SACCADE_TENSORRT_ROOT=$trt" bash "$src/$here/mutation-check.sh" "$src" "$out/mutations"
run binary-hashes sha256sum "$out/cpu/selective_scan_tests" "$out/trt/scan_trt_tests" "$out/trt/scan_trt_engine" \
  "$out/trt/libsaccade_selective_scan_cuda.a" "$out/trt/libsaccade_scan_trt.a" "$out/trt/libsaccade_scan_trt_frame.a" "$out/sanitizer-control"

#!/usr/bin/env bash
# S02 evidence for the tested source commit: clean archive, default CPU build
# and CTest, CUDA build and CTest, the GPU scan report, four compute-sanitizer
# tools with positive controls, and the CUDA mutation check. Requires HEAD to be
# the tested commit with a clean tree, so the CUDA build embeds that identity.
# Usage from the repository root:
#   bash evidence/s02/2026-10-29/capture.sh build-NEW-OUTPUT [REV]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=${1:?usage: capture.sh build-NEW-OUTPUT [REV]}
rev=$(git rev-parse "${2:-HEAD}")
base=60d722cfa45bb6d9763740cf6c8103c9defa9c0d
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
[[ $out == build-* && ! -e $out ]] || { echo "output must be a new build-* path" >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$rev" && -z $(git status --porcelain) ]] || { echo "HEAD must be $rev with a clean tree" >&2; exit 2; }
here=evidence/s02/2026-10-29
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
run source-file-hashes bash -c 'cd "$1" && sha256sum CMakeLists.txt include/saccade/selective_scan_cuda.hpp cuda/selective_scan.cu tests/selective_scan_cuda_tests.cpp docs/SCAN_CUDA.md evidence/s02/2026-10-29/*.sh evidence/s02/2026-10-29/*.cu' _ "$src"
run changed-since-baseline git diff --stat "$base" "$rev"
# S01 reference, its fixtures and every earlier runtime stay byte-identical.
run s01-and-runtime-unchanged git diff "$base" "$rev" -- src include/saccade/selective_scan.hpp include/saccade/protocol.hpp \
  include/saccade/sha256.hpp include/saccade/synthetic_config.hpp include/saccade/synthetic_runtime.hpp include/saccade/file_io.hpp \
  include/saccade/model_manifest.hpp include/saccade/cleanup.hpp cuda/synthetic_runtime.cu tools cmake fixtures \
  tests/selective_scan_tests.cpp tests/protocol_tests.cpp tests/synthetic_config_tests.cpp tests/file_io_tests.cpp \
  tests/cleanup_tests.cpp tests/model_manifest_tests.cpp tests/cuda_synthetic_tests.cpp tests/cuda_cleanup_tests.cpp
run cpu-configure cmake -S "$src" -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release
run cpu-build cmake --build "$out/cpu" --parallel 4
run cpu-ctest ctest --test-dir "$out/cpu" --output-on-failure
run cuda-configure cmake -S "$src" -B "$out/cuda" -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON \
  "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120
run cuda-options bash -c 'grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|SACCADE_ENABLE_CUDA):" "$1/CMakeCache.txt"' _ "$out/cuda"
run cuda-build cmake --build "$out/cuda" --parallel 4 --verbose
run cuda-ctest ctest --test-dir "$out/cuda" --output-on-failure
run scan-cuda-report "$out/cuda/selective_scan_cuda_tests" "$cap/scan-cuda-report.tsv"
run kernel-resources "$(dirname "$nvcc")/cuobjdump" --dump-resource-usage "$out/cuda/libsaccade_selective_scan_cuda.a"
for tool in memcheck racecheck synccheck; do
  run "sanitizer-$tool" "$sanitizer" --tool "$tool" --error-exitcode 99 "$out/cuda/selective_scan_cuda_tests"
done
# initcheck makes the enqueue wait on a stream held by a host callback; only
# that test is skipped (docs/SCAN_CUDA.md §6).
run sanitizer-initcheck "$sanitizer" --tool initcheck --error-exitcode 99 "$out/cuda/selective_scan_cuda_tests" --no-held-stream
run control-build "$nvcc" -std=c++20 -arch=sm_120 -lineinfo -o "$out/sanitizer-control" "$src/$here/sanitizer-control.cu"
for tool in memcheck initcheck racecheck synccheck; do # each tool must report its planted defect
  run "control-$tool" bash -c '"$1" --tool "$2" --error-exitcode 99 "$3" "$2"; code=$?; echo "tool exit=$code"; [[ $code == 99 ]]' \
    _ "$sanitizer" "$tool" "$out/sanitizer-control"
done
run control-synccheck-shuffle-mask bash -c '"$1" --tool synccheck --error-exitcode 99 "$2" shuffle-mask; echo "tool exit=$?"' \
  _ "$sanitizer" "$out/sanitizer-control"
run mutation-check env "NVCC=$nvcc" bash "$src/$here/mutation-check.sh" "$src" "$out/mutations"
run historical-evidence-unchanged git diff "$base" "$rev" -- evidence/cuda-synthetic evidence/p01-a evidence/scan-contract evidence/s01
run historical-evidence-hashes bash "$src/$here/historical-hashes.sh"
run binary-hashes sha256sum "$out/cpu/selective_scan_tests" "$out/cuda/selective_scan_cuda_tests" \
  "$out/cuda/libsaccade_selective_scan_cuda.a" "$out/sanitizer-control"

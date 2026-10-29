#!/usr/bin/env bash
# S01 evidence for the tested source commit: clean archive, default CPU build
# and CTest, ASan/UBSan build and CTest, independent expected regeneration and
# fixture mutation sensitivity. No Python, GPU, CUDA, TensorRT or protobuf.
# Usage from the repository root:
#   bash evidence/s01/2026-10-29/reference/capture.sh build-NEW-OUTPUT [REV]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=${1:?usage: capture.sh build-NEW-OUTPUT [REV]}
rev=${2:-97f9b8c43f4127feb4bd1056893648f9b6cfc6c5}
base=2aef5f8feaec6126f814cd1ed1cabf85f8994c9b
[[ $out == build-* && ! -e $out ]] || { echo "output must be a new build-* path" >&2; exit 2; }
here=evidence/s01/2026-10-29/reference
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
run environment bash -c 'date -Is; uname -srmo; git --version; cmake --version | head -1; c++ --version | head -1; bc --version | head -1; ldd --version | head -1'
run source git rev-parse HEAD origin/main "$rev"
run working-tree git status --porcelain=v1
run source-archive git archive --format=tar --output="$out/source.tar" "$rev"
run archive-hash sha256sum "$out/source.tar"
run source-extract tar -xf "$out/source.tar" -C "$src"
run source-file-hashes bash -c 'cd "$1" && sha256sum CMakeLists.txt include/saccade/selective_scan.hpp src/selective_scan.cpp tests/selective_scan_tests.cpp fixtures/selective-scan/* docs/SCAN_CONTRACT.md .github/workflows/protocol.yml' _ "$src"
run changed-since-baseline git diff --stat "$base" "$rev"
run shared-runtime-unchanged git diff "$base" "$rev" -- src/protocol.cpp src/sha256.cpp src/synthetic_config.cpp src/file_io.cpp src/model_manifest.cpp cuda tools include/saccade/protocol.hpp include/saccade/sha256.hpp include/saccade/synthetic_config.hpp include/saccade/synthetic_runtime.hpp include/saccade/file_io.hpp include/saccade/model_manifest.hpp cmake fixtures/model-manifest
run cpu-configure cmake -S "$src" -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release
run cpu-options bash -c 'grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CXX_FLAGS_RELEASE|SACCADE_ENABLE_CUDA|SACCADE_ENABLE_TENSORRT):" "$1/CMakeCache.txt"' _ "$out/cpu"
run cpu-build cmake --build "$out/cpu" --parallel 4 --verbose
run cpu-ctest ctest --test-dir "$out/cpu" --verbose
run cpu-reference-report "$out/cpu/selective_scan_tests" "$src/fixtures/selective-scan" "$cap/reference-report.tsv"
run cpu-linkage ldd "$out/cpu/selective_scan_tests"
san='-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer'
run sanitizer-configure cmake -S "$src" -B "$out/sanitize" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  "-DCMAKE_CXX_FLAGS=$san" "-DCMAKE_EXE_LINKER_FLAGS=$san"
run sanitizer-build cmake --build "$out/sanitize" --parallel 4 --verbose
run sanitizer-ctest env ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 \
  ctest --test-dir "$out/sanitize" --verbose
run expected-regenerate bash -c 'bash "$1/fixtures/selective-scan/derive.sh" > "$2/expected.regenerated.tsv"' _ "$src" "$out"
run expected-reproduce cmp "$out/expected.regenerated.tsv" "$src/fixtures/selective-scan/expected.tsv"
run expected-hash sha256sum "$src/fixtures/selective-scan/expected.tsv" "$out/expected.regenerated.tsv"
run mutation-check bash "$here/mutation-check.sh" "$src" "$out/mutations"
run historical-evidence-unchanged git diff "$base" "$rev" -- evidence
run historical-evidence-hashes bash "$here/historical-hashes.sh"
run binary-hashes sha256sum "$out/cpu/selective_scan_tests" "$out/sanitize/selective_scan_tests"

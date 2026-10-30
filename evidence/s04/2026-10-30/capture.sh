#!/usr/bin/env bash
# Capture S04 only from a clean tested source commit. Source + commands + raw
# logs are separate from the later evidence/docs commit. Run from repo root:
#   bash evidence/s04/2026-10-30/capture.sh build-s04-evidence [SOURCE_REV]
# Needs the candidate checkpoint (CHECKPOINT, default the legacy T1 best.ckpt),
# the pinned TensorRT SDK, CUDA, compute-sanitizer and a GPU.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=${1:?usage: capture.sh build-NEW-OUTPUT [REV]}
rev=$(git rev-parse "${2:-HEAD}")
base=f02dd7a58aed87adec6265bc27bd54899b4f48ed   # S03 delivery
policy=4d38bcc51caf829afe737959e334d0c85b1eba0b # S04 policy commit
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
trt=${SACCADE_TENSORRT_ROOT:-/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk}
checkpoint=${CHECKPOINT:-/home/ray/developer/ai/saccade/runs/mamba_gt_yolo26m_v14replica_t3_t1/best.ckpt}
checkpoint_sha=a5a0e7091cd854ad68210ae4bbd43af73c7d3d098ea119f02ea65fab59c0676c
bundle_sha=e1a8660fb9ad510f20ccc3d56a4feac61c6a2e5f771aab531caa741e7677f310
[[ $out == build-* && ! -e $out ]] || { echo 'output must be a new build-* path' >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$rev" && -z $(git status --porcelain) ]] || { echo "HEAD must be $rev with a clean tree" >&2; exit 2; }
here=evidence/s04/2026-10-30
mkdir -p "$out/capture" "$out/source" "$out/stage"
out=$(cd "$out" && pwd)
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
run environment bash -c 'date -Is; uname -srmo; git --version; cmake --version | head -1; c++ --version | head -1; "$1" --version | tail -2; "$2" --version | head -3; nvidia-smi --query-gpu=name,driver_version,uuid,pci.bus_id,compute_cap --format=csv; ldd --version | head -1; nproc' _ "$nvcc" "$sanitizer"
run source git rev-parse HEAD "$rev" "$base" "$policy"
run working-tree git status --porcelain=v1
run source-archive git archive --format=tar --output="$out/source.tar" "$rev"
run archive-hash sha256sum "$out/source.tar"
run source-extract tar -xf "$out/source.tar" -C "$src"
run source-file-hashes bash -c 'cd "$1" && sha256sum CMakeLists.txt cmake/*.cmake include/saccade/*.hpp src/*.cpp cuda/*.cu tests/*.cpp tests/*.hpp tests/*.cu tools/*.cpp fixtures/model-manifest/* manifests/s04/* docs/HEAD_POLICY.md evidence/s04/2026-10-30/*.sh' _ "$src"
run changed-since-s03 git diff --stat "$base" "$rev"
run policy-unchanged-since-policy-commit git diff --exit-code "$policy" "$rev" -- docs/HEAD_POLICY.md src/head_reference.cpp include/saccade/head_reference.hpp
run s01-s02-s03-kernels-reference-plugins-unchanged git diff --exit-code "$base" "$rev" -- cuda/selective_scan.cu src/selective_scan.cpp \
  include/saccade/selective_scan.hpp include/saccade/selective_scan_cuda.hpp fixtures/selective-scan tests/selective_scan_tests.cpp \
  src/scan_trt_plugin.cpp include/saccade/scan_trt_plugin.hpp src/scan_trt_network.cpp include/saccade/scan_trt_network.hpp
run historical-evidence-unchanged git diff --exit-code "$base" "$rev" -- evidence/s03 evidence/s02 evidence/s01 evidence/cuda-synthetic evidence/p01-a evidence/scan-contract
run historical-evidence-hashes bash -c 'cd "$1" && sha256sum --quiet -c evidence/s03/2026-10-29/SHA256SUMS && echo "S03 all entries OK"' _ "$src"
run checkpoint-identity bash -c 'stat -L -c "%s bytes" "$1" && sha256sum "$1" && [[ $(sha256sum < "$1" | cut -d" " -f1) == "$2" ]]' _ "$checkpoint" "$checkpoint_sha"

run cpu-configure cmake -S "$src" -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release "-DSACCADE_HEAD_CHECKPOINT=$checkpoint"
run cpu-build cmake --build "$out/cpu" --parallel 8
run cpu-ctest ctest --test-dir "$out/cpu" --output-on-failure
run weights-convert "$out/cpu/head_weights" convert "$checkpoint" "$checkpoint_sha" "$out/stage/t1-head.swb" "$cap/t1-head-mapping.tsv"
run weights-verify "$out/cpu/head_weights" verify "$out/stage/t1-head.swb" "$bundle_sha"
run reference-report "$out/cpu/head_reference_tests" "$out/stage/t1-head.swb" "$bundle_sha"
run cpu-sanitize-configure cmake -S "$src" -B "$out/cpu-asan" -DCMAKE_BUILD_TYPE=RelWithDebInfo "-DSACCADE_HEAD_CHECKPOINT=$checkpoint" \
  "-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
run cpu-sanitize-build cmake --build "$out/cpu-asan" --parallel 8
run cpu-sanitize-ctest ctest --test-dir "$out/cpu-asan" --output-on-failure

run trt-configure cmake -S "$src" -B "$out/trt" -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_TENSORRT=ON \
  "-DSACCADE_TENSORRT_ROOT=$trt" "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 "-DSACCADE_HEAD_CHECKPOINT=$checkpoint"
run trt-options bash -c 'grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|SACCADE_ENABLE_CUDA|SACCADE_ENABLE_TENSORRT|SACCADE_TENSORRT_ROOT|SACCADE_HEAD_CHECKPOINT):" "$1/CMakeCache.txt"' _ "$out/trt"
run trt-build cmake --build "$out/trt" --parallel 8 --verbose
run trt-ctest ctest --test-dir "$out/trt" --output-on-failure --verbose
engine=$out/trt/head-trt/head.engine
engine_sha=$(sha256sum "$engine" | cut -d' ' -f1)
run engine-identity bash -c 'stat -c "%s bytes" "$1" && sha256sum "$1" "$(cat "$2")"' _ "$engine" "$out/trt/head-trt/bundle.path"
run engine-build-report cp "$out/trt/head-trt/build-report.txt" "$cap/head-build-report.txt"
run engine-run-report cp "$out/trt/head-trt/run-report.txt" "$cap/head-run-report.txt"
run head-suite-report cp "$out/trt/head-trt/suite-report.tsv" "$cap/head-suite-report.tsv"
run head-api-trace "$out/trt/head_trt_tests" "$engine" "$engine_sha" "$out/stage/t1-head.swb" "$bundle_sha" --group api_trace
run scan-trt-report "$out/trt/scan_trt_tests" "$cap/scan-trt-report.tsv"
run engine-rebuild-determinism bash -c '"$1" build "$2" "$3" "$4" "$5" && sha256sum "$4" "$6"' _ "$out/trt/head_trt" "$out/stage/t1-head.swb" "$bundle_sha" \
  "$out/stage/rebuild.engine" "$out/stage/rebuild-report.txt" "$engine"

# Manifest: declaration and file integrity in a staging bundle directory.
cp "$src/manifests/s04/yolo26m-head-t1.json" "$src/docs/HEAD_POLICY.md" "$out/stage/"
ln -s "$checkpoint" "$out/stage/best.ckpt"
run manifest-validate "$out/cpu/model_manifest" validate "$out/stage/yolo26m-head-t1.json" "$cap/yolo26m-head-t1.canonical.json"
run manifest-verify "$out/cpu/model_manifest" verify "$out/stage/yolo26m-head-t1.json"
run manifest-schema1-fixture "$out/cpu/model_manifest" verify "$src/fixtures/model-manifest/contract.json"

# Optional matching backbone/neck: ordered I/O only (structure, no weights
# attribution, no inference), with the P01-A inspector and the pinned schema.
backbone=${BACKBONE_ONNX:-/home/ray/developer/ai/saccade/models/yolo/yolo26m_backbone_640.onnx}
onnx_proto=${ONNX_PROTO:-build-artifact-inspector/onnx.proto}
mkdir -p "$out/onnx"
run backbone-identity bash -c 'stat -L -c "%s bytes" "$1" && sha256sum "$1"' _ "$backbone"
run onnx-schema-hash bash -c 'cp "$1" "$2/onnx.proto" && echo "5943f7cb1d0359edfbc7871cb8ec3095ca942c04dea28ee15c926d51e1185efe  $2/onnx.proto" | sha256sum -c' _ "$onnx_proto" "$out/onnx"
run onnx-schema-generate protoc --cpp_out="$out/onnx" -I "$out/onnx" "$out/onnx/onnx.proto"
run onnx-inspect-build bash -c 'c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -I "$1/include" -I "$2" "$1/tools/onnx_contract_inspect.cpp" "$1/src/file_io.cpp" "$2/onnx.pb.cc" $(pkg-config --cflags --libs protobuf) -o "$2/onnx_contract_inspect"' _ "$src" "$out/onnx"
run backbone-ordered-io bash -c '"$1" "$2" | grep -E "^(scope|ir_version|producer|opset|metadata|graph.root|input|output)"' _ "$out/onnx/onnx_contract_inspect" "$backbone"

suite=("$out/trt/head_trt_tests" "$engine" "$engine_sha" "$out/stage/t1-head.swb" "$bundle_sha" --skip intermediates)
for tool in memcheck racecheck synccheck; do
  run "sanitizer-head-$tool" "$sanitizer" --tool "$tool" --error-exitcode 99 "${suite[@]}"
  run "sanitizer-scan-trt-$tool" "$sanitizer" --tool "$tool" --error-exitcode 99 "$out/trt/scan_trt_tests"
done
run sanitizer-head-initcheck "$sanitizer" --tool initcheck --error-exitcode 99 "${suite[@]}" --no-held-stream
run sanitizer-scan-trt-initcheck "$sanitizer" --tool initcheck --error-exitcode 99 "$out/trt/scan_trt_tests" --no-held-stream
printf '%s\n' 'head intermediates: excluded from sanitizers (rebuilds a debug engine); native run in trt-ctest head_trt' \
  'head api_trace: explicit group, separate native CUPTI process (head-api-trace)' \
  'scan_trt api_trace / async_fault: explicit groups, native only (trt-ctest)' \
  'initcheck: --no-held-stream skips the callback-held stream groups only' > "$cap/sanitizer-skips.txt"
run control-build "$nvcc" -std=c++20 -arch=sm_120 -lineinfo -o "$out/sanitizer-control" "$src/evidence/s02/2026-10-29/sanitizer-control.cu"
for tool in memcheck initcheck racecheck synccheck; do
  run "control-$tool" bash -c '"$1" --tool "$2" --error-exitcode 99 "$3" "$2"; code=$?; echo "tool exit=$code"; [[ $code == 99 ]]' \
    _ "$sanitizer" "$tool" "$out/sanitizer-control"
done
run mutation-check env "NVCC=$nvcc" "SACCADE_TENSORRT_ROOT=$trt" bash "$src/$here/mutation-check.sh" "$src" "$out/mutations" "$out/stage/t1-head.swb" "$bundle_sha"
run mutation-results cp "$out/mutations/results.tsv" "$cap/mutation-results.tsv"
run binary-hashes sha256sum "$out/cpu/head_weights" "$out/cpu/head_reference_tests" "$out/cpu/head_weights_tests" "$out/cpu/model_manifest" \
  "$out/trt/head_trt" "$out/trt/head_trt_tests" "$out/trt/scan_trt_tests" "$out/trt/libsaccade_head_trt.a" "$out/trt/libsaccade_scan_trt.a" \
  "$out/trt/libsaccade_scan_trt_frame.a" "$out/trt/libsaccade_selective_scan_cuda.a" "$engine" "$out/stage/t1-head.swb" "$out/sanitizer-control"

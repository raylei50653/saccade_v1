#!/usr/bin/env bash
# Capture the T01 backbone engine and F03 complete model (docs/MODEL_TRT.md,
# docs/BACKBONE_POLICY.md) only from a clean tested source commit; raw logs are
# separate from the later evidence/docs commit. Run from repo root:
#   bash evidence/t01/2026-10-30/f03/capture.sh build-NEW-OUTPUT [SOURCE_REV]
# Needs the T01 artifacts under ~/.cache/saccade-v1/t01 (teacher e12 and T1
# checkpoints, backbone ONNX + backbone-export.json, caches, src, venv-export),
# the pinned TensorRT SDK with its ONNX parser, CUDA, compute-sanitizer and a GPU.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
out=${1:?usage: capture.sh build-NEW-OUTPUT [REV]}
rev=$(git rev-parse "${2:-HEAD}")
s04_tested=3857e243e8fc32663aa688f073343002ad12cb30      # S04 tested source
head_policy=4d38bcc51caf829afe737959e334d0c85b1eba0b     # HEAD_POLICY commit
head_pass=f9077c36e5bbe5aafd8c862fdb62cafa28a17562       # T01 native head record
backbone_policy=fb10377c2f37062126dccbfcc403e0d3282dea8f # BACKBONE_POLICY commit
nvcc=${NVCC:-/opt/cuda/bin/nvcc}
sanitizer=${SANITIZER:-/opt/cuda/bin/compute-sanitizer}
trt=${SACCADE_TENSORRT_ROOT:-/home/ray/.cache/saccade-native-trt/10.16.1.11/sdk}
t01=${T01_ROOT:-$HOME/.cache/saccade-v1/t01}
py=$t01/venv-export/bin/python
teacher=$t01/runs/teacher/attempt-01/run/epoch_0012.ckpt
teacher_sha=14e86231f9c0e3aacea8ffcc9dd8c7dfb0f3e7a01b848ae6ea2c57a3d2899e1b
checkpoint=$t01/runs/t1/attempt-01/run/epoch_0001.ckpt
checkpoint_sha=070580f9f5755e57acd9ee7ed9b4307672f4d9996332b40e6b61ee7cc6e6962b
bundle_sha=4fccfda6342b88d43fc2bf676228810b974b9e9956f45fa53352124098f046a3
onnx=$t01/export/yolo26m-backbone-t01.onnx
onnx_sha=007c32600832b1be94b077c513210346f01ff280d95cd92b4200d8ba7a0c236f
inputs_sha=14dda6867e23ed8a2d83086224d71a445a7fdcb606c7507a3f39c0e5faf60aa0    # BACKBONE_POLICY §4
reference_sha=ccb6f9da04140f81f0b711fc9826a68de20ff12353b6af460b755a06a2419584 # backbone-reference.json at fb10377
head_engine_sha=5b6d30fab444a6a6f649b187f3bc7d6f315a3a9256b48374872df4b35276024a # HEAD_TRT §10
labels=MOT17-02-SDP/1,MOT17-10-SDP/1,MOT17-13-SDP/400,zeros,ones,random/1
low_memory_mib=4096
[[ $out == build-* && ! -e $out ]] || { echo 'output must be a new build-* path' >&2; exit 2; }
[[ $(git rev-parse HEAD) == "$rev" && -z $(git status --porcelain) ]] || { echo "HEAD must be $rev with a clean tree" >&2; exit 2; }
mkdir -p "$out/capture" "$out/source" "$out/stage" "$out/t01root/export" "$out/backbone-dump" "$out/model-dump" "$out/chain-dump"
out=$(cd "$out" && pwd)
cap=$out/capture src=$out/source root=$out/t01root
# Memory samples every 10 s; below 4 GiB available the whole capture is stopped.
(
  while true; do
    date -Is; free -m
    avail=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
    if (( avail < low_memory_mib )); then
      echo "LOW MEMORY: ${avail} MiB available < ${low_memory_mib} MiB; stopping the capture"
      kill -TERM 0
    fi
    sleep 10
  done
) > "$cap/memory.log" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true' EXIT
clean_source() {
  if [[ $(git rev-parse HEAD) != "$rev" || -n $(git status --porcelain) ]]; then
    { date -Is; git rev-parse HEAD; git status --porcelain; } >> "$cap/source-drift.log"
    echo 'source HEAD or working tree changed during capture' >&2
    return 2
  fi
}
run() {
  local name=$1 code=0
  shift
  clean_source || return $?
  printf '%q ' "$@" > "$cap/$name.command"; printf '\n' >> "$cap/$name.command"
  "$@" > "$cap/$name.log" 2>&1 || code=$?
  printf '%s\n' "$code" > "$cap/$name.exit"
  printf '%s: exit=%s\n' "$name" "$code"
  clean_source || return $?
  return "$code"
}
identity() { # FILE SHA: size, hash, and the hash must be SHA
  bash -c 'stat -L -c "%s bytes" "$1" && sha256sum "$1" && [[ $(sha256sum < "$1" | cut -d" " -f1) == "$2" ]]' _ "$1" "$2"
}
hashes=(-DSACCADE_HEAD_CHECKPOINT="$checkpoint" -DSACCADE_HEAD_CHECKPOINT_SHA256="$checkpoint_sha" -DSACCADE_HEAD_BUNDLE_SHA256="$bundle_sha")

run environment bash -c 'date -Is; uname -srmo; git --version; cmake --version | head -1; c++ --version | head -1; "$1" --version | tail -2; "$2" --version | head -3; nvidia-smi --query-gpu=name,driver_version,uuid,pci.bus_id,compute_cap --format=csv; ldd --version | head -1; nproc; "$3" --version' _ "$nvcc" "$sanitizer" "$py"
run source git rev-parse HEAD "$rev" "$s04_tested" "$head_policy" "$head_pass" "$backbone_policy"
run working-tree git status --porcelain=v1
run source-archive git archive --format=tar --output="$out/source.tar" "$rev"
run archive-hash sha256sum "$out/source.tar"
run source-extract tar -xf "$out/source.tar" -C "$src"
# GPU kernels, plugins, frame session, weight delivery and the head comparator are the S04 tested source.
run gpu-code-unchanged-since-s04 git diff --exit-code "$s04_tested" "$rev" -- cuda src/scan_trt_plugin.cpp src/scan_trt_network.cpp \
  src/scan_trt_frame.cpp src/selective_scan.cpp src/head_weights.cpp src/torch_checkpoint.cpp src/head_compare.cpp \
  include/saccade/scan_trt_plugin.hpp include/saccade/scan_trt_network.hpp include/saccade/scan_trt_frame.hpp \
  include/saccade/scan_frame_kernels.hpp include/saccade/selective_scan.hpp include/saccade/selective_scan_cuda.hpp \
  include/saccade/head_weights.hpp include/saccade/torch_checkpoint.hpp include/saccade/head_compare.hpp
run head-policy-unchanged git diff --exit-code "$head_policy" "$rev" -- docs/HEAD_POLICY.md
run backbone-policy-unchanged git diff --exit-code "$backbone_policy" "$rev" -- docs/BACKBONE_POLICY.md evidence/t01/2026-10-30/backbone-policy
run earlier-evidence-unchanged git diff --exit-code "$head_pass" "$rev" -- evidence/s04 evidence/s03 evidence/s02 evidence/s01 evidence/cuda-synthetic \
  evidence/p01-a evidence/scan-contract evidence/t01/2026-10-30/formal evidence/t01/2026-10-30/native-head evidence/t01/2026-10-30/precision \
  manifests/s04 manifests/t01/yolo26m-head-t1.json
run native-changes-since-t01-head git diff --stat "$head_pass" "$rev" -- src include cuda tools tests cmake CMakeLists.txt training
# The only changes to existing head code: head_reference_from_refs and the HeadBuildOptions::features hook.
run head-code-changes-since-t01-head git diff "$head_pass" "$rev" -- src/head_reference.cpp include/saccade/head_reference.hpp \
  src/head_trt_network.cpp include/saccade/head_trt_network.hpp
run artifact-identities bash -c 'set -e; for pair in "$@"; do f=${pair%%=*}; s=${pair#*=}; stat -L -c "%s bytes %n" "$f"; sha256sum "$f"; [[ $(sha256sum < "$f" | cut -d" " -f1) == "$s" ]]; done' \
  _ "$teacher=$teacher_sha" "$checkpoint=$checkpoint_sha" "$onnx=$onnx_sha"
run selection-records python3 -c 'import json, sys
for path, sha in zip(sys.argv[1::2], sys.argv[2::2]):
    d = json.load(open(path)); print(path, json.dumps(d)[:400])
    assert d["sha256"] == sha, path' "$t01/runs/teacher/selection.json" "$teacher_sha" "$t01/runs/t1/selection.json" "$checkpoint_sha"
run onnx-export-record python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print({k: d[k] for k in ("onnx_sha256", "teacher_sha256", "initializers", "float_initializers_unmatched", "teacher_float_tensors_not_in_graph_outside_detect")}); sys.exit(d["onnx_sha256"] != sys.argv[2] or d["teacher_sha256"] != sys.argv[3] or bool(d["float_initializers_unmatched"]))' \
  "$t01/export/backbone-export.json" "$onnx_sha" "$teacher_sha"
run sdk-files bash -c 'cd "$1" && find . \( -type f -o -type l \) | sort | while read -r f; do if [[ -L $f ]]; then echo "L $f -> $(readlink "$f")"; else sha256sum "$f"; fi; done' _ "$trt"

# Inputs and binary64 reference regenerated in a separate artifact root; they must equal the policy commit's.
ln -s "$t01/src" "$root/src"
ln -s "$t01/runs" "$root/runs"
ln -s "$t01/export/backbone-export.json" "$root/export/backbone-export.json"
ref=(env SACCADE_T01_ROOT="$root" "$py" "$src/training/t01/t01_backbone_ref.py")
run reference-inputs "${ref[@]}" inputs
run reference-inputs-identity identity "$root/export/f03-inputs.f32" "$inputs_sha"
run reference-backbone "${ref[@]}" reference
run reference-backbone-identity identity "$root/export/backbone-reference.f64" "$reference_sha"
run reference-precheck "${ref[@]}" precheck
run reference-records cp "$root/export/f03-inputs.json" "$root/export/backbone-reference.json" "$root/export/backbone-precheck.json" "$cap/"
images=$root/export/f03-inputs.f32 bbref=$root/export/backbone-reference.f64

run cpu-configure cmake -S "$src" -B "$out/cpu" -DCMAKE_BUILD_TYPE=Release "${hashes[@]}"
run cpu-build cmake --build "$out/cpu" --parallel 8
run cpu-ctest ctest --test-dir "$out/cpu" --output-on-failure
run weights-convert "$out/cpu/head_weights" convert "$checkpoint" "$checkpoint_sha" "$out/stage/t1-head.swb" "$cap/t1-head-mapping.tsv"
run weights-verify "$out/cpu/head_weights" verify "$out/stage/t1-head.swb" "$bundle_sha"
run reference-report "$out/cpu/head_reference_tests" "$out/stage/t1-head.swb" "$bundle_sha"
bundle=$out/stage/t1-head.swb

run trt-configure cmake -S "$src" -B "$out/trt" -DCMAKE_BUILD_TYPE=Release -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_TENSORRT=ON \
  -DSACCADE_ENABLE_MODEL_TRT=ON "-DSACCADE_TENSORRT_ROOT=$trt" "-DCMAKE_CUDA_COMPILER=$nvcc" -DCMAKE_CUDA_ARCHITECTURES=120 "${hashes[@]}"
run trt-options bash -c 'grep -E "^(CMAKE_BUILD_TYPE|CMAKE_CXX_COMPILER|CMAKE_CUDA_COMPILER|CMAKE_CUDA_ARCHITECTURES|CMAKE_CUDA_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|SACCADE_ENABLE_CUDA|SACCADE_ENABLE_TENSORRT|SACCADE_ENABLE_MODEL_TRT|SACCADE_TENSORRT_ROOT|SACCADE_TENSORRT_LIBRARY|SACCADE_TENSORRT_ONNX_PARSER_LIBRARY|SACCADE_TENSORRT_ONNX_PARSER_INCLUDE|SACCADE_HEAD_CHECKPOINT|SACCADE_HEAD_CHECKPOINT_SHA256|SACCADE_HEAD_BUNDLE_SHA256):" "$1/CMakeCache.txt"' _ "$out/trt"
run trt-build cmake --build "$out/trt" --parallel 8 --verbose
run trt-ctest ctest --test-dir "$out/trt" --output-on-failure --verbose
run head-suite-report cp "$out/trt/head-trt/suite-report.tsv" "$cap/head-suite-report.tsv"
head_engine=$out/trt/head-trt/head.engine
# The head network code with the features hook unset builds the HEAD_TRT §10 engine bytes.
run head-engine-identity identity "$head_engine" "$head_engine_sha"
run head-engine-rebuild bash -c '"$1" build "$2" "$3" "$4" "$5" && cmp "$4" "$6"' _ "$out/trt/head_trt" "$bundle" "$bundle_sha" \
  "$out/stage/head-rebuild.engine" "$cap/head-rebuild-build-report.txt" "$head_engine"

# Backbone engine (t01-backbone-fp32/1).
bb_engine=$out/stage/backbone.engine
run backbone-build "$out/trt/backbone_trt" build "$onnx" "$onnx_sha" "$bb_engine" "$cap/backbone-build-report.txt"
bb_sha=$(sha256sum "$bb_engine" | cut -d' ' -f1)
run backbone-rebuild-record bash -c '"$1" build "$2" "$3" "$4" "$5" >/dev/null && sha256sum "$4" "$6" && { cmp -s "$4" "$6" && echo "rebuild: identical plan bytes" || echo "rebuild: plan bytes differ (recorded, not a criterion)"; }' \
  _ "$out/trt/backbone_trt" "$onnx" "$onnx_sha" "$out/stage/backbone-rebuild.engine" "$cap/backbone-rebuild-build-report.txt" "$bb_engine"
run backbone-run "$out/trt/backbone_trt" run "$bb_engine" "$bb_sha" "$images" "$inputs_sha" "$cap/backbone-run-report.txt" "$out/backbone-dump"
run backbone-compare "${ref[@]}" compare "$out/backbone-dump"
run backbone-compare-report cp "$root/export/backbone-compare.json" "$cap/backbone-compare.json"

# F03 model engine (t01-f03-fp32/1).
model_engine=$out/stage/model.engine
run model-build "$out/trt/model_trt" build "$onnx" "$onnx_sha" "$bundle" "$bundle_sha" "$model_engine" "$cap/model-build-report.txt"
model_sha=$(sha256sum "$model_engine" | cut -d' ' -f1)
run model-rebuild-record bash -c '"$1" build "$2" "$3" "$4" "$5" "$6" "$7" >/dev/null && sha256sum "$6" "$8" && { cmp -s "$6" "$8" && echo "rebuild: identical plan bytes" || echo "rebuild: plan bytes differ (recorded, not a criterion)"; }' \
  _ "$out/trt/model_trt" "$onnx" "$onnx_sha" "$bundle" "$bundle_sha" "$out/stage/model-rebuild.engine" "$cap/model-rebuild-build-report.txt" "$model_engine"
model_run=(env CUDA_MODULE_LOADING=LAZY "$out/trt/model_trt" run "$model_engine" "$model_sha" "$bundle" "$bundle_sha" "$images" "$inputs_sha" "$bbref" "$reference_sha")
run model-run "${model_run[@]}" "$cap/model-run-report.txt" --labels "$labels" --dump "$out/model-dump"
run model-run-unregistered "$out/trt/head_trt" run-unregistered "$model_engine"

# Two-stage chain (backbone engine -> head engine) under the S04 policy, and the diagnostics.
run chain-features bash -c 'cat "$1"/input-{0,1,2,3,4,5}.f32 > "$2" && sha256sum "$2"' _ "$out/backbone-dump" "$out/stage/backbone-features.f32"
chain_sha=$(sha256sum "$out/stage/backbone-features.f32" | cut -d' ' -f1)
run chain-head-run env CUDA_MODULE_LOADING=LAZY "$out/trt/head_trt" run "$head_engine" "$head_engine_sha" "$bundle" "$bundle_sha" \
  "$cap/chain-head-run-report.txt" --features "$out/stage/backbone-features.f32" "$chain_sha" --dump "$out/chain-dump"
run diagnostics "${ref[@]}" diagnose "$out/model-dump" "$out/chain-dump"
run diagnostics-report cp "$root/export/f03-diagnostics.json" "$cap/f03-diagnostics.json"

# F03 sanitizers (BACKBONE_POLICY §7): first real input, failure sequence and graph replay.
run sanitizer-model-memcheck "$sanitizer" --tool memcheck --report-api-errors explicit --error-exitcode 99 "${model_run[@]:2}" \
  "$cap/sanitizer-model-memcheck-report.txt" --inputs 1
run sanitizer-model-initcheck "$sanitizer" --tool initcheck --report-api-errors explicit --error-exitcode 99 "${model_run[@]:2}" \
  "$cap/sanitizer-model-initcheck-report.txt" --inputs 1
printf '%s\n' 'F03 memcheck/initcheck: model_trt run --inputs 1 (MOT17-02-SDP/1, its rerun, the failure sequence and graph replay)' \
  'F03 and head racecheck/synccheck: UNEXECUTED (as in S04: TensorRT instrumentation cost; WSL crash during S04 development, cause undetermined)' \
  'head and S03 suites sanitizers: not rerun here; kernels, plugins and frame session unchanged since S04 tested source (gpu-code-unchanged-since-s04); the T01 head memcheck/initcheck are in evidence/t01/2026-10-30/native-head' \
  > "$cap/sanitizer-skips.txt"

# Manifest (schema 3): declaration and file integrity in the staging directory.
cp "$src/manifests/t01/yolo26m-t1.json" "$src/docs/BACKBONE_POLICY.md" "$src/docs/HEAD_POLICY.md" "$out/stage/"
ln -s "$teacher" "$out/stage/epoch_0012.ckpt"
ln -s "$checkpoint" "$out/stage/epoch_0001.ckpt"
ln -s "$onnx" "$out/stage/yolo26m-backbone-t01.onnx"
run manifest-validate "$out/cpu/model_manifest" validate "$out/stage/yolo26m-t1.json" "$cap/yolo26m-t1.canonical.json"
run manifest-verify "$out/cpu/model_manifest" verify "$out/stage/yolo26m-t1.json"
run binary-hashes sha256sum "$out/cpu/head_weights" "$out/cpu/head_reference_tests" "$out/cpu/model_manifest" "$out/trt/head_trt" \
  "$out/trt/backbone_trt" "$out/trt/model_trt" "$out/trt/libsaccade_model_trt.a" "$out/trt/libsaccade_head_trt.a" "$out/trt/libsaccade_scan_trt.a" \
  "$out/trt/libsaccade_scan_trt_frame.a" "$head_engine" "$bb_engine" "$model_engine" "$bundle" "$images" "$bbref" \
  "$out"/backbone-dump/*.f32 "$out"/model-dump/*.f32 "$out"/chain-dump/*.f32
run build-source-identity bash -c '
  set -euo pipefail
  rev=$1; shift
  for report in "$@"; do
    grep -Eq "(^|[[:space:]])build_commit=$rev$" "$report"
    grep -Eq "(^|[[:space:]])build_dirty=false$" "$report"
    if grep -q "build_dirty=true" "$report"; then exit 1; fi
    echo "$report: tested commit and build_dirty=false verified"
  done
' _ "$rev" "$cap/backbone-build-report.txt" "$cap/backbone-run-report.txt" "$cap/model-build-report.txt" "$cap/model-run-report.txt" \
  "$cap/chain-head-run-report.txt" "$cap/sanitizer-model-memcheck-report.txt" "$cap/sanitizer-model-initcheck-report.txt" "$cap/head-suite-report.tsv"
run working-tree-final git status --porcelain=v1
echo "capture complete: $out"

#!/bin/sh
# New downloads only. No Python, checkpoint deserialization, or old project.
# Invoke from the repository root. Override OUTPUT_DIR to preserve this capture.
set -eu
output_dir=${OUTPUT_DIR:-build-artifact-replay/evidence}
work_dir=build-artifact-inspector
artifact_dir=${ARTIFACT_DIR:-/home/ray/.cache/saccade-v1/p01-a/artifact}
mkdir -p "$output_dir" "$work_dir" "$artifact_dir"
run() {
    name=$1
    shift
    printf '%s\n' "$*" > "$output_dir/$name.command"
    set +e
    "$@" > "$output_dir/$name.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "$output_dir/$name.exit"
    if [ "$status" -ne 0 ]; then
        cat "$output_dir/$name.log" >&2
        exit "$status"
    fi
}
run download-model curl --fail --location --silent --show-error \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.onnx \
    -o "$artifact_dir/yolo26s.onnx"
run artifact-size stat -c '%s %n' "$artifact_dir/yolo26s.onnx"
run artifact-hash sha256sum "$artifact_dir/yolo26s.onnx"
printf '%s  %s\n' d26b65c432111eb95798cd2320603d4d75627605dbec6c6b7f98c499a80e7321 \
    "$artifact_dir/yolo26s.onnx" > "$output_dir/artifact.expected.sha256"
run artifact-hash-check sha256sum -c "$output_dir/artifact.expected.sha256"
run assets-release-metadata curl --fail --location --silent --show-error \
    https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0 \
    -o "$output_dir/assets-v8.4.0-release.json"
run assets-tag-metadata curl --fail --location --silent --show-error \
    https://api.github.com/repos/ultralytics/assets/git/ref/tags/v8.4.0 \
    -o "$output_dir/assets-v8.4.0-ref.json"
run onnx-tag-metadata curl --fail --location --silent --show-error \
    https://api.github.com/repos/onnx/onnx/git/ref/tags/v1.20.1 \
    -o "$output_dir/onnx-v1.20.1-ref.json"
run download-schema curl --fail --location --silent --show-error \
    https://raw.githubusercontent.com/onnx/onnx/d3f6b795aedb48eaecc881bf5e8f5dd6efbe25b3/onnx/onnx.proto \
    -o "$work_dir/onnx.proto"
run schema-hash sha256sum "$work_dir/onnx.proto"
run compiler-version c++ --version
run protoc-version protoc --version
run protobuf-version pkg-config --modversion protobuf
run protobuf-flags pkg-config --cflags --libs protobuf
run generate-schema protoc --cpp_out="$work_dir" -I "$work_dir" "$work_dir/onnx.proto"
# pkg-config supplies compiler flags from the native installed SDK, not data
# from a model or remote response. Intentional shell word splitting below.
run compile-inspector c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror \
    -I include -I "$work_dir" tools/onnx_contract_inspect.cpp src/file_io.cpp \
    "$work_dir/onnx.pb.cc" $(pkg-config --cflags --libs protobuf) \
    -o "$work_dir/onnx_contract_inspect"
run inspect "$work_dir/onnx_contract_inspect" "$artifact_dir/yolo26s.onnx"
run inspector-hash sha256sum "$work_dir/onnx_contract_inspect"
run inspector-source-hash sha256sum tools/onnx_contract_inspect.cpp src/file_io.cpp
run build-base git rev-parse HEAD
run build-status git status --porcelain
run inspection-reproduce cmp "$output_dir/inspect.log" evidence/p01-a/2026-10-28/artifact/onnx-inspection.tsv

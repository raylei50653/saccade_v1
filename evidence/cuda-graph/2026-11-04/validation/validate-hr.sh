#!/usr/bin/env bash
set -euo pipefail
root=/home/ray/developer/ai/saccade_v1
cd "$root"
out="$root/build/cuda-graph/hr-full"
mkdir -p "$out"
engine=/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine
sequence=/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP
for mode in ordinary graph serial; do
  args=()
  [[ $mode != ordinary ]] || args+=(--no-cuda-graph)
  [[ $mode != serial ]] || args+=(--serial)
  "$root/build/cuda-graph/track_sequence" "$engine" "$sequence" "$out/$mode.txt" \
    --profile hr-1280x768 --packets "$out/$mode.bin" "${args[@]}" > "$out/$mode.log" 2>&1
done
cmp "$out/ordinary.bin" "$out/graph.bin"
cmp "$out/ordinary.bin" "$out/serial.bin"
cmp "$out/ordinary.txt" "$out/graph.txt"
cmp "$out/ordinary.txt" "$out/serial.txt"
[[ $(sha256sum < "$out/graph.txt" | cut -d' ' -f1) == 1fb713f30444d2c6e3e9ae30d34621f3155b0967b3a52a9244f107e28547fba7 ]]
(cd "$out" && sha256sum ordinary.bin graph.bin serial.bin ordinary.txt graph.txt serial.txt) > "$out/outputs.sha256"
printf 'PASS: HR 654 frames ordinary/graph/serial complete packets and MOT identical; historical MOT hash preserved\n' > "$out/status.txt"

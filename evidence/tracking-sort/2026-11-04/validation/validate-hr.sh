#!/usr/bin/env bash
set -euo pipefail
cd /home/ray/developer/ai/saccade_v1
out=build/tracking-sort-evidence/hr
mkdir -p "$out"
engine=/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine
sequence=/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP
for mode in graph serial ordinary; do
    args=()
    [[ $mode != serial ]] || args+=(--serial)
    [[ $mode != ordinary ]] || args+=(--no-cuda-graph)
    build/tracking-sort/track_sequence "$engine" "$sequence" "$out/$mode.txt" --packets "$out/$mode.bin" \
        --profile hr-1280x768 "${args[@]}" > "$out/$mode.log" 2>&1
    cmp "$out/$mode.bin" build/cuda-graph/hr-full/final-graph.bin
    cmp "$out/$mode.txt" build/cuda-graph/hr-full/final-graph.txt
done
(cd "$out" && sha256sum *.bin *.txt) > "$out/outputs.sha256"
printf 'PASS: HR 654 frames graph/serial/ordinary match original packet and MOT bytes\n' > "$out/status.txt"

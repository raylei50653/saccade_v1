#!/usr/bin/env bash
set -euo pipefail
cd /home/ray/developer/ai/saccade_v1
engine=/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine
sequence=/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP
series=${1:-latency}
for run in 1-baseline 2-optimized 3-optimized 4-baseline; do
    kind=${run#*-}
    binary=build/tracking-sort/tracking_benchmark
    [[ $kind != baseline ]] || binary=build/tracking-sort-baseline/tracking_benchmark
    "$binary" "$engine" "$sequence" "build/tracking-sort-evidence/$series-$run" \
        --frames 120 --warmup 20 --rounds 1 --only graph > "build/tracking-sort-evidence/$series-$run.log" 2>&1
done

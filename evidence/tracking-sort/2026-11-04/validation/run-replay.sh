#!/usr/bin/env bash
set -euo pipefail
cd /home/ray/developer/ai/saccade_v1
records=/home/ray/developer/ai/saccade_v1-worktrees/f04-parallel/build/f04-eval2/scdr/MOT17-10-SDP.scdr
motion=/home/ray/developer/ai/saccade_v1-worktrees/m1-data/MOT17-10-SDP.scmo
series=${1:-replay}
for run in 1-baseline 2-optimized 3-optimized 4-baseline; do
    kind=${run#*-}
    binary=build/tracking-sort/tracking_batch_benchmark
    [[ $kind != baseline ]] || binary=build/tracking-sort-baseline/tracking_batch_benchmark
    "$binary" "$records" "$motion" "build/tracking-sort-evidence/$series-$run" \
        --frames 140 --iterations 8 --rounds 3 > "build/tracking-sort-evidence/$series-$run.log" 2>&1
done

#!/usr/bin/env bash
set -euo pipefail
cap=evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/development
run() {
 local name=$1 status=0; shift
 printf '%q ' "$@" > "$cap/$name.command"
 printf '\n' >> "$cap/$name.command"
 date -Is > "$cap/$name.started"
 printf '%s started\n' "$name"
 /usr/bin/time -v -o "$cap/$name.time" "$@" > "$cap/$name.log" 2>&1 || status=$?
 printf '%s\n' "$status" > "$cap/$name.exit"
 printf '%s exit=%s\n' "$name" "$status"
 return "$status"
}
finish() {
 local status=$?; trap - EXIT
 printf 'exit=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$cap/status.txt"
 cp build/development/CMakeCache.txt "$cap/CMakeCache.txt"
 cp build/development/CTestTestfile.cmake "$cap/CTestTestfile.cmake"
 if [[ -f build/development/Testing/Temporary/LastTest.log ]]; then cp build/development/Testing/Temporary/LastTest.log "$cap/LastTest.log"; fi
 if [[ -f build/development/Testing/Temporary/LastTestsFailed.log ]]; then cp build/development/Testing/Temporary/LastTestsFailed.log "$cap/LastTestsFailed.log"; fi
 exit "$status"
}
trap finish EXIT
git rev-parse HEAD > "$cap/source.txt"
run configure cmake -S . -B build/development \
 -DSACCADE_ENABLE_CUDA=ON -DSACCADE_ENABLE_PREPROCESS=ON -DSACCADE_ENABLE_POSTPROCESS=ON \
 -DSACCADE_ENABLE_MODEL_PRODUCER=ON -DSACCADE_ENABLE_TENSORRT=ON -DSACCADE_ENABLE_MODEL_TRT=ON \
 -DSACCADE_ENABLE_DETECTION_SESSION=ON -DSACCADE_ENABLE_FLOW=ON \
 -DSACCADE_DETECTION_ENGINE=/home/ray/developer/ai/saccade_v1/build/t01-f03-evidence/stage/model.engine \
 -DSACCADE_TRACKING_ENGINE=/home/ray/developer/ai/saccade_v1/build/t01-f03-evidence/stage/model.engine \
 -DSACCADE_DETECTION_REFERENCES=/home/ray/developer/ai/saccade_v1/build/preprocess-evidence/refs \
 -DSACCADE_TRACKING_SEQUENCES=/home/ray/developer/ai/saccade/datasets/MOT17/train \
 -DSACCADE_TRACKING_RECORDS=/home/ray/developer/ai/saccade_v1-worktrees/m1-data-v2
run build cmake --build build/development --parallel 8
run ctest ctest --test-dir build/development --output-on-failure --stop-on-failure --parallel 1

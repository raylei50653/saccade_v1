#!/usr/bin/env bash
set -euo pipefail
cd /home/ray/developer/ai/saccade_v1
export SACCADE_TRACKING_ENGINE="$PWD/build/t01-f03-evidence/stage/model.engine"
export SACCADE_DETECTION_REFERENCES="$PWD/build/preprocess-evidence/refs"
export SACCADE_TRACKING_SEQUENCES=/home/ray/developer/ai/saccade/datasets/MOT17/train
export SACCADE_TRACKING_RECORDS=/home/ray/developer/ai/saccade_v1-worktrees/m1-data
export PYTHONDONTWRITEBYTECODE=1
driver="$PWD/build/mot-full-20261104-driver"
finish() {
  status=$?
  printf 'exit=%s\nfinished=%s\n' "$status" "$(date -Is)" > "$driver/status.txt"
  exit "$status"
}
trap finish EXIT
printf 'started=%s\npid=%s\n' "$(date -Is)" "$$" > "$driver/started.txt"
printf 'Stage 1: current-source tracking acceptance\n'
bash scripts/run_tracking_session_acceptance.sh build/mot-full-20261104-acceptance --allow-dirty
printf 'Stage 2: full seven-sequence HR MOT and gates\n'
bash scripts/run_mot17_evaluation.sh build/mot-full-20261104-acceptance build/mot-full-20261104-evaluation --allow-dirty
printf 'Stage 3: post-M4 quality scoring\n'
bash scripts/score_mot17_evaluation.sh build/mot-full-20261104-evaluation --allow-dirty
printf 'All stages completed\n'

#!/usr/bin/env bash
# Clean-source gate for 4329ed7 (CUDA Graph + tracking sort + resource schedule):
# the F04 POSTPROCESS acceptance and mutations, the tracking session
# acceptance (CTest, HR MOT17-10, four sanitizers) and the HR 7-sequence run
# checked against the M4 SCDR/MOT hashes. Mirrors f04-accept2.sh.
set -uo pipefail
cd /home/ray/developer/ai/saccade_v1
lock=$HOME/.cache/saccade-v1/gpu.lock
logs=build/cg-accept-logs
mkdir -p "$logs"
export SACCADE_F04_F03_DUMPS=/home/ray/developer/ai/saccade_v1/build/t01-f03-evidence/model-dump
export SACCADE_TRACKING_ENGINE=/home/ray/developer/ai/saccade_v1/build/t01-f03-evidence/stage/model.engine
export SACCADE_DETECTION_REFERENCES=/home/ray/developer/ai/saccade_v1/build/preprocess-evidence/refs
export SACCADE_TRACKING_SEQUENCES=/home/ray/developer/ai/saccade/datasets/MOT17/train
export SACCADE_TRACKING_RECORDS=/home/ray/developer/ai/saccade_v1-worktrees/m1-data

step() {
  local name=$1 status=0
  shift
  printf '%q ' "$@" > "$logs/$name.command"; printf '\n' >> "$logs/$name.command"
  date -Is > "$logs/$name.started"
  echo "== $name started $(date -Is)"
  "$@" > "$logs/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$logs/$name.exit"
  date -Is > "$logs/$name.finished"
  echo "== $name exit=$status $(date -Is)"
  tail -n 5 "$logs/$name.log"
  [[ $status == 0 ]] || { echo "STOP at $name"; exit 1; }
}

git rev-parse HEAD > "$logs/revision.txt"
[[ $(git rev-parse HEAD) == 4329ed7* ]] || { echo 'HEAD is not 4329ed7'; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo 'dirty tree'; exit 1; }

step 1-cpu bash -c 'cmake -S . -B build/cpu -DCMAKE_BUILD_TYPE=Release && cmake --build build/cpu --parallel 8 && ctest --test-dir build/cpu --output-on-failure'
step 2-f04-acceptance flock "$lock" bash scripts/run_postprocess_acceptance.sh build/cg-f04-accept
step 3-mutations flock "$lock" bash scripts/check_postprocess_mutations.sh build/cg-f04-accept/cuda build/cg-f04-mutations
step 4-tracking-acceptance flock "$lock" bash scripts/run_tracking_session_acceptance.sh build/cg-tracking
step 5-mot17-run flock "$lock" bash scripts/run_mot17_evaluation.sh build/cg-tracking build/cg-eval
step 5-m4-hashes bash -c 'cd build/cg-eval && sha256sum -c /home/ray/developer/ai/saccade_v1/evidence/m4/2026-11-03-formal-cad5298/evaluation/outputs.sha256'
echo "ALL STEPS DONE $(date -Is)"

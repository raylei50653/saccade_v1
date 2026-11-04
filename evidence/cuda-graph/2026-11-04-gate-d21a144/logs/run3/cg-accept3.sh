#!/usr/bin/env bash
# Re-run of steps 4-5 of cg-accept.sh at e7027c3 (explicit API reporting for session memchecks).
# the F04 POSTPROCESS acceptance and mutations, the tracking session
# acceptance (CTest, HR MOT17-10, four sanitizers) and the HR 7-sequence run
# checked against the M4 SCDR/MOT hashes. Mirrors f04-accept2.sh.
set -uo pipefail
cd /home/ray/developer/ai/saccade_v1
lock=$HOME/.cache/saccade-v1/gpu.lock
logs=build/cg-accept3-logs
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
[[ $(git rev-parse HEAD) == e7027c3* ]] || { echo 'HEAD is not e7027c3'; exit 1; }
[[ -z $(git status --porcelain) ]] || { echo 'dirty tree'; exit 1; }

step 4-tracking-acceptance flock "$lock" bash scripts/run_tracking_session_acceptance.sh build/cg-tracking3
step 5-mot17-run flock "$lock" bash scripts/run_mot17_evaluation.sh build/cg-tracking3 build/cg-eval3
step 5-m4-hashes bash -c 'cd build/cg-eval3 && sha256sum -c /home/ray/developer/ai/saccade_v1/evidence/m4/2026-11-03-formal-cad5298/evaluation/outputs.sha256'
echo "ALL STEPS DONE $(date -Is)"

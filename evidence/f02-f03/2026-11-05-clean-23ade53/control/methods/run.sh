#!/usr/bin/env bash
set -euo pipefail
cd /home/ray/developer/ai/saccade_v1-worktrees/f02-f03-clean-20261105
export SACCADE_TRACKING_ENGINE=/home/ray/developer/ai/saccade_v1/build/t01-f03-evidence/stage/model.engine
export SACCADE_DETECTION_REFERENCES=/home/ray/developer/ai/saccade_v1/build/preprocess-evidence/refs
export SACCADE_TRACKING_SEQUENCES=/home/ray/developer/ai/saccade/datasets/MOT17/train
export SACCADE_TRACKING_RECORDS=/home/ray/developer/ai/saccade_v1-worktrees/m1-data
export SACCADE_ACCEPTANCE_TIMEOUT=900
control=$PWD/build/closeout-control
acceptance=$PWD/build/clean-acceptance
cap=$control/capture
mkdir "$cap"
[[ -z $(git status --porcelain) ]]
revision=$(git rev-parse HEAD)
printf '%s\n' "$revision" > "$cap/revision.txt"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status-before.txt"
git ls-files -z | sort -z | xargs -0 sha256sum > "$cap/source-files.sha256"
last=preflight
finish() {
  local status=$?
  trap - EXIT
  printf 'exit=%s\nlast=%s\nfinished=%s\n' "$status" "$last" "$(date -Is)" > "$cap/status.txt"
  exit "$status"
}
trap finish EXIT
run() {
  local name=$1 status=0
  shift
  [[ $(git rev-parse HEAD) == "$revision" && -z $(git status --porcelain) ]]
  last=$name
  printf '%q ' "$@" > "$cap/$name.command"
  printf '\n' >> "$cap/$name.command"
  date -Is > "$cap/$name.started"
  echo "$name: started"
  /usr/bin/time -v -o "$cap/$name.time" "$@" > "$cap/$name.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$cap/$name.exit"
  echo "$name: exit=$status"
  [[ $(git rev-parse HEAD) == "$revision" && -z $(git status --porcelain) ]]
  return "$status"
}
run formal-acceptance bash scripts/run_tracking_session_acceptance.sh build/clean-acceptance
[[ $(cat "$acceptance/capture/acceptance.txt") == PASS ]]
# Archive the accepted executables and libraries before any supplementary work.
run binary-archive bash -c 'set -euo pipefail; cd "$1/cuda"; find . -maxdepth 1 -type f \( -perm -u+x -o -name "*.a" \) -print0 | sort -z | tar --null -T - -czf "$2/accepted-binaries.tar.gz"' _ "$acceptance" "$control"
sha256sum "$control/accepted-binaries.tar.gz" > "$cap/accepted-binaries.sha256"
run hr-python-reference "$acceptance/cuda/preprocess_profile_tests" /home/ray/.cache/saccade-v1/hr-deploy/refs
# Supplement the original gate with the complete F02 and scan unit suites.
# No kernel filters or launch-count limits; initcheck excludes held callbacks
# according to the existing harness contract. No source or binary is rebuilt.
sanitizer=/opt/cuda/bin/compute-sanitizer
for tool in memcheck initcheck racecheck synccheck; do
  for test in resize_cuda_tests selective_scan_cuda_tests; do
    args=()
    [[ $tool != initcheck ]] || args+=(--no-held-stream)
    run "$test-$tool" timeout -k 10 600 "$sanitizer" --tool "$tool" --error-exitcode 99 "$acceptance/cuda/$test" "${args[@]}"
    if [[ $tool == racecheck ]]; then
      rg -Fxq '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' "$cap/$test-$tool.log"
    else
      rg -Fxq '========= ERROR SUMMARY: 0 errors' "$cap/$test-$tool.log"
    fi
    rg -q 'PASS|checks passed' "$cap/$test-$tool.log"
  done
done
run hr-seven-sequence bash scripts/run_mot17_evaluation.sh build/clean-acceptance build/clean-evaluation
run binary-check sha256sum -c "$acceptance/capture/binaries.sha256"
run source-check sha256sum -c "$cap/source-files.sha256"
git status --porcelain=v1 --untracked-files=all > "$cap/source-status-after.txt"
cmp "$cap/source-status-before.txt" "$cap/source-status-after.txt"
printf 'PASS: original v1-motion/1 gate, F02/scan component sanitizers, HR seven-sequence regression\n' > "$cap/acceptance.txt"

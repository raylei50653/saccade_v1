#!/usr/bin/env bash
# Shared paths for offline evaluation. Source this file; no legacy checkout is
# consulted. Dataset overrides name the directory containing sequence folders.
# These variables are also consumed by the scripts that source this file.
# shellcheck disable=SC2034
saccade_eval_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
saccade_eval_root=${SACCADE_EVAL_ROOT:-$HOME/.cache/saccade-v1/eval}
saccade_eval_python=${SACCADE_EVAL_PYTHON:-$saccade_eval_root/venv/bin/python}
saccade_eval_trackeval=$saccade_eval_repo/third_party/TrackEval
saccade_eval_datasets=${SACCADE_DATASET_ROOT:-$(dirname "$saccade_eval_repo")/dataset}
saccade_eval_mot17_gt=${SACCADE_MOT17_GT:-$saccade_eval_datasets/MOT17/train}
saccade_eval_pp22_gt=${SACCADE_PP22_GT:-$saccade_eval_datasets/PersonPath22/mot_test_kf}

saccade_eval_require() {
  command -v "$saccade_eval_python" >/dev/null || {
    echo "evaluation Python not found: $saccade_eval_python; run $saccade_eval_repo/scripts/setup_eval.sh or set SACCADE_EVAL_PYTHON" >&2
    return 2
  }
  [[ -f $saccade_eval_trackeval/trackeval/eval.py ]] || {
    echo "missing repository TrackEval: $saccade_eval_trackeval" >&2
    return 2
  }
  "$saccade_eval_python" -I -B -c 'import numpy; import scipy.optimize' >/dev/null 2>&1 || {
    echo "evaluation Python requires NumPy and SciPy: $saccade_eval_python; run $saccade_eval_repo/scripts/setup_eval.sh or set SACCADE_EVAL_PYTHON" >&2
    return 2
  }
}

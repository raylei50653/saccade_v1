#!/usr/bin/env bash
# Offline MOT17 train evaluation of v1 MOT files with TrackEval (HOTA, CLEAR,
# Identity). Measurement tool only; the v1 runtime does not use Python.
# usage: eval_mot17.sh RESULTS_DIR [SEQUENCE...]
#   RESULTS_DIR holds <SEQUENCE>.txt (e.g. MOT17-10-SDP.txt); without
#   sequences, every *.txt in RESULTS_DIR is evaluated. Writes
#   RESULTS_DIR/trackeval/ (combined and per-sequence tables).
set -euo pipefail
[[ $# -ge 1 && -d $1 ]] || { echo 'usage: eval_mot17.sh RESULTS_DIR [SEQUENCE...]' >&2; exit 2; }
results=$(cd "$1" && pwd)
shift
legacy=${SACCADE_LEGACY_ROOT:-/home/ray/developer/ai/saccade}
python=$legacy/.venv/bin/python
trackeval=$legacy/third_party/TrackEval
gt=$legacy/datasets/MOT17/train
if [[ $# -gt 0 ]]; then
  seqs=("$@")
else
  seqs=()
  for f in "$results"/*.txt; do seqs+=("$(basename "$f" .txt)"); done
fi
[[ ${#seqs[@]} -gt 0 ]] || { echo 'no result files' >&2; exit 2; }
out=$results/trackeval
rm -rf "$out"
mkdir -p "$out/trackers/v1"
for s in "${seqs[@]}"; do
  [[ -f $results/$s.txt && -f $gt/$s/gt/gt.txt ]] || { echo "missing result or GT for $s" >&2; exit 2; }
  cp "$results/$s.txt" "$out/trackers/v1/$s.txt"
done
cat > "$out/run.py" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
for name, alias in (('float', float), ('int', int), ('bool', bool)):
    if name not in np.__dict__:
        setattr(np, name, alias)  # TrackEval predates numpy 2
import trackeval
gt, out, seqs = sys.argv[2], sys.argv[3], sys.argv[4:]
evaluator = trackeval.Evaluator({**trackeval.Evaluator.get_default_eval_config(),
    'USE_PARALLEL': False, 'PRINT_CONFIG': False, 'PLOT_CURVES': False, 'TIME_PROGRESS': False})
dataset = trackeval.datasets.MotChallenge2DBox({**trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
    'GT_FOLDER': gt, 'SKIP_SPLIT_FOL': True, 'SEQ_INFO': {s: None for s in seqs},
    'TRACKERS_FOLDER': out + '/trackers', 'TRACKER_SUB_FOLDER': '', 'TRACKERS_TO_EVAL': ['v1'],
    'OUTPUT_FOLDER': out + '/results', 'BENCHMARK': 'MOT17', 'SPLIT_TO_EVAL': 'train', 'PRINT_CONFIG': False})
evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
summary = open(out + '/results/v1/pedestrian_summary.txt').read().splitlines()
row = dict(zip(summary[0].split(), summary[1].split()))
print('RESULT ' + ' '.join(f'{k}={row[k]}' for k in ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW') if k in row))
PY
"$python" "$out/run.py" "$trackeval" "$gt" "$out" "${seqs[@]}" > "$out/trackeval.log" 2>&1 \
  || { tail -20 "$out/trackeval.log" >&2; exit 1; }
grep '^RESULT ' "$out/trackeval.log" | cut -c8-

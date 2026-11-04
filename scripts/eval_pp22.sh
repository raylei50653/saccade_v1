#!/usr/bin/env bash
# PersonPath22 mot_test_kf evaluation: track at original FPS, then keep only
# MOT rows on frames present in GT (PLAN-2 section 2.1). TrackEval settings
# and final output line match eval_mot17.sh.
# usage: eval_pp22.sh RESULTS_DIR [SEQUENCE...]
#   RESULTS_DIR holds <SEQUENCE>.txt (e.g. uid_vid_00008.mp4.txt); without
#   sequences, every *.txt in RESULTS_DIR is evaluated. Writes
#   RESULTS_DIR/trackeval/ (combined and per-sequence tables).
set -euo pipefail
[[ $# -ge 1 && -d $1 ]] || { echo 'usage: eval_pp22.sh RESULTS_DIR [SEQUENCE...]' >&2; exit 2; }
results=$(cd "$1" && pwd)
shift
legacy=${SACCADE_LEGACY_ROOT:-/home/ray/developer/ai/saccade}
python=$legacy/.venv/bin/python
trackeval=$legacy/third_party/TrackEval
gt=$legacy/datasets/PersonPath22/mot_test_kf
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
done
cat > "$out/run.py" <<'PY'
import sys
import csv
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np
for name, alias in (('float', float), ('int', int), ('bool', bool)):
    if name not in np.__dict__:
        setattr(np, name, alias)  # TrackEval predates numpy 2
import trackeval
gt, out, results, seqs = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
# Keep the original frame numbers and the full seqLength. Empty intervening
# frames have neither GT nor predictions and contribute no detections.
with open(out + '/keyframe-filter.csv', 'w', newline='') as audit:
    writer = csv.writer(audit, lineterminator='\n')
    writer.writerow(('sequence', 'keyframes', 'input_rows', 'kept_rows'))
    for seq in seqs:
        with open(Path(gt) / seq / 'gt/gt.txt') as truth:
            keyframes = {int(line.split(',', 1)[0]) for line in truth if line.strip()}
        total = kept = 0
        with open(Path(results) / (seq + '.txt')) as source, open(Path(out) / 'trackers/v1' / (seq + '.txt'), 'w') as target:
            for line in source:
                if not line.strip():
                    continue
                total += 1
                if int(line.split(',', 1)[0]) in keyframes:
                    target.write(line)
                    kept += 1
        writer.writerow((seq, len(keyframes), total, kept))
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
"$python" "$out/run.py" "$trackeval" "$gt" "$out" "$results" "${seqs[@]}" > "$out/trackeval.log" 2>&1 \
  || { tail -20 "$out/trackeval.log" >&2; exit 1; }
grep '^RESULT ' "$out/trackeval.log" | cut -c8-

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
    'USE_PARALLEL': False, 'PRINT_CONFIG': False, 'PLOT_CURVES': False, 'TIME_PROGRESS': False,
    'LOG_ON_ERROR': out + '/error_log.txt'})
dataset = trackeval.datasets.MotChallenge2DBox({**trackeval.datasets.MotChallenge2DBox.get_default_dataset_config(),
    'GT_FOLDER': gt, 'SKIP_SPLIT_FOL': True, 'SEQ_INFO': {s: None for s in seqs},
    'TRACKERS_FOLDER': out + '/trackers', 'TRACKER_SUB_FOLDER': '', 'TRACKERS_TO_EVAL': ['v1'],
    'OUTPUT_FOLDER': out + '/results', 'BENCHMARK': 'MOT17', 'SPLIT_TO_EVAL': 'train', 'PRINT_CONFIG': False})
evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
summary = open(out + '/results/v1/pedestrian_summary.txt').read().splitlines()
row = dict(zip(summary[0].split(), summary[1].split()))
print('RESULT ' + ' '.join(f'{k}={row[k]}' for k in ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW') if k in row))

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

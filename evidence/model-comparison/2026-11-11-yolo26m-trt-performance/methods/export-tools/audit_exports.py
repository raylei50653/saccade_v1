"""Independent initializer-to-fixed-weight audit of completed ONNX graphs (CPU)."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import onnx
from onnx import numpy_helper
import torch
sys.path[:0] = ['/home/ray/.cache/saccade-v1/t01/src/src', '/home/ray/.cache/saccade-v1/t01/src']
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/ray/developer/ai/dataset/yolo26m.pt')
TEACHER = Path('/home/ray/developer/ai/saccade_v1_dev/build-hr/HR01-UA2-20261102/teacher/attempt-001/epoch_0004.ckpt')
EXPECTED_BASE = '401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7'
EXPECTED_TEACHER = '08be5abae55d26b762e262948ded7d3deb11295aa0509ee04ad3582a2c9774cd'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''): h.update(b)
    return h.hexdigest()


torch.set_num_threads(8)
assert sha(BASE) == EXPECTED_BASE and sha(TEACHER) == EXPECTED_TEACHER
rows = []
for arm in ('official', 'finetuned', 'official-default'):
    model = YOLO(str(BASE)).model.float().eval()
    if arm == 'finetuned':
        payload = torch.load(TEACHER, map_location='cpu', weights_only=False)
        state = {k.removeprefix('yolo_model.').replace('._orig_mod.', '.'): v
                 for k, v in payload['model'].items() if k.startswith('yolo_model.')}
        model.load_state_dict(state, strict=True)
        del payload, state
    hashes = defaultdict(list)
    for name, tensor in model.state_dict().items():
        a = tensor.detach().cpu().contiguous().numpy()
        hashes[(a.dtype.str, a.shape, hashlib.sha256(a.tobytes()).hexdigest())].append(name)
    path = ROOT / 'artifacts' / arm / 'model.onnx'
    graph = onnx.load(str(path))
    initializers = []
    for item in graph.graph.initializer:
        a = np.ascontiguousarray(numpy_helper.to_array(item))
        h = hashlib.sha256(a.tobytes()).hexdigest()
        matches = hashes.get((a.dtype.str, a.shape, h), [])
        initializers.append({'name': item.name, 'shape': list(a.shape), 'dtype': a.dtype.str,
                             'sha256': h, 'fixed_weight_keys': matches, 'matched': bool(matches)})
    unmatched = [x['name'] for x in initializers if not x['matched']]
    row = {'arm': arm, 'onnx_sha256': sha(path), 'status': 'PASS' if not unmatched else 'FAIL',
           'base_sha256': EXPECTED_BASE, 'teacher_sha256': EXPECTED_TEACHER if arm == 'finetuned' else None,
           'teacher_state': 'model primary EMA' if arm == 'finetuned' else None,
           'initializers': initializers, 'unmatched_initializers': unmatched,
           'scope': 'every ONNX initializer byte-matched against literal fixed checkpoint tensors; derived Constant operators are not weight initializers'}
    (path.parent / 'initializer-audit.json').write_text(json.dumps(row, indent=2, sort_keys=True) + '\n')
    rows.append({'arm': arm, 'status': row['status'], 'initializer_count': len(initializers), 'unmatched': unmatched})
    del model
assert sha(BASE) == EXPECTED_BASE and sha(TEACHER) == EXPECTED_TEACHER
print(json.dumps(rows))
assert all(x['status'] == 'PASS' for x in rows)

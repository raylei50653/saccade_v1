"""Backbone forward time with cuDNN TF32 on and off (cost of the fp32-strict policy).

    venv/bin/python tf32_timing.py TEACHER_CKPT

Run without NVIDIA_TF32_OVERRIDE in the environment (it would force FP32 in both rows).
"""

import sys
import time
from pathlib import Path

import torch

T01 = Path(__file__).resolve().parents[5] / "training" / "t01"
sys.path.insert(0, str(T01))
from t01_export import Backbone, load_teacher_yolo, setup  # noqa: E402

recipe, _, _ = setup(smoke=True)
m, _ = load_teacher_yolo(recipe, Path(sys.argv[1]))
net = Backbone(m.cuda()).eval()
for bs in (1, 16):
    x = torch.rand(bs, 3, 640, 640, device="cuda")
    for tf32 in (True, False):
        torch.backends.cudnn.allow_tf32 = tf32
        with torch.no_grad():
            for _ in range(5):
                net(x)
            torch.cuda.synchronize()
            t = time.perf_counter()
            n = 20
            for _ in range(n):
                net(x)
            torch.cuda.synchronize()
        print(f"batch {bs:2d} tf32={tf32!s:5s} {1000 * (time.perf_counter() - t) / n / bs:6.2f} ms/frame")

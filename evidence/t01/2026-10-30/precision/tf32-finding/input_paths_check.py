"""TF32 finding: RGB order, layout, normalization and compute precision across the T01 paths.

    venv/bin/python input_paths_check.py OLD_SMOKE_ROOT OUT.json

Run against the smoke root built before the precision controls (TF32 caches,
moved to ~/.cache/saccade-v1/t01/smoke-tf32-20261030). Decomposes the gap
between the training cache and strict-FP32 features, and measures how far
features move under wrong input conventions (BGR, no /255, HWC read as CHW,
no antialias, CPU fancy decode). Relative L2 over each whole feature map.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

T01 = Path(__file__).resolve().parents[5] / "training" / "t01"
sys.path.insert(0, str(T01))
from t01_common import read_json  # noqa: E402
from t01_export import VERIFY_FRAMES, Backbone, load_teacher_yolo, setup  # noqa: E402

old_root, out_path = Path(sys.argv[1]), Path(sys.argv[2])
recipe, _, _ = setup(smoke=True)
import t01_eval  # noqa: E402
from saccade.perception.temporal_yolo.dataset import gpu_decode_clip_batch  # noqa: E402
from torchvision.io import decode_jpeg  # noqa: E402

dev = torch.device("cuda")
root = Path(recipe["dataset"]["root"]) / "train"
paths = [root / s / "img1" / f"{f:06d}.jpg" for s, f in VERIFY_FRAMES]
report: dict = {"old_smoke_root": str(old_root), "frames": [f"{s}/{f}" for s, f in VERIFY_FRAMES]}

# 1. What the shared preprocessing function hands to the model.
x = gpu_decode_clip_batch([[p.read_bytes()] for p in paths], 640, dev)[:, 0]
x_eval = t01_eval.preprocess_frames(paths, dev)
raw = decode_jpeg([torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.uint8) for p in paths], device=dev)
report["input"] = {
    "shape": list(x.shape), "dtype": str(x.dtype), "stride": list(x.stride()), "contiguous": x.is_contiguous(),
    "min": float(x.min()), "max": float(x.max()),
    "eval_path_bitwise_equal": bool(torch.equal(x, x_eval)),
    "nvjpeg_raw": {"shape": list(raw[0].shape), "dtype": str(raw[0].dtype), "stride": list(raw[0].stride())},
    "raw_channel_means": [[round(float(c), 2) for c in r.float().mean(dim=(1, 2))] for r in raw],
}

# 2. Features through each variant, with the teacher the old cache was built from.
cache_dir = old_root / "cache_val" / "cache"
manifest = read_json(cache_dir / "manifest.json")
teacher_path = next(p for p in sorted(old_root.glob("teacher/attempt-*/run/epoch_*.ckpt"))
                    if __import__("t01_common").sha256_file(p) == manifest["teacher_checkpoint_sha256"])
report["teacher_checkpoint"] = str(teacher_path)
teacher, _ = load_teacher_yolo(recipe, teacher_path)
net = Backbone(teacher.to(dev)).eval()


def feats(inp: torch.Tensor, tf32: bool) -> list[np.ndarray]:
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with torch.no_grad():
        return [t.cpu().numpy() for t in net(inp)]


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))


no_aa = torch.stack([F.interpolate((r.float() / 255).unsqueeze(0), (640, 640), mode="bilinear",
                                   align_corners=False, antialias=False).squeeze(0) for r in raw])
cpu_dec = torch.stack([F.interpolate((decode_jpeg(torch.frombuffer(bytearray(p.read_bytes()), dtype=torch.uint8)).float()
                                      / 255).unsqueeze(0).to(dev), (640, 640), mode="bilinear",
                                     align_corners=False, antialias=True).squeeze(0) for p in paths])
resize_then_div = torch.stack([F.interpolate(r.float().unsqueeze(0), (640, 640), mode="bilinear",
                                             align_corners=False, antialias=True).squeeze(0) / 255 for r in raw])
nhwc_misread = x.permute(0, 2, 3, 1).contiguous().view(x.shape)  # HWC bytes read as CHW
report["input"]["resize_then_div_max_abs_vs_contract"] = float((resize_then_div - x).abs().max())

variants = {
    "contract_fp32_strict": (x, False),
    "contract_tf32_conv (as the old cache/GT1 ran)": (x, True),
    "resize_then_div255": (resize_then_div, False),
    "cpu_libjpeg_fancy_decode": (cpu_dec, False),
    "nvjpeg_no_antialias": (no_aa, False),
    "BGR_swapped": (x.flip(1), False),
    "not_divided_by_255": (x * 255, False),
    "HWC_bytes_read_as_CHW": (nhwc_misread, False),
}
rows = []
for i, (seq, fid) in enumerate(VERIFY_FRAMES):
    cached = torch.load(cache_dir / seq / f"{fid:06d}.pt", map_location="cpu", weights_only=True)
    c = [cached[k].float().numpy()[None] for k in ("p3", "p4", "p5")]
    ref = feats(x[i:i + 1], False)
    row = {"frame": f"{seq}/{fid}",
           "ref_rounded_fp16_vs_ref": [rel(r.astype(np.float16).astype(np.float32), r) for r in ref]}
    for name, (inp, tf32) in variants.items():
        f = feats(inp[i:i + 1], tf32)
        row[name] = {"vs_fp32_reference": [rel(a, b) for a, b in zip(f, ref)],
                     "vs_training_cache": [rel(a, b) for a, b in zip(f, c)]}
    tf = feats(x[i:i + 1], True)
    row["tf32_rounded_fp16_vs_cache_max_abs"] = [float(np.abs(a.astype(np.float16).astype(np.float32) - b).max())
                                                 for a, b in zip(tf, c)]
    rows.append(row)
report["per_frame"] = rows
out_path.write_text(json.dumps(report, indent=1) + "\n")

names = list(variants)
print(f"{'variant':46s} {'vs FP32 ref (p3/p4/p5, worst frame)':>36s}   {'vs old training cache':>28s}")
w = np.max([r["ref_rounded_fp16_vs_ref"] for r in rows], axis=0)
print(f"{'FP32 ref rounded to fp16':46s} {' / '.join(f'{v:.2e}' for v in w):>36s}")
for n in names:
    a = np.max([r[n]["vs_fp32_reference"] for r in rows], axis=0)
    b = np.max([r[n]["vs_training_cache"] for r in rows], axis=0)
    print(f"{n:46s} {' / '.join(f'{v:.2e}' for v in a):>36s}   {' / '.join(f'{v:.2e}' for v in b):>28s}")
print("TF32 result rounded to fp16 vs old cache, max abs:", [r["tf32_rounded_fp16_vs_cache_max_abs"] for r in rows])

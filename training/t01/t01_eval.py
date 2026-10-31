"""T01 detector validation / evaluation (policy t01-selection/1 in recipe.json).

Students run T=1 in strict FP32 (policy fp32-strict of t01_precision: TF32 off
for cuDNN and cuBLAS, read back and probed in this process; deterministic cuDNN)
on cached teacher features; teacher diagnostics run the gated teacher on decoded frames.
Both use one decode: anchors (x+0.5, y+0.5)*stride, raw ltrb distances,
score = max_c sigmoid, class 0 only, top-300, NMS 0.5, greedy IoU-0.5 match.
"""

from __future__ import annotations

import configparser
import csv
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision

import t01_precision

CONF_FLOOR = 0.001
THRESHOLDS = (0.001, 0.10, 0.25)
MAX_DET = 300
NMS_IOU = 0.5
MATCH_IOU = 0.5
STRIDES = (8, 16, 32)
MIN_SIDE_BINS = (("min_lt4", 0.0, 4.0), ("min_4to8", 4.0, 8.0), ("min_8to16", 8.0, 16.0), ("min_ge16", 16.0, float("inf")))
HEIGHT_BINS = (("h_lt32", 0.0, 32.0), ("h_32to64", 32.0, 64.0), ("h_64to128", 64.0, 128.0), ("h_ge128", 128.0, float("inf")))
SMALL_BINS = ("min_4to8", "min_8to16")
EVAL_BATCH = 8


def strict_fp32() -> dict[str, Any]:
    """Apply fp32-strict in this process; returns the checked readback (raises on mismatch)."""
    t01_precision.apply("fp32-strict")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return t01_precision.checked_readback("fp32-strict")


def use_source(source_root: Path) -> None:
    sys.path[:0] = [str(source_root), str(source_root / "src")]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _bin(value: float, bins: tuple[tuple[str, float, float], ...]) -> str:
    return next(name for name, lo, hi in bins if lo <= value < hi)


def load_sequence(data_root: Path, seq: str) -> dict[str, Any]:
    seq_dir = data_root / "train" / seq
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini")
    width, height = int(parser["Sequence"]["imWidth"]), int(parser["Sequence"]["imHeight"])
    frames = sorted((seq_dir / "img1").glob("*.jpg"))
    gt: dict[int, list[list[float]]] = {}
    with (seq_dir / "gt" / "gt.txt").open() as handle:
        for row in csv.reader(handle):
            if int(row[6]) != 1 or int(row[7]) != 1 or float(row[8]) < 0.1:
                continue
            x, y, w, h = (float(v) for v in row[2:6])
            gt.setdefault(int(row[0]), []).append([x, y, x + w, y + h])
    return {"seq": seq, "width": width, "height": height, "frames": frames, "gt": gt}


def preprocess_frames(paths: list[Path], device: torch.device) -> torch.Tensor:
    """The T01 input contract, through the same function the trainers use: (N,3,640,640) float in [0,1]."""
    from saccade.perception.temporal_yolo.dataset import gpu_decode_clip_batch  # noqa: PLC0415

    return gpu_decode_clip_batch([[p.read_bytes()] for p in paths], 640, device)[:, 0]


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def _anchors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    points, strides = [], []
    for s in STRIDES:
        n = 640 // s
        ys, xs = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")
        points.append(torch.stack([xs + 0.5, ys + 0.5], -1).reshape(-1, 2))
        strides.append(torch.full((n * n,), float(s)))
    return torch.cat(points).to(device), torch.cat(strides).to(device)


def decode(cls_flat: torch.Tensor, reg_flat: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """cls (B,80,8400) logits, reg (B,4,8400) raw ltrb -> per-image detections in 640 space."""
    anchors, strides = _anchors(cls_flat.device)
    out = []
    for b in range(cls_flat.shape[0]):
        reg = reg_flat[b].T  # (8400, 4)
        x1y1 = (anchors - reg[:, :2]) * strides[:, None]
        x2y2 = (anchors + reg[:, 2:]) * strides[:, None]
        boxes = torch.cat([x1y1, x2y2], 1)
        scores, classes = cls_flat[b].sigmoid().max(dim=0)
        keep = scores >= CONF_FLOOR
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        if scores.numel() > MAX_DET:
            top = scores.topk(MAX_DET).indices
            boxes, scores, classes = boxes[top], scores[top], classes[top]
        n_pre = int(scores.numel())
        invalid = int(((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).sum())
        person = classes == 0
        boxes, scores = boxes[person], scores[person]
        if boxes.numel():
            k = torchvision.ops.nms(boxes, scores, NMS_IOU)
            boxes, scores = boxes[k], scores[k]
        out.append({"boxes": boxes, "scores": scores, "n_topk": n_pre, "n_person": int(person.sum()),
                    "invalid_topk": invalid, "class_hist": torch.bincount(classes, minlength=80)})
    return out


def flatten_levels(cls: list[torch.Tensor], reg: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([c.flatten(2) for c in cls], 2), torch.cat([r.flatten(2) for r in reg], 2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class Accumulator:
    def __init__(self) -> None:
        self.records: list[tuple[float, int]] = []  # (score, tp) over all frames
        self.n_gt = 0
        self.bins: dict[float, dict[str, list[int]]] = {t: {} for t in THRESHOLDS}
        self.preds_at: dict[float, int] = {t: 0 for t in THRESHOLDS}
        self.empty_frames: dict[float, int] = {t: 0 for t in THRESHOLDS}
        self.frames = 0
        self.frames_without_gt = 0
        self.nonfinite = 0
        self.invalid_topk = 0
        self.n_topk = 0
        self.n_person = 0
        self.class_hist = torch.zeros(80, dtype=torch.long)
        self.score_min = float("inf")
        self.score_max = float("-inf")
        self.digest = hashlib.sha256()

    def add(self, det: dict[str, torch.Tensor], gt: list[list[float]], width: int, height: int) -> None:
        sx, sy = 640.0 / width, 640.0 / height
        boxes = det["boxes"].detach().cpu().double()
        scores = det["scores"].detach().cpu().double()
        boxes[:, [0, 2]] /= sx
        boxes[:, [1, 3]] /= sy
        self.digest.update(boxes.numpy().tobytes())
        self.digest.update(scores.numpy().tobytes())
        self.frames += 1
        self.invalid_topk += det["invalid_topk"]
        self.n_topk += det["n_topk"]
        self.n_person += det["n_person"]
        self.class_hist += det["class_hist"].cpu()
        if scores.numel():
            self.score_min = min(self.score_min, float(scores.min()))
            self.score_max = max(self.score_max, float(scores.max()))
        gt_t = torch.tensor(gt, dtype=torch.float64).reshape(-1, 4)
        if gt_t.shape[0] == 0:
            self.frames_without_gt += 1
        self.n_gt += gt_t.shape[0]
        order = scores.argsort(descending=True)
        boxes, scores = boxes[order], scores[order]
        matched_by = torch.full((gt_t.shape[0],), -1, dtype=torch.long)
        if gt_t.shape[0] and boxes.shape[0]:
            ious = torchvision.ops.box_iou(boxes, gt_t)
            used = torch.zeros(gt_t.shape[0], dtype=torch.bool)
            for i in range(boxes.shape[0]):
                avail = ious[i].masked_fill(used, -1.0)
                best, j = avail.max(0)
                if float(best) >= MATCH_IOU:
                    used[j] = True
                    matched_by[j] = i
        tp = torch.zeros(boxes.shape[0], dtype=torch.long)
        tp[matched_by[matched_by >= 0]] = 1
        self.records.extend(zip(scores.tolist(), tp.tolist()))
        for t in THRESHOLDS:
            n_above = int((scores >= t).sum())
            self.preds_at[t] += n_above
            if n_above == 0:
                self.empty_frames[t] += 1
            for g in range(gt_t.shape[0]):
                w, h = float(gt_t[g, 2] - gt_t[g, 0]), float(gt_t[g, 3] - gt_t[g, 1])
                hit = int(matched_by[g] >= 0 and float(scores[matched_by[g]]) >= t)
                for name in ("all", _bin(min(w * sx, h * sy), MIN_SIDE_BINS), _bin(h, HEIGHT_BINS)):
                    c = self.bins[t].setdefault(name, [0, 0])
                    c[0] += 1
                    c[1] += hit

    def result(self) -> dict[str, Any]:
        recs = sorted(self.records, key=lambda r: -r[0])
        tp = np.array([r[1] for r in recs], dtype=np.float64)
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        recall = ctp / max(self.n_gt, 1)
        precision = ctp / np.maximum(ctp + cfp, 1e-12)
        envelope = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
        ap = 0.0
        for r in np.linspace(0, 1, 101):
            idx = np.searchsorted(recall, r, side="left")
            ap += float(envelope[idx]) if idx < len(envelope) else 0.0
        ap /= 101
        thresholds = {}
        for t in THRESHOLDS:
            bins = {k: {"gt": v[0], "matched": v[1], "fn": v[0] - v[1], "recall": v[1] / v[0] if v[0] else None}
                    for k, v in sorted(self.bins[t].items())}
            matched = bins.get("all", {"matched": 0})["matched"]
            thresholds[f"{t:g}"] = {"predictions": self.preds_at[t], "tp": matched, "fp": self.preds_at[t] - matched,
                                    "fn": self.n_gt - matched, "empty_frames": self.empty_frames[t], "bins": bins}
        small = [self.bins[0.25].get(k, [0, 0]) for k in SMALL_BINS]
        small_gt, small_hit = sum(c[0] for c in small), sum(c[1] for c in small)
        return {
            "frames": self.frames,
            "frames_without_gt": self.frames_without_gt,
            "gt": self.n_gt,
            "ap50": ap,
            "small_recall_at_0.25": small_hit / small_gt if small_gt else None,
            "thresholds": thresholds,
            "checks": {
                "nonfinite_raw_outputs": self.nonfinite,
                "topk_detections": self.n_topk,
                "topk_invalid_boxes": self.invalid_topk,
                "topk_person_fraction": self.n_person / self.n_topk if self.n_topk else None,
                "topk_top_classes": [[int(i), int(self.class_hist[i])] for i in self.class_hist.argsort(descending=True)[:5]],
                "kept_score_range": [self.score_min, self.score_max] if self.score_min != float("inf") else None,
            },
            "detections_sha256": self.digest.hexdigest(),
        }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_student(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    from saccade.perception.temporal_yolo.mamba_head import MambaDetectionHead  # noqa: PLC0415

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = state["mamba_args"]
    head = MambaDetectionHead(
        in_channels=tuple(a["in_channels"]), d_model=a["d_model"], d_state=a["d_state"], num_blocks=a["num_blocks"],
        num_classes=a["num_classes"], reg_max=a.get("reg_max", 1), head_depth=a.get("head_depth", 1),
        spatial_reduction=a["spatial_reduction"], use_pixel_shuffle=a["use_pixel_shuffle"],
        use_cross_scan=a["use_cross_scan"], use_hybrid_head=a.get("use_hybrid_head", False),
        use_temporal_mamba=a.get("use_temporal_mamba", False),
        use_temporal_attention=a.get("use_temporal_attention", False), per_channel_a=a.get("per_channel_a", False),
        scan_stop_grad=a.get("scan_stop_grad", False), legacy_n1_scan=a.get("legacy_n1_scan", False),
        reduction_variant=a.get("reduction_variant", "conv"),
    )
    head.load_state_dict(state["student"], strict=True)
    head.to(device).eval()
    info = {"epoch": state["epoch"], "mamba_args": {k: v for k, v in a.items() if not isinstance(v, (list, dict)) or k == "in_channels"},
            "t01_identity": state.get("t01_identity"), "args_seqs": state.get("args", {}).get("seqs"),
            "selection": state.get("selection"), "epoch_loss": state.get("epoch_loss"),
            "t01_compute_precision": state.get("t01_compute_precision")}
    return head, info


def evaluate_student(ckpt_path: Path, cache_dir: Path, data_root: Path, seqs: list[str], device: torch.device) -> dict[str, Any]:
    precision = {"start": strict_fp32()}
    head, info = build_student(ckpt_path, device)
    per_seq = {}
    total = Accumulator()
    for seq in seqs:
        data = load_sequence(data_root, seq)
        acc = Accumulator()
        fids = [int(p.stem) for p in data["frames"]]
        for start in range(0, len(fids), EVAL_BATCH):
            chunk = fids[start:start + EVAL_BATCH]
            cached = [torch.load(cache_dir / seq / f"{fid:06d}.pt", map_location="cpu", weights_only=True) for fid in chunk]
            feats = [torch.stack([c[s] for c in cached]).to(device).float() for s in ("p3", "p4", "p5")]
            with torch.no_grad():
                cls, reg = head(feats)
            cls_f, reg_f = flatten_levels(cls, reg)
            bad = int((~torch.isfinite(cls_f)).sum() + (~torch.isfinite(reg_f)).sum())
            dets = decode(cls_f, reg_f)
            for fid, det in zip(chunk, dets):
                for a in (acc, total):
                    a.add(det, data["gt"].get(fid, []), data["width"], data["height"])
            acc.nonfinite += bad
            total.nonfinite += bad
        per_seq[seq] = acc.result()
    precision["end"] = t01_precision.checked_readback("fp32-strict")
    return {"checkpoint": str(ckpt_path), "checkpoint_info": info, "sequences": per_seq, "total": total.result(),
            "compute_precision": precision}


def evaluate_teacher(ckpt_path: Path, yolo_weights: Path, data_root: Path, seqs: list[str], device: torch.device) -> dict[str, Any]:
    """Teacher diagnostics: ungated gated-YOLO, one2many raw outputs through the same decode."""
    from saccade.perception.temporal_yolo.yolo_gated_detector import GatedDetConfig, build_gated_yolo_detector  # noqa: PLC0415

    precision = {"start": strict_fp32()}
    model = build_gated_yolo_detector(str(yolo_weights), GatedDetConfig(freeze_backbone=True), device,
                                      weights_path=str(ckpt_path))
    model.eval()
    per_seq = {}
    total = Accumulator()
    for seq in seqs:
        data = load_sequence(data_root, seq)
        acc = Accumulator()
        for start in range(0, len(data["frames"]), EVAL_BATCH):
            paths = data["frames"][start:start + EVAL_BATCH]
            x = preprocess_frames(paths, device)
            with torch.no_grad():
                out = model(x, gate_input=None)
            many = out[1]["one2many"]
            cls_f, reg_f = many["scores"], many["boxes"]
            bad = int((~torch.isfinite(cls_f)).sum() + (~torch.isfinite(reg_f)).sum())
            for p, det in zip(paths, decode(cls_f, reg_f)):
                for a in (acc, total):
                    a.add(det, data["gt"].get(int(p.stem), []), data["width"], data["height"])
            acc.nonfinite += bad
            total.nonfinite += bad
        per_seq[seq] = acc.result()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = {"epoch": ckpt["epoch"], "epoch_loss": ckpt.get("epoch_loss"), "t01_identity": ckpt.get("t01_identity"),
            "training_sequences": ckpt.get("provenance", {}).get("training_sequences"),
            "t01_compute_precision": ckpt.get("t01_compute_precision")}
    precision["end"] = t01_precision.checked_readback("fp32-strict")
    return {"checkpoint": str(ckpt_path), "checkpoint_info": info, "sequences": per_seq, "total": total.result(),
            "compute_precision": precision}


def selection_key(result: dict[str, Any]) -> tuple[float, float]:
    t = result["total"]
    return (t["ap50"], t["small_recall_at_0.25"] or 0.0)


def eligible(result: dict[str, Any]) -> bool:
    return result["total"]["checks"]["nonfinite_raw_outputs"] == 0


def _check_decode_matches_teacher_head(model: Any, device: torch.device) -> float:
    """Diagnostic: one2many 'boxes'/'scores' equal Detect cv2/cv3 on the same features."""
    x = torch.rand(1, 3, 640, 640, device=device)
    with torch.no_grad():
        out = model(x, gate_input=None)
        feats = out[1]["one2many"]["feats"]
        head = model.yolo_model.model[-1]
        cls = [head.cv3[i](feats[i]) for i in range(3)]
        reg = [head.cv2[i](feats[i]) for i in range(3)]
    c, r = flatten_levels(cls, reg)
    return float(max((c - out[1]["one2many"]["scores"]).abs().max(), (r - out[1]["one2many"]["boxes"]).abs().max()))


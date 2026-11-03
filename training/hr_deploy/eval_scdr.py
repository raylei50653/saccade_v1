"""Evaluate kept SCDR v1 detections with T01 AP50, GT and matching rules (CPU)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "t01"))

import numpy as np
import torch

from t01_eval import Accumulator, load_sequence

FILE_HEADER = struct.Struct("<4sIIQQ")
FRAME_HEADER = struct.Struct("<QQ12I")
ROW = np.dtype([("box", "<f4", 4), ("score", "<f4"), ("class", "<u4"),
                ("candidate", "<u4"), ("flags", "<u4")])
FRAME_BYTES = 64 + 300 * 32


def evaluate(path: Path, seq_dir: Path) -> dict:
    seq_dir = seq_dir.resolve()
    data = load_sequence(seq_dir.parent.parent, seq_dir.name)
    acc = Accumulator()
    with path.open("rb") as stream:
        magic, version, frame_bytes, epoch, count = FILE_HEADER.unpack(stream.read(FILE_HEADER.size))
        if (magic, version, frame_bytes) != (b"SCDR", 1, FRAME_BYTES):
            raise ValueError("SCDR magic/version/size differs from v1")
        for _ in range(count):
            raw = stream.read(frame_bytes)
            if len(raw) != frame_bytes:
                raise ValueError("truncated SCDR frame")
            h = FRAME_HEADER.unpack_from(raw)
            rows = np.frombuffer(raw, dtype=ROW, count=h[3], offset=64)
            kept = rows[rows["flags"] == 1]
            # SCDR boxes are in original pixels; Accumulator consumes 640-stretch.
            boxes = torch.from_numpy(kept["box"].astype(np.float64))
            boxes[:, [0, 2]] *= 640.0 / data["width"]
            boxes[:, [1, 3]] *= 640.0 / data["height"]
            classes = torch.from_numpy(rows["class"].astype(np.int64))
            det = {"boxes": boxes, "scores": torch.from_numpy(kept["score"].copy()),
                   "n_topk": h[3] + h[11] + h[12], "n_person": h[3],
                   "invalid_topk": h[12], "class_hist": torch.bincount(classes, minlength=80)}
            acc.add(det, data["gt"].get(h[1] + 1, []), data["width"], data["height"])
        if stream.read(1):
            raise ValueError("trailing SCDR bytes")
    result = acc.result()
    del result["checks"]  # SCDR does not contain the raw-head diagnostics.
    point = result["thresholds"]["0.25"]
    result.update({"scdr": str(path.resolve()), "scdr_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "sequence": seq_dir.name, "epoch": epoch, "ap50_percent": 100 * result["ap50"],
                   "recall_at_0.25": point["tp"] / result["gt"], "fp_at_0.25": point["fp"],
                   "coordinates": "SCDR original pixels -> binary64 640 stretch -> T01 Accumulator"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scdr", type=Path)
    parser.add_argument("seq_dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = evaluate(args.scdr, args.seq_dir)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")


if __name__ == "__main__":
    main()

"""Build the versioned T01 split manifest (manifests/t01/split.json).

The split itself is fixed in recipe.json; this tool only records data identity,
counts, GT statistics and leakage checks for it. It reads labels and pixels but
never runs a model, so it cannot leak model results into the split choice.

    venv/bin/python training/t01/t01_split.py
"""

from __future__ import annotations

import configparser
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from t01_common import REPO, load_recipe, sha256_bytes, sha256_file, write_json

MIN_SIDE_BINS = (("min_lt4", 0.0, 4.0), ("min_4to8", 4.0, 8.0), ("min_8to16", 8.0, 16.0), ("min_ge16", 16.0, float("inf")))
HEIGHT_BINS = (("h_lt32", 0.0, 32.0), ("h_32to64", 32.0, 64.0), ("h_64to128", 64.0, 128.0), ("h_ge128", 128.0, float("inf")))
THUMB = 32
NEAR_DUP_NCC = 0.95


def _bin(value: float, bins: tuple[tuple[str, float, float], ...]) -> str:
    return next(name for name, lo, hi in bins if lo <= value < hi)


def _seqinfo(seq_dir: Path) -> dict[str, int]:
    parser = configparser.ConfigParser()
    parser.read(seq_dir / "seqinfo.ini")
    s = parser["Sequence"]
    return {k: int(s[k]) for k in ("imWidth", "imHeight", "frameRate", "seqLength")}


def _gt_stats(gt_file: Path, width: int, height: int) -> dict[str, Any]:
    counts = {name: 0 for name, _, _ in MIN_SIDE_BINS + HEIGHT_BINS}
    boxes = 0
    ids: set[int] = set()
    frames: set[int] = set()
    rows_total = 0
    with gt_file.open() as handle:
        for row in csv.reader(handle):
            rows_total += 1
            if int(row[6]) != 1 or int(row[7]) != 1 or float(row[8]) < 0.1:
                continue
            w, h = float(row[4]), float(row[5])
            boxes += 1
            ids.add(int(row[1]))
            frames.add(int(row[0]))
            counts[_bin(min(w * 640.0 / width, h * 640.0 / height), MIN_SIDE_BINS)] += 1
            counts[_bin(h, HEIGHT_BINS)] += 1
    return {"gt_rows": rows_total, "valid_boxes": boxes, "track_ids": len(ids), "frames_with_gt": len(frames), "bins": counts}


def _thumb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img.draft("L", (THUMB * 4, THUMB * 4))
        small = img.convert("L").resize((THUMB, THUMB), Image.BILINEAR)
    v = np.asarray(small, dtype=np.float32).reshape(-1)
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def main() -> None:
    recipe = load_recipe()
    root = Path(recipe["dataset"]["root"])
    roles = {seq: role for role in ("train", "val", "eval") for seq in recipe["split"][role]}
    if len(roles) != sum(len(v) for v in recipe["split"].values()):
        raise SystemExit("a sequence appears in more than one split role")

    sequences: dict[str, Any] = {}
    file_rows: list[str] = []
    thumbs: dict[str, np.ndarray] = {}
    variants: dict[str, Any] = {}
    for seq_dir in sorted((root / "train").glob("MOT17-*-SDP")):
        seq = seq_dir.name
        info = _seqinfo(seq_dir)
        frames = sorted((seq_dir / "img1").glob("*.jpg"))
        expected = [f"{i:06d}.jpg" for i in range(1, info["seqLength"] + 1)]
        if [p.name for p in frames] != expected:
            raise SystemExit(f"{seq}: img1 is not the contiguous 000001..{info['seqLength']:06d} list")
        with ThreadPoolExecutor(max_workers=16) as pool:
            digests = list(pool.map(sha256_file, frames))
            tvecs = list(pool.map(_thumb, frames))
        thumbs[seq] = np.stack(tvecs)
        lines = "".join(f"{d}  {p.name}\n" for p, d in zip(frames, digests))
        for p, d in zip(frames, digests):
            file_rows.append(f"{seq}\t{roles.get(seq, 'unused')}\timg1/{p.name}\t{d}\t{p.stat().st_size}")
        gt = seq_dir / "gt" / "gt.txt"
        file_rows.append(f"{seq}\t{roles.get(seq, 'unused')}\tgt/gt.txt\t{sha256_file(gt)}\t{gt.stat().st_size}")
        file_rows.append(f"{seq}\t{roles.get(seq, 'unused')}\tseqinfo.ini\t{sha256_file(seq_dir / 'seqinfo.ini')}\t{(seq_dir / 'seqinfo.ini').stat().st_size}")
        sequences[seq] = {
            "role": roles.get(seq, "unused"),
            "seqinfo": info,
            "frames": len(frames),
            "img1_list_sha256": sha256_bytes(lines.encode()),
            "img1_bytes": sum(p.stat().st_size for p in frames),
            "gt_sha256": sha256_file(gt),
            "seqinfo_sha256": sha256_file(seq_dir / "seqinfo.ini"),
            "gt_stats": _gt_stats(gt, info["imWidth"], info["imHeight"]),
        }
        base = seq[: -len("-SDP")]
        same = {}
        for det in ("DPM", "FRCNN"):
            other = root / "train" / f"{base}-{det}"
            if not other.is_dir():
                same[det] = "absent"
                continue
            other_frames = sorted((other / "img1").glob("*.jpg"))
            with ThreadPoolExecutor(max_workers=16) as pool:
                other_digests = list(pool.map(sha256_file, other_frames))
            same[det] = {
                "img1_identical": [p.name for p in other_frames] == [p.name for p in frames] and other_digests == digests,
                "gt_identical": sha256_file(other / "gt" / "gt.txt") == sequences[seq]["gt_sha256"],
            }
        variants[base] = same

    missing = sorted(set(roles) - set(sequences))
    if missing:
        raise SystemExit(f"split names unknown sequences: {missing}")

    # Near-duplicate check: normalized cross-correlation of 32x32 grayscale
    # thumbnails between every held-out frame and every frame of other roles.
    def best_match(seq: str, others: list[str]) -> dict[str, Any]:
        a = thumbs[seq]
        b = np.concatenate([thumbs[o] for o in others])
        owner = [(o, i + 1) for o in others for i in range(len(thumbs[o]))]
        ncc = a @ b.T
        flat = int(np.argmax(ncc))
        i, j = divmod(flat, ncc.shape[1])
        row_max = ncc.max(axis=1)
        return {
            "against": others,
            "max_ncc": float(ncc[i, j]),
            "max_pair": {"frame": i + 1, "other": owner[j][0], "other_frame": owner[j][1]},
            "frames_above_threshold": int((row_max >= NEAR_DUP_NCC).sum()),
        }

    train = recipe["split"]["train"]
    leakage = {
        "method": f"normalized cross-correlation of {THUMB}x{THUMB} grayscale thumbnails (stretch, mean removed); threshold {NEAR_DUP_NCC}",
        "val_vs_train": {s: best_match(s, train) for s in recipe["split"]["val"]},
        "eval_vs_train": {s: best_match(s, train) for s in recipe["split"]["eval"]},
        "eval_vs_val": {s: best_match(s, recipe["split"]["val"]) for s in recipe["split"]["eval"]},
        "within_sequence_reference": {s: float(np.median(np.sum(thumbs[s][1:] * thumbs[s][:-1], axis=1))) for s in sequences},
    }

    files_text = "#sequence\trole\tfile\tsha256\tbytes\n" + "\n".join(file_rows) + "\n"
    files_path = REPO / "manifests" / "t01" / "split-files.tsv"
    files_path.parent.mkdir(parents=True, exist_ok=True)
    files_path.write_text(files_text, encoding="utf-8")

    def total(role: str) -> int:
        return sum(v["frames"] for v in sequences.values() if v["role"] == role)

    manifest = {
        "schema": "saccade-v1-t01-split/1",
        "dataset": {
            "name": "MOT17 train (SDP variant directories)",
            "root": str(root),
            "provenance": "local copy; original download URL, archive hash and date are not recorded locally",
            "license": "not shipped with the local copy; MOTChallenge publishes MOT17 under CC BY-NC-SA 3.0 (not verified from a local artifact)",
            "detector_variants": "DPM/FRCNN/SDP directories share byte-identical img1 frames and gt.txt (see variants); only the SDP directories are read, and isolation is by base sequence",
            "test_split": "MOT17 test sequences have no public GT and are not used",
        },
        "roles": recipe["split"],
        "frames": {role: total(role) for role in ("train", "val", "eval")},
        "isolation": "sequence level: every frame of a val/eval sequence is excluded from teacher, cache, distill, GT1, T3 and T1 training; the seven sequences are distinct scenes per the MOT16/17 release",
        "limitations": [
            "one sequence each for validation and final evaluation; both come from the same dataset and camera era as training (held-out sequences, not a test-server result)",
            "validation MOT17-10 is a moving night-time camera; evaluation MOT17-02 is a static daytime camera; results do not cover other domains",
        ],
        "gt_filter": "mark==1, class==1 (pedestrian), visibility>=0.1; pedestrians map to detector class 0",
        "sequences": sequences,
        "variants": variants,
        "leakage": leakage,
        "files_tsv": "manifests/t01/split-files.tsv",
        "files_tsv_sha256": hashlib.sha256(files_text.encode()).hexdigest(),
    }
    write_json(REPO / "manifests" / "t01" / "split.json", manifest)
    print(f"split.json written: frames={manifest['frames']}")
    for k in ("val_vs_train", "eval_vs_train", "eval_vs_val"):
        for s, v in leakage[k].items():
            print(f"  {k} {s}: max_ncc={v['max_ncc']:.4f} pair={v['max_pair']} above={v['frames_above_threshold']}")


if __name__ == "__main__":
    main()

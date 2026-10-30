"""One-line progress of a T01 pipeline: current stage, epoch/batch, stage and total ETA.

    python3 t01_progress.py [--smoke]

Read-only: parses stage.json, attempt.json and the trainers' logs. The running
stage's ETA comes from its own measured rate; stages not yet started use the
budget of docs/T01_TRAINING.md §6 (fp32-strict cache/GT1 included).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

from t01_common import artifact_root, load_recipe

ORDER = ("teacher", "cache_train", "cache_val", "distill", "gt1", "t3", "t1", "final")
# Budget minutes (smoke-based, fp32-strict cache/GT1 included): the stage itself, then its
# per-epoch val evaluation in the runner (15 s per teacher checkpoint, 11 s per student checkpoint).
TRAIN_MIN = {"teacher": 46, "cache_train": 4, "cache_val": 1, "distill": 25, "gt1": 28, "t3": 12, "t1": 13, "final": 6}
EVAL_MIN = {"teacher": 8, "distill": 6, "gt1": 6, "t3": 3, "t1": 3}
BATCH = re.compile(r"epoch\s+(\d+)/(\d+)\s+\[\s*[\d.]+%\s+(\d+)/(\d+)\]")
EPOCH_DONE = re.compile(r"(?:^|\n)\s*[Ee]poch\s+(\d+)(?:/(\d+))?\s+loss=")
CACHE = re.compile(r"\[Precompute\] (\d+)/(\d+)")


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def epochs_total(recipe: dict, stage: str, attempt: dict) -> int | None:
    if attempt.get("smoke"):
        return attempt["smoke"]["epochs"]
    return recipe["stages"].get(stage, {}).get("epochs")


def stage_status(root: Path, recipe: dict, stage: str) -> tuple[str, str, float | None]:
    """(state, detail, remaining minutes or None if unknown)."""
    if stage == "final":
        if (root / "final" / "final_eval.json").is_file():
            return "done", "", 0.0
        return ("running" if (root / "final" / "opened.json").is_file() else "pending"), "final eval", None
    rec = read(root / stage / "stage.json")
    status = rec.get("status")
    if status is None:
        return "pending", "", None
    if status != "complete" and status != "running":
        return "failed", f"{stage} {rec.get('attempt')} {status}", None
    if status == "complete":
        n_val = len(list((root / stage / "val").glob("epoch_*.json")))
        if stage.startswith("cache_") or (root / stage / "selection.json").is_file():
            return "done", "", 0.0
        n_ep = len(rec.get("epochs_present", []))
        return "running", f"val evaluation {n_val}/{n_ep} checkpoints", max(n_ep - n_val, 0) * (15 if stage == "teacher" else 11) / 60
    attempt_dir = root / stage / rec["attempt"]
    attempt = read(attempt_dir / "attempt.json")
    log = (attempt_dir / "train.log").read_bytes().decode("utf-8", "replace").replace("\r", "\n")
    elapsed = time.time() - datetime.fromisoformat(attempt["started"]).timestamp()
    if stage.startswith("cache_"):
        m = CACHE.findall(log)
        if not m:
            return "running", "loading", None
        done, total = map(int, m[-1])
        return "running", f"frames {done}/{total}", elapsed / max(done, 1) * (total - done) / 60
    total = epochs_total(recipe, stage, attempt) or 0
    finished = [int(e) for e, _ in EPOCH_DONE.findall(log)]
    done_epochs = max(finished, default=0)
    batch = BATCH.findall(log)
    frac, detail = 0.0, f"epoch {done_epochs}/{total} done"
    if batch:
        ep, _, b, nb = map(int, batch[-1])
        if ep > done_epochs:
            frac = int(b) / int(nb)
            detail = f"epoch {ep}/{total} batch {b}/{nb}"
    resumed_from = 0
    if attempt.get("resume"):
        m = re.search(r"epoch_(\d+)\.ckpt", attempt["resume"])
        resumed_from = int(m.group(1)) if m else 0
    progress = done_epochs - resumed_from + frac
    if progress <= 0:
        return "running", detail + " (no rate yet)", None
    per_epoch = elapsed / progress
    remaining = per_epoch * (total - done_epochs - frac) / 60
    return "running", f"{detail} · {per_epoch / 60:.1f} min/epoch", remaining


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    recipe = load_recipe()
    root = artifact_root(recipe) / ("smoke" if a.smoke else "runs")
    parts, total_left, current = [], 0.0, None
    for i, stage in enumerate(ORDER):
        state, detail, left = stage_status(root, recipe, stage)
        if state == "failed":
            print(f"[{datetime.now():%H:%M}] FAILED: {detail}")
            return
        if state == "running" and current is None:
            current = f"{stage} ({i + 1}/{len(ORDER)}) {detail}"
            stage_left = left if left is not None else TRAIN_MIN[stage]
            if "val evaluation" not in detail:
                stage_left += EVAL_MIN.get(stage, 0)
            parts.append(f"本 stage（含 val 評估）剩約 {stage_left:.0f} min")
            total_left += stage_left
        elif state == "pending":
            total_left += TRAIN_MIN[stage] + EVAL_MIN.get(stage, 0)
    if current is None:
        done = all(stage_status(root, recipe, s)[0] == "done" for s in ORDER)
        print(f"[{datetime.now():%H:%M}] {'pipeline complete' if done else 'idle (no stage running)'}")
        return
    print(f"[{datetime.now():%H:%M}] {current} · {' · '.join(parts)} · 全部剩約 {total_left:.0f} min")


if __name__ == "__main__":
    main()

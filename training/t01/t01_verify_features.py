"""Verify that T01 training features are computed like the deployment features.

    venv-export/bin/python t01_verify_features.py --out REPORT.json [--smoke]
        [--negative-root OLD_SMOKE_ROOT]

Compares, for fixed frames of every cached sequence (first, 1/3, 2/3, last):

- FP32 implementation: ONNX Runtime (CPU, FP32) vs the PyTorch fp32-strict
  backbone, batch 1;
- batch invariance: PyTorch fp32-strict at batch 4 (GT1's clip batch) and 8
  (evaluation batch) vs batch 1; the cache path itself forwards batch 1;
- FP16 storage: the training cache (fp16 files) vs the PyTorch batch-1 FP32
  reference, with the pure rounding error and the bitwise agreement with the
  rounded reference reported separately;
- train vs deploy: the training cache vs ONNX Runtime;
- batched vs single nvJPEG decode of the same frames (bitwise).

With --negative-root (a smoke root built before the precision controls, TF32
caches): the runner and both cache-consuming trainers must reject its caches,
and its features must fail the FP16 storage tolerance against an FP32
recompute with the same teacher.

The tolerances below were fixed on 2026-10-30 before the first acceptance run
and are not changed after seeing results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import t01_precision
from t01_common import artifact_root, read_json, sha256_file, v1_source_state, write_json
from t01_export import Backbone, frames_from_paths, load_teacher_yolo, ort_session, rel_l2, setup, strict_fp32

TOLERANCES = {
    # Two FP32 implementations (different kernels / accumulation order): observed 0.5-1.8e-5 before T01 smoke.
    "fp32_implementation_rel_l2": 1e-4,
    "batch_invariance_rel_l2": 1e-4,
    # float16 round-to-nearest: |fl(x) - x| <= 2^-11 |x| for normal values, hence for the L2 ratio.
    "fp16_storage_rel_l2": 2.0**-11,
    "train_vs_deploy_rel_l2": 2.0**-11 + 1e-4,
}
BATCH_SIZES = (4, 8)
LEVELS = ("p3", "p4", "p5")


def picks(manifest: dict[str, Any]) -> list[tuple[str, int]]:
    out = []
    for seq in manifest["sequences"]:
        n = int(manifest["frame_counts"][seq])
        out += [(seq, fid) for fid in sorted({1, 1 + n // 3, 1 + 2 * n // 3, n})]
    return out


def worst(rows: list[dict[str, Any]], key: str) -> float:
    return max(max(r[key][lv] for lv in LEVELS) for r in rows)


def old_teacher(old_root: Path, manifest: dict[str, Any]) -> Path:
    """The old cache's teacher checkpoint inside the (moved) old root, found by its hash."""
    for path in sorted(old_root.glob("teacher/attempt-*/run/epoch_*.ckpt")):
        if sha256_file(path) == manifest["teacher_checkpoint_sha256"]:
            return path
    raise SystemExit(f"no checkpoint under {old_root} matches the old cache's teacher")


def negative_trainers(recipe: dict[str, Any], old_root: Path, scratch: Path) -> dict[str, Any]:
    """Run both cache-consuming trainers against the old TF32 cache; each must refuse it."""
    root = artifact_root(recipe)
    py = str(root / "venv" / "bin" / "python")
    here = Path(__file__).resolve().parent
    old_cache = old_root / "cache_train" / "cache"
    manifest = read_json(old_cache / "manifest.json")
    teacher = old_teacher(old_root, manifest)
    # Relocated copy of the old manifest (teacher path updated after the move; all else unchanged),
    # so the precision keys are the only difference the trainers can see.
    relocated = scratch / "old-cache-manifest-relocated"
    relocated.mkdir(parents=True)
    write_json(relocated / "manifest.json", {**manifest, "teacher_checkpoint_path": str(teacher)})
    common = ["--data-root", recipe["dataset"]["root"], "--yolo-weights", recipe["base_weights"]["path"],
              "--seed", str(recipe["seed"]), "--teacher-ckpt", str(teacher), "--seqs", "MOT17-09-SDP"]
    gt1 = sorted((old_root / "gt1").glob("attempt-*/run/epoch_0002.ckpt"))[-1]
    cases = {
        "distill (train_mamba_head.py)": ("scripts/train/temporal_yolo/train_mamba_head.py",
                                          common + ["--cache-dir", str(relocated), "--run-dir", str(scratch / "distill")]
                                          + recipe["stages"]["distill"]["args"] + ["--epochs", "1", "--warmup-epochs", "1",
                                                                                   "--max-batches-per-epoch", "1"],
                                          "Cache manifest mismatch"),
        "t3/t1 (train_mamba_gt.py)": ("scripts/train/temporal_yolo/train_mamba_gt.py",
                                      common + ["--cache-dir", str(old_cache), "--mamba-ckpt", str(gt1),
                                                "--run-dir", str(scratch / "t3"), "--holdout-seqs", "MOT17-10-SDP,MOT17-02-SDP"]
                                      + recipe["stages"]["t3"]["args"] + ["--epochs", "1", "--warmup-epochs", "1",
                                                                          "--max-batches-per-epoch", "1"],
                                      "Teacher cache compute precision/storage does not match T01"),
    }
    out = {}
    for name, (script, args, expected) in cases.items():
        env = {k: v for k, v in os.environ.items() if k != "NVIDIA_TF32_OVERRIDE"}
        env.update({"SACCADE_T01_PRECISION": "torch-default", "SACCADE_T01_GPU_DECODE_ONLY": "1",
                    "SACCADE_T01_PRECISION_RECORD": str(scratch / f"precision-{script.rsplit('/', 1)[1]}.json"),
                    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"})
        proc = subprocess.run([py, str(here / "t01_exec.py"), str(root / "src"), script, *args], cwd=root / "src",
                              env=env, capture_output=True, text=True)
        text = proc.stdout + proc.stderr
        # The exception message is the last matching line (earlier ones are traceback source lines).
        line = next((ln for ln in reversed(text.splitlines()) if expected in ln), None)
        refused = proc.returncode != 0 and line is not None
        if refused and "mismatch" in expected:  # only the precision keys may differ
            refused = all(k in line for k in ("compute_precision", "storage_dtype")) and "teacher_checkpoint" not in line
        out[name] = {"exit_code": proc.returncode, "refused_for_precision": refused, "message": line,
                     "tail": text.splitlines()[-3:] if not refused else None}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--negative-root", type=Path, default=None)
    a = parser.parse_args()
    started = time.time()
    recipe, runs, export = setup(a.smoke)
    precision = strict_fp32()
    import t01_run  # noqa: PLC0415  (after setup: same guard and source path)

    device = torch.device("cuda")
    sel = read_json(runs / "teacher" / "selection.json")
    teacher_path = Path(sel["checkpoint"])
    if sha256_file(teacher_path) != sel["sha256"]:
        raise SystemExit("teacher checkpoint differs from its selection record")
    exp = read_json(export / "backbone-export.json")
    if exp["teacher_sha256"] != sel["sha256"] or sha256_file(Path(exp["onnx"])) != exp["onnx_sha256"]:
        raise SystemExit("the ONNX export is not of the selected teacher")
    teacher, _ = load_teacher_yolo(recipe, teacher_path)
    net = Backbone(teacher.to(device)).eval()
    sess, ort_info = ort_session(Path(exp["onnx"]))
    data = Path(recipe["dataset"]["root"]) / "train"

    report: dict[str, Any] = {
        "schema": "saccade-v1-t01-feature-precision/1", "tolerances": TOLERANCES, "batch_sizes": list(BATCH_SIZES),
        "v1": v1_source_state(), "smoke": a.smoke, "teacher_checkpoint": str(teacher_path), "teacher_sha256": sel["sha256"],
        "onnx": exp["onnx"], "onnx_sha256": exp["onnx_sha256"], "onnxruntime": ort_info,
        "torch": torch.__version__, "compute_precision_start": precision, "caches": {}, "frames": [],
    }
    for name in ("cache_train", "cache_val"):
        cache = runs / name / "cache"
        manifest = t01_run.check_cache_manifest(cache)
        if manifest["teacher_checkpoint_sha256"] != sel["sha256"]:
            raise SystemExit(f"{cache} was built from another teacher")
        report["caches"][name] = {"path": str(cache), "manifest_sha256": sha256_file(cache / "manifest.json"),
                                  **{k: manifest[k] for k in ("storage_dtype", "dtype", "compute_precision", "sequences",
                                                              "total_frames")}}
        for seq, fid in picks(manifest):
            report["frames"].append({"cache": name, "seq": seq, "fid": fid, "path": str(cache / seq / f"{fid:06d}.pt")})

    paths = [data / f["seq"] / "img1" / f"{f['fid']:06d}.jpg" for f in report["frames"]]
    singles = torch.cat([frames_from_paths([p], device) for p in paths])
    batched = frames_from_paths(paths, device)
    report["decode_batch_vs_single_bitwise"] = bool(torch.equal(singles, batched))

    with torch.no_grad():
        ref = [[t.cpu().numpy() for t in net(singles[i:i + 1])] for i in range(len(paths))]
        by_batch = {}
        for bs in BATCH_SIZES:
            outs: list[list[np.ndarray]] = []
            for s in range(0, len(paths), bs):
                lv = [t.cpu().numpy() for t in net(singles[s:s + bs])]
                outs += [[lv[k][j:j + 1] for k in range(3)] for j in range(lv[0].shape[0])]
            by_batch[bs] = outs
    rows = []
    for i, f in enumerate(report["frames"]):
        ort_out = sess.run(["p3_feat", "p4_feat", "p5_feat"], {"images": singles[i:i + 1].cpu().numpy()})
        cached = torch.load(f["path"], map_location="cpu", weights_only=True)
        row: dict[str, Any] = {"frame": f"{f['seq']}/{f['fid']}", "cache": f["cache"]}
        for key in ("onnx_vs_pytorch", "cache_vs_pytorch", "cache_vs_onnx", "rounding_only", *(f"batch{b}_vs_batch1" for b in BATCH_SIZES),
                    *(f"cache_vs_batch{b}" for b in BATCH_SIZES), "cache_bitwise_equal_fraction", "cache_max_abs"):
            row[key] = {}
        for k, lv in enumerate(LEVELS):
            r = ref[i][k]
            c = cached[lv].float().numpy()[None]
            if cached[lv].dtype != torch.float16:
                raise SystemExit(f"{f['path']}: {lv} stored as {cached[lv].dtype}")
            row["onnx_vs_pytorch"][lv] = rel_l2(ort_out[k], r)
            row["cache_vs_pytorch"][lv] = rel_l2(c, r)
            row["cache_vs_onnx"][lv] = rel_l2(c, ort_out[k])
            rounded = r.astype(np.float16).astype(np.float32)
            row["rounding_only"][lv] = rel_l2(rounded, r)
            row["cache_bitwise_equal_fraction"][lv] = float(np.mean(c == rounded))
            row["cache_max_abs"][lv] = float(np.abs(c - r).max())
            for b in BATCH_SIZES:
                row[f"batch{b}_vs_batch1"][lv] = rel_l2(by_batch[b][i][k], r)
                row[f"cache_vs_batch{b}"][lv] = rel_l2(c, by_batch[b][i][k])
        rows.append(row)
    report["per_frame"] = rows

    checks = {
        "precision_readback_fp32_strict": {"pass": not t01_precision.check("fp32-strict", precision)},
        "onnxruntime_provider": {"used": ort_info["used"], "tf32": ort_info["tf32"],
                                 "pass": ort_info["used"][0] in ("CPUExecutionProvider", "CUDAExecutionProvider")},
        "decode_batch_vs_single_bitwise": {"pass": report["decode_batch_vs_single_bitwise"]},
        "fp32_implementation": {"worst": worst(rows, "onnx_vs_pytorch"), "tol": TOLERANCES["fp32_implementation_rel_l2"]},
        "fp16_storage": {"worst": worst(rows, "cache_vs_pytorch"), "tol": TOLERANCES["fp16_storage_rel_l2"],
                         "rounding_only_worst": worst(rows, "rounding_only"),
                         "min_bitwise_equal_fraction": min(min(r["cache_bitwise_equal_fraction"].values()) for r in rows)},
        "train_vs_deploy": {"worst": worst(rows, "cache_vs_onnx"), "tol": TOLERANCES["train_vs_deploy_rel_l2"]},
    }
    for b in BATCH_SIZES:
        checks[f"batch{b}_invariance"] = {"worst": worst(rows, f"batch{b}_vs_batch1"), "tol": TOLERANCES["batch_invariance_rel_l2"]}
        checks[f"cache_vs_batch{b}"] = {"worst": worst(rows, f"cache_vs_batch{b}"),
                                        "tol": TOLERANCES["fp16_storage_rel_l2"] + TOLERANCES["batch_invariance_rel_l2"]}
    for c in checks.values():
        if "tol" in c:
            c["pass"] = c["worst"] <= c["tol"]

    if a.negative_root is not None:
        neg: dict[str, Any] = {"root": str(a.negative_root)}
        old_val = a.negative_root / "cache_val" / "cache"
        try:
            t01_run.check_cache_manifest(old_val)
            neg["runner_rejects_old_cache"] = {"pass": False}
        except SystemExit as exc:
            neg["runner_rejects_old_cache"] = {"pass": True, "message": str(exc)}
        old_model, _ = load_teacher_yolo(recipe, old_teacher(a.negative_root, read_json(old_val / "manifest.json")))
        old_net = Backbone(old_model.to(device)).eval()
        old_rows = []
        val_frames = [f for f in report["frames"] if f["cache"] == "cache_val"]
        with torch.no_grad():
            for f in val_frames:
                x = frames_from_paths([data / f["seq"] / "img1" / f"{f['fid']:06d}.jpg"], device)
                r = [t.cpu().numpy() for t in old_net(x)]
                c = torch.load(old_val / f["seq"] / f"{f['fid']:06d}.pt", map_location="cpu", weights_only=True)
                old_rows.append({"frame": f"{f['seq']}/{f['fid']}",
                                 "cache_vs_pytorch": {lv: rel_l2(c[lv].float().numpy()[None], r[k]) for k, lv in enumerate(LEVELS)}})
        neg["old_tf32_cache_vs_fp32"] = {"rows": old_rows, "min": min(min(r["cache_vs_pytorch"].values()) for r in old_rows),
                                         "tol": TOLERANCES["fp16_storage_rel_l2"]}
        neg["old_tf32_cache_vs_fp32"]["pass"] = neg["old_tf32_cache_vs_fp32"]["min"] > TOLERANCES["fp16_storage_rel_l2"]
        with tempfile.TemporaryDirectory(dir=artifact_root(recipe)) as tmp:
            trainers = negative_trainers(recipe, a.negative_root, Path(tmp))
        neg["trainers_reject_old_cache"] = {"cases": trainers, "pass": all(v["refused_for_precision"] for v in trainers.values())}
        report["negative_controls"] = neg
        for name in ("runner_rejects_old_cache", "old_tf32_cache_vs_fp32", "trainers_reject_old_cache"):
            checks[f"negative:{name}"] = {"pass": neg[name]["pass"]}

    report["compute_precision_end"] = t01_precision.checked_readback("fp32-strict")
    report["checks"] = checks
    report["status"] = "pass" if all(c["pass"] for c in checks.values()) else "fail"
    report["seconds"] = round(time.time() - started, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, report)
    for name, c in checks.items():
        detail = f"worst {c['worst']:.3e} <= {c['tol']:.3e}" if "tol" in c else ""
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {name:34s} {detail}")
    print(f"{report['status'].upper()}: {a.out}")
    sys.exit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

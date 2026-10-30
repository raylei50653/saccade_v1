"""T01 orchestration: pinned old training source, environment record, stage runs.

    venv/bin/python training/t01/t01_run.py prepare-source [--write-lock]
    venv/bin/python training/t01/t01_run.py env
    venv/bin/python training/t01/t01_run.py stage NAME [--resume CKPT] [--smoke ...]
    venv/bin/python training/t01/t01_run.py select NAME

Formal stage runs require a clean saccade_v1 tree and a verified source tree;
every attempt writes stage.json (identity, command, exit, memory) before and
after the trainer process and keeps the full log.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from t01_common import (
    REPO,
    RECIPE_PATH,
    artifact_root,
    load_recipe,
    read_json,
    sha256_file,
    tree_digest,
    v1_source_state,
    write_json,
)

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable
LOW_MEMORY_GIB = 4.0
STUDENT_STAGES = ("distill", "gt1", "t3", "t1")
RESUME_KINDS = {
    "teacher": "exact resume: model, optimizer, scheduler, grad scaler, Python/torch/CUDA RNG and data-loader generator",
    "student": "optimizer/scheduler resume: model, optimizer, scheduler, epoch and best loss; RNG and data order are NOT restored",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
def source_dir(recipe: dict[str, Any]) -> Path:
    return artifact_root(recipe) / "src"


def prepare_source(write_lock: bool) -> None:
    recipe = load_recipe()
    old = recipe["old_source"]
    target = source_dir(recipe)
    if target.exists():
        raise SystemExit(f"{target} exists; remove it to re-export")
    archive = subprocess.run(
        ["git", "-C", old["local_clone"], "archive", old["revision"], "--", *old["paths"]],
        check=True, capture_output=True,
    ).stdout
    target.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    pristine_digest, _ = tree_digest(target)
    patch = REPO / old["patch"]
    subprocess.run(["patch", "-p1", "--forward", "--batch", "-i", str(patch)], cwd=target, check=True)
    patched_digest, rows = tree_digest(target)
    lock = {
        "schema": "saccade-v1-t01-source-lock/1",
        "revision": old["revision"],
        "paths": old["paths"],
        "archive_sha256": __import__("hashlib").sha256(archive).hexdigest(),
        "pristine_tree_sha256": pristine_digest,
        "patch": old["patch"],
        "patch_sha256": sha256_file(patch),
        "patched_tree_sha256": patched_digest,
        "files": len(rows),
    }
    lock_path = REPO / old["lock"]
    if write_lock:
        write_json(lock_path, lock)
        print(f"wrote {lock_path}")
    else:
        expected = read_json(lock_path)
        if expected != lock:
            raise SystemExit(f"source differs from {lock_path}:\nexpected {expected}\nactual   {lock}")
    write_json(target.parent / "source.json", {**lock, "exported_at": now(), "location": str(target)})
    print(f"source ready: {target} tree={patched_digest}")


def verify_source(recipe: dict[str, Any]) -> dict[str, Any]:
    lock = read_json(REPO / recipe["old_source"]["lock"])
    digest, _ = tree_digest(source_dir(recipe))
    if digest != lock["patched_tree_sha256"]:
        raise SystemExit(f"source tree {digest} differs from the lock {lock['patched_tree_sha256']}")
    if sha256_file(REPO / recipe["old_source"]["patch"]) != lock["patch_sha256"]:
        raise SystemExit("patch file differs from the lock")
    return lock


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def environment() -> dict[str, Any]:
    import torch  # noqa: PLC0415

    def run(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable: {exc}"

    meminfo = Path("/proc/meminfo").read_text().splitlines()
    return {
        "recorded_at": now(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")), ""),
        "cpu_count": os.cpu_count(),
        "mem_total_kib": int(next(l.split()[1] for l in meminfo if l.startswith("MemTotal"))),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "tf32": {"cuda_matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32},
        "nvidia_smi": run("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "packages": run("uv", "pip", "freeze", "--python", sys.executable).splitlines(),
    }


# ---------------------------------------------------------------------------
# Stage commands
# ---------------------------------------------------------------------------
def runs_dir(recipe: dict[str, Any], smoke: bool) -> Path:
    return artifact_root(recipe) / ("smoke" if smoke else "runs")


def stage_epochs(stage_dir: Path) -> dict[int, Path]:
    """Epoch checkpoints across attempts; a later attempt wins for an epoch."""
    found: dict[int, Path] = {}
    for attempt in sorted(stage_dir.glob("attempt-*")):
        for ckpt in sorted((attempt / "run").glob("epoch_*.ckpt")):
            found[int(ckpt.stem.split("_")[1])] = ckpt
    return found


def selected_checkpoint(recipe: dict[str, Any], stage: str, smoke: bool) -> Path:
    record = runs_dir(recipe, smoke) / stage / "selection.json"
    if not record.is_file():
        raise SystemExit(f"{stage} has no selection record ({record}); select it first")
    sel = read_json(record)
    path = Path(sel["checkpoint"])
    if sha256_file(path) != sel["sha256"]:
        raise SystemExit(f"{path} no longer matches its selection record")
    return path


def cache_dir(recipe: dict[str, Any], name: str, smoke: bool) -> Path:
    return runs_dir(recipe, smoke) / name / "cache"


def build_command(recipe: dict[str, Any], stage: str, run_dir: Path, smoke: dict[str, Any] | None,
                  resume: str | None) -> list[str]:
    spec = recipe["stages"][stage]
    data = recipe["dataset"]["root"]
    yolo = recipe["base_weights"]["path"]
    seed = str(recipe["seed"])
    train = ",".join(smoke["seqs"] if smoke else recipe["split"]["train"])
    holdout = ",".join(recipe["split"]["val"] + recipe["split"]["eval"])
    is_smoke = smoke is not None
    args = list(spec["args"])
    if smoke and "--epochs" in args:
        args = _override(args, "--epochs", str(smoke["epochs"]))
        if stage != "teacher":  # keep warmup inside the shortened smoke schedule
            args = _override(args, "--warmup-epochs", "1") if "--warmup-epochs" in args else args + ["--warmup-epochs", "1"]
    cmd = [PYTHON, str(HERE / "t01_exec.py"), str(source_dir(recipe)), spec["script"]]
    common = ["--data-root", data, "--yolo-weights", yolo, "--seed", seed]
    if stage == "teacher":
        cmd += common + ["--run-dir", str(run_dir), "--seqs", train, "--holdout-seqs", holdout] + args
        if resume:
            cmd += ["--resume", resume]
        return cmd
    teacher = str(selected_checkpoint(recipe, "teacher", is_smoke))
    if stage.startswith("cache_"):
        role = spec["sequences"]
        seqs = ",".join(smoke["cache_seqs"][role] if smoke else recipe["split"][role])
        return cmd + common + ["--teacher-ckpt", teacher, "--seqs", seqs, "--run-dir", str(run_dir),
                               "--precompute-dir", str(cache_dir(recipe, stage, is_smoke))] + args
    if stage == "distill":
        cmd += common + ["--teacher-ckpt", teacher, "--cache-dir", str(cache_dir(recipe, "cache_train", is_smoke)),
                         "--seqs", train, "--run-dir", str(run_dir)] + args
        if smoke:
            cmd += ["--max-batches-per-epoch", str(smoke["batches"])]
        if resume:
            cmd += ["--resume", resume]
        return cmd
    parent = str(selected_checkpoint(recipe, spec["parent"], is_smoke))
    cmd += common + ["--teacher-ckpt", teacher, "--mamba-ckpt", parent, "--run-dir", str(run_dir),
                     "--seqs", train, "--holdout-seqs", holdout] + args
    if spec.get("cache"):
        cmd += ["--cache-dir", str(cache_dir(recipe, "cache_train", is_smoke))]
    if smoke:
        cmd += ["--max-batches-per-epoch", str(smoke["batches"])]
    if resume:
        cmd += ["--resume", resume]
    return cmd


def _override(args: list[str], flag: str, value: str) -> list[str]:
    out = list(args)
    out[out.index(flag) + 1] = value
    return out


class Watchdog(threading.Thread):
    """Samples host memory and the trainer's process tree; stops it on low memory."""

    def __init__(self, proc: subprocess.Popen[bytes], log: Path, limit_gib: float):
        super().__init__(daemon=True)
        self.proc, self.log, self.limit = proc, log, limit_gib * 1024**3
        self.min_available = float("inf")
        self.max_rss = 0
        self.max_gpu_mib = 0
        self.tripped = False
        self.stop = threading.Event()

    def run(self) -> None:
        import psutil  # noqa: PLC0415

        last_gpu = 0.0
        with self.log.open("a") as out:
            while not self.stop.wait(2.0):
                avail = psutil.virtual_memory().available
                try:
                    root = psutil.Process(self.proc.pid)
                    rss = sum(p.memory_info().rss for p in [root, *root.children(recursive=True)])
                except psutil.Error:
                    rss = 0
                self.min_available = min(self.min_available, avail)
                self.max_rss = max(self.max_rss, rss)
                if time.time() - last_gpu > 30:
                    last_gpu = time.time()
                    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                          capture_output=True, text=True).stdout.strip()
                    if used.isdigit():
                        self.max_gpu_mib = max(self.max_gpu_mib, int(used))
                    out.write(f"{now()}\tavail_mib={avail >> 20}\trss_mib={rss >> 20}\tgpu_used_mib={used}\n")
                    out.flush()
                if avail < self.limit and not self.tripped:
                    self.tripped = True
                    out.write(f"{now()}\tLOW MEMORY: available {avail >> 20} MiB < {int(self.limit) >> 20} MiB; stopping trainer\n")
                    out.flush()
                    os.killpg(self.proc.pid, signal.SIGTERM)
                    try:
                        self.proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.proc.pid, signal.SIGKILL)


def run_stage(stage: str, resume: str | None, smoke: dict[str, Any] | None) -> int:
    recipe = load_recipe()
    lock = verify_source(recipe)
    state = v1_source_state()
    if smoke is None and state["dirty"]:
        raise SystemExit(f"formal runs need a clean saccade_v1 tree: {state['status']}")
    stage_dir = runs_dir(recipe, smoke is not None) / stage
    record = stage_dir / "stage.json"
    if record.is_file() and read_json(record).get("status") == "complete":
        raise SystemExit(f"{stage} is already complete ({record})")
    attempts = sorted(stage_dir.glob("attempt-*"))
    attempt = stage_dir / f"attempt-{len(attempts) + 1:02d}"
    run_dir = attempt / "run"
    cmd = build_command(recipe, stage, run_dir, smoke, resume)
    if stage == "cache_eval" and smoke is None:
        selected_checkpoint(recipe, "t1", False)  # the eval split opens only after T1 selection
    split_path = REPO / recipe["dataset"]["split_manifest"]
    identity = {
        "source_id": f"saccade@{lock['revision'][:12]}+t01-patch:{lock['patch_sha256'][:12]}",
        "old_revision": lock["revision"],
        "patch_sha256": lock["patch_sha256"],
        "source_tree_sha256": lock["patched_tree_sha256"],
        "v1_commit": state["commit"],
        "v1_dirty": state["dirty"],
        "recipe_sha256": sha256_file(RECIPE_PATH),
        "split_sha256": sha256_file(split_path),
        "stage": stage,
        "attempt": attempt.name,
        "smoke": smoke is not None,
    }
    attempt.mkdir(parents=True)
    env = {**os.environ, "SACCADE_T01_IDENTITY": json.dumps(identity, sort_keys=True), "PYTHONNOUSERSITE": "1",
           "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    payload: dict[str, Any] = {
        "stage": stage,
        "attempt": attempt.name,
        "status": "running",
        "started": now(),
        "command": cmd,
        "command_text": shlex.join(cmd),
        "resume": resume,
        "resume_kind": (RESUME_KINDS["teacher" if stage == "teacher" else "student"] if resume else None),
        "smoke": smoke,
        "identity": identity,
        "environment": environment(),
    }
    write_json(attempt / "attempt.json", payload)
    write_json(record, {"status": "running", "attempt": attempt.name, "updated": now()})
    t0 = time.time()
    with (attempt / "train.log").open("wb") as log:
        proc = subprocess.Popen(cmd, cwd=source_dir(recipe), env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        dog = Watchdog(proc, attempt / "memory.log", LOW_MEMORY_GIB)
        dog.start()
        code = proc.wait()
        dog.stop.set()
        dog.join()
    payload.update({
        "status": "complete" if code == 0 and not dog.tripped else "failed",
        "exit_code": code,
        "finished": now(),
        "wall_seconds": round(time.time() - t0, 1),
        "low_memory_stop": dog.tripped,
        "min_available_mib": int(dog.min_available) >> 20 if dog.min_available != float("inf") else None,
        "max_tree_rss_mib": dog.max_rss >> 20,
        "max_gpu_used_mib": dog.max_gpu_mib,
        "epochs_present": sorted(stage_epochs(stage_dir)),
    })
    if stage == "teacher" and code == 0:
        expected = int(recipe["stages"]["teacher"]["epochs"]) if smoke is None else smoke["epochs"]
        if max(stage_epochs(stage_dir), default=0) != expected:
            payload["status"] = "failed"
            payload["failure"] = "teacher exited 0 without its final epoch checkpoint"
    write_json(attempt / "attempt.json", payload)
    write_json(record, {"status": payload["status"], "attempt": attempt.name, "updated": now(),
                        "epochs_present": payload["epochs_present"]})
    print(f"{stage} {attempt.name}: {payload['status']} (exit {code}, {payload['wall_seconds']} s)")
    return 0 if payload["status"] == "complete" else 1


def select_fixed(stage: str, epoch: int, smoke: bool) -> None:
    """Fixed-epoch selection (teacher e12 rule)."""
    recipe = load_recipe()
    stage_dir = runs_dir(recipe, smoke) / stage
    if read_json(stage_dir / "stage.json").get("status") != "complete":
        raise SystemExit(f"{stage} is not complete")
    path = stage_epochs(stage_dir)[epoch]
    write_json(stage_dir / "selection.json", {
        "stage": stage, "rule": recipe["stages"][stage].get("selected_epoch_rule", f"fixed epoch {epoch}"),
        "epoch": epoch, "checkpoint": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
        "selected_at": now(),
    })
    print(f"{stage}: selected epoch {epoch} -> {path}")


# ---------------------------------------------------------------------------
# Validation, selection, final evaluation
# ---------------------------------------------------------------------------
def _eval_imports(recipe: dict[str, Any]) -> Any:
    from t01_exec import _BlockOldExtensions  # noqa: PLC0415

    if not any(isinstance(f, _BlockOldExtensions) for f in sys.meta_path):
        sys.meta_path.insert(0, _BlockOldExtensions())
    import t01_eval  # noqa: PLC0415

    t01_eval.use_source(source_dir(recipe))
    return t01_eval


def eval_stage(stage: str, smoke: bool) -> None:
    import torch  # noqa: PLC0415

    recipe = load_recipe()
    verify_source(recipe)
    ev = _eval_imports(recipe)
    stage_dir = runs_dir(recipe, smoke) / stage
    if read_json(stage_dir / "stage.json").get("status") != "complete":
        raise SystemExit(f"{stage} is not complete")
    data_root = Path(recipe["dataset"]["root"])
    val = recipe["split"]["val"]
    device = torch.device("cuda")
    for epoch, ckpt in sorted(stage_epochs(stage_dir).items()):
        out = stage_dir / "val" / f"epoch_{epoch:04d}.json"
        digest = sha256_file(ckpt)
        if out.is_file() and read_json(out).get("checkpoint_sha256") == digest:
            continue
        t0 = time.time()
        if stage == "teacher":
            res = ev.evaluate_teacher(ckpt, Path(recipe["base_weights"]["path"]), data_root, val, device)
        else:
            res = ev.evaluate_student(ckpt, cache_dir(recipe, "cache_val", smoke), data_root, val, device)
        res.update({"stage": stage, "epoch": epoch, "split": "val", "sequences_evaluated": val,
                    "checkpoint_sha256": digest, "policy": recipe["selection"]["policy"],
                    "evaluated_at": now(), "seconds": round(time.time() - t0, 1)})
        write_json(out, res)
        t = res["total"]
        print(f"{stage} e{epoch:02d}: AP50={t['ap50']:.4f} small@0.25={t['small_recall_at_0.25']} "
              f"nonfinite={t['checks']['nonfinite_raw_outputs']} ({res['seconds']} s)")


def select_stage(stage: str, smoke: bool) -> None:
    recipe = load_recipe()
    stage_dir = runs_dir(recipe, smoke) / stage
    if stage == "teacher":
        select_fixed(stage, 12 if not smoke else max(stage_epochs(stage_dir)), smoke)
        return
    if stage not in STUDENT_STAGES:
        raise SystemExit(f"{stage} has no selection")
    epochs = stage_epochs(stage_dir)
    rows = []
    for epoch, ckpt in sorted(epochs.items()):
        res_path = stage_dir / "val" / f"epoch_{epoch:04d}.json"
        if not res_path.is_file():
            raise SystemExit(f"missing validation result {res_path}; run eval first")
        res = read_json(res_path)
        if res["checkpoint_sha256"] != sha256_file(ckpt):
            raise SystemExit(f"{res_path} was computed for different checkpoint bytes")
        t = res["total"]
        rows.append({"epoch": epoch, "checkpoint": str(ckpt), "sha256": res["checkpoint_sha256"],
                     "eligible": t["checks"]["nonfinite_raw_outputs"] == 0, "ap50": t["ap50"],
                     "small_recall_at_0.25": t["small_recall_at_0.25"],
                     "recall_at_0.25": t["thresholds"]["0.25"]["bins"].get("all", {}).get("recall"),
                     "fp_at_0.25": t["thresholds"]["0.25"]["fp"],
                     "train_loss": res["checkpoint_info"].get("epoch_loss")})
    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        raise SystemExit(f"{stage}: no eligible checkpoint")
    best = max(eligible, key=lambda r: (r["ap50"], r["small_recall_at_0.25"] or 0.0, r["epoch"]))
    loss_best = min(rows, key=lambda r: (r["train_loss"] if r["train_loss"] is not None else float("inf")))
    write_json(stage_dir / "selection.json", {
        "stage": stage, "policy": recipe["selection"]["policy"],
        "rule": "max val AP50; ties: small recall@0.25 (min side 4-16 px), then later epoch; non-finite outputs ineligible",
        "epoch": best["epoch"], "checkpoint": best["checkpoint"], "sha256": best["sha256"],
        "bytes": Path(best["checkpoint"]).stat().st_size, "selected_at": now(),
        "train_loss_best_epoch_diagnostic": loss_best["epoch"], "table": rows,
    })
    print(f"{stage}: selected epoch {best['epoch']} (AP50 {best['ap50']:.4f}); train-loss-best epoch {loss_best['epoch']}")


def final_eval(smoke: bool) -> None:
    import torch  # noqa: PLC0415

    recipe = load_recipe()
    root = runs_dir(recipe, smoke)
    t1 = selected_checkpoint(recipe, "t1", smoke)
    final = root / "final"
    opened = final / "opened.json"
    if not opened.is_file():
        write_json(opened, {"opened_at": now(), "t1_selection_sha256": sha256_file(root / "t1" / "selection.json"),
                            "t1_checkpoint": str(t1), "note": "eval split opened after the T1 selection record"})
    if not (root / "cache_eval" / "stage.json").is_file() or read_json(root / "cache_eval" / "stage.json")["status"] != "complete":
        smoke_cfg = None if not smoke else {"epochs": 1, "batches": 0, "seqs": [], "cache_seqs": {"eval": recipe["split"]["eval"]}}
        if run_stage("cache_eval", None, smoke_cfg) != 0:
            raise SystemExit("cache_eval failed")
    ev = _eval_imports(recipe)
    data_root = Path(recipe["dataset"]["root"])
    device = torch.device("cuda")
    eval_seqs, val_seqs = recipe["split"]["eval"], recipe["split"]["val"]
    eval_cache, val_cache = cache_dir(recipe, "cache_eval", smoke), cache_dir(recipe, "cache_val", smoke)
    report: dict[str, Any] = {"policy": recipe["selection"]["policy"], "t1_checkpoint": str(t1),
                              "t1_sha256": sha256_file(t1), "started": now()}
    runs = [ev.evaluate_student(t1, eval_cache, data_root, eval_seqs, device) for _ in range(2)]
    report["t1_eval"] = runs[0]
    report["t1_eval_repeat_identical"] = (runs[0]["total"]["detections_sha256"] == runs[1]["total"]["detections_sha256"]
                                          and runs[0]["total"]["ap50"] == runs[1]["total"]["ap50"])
    val_runs = [ev.evaluate_student(t1, val_cache, data_root, val_seqs, device) for _ in range(2)]
    report["t1_val_repeat_identical"] = val_runs[0]["total"]["detections_sha256"] == val_runs[1]["total"]["detections_sha256"]
    report["t1_val"] = val_runs[0]
    report["lineage_on_eval_diagnostic"] = {}
    for stage in ("distill", "gt1", "t3"):
        ckpt = selected_checkpoint(recipe, stage, smoke)
        report["lineage_on_eval_diagnostic"][stage] = ev.evaluate_student(ckpt, eval_cache, data_root, eval_seqs, device)["total"]
    teacher = selected_checkpoint(recipe, "teacher", smoke)
    report["teacher_on_eval_diagnostic"] = ev.evaluate_teacher(teacher, Path(recipe["base_weights"]["path"]), data_root, eval_seqs, device)["total"]
    report["finished"] = now()
    write_json(final / "final_eval.json", report)
    t = report["t1_eval"]["total"]
    print(f"final eval {eval_seqs}: AP50={t['ap50']:.4f} repeat_identical={report['t1_eval_repeat_identical']}")


PIPELINE = ("teacher", "cache_train", "cache_val", "distill", "gt1", "t3", "t1")


def pipeline() -> int:
    recipe = load_recipe()
    root = runs_dir(recipe, False)
    for stage in PIPELINE:
        rec = root / stage / "stage.json"
        if not (rec.is_file() and read_json(rec).get("status") == "complete"):
            if run_stage(stage, None, None) != 0:
                print(f"pipeline stopped: {stage} failed")
                return 1
        if stage == "teacher" or stage in STUDENT_STAGES:
            eval_stage(stage, False)
            if not (root / stage / "selection.json").is_file():
                select_stage(stage, False)
    final_eval(False)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-source")
    p.add_argument("--write-lock", action="store_true")
    sub.add_parser("env")
    p = sub.add_parser("stage")
    p.add_argument("name")
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke-epochs", type=int, default=0, help="smoke run: epochs (0 = formal run)")
    p.add_argument("--smoke-batches", type=int, default=20)
    p.add_argument("--smoke-seqs", default="MOT17-09-SDP")
    p.add_argument("--smoke-val-seqs", default="MOT17-10-SDP")
    for name in ("eval", "select"):
        p = sub.add_parser(name)
        p.add_argument("name")
        p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("final-eval")
    p.add_argument("--smoke", action="store_true")
    sub.add_parser("pipeline")
    p = sub.add_parser("select-fixed")
    p.add_argument("name")
    p.add_argument("epoch", type=int)
    p.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    if a.cmd == "prepare-source":
        prepare_source(a.write_lock)
    elif a.cmd == "env":
        recipe = load_recipe()
        write_json(artifact_root(recipe) / "env.json", environment())
        print(json.dumps({k: v for k, v in environment().items() if k != "packages"}, indent=2))
    elif a.cmd == "stage":
        smoke = None
        if a.smoke_epochs:
            seqs = a.smoke_seqs.split(",")
            smoke = {"epochs": a.smoke_epochs, "batches": a.smoke_batches, "seqs": seqs,
                     "cache_seqs": {"train": seqs, "val": a.smoke_val_seqs.split(","), "eval": ["MOT17-02-SDP"]}}
        sys.exit(run_stage(a.name, a.resume, smoke))
    elif a.cmd == "select-fixed":
        select_fixed(a.name, a.epoch, a.smoke)
    elif a.cmd == "eval":
        eval_stage(a.name, a.smoke)
    elif a.cmd == "select":
        select_stage(a.name, a.smoke)
    elif a.cmd == "final-eval":
        final_eval(a.smoke)
    elif a.cmd == "pipeline":
        sys.exit(pipeline())


if __name__ == "__main__":
    main()

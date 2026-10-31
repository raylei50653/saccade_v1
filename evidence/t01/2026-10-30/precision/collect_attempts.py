"""Collect the compute-precision and storage records of every attempt under a T01 runs root.

    python3 collect_attempts.py RUNS_ROOT OUT.json
"""

import json
import sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
PROBE = ("conv_math", "matmul_math", "conv_rel_err_vs_fp64", "matmul_rel_err_vs_fp64")
FLAGS = ("cudnn_allow_tf32", "cuda_matmul_allow_tf32", "float32_matmul_precision", "nvidia_tf32_override")
rows = []
for attempt in sorted(root.glob("*/attempt-*/attempt.json")):
    a = json.loads(attempt.read_text())
    cp = a.get("compute_precision", {})
    rec = cp.get("in_process") or {}
    storage = a.get("storage", {})
    rows.append({
        "stage": a["stage"], "attempt": a["attempt"], "status": a["status"], "exit_code": a.get("exit_code"),
        "resume": a.get("resume"), "wall_seconds": a.get("wall_seconds"),
        "env_overrides": a.get("env_overrides"), "policy": cp.get("policy"), "autocast": cp.get("autocast"),
        "problems": cp.get("problems"),
        "start": {k: rec.get("start", {}).get(k) for k in FLAGS + PROBE},
        "exit": {k: rec.get("exit", {}).get(k) for k in FLAGS + PROBE},
        "monitor_samples": rec.get("samples"), "deviations": rec.get("deviations"),
        "storage_declared": storage.get("declared"),
        "cache_manifest": storage.get("cache_manifest"),
        "checkpoints": [{"checkpoint": c["checkpoint"], "tensor_dtypes": c["tensor_dtypes"],
                         "save_time": c["save_time_precision"]} for c in storage.get("checkpoints", [])],
        "storage_problems": storage.get("problems"),
    })
evals = []
for res in sorted(root.glob("*/val/epoch_*.json")) + sorted(root.glob("final/final_eval.json")):
    r = json.loads(res.read_text())
    for name, block in ([("result", r)] if "compute_precision" in r else
                        [(k, r[k]) for k in ("t1_eval", "t1_val") if isinstance(r.get(k), dict)]):
        p = block.get("compute_precision", {})
        evals.append({"file": str(res.relative_to(root)), "block": name,
                      **{f"{w}_{k}": p.get(w, {}).get(k) for w in ("start", "end") for k in ("policy", *PROBE[:2])},
                      "checkpoint_save_time_policy": (block.get("checkpoint_info", {}).get("t01_compute_precision") or {}).get("policy")})
out.write_text(json.dumps({"root": str(root), "attempts": rows, "evaluations": evals}, indent=1) + "\n")
print(f"{len(rows)} attempts, {len(evals)} evaluation records -> {out}")
for r in rows:
    print(f"{r['stage']:12s} {r['attempt']} {r['status']:8s} policy={r['policy']:13s} start={r['start']['conv_math']}/{r['start']['matmul_math']} "
          f"exit={r['exit']['conv_math']}/{r['exit']['matmul_math']} samples={r['monitor_samples']} deviations={len(r['deviations'] or [])} "
          f"problems={r['problems']} {r['storage_problems'] or ''}")

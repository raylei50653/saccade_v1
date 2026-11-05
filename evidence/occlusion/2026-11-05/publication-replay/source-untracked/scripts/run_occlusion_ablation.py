#!/usr/bin/env python3
"""Fixed-input, CPU-only OAO/front-occluder experiment on MOT17-10 val."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "MOT17-10-SDP"
ARMS = ("baseline", "oao", "front", "both")
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def source_hashes(output):
    paths = git("ls-files", "-c", "-o", "--exclude-standard", "-z", "--",
                "CMakeLists.txt", "cmake", "include", "src", "cuda", "tools",
                "scripts", "tests", "third_party").decode().split("\0")
    return {path: sha256(ROOT / path) for path in sorted(set(paths))
            if path and (ROOT / path).is_file()
            and not (ROOT / path).resolve().is_relative_to(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True,
                        help="already-built occlusion_ablation executable")
    parser.add_argument("--records", type=Path, required=True,
                        help="current G1 HR records directory (hr-records-v2)")
    parser.add_argument("--output", type=Path, required=True,
                        help="new output directory; existing paths are rejected")
    parser.add_argument("--skip-score", action="store_true",
                        help="replay and compare baseline without reading val GT")
    args = parser.parse_args()
    binary, records, output = (p.resolve() for p in
                               (args.binary, args.records, args.output))
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error(f"binary is not executable: {binary}")
    inputs = {suffix: records / f"{SEQUENCE}.{suffix}"
              for suffix in ("scdr", "scmo", "txt")}
    for path in inputs.values():
        if not path.is_file():
            parser.error(f"missing input: {path}")

    # Capture source before creating output, avoiding untracked output recursion.
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    tracked_status = git("status", "--porcelain=v1", "--untracked-files=no")
    diff = git("diff", "--binary", "HEAD")
    sources = source_hashes(output)
    frozen = {str(path): sha256(path) for path in [binary, *inputs.values()]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "source-status.txt").write_bytes(status)
    (output / "source-status-tracked.txt").write_bytes(tracked_status)
    (output / "source.diff").write_bytes(diff)
    write_json(output / "source-files.sha256.json", sources)
    # git diff cannot contain newly added, unstaged source files.
    untracked = set(git("ls-files", "-o", "--exclude-standard", "-z").decode().split("\0"))
    for path in sorted(untracked.intersection(sources)):
        destination = output / "source-untracked" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, destination)
    manifest = {
        "schema": "saccade-occlusion-ablation/1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": git("rev-parse", "HEAD").decode().strip(),
        "source_dirty": bool(status.strip()),
        "formal_acceptance": False,
        "measurement": "CPU tracker replay; not end-to-end FPS or GPU acceptance",
        "motion": "G1 NVOFA v1-motion/2, supplied frozen HR records",
        "sequence": SEQUENCE,
        "split": "validation; no holdout sequences read",
        "arms": list(ARMS),
        "scoring": not args.skip_score,
        "repository": str(ROOT),
        "binary": str(binary),
        "records": str(records),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "inputs_and_binary_sha256": frozen,
        "environment": {"platform": platform.platform(), "python": sys.version,
                        "SACCADE_LEGACY_ROOT": os.environ.get("SACCADE_LEGACY_ROOT")},
        "commands": [],
        "status": "running",
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)

    def run(label, command):
        entry = {"label": label, "argv": [str(x) for x in command], "cwd": str(ROOT)}
        write_json(output / f"{label}.command.json", entry)
        started = time.perf_counter()
        with (output / f"{label}.stdout.log").open("wb") as stdout, \
             (output / f"{label}.stderr.log").open("wb") as stderr:
            result = subprocess.run(entry["argv"], cwd=ROOT, stdout=stdout, stderr=stderr)
        entry.update(exit_code=result.returncode,
                     wall_ms=(time.perf_counter() - started) * 1000)
        (output / f"{label}.exit").write_text(f"{result.returncode}\n")
        manifest["commands"].append(entry)
        write_json(manifest_path, manifest)
        if result.returncode:
            raise RuntimeError(f"{label} exited {result.returncode}; see saved logs")

    score_sources = {}
    try:
        if not args.skip_score:
            legacy = Path(os.environ.get("SACCADE_LEGACY_ROOT",
                                         "/home/ray/developer/ai/saccade")).resolve()
            gt = legacy / "datasets/MOT17/train" / SEQUENCE
            scorer = ROOT / "scripts/eval_mot17.sh"
            score_sources = {str(path): sha256(path) for path in
                             sorted((legacy / "third_party/TrackEval").rglob("*.py"))}
            for path in (gt / "gt/gt.txt", gt / "seqinfo.ini", scorer):
                score_sources[str(path)] = sha256(path)
            write_json(output / "scoring-inputs.sha256.json", score_sources)
            run("scoring-environment", [legacy / ".venv/bin/python", "-c",
                "import sys,numpy; print(sys.version); print('numpy',numpy.__version__)"])

        rows = []
        for arm in ARMS:
            arm_dir = output / arm
            arm_dir.mkdir()
            mot, diag = arm_dir / f"{SEQUENCE}.txt", arm_dir / "diagnostics.json"
            print(f"Replaying {arm} on {SEQUENCE}", flush=True)
            run(f"replay-{arm}", [binary, inputs["scdr"], inputs["scmo"], mot, arm, diag])
            row = {"arm": arm, "mot_sha256": sha256(mot),
                   "diagnostics": json.loads(diag.read_text())}
            if arm == "baseline":
                manifest["baseline_bytes_equal"] = mot.read_bytes() == inputs["txt"].read_bytes()
                if not manifest["baseline_bytes_equal"]:
                    raise RuntimeError("baseline MOT differs from supplied G1 reference; stopped")
            if not args.skip_score:
                run(f"score-{arm}", ["bash", scorer, arm_dir, SEQUENCE])
                lines = (arm_dir / "trackeval/results/v1/pedestrian_summary.txt").read_text().splitlines()
                if len(lines) != 2 or len(lines[0].split()) != len(lines[1].split()):
                    raise RuntimeError(f"malformed TrackEval summary for {arm}")
                row["metrics"] = dict(zip(lines[0].split(), map(float, lines[1].split())))
                for name in METRICS:
                    if name not in row["metrics"]:
                        raise RuntimeError(f"missing {name} metric for {arm}")
            rows.append(row)
            write_json(output / "summary.json", rows)

        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["arm", *METRICS, "mot_sha256"])
            writer.writeheader()
            for row in rows:
                writer.writerow({"arm": row["arm"], "mot_sha256": row["mot_sha256"],
                                 **{key: row.get("metrics", {}).get(key, "") for key in METRICS}})
        after = {path: sha256(Path(path)) for path in frozen}
        manifest["inputs_and_binary_unchanged"] = frozen == after
        manifest["source_unchanged"] = sources == source_hashes(output)
        manifest["scoring_inputs_unchanged"] = (
            {path: sha256(Path(path)) for path in score_sources} == score_sources
            if not args.skip_score else None)
        if (not manifest["inputs_and_binary_unchanged"] or not manifest["source_unchanged"]
                or manifest["scoring_inputs_unchanged"] is False):
            raise RuntimeError("source, binary, inputs, or scoring dependencies changed during experiment")
        manifest["status"] = "complete"
        print("arm       " + " ".join(f"{name:>9}" for name in METRICS))
        for row in rows:
            print(f"{row['arm']:<10}" + " ".join(
                f"{row['metrics'][key]:9.3f}" if "metrics" in row else f"{'skipped':>9}"
                for key in METRICS))
        print(f"Evidence: {output}")
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        print(f"occlusion ablation: {error}", file=sys.stderr)
        return 1
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts_sha256"] = {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file() and path != manifest_path}
        write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

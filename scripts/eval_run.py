#!/usr/bin/env python3
"""Capture, score and measure one benchmark with the fixed HR deployment."""

import argparse
import configparser
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
ENGINE = Path("/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine")
BENCHMARKS = ("MOT17", "MOT20", "MOT15", "DanceTrack", "PP22")
QUALITY = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW")
SPEED = ("single_mean_ms", "single_p50_ms", "single_p95_ms", "single_p99_ms",
         "queued_ms_per_frame", "queued_fps")
COLUMNS = ("sequence", "frames", *QUALITY, *SPEED, "speed_note")
SCOPE = (
    "FPS is queued completion throughput with preloaded JPEGs, measured on frames 21–140 of each sequence. It excludes disk I/O and output files.",
    "It is neither per-frame latency nor live-stream or camera FPS.",
    "Clocks are not locked.",
)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eval_python():
    root = os.environ.get("SACCADE_EVAL_ROOT") or str(Path.home() / ".cache/saccade-v1/eval")
    return os.environ.get("SACCADE_EVAL_PYTHON") or str(Path(root) / "venv/bin/python")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--out", required=True)
    parser.add_argument("--engine", type=Path, default=ENGINE)
    parser.add_argument("--no-speed", action="store_true")
    parser.add_argument("sequences", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"build/[A-Za-z0-9][A-Za-z0-9._-]*", args.out):
        parser.error("--out must match build/[A-Za-z0-9][A-Za-z0-9._-]*")
    args.out = REPO / args.out
    if os.path.lexists(args.out):
        parser.error("--out must not exist")
    args.build = (REPO / args.build).resolve()
    for name in ("track_sequence", "tracking_benchmark"):
        if not (args.build / name).is_file():
            parser.error(f"--build must contain {name}")
    args.engine = args.engine.resolve()
    args.sequences = [path.resolve() for path in args.sequences]
    if len({path.name for path in args.sequences}) != len(args.sequences):
        parser.error("sequence names must be unique")
    if len({path.parent for path in args.sequences}) != 1:
        parser.error("sequences must share one parent directory (GT root)")
    args.frames = {}
    args.inputs = {}
    args.image_views = {}
    for path in args.sequences:
        if not (path / "seqinfo.ini").is_file() or not (path / "gt/gt.txt").is_file():
            parser.error(f"missing seqinfo.ini or gt/gt.txt: {path}")
        try:
            info = configparser.ConfigParser()
            info.read_string((path / "seqinfo.ini").read_text())
            sequence = info["Sequence"]
            image_dir = path / sequence["imDir"]
            extension = sequence["imExt"]
            args.frames[path.name] = int(sequence["seqLength"])
        except (OSError, ValueError, KeyError, configparser.Error) as error:
            parser.error(f"cannot read sequence layout: {path}: {error}")
        if not image_dir.is_dir():
            parser.error(f"missing image directory: {image_dir}")
        args.inputs[path.name] = path
        if (image_dir / f"{1:06d}{extension}").is_file():
            continue
        if (image_dir / f"{1:08d}{extension}").is_file():
            if sequence["imDir"] != "img1":
                parser.error(f"8-digit image view requires imDir == img1: {path}")
            view = args.out / "inputs" / path.name
            args.inputs[path.name] = view
            args.image_views[path.name] = {
                "original_path": str(path), "view_path": str(view),
                "source_digits": 8, "link_count": args.frames[path.name],
                "imExt": extension,
            }
        else:
            parser.error(f"missing images (expected 000001{extension} or 00000001{extension}): {image_dir}")
    return args


def identity(path):
    """Retain all occurrences: nested native identities repeat source fields."""
    result = {key: [] for key in ("source_commit", "source_dirty", "tracker_config_sha256")}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key in result:
            result[key].append(value)
    return result


def prepare_scoring(args):
    stage = args.out / "scoring"
    filters = []
    for seq in args.sequences:
        name = seq.name
        target = stage / "gt" / name
        (target / "gt").mkdir(parents=True)
        shutil.copyfile(seq / "seqinfo.ini", target / "seqinfo.ini")
        truth = (seq / "gt/gt.txt").read_bytes()
        if args.benchmark == "DanceTrack":
            truth = b"".join(b",".join(line.split(b",")[:7] + [b"1", b"1"]) + b"\n"
                             for line in truth.splitlines() if line.strip())
        (target / "gt/gt.txt").write_bytes(truth)
        prediction = args.out / "quality" / name / "mot.txt"
        destination = stage / "trackers/v1" / (name + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.benchmark == "PP22":
            keyframes = {int(line.split(b",", 1)[0]) for line in truth.splitlines() if line.strip()}
            rows = [line for line in prediction.read_bytes().splitlines(keepends=True) if line.strip()]
            kept = [line for line in rows if int(line.split(b",", 1)[0]) in keyframes]
            destination.write_bytes(b"".join(kept))
            filters.append({"sequence": name, "keyframes": len(keyframes),
                            "input_rows": len(rows), "kept_rows": len(kept)})
        else:
            shutil.copyfile(prediction, destination)
    dump(stage / "keyframe-filter.json", filters)
    benchmark = args.benchmark if args.benchmark in ("MOT15", "MOT20") else "MOT17"
    dump(stage / "config.json", {"dataset": {
        "GT_FOLDER": str(stage / "gt"), "TRACKERS_FOLDER": str(stage / "trackers"),
        "OUTPUT_FOLDER": str(stage / "results"), "SKIP_SPLIT_FOL": True,
        "TRACKER_SUB_FOLDER": "", "TRACKERS_TO_EVAL": ["v1"],
        "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": benchmark,
        "SPLIT_TO_EVAL": "train", "DO_PREPROC": True, "PRINT_CONFIG": False,
        "SEQ_INFO": {seq.name: None for seq in args.sequences},
    }})
    return stage


def speed_result(directory, capture_identity):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "PASS":
        raise RuntimeError(f"tracking_benchmark status is not PASS: {directory}")
    graph, = [group for group in summary["groups"] if group["mode"] == "graph"]
    sessions = []
    for path in sorted(directory.glob("*.identity.txt")):
        record = identity(path)
        hashes = record["tracker_config_sha256"]
        record["tracker_config_check"] = (
            "PASS" if hashes and all(value in capture_identity["tracker_config_sha256"] for value in hashes)
            else "FAIL" if hashes else "not exposed")
        sessions.append({"file": path.name, **record})
    check = ("FAIL" if any(row["tracker_config_check"] == "FAIL" for row in sessions)
             else "PASS" if any(row["tracker_config_check"] == "PASS" for row in sessions)
             else "not exposed")
    audit = {"status": summary["status"], "graph": graph, "sessions": sessions,
             "tracker_config_check": check}
    values = {f"single_{name}_ms": graph[f"latency_{name}_us"] / 1000
              for name in ("mean", "p50", "p95", "p99")}
    values.update(queued_ms_per_frame=graph["queued_us_per_frame_mean"] / 1000,
                  queued_fps=1e6 / graph["queued_us_per_frame_mean"], speed_note="")
    return values, audit


def write_results(out, rows, environment):
    dump(out / "results.json", rows)
    with (out / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    identities = list(environment["captures"].values())
    identities += [session for speed in environment["speed"].values()
                   for session in speed.get("sessions", [])]
    mismatch = any(value != environment["git_head"] for item in identities for value in item["source_commit"])
    dirty = any(value != "false" for item in identities for value in item["source_dirty"])
    lines = []
    if dirty or mismatch:
        lines.append("**WARNING: binaries are dirty or their source_commit differs from repository HEAD; see environment.json.**\n")
    lines += [f"# {environment['benchmark']} evaluation\n",
              "| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    for row in rows:
        cells = []
        for key in COLUMNS:
            value = row[key]
            if value is None:
                cells.append("")
            elif key in (*QUALITY[:-1], *SPEED):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", *SCOPE, ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main(argv=None):
    args = arguments(argv)
    args.out.mkdir(parents=True)
    commands = {"argv": [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                         *(sys.argv[1:] if argv is None else argv)],
                "cwd": str(Path.cwd()), "steps": []}
    environment = {"benchmark": args.benchmark, "gt_root": str(args.sequences[0].parent),
                   "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                   "git_status_porcelain": subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True),
                   "eval_python": eval_python(), "captures": {}, "speed": {},
                   "image_views": {},
                   "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in (
                       ("track_sequence", args.build / "track_sequence"),
                       ("tracking_benchmark", args.build / "tracking_benchmark"), ("engine", args.engine))}}

    def run(label, command, log, gpu=False):
        if gpu:
            command = ["timeout", "-k", "30s", "3600s", *map(str, command)]
        else:
            command = list(map(str, command))
        record = {"step": label, "argv": command, "cwd": str(REPO),
                  "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "log": str(log.relative_to(args.out))}
        commands["steps"].append(record)
        dump(args.out / "command.json", commands)
        print(label, flush=True)
        start = time.monotonic()
        with log.open("wb") as output:
            result = subprocess.run(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, wall_seconds=time.monotonic() - start)
        dump(args.out / "command.json", commands)
        if result.returncode:
            raise RuntimeError(f"{label} failed ({result.returncode}); see {log}")

    dump(args.out / "command.json", commands)
    dump(args.out / "environment.json", environment)
    try:
        for name, record in args.image_views.items():
            original, view = Path(record["original_path"]), Path(record["view_path"])
            (view / "img1").mkdir(parents=True)
            shutil.copyfile(original / "seqinfo.ini", view / "seqinfo.ini")
            for fid in range(1, args.frames[name] + 1):
                source = (original / "img1" / f"{fid:08d}{record['imExt']}").resolve(strict=True)
                (view / "img1" / f"{fid:06d}{record['imExt']}").symlink_to(source)
            environment["image_views"][name] = {key: value for key, value in record.items() if key != "imExt"}
        lock_path = Path.home() / ".cache/saccade-v1/gpu.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], text=True)
            environment["gpus"] = list(csv.DictReader(gpu.splitlines(), skipinitialspace=True))
            with (args.out / "telemetry.csv").open("wb") as telemetry_output:
                telemetry = subprocess.Popen([
                    "nvidia-smi", "--query-gpu=timestamp,uuid,pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
                    "--format=csv", "-l", "1"], stdout=telemetry_output, stderr=subprocess.STDOUT)
                try:
                    for seq in args.sequences:
                        dest = args.out / "quality" / seq.name
                        dest.mkdir(parents=True)
                        run("capture " + seq.name, [args.build / "track_sequence", args.engine, args.inputs[seq.name], dest / "mot.txt",
                            "--profile", "hr-1280x768", "--set", "tracker.kalman.nsa_enabled=false",
                            "--set", "tracker.rows.measurement_box=false", "--resolved-config", dest / "resolved.config"],
                            dest / "track.log", gpu=True)
                        environment["captures"][seq.name] = identity(dest / "track.log")
                    (args.out / "speed").mkdir()
                    for seq in args.sequences:
                        note = ("disabled (--no-speed)" if args.no_speed else
                                "fewer than 140 frames" if args.frames[seq.name] < 140 else "")
                        if note:
                            environment["speed"][seq.name] = {"speed_note": note, "tracker_config_check": "not run"}
                            continue
                        dest = args.out / "speed" / seq.name
                        run("speed " + seq.name, [args.build / "tracking_benchmark", args.engine, args.inputs[seq.name], dest,
                            "--frames", "120", "--warmup", "20", "--rounds", "2", "--only", "graph"],
                            args.out / "speed" / (seq.name + ".log"), gpu=True)
                        values, audit = speed_result(dest, environment["captures"][seq.name])
                        environment["speed"][seq.name] = {**audit, "values": values}
                        if audit["tracker_config_check"] == "FAIL":
                            raise RuntimeError(f"tracker config hash differs between capture and speed: {seq.name}")
                finally:
                    telemetry.terminate()
                    telemetry.wait(timeout=10)
        stage = prepare_scoring(args)
        run("scoring", [eval_python(), "-I", "-B", REPO / "scripts/eval_trackeval.py", stage], stage / "trackeval.log")
        rows = json.loads((stage / "quality.json").read_text())
        for row in rows:
            row.update({key: None for key in SPEED}, speed_note="")
            if row["sequence"] != "COMBINED":
                speed = environment["speed"][row["sequence"]]
                row.update(speed["values"] if "values" in speed else {"speed_note": speed["speed_note"]})
        write_results(args.out, rows, environment)
        environment["status"] = "PASS"
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        environment.update(status="FAIL", error=str(error))
        print(str(error), file=sys.stderr)
        return 1
    finally:
        dump(args.out / "environment.json", environment)
    print(args.out / "RESULTS.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

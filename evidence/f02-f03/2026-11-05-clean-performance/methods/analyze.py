#!/usr/bin/env python3
"""Read-only validation of clean-source ABBA payloads; write derived tables."""
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

COMMITS = {
    "baseline": "f4865cdb27f0e624b8cc26219e78f4aa938690df",
    "candidate": "23ade536906d07ccf09859f1b580b044dafe95dc",
}
ENGINE = "8292cc233019d08f1b1b6cfbc12f1fe1644f66f2164fd8f50a69b660a7817876"
IMAGES = "0d2ad63d8d74b02858099d48b100887667d0b6040cca2bfd39d1faf7edec5611"
PACKET_BYTES = 12368


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def stats(values):
    require(bool(values), "empty sample group")
    require(all(math.isfinite(v) and v > 0 for v in values), "invalid timing sample")
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean": statistics.mean(values),
        "p50": ordered[math.ceil(len(values) * .50) - 1],
        "p95": ordered[math.ceil(len(values) * .95) - 1],
        "p99": ordered[math.ceil(len(values) * .99) - 1],
        "min": ordered[0],
        "max": ordered[-1],
    }


def comparison(a, b):
    return {
        "baseline": stats(a), "candidate": stats(b),
        "mean_reduction_percent": 100 * (1 - statistics.mean(b) / statistics.mean(a)),
        "throughput_speedup": statistics.mean(a) / statistics.mean(b),
    }


def identity(path, backend, model=False):
    pairs = defaultdict(list)
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            pairs[key].append(value)
    commit_key, dirty_key = ("build_commit", "build_dirty") if model else ("source_commit", "source_dirty")
    require(pairs[commit_key] and set(pairs[commit_key]) == {COMMITS[backend]}, f"wrong commit: {path}")
    require(pairs[dirty_key] and set(pairs[dirty_key]) == {"false"}, f"dirty identity: {path}")
    require(pairs["engine_sha256"] and set(pairs["engine_sha256"]) == {ENGINE}, f"wrong engine: {path}")
    require(pairs["profile"] == ["hr-1280x768"], f"wrong profile: {path}")
    if model:
        require(pairs["images_sha256"] == [IMAGES], f"wrong model inputs: {path}")
    else:
        require(pairs["motion_profile"] == ["v1-motion/1"], f"wrong motion profile: {path}")
        require(pairs["cuda_graph"] == ["true"], f"graph disabled: {path}")


def write_csv(path, records):
    require(records, f"empty CSV: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def context(record):
    return {"outer_round": record["round"], "outer_position": record["position"],
            "backend": record["backend"], "directory": record["directory"]}


def main(root):
    records = json.loads((root / "records.json").read_text())
    require(len(records) == 16, "expected exactly 16 benchmark processes")
    require(len({str(Path(r["directory"]).resolve()) for r in records}) == 16, "reused output directory")
    for workload in ("model", "tracking"):
        selected = [r for r in records if r["workload"] == workload]
        require(len(selected) == 8, f"expected eight {workload} processes")
        for round_id in range(2):
            selected_round = [r for r in selected if r["round"] == round_id]
            require(sorted(r["position"] for r in selected_round) == list(range(4)), "invalid ABBA positions")
            require([r["backend"] for r in sorted(selected_round, key=lambda r: r["position"])] ==
                    ["baseline", "candidate", "candidate", "baseline"], "not ABBA")

    model_samples, model_runs, tracking_runs, tracking_latencies = [], [], [], []
    model_reference = None
    packet_reference = None
    input_reference = None
    packet_paths = []
    for record in records:
        directory = Path(record["directory"])
        require(not (directory / "FAILURE.txt").exists(), f"failed benchmark: {directory}")
        summary = json.loads((directory / "summary.json").read_text())
        require(summary["status"] == "PASS", f"summary not PASS: {directory}")
        status_path = directory / "status.json"
        if status_path.exists():
            require(json.loads(status_path.read_text())["status"] == "PASS", f"status not PASS: {directory}")
        metadata = context(record)
        backend = record["backend"]
        if record["workload"] == "model":
            identity(directory / "identity.txt", backend, model=True)
            require(summary["frames_verified"] == 4 and summary["timing_input"] == 0, "wrong model verification scope")
            require(summary["iterations_per_run"] == 200 and summary["warmup_per_run"] == 40 and
                    summary["rounds"] == 1, "wrong model replay bounds")
            require(summary["ordinary_repeat_and_graph_bits_equal"] is True, "internal model equivalence failed")
            paths = {p.name: p for p in directory.glob("input-*") if p.suffix in (".f32", ".status")}
            expected = {f"input-{i}{suffix}" for i in range(4) for suffix in (".f32", ".status")}
            require(set(paths) == expected, f"incomplete model outputs: {directory}")
            payloads = {name: path.read_bytes() for name, path in paths.items()}
            require(all(payloads.values()), f"empty model output: {directory}")
            require(all(len(payloads[f"input-{i}.status"]) == 72 for i in range(4)), "expected nine 8-byte statuses")
            if model_reference is None:
                model_reference = payloads
            require(payloads == model_reference, f"model outputs/status bytes differ: {directory}")
            samples = rows(directory / "samples.csv")
            require(len(samples) == 800, f"wrong model sample count: {directory}")
            require(Counter((r["round"], r["ordinal"], r["mode"]) for r in samples) ==
                    Counter({("0", "0", "ordinary"): 200, ("0", "1", "graph"): 200,
                             ("0", "2", "graph"): 200, ("0", "3", "ordinary"): 200}), "wrong model schedule")
            for ordinal in range(4):
                require(sorted(int(r["iteration"]) for r in samples if int(r["ordinal"]) == ordinal) ==
                        list(range(200)), "duplicate or missing model iterations")
            for sample in samples:
                model_samples.append({**metadata, **sample})
            for mode in ("ordinary", "graph"):
                values = [float(r["gpu_ms"]) for r in samples if r["mode"] == mode]
                model_runs.append({**metadata, "mode": mode, "gpu_ms": stats(values),
                                   "wall_ms": stats([float(r["wall_ms"]) for r in samples if r["mode"] == mode])})
        else:
            require(status_path.exists(), f"missing tracking final status: {directory}")
            require(summary["frames_per_session"] == 120 and summary["warmup_frames"] == 20 and
                    summary["abba_rounds"] == 1, "wrong tracking replay bounds")
            inputs = (directory / "inputs.sha256").read_bytes()
            if input_reference is None:
                input_reference = inputs
            require(inputs == input_reference, f"tracking input manifests differ: {directory}")
            runs = rows(directory / "runs.csv")
            require(len(runs) == 4, "expected four tracking sessions per process")
            require(Counter((r["round"], r["ordinal"], r["graph"], r["style"]) for r in runs) ==
                    Counter({("0", "2", "1", "single"): 1, ("0", "2", "1", "queued"): 1,
                             ("0", "3", "1", "single"): 1, ("0", "3", "1", "queued"): 1}), "wrong tracking schedule")
            require(len(list(directory.glob("*.bin"))) == 4, "unexpected tracking packet files")
            for run in runs:
                stem = f"r{run['round']}-{run['ordinal']}-graph-{run['style']}"
                identity(directory / (stem + ".identity.txt"), backend)
                packet_path = directory / (stem + ".bin")
                packet = packet_path.read_bytes()
                require(len(packet) == 140 * PACKET_BYTES, f"not 140 complete packets: {packet_path}")
                if packet_reference is None:
                    packet_reference = packet
                require(packet == packet_reference, f"full packet bytes differ: {packet_path}")
                require(digest(packet) == run["packet_sha256"] == summary["packet_sha256"], "packet hash receipt mismatch")
                packet_paths.append(str(packet_path))
                require(math.isclose(float(run["wall_us"]) / 120, float(run["wall_us_per_frame"]),
                                     rel_tol=1e-8, abs_tol=1e-5), "tracking interval denominator mismatch")
                tracking_runs.append({**metadata, **run})
                if run["style"] == "single":
                    latencies = rows(directory / (stem + ".csv"))
                    require([int(r["frame"]) for r in latencies] == list(range(20, 140)), "missing tracking frames")
                    for latency in latencies:
                        tracking_latencies.append({**metadata, "inner_round": run["round"],
                                                   "ordinal": run["ordinal"], **latency})

    require(len(model_samples) == 6400 and len(model_runs) == 16, "wrong aggregate model count")
    require(len(tracking_runs) == 32 and len(tracking_latencies) == 1920, "wrong aggregate tracking count")
    require(len(packet_paths) == 32, "expected 32 packet files")
    analysis = {
        "status": "PASS", "schema": "f02-f03-clean-performance-analysis/1", "commits": COMMITS,
        "engine_sha256": ENGINE, "model_images_sha256": IMAGES,
        "measurement_scope": "v1-motion/1; fixed engine/GPU; model input 0 timed, four inputs verified; HR tracking prefix 140 frames",
        "raw_model_capture_labels_preserved": "development-optimization/formal_acceptance=false are fixed harness labels; clean build identities checked separately",
        "model_output_status_comparison": {"mode": "direct byte comparison", "processes": 8,
            "inputs": 4, "payload_sha256": {name: digest(data) for name, data in model_reference.items()}},
        "tracking_packet_comparison": {"mode": "direct byte comparison", "files": len(packet_paths),
            "frames_per_file": 140, "bytes_per_file": len(packet_reference), "sha256": digest(packet_reference),
            "files_compared": packet_paths, "input_manifest_sha256": digest(input_reference)},
        "model_ms": {}, "tracking_us": {}, "per_round": [], "model_processes": model_runs,
        "largest_single_latencies": sorted(tracking_latencies,
            key=lambda r: float(r["enqueue_to_receive_us"]), reverse=True)[:20],
        "outlier_policy": "All samples retained; no trimming, no outlier exclusions",
    }
    for mode in ("ordinary", "graph"):
        def model_values(backend, round_id=None):
            return [float(r["gpu_ms"]) for r in model_samples if r["mode"] == mode and
                    r["backend"] == backend and (round_id is None or r["outer_round"] == round_id)]
        require(len(model_values("baseline")) == len(model_values("candidate")) == 1600, "wrong model pooled count")
        analysis["model_ms"][mode] = comparison(model_values("baseline"), model_values("candidate"))
        for round_id in range(2):
            analysis["per_round"].append({"workload": "model", "mode": mode, "round": round_id,
                **comparison(model_values("baseline", round_id), model_values("candidate", round_id))})
    for style in ("single", "queued"):
        def tracking_values(backend, round_id=None):
            source = tracking_latencies if style == "single" else tracking_runs
            key = "enqueue_to_receive_us" if style == "single" else "wall_us_per_frame"
            return [float(r[key]) for r in source if r["backend"] == backend and
                    (style == "single" or r["style"] == "queued") and
                    (round_id is None or r["outer_round"] == round_id)]
        expected_count = 960 if style == "single" else 8
        require(len(tracking_values("baseline")) == len(tracking_values("candidate")) == expected_count,
                "wrong tracking pooled count")
        analysis["tracking_us"][style] = comparison(tracking_values("baseline"), tracking_values("candidate"))
        if style == "queued":
            analysis["tracking_us"][style]["scope"] = "throughput wall interval per frame; not frame delivery latency"
            for backend in COMMITS:
                values = tracking_values(backend)
                analysis["tracking_us"][style][backend]["run_intervals_us_per_frame"] = values
                analysis["tracking_us"][style][backend]["fps"] = 1e6 / statistics.mean(values)
        for round_id in range(2):
            analysis["per_round"].append({"workload": "tracking", "mode": style, "round": round_id,
                **comparison(tracking_values("baseline", round_id), tracking_values("candidate", round_id))})
    write_csv(root / "model-samples.csv", model_samples)
    write_csv(root / "tracking-runs.csv", tracking_runs)
    write_csv(root / "tracking-latencies.csv", tracking_latencies)
    (root / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print("PASS: 16 clean ABBA processes; 4-input raw/status bytes and 32 complete packet files identical")
    for label, result in [("F03 Graph ms", analysis["model_ms"]["graph"]),
                          ("single latency us", analysis["tracking_us"]["single"]),
                          ("queued interval us/frame", analysis["tracking_us"]["queued"])]:
        print(f"{label}: {result['baseline']['mean']:.6f} -> {result['candidate']['mean']:.6f} "
              f"({result['mean_reduction_percent']:.3f}% reduction)")


if __name__ == "__main__":
    try:
        require(len(sys.argv) == 2, "usage: analyze.py CAMPAIGN_ROOT")
        main(Path(sys.argv[1]).resolve())
    except Exception as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        raise SystemExit(1)

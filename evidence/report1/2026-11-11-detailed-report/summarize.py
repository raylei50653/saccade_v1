#!/usr/bin/env python3
"""Recalculate report tables from pinned Git objects; no build/GPU execution.

From any directory: python3 /path/to/this/summarize.py > /new/summary.json
The script writes JSON only to stdout and does not modify historical evidence.
"""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess

BASE = "3cbaa518f0dc37d691404e95ed85c0109c81130d"
ROOT = Path(__file__).resolve().parents[3]
G1 = "evidence/tracking-session/2026-11-05-v1-motion-2-acceptance-2/capture/"
CLEAN = "evidence/f02-f03/2026-11-05-clean-performance/"
NSA = "evidence/nsa-v2/2026-11-09-formal-527ffe5/SUMMARY.json"
sources = {}


def read(path):
    data = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
    sources[path] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    return data.decode("utf-8")


def rows(path):
    return list(csv.DictReader(io.StringIO(read(path))))


def stats(values):
    values = sorted(values)
    if not values:
        raise ValueError("empty sample set")
    return {"n": len(values), "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p50": values[math.ceil(len(values) * .50) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1],
            "p99": values[math.ceil(len(values) * .99) - 1],
            "min": values[0], "max": values[-1]}


serial = rows(G1 + "serial-latency.csv")
throughput = rows(G1 + "throughput.csv")
throughput_runs = rows(G1 + "throughput-runs.csv")
overlap = rows(G1 + "ofa-overlap.csv")
ofa_trials = rows(G1 + "ofa-trials.csv")
serial_frames = sum(int(r["frames"]) for r in serial)
serial_wall = sum(float(r["wall_ms"]) for r in serial)
total = next(r for r in throughput if r["sequence"] == "Total")
g1_mean = float(total["after_ms_per_frame"])
repeat_means = []
for repeat in ("1", "2", "3"):
    means = {}
    for arm in ("before", "after"):
        selected = [r for r in throughput_runs if r["repeat"] == repeat and r["arm"] == arm]
        means[arm] = sum(float(r["wall_ms"]) for r in selected) / sum(int(r["frames"]) for r in selected)
    repeat_means.append({"repeat": int(repeat), **means,
                         "change_percent": (means["after"] / means["before"] - 1) * 100})

nsa = json.loads(read(NSA))["step5_record_only"]["native_ms_per_frame"]
nsa_stats = {}
for arm in ("default", "enabled"):
    selected = [r for r in nsa if r["arm"] == arm]
    frames = sum(r["frames"] for r in selected)
    mean = sum(r["ms_per_frame"] * r["frames"] for r in selected) / frames
    sequences = sorted({r["sequence"] for r in selected})
    medians = [{"sequence": seq, "frames": next(r["frames"] for r in selected if r["sequence"] == seq),
                "median_ms_per_frame": statistics.median(r["ms_per_frame"] for r in selected if r["sequence"] == seq)}
               for seq in sequences]
    weighted_medians = sum(r["frames"] * r["median_ms_per_frame"] for r in medians) / sum(r["frames"] for r in medians)
    nsa_stats[arm] = {"runs": len(selected), "frames": frames, "weighted_mean_ms_per_frame": mean,
                      "fps": 1000 / mean, "sequence_medians": medians,
                      "weighted_sequence_medians_ms_per_frame": weighted_medians}

latencies = rows(CLEAN + "tracking-latencies.csv")
runs = rows(CLEAN + "tracking-runs.csv")
analysis = json.loads(read(CLEAN + "analysis.json"))
clean_stats = {}
for backend in ("baseline", "candidate"):
    selected = [r for r in runs if r["backend"] == backend]
    single = stats(float(r["enqueue_to_receive_us"]) / 1000 for r in latencies if r["backend"] == backend)
    queued = stats(float(r["wall_us_per_frame"]) / 1000 for r in selected if r["style"] == "queued")
    clean_stats[backend] = {"single_ms": single, "queued_interval_ms_per_frame": queued,
                            "queued_fps": 1000 / queued["mean"],
                            "constructor_ms": stats(float(r["init_us"]) / 1000 for r in selected),
                            "first_frame_ms": stats(float(r["first_frame_us"]) / 1000 for r in selected)}

for path in ("src/tracking_session.cpp", "cuda/flow_motion.cu", "tools/track_sequence.cpp",
             "tools/track_stream.cpp", "tools/tracking_benchmark.cpp", "tools/resource_schedule_probe.cpp",
             "include/saccade/model_profile.hpp", "include/saccade/tracking_session.hpp",
             "src/head_trt_network.cpp", "src/model_trt_network.cpp", "docs/TRACKING.md", "docs/STREAMING.md",
             "evidence/resource-schedule/2026-11-04/REPORT.md", "evidence/resource-schedule/2026-11-04/stages/stage_summary.csv"):
    read(path)

def overlap_stats(selected):
    return {"n": len(selected), "contained": sum(r["ofa_bounds_contained_in_f03"] == "True" for r in selected),
            "ofa_indirect_bound_ms": stats((int(r["ofa_latest_end_ns"]) - int(r["ofa_earliest_start_ns"])) / 1e6 for r in selected),
            "f03_envelope_ms": stats((int(r["f03_end_ns"]) - int(r["f03_start_ns"])) / 1e6 for r in selected)}

result = {"schema": "report1-latency-source-summary/1", "report_base": BASE,
          "new_runtime_or_gpu_measurements": False,
          "quantile": "nearest-rank: sorted[ceil(n*p)-1]; median averages middle pair; no outliers removed",
          "g1_motion2": {"serial_sequence_averages": serial, "serial_total_frames": serial_frames,
                          "serial_total_wall_ms": serial_wall, "serial_weighted_ms_per_frame": serial_wall / serial_frames,
                          "serial_per_frame_quantiles": None, "pipelined": throughput,
                          "pipelined_total_fps": 1000 / g1_mean, "repeats": repeat_means,
                          "ofa_all29": overlap_stats(overlap),
                          "ofa_frames2_through29": overlap_stats([r for r in overlap if int(r["frame"]) >= 2]),
                          "ofa_host_api_ms_nearest_rank": stats(float(r["host_ms"]) for r in ofa_trials)},
          "nsa_record_only": {"arms": nsa_stats,
                               "enabled_change_percent": (nsa_stats["enabled"]["weighted_mean_ms_per_frame"] /
                                                          nsa_stats["default"]["weighted_mean_ms_per_frame"] - 1) * 100},
          "clean_motion1": {"recalculated": clean_stats, "original_model_ms": analysis["model_ms"]},
          "sources_from_report_base": dict(sorted(sources.items()))}
print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

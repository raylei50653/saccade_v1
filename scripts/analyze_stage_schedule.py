#!/usr/bin/env python3
"""Analyze a resource_schedule_probe Nsight Systems SQLite export.

The probe's NVTX frame enqueue ranges, CUDA Runtime correlations and graph
launch correlations establish frame identity. Nsight may assign synthetic
stream IDs to graph kernels: logical D/M is determined by the owned graph
contents, not by treating those IDs as extra concurrent execution streams.

usage: python3 -I scripts/analyze_stage_schedule.py TRACE.sqlite OUTPUT_DIR \
    --warmup 20 --frames 120

Kernel busy time is the union of activity intervals; envelope is first kernel
start to last kernel end and includes gaps. Neither measures the hardware
JPEG decoder itself. Profiler timings are kept separate from unprofiled wall
latency. A complete frame must have the expected stage structure and one
12,368-byte packet D2H; missing correlations/unknown work fail closed.
"""

import argparse
import bisect
import collections
import csv
import html
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys


U = "saccade::<unnamed>::"
S = "saccade::detail::<unnamed>::"
STAGES = [
    "F01_compressed_H2D", "F01_nvJPEG_SM_conversion", "F02_resize", "F05_motion_image",
    "F03_engine", "F03_producer", "F04_postprocess", "M_begin",
    "F06_GMC_including_cuFFT", "F08a_predict", "F04b_batch",
    "F07_F08b_update", "M_control", "M_candidates", "M_commit",
    "M_packet", "packet_D2H",
]
EXPECTED = {
    "F01_compressed_H2D": 1,
    "F02_resize": 2, "F05_motion_image": 2, "F03_producer": 3,
    "F04_postprocess": 5, "M_begin": 1, "F08a_predict": 2,
    "F04b_batch": 1, "F07_F08b_update": 2, "M_control": 4,
    "M_candidates": 2, "M_commit": 1, "M_packet": 1,
    "packet_D2H": 1,
}


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def duration(intervals):
    return sum(end - start for start, end in merged(intervals))


def intersection(a, b):
    a, b = merged(a), merged(b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def stats(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values), "mean": statistics.fmean(values),
        "p50": values[math.ceil(.50 * len(values)) - 1],
        "p95": values[math.ceil(.95 * len(values)) - 1],
        "p99": values[math.ceil(.99 * len(values)) - 1],
        "min": values[0], "max": values[-1],
    }


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def custom_stage(name):
    if name.startswith("void nvjpeg::"):
        return "F01_nvJPEG_SM_conversion", "JPEG_SM"
    if name.startswith(U):
        rest = name[len(U):]
        function = rest.split("(", 1)[0]
        if function in {"horizontal", "vertical"}:
            return "F02_resize", "D"
        if function in {"image_kernel", "image_header"}:
            return "F05_motion_image", "D"
        if function in {"finite", "aggregate"} or rest.startswith("initialize(saccade::PostprocessProducerStatus *"):
            return "F03_producer", "D"
        if function in {"decode", "adaptive_prefix", "dense_merge"} or rest.startswith((
            "initialize(const saccade::PostprocessProducerStatus *",
            "finalize(saccade::DetectionCandidates *",
        )):
            return "F04_postprocess", "D"
        if function in {"reset_numeric", "motion_stage", "deliver", "cross_power", "reduce_correlation", "context_kernel"}:
            return "F06_GMC_including_cuFFT", "M"
        if function == "predict_kernel":
            return "F08a_predict", "M"
        if function == "batch_kernel":
            return "F04b_batch", "M"
        if function == "update_kernel":
            return "F07_F08b_update", "M"
    if name.startswith("void regular_fft"):
        return "F06_GMC_including_cuFFT", "M"
    if name.startswith(S):
        function = name[len(S):].split("(", 1)[0]
        return {
            "begin": ("M_begin", "M"), "stage": ("M_control", "M"),
            "candidates": ("M_candidates", "M"),
            "commit": ("M_commit", "M"), "packet": ("M_packet", "M"),
        }.get(function, (None, None))
    return None, None


def render_schedule(prefix, events, stage_rows, frame_ranges, receive_ranges, mode, first_frame):
    """Render measured intervals with stdlib SVG and the local rsvg converter."""
    wanted = set(range(first_frame, first_frame + 5))
    values = [e for e in events if e["mode"] == mode and e["frame"] in wanted]
    stages = [r for r in stage_rows if r["mode"] == mode and r["frame"] in wanted]
    if not values:
        raise ValueError(f"no {mode} activity for schedule plot")
    base = frame_ranges[mode, first_frame]["start"]
    last = max(e["end"] for e in values)
    xmax = max(5, math.ceil((last - base) / 1e6 / 5) * 5)
    width, height, left, right = 1800, 865, 285, 1740
    scale = (right - left) / xmax
    x = lambda ns: left + (ns - base) / 1e6 * scale
    lane_y = {"CPU": 175, "COPY": 270, "JPEG_SM": 365, "D": 460, "M": 555}
    colors = {
        "F01_compressed_H2D": "#9b59b6", "F01_nvJPEG_SM_conversion": "#a66ed1",
        "F02_resize": "#48a9dc", "F05_motion_image": "#4fbd8b",
        "F03_engine": "#286bc0", "F03_producer": "#da9a31", "F04_postprocess": "#e47734",
        "F06_GMC_including_cuFFT": "#299e76", "F08a_predict": "#289fab",
        "F04b_batch": "#c44972", "F07_F08b_update": "#c97427",
        "packet_D2H": "#c5a129",
    }
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7" fill="#5b6878"/></marker></defs>',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<g font-family="DejaVu Sans, sans-serif" fill="#182c43">',
             f'<text x="50" y="46" font-size="27" font-weight="bold">Measured {html.escape(mode)} schedule: frames {first_frame} to {first_frame + 4}</text>',
             '<text x="50" y="77" font-size="17">HR 1280x768 FP16 / CUDA Graph / RTX 5070 Ti Laptop GPU / Nsight Systems CUDA node trace</text>',
             '<text x="50" y="106" font-size="15" fill="#566779">Time is relative to the first displayed frame enqueue. One CPU owner submits the session.</text>']
    for tick in range(0, xmax + 1):
        xx = left + tick * scale
        major = tick % 5 == 0
        parts.append(f'<line x1="{xx:.2f}" y1="133" x2="{xx:.2f}" y2="625" stroke="{"#cbd5e1" if major else "#edf0f4"}" stroke-width="1"/>')
        if major:
            parts.append(f'<text x="{xx:.2f}" y="650" font-size="14" text-anchor="middle">{tick}</text>')
    labels = {
        "CPU": ["CPU host thread", "E = enqueue (includes waits)", "R = blocking receive"],
        "COPY": ["GPU copy engine", "JPEG compressed H2D", "12,368-byte packet D2H"],
        "JPEG_SM": ["GPU SM: nvJPEG", "Color conversion only", "Private CUDA stream"],
        "D": ["GPU logical D stream", "Resize, motion image", "TensorRT, producer, F04"],
        "M": ["GPU logical M stream", "GMC / predict / tracking", "Control, commit, packet"],
    }
    for lane, yy in lane_y.items():
        parts.append(f'<rect x="{left}" y="{yy-31}" width="{right-left}" height="65" fill="#f5f7fa"/>')
        for i, label in enumerate(labels[lane]):
            parts.append(f'<text x="50" y="{yy-12+i*19}" font-size="{16 if i == 0 else 13}" font-weight="{"bold" if i == 0 else "normal"}">{html.escape(label)}</text>')
    # Pale envelopes show gaps explicitly; tiny control stages are plotted
    # only as their real activity intervals, never as a long control envelope.
    for r in stages:
        stage, side = r["stage"], r["logical_stream"]
        if stage not in colors or side not in {"D", "M"}:
            continue
        xx, ww = x(r["start_ns"]), max(.5, (r["end_ns"] - r["start_ns"]) / 1e6 * scale)
        yy = lane_y[side]
        parts.append(f'<rect x="{xx:.3f}" y="{yy-22}" width="{ww:.3f}" height="44" rx="2" fill="{colors[stage]}" opacity="0.20"/>')
        if stage == "F03_engine":
            parts.append(f'<text x="{xx+ww/2:.3f}" y="{yy-8}" font-size="13" text-anchor="middle" fill="#174b88">F03 frame {r["frame"]}</text>')
    for event in values:
        stage = event["stage"]
        lane = "COPY" if event["kind"] == "memcpy" else event["logical_stream"]
        yy = lane_y[lane]
        xx, ww = x(event["start"]), max(.7, (event["end"] - event["start"]) / 1e6 * scale)
        hh = 14 if lane in {"D", "M"} else 26
        offset = 7 if lane in {"D", "M"} else 0
        parts.append(f'<rect x="{xx:.3f}" y="{yy-hh/2+offset:.2f}" width="{ww:.3f}" height="{hh}" fill="{colors.get(stage,"#7c8794")}"><title>{html.escape(stage)} frame={event["frame"]} {(event["end"]-event["start"])/1000:.3f} us</title></rect>')
        if lane in {"COPY", "JPEG_SM"}:
            parts.append(f'<text x="{xx:.3f}" y="{yy-18}" font-size="11" text-anchor="middle">{event["frame"]}</text>')
    for key in sorted(frame_ranges):
        if key[0] != mode or key[1] not in wanted:
            continue
        for source, short, color in [(frame_ranges, "E", "#63778e"), (receive_ranges, "R", "#b2bccc")]:
            r = source.get(key)
            if r is None:
                continue
            xx, ww = x(r["start"]), max(.7, (r["end"]-r["start"])/1e6*scale)
            parts.append(f'<rect x="{xx:.3f}" y="{lane_y["CPU"]-18}" width="{ww:.3f}" height="36" fill="{color}" rx="2"/>')
            if ww > 23:
                parts.append(f'<text x="{xx+ww/2:.3f}" y="{lane_y["CPU"]+5}" font-size="12" fill="#ffffff" text-anchor="middle">{short}{key[1]}</text>')
    for stage, label in [("F05_motion_image", "MI ready"), ("F04_postprocess", "CS ready")]:
        r = next(r for r in stages if r["frame"] == first_frame and r["stage"] == stage)
        xx = x(r["end_ns"])
        parts.append(f'<line x1="{xx:.2f}" y1="{lane_y["D"]+25}" x2="{xx:.2f}" y2="{lane_y["M"]-24}" stroke="#5b6878" stroke-dasharray="3 3" marker-end="url(#arrow)"/>')
        parts.append(f'<text x="{xx+5:.2f}" y="{(lane_y["D"]+lane_y["M"])/2+4}" font-size="11" fill="#5b6878">{label}</text>')
    parts.append(f'<text x="{(left+right)/2}" y="675" text-anchor="middle" font-size="15">Milliseconds from frame {first_frame} enqueue</text>')
    legend = [
        ("F01_compressed_H2D", "JPEG H2D"), ("F01_nvJPEG_SM_conversion", "nvJPEG SM conversion"),
        ("F02_resize", "F02 resize"), ("F05_motion_image", "F05 motion image"),
        ("F03_engine", "F03 engine"), ("F03_producer", "Producer"), ("F04_postprocess", "F04"),
        ("F06_GMC_including_cuFFT", "GMC / cuFFT"), ("F08a_predict", "Predict"),
        ("F04b_batch", "F04b"), ("F07_F08b_update", "Update"), ("packet_D2H", "Packet D2H"),
    ]
    for i, (stage, label) in enumerate(legend):
        xx, yy = 50 + (i % 6) * 285, 709 + (i // 6) * 27
        parts.append(f'<rect x="{xx}" y="{yy-11}" width="15" height="15" fill="{colors[stage]}"/>')
        parts.append(f'<text x="{xx+23}" y="{yy+1}" font-size="13">{label}</text>')
    parts.extend([
        '<text x="50" y="782" font-size="14">Pale bars = stage envelopes. Dark bars = actual kernel / copy activity. Small control kernels are gray.</text>',
        '<text x="50" y="808" font-size="14">Hardware JPEG decoder activity is not exposed here. D, M and nvJPEG conversion share the same GPU SM resources.</text>',
        '<text x="50" y="834" font-size="14">Profiled timings include tracing overhead. Activity overlap is not GPU utilization or an isolated kernel cost.</text>',
        '</g></svg>',
    ])
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    svg = prefix.with_suffix(".svg")
    svg.write_text("\n".join(parts) + "\n")
    subprocess.run(["rsvg-convert", "-o", str(prefix.with_suffix(".png")), str(svg)], check=True)


def analyze(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(f"file:{Path(args.sqlite).resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    failures = []
    ranges = []
    for row in db.execute(
        "SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value) AS label "
        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId "
        "WHERE n.end IS NOT NULL ORDER BY n.start"
    ):
        ranges.append(dict(row))
    modes = [r for r in ranges if re.fullmatch(r"mode:[\w-]+", r["label"] or "")]
    frames_by_tid = collections.defaultdict(list)
    frame_ranges = {}
    receive_ranges = {}
    for r in ranges:
        match = re.fullmatch(r"frame:(\d+):(enqueue|receive)", r["label"] or "")
        if not match:
            continue
        containers = [m for m in modes if m["globalTid"] == r["globalTid"] and m["start"] <= r["start"] and r["end"] <= m["end"]]
        if len(containers) != 1:
            failures.append(f"frame range has {len(containers)} mode containers: {r['label']}")
            continue
        mode, frame, kind = containers[0]["label"][5:], int(match[1]), match[2]
        r.update(mode=mode, frame=frame)
        target = frame_ranges if kind == "enqueue" else receive_ranges
        if (mode, frame) in target:
            failures.append(f"duplicate {mode} frame={frame} {kind} range")
        target[mode, frame] = r
        if kind == "enqueue":
            frames_by_tid[r["globalTid"]].append(r)
    for values in frames_by_tid.values():
        values.sort(key=lambda r: r["start"])
    starts_by_tid = {tid: [r["start"] for r in values] for tid, values in frames_by_tid.items()}

    def frame_for_api(tid, start):
        values = frames_by_tid.get(tid, [])
        index = bisect.bisect_right(starts_by_tid.get(tid, []), start) - 1
        if index >= 0 and start <= values[index]["end"]:
            r = values[index]
            return r["mode"], r["frame"]
        return None

    runtime = {}
    for row in db.execute(
        "SELECT r.*,s.value AS api FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON s.id=r.nameId ORDER BY r.start"
    ):
        d = dict(row)
        d["frame_key"] = frame_for_api(d["globalTid"], d["start"])
        runtime[d["correlationId"]] = d
    selected_keys = sorted(key for key in frame_ranges if args.warmup <= key[1] < args.warmup + args.frames)
    mode_names = sorted({m["label"][5:] for m in modes})
    if not mode_names:
        failures.append("no mode NVTX ranges")
    for mode in mode_names:
        found = {frame for m, frame in selected_keys if m == mode}
        wanted = set(range(args.warmup, args.warmup + args.frames))
        if found != wanted:
            failures.append(f"{mode}: frame ranges differ from expected sample IDs")
    by_frame = collections.defaultdict(list)
    all_kernels = []
    unmapped_kernels = []
    for row in db.execute(
        "SELECT k.*,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start"
    ):
        k = dict(row)
        api = runtime.get(k["correlationId"])
        key = api["frame_key"] if api else None
        if key is None:
            unmapped_kernels.append(k)
            continue
        k["mode"], k["frame"] = key
        k["api"] = api["api"]
        k["kind"] = "kernel"
        all_kernels.append(k)
        if key in selected_keys:
            by_frame[key].append(k)
    packet_copies = collections.defaultdict(list)
    jpeg_copies = collections.defaultdict(list)
    extra_copies = []
    for row in db.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"):
        c = dict(row)
        api = runtime.get(c["correlationId"])
        key = api["frame_key"] if api else None
        if key not in selected_keys:
            continue
        c["mode"], c["frame"] = key
        c["kind"] = "memcpy"
        copy_kind = {0: "Unknown", 1: "HostToDevice", 2: "DeviceToHost", 3: "HostToArray",
                     4: "ArrayToHost", 5: "ArrayToArray", 6: "ArrayToDevice", 7: "DeviceToArray",
                     8: "DeviceToDevice", 9: "HostToHost", 10: "PeerToPeer"}.get(c["copyKind"], f"copyKind={c['copyKind']}")
        c["name"] = f"{c['bytes']}-byte {copy_kind}"
        # CUDA copy kind 2 is DeviceToHost, independent of memory-kind enum.
        if c["bytes"] == 12368 and c["copyKind"] == 2:
            c["stage"], c["logical_stream"] = "packet_D2H", "M"
            packet_copies[key].append(c)
        elif c["copyKind"] == 1:
            c["stage"], c["logical_stream"] = "F01_compressed_H2D", "JPEG_COPY"
            jpeg_copies[key].append(c)
        else:
            extra_copies.append(c)

    events = []
    per_frame_stages = []
    frame_metrics = []
    resource_groups = collections.defaultdict(list)
    stage_intervals = collections.defaultdict(list)
    side_intervals = collections.defaultdict(list)
    frame_side_intervals = collections.defaultdict(list)
    graph_metadata = collections.defaultdict(set)
    ordering_checks = collections.Counter()
    for key in selected_keys:
        mode, frame = key
        kernels = by_frame[key]
        if not kernels:
            failures.append(f"{key}: no mapped kernels")
            continue
        d_graph_correlations = {
            k["correlationId"] for k in kernels
            if custom_stage(k["name"])[0] == "F03_producer" and k.get("graphId")
        }
        first_producer = min((k["start"] for k in kernels if custom_stage(k["name"])[0] == "F03_producer"), default=-1)
        last_image = max((k["end"] for k in kernels if custom_stage(k["name"])[0] == "F05_motion_image"), default=-1)
        direct_d_streams = {k["streamId"] for k in kernels if custom_stage(k["name"])[0] == "F02_resize"}
        by_stage = collections.defaultdict(list)
        for k in kernels:
            stage, side = custom_stage(k["name"])
            if stage is None and (k["correlationId"] in d_graph_correlations or
                                  (k["streamId"] in direct_d_streams and last_image <= k["start"] < first_producer)):
                stage, side = "F03_engine", "D"
            if stage is None:
                failures.append(f"{key}: unclassified kernel: {k['name']}")
                continue
            k["stage"], k["logical_stream"] = stage, side
            by_stage[stage].append(k)
            events.append(k)
            interval = k["start"], k["end"]
            stage_intervals[mode, stage].append(interval)
            side_intervals[mode, side].append(interval)
            frame_side_intervals[mode, frame, side].append(interval)
            resource_groups[mode, stage, k["name"]].append(k)
            if k.get("graphId"):
                graph_metadata[mode, side].add((k["graphId"], k["streamId"]))
        by_stage["packet_D2H"] = packet_copies[key]
        by_stage["F01_compressed_H2D"] = jpeg_copies[key]
        for stage, copies in [("packet_D2H", packet_copies[key]), ("F01_compressed_H2D", jpeg_copies[key])]:
            for c in copies:
                events.append(c)
                stage_intervals[mode, stage].append((c["start"], c["end"]))
        for stage, count in EXPECTED.items():
            actual = len(by_stage[stage])
            if actual != count:
                failures.append(f"{key}: {stage} has {actual} activities; expected {count}")
        for stage in {"F01_nvJPEG_SM_conversion", "F03_engine", "F06_GMC_including_cuFFT"}:
            if not by_stage[stage]:
                failures.append(f"{key}: missing {stage}")
        # These are observable necessary ready/commit edges, not a proof of
        # arbitrary memory safety. They also reject a misassigned graph frame.
        for before, after in [
            ("F02_resize", "F05_motion_image"),
            ("F01_compressed_H2D", "F01_nvJPEG_SM_conversion"),
            ("F01_nvJPEG_SM_conversion", "F02_resize"),
            ("F05_motion_image", "F03_engine"),
            ("F03_engine", "F03_producer"),
            ("F03_producer", "F04_postprocess"),
            ("F05_motion_image", "F06_GMC_including_cuFFT"),
            ("F06_GMC_including_cuFFT", "F08a_predict"),
            ("F08a_predict", "M_candidates"),
            ("F04_postprocess", "M_candidates"),
            ("F04b_batch", "F07_F08b_update"),
            ("F07_F08b_update", "M_commit"),
            ("M_commit", "M_packet"),
            ("M_packet", "packet_D2H"),
        ]:
            if by_stage[before] and by_stage[after]:
                ordering_checks[f"{before} -> {after}"] += 1
                if max(v["end"] for v in by_stage[before]) > min(v["start"] for v in by_stage[after]):
                    failures.append(f"{key}: observable ordering violated: {before} -> {after}")
        for stage in STAGES:
            values = by_stage[stage]
            if not values:
                continue
            start = min(v["start"] for v in values)
            end = max(v["end"] for v in values)
            per_frame_stages.append({
                "mode": mode, "frame": frame, "stage": stage,
                "logical_stream": values[0]["logical_stream"],
                "profiler_stream_ids": ";".join(map(str, sorted({v["streamId"] for v in values}))),
                "activities": len(values), "start_ns": start, "end_ns": end,
                "transfer_bytes": sum(v.get("bytes", 0) for v in values),
                "busy_us": duration((v["start"], v["end"]) for v in values) / 1000,
                "envelope_us": (end - start) / 1000,
            })
        d = frame_ranges[key]
        receive = receive_ranges.get(key)
        f02 = by_stage["F02_resize"]
        f04 = by_stage["F04_postprocess"]
        predict = by_stage["F08a_predict"]
        candidates = by_stage["M_candidates"]
        f06 = by_stage["F06_GMC_including_cuFFT"]
        engine = by_stage["F03_engine"]
        row = {"mode": mode, "frame": frame, "enqueue_start_ns": d["start"],
               "cpu_enqueue_us": (d["end"] - d["start"]) / 1000,
               "cpu_receive_us": (receive["end"] - receive["start"]) / 1000 if receive else None}
        if f02:
            row["enqueue_to_F02_start_us_includes_host_and_queue_waits"] = (min(v["start"] for v in f02) - d["start"]) / 1000
        if packet_copies[key]:
            row["enqueue_to_packet_D2H_end_us"] = (max(v["end"] for v in packet_copies[key]) - d["start"]) / 1000
        if predict and candidates:
            row["predict_end_to_candidates_start_us"] = (min(v["start"] for v in candidates) - max(v["end"] for v in predict)) / 1000
        if f04 and candidates:
            row["F04_end_to_candidates_start_us"] = (min(v["start"] for v in candidates) - max(v["end"] for v in f04)) / 1000
        row["same_frame_F03_F06_kernel_overlap_us"] = intersection(
            [(v["start"], v["end"]) for v in engine],
            [(v["start"], v["end"]) for v in f06]) / 1000
        row["same_frame_D_M_kernel_overlap_us"] = intersection(
            frame_side_intervals[mode, frame, "D"], frame_side_intervals[mode, frame, "M"]) / 1000
        frame_metrics.append(row)

    # A missing correlation inside a measured device window would silently
    # undercount activity. Initialization outside these windows is excluded.
    for mode in mode_names:
        selected_events = [e for e in events if e["mode"] == mode]
        if not selected_events:
            continue
        low, high = min(e["start"] for e in selected_events), max(e["end"] for e in selected_events)
        for k in unmapped_kernels:
            if low <= k["start"] < high:
                failures.append(f"{mode}: kernel in sampled window lacks frame correlation: {k['name']}")

    for mode in mode_names:
        for stage in ["F03_engine", "F06_GMC_including_cuFFT"]:
            counts = {r["activities"] for r in per_frame_stages if r["mode"] == mode and r["stage"] == stage}
            if len(counts) != 1:
                failures.append(f"{mode}: {stage} activity count changed across sampled frames: {sorted(counts)}")
    summaries = []
    for mode in mode_names:
        for stage in STAGES:
            rows = [r for r in per_frame_stages if r["mode"] == mode and r["stage"] == stage]
            summaries.append({"mode": mode, "stage": stage,
                              "busy_us": stats([r["busy_us"] for r in rows]),
                              "envelope_us": stats([r["envelope_us"] for r in rows]),
                              "transfer_bytes": stats([r["transfer_bytes"] for r in rows]),
                              "activities_per_frame": sorted({r["activities"] for r in rows})})
    resources = []
    for (mode, stage, name), values in sorted(resource_groups.items()):
        first = values[0]
        resources.append({
            "mode": mode, "stage": stage, "name": name, "calls": len(values),
            "mean_us": statistics.fmean((v["end"] - v["start"]) / 1000 for v in values),
            "total_us": sum(v["end"] - v["start"] for v in values) / 1000,
            "registers_per_thread_max": max(v["registersPerThread"] for v in values),
            "static_shared_bytes_max": max(v["staticSharedMemory"] for v in values),
            "dynamic_shared_bytes_max": max(v["dynamicSharedMemory"] for v in values),
            "local_bytes_per_thread_max": max(v["localMemoryPerThread"] for v in values),
            "grid_shapes": ";".join(sorted({f"{v['gridX']}x{v['gridY']}x{v['gridZ']}" for v in values})),
            "block_shapes": ";".join(sorted({f"{v['blockX']}x{v['blockY']}x{v['blockZ']}" for v in values})),
            "profiler_stream_ids": ";".join(map(str, sorted({v["streamId"] for v in values}))),
        })
    overlaps = []
    side_summaries = []
    for mode in mode_names:
        for i, stage_a in enumerate(STAGES):
            for stage_b in STAGES[i + 1:]:
                value = intersection(stage_intervals[mode, stage_a], stage_intervals[mode, stage_b])
                if value:
                    overlaps.append({"mode": mode, "stage_a": stage_a, "stage_b": stage_b,
                                     "overlap_us": value / 1000, "overlap_us_per_sample_frame": value / (1000 * args.frames)})
        d, m = side_intervals[mode, "D"], side_intervals[mode, "M"]
        overlap = intersection(d, m)
        next_frame_overlap = []
        for frame in range(args.warmup, args.warmup + args.frames - 1):
            next_frame_overlap.append(intersection(frame_side_intervals[mode, frame, "M"], frame_side_intervals[mode, frame + 1, "D"]) / 1000)
        side_summaries.append({
            "mode": mode, "D_kernel_busy_us": duration(d) / 1000,
            "M_kernel_busy_us": duration(m) / 1000,
            "D_M_kernel_overlap_us": overlap / 1000,
            "D_M_overlap_us_per_sample_frame": overlap / (1000 * args.frames),
            "fraction_of_M_kernel_busy_overlapped_by_D": overlap / duration(m) if duration(m) else 0,
            "M_k_D_kplus1_kernel_overlap_us": stats(next_frame_overlap),
        })
    frame_fields = sorted({field for row in frame_metrics for field in row}, key=lambda f: (f not in {"mode", "frame"}, f))
    write_csv(out / "stage_samples.csv", per_frame_stages)
    write_csv(out / "frame_metrics.csv", frame_metrics, frame_fields)
    write_csv(out / "kernel_resources.csv", resources)
    write_csv(out / "stage_overlap.csv", overlaps, ["mode", "stage_a", "stage_b", "overlap_us", "overlap_us_per_sample_frame"])
    summary_rows = []
    for row in summaries:
        flattened = {"mode": row["mode"], "stage": row["stage"], "samples": row["busy_us"]["n"]}
        for metric in ["busy_us", "envelope_us"]:
            for stat in ["mean", "p50", "p95", "p99", "min", "max"]:
                flattened[metric + "_" + stat] = row[metric].get(stat)
        summary_rows.append(flattened)
    write_csv(out / "stage_summary.csv", summary_rows)
    summary = {
        "schema": "saccade-stage-schedule/1", "status": "FAIL" if failures else "PASS",
        "sqlite": str(Path(args.sqlite).resolve()), "warmup": args.warmup,
        "samples_per_mode": args.frames, "modes": mode_names,
        "timing_scope": "Nsight-profiled GPU kernel activity; not unprofiled wall latency",
        "busy_definition": "union of kernel or copy intervals within a stage/frame",
        "envelope_definition": "first activity start to last activity end, including intervening gaps",
        "F01_scope": "compressed HostToDevice transfer and CUDA SM color conversion separately; hardware JPEG decoder duration is not visible in CUDA activity",
        "input_availability_scope": "enqueue start to F02 start includes host previous-RGB wait, staging/header work, decoder and stream queue waits; not isolated F01 latency",
        "logical_stream_scope": "graph kernels may have synthetic profiler stream IDs; logical D/M follows graph content and launch frame correlation",
        "stage_summaries": summaries, "overlap": side_summaries,
        "graph_profiler_streams": [{"mode": mode, "logical_stream": side, "graph_stream_pairs": sorted(pairs)} for (mode, side), pairs in sorted(graph_metadata.items())],
        "observable_ordering_checks": dict(sorted(ordering_checks.items())),
        "ordering_scope": "necessary same-frame ready and commit edges only; does not establish arbitrary memory safety or lifetime safety",
        "extra_sampled_memcopies": extra_copies,
        "failures": sorted(set(failures)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "timeline.json").write_text(json.dumps({"units": "ns", "rows": per_frame_stages}, indent=2) + "\n")
    if args.plot_prefix and not failures:
        render_schedule(args.plot_prefix, events, per_frame_stages, frame_ranges, receive_ranges, "queued", args.warmup)
    print(json.dumps({"status": summary["status"], "modes": mode_names,
                      "sampled_frames": len(selected_keys), "kernels": sum(len(v) for v in by_frame.values()),
                      "failures": summary["failures"][:20], "output": str(out)}, indent=2))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("output")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--plot-prefix", help="write .svg/.png schedule (requires rsvg-convert); fixed first five queued sample frames")
    args = parser.parse_args()
    if args.warmup < 1 or args.frames < 1:
        parser.error("positive warmup and frames required")
    try:
        return analyze(args)
    except (sqlite3.Error, OSError, ValueError, KeyError) as error:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        (Path(args.output) / "summary.json").write_text(json.dumps({"status": "FAIL", "error": str(error)}, indent=2) + "\n")
        print(f"analysis failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

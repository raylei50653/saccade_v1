#!/usr/bin/env python3
"""Read-only NVDEC/CUDA trace audit; CPU API time is not GPU engine activity.

python build/streaming-nvdec/analyze_trace.py TRACE.sqlite --frames 3
An optional --output FILE is created exclusively; the input DB is never written.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import sqlite3


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def group_copies(copies):
    groups = collections.defaultdict(lambda: {"count": 0, "bytes": 0, "gpu_duration_ns": 0})
    for copy in copies:
        key = (copy["kind"], copy["bytes"], copy["streamId"])
        group = groups[key]
        group["count"] += 1
        group["bytes"] += copy["bytes"]
        group["gpu_duration_ns"] += copy["end"] - copy["start"]
    return [{"kind": kind, "bytes_each": size, "streamId": stream, **value}
            for (kind, size, stream), value in sorted(groups.items())]


def analyze(path, frames, motion_bytes):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"StringIds", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "ENUM_CUDA_MEMCPY_OPER"}
    if not required <= tables:
        raise ValueError("trace lacks required CUDA activity tables: " + str(sorted(required - tables)))
    names = dict(db.execute("SELECT id,value FROM StringIds"))
    kind_labels = dict(db.execute("SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER"))
    kind_names = dict(db.execute("SELECT id,name FROM ENUM_CUDA_MEMCPY_OPER"))
    kernels = [dict(row) for row in db.execute("SELECT start,end,streamId,correlationId,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")]
    for kernel in kernels:
        kernel["name"] = names[kernel.pop("demangledName")]
    colors = [kernel for kernel in kernels if "saccade::" in kernel["name"] and
              "convert<" in kernel["name"] and "Coefficients" in kernel["name"]]
    if not colors:
        raise ValueError("no native device-video color conversion kernel in trace")
    boundary = colors[0]["start"]

    api_by_correlation = collections.defaultdict(list)
    api_summary = collections.Counter()
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables:
            continue
        for row in db.execute(f'SELECT * FROM "{table}"'):
            api = dict(row)
            api["name"] = names[api["nameId"]]
            api["table"] = table
            api_by_correlation[api["correlationId"]].append(api)
    color_calls = api_by_correlation.get(colors[0]["correlationId"], [])
    api_boundary = min((api["start"] for api in color_calls), default=None)
    if api_boundary is not None:
        for apis in api_by_correlation.values():
            for api in apis:
                if api["start"] >= api_boundary and any(term in api["name"].lower() for term in ("memcpy", "memset")):
                    api_summary[(api["table"], api["name"])] += 1

    copies = []
    for row in db.execute("SELECT start,end,streamId,correlationId,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"):
        copy = dict(row)
        copy["kind"] = kind_labels[copy["copyKind"]]
        copy["kind_enum"] = kind_names[copy["copyKind"]]
        copy["apis"] = [{"table": api["table"], "name": api["name"], "start": api["start"],
                         "end": api["end"], "returnValue": api.get("returnValue")}
                        for api in api_by_correlation.get(copy["correlationId"], [])]
        copies.append(copy)
    after = [copy for copy in copies if copy["start"] >= boundary]
    before = [copy for copy in copies if copy["start"] < boundary]
    h2d = [copy for copy in after if copy["copyKind"] in (1, 3, 11)]
    d2h = [copy for copy in after if copy["copyKind"] in (2, 4, 12)]
    unexpected_d2h = [copy for copy in d2h if copy["bytes"] not in (12368, motion_bytes)]
    expected_counts = {12368: frames, motion_bytes: frames}
    actual_counts = collections.Counter(copy["bytes"] for copy in d2h)

    video_apis = []
    video_metadata = []
    video_tables = sorted(table for table in tables if "NVVIDEO" in table.upper())
    for table in video_tables:
        columns = {row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
        if not {"start", "end", "nameId"} <= columns:
            continue
        # These are host API calls. Their duration cannot establish hardware
        # decode-engine busy time or overlap with CUDA kernels.
        for row in db.execute(f'SELECT nameId,count(*) calls,sum(end-start) total_ns,min(start) first_ns,max(end) last_ns FROM "{table}" GROUP BY nameId'):
            if names[row["nameId"]] == "GpuMetrics":
                video_metadata.append({"table": table, "name": "GpuMetrics", "records": row["calls"]})
                continue
            video_apis.append({"table": table, "name": names[row["nameId"]], "calls": row["calls"],
                               "host_api_total_ns": row["total_ns"], "first_ns": row["first_ns"], "last_ns": row["last_ns"]})
    decode_calls = sum(row["calls"] for row in video_apis if "DecodePicture" in row["name"])
    checks = {
        "color_kernel_count_matches_frames": len(colors) == frames,
        "nvvideo_decode_picture_api_recorded": decode_calls > 0,
        "no_h2d_activity_after_first_color": not h2d,
        "no_d2h_except_packet_and_motion_after_first_color": not unexpected_d2h,
        "packet_and_motion_d2h_counts_match_frames": dict(actual_counts) == expected_counts,
    }
    result = {
        "database": str(path.resolve()), "database_sha256": sha256(path),
        "formal_acceptance": False,
        "scope": "observed CUDA activity after the first native video color kernel; initialization separate",
        "first_color_kernel_start_ns": boundary, "first_color_host_api_start_ns": api_boundary,
        "first_color_gpu_minus_api_start_ns": boundary - api_boundary if api_boundary is not None else None,
        "color_kernels": colors, "expected_frames": frames,
        "motion_context_bytes": motion_bytes,
        "all_gpu_copy_summary": group_copies(copies),
        "initialization_gpu_copy_summary": group_copies(before),
        "after_first_color_gpu_copy_summary": group_copies(after),
        "after_first_color_gpu_copy_details": after,
        "post_boundary_cpu_mem_api_summary": [{"table": table, "name": name, "calls": count}
                                               for (table, name), count in sorted(api_summary.items())],
        "nvvideo_tables": video_tables,
        "nvvideo_host_api_summary": video_apis,
        "nvvideo_metadata_records": video_metadata,
        "checks": checks, "observed_transfer_contract_pass": all(checks.values()),
        "limitations": [
            "CUDA runtime/driver API durations and GPU activity durations are separate observations.",
            "NvVideo API intervals identify calls, not hardware decode engine utilization or measured overlap.",
            "D2H attribution uses observed transfer size plus expected source copies; source does not download pixels.",
            "This is not full-library race/synchronization safety or cross-environment support proof.",
            "First-conversion boundary excludes setup copies and the first decode before session construction.",
            "Host/GPU timestamp alignment is finite; cross-domain microsecond offsets are not causal-order proof.",
        ],
    }
    db.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--motion-bytes", type=int, default=56,
                        help="sizeof(MotionContext), compiled locally from include/saccade/motion.hpp")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.database, args.frames, args.motion_bytes)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as output:
            output.write(payload)
    print(payload)


if __name__ == "__main__":
    main()

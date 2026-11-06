#!/usr/bin/env python3
"""Inventory Compute Sanitizer launch dumps; completion comes from capture evidence.

Usage: python3 analyze_launch_inventory.py --output NEW_DIRECTORY LOG [LOG ...]
Use --provisional for snapshots of running or incomplete captures. An inventory is
an observation of launches, never a claim that those launches finished checking.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re


LAUNCH = re.compile(r"^Launch #(?P<number>\d+)(?P<suffix>.*)$")
DIMS = re.compile(r"^\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$")
SOURCES = (
    ("tracking_session_cu_", "custom/tracking-session"),
    ("selective_scan_cu_", "custom/scan"),
    ("model_producer_cu_", "custom/producer"),
    ("postprocess_cu_", "custom/postprocess"),
    ("flow_motion_cu_", "custom/flow-motion"),
    ("tracking_cu_", "custom/tracking"),
    ("resize_cu_", "custom/resize"),
    ("motion_cu_", "custom/motion"),
    ("video_preprocess_cu_", "custom/video-preprocess"),
)
VENDORS = (("libnvidia-opticalflow.so", "nvidia-opticalflow"),
           ("libnvjpeg.so", "nvjpeg"), ("libnvinfer.so", "nvinfer"))
INTERNAL = re.compile(r"Internal (?:Sanitizer|Racecheck|Memcheck|Synccheck|Initcheck)|"
                      r"didn't track|did not track|not tracked|Unable to allocate|"
                      r"errors might go undetected|records might be incomplete", re.I)


def json_file(path):
    return json.loads(path.read_text()) if path.is_file() else None


def filter_flag(suffix):
    """A bare observed launch label is the tool's checked-launch form."""
    suffix = suffix.strip()
    if not suffix:
        return False, "bare-launch-label"
    match = re.fullmatch(r"[\[(]\s*filtered(?:\s*[:=]\s*(yes|no|true|false|0|1))?\s*[\])]", suffix, re.I)
    if match:
        return match[1] is None or match[1].lower() in ("yes", "true", "1"), "explicit-filter-label"
    if re.fullmatch(r"[\[(]\s*unfiltered\s*[\])]", suffix, re.I):
        return False, "explicit-unfiltered-label"
    return None, "unknown-launch-label-suffix"


def classify(record):
    kernel = record.get("kernel", "")
    libraries = sorted({vendor for fragment, vendor in VENDORS
                        if any(fragment in frame for frame in record["host_frames"])})
    record["host_libraries"] = libraries
    for fragment, label in SOURCES:
        if fragment in kernel:
            return label, "mangled-source-tag:" + fragment
    if len(libraries) == 1:
        return libraries[0], "host-backtrace-library"
    if libraries:
        return "ambiguous-vendor", "multiple-host-libraries"
    return "unknown", "no-supported-source-or-library-evidence"


def parse_launches(text):
    records = []
    current = None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("========="):
            continue
        body = line[9:].strip()
        launch = LAUNCH.fullmatch(body)
        if launch:
            filtered, evidence = filter_flag(launch["suffix"])
            current = {"launch_number": int(launch["number"]), "line": number,
                       "raw_launch_label": body, "filtered": filtered,
                       "filtered_evidence": evidence, "host_frames": [], "extra_metadata": []}
            records.append(current)
            continue
        if current is None:
            continue
        if not body:
            current = None
        elif body.startswith("Kernel:"):
            current["kernel"] = body.split(":", 1)[1].strip()
        elif body.startswith("Host Frame:"):
            current["host_frames"].append(body.removeprefix("Host Frame:").strip())
        elif re.match(r"^(Grid|Block|Cluster(?: size)?):", body, re.I):
            field, value = body.split(":", 1)
            match = DIMS.fullmatch(value.strip())
            key = "cluster" if field.lower().startswith("cluster") else field.lower()
            current[key] = list(map(int, match.groups())) if match else None
            if not match:
                current["extra_metadata"].append(body)
        elif body.startswith("Stream:"):
            current["stream"] = body.split(":", 1)[1].strip()
        elif "graph" in body.lower() and not body.startswith(("ERROR", "WARNING", "Error")):
            current["extra_metadata"].append(body)
        elif not body.startswith("Saved host backtrace up to driver entry point at kernel launch time"):
            # Subsequent API/device-error backtraces are not launch ownership.
            current = None
    for record in records:
        record.setdefault("cluster", None)
        record["classification"], record["classification_basis"] = classify(record)
        record["record_complete"] = all(record.get(key) for key in ("kernel", "grid", "block", "stream")) and record["filtered"] is not None
    return records


def diagnostic_inventory(text):
    """Separate actual diagnostic headers from dump metadata, without changing status."""
    reports, unknown = [], []
    current = None
    for number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("========="):
            continue
        body = line[9:].strip()
        if not body:
            current = None
            continue
        if LAUNCH.fullmatch(body):
            current = None
            continue
        if body.startswith("Host Frame:"):
            if current is not None:
                current["host_frames"].append(body.removeprefix("Host Frame:").strip())
            continue
        if re.match(r"^(?:Kernel:|Grid:|Block:|Cluster(?: size)?:|Stream:|Saved host backtrace|COMPUTE-SANITIZER|ERROR SUMMARY:|RACECHECK SUMMARY:)", body):
            continue
        if re.match(r"^(?:Program hit |Invalid |Uninitialized |(?:ERROR|WARNING|Error|Warning):|Barrier error detected|Internal )", body):
            current = {"header": body, "line": number, "host_frames": []}
            reports.append(current)
        elif current is not None:
            current.setdefault("continuation", []).append(body)
        else:
            unknown.append({"line": number, "text": body})
    for report in reports:
        report["host_libraries"] = sorted({vendor for fragment, vendor in VENDORS
            if any(fragment in frame for frame in report["host_frames"])})
    return {"diagnostic_report_total": len(reports),
            "diagnostic_header_counts": dict(Counter(r["header"] for r in reports)),
            "reports": reports, "unknown_nonmetadata_lines": unknown,
            "interpretation": "Diagnostic inventory only. ERROR SUMMARY and exit status remain literal; no launch-count subtraction or acceptance promotion."}


def inferred_owners(records):
    """Keep graph-owner inference distinct from directly observed host evidence."""
    known = {}
    for record in records:
        if record["classification"] not in ("unknown", "ambiguous-vendor"):
            known.setdefault(record["kernel"], {}).setdefault(record["classification"], []).append(record["launch_number"])
    for record in records:
        record["inferred_owner"] = record["classification"]
        record["owner_inference_basis"] = "direct-classification"
        record["owner_witness_launch_numbers"] = []
        if record["classification"] != "unknown":
            continue
        candidates = known.get(record["kernel"], {})
        if len(candidates) == 1:
            owner = next(iter(candidates))
            record["inferred_owner"] = owner
            record["owner_inference_basis"] = "same-exact-kernel-name-as-direct-owner-witness;module-identity-not-proven"
            record["owner_witness_launch_numbers"] = candidates[owner][:8]
        elif record["kernel"].startswith("_ZN3cub"):
            record["inferred_owner"] = "cuda-cub"
            record["owner_inference_basis"] = "mangled-cub-namespace;host-library-not-established"


def execution_evidence(log, text):
    execution_path = log.with_name(log.stem + ".execution.json")
    execution = json_file(execution_path)
    target_file = json_file(log.with_name(log.stem + ".target.json"))
    command = json_file(log.with_name(log.stem + ".command.json")) or {}
    dump_enabled = "--dump-kernel-launches" in command.get("target", [])
    target = (execution or {}).get("target", target_file or {})
    summaries = re.findall(r"^========= (?:RACECHECK SUMMARY:.*|ERROR SUMMARY:.*)$", text, re.M)
    applications = [{"frames": int(n), "mode": mode} for n, mode in
                    re.findall(r"^track_sequence frames=(\d+) mode=(\S+) wall_ms=", text, re.M)]
    controls = re.findall(r"^control=(\S+) launch=(\d+) completion=(\d+)\s*$", text, re.M)
    memory_events = [(execution or {}).get("sampled_memory_events", {})]
    for field in ("memory.events", "memory.events.local"):
        memory_events.append(dict(line.split() for line in target.get(field, "").splitlines()
                                  if len(line.split()) == 2))
    resource_failure = bool((execution or {}).get("outer_timeout") or
        (execution or {}).get("returncode") in (124, 137, -9) or target.get("returncode") in (124, 137, -9) or
        any(int(events.get(key, 0)) for events in memory_events for key in ("oom", "oom_kill", "oom_group_kill")) or
        any(value in ("oom-kill", "timeout", "resources") for value in (execution or {}).get("sampled_systemd_results", [])))
    expected = (json_file(log.parent / "manifest.json") or {}).get("frames")
    application_marker = len(applications) == 1 and (expected is None or applications[0]["frames"] == expected)
    control_marker = len(controls) == 1
    final_records = bool(execution and execution.get("finished") and target.get("finished") and
                         execution.get("returncode") in (0, 99) and target.get("returncode") in (0, 99))
    return {"execution_record": str(execution_path),
            "dump_kernel_launches_enabled": dump_enabled,
            "acceptance_classification": "INVENTORY_ONLY_NOT_ACCEPTANCE" if dump_enabled else "NOT_CLASSIFIED_BY_THIS_ANALYZER",
            "execution_record_present": execution is not None, "target_record_present": bool(target),
            "final_execution_records": final_records, "application_markers": applications,
            "control_markers": controls, "expected_frames": expected, "resource_failure": resource_failure,
            "internal_tool_error": bool(INTERNAL.search(text)), "sanitizer_summaries": summaries,
            "capture_completed": final_records and (application_marker or control_marker) and not resource_failure and not INTERNAL.search(text),
            "returncode": (execution or {}).get("returncode"), "target_returncode": target.get("returncode")}


def dims(value):
    return "x".join(map(str, value)) if value is not None else "not_reported"


def key(record):
    return (record.get("kernel", ""), dims(record.get("grid")), dims(record.get("block")), dims(record.get("cluster")))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    args.output.mkdir(parents=True)
    runs, launch_rows, count_rows = [], [], []
    for index, supplied in enumerate(args.logs):
        log = supplied.resolve()
        before = log.stat()
        data = log.read_bytes()
        after = log.stat()
        text = data.decode(errors="replace")
        records = parse_launches(text)
        inferred_owners(records)
        diagnostics = diagnostic_inventory(text)
        evidence = execution_evidence(log, text)
        counts = Counter((key(r), r["filtered"], r["classification"]) for r in records)
        label = f"{index:02d}-{log.stem}"
        summary = {"id": label, "log": str(log), "snapshot_sha256": hashlib.sha256(data).hexdigest(),
                   "snapshot_bytes": len(data), "size_changed_during_read": before.st_size != after.st_size,
                   "launched_total": len(records), "filtered_total": sum(r["filtered"] is True for r in records),
                   "unique_kernel_total": len({r.get("kernel", "") for r in records}),
                   "unique_kernel_dimension_total": len({key(r) for r in records}),
                   "filter_unknown_total": sum(r["filtered"] is None for r in records),
                   "checked_label_total": sum(r["filtered"] is False for r in records),
                   "incomplete_launch_records": sum(not r["record_complete"] for r in records),
                   "classification_counts": dict(Counter(r["classification"] for r in records)),
                   "inferred_owner_counts": dict(Counter(r["inferred_owner"] for r in records)),
                   "launch_numbers_sequential_from_one": bool(records) and [r["launch_number"] for r in records] == list(range(1, len(records) + 1)),
                   "completion": evidence,
                   "diagnostic_report_total": diagnostics["diagnostic_report_total"],
                   "diagnostic_header_counts": diagnostics["diagnostic_header_counts"],
                   "unknown_nonmetadata_line_count": len(diagnostics["unknown_nonmetadata_lines"]),
                   "scope": "Observed dump launch records only; a logged launch does not prove checking completed."}
        (args.output / f"{label}.json").write_text(json.dumps({"summary": summary, "launches": records, "diagnostics": diagnostics}, indent=2) + "\n")
        for record in records:
            launch_rows.append({"run": label, "launch_number": record["launch_number"], "kernel": record.get("kernel", ""),
                "grid": dims(record.get("grid")), "block": dims(record.get("block")), "cluster": dims(record.get("cluster")),
                "stream": record.get("stream", ""), "filtered": str(record["filtered"]).lower(),
                "classification": record["classification"], "classification_basis": record["classification_basis"],
                "inferred_owner": record["inferred_owner"], "owner_inference_basis": record["owner_inference_basis"],
                "host_libraries": ";".join(record["host_libraries"]), "record_complete": record["record_complete"]})
        for (signature, filtered, classification), count in sorted(counts.items(), key=lambda pair: str(pair[0])):
            count_rows.append(dict(zip(("run", "kernel", "grid", "block", "cluster", "filtered", "classification", "count"),
                                      (label, *signature, str(filtered).lower(), classification, count))))
        runs.append({"summary": summary, "counts": Counter(key(r) for r in records)})
    signatures = sorted({signature for run in runs for signature in run["counts"]})
    comparison_rows = []
    for signature in signatures:
        row = dict(zip(("kernel", "grid", "block", "cluster"), signature))
        row.update({run["summary"]["id"]: run["counts"][signature] for run in runs})
        row["counts_equal_across_logs"] = len(set(run["counts"][signature] for run in runs)) <= 1
        comparison_rows.append(row)
    report = {"schema": "sanitizer-launch-inventory/1", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "acceptance_classification": "INVENTORY_ONLY_NOT_ACCEPTANCE",
              "literal_summary_policy": "Dump launches can contribute to ERROR SUMMARY and exit99. Preserve literal counts and statuses; never subtract launch counts to create a PASS.",
              "provisional": args.provisional, "runs": [r["summary"] for r in runs],
              "all_captures_completed": bool(runs) and all(r["summary"]["completion"]["capture_completed"] for r in runs),
              "all_launch_records_complete": bool(runs) and all(r["summary"]["launched_total"] and not r["summary"]["incomplete_launch_records"] for r in runs),
              "all_inventory_counts_equal": all(row["counts_equal_across_logs"] for row in comparison_rows),
              "comparison_note": "Counts compare observations, not instrumentation success. Controls and different modes may legitimately differ. No full-session PASS is inferred."}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.output / "launches.csv", launch_rows, ("run", "launch_number", "kernel", "grid", "block", "cluster", "stream", "filtered", "classification", "classification_basis", "inferred_owner", "owner_inference_basis", "host_libraries", "record_complete"))
    write_csv(args.output / "counts.csv", count_rows, ("run", "kernel", "grid", "block", "cluster", "filtered", "classification", "count"))
    write_csv(args.output / "comparison.csv", comparison_rows, ("kernel", "grid", "block", "cluster", *[r["summary"]["id"] for r in runs], "counts_equal_across_logs"))
    print(json.dumps({"output": str(args.output), "provisional": args.provisional, "all_captures_completed": report["all_captures_completed"],
                      "runs": [{"log": Path(r["summary"]["log"]).name, "launches": r["summary"]["launched_total"],
                                "filtered": r["summary"]["filtered_total"], "complete": r["summary"]["completion"]["capture_completed"]} for r in runs]}))


if __name__ == "__main__":
    main()

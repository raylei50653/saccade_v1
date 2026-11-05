#!/usr/bin/env python3
"""Classify complete memcheck logs without suppressing unexpected diagnostics.

Examples:
  python3 classify_logs.py LOG... --output classification.json
  python3 classify_logs.py probe.log

PROBE BEGIN/END markers are observations of the host call active when a report
was printed; they do not establish where asynchronous device faults originated.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


MARKER = re.compile(r"PROBE (BEGIN|END) session=(\d+) phase=(\S+)(?: status=(\S+))?")
SUMMARY = re.compile(r"ERROR SUMMARY:\s*(\d+)\s+errors?\b")
LIB_FRAME = re.compile(r"\[(0x[0-9a-fA-F]+)\] in (\S+)")
MEMORY_DIAGNOSIS = re.compile(
    r"^(?:Invalid (?:__|global|shared|local|read|write|free)|Misaligned|"
    r"Uninitialized|Leak of|Malloc/Free error|Memory access error)|"
    r"CUDA_ERROR_(?:ILLEGAL_ADDRESS|MISALIGNED_ADDRESS)", re.IGNORECASE)
OTHER_DIAGNOSIS = re.compile(
    r"^(?:Program hit|Invalid|Misaligned|Uninitialized|Leak of|Malloc/Free error|"
    r"Memory access error|Error:|Warning:|Fatal)")
BOILERPLATE = ("COMPUTE-SANITIZER", "Saved host backtrace", "Saved device backtrace")


def parse(path):
    data = path.read_bytes()
    reports, summaries, markers, marker_issues = [], [], [], []
    target_output = []
    active = []
    current = None

    def start_report(message, line):
        kind = ("memory" if MEMORY_DIAGNOSIS.search(message) else
                "api" if message.startswith("Program hit") else "unknown")
        report = {"line": line, "message": message, "kind": kind,
                  "phase": active[-1].copy() if active else None,
                  "details": [], "library_stack": []}
        reports.append(report)
        return report

    for line_no, raw in enumerate(data.decode(errors="replace").splitlines(), 1):
        marker = MARKER.search(raw)
        if marker:
            action, session, phase, status = marker.groups()
            event = {"line": line_no, "action": action,
                     "session": int(session), "phase": phase}
            if status is not None:
                event["status"] = status
            markers.append(event)
            phase_key = {"session": int(session), "phase": phase}
            if action == "BEGIN":
                active.append(phase_key)
            elif active and active[-1] == phase_key:
                active.pop()
            else:
                marker_issues.append({"line": line_no, "message": "unmatched END",
                                      "marker": event, "active": active.copy()})
            # Markers normally occupy a dedicated line. Preserve any sanitizer
            # material before one if merged stdout/stderr interleaved a line.
            raw = raw[:marker.start()].rstrip()
            if not raw:
                continue

        if not raw.startswith("========="):
            if raw.strip():
                target_output.append({"line": line_no, "text": raw})
            continue
        message = raw[len("========="):].strip()
        if not message:
            current = None
            continue
        summary = SUMMARY.match(message)
        if summary:
            summaries.append({"line": line_no, "errors": int(summary[1]), "text": message})
            current = None
            continue
        if any(message.startswith(prefix) for prefix in BOILERPLATE):
            continue
        if OTHER_DIAGNOSIS.search(message):
            current = start_report(message, line_no)
            continue
        if current is None:
            current = start_report(message, line_no)
            continue
        current["details"].append(message)
        if message.startswith("Host Frame:"):
            frame = LIB_FRAME.search(message)
            if frame:
                offset, library = frame.groups()
                library = Path(library).name
                if library.startswith(("libnvcuvid.so", "libnvidia-opticalflow.so")):
                    current["library_stack"].append({"library": library, "offset": offset.lower()})

    if active:
        marker_issues.append({"message": "unclosed BEGIN markers", "active": active})
    families = {}
    for report in reports:
        stack = report["library_stack"]
        # A family preserves every library frame. Unstacked diagnoses cannot
        # accidentally be merged into an NVOF family.
        identity = json.dumps([report["message"], stack], sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()[:12]
        report["stack_family"] = key
        if key not in families:
            families[key] = {"id": key, "message": report["message"],
                             "kind": report["kind"], "library_stack": stack,
                             "count": 0, "first_line": report["line"]}
        families[key]["count"] += 1

    phase_counts = defaultdict(Counter)
    for event in markers:
        key = (str(event["session"]), event["phase"])
        for field in ("errors", "api_errors", "memory_errors", "unknown_errors"):
            phase_counts[key][field] += 0
    for report in reports:
        phase = report["phase"]
        key = (str(phase["session"]), phase["phase"]) if phase else (None, None)
        phase_counts[key]["errors"] += 1
        phase_counts[key][report["kind"] + "_errors"] += 1
    kinds = Counter(report["kind"] for report in reports)
    messages = Counter(report["message"] for report in reports)
    return {
        "path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data), "errors": len(reports),
        "api_errors": kinds["api"], "memory_errors": kinds["memory"],
        "unknown_diagnoses": kinds["unknown"],
        "messages": dict(messages), "summaries": summaries,
        "summary_matches_reports": (len(summaries) == 1 and
                                    summaries[0]["errors"] == len(reports)),
        "stack_families": list(families.values()),
        "phase_counts": [{"session": int(session) if session is not None else None,
                          "phase": phase, **dict(counts)}
                         for (session, phase), counts in phase_counts.items()],
        "markers": markers, "marker_issues": marker_issues,
        "target_output": target_output, "reports": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {"schema": "saccade.nvof.memcheck-classification/1",
              "scope": "Diagnosis classification only; no suppression or sanitizer PASS decision.",
              "logs": [parse(path) for path in args.logs]}
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(output)
        for log in result["logs"]:
            print(f"{log['path']}: reports={log['errors']} api={log['api_errors']} "
                  f"memory={log['memory_errors']} unknown={log['unknown_diagnoses']} "
                  f"families={len(log['stack_families'])} "
                  f"summary_matches={log['summary_matches_reports']}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()

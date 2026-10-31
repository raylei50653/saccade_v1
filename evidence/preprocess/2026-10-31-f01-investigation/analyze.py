#!/usr/bin/env python3
"""Offline diagnostic log analysis. Never runs GPU code or changes captures."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parent
CAP = ROOT / "capture"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def values(line):
    return dict(re.findall(r"([A-Za-z_][\w]*)=([^\s]+)", line))


def diagnostics(text):
    """Each uninitialized report starts with a read size, then a kernel/thread."""
    records = []
    pattern = re.compile(
        r"^========= Uninitialized __global__ memory read of size (\d+) bytes\n"
        r"=========\s+at (.+)\n"
        r"=========\s+by thread \((\d+),(\d+),(\d+)\) in block \((\d+),(\d+),(\d+)\)\n"
        r"=========\s+Address (0x[0-9a-fA-F]+)", re.M)
    for match in pattern.finditer(text):
        size, kernel, tx, ty, tz, bx, by, bz, address = match.groups()
        records.append({"read_bytes": int(size), "kernel": kernel,
                        "thread": [int(tx), int(ty), int(tz)],
                        "block": [int(bx), int(by), int(bz)],
                        "address": int(address, 16)})
    return records


def pixel_analysis(path):
    if not path.is_file() or not path.with_suffix(".exit").is_file():
        return {"status": "pending", "path": str(path)}
    log = path.read_text()
    records = diagnostics(log)
    rgba = [r for r in records if "nvjpeg::rgba2rgb" in r["kernel"]]
    for r in rgba:
        r["x"] = r["block"][0] * 32 + r["thread"][0]
        r["y"] = r["block"][1] * 8 + r["thread"][1]
    coords = Counter((r["x"], r["y"]) for r in rgba)
    valid = {(x, y) for y in range(27) for x in range(35)}
    actual = set(coords)
    by_coordinate = {(r["x"], r["y"]): r["address"] for r in rgba}
    horizontal = []
    vertical = []
    for (x, y), address in by_coordinate.items():
        if (x+1, y) in by_coordinate:
            horizontal.append(by_coordinate[x+1, y]-address)
        if (x, y+1) in by_coordinate:
            vertical.append(by_coordinate[x, y+1]-address)
    row_pitches = Counter(vertical)
    x_strides = Counter(horizontal)
    pitch = next(iter(row_pitches)) if len(row_pitches) == 1 else None
    bases = {r["address"]-4*r["x"]-pitch*r["y"] for r in rgba} if pitch is not None else set()
    summary_counts = [int(n) for n in re.findall(r"ERROR SUMMARY: (\d+) errors?\s*$", log, re.M)]
    inspect_reports = [r for r in records if "inspect_caller_output" in r["kernel"]]
    checks = {
        "summary_945_errors": summary_counts == [945],
        "945_rgba_records": len(rgba) == 945,
        "945_unique_pixels": len(coords) == 945,
        "every_valid_pixel_once": actual == valid and all(n == 1 for n in coords.values()),
        "all_reads_three_bytes": bool(rgba) and all(r["read_bytes"] == 3 for r in rgba),
        "all_records_rgba": len(records) == len(rgba),
        "no_inspect_kernel_reports": not inspect_reports,
        "horizontal_stride_four": set(x_strides) == {4},
        "uniform_row_pitch": pitch is not None,
        "uniform_base_address": len(bases) == 1,
        "exit_99": path.with_suffix(".exit").read_text().strip() == "99",
    }
    return {"status": "verified" if all(checks.values()) else "check-failed",
            "path": str(path.relative_to(ROOT)), "checks": checks,
            "record_count": len(records), "rgba_record_count": len(rgba),
            "pixel_occurrence_histogram": dict(Counter(coords.values())),
            "missing_pixels": sorted(valid-actual), "invalid_pixels": sorted(actual-valid),
            "horizontal_stride_bytes_histogram": dict(x_strides),
            "row_pitch_bytes_histogram": dict(row_pitches),
            "inferred_rgba_row_pitch_bytes": pitch,
            "inferred_rgba_base_address": hex(next(iter(bases))) if len(bases) == 1 else None,
            "read_size_histogram": dict(Counter(r["read_bytes"] for r in records)),
            "kernel_histogram": dict(Counter(r["kernel"] for r in records)),
            "summary_errors": summary_counts,
            "coordinates": [{"x": r["x"], "y": r["y"], "address": hex(r["address"]),
                             "read_bytes": r["read_bytes"]} for r in rgba]}


def command_analysis(command_path):
    name = command_path.stem
    exit_path = CAP / (name+".exit")
    if not exit_path.is_file():
        return {"name": name, "status": "pending"}
    log_path = CAP / (name+".log")
    log = log_path.read_text() if log_path.is_file() else ""
    command = command_path.read_text().strip()
    tokens = shlex.split(command)
    records = diagnostics(log)
    kernels = Counter(re.findall(r"^=========\s+at (.+)$", log, re.M))
    result_lines = [line for line in log.splitlines() if line.startswith("result=")]
    identity_lines = [line for line in log.splitlines() if line.startswith("nvjpeg=")]
    output = ROOT / (name+".bin")
    status = int(exit_path.read_text())
    summary_errors = [int(n) for n in re.findall(r"ERROR SUMMARY: (\d+) errors?\s*$", log, re.M)]
    api_failures = [line for line in log.splitlines()
                    if re.match(r"(?:api|cleanup)=", line)
                    and re.search(r"(?:nvjpeg|cuda)_status=[1-9]\d*", line)]
    tool = None
    if "--tool" in tokens:
        tool = tokens[tokens.index("--tool")+1]
    return {"name": name, "status": "complete", "exit": status,
            "command": command, "tool": tool,
            "summary_errors": summary_errors,
            "unprinted_errors": [int(n) for n in re.findall(r"ERROR SUMMARY: (\d+) errors were not printed", log)],
            "printed_uninitialized_records": len(records),
            "sdk_kernel_reports": dict((k, n) for k, n in kernels.items() if "nvjpeg::" in k),
            "caller_inspect_kernel_reports": sum(n for k, n in kernels.items() if "inspect_caller_output" in k),
            "all_reported_kernels": dict(kernels),
            "result_lines": result_lines,
            "results": [values(line) for line in result_lines],
            "identity": [values(line) for line in identity_lines],
            "api_failures": api_failures,
            "output_bytes": output.stat().st_size if output.is_file() else None,
            "output_sha256": sha256(output),
            "output_matches_planar_rgb_reference": output.read_bytes() == (ROOT/"baseline.rgb").read_bytes() if output.is_file() else None,
            "log_sha256": sha256(log_path)}


def main():
    commands = [command_analysis(path) for path in sorted(CAP.glob("*.command"))]
    pixels = pixel_analysis(CAP/"new-rgb-once-full-initcheck.log")
    output = {
        "schema": "f01-investigation-offline-summary/1",
        "scope": "Standalone diagnostic evidence; completion or exact RGB does not turn initcheck failure into PASS.",
        "source_sha256": sha256(ROOT/"repro.cu"), "binary_sha256": sha256(ROOT/"repro"),
        "baseline_rgb_sha256": sha256(ROOT/"baseline.rgb"),
        "pending_commands": [c["name"] for c in commands if c["status"] == "pending"],
        "commands": commands, "full_pixel_analysis": pixels,
        "output_hashes": {p.name: sha256(p) for p in sorted(ROOT.glob("*.bin"))},
    }
    (ROOT/"summary.json").write_text(json.dumps(output, indent=2)+"\n")
    concise_pixels = {k: v for k, v in pixels.items() if k != "coordinates"}
    print(json.dumps({"commands": len(commands), "pending": output["pending_commands"],
                      "full_pixel_analysis": concise_pixels}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Parse paired cuobjdump/ptxas reports without launching GPU work.

Usage: python summarize_static.py INPUT_DIRECTORY OUTPUT_CSV [HISTORICAL_CSV]

The ptxas smem field is user-declared static shared memory. The cuobjdump
SHARED field is kept separately: on this sm_120 build nonzero user shared
sections have an additional 1024 bytes. This is not added to ptxas smem or
compared against Nsight's static_shared_bytes_max as if it were user data.
Compiler spill byte counts describe compiled spill slots/instructions, not
dynamic memory traffic. Historical rows are matched by stage and unambiguous
short function name, because profiler demangling uses a different convention.
"""

import csv
import json
from pathlib import Path
import re
import subprocess
import sys


STAGES = {
    "selective_scan": {"F03_engine"},
    "scan_frame": set(),
    "resize": {"F02_preprocess", "F02_resize", "F02"},
    "postprocess": {"F04_postprocess"},
    "model_producer": {"F03_producer"},
    "detection_session": set(),
    "motion": {"F05_motion_image", "F06_GMC_including_cuFFT"},
    "tracking": {"F08a_predict", "F04b_batch", "F07_F08b_update"},
    "tracking_session": {"M_begin", "M_candidates", "M_control", "M_commit", "M_packet"},
}


def short_name(name):
    # All production entrypoints in these nine units are non-template kernels.
    prefix = name.split("(", 1)[0]
    prefix = name.replace("(anonymous namespace)", "<unnamed>").split("(", 1)[0]
    return prefix.rsplit("::", 1)[-1].split()[-1]


def main():
    root = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2])
    history = list(csv.DictReader(open(sys.argv[3], newline=""))) if len(sys.argv) > 3 else []
    manifest = json.loads((root / "commands.json").read_text())
    rows = []
    for unit in manifest["units"]:
        if unit["exit_code"] != 0:
            raise ValueError(f"Compilation failed: {unit['source']}")
        stem = Path(unit["source"]).stem
        resource_path = root / f"{stem}.resources.txt"
        compile_path = root / f"{stem}.compile.log"
        resource_text = resource_path.read_text()
        compile_text = compile_path.read_text()
        compiled = {}
        for match in re.finditer(
            r"Compiling entry function '([^']+)' for '(sm_\d+)'(.*?)(?=ptxas info\s*: Compiling entry function|\Z)",
            compile_text, re.S,
        ):
            name, architecture, body = match.groups()
            stack = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", body)
            regs = re.search(r"Used (\d+) registers", body)
            barriers = re.search(r"used (\d+) barriers", body)
            shared = re.search(r"(\d+) bytes smem", body)
            if not stack or not regs or not barriers:
                raise ValueError(f"Missing compiler fields for {name}")
            compiled[name] = {
                "architecture": architecture,
                "ptxas_registers_per_thread": int(regs[1]),
                "ptxas_user_static_shared_bytes": int(shared[1]) if shared else 0,
                "ptxas_stack_frame_bytes_per_thread": int(stack[1]),
                "ptxas_spill_store_bytes": int(stack[2]),
                "ptxas_spill_load_bytes": int(stack[3]),
                "ptxas_barriers": int(barriers[1]),
            }
        resources = re.findall(r" Function (\S+):\n\s*([^\n]+)", resource_text)
        if len(resources) != len(compiled):
            raise ValueError(f"Function counts differ in {stem}: {len(resources)} vs {len(compiled)}")
        names = [name for name, _ in resources]
        demangled = subprocess.check_output(["c++filt", *names], text=True).splitlines()
        if len(demangled) != len(names):
            raise ValueError(f"Demangler count mismatch in {stem}")
        for (name, text), human in zip(resources, demangled):
            fields = {key: int(value) for key, value in re.findall(r"([A-Z]+(?:\[\d+\])?):(\d+)", text)}
            compiler = compiled[name]
            if fields["REG"] != compiler["ptxas_registers_per_thread"]:
                raise ValueError(f"Register counts differ for {name}")
            row = {
                "source": "cuda/" + stem + ".cu",
                "source_sha256": unit["sha256"],
                "source_commit": manifest["source_commit"],
                "mangled_name": name,
                "demangled_name": human,
                "short_name": short_name(human),
                **compiler,
                "cubin_registers_per_thread": fields["REG"],
                "cubin_shared_bytes_including_reserved": fields["SHARED"],
                "cubin_shared_minus_ptxas_user_bytes": fields["SHARED"] - compiler["ptxas_user_static_shared_bytes"],
                "cubin_stack_bytes_per_thread": fields["STACK"],
                "cubin_local_bytes_per_thread": fields["LOCAL"],
                "cubin_constant0_bytes": fields.get("CONSTANT[0]", 0),
                "resource_log": str(resource_path),
                "compiler_log": str(compile_path),
            }
            old = [h for h in history if h["stage"] in STAGES[stem] and short_name(h["name"]) == row["short_name"]]
            if len({h["name"] for h in old}) > 1:
                raise ValueError(f"Ambiguous historical match for {stem}:{row['short_name']}")
            row["historical_stages"] = ";".join(sorted({h["stage"] for h in old}))
            row["historical_registers_per_thread_max"] = max((int(h["registers_per_thread_max"]) for h in old), default="")
            row["historical_user_static_shared_bytes_max"] = max((int(h["static_shared_bytes_max"]) for h in old), default="")
            row["historical_local_bytes_per_thread_max"] = max((int(h["local_bytes_per_thread_max"]) for h in old), default="")
            different = []
            if old:
                for historical, current in (
                    ("historical_registers_per_thread_max", "ptxas_registers_per_thread"),
                    ("historical_user_static_shared_bytes_max", "ptxas_user_static_shared_bytes"),
                    ("historical_local_bytes_per_thread_max", "cubin_local_bytes_per_thread"),
                ):
                    if row[historical] != row[current]:
                        different.append(f"{historical}={row[historical]} vs {current}={row[current]}")
            row["historical_comparison"] = "unmatched" if not old else "; ".join(different) if different else "equal_registers_user_shared_local"
            rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "functions": len(rows),
        "units": len(manifest["units"]),
        "historical_matched": sum(r["historical_comparison"] != "unmatched" for r in rows),
        "historical_discrepancies": [
            {"source": r["source"], "function": r["short_name"], "comparison": r["historical_comparison"]}
            for r in rows if r["historical_comparison"] not in {"unmatched", "equal_registers_user_shared_local"}
        ],
        "nonzero_stack_local_or_spills": [
            {k: r[k] for k in ("source", "short_name", "ptxas_stack_frame_bytes_per_thread", "cubin_local_bytes_per_thread", "ptxas_spill_store_bytes", "ptxas_spill_load_bytes")}
            for r in rows if any(r[k] for k in ("ptxas_stack_frame_bytes_per_thread", "cubin_local_bytes_per_thread", "ptxas_spill_store_bytes", "ptxas_spill_load_bytes"))
        ],
        "shared_overheads": sorted({r["cubin_shared_minus_ptxas_user_bytes"] for r in rows}),
    }, indent=2))


if __name__ == "__main__":
    main()

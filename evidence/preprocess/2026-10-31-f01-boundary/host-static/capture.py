#!/usr/bin/env python3
"""CPU-only F01 host-library inspection and offline analysis of an old trace.

Does not load CUDA libraries or execute a GPU program. Each run needs a fresh
output directory; all addresses are offsets into the hashed ELF, not live VAs.
"""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--library", type=Path,
                        default=Path("/opt/cuda/targets/x86_64-linux/lib/libnvjpeg.so.13.2.3.58"))
    parser.add_argument("--nvcuvid", type=Path, default=Path("/usr/lib/wsl/lib/libnvcuvid.so.1"))
    parser.add_argument("--trace", type=Path,
                        default=Path("evidence/preprocess/2026-10-31-f01-investigation/surface-poison/raw/nsys-rgb.sqlite"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    command_dir = args.out / "commands"
    command_dir.mkdir()
    commands = []

    def run(label, argv):
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result = subprocess.run(argv, capture_output=True, timeout=60)
        (command_dir / f"{label}.stdout").write_bytes(result.stdout)
        (command_dir / f"{label}.stderr").write_bytes(result.stderr)
        (command_dir / f"{label}.exit").write_text(f"{result.returncode}\n")
        record = {"label": label, "argv": argv, "cwd": str(Path.cwd()),
                  "start_utc": start, "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "exit": result.returncode}
        (command_dir / f"{label}.command.json").write_text(json.dumps(record, indent=2) + "\n")
        commands.append(record)
        if result.returncode:
            raise RuntimeError(f"{label} exited {result.returncode}; capture preserved")
        return result.stdout.decode("utf-8", errors="replace")

    identities = {str(p): {"sha256": digest(p), "bytes": p.stat().st_size}
                  for p in (args.library, args.nvcuvid, args.trace, Path(__file__))}
    run("git-head", ["git", "rev-parse", "HEAD"])
    run("git-status", ["git", "status", "--porcelain=v1"])
    for tool in ("readelf", "objdump", "nm", "strings"):
        run(f"version-{tool}", [tool, "--version"])
    dynamic = run("nvjpeg-dynamic", ["readelf", "-dW", str(args.library)])
    run("nvjpeg-sections", ["readelf", "-SW", str(args.library)])
    relocations = run("nvjpeg-relocations", ["readelf", "-rW", str(args.library)])
    run("nvjpeg-imports", ["nm", "-D", "--undefined-only", str(args.library)])
    run("nvjpeg-exports", ["nm", "-D", "--defined-only", str(args.library)])
    strings = run("nvjpeg-strings", ["strings", "-t", "x", str(args.library)])
    run("nvcuvid-dynamic", ["readelf", "-dW", str(args.nvcuvid)])
    run("nvcuvid-exports", ["nm", "-D", "--defined-only", str(args.nvcuvid)])

    def disasm(label, library, start, end):
        return run(label, ["objdump", "-d", "-M", "intel", "--no-show-raw-insn",
                           f"--start-address={start:#x}", f"--stop-address={end:#x}", str(library)])

    dispatch = disasm("decode-batched-dispatch", args.library, 0x73870, 0x7391a)
    loader = disasm("nvjpeg-private-loader", args.library, 0x6fa70, 0x6fd80)
    disasm("nvjpeg-cuda-proc-loader", args.library, 0x18de30, 0x18e390)
    disasm("nvjpeg-v2-decode-wrapper", args.library, 0xd0900, 0xd0917)
    disasm("nvjpeg-hw-decode-prefix", args.library, 0xd7d70, 0xd8070)
    disasm("nvjpeg-hwv2-decode-prefix", args.library, 0xd4bf0, 0xd4dc0)
    getter = disasm("nvcuvid-private-table-getter", args.nvcuvid, 0x4d220, 0x4d32e)
    disasm("nvcuvid-private-wrappers", args.nvcuvid, 0x4cb60, 0x4d220)

    # RTTI/RELATIVE-relocation reconstruction, only for the two batched classes.
    rels = {}
    for line in relocations.splitlines():
        m = re.match(r"([0-9a-f]+)\s+[0-9a-f]+\s+R_X86_64_RELATIVE\s+([0-9a-f]+)\s*$", line)
        if m:
            rels[int(m[1], 16)] = int(m[2], 16)
    rtti = {}
    for line in strings.splitlines():
        m = re.match(r"\s*([0-9a-f]+) (N6nvjpeg18decoder_hw_batched(?:19NVJPGBatchedDecoder|21NVJPGV2BatchedDecoder)E)$", line)
        if not m:
            continue
        name_offset = int(m[1], 16)
        candidates = []
        for typeinfo in [off - 8 for off, value in rels.items() if value == name_offset]:
            for ap in [off + 8 for off, value in rels.items() if value == typeinfo]:
                if rels.get(ap + 0x38) not in (0xd7d70, 0xd0900):
                    continue
                candidates.append({"typeinfo_offset": hex(typeinfo), "vtable_address_point": hex(ap),
                                   "methods": {hex(o): hex(rels.get(ap + o, 0)) for o in range(0, 0x58, 8)}})
        rtti[m[2]] = {"name_offset": hex(name_offset), "candidates": candidates}

    # __std_2U4S4U_X08 builds a private table by LEA target/MOV [rdi+slot],rax.
    private_table = {}
    target = None
    for line in getter.splitlines():
        m = re.search(r"lea\s+rax,.*# ([0-9a-f]+) ", line)
        if m:
            target = int(m[1], 16)
        m = re.search(r"mov\s+QWORD PTR \[rdi(?:\+(0x[0-9a-f]+))?\],rax", line)
        if m and target is not None:
            private_table[hex(int(m[1], 16) if m[1] else 0)] = hex(target)

    connection = sqlite3.connect(f"file:{args.trace.resolve()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    table_names = [r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    ids = {r["id"]: r["value"] for r in connection.execute("SELECT * FROM StringIds")}
    queries = {}

    def query(label, sql):
        queries[label] = sql
        return [dict(r) for r in connection.execute(sql)]

    runtime = query("runtime", "SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")
    for row in runtime:
        row["name"] = ids[row["nameId"]]
    kernels = query("kernels", "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    for row in kernels:
        row["name"] = ids[row["demangledName"]]
    events = query("events", "SELECT * FROM CUPTI_ACTIVITY_KIND_CUDA_EVENT ORDER BY correlationId")
    syncs = query("synchronization", "SELECT * FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION ORDER BY start")
    memcpy = query("memcpy", "SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start")
    metadata = query("capture_metadata", "SELECT * FROM META_DATA_CAPTURE")
    diagnostics = query("diagnostics", "SELECT * FROM DIAGNOSTIC_EVENT ORDER BY timestamp")
    streams = query("streams", "SELECT * FROM TARGET_INFO_CUDA_STREAM")
    api_by_corr = {r["correlationId"]: r for r in runtime}
    handoffs = []
    converters = [r for r in kernels if "nvjpeg::rgba2rgb<" in r["name"]]
    inspectors = [r for r in kernels if "inspect_caller_output" in r["name"]]
    for i, converter in enumerate(converters):
        next_converter_start = converters[i + 1]["start"] if i + 1 < len(converters) else 2**63
        records = [r for r in events if r["streamId"] == converter["streamId"]
                   and api_by_corr[r["correlationId"]]["start"] >= api_by_corr[converter["correlationId"]]["end"]
                   and api_by_corr[r["correlationId"]]["start"] < next_converter_start]
        if not records:
            continue
        record = min(records, key=lambda r: api_by_corr[r["correlationId"]]["start"])
        waits = [r for r in syncs if r["syncType"] == 2 and r["eventId"] == record["eventId"]
                 and r["eventSyncId"] == record["eventSyncId"] and r["contextId"] == record["contextId"]
                 and r["globalPid"] == record["globalPid"]]
        for wait in waits:
            candidates = [r for r in inspectors if r["streamId"] == wait["streamId"]
                          and r["start"] > converter["end"] and r["start"] < next_converter_start]
            handoffs.append({"converter": converter, "record": record,
                             "record_api": api_by_corr[record["correlationId"]], "wait": wait,
                             "wait_api": api_by_corr[wait["correlationId"]],
                             "inspectors_after_converter": candidates,
                             "claim": "record/wait match by process, context, eventId and eventSyncId; does not expose hardware producer"})
    trace_analysis = {"sqlite": str(args.trace), "sha256": digest(args.trace), "tables": table_names,
                      "runtime_api_counts": dict(collections.Counter(r["name"] for r in runtime)),
                      "all_runtime_callchain_ids_null": all(r["callchainId"] is None for r in runtime),
                      "driver_activity_table_present": "CUPTI_ACTIVITY_KIND_DRIVER" in table_names,
                      "cuda_callchains_table_present": "CUDA_CALLCHAINS" in table_names,
                      "runtime": runtime, "kernels": kernels, "events": events, "synchronization": syncs,
                      "memcpy": memcpy, "metadata": metadata, "diagnostics": diagnostics, "streams": streams,
                      "handoffs": handoffs, "queries": queries}
    (args.out / "trace-analysis.json").write_text(json.dumps(trace_analysis, indent=2) + "\n")
    needed = re.findall(r"Shared library: \[(.*?)\]", dynamic)
    analysis = {"kind": "CPU-only static inspection and offline historical trace analysis; no new GPU acceptance",
                "identities": identities, "dt_needed": needed,
                "nvjpeg_exports_only_nvjpeg_api": True,
                "decode_entry": "0x73870", "backend_dispatch_call": "0x73905: call QWORD PTR [r10+0x38]",
                "batched_rtti_vtables": rtti, "nvcuvid_private_function_table": private_table,
                "matched_converter_caller_handoffs": len(handoffs),
                "trace_skip_some_api_calls": next((r["value"] for r in metadata if r["name"] == "CUDA_SKIP_SOME_API_CALLS"), None),
                "checks": {"dispatch_indirect_call_found": "call   QWORD PTR [r10+0x38]" in dispatch,
                           "private_dlopen_found": "# 2a2e2b " in loader,
                           "two_batched_types_found": len(rtti) == 2,
                           "private_table_21_slots": len(private_table) == 21,
                           "two_converter_caller_handoffs": len(handoffs) == 2}}
    (args.out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (args.out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    if not all(analysis["checks"].values()):
        raise RuntimeError(f"expected local-binary controls changed: {analysis['checks']}")
    print(json.dumps({"checks": analysis["checks"], "private_table": private_table,
                      "handoffs": len(handoffs), "gpu_operations_executed": 0}, indent=2))


if __name__ == "__main__":
    main()

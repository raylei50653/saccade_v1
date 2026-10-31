#!/usr/bin/env python3
"""Offline check of the one narrow inner-provider call observation."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
capture = root / "gdb-inner-capture"
status = json.loads((capture / "status.json").read_text())
rows = status["observations"]
getter = next(r for r in rows if r["kind"] == "private_getter_entry")
call = next(r for r in rows if r["kind"] == "private_inner_call")
returned = next(r for r in rows if r["kind"] == "private_inner_return")
base = int(getter["pc"], 16)-0x4d220
reference = (root.parents[1] / "2026-10-31-f01-investigation" / "baseline.rgb").read_bytes()
rgb = (capture / "output.rgb").read_bytes()
disasm = (root / "inner-static" / "actual-inner-callee.stdout").read_text()
checks = {
    "debugger_and_target_completed": status["diagnostic_complete"] and status["target_exit_observed"],
    "one_inner_observation": sum(r["kind"] == "private_inner_call" for r in rows) == 1,
    "correct_callsite": int(call["pc"], 16)-base == 0x4e028,
    "callee_address_matches_module_offset": int(call["target"], 16)-base == 0x17ee0
        and call["target_module"]["offset"] == "0x17ee0",
    "callee_mapped_file_identity": call["target_module"]["path"] == "/usr/lib/wsl/lib/libnvcuvid.so.1"
        and call["target_module"]["mapped_file_sha256"] == "eef76fcf52bb7a594e6ef4dcc70f0e2579c183240037a67099646911acc273cf",
    "outer_inner_return_zero": int(returned["pc"], 16)-base == 0x4e02e and returned["eax"] == 0,
    "captured_branch_cl_zero": int(call["args"]["rcx"], 16) & 255 == 0,
    "static_branch_and_private_call_present": "17eec:" in disasm and "je     17f30" in disasm
        and "17f53:" in disasm and "call   QWORD PTR [rax+0x40]" in disasm,
    "full_rgb_exact": rgb == reference,
    "raw_log_complete": "result=complete repeat_exact=1 rgb_equivalence=exact" in (capture / "stdout.log").read_text(),
}
result = {"scope": "normal decode with one-shot CPU breakpoints; actual callee resolved, next private call not executed under an additional breakpoint",
          "checks": checks, "inner_call": call, "inner_return": returned,
          "callee_module_offset": "libnvcuvid.so.1+0x17ee0",
          "selected_branch": "CL=0 -> +0x17f30 -> indirect call at+0x17f53 through [innerObject+0x20] table slot+0x40",
          "unresolved": "actual target and semantic/synchronization guarantee of the table+0x40 function",
          "rgb_bytes": len(rgb), "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
          "not_claimed": ["R8=5 meaning", "private slot is named public CUDA wait API",
                          "all memory paths share this callee", "hardware-to-converter completion theorem",
                          "RTTI name: provider object's typeinfo was unavailable"]}
(root / "inner-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"checks": checks, "callee": result["callee_module_offset"], "next_boundary": result["unresolved"]}, indent=2))
if not all(checks.values()):
    raise SystemExit(1)

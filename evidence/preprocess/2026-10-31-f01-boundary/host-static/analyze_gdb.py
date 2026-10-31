#!/usr/bin/env python3
"""Recheck saved GDB observations against the captured static offsets, offline."""
import hashlib
import json
from pathlib import Path


root = Path(__file__).resolve().parent
capture = root / "gdb-capture-4"
status = json.loads((capture / "status.json").read_text())
static = json.loads((root / "capture" / "analysis.json").read_text())
observations = status["observations"]
entry = next(r for r in observations if r["kind"] == "decode_entry")
base = int(entry["entry"], 16) - 0x73870
dispatches = [r for r in observations if r["kind"] == "actual_backend_dispatch"]
getter_entry = next(r for r in observations if r["kind"] == "private_getter_entry")
getter_return = next(r for r in observations if r["kind"] == "private_getter_return")
provider_base = int(getter_entry["pc"], 16) - 0x4d220
slot = next(r for r in observations if r["kind"] == "private_slot_70_entry")
normal = [{"pc_offset": hex(int(r["pc"], 16)-base),
           "target_offset": hex(int(r["target"], 16)-base),
           "vptr_offset": hex(int(r["vptr"], 16)-base), "rtti_name": r["rtti_name"]} for r in dispatches]
private = {slot: hex(int(target, 16)-provider_base) for slot, target in getter_return["values"].items()}
modules = entry["modules"]
output = (capture / "output.rgb").read_bytes()
reference = (root.parents[1] / "2026-10-31-f01-investigation" / "baseline.rgb").read_bytes()
checks = {
    "debugger_and_inferior_completed": status["diagnostic_complete"] and status["target_exit_observed"],
    "two_dispatch_observations": len(normal) == 2,
    "dispatch_matches_v2_static_vtable": all(r["pc_offset"] == "0x73905" and r["target_offset"] == "0xd0900"
                                             and r["vptr_offset"] == "0x809c78"
                                             and r["rtti_name"] == "N6nvjpeg18decoder_hw_batched21NVJPGV2BatchedDecoderE" for r in normal),
    "private_getter_table_matches_static": all(static["nvcuvid_private_function_table"][slot] == target for slot, target in private.items()),
    "private_slot_70_matches_provider": int(slot["pc"], 16)-provider_base == 0x4ce10,
    "nvjpeg_actual_mapping_hash_matches_static": modules["/opt/cuda/targets/x86_64-linux/lib/libnvjpeg.so.13.2.3.58"]["sha256"]
        == static["identities"]["/opt/cuda/targets/x86_64-linux/lib/libnvjpeg.so.13.2.3.58"]["sha256"],
    "nvcuvid_actual_mapping_hash_matches_static": modules["/usr/lib/wsl/lib/libnvcuvid.so.1"]["sha256"]
        == static["identities"]["/usr/lib/wsl/lib/libnvcuvid.so.1"]["sha256"],
    "rgb_full_bytes_exact": output == reference,
    "raw_log_complete": "result=complete repeat_exact=1 rgb_equivalence=exact" in (capture / "stdout.log").read_text(),
    "no_debugger_python_exception": "Python Exception" not in (capture / "stderr.log").read_text(),
}
result = {"scope": "bounded normal decoder under CPU debugger; no sanitizer acceptance or hardware fence proof",
          "checks": checks, "dispatches": normal, "private_table": private,
          "nvjpeg_load_base": hex(base), "nvcuvid_load_base": hex(provider_base),
          "private_slot_70_offset": hex(int(slot["pc"], 16)-provider_base),
          "rgb_bytes": len(output), "rgb_sha256": hashlib.sha256(output).hexdigest(),
          "actual_modules": modules,
          "limitations": ["CPU breakpoints alter timing and execute only one fixed JPEG twice",
                          "private slot semantics and hardware-producer completion not established by names/offsets",
                          "module hashes are mapped file hashes captured at stop, not dumps of all live executable pages",
                          "private getter has no debug return type; filled table observed but no typed getter return code claimed"]}
(root / "gdb-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"checks": checks, "dispatches": normal, "rgb_bytes": len(output)}, indent=2))
if not all(checks.values()):
    raise SystemExit(1)

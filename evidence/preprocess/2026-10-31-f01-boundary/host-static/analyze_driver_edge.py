#!/usr/bin/env python3
"""Offline validation of the final single edge into the real WSL driver."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
capture = root / "gdb-driver-edge-capture"
status = json.loads((capture / "status.json").read_text())
rows = status["observations"]
getter = next(r for r in rows if r["kind"] == "private_getter_entry")
entry = next(r for r in rows if r["kind"] == "decode_entry")
call = next(r for r in rows if r["kind"] == "private_driver_edge_call")
returned = next(r for r in rows if r["kind"] == "private_driver_edge_return")
provider_base = int(getter["pc"], 16)-0x4d220
module = call["target_module"]
driver_path = "/usr/lib/wsl/drivers/nvamsi.inf_amd64_e6537d0cc841801c/libcuda.so.1.1"
driver_hash = "70d8cd3774f4b0cf9359af3ca894f8e2586687cdc4c1fbb63af786fb69d90354"
reference = (root.parents[1] / "2026-10-31-f01-investigation" / "baseline.rgb").read_bytes()
rgb = (capture / "output.rgb").read_bytes()
disasm = (root / "driver-edge-static" / "actual-driver-callee.stdout").read_text()
static_identity = json.loads((root / "driver-edge-static" / "identity.json").read_text())
checks = {
    "debugger_and_target_completed": status["diagnostic_complete"] and status["target_exit_observed"],
    "one_edge_observation": sum(r["kind"] == "private_driver_edge_call" for r in rows) == 1,
    "provider_callsite": int(call["pc"], 16)-provider_base == 0x17f53,
    "provider_return_site_and_zero": int(returned["pc"], 16)-provider_base == 0x17f56 and returned["eax"] == 0,
    "target_is_real_driver": module["path"] == driver_path and module["mapped_file_sha256"] == driver_hash
        and static_identity["path"] == driver_path and static_identity["sha256"] == driver_hash,
    "target_offset": module["offset"] == "0x4c7a50",
    "mapping_identity_matches_entry": entry["modules"][driver_path]["sha256"] == module["mapped_file_sha256"],
    "no_exported_name_claimed": call["symbol"].startswith("No symbol matches"),
    "driver_static_callee_captured": "4c7a50:" in disasm and "4c7ae6:" in disasm and "call   4c76e0" in disasm,
    "full_rgb_exact": rgb == reference,
    "raw_log_complete": "result=complete repeat_exact=1 rgb_equivalence=exact" in (capture / "stdout.log").read_text(),
    "no_debugger_exception": "Python Exception" not in (capture / "stderr.log").read_text(),
}
result = {"scope": "final bounded normal CPU-debugger edge; no deeper GPU probing or synchronization theorem",
          "checks": checks, "edge_call": call, "edge_return": returned,
          "actual_boundary": "real WSL driver libcuda.so.1.1+0x4c7a50; no public symbol",
          "static_summary": "calls internal resolver helper+0x1303fa0 then checks state; success branch passes a 32-bit single-element list to helper+0x4c76e0",
          "unresolved": ["semantics and hardware completion/visibility guarantee of hidden driver operation",
                         "which state branches execute inside the driver", "initcheck shadow-state transition for private writes"],
          "rgb_bytes": len(rgb), "rgb_sha256": hashlib.sha256(rgb).hexdigest()}
(root / "driver-edge-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"checks": checks, "boundary": result["actual_boundary"]}, indent=2))
if not all(checks.values()):
    raise SystemExit(1)

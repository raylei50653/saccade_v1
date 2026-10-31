#!/usr/bin/env python3
"""Offline cross-check of the captured disassemblies and reconstructed contract.

This executes no SDK binary and no GPU work. It verifies the saved outputs of two
NVIDIA disassemblers, the complete memory-address dataflow, and the host ABI.
It is not an emulator or a formal proof of NVIDIA instruction semantics.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re


def instructions(text):
    # nvdisasm names the unreachable tail self-loop; cuobjdump prints its
    # numeric target. Resolve the captured label to its next instruction offset.
    labels = {m[1]: int(m[2], 16) for m in re.finditer(
        r"(?m)^(\.L_[A-Za-z0-9_]+):\s*\n\s*/\*([0-9a-f]{4})\*/", text)}
    result = {}
    for match in re.finditer(r"/\*([0-9a-f]{4})\*/\s+(.*?);", text):
        instruction = re.sub(r"\s+", " ", match[2]).strip()
        for label, offset in labels.items():
            instruction = instruction.replace("`(" + label + ")", hex(offset))
        result[int(match[1], 16)] = instruction
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    capture = args.capture
    commands = capture / "commands"
    for status_file in commands.glob("*.status.json"):
        status = json.loads(status_file.read_text())
        if status["timeout"] or status["exit_code"] != 0:
            raise SystemExit(f"failed captured command: {status_file}")
        output = capture / status["stored_stdout"]
        content = gzip.decompress(output.read_bytes()) if output.suffix == ".gz" else output.read_bytes()
        if hashlib.sha256(content).hexdigest() != status["stdout_sha256"]:
            raise SystemExit(f"raw stdout hash mismatch: {output}")
        stderr = commands / (status["label"] + ".stderr.txt")
        if hashlib.sha256(stderr.read_bytes()).hexdigest() != status["stderr_sha256"]:
            raise SystemExit(f"stderr hash mismatch: {stderr}")
    sass = instructions((commands / "rgba2rgb-sass.stdout.txt").read_text())
    nvdisasm = instructions((commands / "rgba2rgb-nvdisasm.stdout.txt").read_text())
    if sass != nvdisasm:
        raise SystemExit("disassemblers disagree")
    # Exact operands of all live instructions involved in coordinate, address,
    # load, byte extraction or output store reconstruction, plus both exits.
    expected = {
        0x10: "S2R R11, SR_TID.Y", 0x20: "LDCU.64 UR4, c[0x0][0x3c0]",
        0x30: "S2R R0, SR_CTAID.Y", 0x40: "S2R R6, SR_TID.X",
        0x50: "S2R R7, SR_CTAID.X", 0x60: "LEA R11, R0, R11, 0x3",
        0x70: "ISETP.GE.AND P0, PT, R11, UR5, PT", 0x80: "LEA R6, R7, R6, 0x5",
        0x90: "ISETP.GE.OR P0, PT, R6, UR4, P0", 0xa0: "@P0 EXIT",
        0xb0: "LDCU.64 UR6, c[0x0][0x3a0]", 0xc0: "IMAD.WIDE R2, R6, 0x4, RZ",
        0xd0: "LDCU.64 UR8, c[0x0][0x380]", 0xe0: "LDCU.64 UR4, c[0x0][0x358]",
        0xf0: "LDCU.128 UR12, c[0x0][0x3f0]", 0x100: "SHF.R.S32.HI R7, RZ, 0x1f, R6",
        0x110: "LDCU.128 UR16, c[0x0][0x3d0]", 0x120: "IMAD.WIDE.U32 R2, R11, UR6, R2",
        0x130: "IMAD R3, R11, UR7, R3", 0x140: "LDCU.64 UR6, c[0x0][0x3e8]",
        0x150: "IADD.64 R2, R2, UR8", 0x160: "LDCU.64 UR8, c[0x0][0x3c8]",
        0x170: "LDG.E R13, desc[UR4][R2.64]", 0x180: "IMAD.WIDE.U32 R8, R11, UR12, R6",
        0x190: "IMAD R9, R11.reuse, UR13, R9", 0x1a0: "IMAD.WIDE.U32 R4, R11, UR6, R6",
        0x1b0: "IMAD.WIDE.U32 R6, R11, UR14, R6", 0x1c0: "IMAD R5, R11.reuse, UR7, R5",
        0x1d0: "IMAD R7, R11, UR15, R7", 0x1e0: "IADD.64 R2, R4, UR8",
        0x1f0: "IADD.64 R4, R8, UR16", 0x200: "IADD.64 R6, R6, UR18",
        0x210: "PRMT R9, R13.reuse, 0x7771, RZ", 0x220: "STG.E.U8 desc[UR4][R2.64], R13",
        0x230: "PRMT R11, R13, 0x7772, RZ", 0x240: "STG.E.U8 desc[UR4][R4.64], R9",
        0x250: "STG.E.U8 desc[UR4][R6.64], R11", 0x260: "EXIT",
    }
    for offset, text in expected.items():
        if sass.get(offset) != text:
            raise SystemExit(f"unexpected dataflow instruction at {offset:#x}")
    memory = [(offset, text) for offset, text in sass.items() if re.search(r"\b(?:LDG|STG)\b", text)]
    if [offset for offset, _ in memory] != [0x170, 0x220, 0x240, 0x250]:
        raise SystemExit("unexpected global memory operations")
    analysis = json.loads((capture / "analysis.json").read_text())
    if analysis["host_abi"] != json.loads((commands / "host-abi.stdout.txt").read_text()):
        raise SystemExit("ABI reconstruction differs from captured host observation")
    if len(sass) != 48:
        raise SystemExit("unexpected total instruction count")
    result = {"offline_crosscheck": "PASS", "gpu_commands_executed": 0,
              "disassemblers_agree_on_instructions": len(sass),
              "exact_dataflow_instructions_checked": len(expected),
              "global_memory_operations": [{"offset": hex(o), "instruction": t} for o, t in memory],
              "limitation": "saved-output consistency and reconstruction checks, not hardware execution, sanitizer shadow proof or F01 acceptance"}
    if args.out:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

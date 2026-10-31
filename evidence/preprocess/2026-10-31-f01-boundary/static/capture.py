#!/usr/bin/env python3
"""Capture one pinned nvJPEG kernel's static evidence, without GPU execution.

Every subprocess retains argv, cwd, stdout, stderr, actual exit and timestamps.
Large ELF text is gzip-compressed losslessly. The extracted proprietary cubin and
host-only ABI executable remain local in work/; their hashes are recorded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FUNCTION = "_ZN6nvjpeg8rgba2rgbIL20nvjpegOutputFormat_t3ENS_19RGBAToRGBConversion12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS5_"
HISTORICAL_HASH = "c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument("--library", type=Path, default=Path("/opt/cuda/targets/x86_64-linux/lib/libnvjpeg.so.13.2.3.58"))
    parser.add_argument("--cuda", type=Path, default=Path("/opt/cuda"))
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    commands = out / "commands"
    commands.mkdir()
    work = out / "work"
    work.mkdir()
    library = args.library.resolve(strict=True)
    cuobjdump = args.cuda / "bin/cuobjdump"
    nvdisasm = args.cuda / "bin/nvdisasm"
    history = REPO / "evidence/preprocess/2026-10-31-f01-investigation"
    policy = REPO / "docs/PREPROCESS_POLICY.md"
    original_report = REPO / "evidence/preprocess/2026-10-31/REPORT.md"
    immutable = [library, HERE / "capture.py", HERE / "abi.cpp",
                 args.cuda / "include/nvjpeg.h", args.cuda / "include/nppdefs.h",
                 history / "REPORT.md", history / "capture/identity.log",
                 history / "capture/new-rgb-once-full-initcheck.log", policy, original_report]
    before = {str(p): digest(p) for p in immutable}
    statuses = []

    def run(label: str, argv: list[str | Path], *, cwd: Path = work,
            compress: bool = False) -> str:
        argv = [str(a) for a in argv]
        start = dt.datetime.now(dt.timezone.utc).isoformat()
        save_json(commands / f"{label}.command.json", {"argv": argv, "cwd": str(cwd), "timeout_seconds": 60})
        timed_out = False
        try:
            p = subprocess.run(argv, cwd=cwd, capture_output=True, timeout=60)
            stdout, stderr, code = p.stdout, p.stderr, p.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = exc.stdout or b"", exc.stderr or b"", None
            timed_out = True
        stdout_path = commands / (f"{label}.stdout.txt.gz" if compress else f"{label}.stdout.txt")
        if compress:
            stdout_path.write_bytes(gzip.compress(stdout, mtime=0))
        else:
            stdout_path.write_bytes(stdout)
        (commands / f"{label}.stderr.txt").write_bytes(stderr)
        (commands / f"{label}.exit").write_text("TIMEOUT\n" if timed_out else f"{code}\n")
        status = {"label": label, "started_utc": start,
                  "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "exit_code": code, "timeout": timed_out,
                  "stdout_bytes": len(stdout), "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
                  "stored_stdout": str(stdout_path.relative_to(out)),
                  "stderr_sha256": hashlib.sha256(stderr).hexdigest()}
        save_json(commands / f"{label}.status.json", status)
        statuses.append(status)
        save_json(out / "command-status.json", statuses)
        if timed_out or code != 0:
            raise RuntimeError(f"{label} failed: code={code}, timeout={timed_out}")
        return stdout.decode("utf-8", errors="strict")

    run("git-head", ["git", "rev-parse", "HEAD"], cwd=REPO)
    run("git-status", ["git", "status", "--porcelain=v1"], cwd=REPO)
    run("cuobjdump-version", [cuobjdump, "--version"])
    run("nvdisasm-version", [nvdisasm, "--version"])
    run("readelf-version", ["readelf", "--version"])
    run("host-compiler-version", ["g++", "--version"])
    listing = run("list-cubins", [cuobjdump, "--list-elf", library])
    elf = run("sm120-elf", [cuobjdump, "--dump-elf", "--gpu-architecture", "sm_120", library], compress=True)
    blocks = elf.split("Fatbin elf code:")[1:]
    candidates = [(i, b) for i, b in enumerate(blocks) if "\n.nv.info." + FUNCTION + "\n" in b]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one kernel metadata block, found {len(candidates)}")
    index, block = candidates[0]
    names = [l.split(": ", 1)[1].strip() for l in listing.splitlines() if ".sm_120.cubin" in l]
    cubin_name = names[index]
    run("extract-cubin", [cuobjdump, "--extract-elf", cubin_name, library])
    cubin = work / cubin_name
    if not cubin.is_file():
        raise RuntimeError("cuobjdump did not create expected cubin")
    cubin_sha = digest(cubin)
    sass_raw = run("rgba2rgb-sass", [cuobjdump, "--dump-sass", "--function", FUNCTION, cubin])
    sections = run("cubin-sections", ["readelf", "--wide", "--sections", cubin])
    symbols = run("cubin-symbols", ["readelf", "--wide", "--symbols", cubin])
    sym = [l for l in symbols.splitlines() if l.split() and l.split()[-1] == FUNCTION]
    if len(sym) != 1:
        raise RuntimeError("entry symbol is ambiguous")
    symbol_index = int(sym[0].split(":", 1)[0])
    run("rgba2rgb-nvdisasm", [nvdisasm, "--print-code", "--print-instruction-encoding", "--cuda-function-index", str(symbol_index), cubin])
    run("kernel-info-hex", ["readelf", "--hex-dump=.nv.info." + FUNCTION, cubin])
    run("host-abi-compile", ["g++", "-std=c++17", "-I" + str(args.cuda / "include"), HERE / "abi.cpp", "-o", work / "abi"])
    abi = json.loads(run("host-abi", [work / "abi"]))
    if abi != {"pointer_bytes": 8, "size_t_bytes": 8,
               "nvjpegImage_t": {"size": 64, "align": 8, "channel": 0, "pitch": 32, "components": 4},
               "NppiSize": {"size": 8, "align": 4, "width": 0, "height": 4}, "NVJPEG_OUTPUT_RGB": 3}:
        raise RuntimeError("host ABI differs from reconstruction")
    marker = "\n.nv.info." + FUNCTION + "\n"
    info_tail = block.split(marker, 1)[1]
    next_section = re.search(r"(?m)^\.[A-Za-z]", info_tail)
    info = marker.strip() + "\n" + (info_tail[:next_section.start()] if next_section else info_tail)
    (out / "kernel-info.txt").write_text(info)
    instructions = {}
    for match in re.finditer(r"/\*([0-9a-f]{4})\*/\s+(.*?);", sass_raw):
        instructions[int(match[1], 16)] = re.sub(r"\s+", " ", match[2]).strip()
    (out / "rgba2rgb.sass.txt").write_text("\n".join(f"{off:04x}: {text}" for off, text in instructions.items()) + "\n")
    expected = {
        0x10: "S2R R11, SR_TID.Y", 0x30: "S2R R0, SR_CTAID.Y",
        0x40: "S2R R6, SR_TID.X", 0x50: "S2R R7, SR_CTAID.X",
        0x60: "LEA R11, R0, R11, 0x3", 0x70: "ISETP.GE.AND P0, PT, R11, UR5, PT",
        0x80: "LEA R6, R7, R6, 0x5", 0x90: "ISETP.GE.OR P0, PT, R6, UR4, P0",
        0xa0: "@P0 EXIT", 0xb0: "LDCU.64 UR6, c[0x0][0x3a0]",
        0xc0: "IMAD.WIDE R2, R6, 0x4, RZ", 0xd0: "LDCU.64 UR8, c[0x0][0x380]",
        0x120: "IMAD.WIDE.U32 R2, R11, UR6, R2", 0x150: "IADD.64 R2, R2, UR8",
        0x170: "LDG.E R13, desc[UR4][R2.64]", 0x210: "PRMT R9, R13.reuse, 0x7771, RZ",
        0x220: "STG.E.U8 desc[UR4][R2.64], R13", 0x230: "PRMT R11, R13, 0x7772, RZ",
        0x240: "STG.E.U8 desc[UR4][R4.64], R9", 0x250: "STG.E.U8 desc[UR4][R6.64], R11",
        0x260: "EXIT",
    }
    for offset, text in expected.items():
        if instructions.get(offset) != text:
            raise RuntimeError(f"instruction mismatch at {offset:#x}: {instructions.get(offset)!r}")
    params = [{"ordinal": int(m[1], 16), "offset": int(m[2], 16), "size": int(m[3], 16)}
              for m in re.finditer(r"Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x([0-9a-f]+)\s+Size\s*:\s*0x([0-9a-f]+)", info)]
    params.sort(key=lambda p: p["ordinal"])
    if params != [{"ordinal": 0, "offset": 0, "size": 64}, {"ordinal": 1, "offset": 64, "size": 8}, {"ordinal": 2, "offset": 72, "size": 64}]:
        raise RuntimeError("unexpected kernel parameter metadata")
    unused = re.search(r"EIATTR_UNUSED_LOAD_BYTE_OFFSET.*?Value:\s*(0x[0-9a-f]+)\s+(0x[0-9a-f]+)", info, re.S)
    if not unused or (int(unused[1], 16), int(unused[2], 16)) != (0x170, 0x7):
        raise RuntimeError("unused-load metadata changed")
    if "0x12b 0x880380" not in info or "EIATTR_CBANK_PARAM_SIZE" not in info:
        raise RuntimeError("unexpected constant-bank metadata")
    # These are checks of the reconstructed byte permutation, not an emulator of
    # the GPU or sanitizer. The algebra fixes each output byte for every word.
    def prmt(a: int, b: int, selector: int) -> int:
        source = [(a >> (8 * i)) & 255 for i in range(4)] + [(b >> (8 * i)) & 255 for i in range(4)]
        result = 0
        for i in range(4):
            sel = (selector >> (4 * i)) & 15
            value = source[sel & 7]
            if sel & 8:
                value = 255 if value & 128 else 0
            result |= value << (8 * i)
        return result
    words = [0, 0xffffffff, 0x12345678, 0xa55a09f0] + [1 << i for i in range(32)]
    for word in words:
        if (prmt(word, 0, 0x7771) & 255) != ((word >> 8) & 255):
            raise RuntimeError("first byte permutation reconstruction failed")
        if (prmt(word, 0, 0x7772) & 255) != ((word >> 16) & 255):
            raise RuntimeError("second byte permutation reconstruction failed")
    # Finite historical-grid reconstruction corroborates the SASS algebra.
    points = [(32 * bx + tx, 8 * by + ty) for bx in range(2) for by in range(4)
              for tx in range(32) for ty in range(8)
              if 32 * bx + tx < 35 and 8 * by + ty < 27]
    addresses = [512 * y + 4 * x for x, y in points]
    if len(set(points)) != 945 or len(set(addresses)) != 945 or max(addresses) != 13448:
        raise RuntimeError("historical pixel-grid reconstruction failed")
    history_text = (history / "capture/new-rgb-once-full-initcheck.log").read_text()
    read_reports = re.findall(r"Uninitialized __global__ memory read of size (\d+) bytes", history_text)
    if len(read_reports) != 945 or set(read_reports) != {"3"}:
        raise RuntimeError("historical diagnostic identity/count changed")
    after = {str(p): digest(p) for p in immutable}
    if before != after:
        raise RuntimeError("input drift during capture")
    identity = {"started_inputs": before, "finished_inputs": after,
                "inputs_unchanged": True, "library_sha256": digest(library),
                "library_matches_historical_report": digest(library) == HISTORICAL_HASH,
                "cubin_name": cubin_name, "cubin_sha256": cubin_sha,
                "host_abi_executable_sha256": digest(work / "abi"),
                "tools": {str(p): {"sha256": digest(p.resolve()), "resolved": str(p.resolve())}
                          for p in [cuobjdump, nvdisasm, Path(shutil.which("readelf")), Path(shutil.which("g++"))]},
                "python": sys.version, "gpu_commands_executed": 0,
                "proprietary_cubin_local_only": str(cubin)}
    save_json(out / "identity.json", identity)
    analysis = {
        "classification": "STATIC_RECONSTRUCTION_CONFIRMED_FOR_PINNED_BINARY_NOT_F01_ACCEPTANCE",
        "function": FUNCTION, "architecture": "sm_120", "function_symbol_index": symbol_index,
        "checked_instruction_operands": len(expected), "host_abi": abi,
        "parameters": params, "parameter_bytes": 136, "constant_bank_base": "0x380",
        "parameter_map": {"source_channel_0": "c0[0x380]", "source_pitch_0": "c0[0x3a0]",
                          "width": "c0[0x3c0]", "height": "c0[0x3c4]",
                          "destination_channel_0_1_2": ["c0[0x3c8]", "c0[0x3d0]", "c0[0x3d8]"],
                          "destination_pitch_0_1_2": ["c0[0x3e8]", "c0[0x3f0]", "c0[0x3f8]"]},
        "reconstruction": {"x": "32*blockIdx.x+threadIdx.x", "y": "8*blockIdx.y+threadIdx.y",
                           "early_exit": "x>=width || y>=height",
                           "source_address": "src.channel[0] + y*src.pitch[0] + 4*x",
                           "physical_load": "one 32-bit LDG.E at 0x170 (includes fourth byte)",
                           "output": "dst.channel[c][y*dst.pitch[c]+x] = byte_c(loaded_word), c=0,1,2",
                           "high_byte": "loaded but has no path to any output store",
                           "domain": "nonnegative launch coordinates and positive dimensions fitting signed int; valid allocated 64-bit address arithmetic and pitch; no coordinate/address overflow"},
        "unused_load_metadata": {"raw_values": ["0x170", "0x7"],
                                 "observed": True,
                                 "interpretation": "0x7 agrees with low-three-byte use; no public encoding specification or observation proving initcheck consumes this record"},
        "byte_permutation_model_examples_checked": len(words),
        "historical_grid": {"width": 35, "height": 27, "pitch": 512, "active_pixels": 945,
                            "distinct_load_addresses": 945, "last_load_start_offset": 13448,
                            "last_physical_load_end_exclusive": 13452,
                            "reported_read_bytes": 3, "reported_reads": len(read_reports)},
        "not_proved": ["executed instrumented instruction stream is byte-identical to on-disk cubin",
                       "which internal agent wrote the private surface",
                       "hardware completion happens-before converter for every schedule",
                       "initcheck shadow-state implementation or reason for missing initialization",
                       "runtime numeric correctness beyond already captured cases",
                       "F01 sanitizer PASS or vendor false-positive classification"],
    }
    save_json(out / "analysis.json", analysis)
    print(json.dumps({"out": str(out), "classification": analysis["classification"],
                      "library_matches_historical_report": identity["library_matches_historical_report"],
                      "instructions_checked": len(expected), "kernel_parameter_bytes": 136}, indent=2))


if __name__ == "__main__":
    main()

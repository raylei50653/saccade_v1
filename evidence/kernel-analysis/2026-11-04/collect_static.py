#!/usr/bin/env python3
"""Compile current production CUDA units with existing flags, without editing a build.

Usage: python collect_static.py COMPILE_COMMANDS NEW_OUTPUT_DIRECTORY
Compilation is sequential and does not launch GPU kernels. Binaries stay in build/.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

commands = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=False)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
selected = []
for entry in json.loads(commands.read_text()):
    src = Path(entry["file"])
    if src.parent.name != "cuda" or src.suffix != ".cu" or src.name == "synthetic_runtime.cu":
        continue
    args = shlex.split(entry["command"])
    i = args.index("-o")
    obj = out / (src.stem + ".o")
    args[i + 1] = str(obj)
    # Build identity is refreshed; optimization, arch and arithmetic flags are retained.
    args = [a if not a.startswith("-DSACCADE_BUILD_COMMIT=") else '-DSACCADE_BUILD_COMMIT="' + commit + '"' for a in args]
    args.append("--resource-usage")
    result = subprocess.run(args, cwd=entry["directory"], text=True, capture_output=True)
    (out / (src.stem + ".compile.log")).write_text(result.stdout + result.stderr)
    record = dict(source=str(src), sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                  command=args, cwd=entry["directory"], exit_code=result.returncode)
    selected.append(record)
    if result.returncode:
        print(result.stderr)
        raise SystemExit(result.returncode)
    usage = subprocess.run(["/opt/cuda/bin/cuobjdump", "--dump-resource-usage", str(obj)], text=True, capture_output=True, check=True)
    (out / (src.stem + ".resources.txt")).write_text(usage.stdout)
    print(src.name, "OK", flush=True)
(out / "commands.json").write_text(json.dumps(dict(source_commit=commit, compile_commands=str(commands), units=selected), indent=2) + "\n")

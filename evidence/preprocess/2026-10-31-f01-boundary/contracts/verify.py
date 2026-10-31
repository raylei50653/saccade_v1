#!/usr/bin/env python3
"""Verify pinned local SDK bytes and syntax-check the helper; no GPU execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main() -> int:
    identity = json.loads((HERE / "local-sdk-identity.json").read_text())
    failures = []
    for item in identity["files"]:
        path = Path(item["path"])
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            failures.append(f"SDK drift {path}")
    command = ["c++", "-std=c++17", "-fsyntax-only", "-I/opt/cuda/include",
               str(HERE / "compile_probe.cpp")]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    print(json.dumps({"sdk_files": len(identity["files"]),
                      "identity_failures": failures,
                      "syntax_only_argv": command,
                      "syntax_only_exit": result.returncode,
                      "stderr": result.stderr,
                      "gpu_executed": False}, indent=2))
    return int(bool(failures) or result.returncode != 0)


if __name__ == "__main__":
    sys.exit(main())

"""Capture remaining five unique MOT17 sequences for each frozen detector."""
from pathlib import Path
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
PYTHON = "/home/ray/.cache/saccade-v1/t01/venv/bin/python"
for seq in ("MOT17-04-SDP", "MOT17-05-SDP", "MOT17-09-SDP", "MOT17-11-SDP", "MOT17-13-SDP"):
    for arm in ("official", "finetuned"):
        name = arm + seq[6:8]
        cmd = ["flock", "/home/ray/.cache/saccade-v1/gpu.lock", PYTHON, str(HERE / "export.py"),
               "--arm", arm, "--sequence", seq, "--batch-size", "8"]
        started = time.monotonic()
        print(json.dumps({"status": "STARTING", "arm": arm, "sequence": seq, "command": cmd}), flush=True)
        with (HERE / (name + ".log")).open("x") as stream:
            result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
        receipt = {"command": cmd, "exit_code": result.returncode, "wall_seconds": time.monotonic() - started}
        (HERE / (name + ".command.json")).write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps({"arm": arm, "sequence": seq, **receipt}), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)

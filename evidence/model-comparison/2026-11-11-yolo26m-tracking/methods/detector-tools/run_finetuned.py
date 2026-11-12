from pathlib import Path
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
PYTHON = "/home/ray/.cache/saccade-v1/t01/venv/bin/python"
for seq in ("MOT17-10-SDP", "MOT17-02-SDP"):
    name = "finetuned" + seq[6:8]
    cmd = ["flock", "/home/ray/.cache/saccade-v1/gpu.lock", PYTHON, str(HERE / "export.py"),
           "--arm", "finetuned", "--sequence", seq]
    started = time.monotonic()
    with (HERE / (name + ".log")).open("x") as stream:
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    receipt = {"command": cmd, "exit_code": result.returncode, "wall_seconds": time.monotonic() - started}
    (HERE / (name + ".command.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

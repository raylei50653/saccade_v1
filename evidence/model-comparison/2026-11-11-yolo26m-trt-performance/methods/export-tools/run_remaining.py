from pathlib import Path
import json
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCK = "/home/ray/.cache/saccade-v1/gpu.lock"
PYTHON = "/home/ray/.cache/saccade-v1/t01/venv-export/bin/python"


def run(name, command):
    start = time.monotonic()
    print(json.dumps({"event": "START", "name": name, "command": command}), flush=True)
    with (HERE / (name + ".log")).open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    row = {"command": command, "exit_code": result.returncode, "wall_seconds": time.monotonic() - start}
    (HERE / (name + ".command.json")).write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps({"name": name, **row}), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def build(arm):
    a = ROOT / "artifacts" / arm
    run(arm + "-build", ["flock", LOCK, str(HERE / "build_engine"), str(a / "model.onnx"),
                         str(a / "model.engine"), str(a / "build.json"), str(a / "layers.json")])


deadline = time.monotonic() + 900
pending = ROOT / "artifacts/finetuned/onnx-verify.json"
while not pending.is_file():
    if time.monotonic() > deadline:
        raise SystemExit("finetuned export timed out waiting for verification receipt")
    time.sleep(2)
assert json.loads(pending.read_text())["status"] == "PASS"
build("finetuned")
run("official-default-export", ["flock", LOCK, PYTHON, str(HERE / "export_models.py"), "--arm", "official-default"])
build("official-default")

#!/usr/bin/env python3
"""Bounded normal nvJPEG diagnostic under CPU GDB; never an acceptance run."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("observe_driver_edge.gdb")
    (out / "observe_driver_edge.gdb").write_bytes(source.read_bytes())
    binary = Path("build-f01-investigation/repro")
    expected = Path("evidence/preprocess/2026-10-31-f01-investigation/baseline.rgb")
    inputs = [binary, source, Path(__file__), Path("fixtures/preprocess/baseline.jpg"), expected,
              Path("/opt/cuda/targets/x86_64-linux/lib/libnvjpeg.so.13.2.3.58"),
              Path("/usr/lib/wsl/lib/libnvcuvid.so.1"), Path("/usr/lib/wsl/lib/libcuda.so.1")]
    identity = {str(p): {"sha256": digest(p), "bytes": p.stat().st_size} for p in inputs}
    assert digest(binary) == "58e2c0ea589fd64ef9e334793c2af8230f4e580d6a303888de769313b5da27c3"
    mem = Path("/proc/meminfo").read_text()
    (out / "memory-before.txt").write_text(mem)
    assert int(next(x.split()[1] for x in mem.splitlines() if x.startswith("MemAvailable:"))) >= 4*1024*1024
    argv = ["gdb", "-q", "-nx", "-batch", "-x", str(source), "--args", str(binary),
            "hw", "rgb", "default", "preallocate", "uninit", "2",
            "fixtures/preprocess/baseline.jpg", str(expected), str(out / "output.rgb")]
    record = {"argv": argv, "cwd": str(Path.cwd()), "timeout_seconds": 60,
              "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "env": {k: os.environ[k] for k in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING") if k in os.environ},
              "identity": identity, "scope": "diagnostic CPU debugger changes host timing; no sanitizer or CUDA debugger"}
    (out / "command.json").write_text(json.dumps(record, indent=2) + "\n")
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = p.communicate(timeout=60)
        record.update(debugger_exit=p.returncode, timeout=False)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        stdout, stderr = p.communicate()
        record.update(debugger_exit=p.returncode, timeout=True)
    (out / "stdout.log").write_bytes(stdout)
    (out / "stderr.log").write_bytes(stderr)
    (out / "exit").write_text(str(record["debugger_exit"]) + "\n")
    records = [json.loads(line[len("F01_GDB "):]) for line in stdout.decode(errors="replace").splitlines() if line.startswith("F01_GDB ")]
    record["observations"] = records
    record["target_exit_observed"] = any(r.get("kind") == "inferior_exited" and r.get("exit_code") == 0 for r in records)
    record["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if (out / "output.rgb").exists():
        rgb = (out / "output.rgb").read_bytes()
        record["output"] = {"sha256": digest(out / "output.rgb"), "bytes": len(rgb), "rgb_exact": rgb == expected.read_bytes()}
    record["identity_after"] = {str(p): digest(p) for p in inputs}
    record["diagnostic_complete"] = (not record["timeout"] and record["debugger_exit"] == 0
                                     and record["target_exit_observed"] and record.get("output", {}).get("rgb_exact", False)
                                     and b"Python Exception" not in stderr
                                     and all(record["identity"][str(p)]["sha256"] == record["identity_after"][str(p)] for p in inputs))
    (out / "status.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("debugger_exit", "timeout", "target_exit_observed", "diagnostic_complete", "observations")}, indent=2))
    return 0 if record["diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare, then capture normal public preprocessing use; no diagnostic hooks."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = "2b39ddaf7cefcbdfb8a7fc143ecfd83466618cb3"
REFS = ROOT / "build-preprocess-evidence/refs"
FROZEN = ROOT / "evidence/preprocess/2026-10-31/refs/manifest.json"
POLICY = ROOT / "docs/PREPROCESS_USAGE_POLICY.md"
MANIFEST_SHA = "a3b08bdfe20f13f122e63a0314969832622daf8cedf0e29330e4b325471af5ff"
ARCHIVES = {
    "libsaccade_preprocess.a": "43cc50bbda382f6b29021065b9af40a6de74ae5e7e593271f099c63599383ad6",
    "libsaccade_resize_cuda.a": "59737cb8efa7a616b29942ca085b3941837c5c16d2cd1fff4af60913f75614e1",
    "libsaccade_sha256.a": "5a309bf1b7eac2787aff0e463297ef039459aa6063c6281876b947b1b26d41d8",
    "libsaccade_file_io.a": "6bc43359c7a82130e771680044b6518b326382424bfd1ad1be4eec0d382b61f4",
}
CODE = ["src/preprocess.cpp", "cuda/resize.cu", "src/sha256.cpp", "src/file_io.cpp",
        "include/saccade/preprocess.hpp", "include/saccade/resize_cuda.hpp", "include/saccade/cleanup.hpp",
        "include/saccade/sha256.hpp", "include/saccade/file_io.hpp"]
SEQUENCE = [0, 4, 1, 4, 2, 4, 3, 4, 5, 4, 6, 4, 0]
PLAN = {"schema": "saccade-preprocess-normal-usage/1", "policy": "t01-preprocess-usage/1",
        "sequence": SEQUENCE, "cycles": 4, "sessions": {"serial": ["serial", 1], "queued": ["queued", 3],
        "dual-a": ["dual", 3], "dual-b": ["dual", 3]}, "total_frames": 208,
        "dual_b_sequence": [(index+1)%7 for index in SEQUENCE],
        "delay_clock64_cycles": 10000000, "delay_placement": "same detect stream after enqueue F02 event and before both pinned D2H consumers",
        "preload_delay_kernel": True, "queued_coverage": "each window3 session must observe prior consumer cudaErrorNotReady at next enqueue",
        "readback_slots_per_session": 3, "caller_jpeg_overwrite": True,
        "comparisons": "all RGB bytes and all FP32 bits exact to frozen refs; finite; repeats; frame/mapping; checked close"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def command(out, label, argv, timeout=60):
    record = {"argv": [str(v) for v in argv], "cwd": str(ROOT), "start_utc": now(), "timeout_seconds": timeout,
              "environment": {k: os.environ[k] for k in ["LD_LIBRARY_PATH", "CUDA_MODULE_LOADING", "CUDA_VISIBLE_DEVICES"] if k in os.environ}}
    (out / f"{label}.command.json").write_text(json.dumps(record, indent=2) + "\n")
    process = subprocess.Popen(record["argv"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        record.update(exit=process.returncode, timeout=False)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        record.update(exit=process.returncode, timeout=True)
    (out / f"{label}.stdout.log").write_bytes(stdout)
    (out / f"{label}.stderr.log").write_bytes(stderr)
    (out / f"{label}.exit").write_text(str(record["exit"]) + "\n")
    record["end_utc"] = now()
    (out / f"{label}.status.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def preflight():
    if sha(FROZEN) != MANIFEST_SHA or sha(REFS / "manifest.json") != MANIFEST_SHA:
        raise RuntimeError("frozen manifest drift")
    manifest = json.loads(FROZEN.read_text())
    if sha(REFS / "cases.tsv") != manifest["cases_tsv_sha256"]:
        raise RuntimeError("fixed case list drift")
    reference_hashes = {}
    for case in manifest["cases"][:7]:
        if case["jpeg"] == "-":
            raise RuntimeError("real JPEG case missing")
        for name, expected in case["hashes"].items():
            actual = sha(REFS / name)
            if actual != expected:
                raise RuntimeError(f"frozen reference mismatch: {name}")
            reference_hashes[name] = actual
    libraries = {}
    for name, expected in ARCHIVES.items():
        path = ROOT / "build-preprocess-evidence/build" / name
        if sha(path) != expected:
            raise RuntimeError(f"historical archive drift: {name}")
        libraries[str(path)] = expected
    if sha(ROOT / "build-preprocess-evidence/build/preprocess") != "41fbeea381448745d4f37cafe1e80c05fc75b68ac1d9d27a7b555849f553925f":
        raise RuntimeError("historical formal CLI identity drift")
    code = {}
    for name in CODE:
        expected = subprocess.check_output(["git", "show", BASE + ":" + name], cwd=ROOT)
        actual = (ROOT / name).read_bytes()
        if actual != expected:
            raise RuntimeError(f"runtime source differs from historical source: {name}")
        code[name] = hashlib.sha256(actual).hexdigest()
    cuda = ROOT / "build-preprocess-evidence/build/CMakeCache.txt"
    cache = cuda.read_text()
    for marker in ["SACCADE_ENABLE_PREPROCESS:BOOL=ON", "CMAKE_BUILD_TYPE:STRING=Release", "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=120"]:
        if marker not in cache:
            raise RuntimeError("active historical cache changed: " + marker)
    flags = (ROOT / "build-preprocess-evidence/build/CMakeFiles/saccade_preprocess.dir/flags.make").read_text()
    if BASE not in flags or 'SACCADE_BUILD_DIRTY=\\\"false\\\"' not in flags:
        raise RuntimeError("historical archive build identity flags not clean source")
    dynamic = {}
    for path in [Path("/opt/cuda/targets/x86_64-linux/lib/libnvjpeg.so.13.2.3.58"),
                 Path("/opt/cuda/targets/x86_64-linux/lib/libcudart.so.13")]:
        dynamic[str(path.resolve())] = sha(path)
    return {"checked_utc": now(), "reference_manifest_sha256": MANIFEST_SHA,
            "reference_hashes": reference_hashes, "reference_cases_tsv_sha256": sha(REFS / "cases.tsv"),
            "archive_hashes": libraries, "runtime_source_commit": BASE, "runtime_source_hashes": code,
            "active_cache_sha256": sha(cuda), "dynamic_library_hashes": dynamic,
            "usage_policy_sha256": sha(POLICY), "harness_sha256": sha(HERE / "usage.cu"),
            "capture_script_sha256": sha(HERE / "capture.py"), "verifier_sha256": sha(HERE / "verify.py"),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=ROOT, text=True)}


def prepare(out):
    checked = preflight()
    (out / "preflight.json").write_text(json.dumps(checked, indent=2) + "\n")
    (out / "PLAN.json").write_text(json.dumps(PLAN, indent=2) + "\n")
    for src, dest in [(HERE / "usage.cu", "usage.cu"), (POLICY, "PREPROCESS_USAGE_POLICY.md"),
                      (FROZEN, "frozen-manifest.json"), (REFS / "cases.tsv", "cases.tsv")]:
        (out / dest).write_bytes(src.read_bytes())
    command(out, "compiler-version", ["/opt/cuda/bin/nvcc", "--version"])
    lib = ROOT / "build-preprocess-evidence/build"
    argv = ["/opt/cuda/bin/nvcc", "--allow-unsupported-compiler", "-std=c++20", "-O2", "-lineinfo", "-arch=sm_120",
            "--cudart", "shared", "-Xcompiler=-Wall,-Wextra,-Wpedantic,-Werror,-ffp-contract=off", "-I" + str(ROOT / "include"),
            str(HERE / "usage.cu"), str(lib / "libsaccade_preprocess.a"), str(lib / "libsaccade_file_io.a"),
            str(lib / "libsaccade_sha256.a"), str(lib / "libsaccade_resize_cuda.a"),
            "-L/opt/cuda/targets/x86_64-linux/lib", "-lnvjpeg", "-Xlinker=-rpath,/opt/cuda/targets/x86_64-linux/lib",
            "-ldl", "-lrt", "-lpthread", "-o", str(out / "usage")]
    result = command(out, "compile", argv)
    if result["exit"] or result["timeout"]:
        raise RuntimeError("compile failed; raw outputs preserved")
    after = preflight()
    for key in ["reference_hashes", "archive_hashes", "runtime_source_hashes", "dynamic_library_hashes", "usage_policy_sha256", "harness_sha256", "capture_script_sha256", "verifier_sha256"]:
        if checked[key] != after[key]:
            raise RuntimeError("identity drift during CPU build: " + key)
    (out / "build.json").write_text(json.dumps({"source": checked, "binary_sha256": sha(out / "usage"),
                                                "plan_sha256": sha(out / "PLAN.json"), "built_utc": now()}, indent=2) + "\n")
    command(out, "ldd", ["ldd", str(out / "usage")])
    print("prepared CPU build only", out, "binary", sha(out / "usage"), flush=True)


def run(out, build):
    for key in ["LD_PRELOAD", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH", "CUDA_INJECTION32_PATH"]:
        if os.environ.get(key):
            raise RuntimeError("normal-use capture forbids injection environment: " + key)
    checked = preflight()
    built = json.loads((build / "build.json").read_text())
    if sha(build / "usage") != built["binary_sha256"]:
        raise RuntimeError("prepared binary drift")
    for key in ["reference_hashes", "archive_hashes", "runtime_source_hashes", "dynamic_library_hashes", "usage_policy_sha256", "harness_sha256", "capture_script_sha256", "verifier_sha256"]:
        if checked[key] != built["source"][key]:
            raise RuntimeError("pre-GPU identity differs from prepared source: " + key)
    mem = Path("/proc/meminfo").read_text()
    (out / "memory-before.txt").write_text(mem)
    if int(next(row.split()[1] for row in mem.splitlines() if row.startswith("MemAvailable:"))) < 4*1024*1024:
        raise RuntimeError("at least 4 GiB MemAvailable required")
    (out / "preflight.json").write_text(json.dumps(checked, indent=2) + "\n")
    (out / "PLAN.json").write_text((build / "PLAN.json").read_text())
    (out / "PREPROCESS_USAGE_POLICY.md").write_bytes(POLICY.read_bytes())
    for label, identity_argv in [("gpu-identity", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
                                  ("os-identity", ["uname", "-a"])]:
        identity = command(out, label, identity_argv)
        if identity["exit"] or identity["timeout"]:
            raise RuntimeError("required environment identity failed: " + label)
    argv = [str(build / "usage"), str(REFS), str(out / "run")]
    result = command(out, "normal-usage", argv, timeout=180)
    after = preflight()
    (out / "postflight.json").write_text(json.dumps(after, indent=2) + "\n")
    modules = {}
    for path in (out / "run").glob("*-loaded-maps.txt"):
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 6 or not parts[-1].startswith("/"):
                continue
            name = parts[-1]
            if any(marker in name for marker in ["libnvjpeg", "libnvcuvid", "libcudart", "libcuda.so", "libdxcore"]):
                modules[name] = {"sha256": sha(name), "bytes": Path(name).stat().st_size,
                                 "measurement": "mapped path observed in live process; mapped file hashed after run"}
    (out / "loaded-modules.json").write_text(json.dumps(modules, indent=2) + "\n")
    identity_keys = ["reference_hashes", "archive_hashes", "runtime_source_hashes", "dynamic_library_hashes", "usage_policy_sha256", "harness_sha256", "capture_script_sha256", "verifier_sha256"]
    identity_stable = all(checked[key] == after[key] for key in identity_keys) and sha(build / "usage") == built["binary_sha256"]
    status = {"normal_process_exit": result["exit"], "timeout": result["timeout"], "identity_stable": identity_stable,
              "binary_sha256": sha(build / "usage"), "plan_sha256": sha(out / "PLAN.json"),
              "policy_sha256": sha(out / "PREPROCESS_USAGE_POLICY.md"), "scope": "fixed seven cases/208 normal usage frames, no full pipeline"}
    (out / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    verification = command(out, "offline-verify", [sys.executable, str(HERE / "verify.py"), str(out)])
    print(json.dumps(status, indent=2), flush=True)
    if result["exit"] or result["timeout"] or not identity_stable or verification["exit"]:
        raise RuntimeError("usage capture failed or coverage unmet; preserved completed prefix")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["prepare", "run"])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--build", type=Path)
    a = p.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if a.phase == "prepare":
        prepare(out)
    else:
        if a.build is None:
            raise RuntimeError("run requires --build")
        run(out, a.build.resolve())


if __name__ == "__main__":
    main()

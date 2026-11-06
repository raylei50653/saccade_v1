#!/usr/bin/env python3
"""Capture bounded, unfiltered HR session sanitizer evidence from existing clean binaries.

This is an additional validation capture, not a replacement for a formal gate.
Each attempt is retained, including incomplete and diagnostic runs. No shell is used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid


TOOLS = ("racecheck", "synccheck", "memcheck", "initcheck")
MODES = {
    "graph-pipelined": [],
    "ordinary-pipelined": ["--no-cuda-graph"],
    "ordinary-serial": ["--serial", "--no-cuda-graph"],
    "graph-serial": ["--serial"],
}
DIAGNOSTIC = {
    "racecheck": "Error: Race reported between",
    "synccheck": "Barrier error detected",
    "memcheck": "Invalid __global__ write",
    "initcheck": "Uninitialized __global__ memory read",
}
INTERNAL = re.compile(
    r"Internal (?:Sanitizer|Racecheck|Memcheck|Synccheck|Initcheck)|"
    r"Internal error encountered|Unknown record type|Error in printing record|"
    r"failure to track a kernel|records might be incomplete|"
    r"didn't track|did not track|not tracked|Unable to allocate|"
    r"No attachable process|No CUDA|Target application returned an error|"
    r"process didn't terminate successfully|Failed to initialize|Device not supported|"
    r"unsupported|out of memory|errors might go undetected", re.I
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path, value):
    with Path(path).open("w") as out:
        json.dump(value, out, indent=2, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_text(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def cgroup_sample(group):
    sample = {"time": now(), "control_group": group}
    if group:
        base = Path("/sys/fs/cgroup") / group.lstrip("/")
        for name in ("memory.current", "memory.peak", "memory.swap.current", "memory.swap.peak",
                     "memory.events", "memory.events.local"):
            try:
                sample[name] = (base / name).read_text().strip()
            except OSError:
                pass
    return sample


def target_wrapper(spec_path):
    """Runs inside the isolated service and records final stats before collection."""
    spec = json.loads(Path(spec_path).read_text())
    group = ""
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            group = line[3:]
    started = now()
    result = subprocess.run(spec["command"], env=spec["environment"], check=False)
    record = cgroup_sample(group)
    record.update(started=started, finished=now(), returncode=result.returncode)
    save(spec["target_record"], record)
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


def source_snapshot(source, build, engine, sequence, frames):
    revision = run_text(["git", "-C", str(source), "rev-parse", "HEAD"])
    status = run_text(["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"])
    if status:
        raise RuntimeError("the supplied source worktree must be clean")
    paths = subprocess.check_output([
        "git", "-C", str(source), "ls-files", "-z", "--", "CMakeLists.txt", "cmake", "include",
        "src", "cuda", "tests", "tools", "third_party", "scripts",
    ]).split(b"\0")
    sources = {os.fsdecode(p): digest(source / os.fsdecode(p)) for p in paths if p}
    binaries = {str(p): digest(p) for p in sorted(build.iterdir())
                if p.is_file() and (os.access(p, os.X_OK) or p.suffix in (".a", ".so") or ".so." in p.name)}
    binaries[str(build / "CMakeCache.txt")] = digest(build / "CMakeCache.txt")
    info = sequence_info(sequence)
    inputs = {str(sequence / "seqinfo.ini"): digest(sequence / "seqinfo.ini")}
    for frame in range(1, frames + 1):
        path = sequence / info["imDir"] / f"{frame:06d}{info['imExt']}"
        inputs[str(path)] = digest(path)
    return {"revision": revision, "source_status": status, "source_files": sources,
            "binaries": binaries, "engine": {str(engine): digest(engine)}, "inputs": inputs}


def sequence_info(sequence):
    fields = {}
    for line in (sequence / "seqinfo.ini").read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    for key in ("seqLength", "imDir", "imExt"):
        if key not in fields:
            raise RuntimeError(f"sequence metadata missing {key}")
    return fields


class Capture:
    def __init__(self, args, output):
        self.args, self.output = args, output
        self.active_unit = None
        self.wrapper = output / "target-wrapper.py"
        self.wrapper.write_text(Path(__file__).read_text())
        # A fresh environment prevents inherited sanitizer filters, instrumentation
        # overrides, or launch blocking from silently changing the experiment.
        keys = ("HOME", "USER", "LOGNAME", "PATH", "TMPDIR", "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES", "CUDA_MODULE_LOADING")
        self.environment = {key: os.environ[key] for key in keys if key in os.environ}
        self.environment.setdefault("PATH", "/usr/bin:/bin:/opt/cuda/bin")
        self.removed_environment = {
            key: value for key, value in os.environ.items()
            if key.startswith("NV_COMPUTE_SANITIZER_") or key in (
                "CUDA_LAUNCH_BLOCKING", "CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH")
        }

    def stop(self):
        if self.active_unit:
            subprocess.run(["systemctl", "--user", "stop", self.active_unit],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=False)

    def run(self, name, command):
        unit = f"saccade-hr-{uuid.uuid4().hex[:12]}.service"
        self.active_unit = unit
        target_record = self.output / f"{name}.target.json"
        spec = self.output / f"{name}.spec.json"
        timeout_command = ["/usr/bin/timeout", "--kill-after=10s", str(self.args.timeout), *map(str, command)]
        save(spec, {"command": timeout_command, "environment": self.environment,
                    "target_record": str(target_record)})
        launch = [
            "systemd-run", "--user", "--wait", "--pipe", "--collect", f"--unit={unit}",
            f"--working-directory={self.args.source}", "--property=MemoryAccounting=yes",
            f"--property=MemoryHigh={self.args.memory_high}", f"--property=MemoryMax={self.args.memory_max}",
            "--property=MemorySwapMax=0", "--property=KillMode=control-group", "--property=OOMPolicy=kill",
            f"--property=RuntimeMaxSec={self.args.timeout + 30}",
            "/usr/bin/env", "-i", *[f"{k}={v}" for k, v in self.environment.items()],
            sys.executable, str(self.wrapper), "--_target", str(spec),
        ]
        save(self.output / f"{name}.command.json", {"target": list(map(str, command)), "launch": launch})
        record = {"name": name, "unit": unit, "started": now()}
        print(f"{name}: started", flush=True)
        peak = 0
        events = {}
        systemd_results = set()
        try:
            with (self.output / f"{name}.log").open("w") as log, \
                    (self.output / f"{name}.telemetry.jsonl").open("w") as telemetry:
                process = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + self.args.timeout + 60
                while process.poll() is None:
                    shown = subprocess.run([
                        "systemctl", "--user", "show", unit, "--property=ControlGroup",
                        "--property=MemoryCurrent", "--property=MemoryPeak", "--property=Result",
                        "--property=ExecMainStatus", "--property=ActiveState", "--property=SubState",
                    ], capture_output=True, text=True, timeout=5, check=False)
                    props = dict(line.split("=", 1) for line in shown.stdout.splitlines() if "=" in line)
                    if props.get("Result"):
                        systemd_results.add(props["Result"])
                    sample = cgroup_sample(props.get("ControlGroup", ""))
                    sample["systemd"] = props
                    telemetry.write(json.dumps(sample, sort_keys=True) + "\n")
                    telemetry.flush()
                    os.fsync(telemetry.fileno())
                    for field in (sample.get("memory.peak"), props.get("MemoryPeak")):
                        if field and field.isdigit():
                            peak = max(peak, int(field))
                    if "memory.events" in sample:
                        events = dict(line.split() for line in sample["memory.events"].splitlines())
                    if time.monotonic() > deadline:
                        self.stop()
                        process.wait(timeout=20)
                        record["outer_timeout"] = True
                        break
                    time.sleep(0.5)
                record["returncode"] = process.wait(timeout=20)
        finally:
            self.stop()
            self.active_unit = None
        record.update(finished=now(), sampled_memory_peak_bytes=peak, sampled_memory_events=events,
                      sampled_systemd_results=sorted(systemd_results))
        if target_record.exists():
            record["target"] = json.loads(target_record.read_text())
        save(self.output / f"{name}.execution.json", record)
        print(f"{name}: exit={record['returncode']} peak={peak} bytes", flush=True)
        return record


def output_hashes(output, name):
    return {suffix: digest(output / f"{name}.{suffix}")
            for suffix in ("txt", "bin", "scmo") if (output / f"{name}.{suffix}").is_file()}


def packet_size(source):
    text = (source / "include/saccade/protocol.hpp").read_text()
    values = {}
    for key in ("packet_header_bytes", "packet_capacity", "packet_row_bytes"):
        match = re.search(rf"\b{key}\s*=\s*(\d+)\s*;", text)
        if not match:
            raise RuntimeError(f"cannot establish packet byte size from {key}")
        values[key] = int(match[1])
    return values["packet_header_bytes"] + values["packet_capacity"] * values["packet_row_bytes"]


def summary_ok(tool, text):
    if tool == "racecheck":
        found = re.findall(r"^========= RACECHECK SUMMARY:.*$", text, re.M)
        return found == ["========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"]
    found = re.findall(r"^========= ERROR SUMMARY:.*$", text, re.M)
    return found == ["========= ERROR SUMMARY: 0 errors"]


def clean_sanitizer_log(tool, text, require_launches=False):
    """Unknown sanitizer output is diagnostic, even with a trailing zero summary."""
    if not summary_ok(tool, text) or INTERNAL.search(text):
        return False
    banner = False
    launches = 0
    for line in text.splitlines():
        if not line.startswith("========="):
            continue
        body = line[9:].strip()
        if not body or body == "COMPUTE-SANITIZER":
            banner = banner or body == "COMPUTE-SANITIZER"
            continue
        if body in ("ERROR SUMMARY: 0 errors", "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"):
            continue
        if re.fullmatch(r"Launch #\d+(?: \(filtered: (?:no|false|0)\))?", body):
            launches += 1
            continue
        if re.match(r"^(?:Kernel:|Grid:|Block:|Cluster:|Stream:|Host Frame:)", body):
            continue
        if body.startswith("Saved host backtrace up to driver entry point at kernel launch time"):
            continue
        return False
    return banner and (not require_launches or launches > 0)


def positive_summary_ok(tool, text):
    if tool == "racecheck":
        found = re.findall(r"^========= RACECHECK SUMMARY:.*$", text, re.M)
        match = re.fullmatch(r"========= RACECHECK SUMMARY: ([\d,]+) hazards? displayed "
                             r"\(([\d,]+) errors?, ([\d,]+) warnings?\)", found[0]) if len(found) == 1 else None
        return bool(match and int(match[1].replace(",", "")) > 0 and
                    int(match[2].replace(",", "")) + int(match[3].replace(",", "")) > 0)
    found = re.findall(r"^========= ERROR SUMMARY:.*$", text, re.M)
    match = re.fullmatch(r"========= ERROR SUMMARY: ([\d,]+) errors?", found[0]) if len(found) == 1 else None
    return bool(match and int(match[1].replace(",", "")) > 0)


def incomplete(record, text):
    event_sources = [record.get("sampled_memory_events", {})]
    for key in ("memory.events", "memory.events.local"):
        raw = record.get("target", {}).get(key, "")
        event_sources.append(dict(line.split() for line in raw.splitlines() if len(line.split()) == 2))
    oom = any(int(events.get(key, 0)) for events in event_sources
              for key in ("oom", "oom_kill", "oom_group_kill"))
    unit_failure = any(value in ("oom-kill", "timeout", "resources")
                       for value in record.get("sampled_systemd_results", []))
    return bool(record.get("outer_timeout") or record["returncode"] in (124, 137, -9) or
                record.get("target", {}).get("returncode") in (124, 137, -9) or
                oom or unit_failure or "oom-kill" in text or "timed out" in text.lower())


def application_ok(text, args, revision, engine_sha, mode):
    return all((
        f"source_commit={revision}\n" in text,
        "source_dirty=false\n" in text,
        f"engine_sha256={engine_sha}\n" in text,
        "motion_profile=v1-motion/2\n" in text,
        "profile=hr-1280x768\n" in text,
        ("cuda_graph=true\n" if mode.startswith("graph-") else "cuda_graph=false\n") in text,
        len(re.findall(rf"^track_sequence frames={args.frames} mode={'serial' if mode.endswith('-serial') else 'pipelined'} wall_ms=", text, re.M)) == 1,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-name-regex", required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--tools", nargs="+", choices=TOOLS, default=list(TOOLS))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--memory-high", default="16G")
    parser.add_argument("--memory-max", default="24G")
    parser.add_argument("--dump-kernel-launches", action="store_true",
                        help="Diagnostic inventory only: installed sanitizer counts launch records in its error summary; never accepted as PASS")
    parser.add_argument("--engine", type=Path, default=Path("/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine"))
    parser.add_argument("--sequence", type=Path, default=Path("/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP"))
    parser.add_argument("--sanitizer", type=Path, default=Path("/opt/cuda/bin/compute-sanitizer"))
    args = parser.parse_args()
    for key in ("build", "source", "output", "engine", "sequence", "sanitizer"):
        setattr(args, key, getattr(args, key).resolve())
    if args.frames < 1 or args.timeout < 1:
        parser.error("frames and timeout must be positive")
    if len(set(args.tools)) != len(args.tools) or len(set(args.modes)) != len(args.modes):
        parser.error("tools and modes must not contain duplicates")
    if args.output.exists():
        parser.error("output must be a fresh directory")
    if args.frames > int(sequence_info(args.sequence)["seqLength"]):
        parser.error("requested frames exceed the input sequence length")
    for name in ("track_sequence", "postprocess_sanitizer_control", "CMakeCache.txt"):
        if not (args.build / name).is_file():
            parser.error(f"missing build artifact: {name}")
    cache = (args.build / "CMakeCache.txt").read_text()
    if f"CMAKE_HOME_DIRECTORY:INTERNAL={args.source}\n" not in cache:
        parser.error("build CMakeCache belongs to a different source directory")
    args.output.mkdir(parents=True)
    capture = Capture(args, args.output)
    results = []
    manifest = {"schema": "hr-session-sanitizer/1", "started": now(),
                "formal_acceptance": False, "profile": "hr-1280x768", "motion_profile": "v1-motion/2",
                "frames": args.frames, "sequence": str(args.sequence), "modes": args.modes, "tools": args.tools,
                "kernel_filter": args.kernel_name_regex, "launch_limit": None, "forced_synchronization": False,
                "sanitizer_environment_overrides": {}, "environment": capture.environment,
                "dump_kernel_launches": args.dump_kernel_launches,
                "diagnostic_inventory": args.dump_kernel_launches,
                "print_limit": {t: 1000 if t == "initcheck" else 0 for t in args.tools},
                "removed_environment": capture.removed_environment,
                "runner_sha256": digest(__file__), "sanitizer_sha256": digest(args.sanitizer),
                "target_wrapper_sha256": digest(capture.wrapper),
                "resource_limits": {"timeout_seconds": args.timeout, "memory_high": args.memory_high,
                                    "memory_max": args.memory_max, "memory_swap_max": 0},
                "scope": "SCOPED HR custom kernels plus CUB only, full stated prefix and modes; external library kernels excluded. Not an unfiltered full-session PASS."}
    import csv
    inventory = Path(__file__).resolve().parent / "inventory-completed-scope/counts.csv"
    rows = [r for r in csv.DictReader(inventory.open()) if r['run']=='00-synccheck-graph-pipelined']
    selected = [r for r in rows if re.search(args.kernel_name_regex,r['kernel'])]
    manifest['scope_inventory'] = {'source':str(inventory), 'sha256':digest(inventory),
        'all_names':len({r['kernel'] for r in rows}), 'selected_names':sorted({r['kernel'] for r in selected}),
        'selected_launches':sum(int(r['count']) for r in selected),
        'basis':'Exact same clean binary names from complete unfiltered inventory; regex predicate applied to every observed kernel. No launch limit.'}
    if not selected: raise RuntimeError('filter matches no observed kernels')
    lock_path = Path.home() / ".cache/saccade-v1/gpu.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        print(f"Waiting for GPU lock: {lock_path}", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = source_snapshot(args.source, args.build, args.engine, args.sequence, args.frames)
        save(args.output / "custody-before.json", before)
        manifest["source_revision"] = before["revision"]
        save(args.output / "manifest.json", manifest)
        prefix = args.output / "prefix"
        prefix.mkdir()
        (prefix / "seqinfo.ini").write_text(re.sub(r"(?m)^seqLength=.*$", f"seqLength={args.frames}",
                                                   (args.sequence / "seqinfo.ini").read_text()))
        info = sequence_info(args.sequence)
        (prefix / info["imDir"]).symlink_to(args.sequence / info["imDir"], target_is_directory=True)
        manifest["prefix_seqinfo_sha256"] = digest(prefix / "seqinfo.ini")
        engine_sha = next(iter(before["engine"].values()))
        expected_packet_bytes = args.frames * packet_size(args.source)
        manifest["expected_packet_file_bytes"] = expected_packet_bytes

        def command(name, mode):
            if (digest(prefix / "seqinfo.ini") != manifest["prefix_seqinfo_sha256"] or
                    not (prefix / info["imDir"]).is_symlink() or
                    (prefix / info["imDir"]).resolve() != (args.sequence / info["imDir"]).resolve()):
                raise RuntimeError("actual prefix metadata or input symlink changed")
            return [args.build / "track_sequence", args.engine, prefix, args.output / f"{name}.txt",
                    "--profile", "hr-1280x768", "--packets", args.output / f"{name}.bin",
                    "--motion", args.output / f"{name}.scmo", *MODES[mode]]

        try:
            baseline = None
            baseline_ok = True
            for mode in args.modes:
                name = f"baseline-{mode}"
                record = capture.run(name, command(name, mode))
                text = (args.output / f"{name}.log").read_text(errors="replace")
                hashes = output_hashes(args.output, name)
                okay = (record["returncode"] == 0 and record.get("target", {}).get("returncode") == 0 and
                        not incomplete(record, text) and
                        application_ok(text, args, before["revision"], engine_sha, mode) and len(hashes) == 3 and
                        (args.output / f"{name}.bin").stat().st_size == expected_packet_bytes)
                if baseline is None and okay:
                    baseline = hashes
                okay = okay and hashes == baseline
                baseline_ok = baseline_ok and okay
                results.append({"name": name, "status": "PASS" if okay else "FAIL", "outputs_sha256": hashes,
                                "execution": record})
                save(args.output / "results.json", results)
            if not baseline_ok:
                raise RuntimeError("ordinary baseline completion, identity or cross-mode byte parity failed")
            controls = {}
            for tool in args.tools:
                options = [args.sanitizer, "--tool", tool, "--error-exitcode", "99", "--print-limit",
                           "1000" if tool == "initcheck" else "0"]
                if args.dump_kernel_launches:
                    options += ["--dump-kernel-launches"]
                if tool == "racecheck":
                    options += ["--racecheck-num-workers", "1"]
                if tool == "memcheck":
                    options += ["--report-api-errors", "explicit"]
                name = f"control-{tool}"
                record = capture.run(name, [*options, args.build / "postprocess_sanitizer_control", tool])
                text = (args.output / f"{name}.log").read_text(errors="replace")
                okay = (record["returncode"] == 99 and record.get("target", {}).get("returncode") == 99 and
                        not incomplete(record, text) and
                        DIAGNOSTIC[tool] in text and f"control={tool} launch=" in text and
                        not INTERNAL.search(text.replace("Target application returned an error", "")) and
                        positive_summary_ok(tool, text))
                controls[tool] = okay
                results.append({"name": name, "status": "PASS" if okay else "FAIL", "execution": record,
                                "expected_diagnostic": DIAGNOSTIC[tool]})
                save(args.output / "results.json", results)
                for mode in args.modes:
                    name = f"{tool}-{mode}"
                    record = capture.run(name, [*options, "--kernel-name", f"regex={args.kernel_name_regex}", *command(name, mode)])
                    text = (args.output / f"{name}.log").read_text(errors="replace")
                    hashes = output_hashes(args.output, name)
                    complete = application_ok(text, args, before["revision"], engine_sha, mode)
                    parity = (len(hashes) == 3 and hashes == baseline and
                              (args.output / f"{name}.bin").stat().st_size == expected_packet_bytes)
                    internal = bool(INTERNAL.search(text))
                    log_clean = clean_sanitizer_log(tool, text, require_launches=args.dump_kernel_launches)
                    okay = (record["returncode"] == 0 and record.get("target", {}).get("returncode") == 0 and
                            not args.dump_kernel_launches and not incomplete(record, text) and complete and
                            parity and log_clean and controls[tool])
                    status = "PASS" if okay else "INCOMPLETE" if incomplete(record, text) else "OPEN"
                    results.append({"name": name, "status": status, "target_completed": complete,
                                    "byte_parity": parity, "zero_summary": summary_ok(tool, text),
                                    "internal_tool_error": internal, "positive_control_passed": controls[tool],
                                    "sanitizer_log_clean": log_clean,
                                    "inventory_diagnostic": args.dump_kernel_launches,
                                    "outputs_sha256": hashes, "execution": record})
                    save(args.output / "results.json", results)
                    print(f"{name}: {status}; completed={complete} byte_parity={parity}", flush=True)
        except BaseException as error:
            manifest["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            capture.stop()
            try:
                after = source_snapshot(args.source, args.build, args.engine, args.sequence, args.frames)
                save(args.output / "custody-after.json", after)
                manifest["prefix_custody_unchanged"] = (
                    digest(prefix / "seqinfo.ini") == manifest["prefix_seqinfo_sha256"] and
                    (prefix / info["imDir"]).is_symlink() and
                    (prefix / info["imDir"]).resolve() == (args.sequence / info["imDir"]).resolve())
                manifest["custody_unchanged"] = before == after and manifest["prefix_custody_unchanged"]
                manifest["sanitizer_unchanged"] = digest(args.sanitizer) == manifest["sanitizer_sha256"]
                manifest["target_wrapper_unchanged"] = digest(capture.wrapper) == manifest["target_wrapper_sha256"]
                manifest["custody_unchanged"] = (manifest["custody_unchanged"] and manifest["sanitizer_unchanged"] and
                                                 manifest["target_wrapper_unchanged"])
            except Exception as error:
                manifest["custody_unchanged"] = False
                manifest["custody_error"] = str(error)
            manifest["finished"] = now()
            manifest["status"] = ("PASS" if results and not manifest.get("error") and
                                  manifest["custody_unchanged"] and all(r["status"] == "PASS" for r in results) else "OPEN")
            save(args.output / "manifest.json", manifest)
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--_target":
        sys.exit(target_wrapper(sys.argv[2]))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        print(f"hr-sanitizer-validation: {error}", file=sys.stderr)
        sys.exit(2)

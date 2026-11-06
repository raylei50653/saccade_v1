#!/usr/bin/env python3
"""CPU-only scope and fail-closed checks for the HR sanitizer capture runner."""
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

def discover_repo_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "scripts/run_hr_sanitizer_validation.py").is_file() and (candidate / "CMakeLists.txt").is_file():
            return candidate
    raise RuntimeError("cannot discover repository ancestor for evidence rerun")


ROOT = discover_repo_root()
RUNNER = ROOT / "scripts/run_hr_sanitizer_validation.py"
spec = importlib.util.spec_from_file_location("hr_validation", RUNNER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = []


def check(name, condition, detail=None):
    checks.append({"name": name, "passed": bool(condition), "detail": detail})


for limit, status in [(0, "PASS"), (1, "PASS_FORCED_SYNC"), (16, "PASS_FORCED_SYNC")]:
    check(f"success-status-{limit}", m.success_status(limit) == status)
    cases = [
        ([{"status": "PASS"}, {"status": status}], True, None, status),
        ([{"status": "PASS"}, {"status": "OPEN"}], True, None, "OPEN"),
        ([{"status": "PASS"}, {"status": "INCOMPLETE"}], True, None, "OPEN"),
        ([{"status": status}], False, None, "OPEN"),
        ([{"status": status}], True, "error", "OPEN"),
        ([], True, None, "OPEN"),
    ]
    for number, (rows, custody, error, expected) in enumerate(cases):
        check(f"aggregate-{limit}-{number}", m.aggregate_status(rows, custody, error, limit) == expected)
check("forced-row-cannot-be-native-pass",
      m.aggregate_status([{"status": "PASS_FORCED_SYNC"}], True, None, 0) == "OPEN")

base_command = [sys.executable, "-I", str(RUNNER), "--source", "/nonexistent", "--build",
                "/nonexistent", "--output", "/nonexistent"]
for number, (flags, error) in enumerate([
    (["--tools", "racecheck", "--force-synchronization-limit", "-1"], "must be nonnegative"),
    (["--tools", "synccheck", "--force-synchronization-limit", "1"], "requires --tools racecheck only"),
    (["--tools", "racecheck", "memcheck", "--force-synchronization-limit", "1"], "requires --tools racecheck only"),
    (["--force-synchronization-limit", "1"], "requires --tools racecheck only"),
    (["--tools", "racecheck", "--racecheck-num-workers", "-1"], "racecheck-num-workers must be nonnegative"),
]):
    result = subprocess.run(base_command + flags, capture_output=True, text=True)
    check(f"argument-rejected-{number}", result.returncode == 2 and error in result.stderr)

banner = "========= COMPUTE-SANITIZER\n"
zero = "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"
check("clean-zero-summary", m.clean_sanitizer_log("racecheck", banner + zero))
for name, text in {
    "missing-summary": banner,
    "missing-banner": zero,
    "duplicate-summary": banner + zero + zero,
    "reported-hazard": banner + "========= Error: Race reported between Write and Read\n" + zero,
    "dropped-hazard-count": banner + "========= RACECHECK SUMMARY: 4 hazards displayed (0 errors, 0 warnings)\n",
    "internal-error": banner + "========= Internal Sanitizer Error: failure to track a kernel\n" + zero,
    "unknown-diagnostic": banner + "========= An unrecognized diagnostic\n" + zero,
    "truncated-dump": banner + "========= Launch #1 (filtered: no)\n",
    "filtered-launch": banner + "========= Launch #1 (filtered: yes)\n" + zero,
}.items():
    check(f"log-rejected-{name}", not m.clean_sanitizer_log("racecheck", text))
check("inventory-requires-launches", not m.clean_sanitizer_log("racecheck", banner + zero, True))

for name, record in {
    "timeout": {"returncode": 124},
    "target-timeout": {"returncode": 0, "target": {"returncode": 124}},
    "sampled-oom": {"returncode": 0, "sampled_memory_events": {"oom_kill": 1}},
    "target-oom": {"returncode": 0, "target": {"memory.events": "oom 1\noom_kill 1\n"}},
    "systemd-oom": {"returncode": 0, "sampled_systemd_results": ["oom-kill"]},
}.items():
    check(f"incomplete-{name}", m.incomplete(record, ""))

with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    source = base / "source"
    source.mkdir()
    build = base / "build"
    build.mkdir()
    for name in ("track_sequence", "postprocess_sanitizer_control"):
        (build / name).write_bytes(b"fake")
    (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n")
    engine = base / "engine"
    engine.write_bytes(b"engine")
    sanitizer = base / "sanitizer"
    sanitizer.write_bytes(b"sanitizer")
    sequence = base / "sequence"
    sequence.mkdir()
    (sequence / "img1").mkdir()
    (sequence / "seqinfo.ini").write_text("seqLength=1\nimDir=img1\nimExt=.jpg\n")
    snapshot = {"revision": "abc", "engine": {str(engine): m.digest(engine)}}
    commands = []
    scenario = "clean"

    class FakeCapture:
        def __init__(self, args, output):
            self.args, self.output = args, output
            self.environment, self.removed_environment = {}, {}
            self.wrapper = output / "wrapper.py"
            self.wrapper.write_bytes(b"fake")

        def stop(self):
            pass

        def run(self, name, command):
            commands.append(list(map(str, command)))
            control = name.startswith("control-")
            session = name.startswith("racecheck-")
            if session and scenario == "interrupt":
                raise KeyboardInterrupt("test interruption")
            app = ("source_commit=abc\nsource_dirty=false\n"
                   f"engine_sha256={m.digest(engine)}\n"
                   "motion_profile=v1-motion/2\nprofile=hr-1280x768\ncuda_graph=true\n")
            if not (session and scenario == "missing-completion"):
                app += "track_sequence frames=1 mode=pipelined wall_ms=1\n"
            if control:
                log = (banner + m.DIAGNOSTIC["racecheck"] + "\ncontrol=racecheck launch=1\n" +
                       (zero if scenario == "positive-control-fail" else
                        "========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)\n"))
            elif session:
                log = app + banner
                if scenario == "hazard":
                    log += "========= Error: Race reported between Write and Read\n"
                elif scenario == "internal":
                    log += "========= Internal Sanitizer Error: failure to track a kernel\n"
                if self.args.dump_kernel_launches:
                    log += "========= Launch #1 (filtered: no)\n"
                if scenario not in ("missing-summary", "truncated-dump"):
                    log += zero
            else:
                log = app
            (self.output / (name + ".log")).write_text(log)
            for suffix in (".bin", ".txt", ".scmo"):
                data = b"efgh" if session and scenario == "bad-parity" else b"abcd"
                (self.output / (name + suffix)).write_bytes(data)
            rc = 99 if control else 124 if session and scenario == "incomplete" else 0
            record = {"returncode": rc, "target": {"returncode": rc}}
            if session and scenario == "no-target":
                record.pop("target")
            return record

    cases = [(0, "clean", "PASS", 1), (16, "clean", "PASS_FORCED_SYNC", 1),
             (8, "clean-auto", "PASS_FORCED_SYNC", 0), (8, "clean-four", "PASS_FORCED_SYNC", 4)]
    cases += [(8, name, "OPEN", 1) for name in (
        "hazard", "internal", "missing-summary", "truncated-dump", "dump",
        "incomplete", "no-target", "bad-parity", "missing-completion",
        "positive-control-fail", "bad-custody", "interrupt")]
    for number, (limit, scenario, expected, workers) in enumerate(cases):
        output = base / f"out{number}"
        argv = ["runner", "--source", str(source), "--build", str(build), "--output", str(output),
                "--engine", str(engine), "--sequence", str(sequence), "--sanitizer", str(sanitizer),
                "--frames", "1", "--tools", "racecheck", "--modes", "graph-pipelined",
                "--force-synchronization-limit", str(limit), "--racecheck-num-workers", str(workers)]
        if scenario in ("dump", "truncated-dump"):
            argv.append("--dump-kernel-launches")
        commands.clear()
        snapshots = [snapshot, {**snapshot, "changed": True}] if scenario == "bad-custody" else [snapshot, snapshot]
        interrupted = False
        with patch.object(sys, "argv", argv), patch.object(m, "Capture", FakeCapture), \
                patch.object(m, "source_snapshot", side_effect=snapshots), \
                patch.object(m, "packet_size", return_value=4), \
                patch.object(Path, "home", return_value=base):
            try:
                code = m.main()
            except KeyboardInterrupt:
                code, interrupted = None, True
        manifest = json.loads((output / "manifest.json").read_text())
        results = json.loads((output / "results.json").read_text())
        prefix = f"integration-{limit}-{scenario}"
        check(prefix + "-status", manifest["status"] == expected)
        check(prefix + "-exit", interrupted if scenario == "interrupt" else code == (0 if expected.startswith("PASS") else 1))
        check(prefix + "-forced-flag", manifest["forced_synchronization"] == bool(limit))
        check(prefix + "-limit", manifest["force_synchronization_limit"] == limit)
        check(prefix + "-workers", manifest["racecheck_num_workers"] == workers)
        native = expected == "PASS" and limit == 0
        check(prefix + "-native", manifest["native_schedule_acceptance"] == native)
        if scenario.startswith("clean"):
            check(prefix + "-session-status", results[-1]["status"] == expected)
            check(prefix + "-session-native", results[-1]["native_schedule_acceptance"] == native)
            checked = commands[1:]
            check(prefix + "-command-count", len(checked) == 2)
            for index, command in enumerate(checked):
                if limit:
                    position = command.index("--force-synchronization-limit")
                    okay = command[position + 1] == str(limit)
                else:
                    okay = "--force-synchronization-limit" not in command
                check(prefix + f"-command-option-{index}", okay)
                worker_position = command.index("--racecheck-num-workers")
                check(prefix + f"-worker-option-{index}", command[worker_position + 1] == str(workers))
        elif scenario == "incomplete":
            check(prefix + "-incomplete-row", results[-1]["status"] == "INCOMPLETE")
        elif scenario == "interrupt":
            check(prefix + "-error-retained", "KeyboardInterrupt" in manifest.get("error", ""))

# Exercise actual Capture.run bookkeeping without launching systemd or CUDA.
with tempfile.TemporaryDirectory() as tmp:
    output = Path(tmp)
    args = SimpleNamespace(source=output, timeout=1, memory_high="8G", memory_max="8G")
    capture = m.Capture(args, output)
    for index, (periodic, final, expected) in enumerate([(101, 436, 436), (500, 436, 500), (101, None, 101)]):
        name = f"peak-{index}"
        target = {"returncode": 0}
        if final is not None:
            target["memory.peak"] = str(final)
        (output / f"{name}.target.json").write_text(json.dumps(target))
        process = type("FakeProcess", (), {"poll": lambda self: next(self.polls),
                                           "wait": lambda self, **kwargs: 0})()
        process.polls = iter([None, 0])
        shown = subprocess.CompletedProcess([], 0, stdout=f"ControlGroup=/fake\nMemoryPeak={periodic}\nResult=success\n")
        with patch.object(m.subprocess, "Popen", return_value=process), \
                patch.object(m.subprocess, "run", return_value=shown), \
                patch.object(m, "cgroup_sample", return_value={"memory.peak": str(periodic)}), \
                patch.object(m.time, "sleep"), patch.object(capture, "stop"):
            record = capture.run(name, ["/fake-target"])
        check(f"final-peak-{index}", record["sampled_memory_peak_bytes"] == expected)
        saved = json.loads((output / f"{name}.execution.json").read_text())
        check(f"saved-final-peak-{index}", saved["sampled_memory_peak_bytes"] == expected)


report = {
    "schema": "hr-sanitizer-runner-cpu-tests/1",
    "time": dt.datetime.now(dt.timezone.utc).isoformat(),
    "gpu_work": False,
    "runner": str(RUNNER),
    "runner_sha256": hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
    "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "checks": checks,
    "passed": sum(check["passed"] for check in checks),
    "failed": sum(not check["passed"] for check in checks),
}
report["status"] = "PASS" if not report["failed"] else "FAIL"
Path(__file__).with_name("runner-tests.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"{report['passed']} CPU checks passed; {report['failed']} failed; no GPU work")
sys.exit(bool(report["failed"]))

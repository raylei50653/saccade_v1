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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).with_name("capture_custom.py")
spec = importlib.util.spec_from_file_location("hr_validation", RUNNER)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = []


def check(name, condition, detail=None):
    checks.append({"name": name, "passed": bool(condition), "detail": detail})


compile(RUNNER.read_text(), str(RUNNER), "exec")
check("syntax", True)
help_result = subprocess.run([sys.executable, "-I", str(RUNNER), "--help"], capture_output=True, text=True)
check("help", help_result.returncode == 0 and "PASS_FILTERED_FORCED_SYNC" in help_result.stdout)
inventory = m.reference_inventory()
check("reference-kernel-count", len(inventory["selected_names"]) == 36)
check("reference-launch-count", inventory["selected_launches"] == 2518)
check("fixed-session-predicate", m.CUSTOM_REGEX == r"_ZN7saccade|_ZN3cub")
check("control-included", bool(m.re.search(m.CONTROL_REGEX, "_Z11shared_racePf")))
check("control-not-in-session-predicate", not m.re.search(m.CUSTOM_REGEX, "_Z11shared_racePf"))
check("external-excluded", not m.re.search(m.CUSTOM_REGEX, "__myl_abc"))
check("missing-row-rejected", m.aggregate_status([{"status":"PASS"}], True, None, 8, 1) == "OPEN")
for limit in (0, -1):
    result = subprocess.run([sys.executable, "-I", str(RUNNER), "--output", "/nonexistent", "--force-synchronization-limit", str(limit)], capture_output=True, text=True)
    check(f"nonpositive-limit-{limit}", result.returncode == 2 and "positive force-synchronization-limit" in result.stderr)
negative_workers = subprocess.run([sys.executable, "-I", str(RUNNER), "--output", "/nonexistent", "--racecheck-num-workers", "-1"], capture_output=True, text=True)
check("negative-workers-rejected", negative_workers.returncode == 2 and "racecheck-num-workers must be nonnegative" in negative_workers.stderr)
banner = "========= COMPUTE-SANITIZER\n"
zero = "========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)\n"

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

    cases = [(8, "clean", "PASS_FILTERED_FORCED_SYNC", 1),
             (8, "clean-auto", "PASS_FILTERED_FORCED_SYNC", 0),
             (8, "clean-eight", "PASS_FILTERED_FORCED_SYNC", 8)]
    cases += [(8, name, "OPEN", 1) for name in (
        "hazard", "internal", "missing-summary", "truncated-dump", "dump",
        "incomplete", "no-target", "bad-parity", "missing-completion",
        "positive-control-fail", "bad-custody", "interrupt")]
    for number, (limit, scenario, expected, workers) in enumerate(cases):
        output = base / f"out{number}"
        argv = ["runner", "--source", str(source), "--build", str(build), "--output", str(output),
                "--engine", str(engine), "--sequence", str(sequence), "--sanitizer", str(sanitizer),
                "--frames", "1", "--modes", "graph-pipelined",
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
                worker_index = command.index("--racecheck-num-workers")
                check(prefix + f"-worker-command-{index}", command[worker_index + 1] == str(workers))
                filter_index = command.index("--kernel-name")
                expected_regex = m.CONTROL_REGEX if index == 0 else m.CUSTOM_REGEX
                check(prefix + f"-fixed-filter-{index}", command[filter_index + 1] == "regex=" + expected_regex)
        elif scenario == "incomplete":
            check(prefix + "-incomplete-row", results[-1]["status"] == "INCOMPLETE")
        elif scenario == "interrupt":
            check(prefix + "-error-retained", "KeyboardInterrupt" in manifest.get("error", ""))

report = {
    "schema": "hr-custom-sanitizer-runner-cpu-tests/1",
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
Path(__file__).with_name("custom-runner-tests.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"{report['passed']} CPU checks passed; {report['failed']} failed; no GPU work")
sys.exit(bool(report["failed"]))

#!/usr/bin/env python3
"""Capture bounded HR custom/CUB racecheck evidence; never full/native acceptance.

Derived from the earlier custom capture main routine. Process isolation, output,
identity, summary, completion and custody helpers are imported from the current
workspace runner. The only session kernel predicate is the fixed namespace regex.
"""
import argparse
import csv
import fcntl
import importlib.util
from pathlib import Path
import re
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
ROOT_RUNNER = ROOT / "scripts/run_hr_sanitizer_validation.py"
spec = importlib.util.spec_from_file_location("hr_validation", ROOT_RUNNER)
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
for helper in ("Capture", "MODES", "DIAGNOSTIC", "INTERNAL", "now", "save", "digest",
               "sequence_info", "source_snapshot", "packet_size", "output_hashes",
               "incomplete", "application_ok", "positive_summary_ok", "clean_sanitizer_log",
               "summary_ok"):
    globals()[helper] = getattr(shared, helper)

CUSTOM_REGEX = r"_ZN7saccade|_ZN3cub"
# shared_race is a global test-only function, not in the deployed namespaces.
# Verified from tests/postprocess_sanitizer_control.cu and nm of its clean binary.
CONTROL_REGEX = CUSTOM_REGEX + r"|^_Z11shared_racePf$"


def success_status(_limit):
    return "PASS_FILTERED_FORCED_SYNC"


def aggregate_status(results, custody_unchanged, error, _limit, modes):
    statuses = {"PASS", "PASS_FILTERED_FORCED_SYNC"}
    complete = (not error and custody_unchanged and len(results) == 2 * modes + 1 and
                sum(row["status"] == "PASS_FILTERED_FORCED_SYNC" for row in results) == modes and
                all(row["status"] in statuses for row in results))
    return "PASS_FILTERED_FORCED_SYNC" if complete else "OPEN"


def reference_inventory():
    path = ROOT / "evidence/hr-sanitizer/2026-11-05-a00ab3a/inventory/counts.csv"
    with path.open() as source:
        rows = [row for row in csv.DictReader(source)
                if row["run"] == "00-synccheck-graph-pipelined"]
    selected = [row for row in rows if re.search(CUSTOM_REGEX, row["kernel"])]
    if not selected:
        raise RuntimeError("fixed namespace predicate matches no reference inventory kernels")
    return {"source": str(path), "sha256": digest(path),
            "all_names": len({row["kernel"] for row in rows}),
            "selected_names": sorted({row["kernel"] for row in selected}),
            "selected_launches": sum(int(row["count"]) for row in selected),
            "basis": "Earlier complete 30-frame Graph inventory; reference selection only, not a new per-capture launch coverage measurement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build/hr-sanitizer-20261105/cuda")
    parser.add_argument("--source", type=Path, default=ROOT / "build/hr-sanitizer-20261105-source")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.set_defaults(tools=["racecheck"])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--memory", default="8G", help="Both MemoryHigh and MemoryMax; swap remains disabled")
    parser.add_argument("--racecheck-num-workers", type=int, default=1,
                        help="Racecheck CPU workers; 0 selects the tool's automatic count")
    parser.add_argument("--force-synchronization-limit", type=int, default=8,
                        help="Positive stream launch bound; successful filtered captures are PASS_FILTERED_FORCED_SYNC, never native or full-kernel acceptance")
    parser.add_argument("--dump-kernel-launches", action="store_true",
                        help="Diagnostic inventory only: installed sanitizer counts launch records in its error summary; never accepted as PASS")
    parser.add_argument("--engine", type=Path, default=Path("/home/ray/.cache/saccade-v1/hr-deploy/model-hr-fp16.engine"))
    parser.add_argument("--sequence", type=Path, default=Path("/home/ray/developer/ai/saccade/datasets/MOT17/train/MOT17-10-SDP"))
    parser.add_argument("--sanitizer", type=Path, default=Path("/opt/cuda/bin/compute-sanitizer"))
    args = parser.parse_args()
    args.memory_high = args.memory_max = args.memory
    for key in ("build", "source", "output", "engine", "sequence", "sanitizer"):
        setattr(args, key, getattr(args, key).resolve())
    if args.frames < 1 or args.timeout < 1:
        parser.error("frames and timeout must be positive")
    if args.force_synchronization_limit < 1:
        parser.error("this scoped runner requires a positive force-synchronization-limit")
    if args.racecheck_num_workers < 0:
        parser.error("racecheck-num-workers must be nonnegative")
    if args.force_synchronization_limit and args.tools != ["racecheck"]:
        parser.error("nonzero force-synchronization-limit requires --tools racecheck only")
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
                "kernel_filter": CUSTOM_REGEX, "positive_control_filter": CONTROL_REGEX,
                "full_kernel_coverage": False, "launch_limit": None,
                "forced_synchronization": bool(args.force_synchronization_limit),
                "force_synchronization_limit": args.force_synchronization_limit,
                "racecheck_num_workers": args.racecheck_num_workers,
                "native_schedule_acceptance": False,
                "sanitizer_environment_overrides": {}, "environment": capture.environment,
                "dump_kernel_launches": args.dump_kernel_launches,
                "diagnostic_inventory": args.dump_kernel_launches,
                "print_limit": {t: 1000 if t == "initcheck" else 0 for t in args.tools},
                "removed_environment": capture.removed_environment,
                "runner_sha256": digest(__file__), "shared_runner_sha256": digest(ROOT_RUNNER),
                "sanitizer_sha256": digest(args.sanitizer),
                "target_wrapper_sha256": digest(capture.wrapper),
                "resource_limits": {"timeout_seconds": args.timeout, "memory_high": args.memory_high,
                                    "memory_max": args.memory_max, "memory_swap_max": 0},
                "scope": "SCOPED custom namespace and CUB racecheck only, with tool-forced stream synchronization. Full stated input prefix, baseline parity, and selected modes; external library kernels are excluded. Does not close unfiltered or native queued racecheck."}
    manifest["reference_scope_inventory"] = reference_inventory()
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
                    options += ["--racecheck-num-workers", str(args.racecheck_num_workers)]
                    if args.force_synchronization_limit:
                        options += ["--force-synchronization-limit", str(args.force_synchronization_limit)]
                if tool == "memcheck":
                    options += ["--report-api-errors", "explicit"]
                name = f"control-{tool}"
                record = capture.run(name, [*options, "--kernel-name", f"regex={CONTROL_REGEX}", args.build / "postprocess_sanitizer_control", tool])
                text = (args.output / f"{name}.log").read_text(errors="replace")
                okay = (record["returncode"] == 99 and record.get("target", {}).get("returncode") == 99 and
                        not incomplete(record, text) and
                        DIAGNOSTIC[tool] in text and f"control={tool} launch=" in text and
                        not INTERNAL.search(text.replace("Target application returned an error", "")) and
                        positive_summary_ok(tool, text))
                controls[tool] = okay
                results.append({"name": name, "status": "PASS" if okay else "FAIL", "execution": record,
                                "expected_diagnostic": DIAGNOSTIC[tool], "kernel_filter": CONTROL_REGEX})
                save(args.output / "results.json", results)
                for mode in args.modes:
                    name = f"{tool}-{mode}"
                    record = capture.run(name, [*options, "--kernel-name", f"regex={CUSTOM_REGEX}", *command(name, mode)])
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
                    status = (success_status(args.force_synchronization_limit) if okay else
                              "INCOMPLETE" if incomplete(record, text) else "OPEN")
                    results.append({"name": name, "status": status, "target_completed": complete,
                                    "forced_synchronization": bool(args.force_synchronization_limit),
                                    "kernel_filter": CUSTOM_REGEX, "full_kernel_coverage": False,
                                    "force_synchronization_limit": args.force_synchronization_limit,
                                    "native_schedule_acceptance": bool(okay and not args.force_synchronization_limit),
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
                manifest["shared_runner_unchanged"] = digest(ROOT_RUNNER) == manifest["shared_runner_sha256"]
                manifest["runner_unchanged"] = digest(__file__) == manifest["runner_sha256"]
                manifest["custody_unchanged"] = (manifest["custody_unchanged"] and manifest["sanitizer_unchanged"] and
                                                 manifest["target_wrapper_unchanged"] and manifest["shared_runner_unchanged"] and
                                                 manifest["runner_unchanged"])
            except Exception as error:
                manifest["custody_unchanged"] = False
                manifest["custody_error"] = str(error)
            manifest["finished"] = now()
            manifest["status"] = aggregate_status(results, manifest["custody_unchanged"],
                                                  manifest.get("error"), args.force_synchronization_limit, len(args.modes))
            manifest["native_schedule_acceptance"] = bool(manifest["status"] == "PASS" and
                                                           not args.force_synchronization_limit)
            save(args.output / "manifest.json", manifest)
    return 0 if manifest["status"] == "PASS_FILTERED_FORCED_SYNC" else 1


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
    try:
        sys.exit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        print(f"hr-custom-racecheck-capture: {error}", file=sys.stderr)
        sys.exit(2)

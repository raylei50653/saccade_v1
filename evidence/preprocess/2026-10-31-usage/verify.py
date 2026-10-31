#!/usr/bin/env python3
"""Offline, fail-closed comparison of every normal-use report row to fixed refs."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = Path(sys.argv[1]).resolve()
    here = Path(__file__).resolve().parent
    frozen = here.parents[2] / "evidence/preprocess/2026-10-31/refs/manifest.json"
    expected_manifest = "a3b08bdfe20f13f122e63a0314969832622daf8cedf0e29330e4b325471af5ff"
    manifest = json.loads(frozen.read_text())
    status = json.loads((out / "status.json").read_text())
    before = json.loads((out / "preflight.json").read_text())
    after = json.loads((out / "postflight.json").read_text())
    modules = json.loads((out / "loaded-modules.json").read_text())
    plan = json.loads((out / "PLAN.json").read_text())
    sequence = [0,4,1,4,2,4,3,4,5,4,6,4,0]
    configurations = {"serial": ("serial",1), "queued": ("queued",3), "dual-a": ("dual",3), "dual-b": ("dual",3)}
    checks = {}
    errors = []

    def require(label, value):
        checks[label] = bool(value)
        if not value:
            errors.append(label)

    def rows(name):
        path = out / "run" / name
        if not path.exists():
            errors.append("missing " + name)
            return []
        with path.open(newline="") as file:
            return list(csv.DictReader(file, delimiter="\t"))

    reports = rows("report.tsv")
    submissions = rows("submissions.tsv")
    summaries = rows("summary.tsv")
    require("frozen_manifest", sha(frozen) == expected_manifest and before["reference_manifest_sha256"] == expected_manifest)
    require("normal_process_completed", status["normal_process_exit"] == 0 and not status["timeout"])
    require("identities_stable", status["identity_stable"] and all(before[key] == after[key] for key in
            ["reference_hashes", "archive_hashes", "runtime_source_hashes", "dynamic_library_hashes", "usage_policy_sha256", "harness_sha256", "capture_script_sha256", "verifier_sha256"]))
    require("policy_snapshot", status["policy_sha256"] == before["usage_policy_sha256"] == sha(out / "PREPROCESS_USAGE_POLICY.md"))
    require("fixed_plan", plan["sequence"] == sequence and plan["cycles"] == 4 and plan["total_frames"] == 208
            and plan["dual_b_sequence"] == [(i+1)%7 for i in sequence] and plan["delay_clock64_cycles"] == 10000000)
    require("plan_snapshot", sha(out / "PLAN.json") == status["plan_sha256"])
    require("complete_report", len(reports) == 208 and {r["session"] for r in reports} == set(configurations))
    require("complete_submission_log", len(submissions) == 208 and {r["session"] for r in submissions} == set(configurations))
    require("four_checked_sessions", len(summaries) == 4 and {r["session"] for r in summaries} == set(configurations))
    frozen_cases = manifest["cases"][:7]
    total_rgb = 0
    total_f32 = 0
    for session, (phase, window) in configurations.items():
        expected = [(i+1)%7 if session == "dual-b" else i for i in sequence]*4
        own = [r for r in reports if r["session"] == session]
        submitted = [r for r in submissions if r["session"] == session]
        require(session+"_frame_order", [int(r["frame"]) for r in own] == list(range(52)))
        require(session+"_case_order", [int(r["case"]) for r in own] == expected)
        require(session+"_submission_order", [int(r["frame"]) for r in submitted] == list(range(52))
                and [int(r["case"]) for r in submitted] == expected)
        for row in own:
            case = frozen_cases[int(row["case"])]
            valid = row["phase"] == phase and int(row["window"]) == window
            valid = valid and row["label"] == case["label"] and int(row["width"]) == case["width"] and int(row["height"]) == case["height"]
            valid = valid and int(row["rgb_bytes"]) == 3*case["width"]*case["height"] and int(row["f32_elements"]) == 3*640*640
            valid = valid and all(row[key] == "1" for key in ["rgb_exact", "f32_bitwise", "finite", "repeat_exact", "cross_session_exact", "mapping"])
            valid = valid and row["rgb_sha256"] == row["reference_rgb_sha256"] == case["hashes"][case["rgb"]]
            valid = valid and row["f32_sha256"] == row["reference_f32_sha256"] == case["hashes"][case["f32"]]
            require(session+"_frame_"+row["frame"]+"_all_bytes_and_mapping", valid)
            total_rgb += int(row["rgb_bytes"])
            total_f32 += int(row["f32_elements"])
        query_not_ready = sum(row["prior_consumer_query"] == "600" for row in submitted)
        require(session+"_delay_and_pinned_ring_log", all(int(row["delay_cycles"]) == 10000000
                and row["caller_jpeg_overwritten"] == "1" and int(row["slot"]) == int(row["frame"]) % 3
                and 1 <= int(row["pending_after_submit"]) <= window for row in submitted))
        require(session+"_query_statuses", all(row["prior_consumer_query"] in ("-1", "0", "600")
                and (row["query_last_error"] in ("0", "600") if row["prior_consumer_query"] == "600" else row["query_last_error"] == "-1")
                for row in submitted))
        require(session+"_pending_consumer_observed", window == 1 or query_not_ready > 0)
        summary = next((r for r in summaries if r["session"] == session), None)
        require(session+"_summary", summary is not None and summary["phase"] == phase and int(summary["window"]) == window
                and summary["submitted"] == summary["verified"] == "52" and summary["checked_close"] == "1"
                and int(summary["prior_consumer_not_ready"]) == query_not_ready and int(summary["max_pending"]) == window)
        identity_path = out / "run" / (session+"-identity.txt")
        identity = identity_path.read_text() if identity_path.exists() else ""
        require(session+"_public_native_identity", "source_commit=2b39ddaf7cefcbdfb8a7fc143ecfd83466618cb3\n" in identity
                and "source_dirty=false\n" in identity and "backend=NVJPEG_BACKEND_HARDWARE\n" in identity
                and "output=NVJPEG_OUTPUT_RGB\n" in identity and "output_width=640\noutput_height=640\n" in identity)
    require("mapped_nvjpeg_identity", any("libnvjpeg" in path and record["sha256"] == "c8fb7a95fa58f20eeb58c21760ad4017845fd051c6a70672ff25736696404338"
            for path, record in modules.items()))
    require("delay_preloaded_and_result", "delay_preloaded=1 delay_cycles=10000000" in (out / "normal-usage.stdout.log").read_text()
            and "result=complete sessions=4 frames=208 rgb_exact=1 f32_bitwise=1" in (out / "normal-usage.stdout.log").read_text())
    result = {"pass": not errors, "checks": checks, "errors": errors, "frames_reported": len(reports),
              "rgb_bytes_compared": total_rgb, "f32_elements_compared": total_f32,
              "session_summaries": summaries, "scope": "seven fixed real images; 208 uses; historical source/libraries; finite normal usage only"}
    (out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["pass", "errors", "frames_reported", "rgb_bytes_compared", "f32_elements_compared", "session_summaries"]}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

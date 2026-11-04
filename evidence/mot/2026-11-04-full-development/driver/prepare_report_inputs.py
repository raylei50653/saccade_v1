"""Assemble report inputs without inference, scoring or ground-truth reads.

Run again after scoring to refresh current metrics and manifest comparisons:
python -I build/mot-full-20261104-driver/prepare_report_inputs.py
"""

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path.cwd()
DRIVER = ROOT / "build/mot-full-20261104-driver"
ACCEPTANCE = ROOT / "build/mot-full-20261104-acceptance/capture"
EVALUATION = ROOT / "build/mot-full-20261104-evaluation"
M4 = ROOT / "evidence/m4/2026-11-03-formal-cad5298/evaluation"
D21 = ROOT / "evidence/cuda-graph/2026-11-04-gate-d21a144/mot17-run"
SEQUENCES = Path("/home/ray/developer/ai/saccade/datasets/MOT17/train")
DEPLOY = Path("/home/ray/.cache/saccade-v1/hr-deploy")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def number(value):
    if not value:
        return None
    try:
        return int(value) if re.fullmatch(r"[+-]?\d+", value) else float(value)
    except ValueError:
        return value


def table(path):
    if not path.exists():
        return None
    with path.open() as stream:
        return [{key: number(value) for key, value in row.items()}
                for row in csv.DictReader(stream, delimiter="\t")]


def key_values(path):
    if not path.exists():
        return None
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def manifest(path):
    if not path.exists():
        return None
    return {name.lstrip("*"): sha for line in path.read_text().splitlines()
            for sha, name in [line.split(maxsplit=1)]}


def compare_manifests(current, expected):
    actual, reference = manifest(current), manifest(expected)
    if actual is None:
        return {"status": "PENDING", "current": str(current), "reference": str(expected)}
    matched = [name for name, sha in reference.items() if actual.get(name) == sha]
    changed = {name: {"expected": sha, "actual": actual.get(name)} for name, sha in reference.items()
               if actual.get(name) != sha}
    extra = {name: sha for name, sha in actual.items() if name not in reference}
    return {"status": "PASS" if not changed and not extra else "FAIL", "matched": len(matched),
            "expected": len(reference), "actual": len(actual), "changed": changed, "extra": extra,
            "current": str(current), "reference": str(expected)}


DETAIL_FIELDS = ["CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "Frag", "MT", "PT", "ML",
                 "CLR_Frames", "IDTP", "IDFP", "IDFN", "MOTA", "IDF1"]


def details(directory, current=False):
    groups = {}
    paths = directory.glob("*/trackeval/results/v1/pedestrian_detailed.csv") if current else directory.glob("*/pedestrian_detailed.csv")
    for path in sorted(paths):
        group = path.parents[3].name if current else path.parent.name
        with path.open() as stream:
            groups[group] = [{"sequence": row["seq"], **{field: number(row[field])
                              for field in DETAIL_FIELDS if field in row}} for row in csv.DictReader(stream)]
    return {"units": "MOTA and IDF1 are fractions in detailed CSV; metrics.tsv reports percent.", "groups": groups}


frames = []
for code in ("02", "04", "05", "09", "10", "11", "13"):
    sequence = f"MOT17-{code}-SDP"
    info = key_values(SEQUENCES / sequence / "seqinfo.ini")
    frames.append({"sequence": sequence, "group": "eval" if code == "02" else "val" if code == "10" else "in-domain",
                   "frames": int(info["seqLength"]), "width": int(info["imWidth"]), "height": int(info["imHeight"]),
                   "dataset_fps": float(info["frameRate"]), "images_present": len(list((SEQUENCES / sequence / "img1").glob("*.jpg")))})

baseline_metrics = table(M4 / "metrics.tsv")
by_sequence = {row["row"]: row for row in baseline_metrics if str(row["row"]).startswith("MOT17-")}
by_sequence["MOT17-10-SDP"] = next(row for row in baseline_metrics if row["row"] == "val")
by_sequence["MOT17-02-SDP"] = next(row for row in baseline_metrics if row["row"] == "eval")
deploy_manifest = json.loads((DEPLOY / "manifest.json").read_text())
build_report = {}
for line in (ROOT / "evidence/precision/2026-11-03-formal/build-report.txt").read_text().splitlines():
    if line.startswith("layers_json="):
        break
    if "=" in line:
        key, value = line.split("=", 1)
        if key in {"onnx_sha256", "bundle_sha256", "engine_bytes", "engine_sha256", "precision", "fp16", "tf32",
                   "aux_streams", "profile", "plugins", "build_commit", "build_dirty", "tensorrt_headers",
                   "tensorrt_library", "nvcc", "scan_cuda_flags", "workspace_bytes", "optimization_level"}:
            build_report[key] = value
engine = DEPLOY / "model-hr-fp16.engine"
manifest_sources = {key: {"path": deploy_manifest[key], "declared_sha256": deploy_manifest[key + "_sha256"]}
                    for key in ("student", "teacher", "base")}
source_file = ACCEPTANCE / "source-files.sha256"
source_status = ACCEPTANCE / "source-status.txt"
report = {
    "generated_from": "Existing result files, source, manifests and seqinfo/image membership only; no GT contents, scoring or GPU launched.",
    "frames": {"total": sum(item["frames"] for item in frames), "sequences": frames},
    "m4_baseline": {"revision": "cad52981c7ecf40e13c9c016f351f6137b2b61f3", "metrics_path": str(M4 / "metrics.tsv"),
                    "rows": baseline_metrics, "by_sequence": by_sequence,
                    "detailed": details(M4 / "trackeval"),
                    "quality_comparison_policy": "Current rerun is post-M4; MOT17-02 is no longer a newly opened independent evaluation."},
    "d21_timing_baseline": {"identity": key_values(D21 / "source.txt"), "rows": table(D21 / "timing.tsv"),
                            "scope": "One unprofiled run per sequence, separate historical capture; not ABBA, clock/thermal uncontrolled. Quality not rescored; 21 outputs matched M4."},
    "timing_boundaries": {"source": "tools/track_sequence.cpp:74-94", "start": "After TrackingSession construction, before first JPEG file read",
                          "end": "After session.finish() and all pending ResultPackets received",
                          "includes": ["JPEG file reads", "host submission", "pipeline processing", "SequenceOutput.add", "pending packet drain"],
                          "excludes": ["engine/session construction", "session.close", "write_mot_file", "optional binary packet file writes"],
                          "interpretation": "Pipelined ms_per_frame is sequence wall time divided by full frame count; reciprocal FPS is processing throughput, not per-frame delivery latency.",
                          "baseline_report_erratum": "d21 gate report says MOT writing included; executable source places timer end before write_mot_file."},
    "acceptance_protocol": {
        "entrypoint": "scripts/run_tracking_session_acceptance.sh build/mot-full-20261104-acceptance --allow-dirty",
        "expected_clean_sanitizer_runs": 17,
        "kernel_suites": {"tests": ["tracking_packet_tests", "tracking_cuda_tests", "motion_cuda_tests"],
                          "tools": ["memcheck", "initcheck", "racecheck", "synccheck"], "runs": 12,
                          "initcheck_exceptions": ["--check-api-memory-access no: fixed capacity output tails undefined by contract", "motion_cuda_tests --synthetic: excludes two nvJPEG frames per PREPROCESS_USAGE_POLICY"]},
        "full_session_memcheck": {"runs": 3, "report_api_errors": "explicit", "cases": ["T01 tracking_session_tests MOT17-10 --limit 30", "T01 tracking_session_tests --lifecycle", "HR FP16 track_sequence MOT17-10 prefix 30 frames"],
                                  "scope": "All kernels in these bounded runs, including nvJPEG, TensorRT and cuFFT",
                                  "explicit_reason": "Avoid counting known driver code-recompilation note from TensorRT module loading as an extended API error; memory errors remain enabled."},
        "session_racecheck_synccheck": {"runs": 2, "frames": 10, "engine": "T01", "kernel_filter": "_tracking_session_cu_|_tracking_cu_|_motion_cu_",
                                       "excluded_kernels": ["TensorRT internals", "nvJPEG", "cuFFT", "F02 resize", "scan plugins", "other kernels outside explicit filter"]},
        "positive_controls": {"runs": 4, "expected_exit": 99, "expected_diagnostics": {"memcheck": "Invalid __global__ write", "initcheck": "Uninitialized __global__ memory read", "racecheck": "Error: Race reported between", "synccheck": "Barrier error detected"}},
        "not_claimed": ["full-session initcheck", "TensorRT internals racecheck or synccheck", "all-frame HR sanitizer coverage", "clean-source formal acceptance", "new semantic mutation suite"]},
    "identity": {"acceptance_source": key_values(ACCEPTANCE / "source.txt"),
                 "acceptance_source_file_count": len(source_file.read_text().splitlines()) if source_file.exists() else None,
                 "acceptance_status_line_count": len(source_status.read_text().splitlines()) if source_status.exists() else None,
                 "acceptance_source_diff_sha256": digest(ACCEPTANCE / "source.diff") if (ACCEPTANCE / "source.diff").exists() else None,
                 "evaluation_source": key_values(EVALUATION / "capture/source.txt"),
                 "engine_path": str(engine), "engine_bytes": engine.stat().st_size, "engine_sha256": digest(engine),
                 "deploy_manifest_sha256": digest(DEPLOY / "manifest.json"), "manifest_declared_training_sources": manifest_sources,
                 "deploy_manifest_artifact_count": len(deploy_manifest["artifacts"]), "engine_build_report": build_report,
                 "runtime_model_profile": "hr-1280x768", "runtime_model_io": "FP32 model input/raw outputs, FP32 scan boundaries; FP16 backbone/head tactics",
                 "ctest_reference_engine_profile": "T01 640x640 for full seven-sequence serial/graph CPU-reference packet comparison"},
    "stream_gate_patch": {"path": "scripts/check_stream_placement.py:36", "change": "Add exact tiled_hr kernel name alongside horizontal and vertical to F02 / D rule",
                          "validation_path": str(DRIVER / "stream-checker-tests/results.json"), "validation_cases": 14,
                          "fixture_scope": "Legacy real sqlite replay plus explicitly controlled fused-name edits; actual fused GPU trace belongs to new evaluation gate."},
    "current": {"output_manifest_comparison_to_m4": compare_manifests(EVALUATION / "capture/outputs.sha256", M4 / "outputs.sha256"),
                "metrics": table(EVALUATION / "capture/metrics.tsv"), "timing": table(EVALUATION / "capture/timing.tsv"),
                "detailed": details(EVALUATION / "score", current=True),
                "gates": (EVALUATION / "capture/gates.txt").read_text().strip() if (EVALUATION / "capture/gates.txt").exists() else None,
                "scoring": (EVALUATION / "capture/scoring.txt").read_text().strip() if (EVALUATION / "capture/scoring.txt").exists() else None},
}
(DRIVER / "report-inputs.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"output": str(DRIVER / "report-inputs.json"), "frames": report["frames"]["total"],
                  "m4_rows": len(baseline_metrics), "comparison_status": report["current"]["output_manifest_comparison_to_m4"]["status"]}))

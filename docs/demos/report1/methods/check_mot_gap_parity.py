#!/usr/bin/env python3
"""Independently check display gap filling against F09 and captured MOT text.

Read-only inputs; writes only the requested JSON receipt. No inference, GT,
TrackEval, model access, or quality rescoring. Python reproduces the formal
v1-mot/1 rule independently, then checks the JavaScript module and every row of
the already captured, three-decimal MOT outputs.
"""
from __future__ import annotations

import argparse
import collections
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
DEMO = Path(__file__).resolve().parents[1]
ASSETS = DEMO / "assets"
MODULE = DEMO / "gap-fill.js"
DEFAULT_OUTPUT = ROOT / "evidence/report1/2026-11-11-demo-mot-interpolation/parity.json"
CASES = [
    ("MOT17-10-SDP", "data.json", "2026-11-11-g1-offline-e2e", 654, 10385, 934, 11319, 98),
    ("MOT17-02-SDP", "data-02.json", "2026-11-11-mot17-02-demo", 600, 11454, 2040, 13494, 149),
    ("MOT17-04-SDP", "data-04.json", "2026-11-11-mot17-04-demo", 1050, 43003, 2053, 45056, 123),
]
NODE_SCRIPT = r"""
const fs = require('node:fs');
const modulePath = process.argv[1];
const api = require(modulePath);
const result = [];
for (const asset of process.argv.slice(2)) {
  const data = JSON.parse(fs.readFileSync(asset, 'utf8'));
  const before = JSON.stringify(data.frames);
  const index = api.build(data.frames, {maxMissingFrames:35, minObservations:5});
  const rows = [];
  for (let frame=0; frame<data.frames.length; frame++) {
    for (const row of api.sample(index, frame, 0)) rows.push({frame, ...row});
  }
  result.push({sequence:data.meta.sequence, stats:index.stats,
    motIdByOutputId:index.motIdByOutputId, rows,
    sourceFramesUnchanged:before === JSON.stringify(data.frames)});
}
process.stdout.write(JSON.stringify(result));
"""


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def formal_rows(frames: list[dict]) -> tuple[dict, dict, dict]:
    """Literal independent F09 reproduction: group output ID, ignore UID."""
    mapping: dict[int, int] = {}
    by_id: dict[int, list[dict]] = collections.defaultdict(list)
    observed: dict[tuple[int, int], dict] = {}
    for index, frame in enumerate(frames):
        assert frame["i"] == index
        for row in frame["rows"]:
            output_id, uid = row[:2]
            # These captured demo inputs are all observed, but F09 itself has
            # no flag or UID gate when it consumes its input packet rows.
            assert row[8] == 1
            if output_id not in mapping:
                mapping[output_id] = len(mapping) + 1
            entry = {"frame": index, "outputId": output_id, "motId": mapping[output_id],
                     "uid": uid, "xyxy": row[2:6], "confidence": row[6]}
            key = (index, entry["motId"])
            assert key not in observed
            observed[key] = entry
            by_id[output_id].append(entry)
    filled: dict[tuple[int, int], dict] = {}
    for output_id, rows in by_id.items():
        if len(rows) < 5:
            continue
        for left, right in zip(rows, rows[1:]):
            distance = right["frame"] - left["frame"]
            if distance <= 1 or distance > 36:
                continue
            for frame in range(left["frame"] + 1, right["frame"]):
                t = (frame - left["frame"]) / distance
                entry = {"frame": frame, "outputId": output_id, "motId": mapping[output_id],
                         "fromUid": left["uid"], "toUid": right["uid"],
                         "fromFrame": left["frame"], "toFrame": right["frame"],
                         "missingFrames": distance - 1, "fraction": t,
                         "xyxy": [a + t * (b - a) for a, b in zip(left["xyxy"], right["xyxy"])],
                         "confidence": -1, "method": "v1-mot/1"}
                key = (frame, entry["motId"])
                assert key not in filled and key not in observed
                filled[key] = entry
    return mapping, observed, filled


def verify_case(case: tuple, actual_module: dict) -> dict:
    sequence, asset_name, receipt_name, frame_count, observed_count, fill_count, total_count, uid_cross = case
    asset = ASSETS / asset_name
    capture = ROOT / "evidence/report1" / receipt_name / "raw/run1"
    mot = capture / "mot.txt"
    config = capture / "resolved.config"
    config_lines = set(config.read_text().splitlines())
    assert "output.interpolate_max_gap=35" in config_lines
    assert "output.interpolate_min_rows=5" in config_lines
    data = json.loads(asset.read_text())
    assert data["meta"]["sequence"] == sequence and len(data["frames"]) == frame_count
    mapping, observed, expected = formal_rows(data["frames"])
    assert len(observed) == observed_count and len(expected) == fill_count
    assert actual_module["sequence"] == sequence and actual_module["sourceFramesUnchanged"]
    assert actual_module["motIdByOutputId"] == {str(key): value for key, value in mapping.items()}
    module_rows = {(r["frame"], r["motId"]): r for r in actual_module["rows"]}
    assert len(module_rows) == len(actual_module["rows"]) == fill_count
    assert set(module_rows) == set(expected)
    max_module_error = 0.0
    for key, row in module_rows.items():
        reference = expected[key]
        for field in ["outputId", "motId", "fromUid", "toUid", "fromFrame", "toFrame", "missingFrames", "confidence", "method"]:
            assert row[field] == reference[field], (sequence, key, field)
        assert abs(row["fraction"] - reference["fraction"]) <= 1e-15
        assert len(row["xyxy"]) == 4
        error = max(abs(a - b) for a, b in zip(row["xyxy"], reference["xyxy"]))
        assert error <= 1e-12, (sequence, key, error)
        max_module_error = max(max_module_error, error)
    with mot.open(newline="") as handle:
        text_rows = list(csv.reader(handle))
    assert len(text_rows) == total_count == observed_count + fill_count
    assert len({(int(row[0]) - 1, int(row[1])) for row in text_rows}) == total_count
    all_expected = {**observed, **expected}
    assert {(int(row[0]) - 1, int(row[1])) for row in text_rows} == set(all_expected)
    max_mot_error = 0.0
    actual_interpolated = 0
    for text_row in text_rows:
        key = (int(text_row[0]) - 1, int(text_row[1]))
        reference = all_expected[key]
        x1, y1, x2, y2 = reference["xyxy"]
        expected_fields = [x1, y1, x2 - x1, y2 - y1, reference["confidence"]]
        assert len(text_row) == 10 and text_row[7:] == ["-1", "-1", "-1"]
        for field, value in zip(text_row[2:7], expected_fields):
            error = abs(float(field) - value)
            assert error <= 0.0005000001, (sequence, key, field, value, error)
            max_mot_error = max(max_mot_error, error)
        is_fill = key in expected
        assert (text_row[6] == "-1") == is_fill
        actual_interpolated += is_fill
    assert actual_interpolated == fill_count
    crossing = [row for row in expected.values() if row["fromUid"] != row["toUid"]]
    assert len(crossing) == uid_cross
    witness_id, witness_frame = {"MOT17-10-SDP": (71, 90), "MOT17-02-SDP": (12, 220), "MOT17-04-SDP": (20, 240)}[sequence]
    witness = next(row for row in crossing if row["outputId"] == witness_id and row["frame"] == witness_frame)
    return {"sequence": sequence, "status": "PASS", "frames": frame_count,
            "observed_rows": observed_count, "interpolated_rows": fill_count,
            "final_mot_rows": total_count, "uid_crossing_interpolated_rows": uid_cross,
            "module_full_precision_xyxy_max_abs_error": max_module_error,
            "mot_three_decimal_xywh_conf_max_abs_error": max_mot_error,
            "module_stats": actual_module["stats"], "uid_crossing_witness": witness,
            "inputs": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha(p)} for p in [asset, mot, config]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    before = {p.name: sha(p) for p in sorted(ASSETS.iterdir()) if p.is_file()}
    version = subprocess.check_output(["node", "--version"], text=True).strip()
    command = ["node", "-e", NODE_SCRIPT, str(MODULE), *[str(ASSETS / case[1]) for case in CASES]]
    result = json.loads(subprocess.check_output(command, text=True))
    assert len(result) == len(CASES)
    rows = [verify_case(case, module) for case, module in zip(CASES, result)]
    after = {p.name: sha(p) for p in sorted(ASSETS.iterdir()) if p.is_file()}
    assert before == after
    receipt = {"schema": "saccade-demo-formal-mot-gap-parity/1", "status": "PASS",
               "created_utc": datetime.now(timezone.utc).isoformat(), "node": version,
               "scope": "display gap module vs independently reproduced F09 and existing captured MOT; no inference/GT/scoring",
               "policy": {"profile": "v1-mot/1", "grouping": "output_id only; UID changes permitted",
                          "minimum_observed_rows": 5, "maximum_missing_frames": 35,
                          "confidence": -1, "first_appearance_mot_id_mapping": True,
                          "extrapolation": False},
               "comparison": {"full_precision_module_xyxy_absolute_tolerance": 1e-12,
                              "captured_mot_xywh_conf_absolute_tolerance": 0.0005000001,
                              "mot_precision_decimal_places": 3,
                              "byte_identity_claim": False},
               "cases": rows, "source_assets_unchanged": before,
               "module": {"path": str(MODULE.relative_to(ROOT)), "sha256": sha(MODULE)},
               "checker": {"path": str(Path(__file__).resolve().relative_to(ROOT)), "sha256": sha(Path(__file__))}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print("PASS: 3 sequences; 5027 interpolated rows; 69869 final MOT rows; output-ID mapping and all XYXY match independent F09; captured 3-decimal MOT within rounding bound; source assets unchanged")


if __name__ == "__main__":
    main()

"""Validate all seven unique MOT17 captures and explicit two-sequence imports."""
from pathlib import Path
import hashlib
import json
import struct

ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = {"MOT17-02-SDP": 600, "MOT17-04-SDP": 1050, "MOT17-05-SDP": 837,
             "MOT17-09-SDP": 525, "MOT17-10-SDP": 654, "MOT17-11-SDP": 900, "MOT17-13-SDP": 750}
ARMS = ("official", "finetuned")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cells = []
    sources_new = None
    per_arm_model_state = {}
    inputs_by_seq = {}
    native_source_by_name = {}
    for arm in ARMS:
        for seq, count in SEQUENCES.items():
            directory = ROOT / arm / seq
            receipt = directory / "inference.json"
            x = json.loads(receipt.read_text())
            assert x["status"] == "PASS" and x["frames"] == count and x["raw_nonfinite"] == 0
            assert x["weights_state_before"] == x["weights_state_after"]
            if arm in per_arm_model_state:
                assert per_arm_model_state[arm] == x["weights_state_before"]
            per_arm_model_state[arm] = x["weights_state_before"]
            if seq in inputs_by_seq:
                assert inputs_by_seq[seq] == x["input_hashes"]
            inputs_by_seq[seq] = x["input_hashes"]
            imported = seq in ("MOT17-02-SDP", "MOT17-10-SDP")
            if imported:
                imp = json.loads((directory / "import.json").read_text())
                assert imp["status"] == "PASS" and imp["reused"] and not imp["new_inference"]
                assert imp["original_receipt_paths_preserved"]
                for filename in ("inference.json", "detections.scdr", "progress.json"):
                    entry = imp["files"][filename]
                    assert sha(entry["source"]) == sha(directory / filename) == entry["sha256"]
            else:
                if sources_new is None:
                    sources_new = x["source_hashes"]
                assert sources_new == x["source_hashes"]
                for path, expected in x["source_hashes"].items():
                    assert sha(path) == expected
            for path, expected in x["source_hashes"].items():
                name = Path(path).name
                if name == "export.py":
                    continue  # Only ROOT and CLI choices differ from imported captures.
                if name in native_source_by_name:
                    assert native_source_by_name[name] == expected
                native_source_by_name[name] = expected
            stream = directory / "detections.scdr"
            assert sha(stream) == x["scdr_sha256"] and stream.stat().st_size == x["bytes"] == 28 + 9664 * count
            assert struct.unpack("<4sIIQQ", stream.read_bytes()[:28]) == (b"SCDR", 1, 9664, 1, count)
            assert x["counts"]["kept"] + x["counts"]["suppressed"] == x["counts"]["count"]
            cells.append({"arm": arm, "sequence": seq, "frames": count, "reused": imported,
                          "new_inference": not imported, "inference_receipt": str(receipt),
                          "inference_receipt_sha256": sha(receipt), "scdr": str(stream),
                          "scdr_sha256": x["scdr_sha256"], "counts": x["counts"],
                          "weights": x["weights"], "native_f04_policy": x["policy"]})
    result = {"status": "PASS", "all_fourteen_complete": True, "each_arm_frames": sum(SEQUENCES.values()),
              "new_capture_cells": 10, "reused_capture_cells": 4, "new_capture_frames_total": 8124,
              "reused_capture_frames_total": 2508, "raw_nonfinite_total": 0,
              "new_source_hashes_same_across_ten_runs_and_current": True,
              "native_wrapper_library_contract_hashes_same_in_new_and_reused": True,
              "input_bytes_same_between_arms_for_each_sequence": True,
              "model_state_same_before_after_and_across_sequences_per_arm": True,
              "source_hashes_new_captures": sources_new, "cells": cells,
              "gpu_lock": "/home/ray/.cache/saccade-v1/gpu.lock", "gt_read_in_inference": False,
              "product_speed_measurement": False,
              "runtime_scope": "local fixed-model comparison; no deployment runtime/config change or formal acceptance gate"}
    (ROOT / "detector-tools/delivery.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "all_fourteen_complete", "each_arm_frames",
                                            "new_capture_cells", "reused_capture_cells")}))


if __name__ == "__main__":
    main()

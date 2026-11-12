"""Frozen stock / UA2 teacher FP32 inference -> native scalar F04 SCDR.

No GT, training, tuning, tracker or scoring is performed here. MOT17-02 is
explicitly included only for this newly authorized comparison.
"""
from __future__ import annotations

import argparse
import configparser
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
MAIN = Path("/home/ray/developer/ai/saccade_v1")
DEV = Path("/home/ray/developer/ai/saccade_v1_dev")
SOURCE = Path("/home/ray/.cache/saccade-v1/t01/src")
ROOT = MAIN / "build/yolo26m-threeway-all7-20261111"
HERE = Path(__file__).resolve().parent
BASE = Path("/home/ray/developer/ai/dataset/yolo26m.pt")
TEACHER = DEV / "build-hr/HR01-UA2-20261102/teacher/attempt-001/epoch_0004.ckpt"
EXPECTED_BASE = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
EXPECTED_TEACHER = "08be5abae55d26b762e262948ded7d3deb11295aa0509ee04ad3582a2c9774cd"
FILE_HEADER = struct.Struct("<4sIIQQ")
FRAME_HEADER = struct.Struct("<QQ12I")
FRAME_BYTES = 9664
EPOCH = 1


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def state_sha(model):
    import torch
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_library():
    lib = ctypes.CDLL(str(HERE / "libf04.so"))
    f = lib.f04_convert
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
                  ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_void_p]
    f.restype = ctypes.c_int
    return f


def convert(f, cls, reg, width, height, frame):
    assert cls.dtype.name == reg.dtype.name == "float32"
    assert cls.flags.c_contiguous and reg.flags.c_contiguous
    assert cls.shape == (80, 20160) and reg.shape == (4, 20160)
    buffer = ctypes.create_string_buffer(FRAME_BYTES)
    rc = f(cls.ctypes.data, reg.ctypes.data, width, height, EPOCH, frame, buffer)
    if rc:
        raise RuntimeError(f"native F04 exception on frame {frame}")
    header = FRAME_HEADER.unpack_from(buffer)
    if header[0:3] != (EPOCH, frame, 1) or header[4:7] != (300, 0, 0x34504F46):
        raise RuntimeError(f"native F04 failure {header}")
    if header[3] != header[7] + header[8] or header[3] > 300:
        raise RuntimeError(f"native F04 invalid counts {header}")
    return buffer.raw, header


def cpu_smoke():
    import numpy as np
    f = load_library()
    cls = np.full((80, 20160), -100, dtype=np.float32)
    reg = np.ones((4, 20160), dtype=np.float32)
    cls[0, 0] = 2
    cls[0, 1] = 2
    reg[:, :2] = 2
    raw, h = convert(f, cls, reg, 1920, 1080, 0)
    rows = struct.Struct("<5f3I")
    a, b = rows.unpack_from(raw, 64), rows.unpack_from(raw, 96)
    assert h[3] == 2 and h[7:9] == (1, 1) and a[6:] == (0, 1) and b[6:] == (1, 2), (h, a, b)
    print(json.dumps({"cpu_smoke": "PASS", "header": h, "rows": [a, b]}), flush=True)


def export(arm, seq, data_root, batch_size):
    import numpy as np
    import torch
    import torchvision
    import hr_contract
    import hr_teacher_forward
    import t01_eval
    from saccade.perception.temporal_yolo.yolo_gated_detector import GatedDetConfig, build_gated_yolo_detector

    out = ROOT / arm / seq
    out.mkdir(parents=True, exist_ok=True)
    target = out / "detections.scdr"
    receipt = out / "inference.json"
    if target.exists() or receipt.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    started = time.monotonic()
    started_utc = now()
    info = configparser.ConfigParser()
    info.read(data_root / "train" / seq / "seqinfo.ini")
    cfg = info["Sequence"]
    width, height, count = int(cfg["imWidth"]), int(cfg["imHeight"]), int(cfg["seqLength"])
    paths = [data_root / "train" / seq / cfg["imDir"] / f"{i:06d}{cfg['imExt']}" for i in range(1, count + 1)]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError("incomplete JPEG sequence")
    inputs = {str(p): sha(p) for p in [data_root / "train" / seq / "seqinfo.ini", *paths]}
    assert sha(BASE) == EXPECTED_BASE
    assert sha(TEACHER) == EXPECTED_TEACHER
    precision_start = t01_eval.strict_fp32()
    model = build_gated_yolo_detector(str(BASE), GatedDetConfig(freeze_backbone=True), torch.device("cuda"))
    if arm == "finetuned":
        payload = torch.load(TEACHER, map_location="cpu", weights_only=False)
        state = {k.replace("._orig_mod.", "."): v for k, v in payload["model"].items()}
        model.load_state_dict(state, strict=True)
        del payload, state
    model.eval().requires_grad_(False)
    before = state_sha(model)
    f = load_library()
    aggregate = {k: 0 for k in ("count", "kept", "suppressed", "low_score", "topk_dropped", "class_dropped", "degenerate")}
    stats_fields = {"count": 3, "kept": 7, "suppressed": 8, "low_score": 9,
                    "topk_dropped": 10, "class_dropped": 11, "degenerate": 12}
    raw_nonfinite = 0
    first_decode = None
    temp = out / "detections.scdr.tmp"
    with temp.open("xb") as stream, torch.no_grad():
        stream.write(FILE_HEADER.pack(b"SCDR", 1, FRAME_BYTES, EPOCH, count))
        for start in range(0, count, batch_size):
            chunk = paths[start:start + batch_size]
            images = hr_contract.decode_stretch([[p.read_bytes()] for p in chunk], (1280, 768), "cuda")[:, 0]
            many = hr_teacher_forward.evaluation_predictions(model, images)
            cls_gpu, reg_gpu = many["scores"], many["boxes"]
            bad = int((~torch.isfinite(cls_gpu)).sum() + (~torch.isfinite(reg_gpu)).sum())
            raw_nonfinite += bad
            if bad:
                raise RuntimeError(f"nonfinite raw outputs at {start}")
            cls = cls_gpu.float().cpu().contiguous().numpy()
            reg = reg_gpu.float().cpu().contiguous().numpy()
            if start == 0:
                canonical = hr_contract.decode(cls_gpu[:1], reg_gpu[:1], (1280, 768))[0]
                first_decode = {"canonical_kept": len(canonical["scores"]), "canonical_topk": canonical["n_topk"],
                                "canonical_person": canonical["n_person"]}
            for j in range(len(chunk)):
                raw, header = convert(f, cls[j], reg[j], width, height, start + j)
                rows = np.frombuffer(raw, dtype=np.dtype([("box", "<f4", 4), ("score", "<f4"),
                                       ("class", "<u4"), ("candidate", "<u4"), ("flags", "<u4")]),
                                     count=header[3], offset=64)
                if not np.isfinite(rows["box"]).all() or not np.isfinite(rows["score"]).all():
                    raise RuntimeError("nonfinite native rows")
                if len(set(rows["candidate"].tolist())) != len(rows) or not np.isin(rows["flags"], [1, 2]).all():
                    raise RuntimeError("duplicate identities or invalid flags")
                if start == 0 and j == 0:
                    first_decode["native_kept"] = header[7]
                    first_decode["native_person"] = header[3]
                    first_decode["kept_count_equal"] = header[7] == first_decode["canonical_kept"]
                for key, field in stats_fields.items():
                    aggregate[key] += header[field]
                stream.write(raw)
            completed = start + len(chunk)
            if completed % 64 < batch_size or completed == count:
                progress = {"at": now(), "arm": arm, "sequence": seq, "frames": completed, "total": count,
                            "elapsed_seconds": time.monotonic() - started, "counts": aggregate}
                write_json(out / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            del many, cls_gpu, reg_gpu, images, cls, reg
    torch.cuda.synchronize()
    after = state_sha(model)
    if before != after:
        raise RuntimeError("model state changed during inference")
    precision_end = t01_eval.strict_fp32()
    assert temp.stat().st_size == FILE_HEADER.size + count * FRAME_BYTES
    temp.rename(target)
    sources = [HERE / "export.py", HERE / "f04_wrapper.cpp", HERE / "libf04.so",
               MAIN / "src/postprocess.cpp", MAIN / "include/saccade/postprocess.hpp",
               MAIN / "include/saccade/model_profile.hpp", DEV / "training/hr/hr_contract.py",
               DEV / "training/hr/hr_teacher_forward.py", DEV / "training/t01/t01_eval.py",
               DEV / "training/t01/t01_precision.py",
               SOURCE / "src/saccade/perception/temporal_yolo/yolo_gated_detector.py"]
    if any(sha(p) != expected for p, expected in inputs.items()):
        raise RuntimeError("input bytes changed during inference")
    result = {"status": "PASS", "arm": arm, "sequence": seq, "started_utc": started_utc, "completed_utc": now(),
              "frames": count, "width": width, "height": height, "epoch": EPOCH,
              "batch_size": batch_size, "precision_start": precision_start, "precision_end": precision_end,
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "source_hashes": {str(p): sha(p) for p in sources}, "input_hashes": inputs,
              "weights": {str(BASE): sha(BASE), **({str(TEACHER): sha(TEACHER)} if arm == "finetuned" else {})},
              "weights_state_before": before, "weights_state_after": after, "raw_nonfinite": raw_nonfinite,
              "counts": aggregate, "first_frame_canonical_vs_native": first_decode,
              "scdr": str(target), "scdr_sha256": sha(target), "bytes": target.stat().st_size,
              "policy": "hr-input/1 strict FP32, ungated one2many; native scalar F04 full kept+suppressed; original anchor IDs",
              "gt_read": False, "new_mot17_02_comparison_authorized": seq == "MOT17-02-SDP",
              "wall_seconds": time.monotonic() - started, "product_latency_measurement": False}
    write_json(receipt, result)
    print(json.dumps({"status": "PASS", "arm": arm, "sequence": seq, "frames": count,
                      "scdr_sha256": result["scdr_sha256"], "wall_seconds": result["wall_seconds"]}), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["official", "finetuned"])
    parser.add_argument("--sequence", choices=["MOT17-02-SDP", "MOT17-04-SDP", "MOT17-05-SDP", "MOT17-09-SDP", "MOT17-10-SDP", "MOT17-11-SDP", "MOT17-13-SDP"])
    parser.add_argument("--data-root", type=Path, default=Path("/home/ray/developer/ai/dataset/MOT17"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke:
        cpu_smoke()
        return
    if args.arm is None or args.sequence is None:
        parser.error("--arm and --sequence required")
    sys.path[:0] = [str(DEV / "training/hr"), str(DEV / "training/t01"), str(SOURCE / "src"), str(SOURCE)]
    from t01_exec import _BlockOldExtensions
    sys.meta_path.insert(0, _BlockOldExtensions())
    import torch
    torch.set_num_threads(8)
    export(args.arm, args.sequence, args.data_root, args.batch_size)


if __name__ == "__main__":
    main()

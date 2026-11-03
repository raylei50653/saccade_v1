"""Export the fixed HR Full25 e10 deployment artifacts and numerical references."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["SACCADE_T01_GPU_DECODE_ONLY"] = "1"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESEARCH = Path("/home/ray/developer/ai/saccade_v1_dev")
T01 = Path.home() / ".cache/saccade-v1/t01"
OUT = Path.home() / ".cache/saccade-v1/hr-deploy"
SOURCE = T01 / "src"
DATA = Path("/home/ray/developer/ai/saccade/datasets/MOT17")
BASE = Path("/home/ray/developer/ai/saccade/models/yolo/yolo26m.pt")
TEACHER = RESEARCH / "build-hr/HR01-UA2-20261102/teacher/attempt-001/epoch_0004.ckpt"
STUDENT = RESEARCH / "build-hr-mixed/HR01-MS-FULL25-20261103/mixed/attempt-001/epoch_0010.ckpt"
CACHE = RESEARCH / "build-hr/HR01-UA2D-20261102"
TEACHER_SHA = "08be5abae55d26b762e262948ded7d3deb11295aa0509ee04ad3582a2c9774cd"
STUDENT_SHA = "53bccffe9bdbf37bdcfc81d03e95ccbd2d142d67a557ea015b76d356e4fa5f7d"
BASE_SHA = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
SIZE = (1280, 768)
TRAIN_FRAMES = (("MOT17-04-SDP", 1), ("MOT17-05-SDP", 1), ("MOT17-09-SDP", 1), ("MOT17-13-SDP", 400))
REF_FRAMES = tuple(("MOT17-10-SDP", fid) for fid in (1, 218, 437, 654))
SCALES = ("p3", "p4", "p5")
FEATURE_SHAPES = ((1, 256, 96, 160), (1, 512, 48, 80), (1, 512, 24, 40))
OUTPUT_NAMES = tuple(f"{kind}_{scale}" for kind in ("cls", "reg") for scale in SCALES)
OUTPUT_SHAPES = tuple((1, channels, h, w) for channels in (80, 4) for h, w in ((96, 160), (48, 80), (24, 40)))

sys.path[:0] = [str(REPO / "training/t01"), str(RESEARCH / "training/hr")]
from t01_exec import _BlockOldExtensions
sys.meta_path.insert(0, _BlockOldExtensions())
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE)]

import numpy as np
import torch

import hr_contract
import t01_eval
import t01_precision
from t01_export import Backbone, strict_fp32


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checked_load(path: Path, expected: str):
    if sha(path) != expected:
        raise ValueError(f"checkpoint SHA-256 differs: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def backbone(state_key: str = "model"):
    from ultralytics import YOLO
    if sha(BASE) != BASE_SHA:
        raise ValueError("base YOLO SHA-256 differs")
    state = checked_load(TEACHER, TEACHER_SHA)
    weights = {key.removeprefix("yolo_model."): value for key, value in state[state_key].items()
               if key.startswith("yolo_model.")}
    model = YOLO(str(BASE)).model.float().eval()
    model.load_state_dict(weights, strict=True)
    return Backbone(model).eval(), weights


def frame_path(seq: str, fid: int) -> Path:
    return DATA / "train" / seq / "img1" / f"{fid:06d}.jpg"


def image(seq: str, fid: int):
    return hr_contract.decode_stretch([[frame_path(seq, fid).read_bytes()]], SIZE, "cuda")[:, 0]


def rel_l2(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))


def export_backbone() -> None:
    import onnx
    from onnx import numpy_helper
    model, weights = backbone()
    path = OUT / "yolo26m-backbone-hr.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (torch.zeros(1, 3, 768, 1280),), str(path), export_params=True,
                          opset_version=20, do_constant_folding=False, training=torch.onnx.TrainingMode.EVAL,
                          input_names=["images"], output_names=["p3_feat", "p4_feat", "p5_feat"], dynamo=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph, full_check=True)
    by_hash = {}
    for key, tensor in weights.items():
        if tensor.dtype == torch.float32:
            by_hash.setdefault(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(), []).append(key)
    mapping = []
    for initializer in graph.graph.initializer:
        array = numpy_helper.to_array(initializer)
        digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
        mapping.append({"initializer": initializer.name, "shape": list(array.shape), "sha256": digest,
                        "checkpoint_keys": by_hash.get(digest, [])})
    unmatched = [row["initializer"] for row in mapping if not row["checkpoint_keys"]]
    if unmatched:
        raise ValueError(f"unmatched ONNX initializers: {unmatched}")
    write_json(OUT / "backbone-export.json", {
        "onnx_sha256": sha(path), "onnx_bytes": path.stat().st_size, "opset": 20, "ir_version": graph.ir_version,
        "teacher": str(TEACHER), "teacher_sha256": TEACHER_SHA, "teacher_state_key": "model",
        "teacher_state_meaning": "primary EMA; cache and online loader both read model, not train_model",
        "input_contract": hr_contract.contract_record(SIZE),
        "inputs": [{"name": value.name, "shape": [d.dim_value for d in value.type.tensor_type.shape.dim]}
                   for value in graph.graph.input],
        "outputs": [{"name": value.name, "shape": [d.dim_value for d in value.type.tensor_type.shape.dim]}
                    for value in graph.graph.output],
        "initializers": mapping, "torch": torch.__version__, "onnx": onnx.__version__})
    print(f"backbone: {path} sha256={sha(path)}", flush=True)


def verify_backbone() -> None:
    import onnxruntime as ort
    precision = strict_fp32()
    torch.backends.cudnn.enabled = True
    models = {key: backbone(key)[0].cuda() for key in ("model", "train_model")}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(OUT / "yolo26m-backbone-hr.onnx"), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    rows = []
    with torch.no_grad():
        for seq, fid in (*TRAIN_FRAMES, *REF_FRAMES):
            x = image(seq, fid)
            pt = [value.cpu() for value in models["model"](x)]
            raw = [value.cpu() for value in models["train_model"](x)]
            native = session.run(None, {"images": x.cpu().numpy()})
            role = "cache_val" if seq == "MOT17-10-SDP" else "cache_train"
            cache_path = CACHE / role / "cache" / seq / f"{fid:06d}.pt"
            cache = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
            levels = {}
            for i, key in enumerate(SCALES):
                reference = cache[key][None].float()
                rounded = pt[i].half().float()
                raw_rounded = raw[i].half().float()
                levels[key] = {
                    "shape": list(pt[i].shape), "elements": pt[i].numel(),
                    "model_cache_bitwise_after_cpu_fp16": torch.equal(rounded.view(torch.int32), reference.view(torch.int32)),
                    "model_cache_different_elements": int((rounded.view(torch.int32) != reference.view(torch.int32)).sum()),
                    "model_cache_max_abs_after_cpu_fp16": float((rounded - reference).abs().max()),
                    "train_model_cache_different_elements": int((raw_rounded.view(torch.int32) != reference.view(torch.int32)).sum()),
                    "train_model_cache_rel_l2_after_cpu_fp16": rel_l2(raw_rounded.numpy(), reference.numpy()),
                    "onnx_vs_pytorch_rel_l2": rel_l2(native[i], pt[i].numpy()),
                    "onnx_vs_pytorch_max_abs": float(np.abs(native[i] - pt[i].numpy()).max()),
                    "onnx_vs_cache_rel_l2": rel_l2(native[i], reference.numpy()),
                    "finite": bool(np.isfinite(native[i]).all() and torch.isfinite(pt[i]).all())}
            rows.append({"sequence": seq, "fid": fid, "jpeg_sha256": sha(frame_path(seq, fid)),
                         "cache": str(cache_path), "cache_sha256": sha(cache_path), "levels": levels})
            print(f"verify: {seq}/{fid}: " + str({key: (v["model_cache_different_elements"],
                  v["onnx_vs_pytorch_rel_l2"]) for key, v in levels.items()}), flush=True)
    all_levels = [level for row in rows for level in row["levels"].values()]
    report = {
        "teacher_sha256": TEACHER_SHA, "chosen_state": "model", "raw_negative_control": "train_model",
        "rounding": "strict FP32 GPU batch 1 -> CPU FP16 -> CPU FP32 (hr_cache.to_half)",
        "source_trace": [
            "training/hr/hr_teacher.py: model = primary EMA; train_model = raw",
            "training/hr/hr_cache.py: build_teacher -> build_gated_yolo_detector(weights_path)",
            "pinned yolo_gated_detector.py: sd = state[model]",
            "training/hr_mixed/online_features.py: OnlineFeatures -> hr_cache.build_teacher"],
        "onnx_sha256": sha(OUT / "yolo26m-backbone-hr.onnx"), "onnxruntime": ort.__version__,
        "onnxruntime_providers": session.get_providers(), "precision_start": precision,
        "precision_end": t01_precision.checked_readback("fp32-strict"), "frames": rows,
        "tolerances": {"fp32_implementation_rel_l2": 1e-4, "train_vs_deploy_rel_l2": 2**-11 + 1e-4},
        "cache_identity_pass": all(v["model_cache_bitwise_after_cpu_fp16"] for v in all_levels),
        "onnx_tolerance_pass": all(v["finite"] and v["onnx_vs_pytorch_rel_l2"] <= 1e-4
                                   and v["onnx_vs_cache_rel_l2"] <= 2**-11 + 1e-4 for v in all_levels),
        "scope": "BACKBONE_POLICY secondary rel-L2 tolerances; no binary64 16-sigma native acceptance"}
    write_json(OUT / "backbone-verify.json", report)
    if not report["cache_identity_pass"] or not report["onnx_tolerance_pass"]:
        raise RuntimeError("backbone numerical verification failed")


def write_f32(path: Path, tensors) -> None:
    with path.open("wb") as stream:
        for tensor in tensors:
            stream.write(tensor.detach().cpu().contiguous().numpy().astype("<f4").tobytes())


def references() -> None:
    from torchvision.io import decode_jpeg
    precision = strict_fp32()
    teacher = backbone()[0].cuda()
    checked_load(STUDENT, STUDENT_SHA)
    head, _ = t01_eval.build_student(STUDENT, torch.device("cuda"))
    refs = OUT / "refs"
    for directory in (refs, refs / "backbone", refs / "model"):
        directory.mkdir(parents=True, exist_ok=True)
    cases = ["# label\twidth\theight\toutput_width\toutput_height\tjpeg\trgb\tf32"]
    frames = []
    with (refs / "model-images.f32").open("wb") as images_file, (refs / "head-features.f32").open("wb") as features_file:
        with torch.no_grad():
            for i, (seq, fid) in enumerate(REF_FRAMES):
                stem = f"case-{i:03d}"
                path = frame_path(seq, fid)
                jpeg = path.read_bytes()
                x = image(seq, fid)
                torch.cuda.current_stream().synchronize()
                rgb = decode_jpeg(torch.frombuffer(bytearray(jpeg), dtype=torch.uint8), device="cuda")
                height, width = rgb.shape[1:]
                (refs / f"{stem}.jpg").write_bytes(jpeg)
                rgb.cpu().contiguous().numpy().tofile(refs / f"{stem}.rgb")
                write_f32(refs / f"{stem}.f32", [x])
                images_file.write((refs / f"{stem}.f32").read_bytes())
                feats = teacher(x)
                cls, reg = head(feats)
                write_f32(refs / "backbone" / f"input-{i}.f32", feats)
                features_file.write((refs / "backbone" / f"input-{i}.f32").read_bytes())
                write_f32(refs / "model" / f"input-{i}.f32", (*cls, *reg))
                cases.append("\t".join(map(str, (f"{seq}/{fid}", width, height, *SIZE,
                                                f"{stem}.jpg", f"{stem}.rgb", f"{stem}.f32"))))
                frames.append({"index": i, "sequence": seq, "fid": fid,
                               "backbone": f"backbone/input-{i}.f32", "model": f"model/input-{i}.f32"})
                print(f"references: {seq}/{fid}", flush=True)
    (refs / "cases.tsv").write_text("\n".join(cases) + "\n")
    write_json(refs / "manifest.json", {
        "contract": "hr-input/1", "frames": frames, "teacher_sha256": TEACHER_SHA, "teacher_state_key": "model",
        "student_sha256": STUDENT_SHA, "student_state_key": "student", "precision_start": precision,
        "precision_end": t01_precision.checked_readback("fp32-strict"),
        "head_backend": "pinned pure PyTorch stepwise scan; no old CUDA extension",
        "layout": "little-endian FP32, contiguous NCHW; concatenated in listed order; no headers",
        "backbone_names": list(SCALES), "backbone_shapes": FEATURE_SHAPES,
        "output_names": OUTPUT_NAMES, "output_shapes": OUTPUT_SHAPES,
        "model_images": "model-images.f32", "head_features": "head-features.f32",
        "hashes": {str(path.relative_to(refs)): sha(path) for path in sorted(refs.rglob("*"))
                   if path.is_file() and path.name != "manifest.json"}})


def validation_replay() -> None:
    evidence = RESEARCH / "evidence/hr/2026-11-03/full25"
    validation_path = evidence / "full25-validation.json"
    recorded = json.loads(validation_path.read_text())
    total = recorded["total"]
    events_path = evidence / "full25-events.csv"
    with events_path.open() as stream:
        events = list(csv.DictReader(stream))
    acc = t01_eval.Accumulator()
    acc.n_gt = total["gt"]
    acc.records = [(float(row["score"]), int(row["tp"])) for row in events]
    ap = acc.result()["ap50"]
    tp = sum(int(row["tp"]) for row in events if float(row["score"]) >= 0.25)
    predictions = sum(float(row["score"]) >= 0.25 for row in events)
    selection = json.loads((STUDENT.parent.parent / "selection.json").read_text())["selected"]
    selected_validation = Path(selection["validation"])
    if sha(selected_validation) != selection["validation_sha256"]:
        raise ValueError("selected validation SHA-256 differs")
    if json.loads(selected_validation.read_text())["total"] != total:
        raise ValueError("canonical replay metrics differ from the selected validation")
    point = total["thresholds"]["0.25"]
    if (tp, predictions - tp) != (point["tp"], point["fp"]):
        raise ValueError("recorded event operating point differs")
    report = {
        "student_sha256": sha(STUDENT), "selected_checkpoint_sha256": selection["checkpoint_sha256"],
        "validation_sha256": sha(validation_path), "events_sha256": sha(events_path),
        "selected_validation_sha256": sha(selected_validation), "selected_validation_total_exact": True,
        "method": "T01 Accumulator.result on recorded canonical GPU replay score/TP events; no inference or selection",
        "frames": total["frames"], "gt": total["gt"], "ap50": ap, "ap50_percent": 100 * ap,
        "recorded_ap50": total["ap50"], "tp_at_0.25": tp, "fp_at_0.25": predictions - tp,
        "recorded_detections_sha256": total["detections_sha256"],
        "ap50_exact": ap == total["ap50"], "held_out_evaluated": False}
    write_json(OUT / "student-validation-replay.json", report)
    if not report["ap50_exact"] or report["student_sha256"] != STUDENT_SHA or selection["checkpoint_sha256"] != STUDENT_SHA:
        raise RuntimeError("recorded validation differs from the fixed student")
    print(f"student AP50: {100 * ap:.12f}% TP={tp} FP={predictions - tp}", flush=True)


def manifest() -> None:
    files = [path for path in sorted(OUT.rglob("*")) if path.is_file() and path != OUT / "manifest.json"]
    sources = [*sorted(HERE.glob("*.py")), HERE / "README.md",
               REPO / "src/head_weights.cpp", REPO / "include/saccade/head_weights.hpp",
               REPO / "tests/head_weights_tests.cpp", REPO / "tools/head_weights.cpp",
               REPO / "training/t01/t01_export.py", REPO / "training/t01/t01_eval.py",
               REPO / "training/t01/t01_precision.py",
               RESEARCH / "training/hr/hr_contract.py", RESEARCH / "training/hr/hr_cache.py",
               RESEARCH / "training/hr/hr_teacher.py", RESEARCH / "training/hr_mixed/online_features.py",
               SOURCE / "src/saccade/perception/temporal_yolo/yolo_gated_detector.py",
               SOURCE / "src/saccade/perception/temporal_yolo/mamba_head.py"]
    write_json(OUT / "manifest.json", {
        "schema": "saccade-hr-deploy/1", "student": str(STUDENT), "student_sha256": STUDENT_SHA,
        "teacher": str(TEACHER), "teacher_sha256": TEACHER_SHA, "base": str(BASE), "base_sha256": BASE_SHA,
        "input_contract": hr_contract.contract_record(SIZE), "held_out_evaluated": False,
        "artifacts": {str(path.relative_to(OUT)): {"bytes": path.stat().st_size, "sha256": sha(path)} for path in files},
        "sources": {str(path): sha(path) for path in sources}})
    print(f"manifest: {len(files)} artifacts", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("backbone", "verify-backbone", "references", "validation-replay", "manifest"))
    args = parser.parse_args()
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    {"backbone": export_backbone, "verify-backbone": verify_backbone, "references": references,
     "validation-replay": validation_replay, "manifest": manifest}[args.command]()


if __name__ == "__main__":
    main()

"""Export and verify the T01 matching YOLO26m backbone/neck and real head inputs.

    venv-export/bin/python t01_export.py backbone [--smoke]
    venv-export/bin/python t01_export.py verify-backbone [--smoke]
    venv-export/bin/python t01_export.py features [--smoke]

`backbone` writes an FP32 ONNX of YOLO26m layers 0-22 from the selected
teacher (e12) with ordered outputs p3_feat/p4_feat/p5_feat, BatchNorm kept as
separate nodes (no folding), so every weight initializer can be matched byte
for byte to a teacher checkpoint tensor. `verify-backbone` compares ONNX
Runtime (CPU, FP32) with the PyTorch teacher and with the cached training
features, plus a negative control with the untrained base weights. `features`
writes fixed FP32 P3/P4/P5 frames for the S04 head comparison.

PyTorch references run under the fp32-strict policy (t01_precision), read back
in this process; the ONNX Runtime provider is recorded, and a CUDA execution
provider would be created with use_tf32=0 and checked.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

import t01_precision
from t01_common import artifact_root, load_recipe, read_json, sha256_file, write_json
from t01_exec import _BlockOldExtensions

FEATURE_FRAMES = (("MOT17-02-SDP", 1), ("MOT17-10-SDP", 1), ("MOT17-13-SDP", 400))
VERIFY_FRAMES = (("MOT17-10-SDP", 1), ("MOT17-10-SDP", 327), ("MOT17-10-SDP", 654))
OUTPUT_NAMES = ("p3_feat", "p4_feat", "p5_feat")
LAYER_INDEX = (16, 19, 22)


def strict_fp32() -> dict[str, Any]:
    """Apply fp32-strict before any CUDA math; returns the checked in-process readback."""
    t01_precision.apply("fp32-strict")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return t01_precision.checked_readback("fp32-strict")


def ort_session(onnx_path: Path, provider: str = "CPUExecutionProvider") -> tuple[Any, dict[str, Any]]:
    """ONNX Runtime session with the provider actually used read back; CUDA EP only with use_tf32=0."""
    import onnxruntime as ort  # noqa: PLC0415

    options = {"use_tf32": "0"} if provider == "CUDAExecutionProvider" else {}
    sess = ort.InferenceSession(str(onnx_path), providers=[(provider, options)])
    used = sess.get_providers()
    got = sess.get_provider_options()
    if used[0] != provider:
        raise SystemExit(f"ONNX Runtime fell back to {used} (requested {provider})")
    if "CUDAExecutionProvider" in used and got["CUDAExecutionProvider"].get("use_tf32") != "0":
        raise SystemExit(f"CUDA EP without use_tf32=0: {got['CUDAExecutionProvider']}")
    return sess, {"onnxruntime": ort.__version__, "available": ort.get_available_providers(), "used": used,
                  "provider_options": got,
                  "tf32": "CUDA EP use_tf32=0 (read back)" if "CUDAExecutionProvider" in used
                          else "not applicable: CPU execution provider (no TF32)"}


def setup(smoke: bool) -> tuple[dict[str, Any], Path, Path]:
    recipe = load_recipe()
    root = artifact_root(recipe)
    os.environ["SACCADE_T01_GPU_DECODE_ONLY"] = "1"
    sys.meta_path.insert(0, _BlockOldExtensions())
    sys.path[:0] = [str(root / "src"), str(root / "src" / "src")]
    runs = root / ("smoke" if smoke else "runs")
    return recipe, runs, root / ("smoke-export" if smoke else "export")


class Backbone(nn.Module):
    """YOLO layers 0-22 with the model's own routing; returns layers 16/19/22."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.layers = model.model
        self.save = set(model.save)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        saved: list[torch.Tensor | None] = []
        x: Any = images
        for i in range(23):
            m = self.layers[i]
            if m.f != -1:
                x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
            x = m(x)
            saved.append(x if i in self.save else None)
        return saved[16], saved[19], saved[22]  # type: ignore[return-value]


def load_teacher_yolo(recipe: dict[str, Any], ckpt_path: Path | None) -> tuple[nn.Module, dict[str, Any]]:
    """Fresh, un-patched YOLO26m with the teacher's YOLO tensors (gate alphas reported, not loaded)."""
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(recipe["base_weights"]["path"]).model.float().eval()
    info: dict[str, Any] = {"base_weights_sha256": sha256_file(Path(recipe["base_weights"]["path"]))}
    if ckpt_path is None:
        info["teacher"] = None
        return model, info
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"]
    yolo = {k[len("yolo_model."):]: v for k, v in sd.items() if k.startswith("yolo_model.")}
    other = sorted(k for k in sd if not k.startswith("yolo_model."))
    model.load_state_dict(yolo, strict=True)
    info.update({"teacher_checkpoint": str(ckpt_path), "teacher_sha256": sha256_file(ckpt_path),
                 "teacher_epoch": state["epoch"], "yolo_tensors_loaded": len(yolo),
                 "excluded_keys": {k: "TrackSpatialGate alpha: the gate is inactive at deployment (gate_input=None), as in the feature cache"
                                   for k in other},
                 "excluded_values": {k: [float(x) for x in sd[k].flatten()] for k in other}})
    return model, info


def frames_from_paths(paths: list[Path], device: torch.device) -> torch.Tensor:
    """The T01 input contract through the trainers' own function: (N,3,640,640) float32 RGB in [0,1]."""
    from saccade.perception.temporal_yolo.dataset import gpu_decode_clip_batch  # noqa: PLC0415

    return gpu_decode_clip_batch([[p.read_bytes()] for p in paths], 640, device)[:, 0]


def frames(recipe: dict[str, Any], picks: tuple[tuple[str, int], ...], device: torch.device) -> torch.Tensor:
    root = Path(recipe["dataset"]["root"]) / "train"
    return frames_from_paths([root / seq / "img1" / f"{fid:06d}.jpg" for seq, fid in picks], device)


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm((a - b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))


def export_backbone(smoke: bool) -> None:
    import onnx  # noqa: PLC0415

    recipe, runs, out = setup(smoke)
    sel = read_json(runs / "teacher" / "selection.json")
    teacher = Path(sel["checkpoint"])
    if sha256_file(teacher) != sel["sha256"]:
        raise SystemExit("teacher checkpoint differs from its selection record")
    model, info = load_teacher_yolo(recipe, teacher)
    wrapper = Backbone(model).eval()
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "yolo26m-backbone-t01.onnx"
    dummy = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        torch.onnx.export(wrapper, (dummy,), str(onnx_path), export_params=True, opset_version=20,
                          do_constant_folding=False, training=torch.onnx.TrainingMode.EVAL,
                          input_names=["images"], output_names=list(OUTPUT_NAMES), dynamo=False)
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph, full_check=True)
    # Byte-level mapping: every float initializer against the teacher's YOLO tensors.
    by_hash: dict[str, list[str]] = {}
    for key, tensor in model.state_dict().items():
        if tensor.dtype == torch.float32:
            by_hash.setdefault(hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(), []).append(key)
    rows, unmatched = [], []
    from onnx import numpy_helper  # noqa: PLC0415

    for init in graph.graph.initializer:
        arr = numpy_helper.to_array(init)
        digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
        keys = by_hash.get(digest, []) if arr.dtype == np.float32 else []
        rows.append({"initializer": init.name, "dtype": str(arr.dtype), "shape": list(arr.shape), "sha256": digest, "checkpoint_keys": keys})
        if arr.dtype == np.float32 and arr.size > 1 and not keys:
            unmatched.append(init.name)
    used = {k for r in rows for k in r["checkpoint_keys"]}
    float_keys = [k for k, t in model.state_dict().items() if t.dtype == torch.float32]
    not_in_graph = [k for k in float_keys if k not in used and not k.startswith("model.23.")]
    write_json(out / "backbone-export.json", {
        **info,
        "onnx": str(onnx_path), "onnx_bytes": onnx_path.stat().st_size, "onnx_sha256": sha256_file(onnx_path),
        "opset": 20, "ir_version": graph.ir_version, "producer": f"{graph.producer_name} {graph.producer_version}",
        "inputs": [{"name": i.name, "shape": [d.dim_value for d in i.type.tensor_type.shape.dim], "dtype": "float32"} for i in graph.graph.input],
        "outputs": [{"name": o.name, "shape": [d.dim_value for d in o.type.tensor_type.shape.dim], "dtype": "float32",
                     "source_layer": LAYER_INDEX[i]} for i, o in enumerate(graph.graph.output)],
        "nodes": len(graph.graph.node), "initializers": len(rows),
        "float_initializers_unmatched": unmatched,
        "teacher_float_tensors_not_in_graph_outside_detect": not_in_graph,
        "detect_head_layer_23": "not exported (the Mamba-FPN head replaces it)",
        "initializer_mapping": rows,
        "preprocessing_contract": recipe["preprocessing"]["contract"],
    })
    print(f"ONNX {onnx_path} sha256={sha256_file(onnx_path)} initializers={len(rows)} unmatched={len(unmatched)} not_in_graph={len(not_in_graph)}")


def verify_backbone(smoke: bool) -> None:
    recipe, runs, out = setup(smoke)
    precision = strict_fp32()
    exp = read_json(out / "backbone-export.json")
    onnx_path = Path(exp["onnx"])
    if sha256_file(onnx_path) != exp["onnx_sha256"]:
        raise SystemExit("ONNX differs from its export record")
    device = torch.device("cuda")
    x = frames(recipe, VERIFY_FRAMES, device)
    teacher, _ = load_teacher_yolo(recipe, Path(exp["teacher_checkpoint"]))
    base, _ = load_teacher_yolo(recipe, None)
    sess, ort_info = ort_session(onnx_path)
    cache = runs / "cache_val" / "cache"
    report: dict[str, Any] = {"onnx_sha256": exp["onnx_sha256"], "frames": [f"{s}/{f}" for s, f in VERIFY_FRAMES],
                              "onnxruntime": ort_info, "pytorch_compute_precision": precision, "per_frame": []}
    with torch.no_grad():
        t_gpu = Backbone(teacher.to(device)).eval()
        b_gpu = Backbone(base.to(device)).eval()
        for i, (seq, fid) in enumerate(VERIFY_FRAMES):
            xi = x[i:i + 1]
            ort_out = sess.run(list(OUTPUT_NAMES), {"images": xi.cpu().numpy()})
            pt = [t.cpu().numpy() for t in t_gpu(xi)]
            pb = [t.cpu().numpy() for t in b_gpu(xi)]
            row: dict[str, Any] = {"frame": f"{seq}/{fid}", "levels": {}}
            cached = torch.load(cache / seq / f"{fid:06d}.pt", map_location="cpu", weights_only=True) if (cache / seq).is_dir() else None
            for k, name in enumerate(("p3", "p4", "p5")):
                level = {"onnx_vs_pytorch_rel_l2": rel_l2(ort_out[k], pt[k]),
                         "onnx_vs_pytorch_max_abs": float(np.abs(ort_out[k] - pt[k]).max()),
                         "shape": list(ort_out[k].shape)}
                if cached is not None:
                    c = cached[name].float().numpy()[None]
                    level["onnx_vs_training_cache_rel_l2"] = rel_l2(ort_out[k], c)
                    level["untrained_base_vs_training_cache_rel_l2"] = rel_l2(pb[k], c)
                row["levels"][name] = level
            report["per_frame"].append(row)
    report["cache"] = str(cache) if cache.is_dir() else None
    report["pytorch_compute_precision_end"] = t01_precision.checked_readback("fp32-strict")
    write_json(out / "backbone-verify.json", report)
    for row in report["per_frame"]:
        print(row["frame"], {k: {m: f"{v:.2e}" for m, v in lv.items() if m != "shape"} for k, lv in row["levels"].items()})


def write_features(smoke: bool) -> None:
    recipe, runs, out = setup(smoke)
    precision = strict_fp32()
    exp = read_json(out / "backbone-export.json")
    device = torch.device("cuda")
    teacher, _ = load_teacher_yolo(recipe, Path(exp["teacher_checkpoint"]))
    net = Backbone(teacher.to(device)).eval()
    x = frames(recipe, FEATURE_FRAMES, device)
    blobs = []
    with torch.no_grad():
        for i in range(len(FEATURE_FRAMES)):
            for t in net(x[i:i + 1]):
                blobs.append(t.contiguous().cpu().numpy().astype("<f4").tobytes())
    path = out / "head-features.f32"
    path.write_bytes(b"".join(blobs))
    write_json(out / "head-features.json", {
        "file": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "frames": [f"{s}/{f}" for s, f in FEATURE_FRAMES],
        "layout": "per frame: p3 [1,256,80,80], p4 [1,512,40,40], p5 [1,512,20,20]; FP32 little-endian, NCHW",
        "producer": "PyTorch FP32 (TF32 off) teacher backbone, T01 input contract", "teacher_sha256": exp["teacher_sha256"],
        "compute_precision": precision, "storage_dtype": "float32",
    })
    print(f"{path} sha256={sha256_file(path)} bytes={path.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("backbone", "verify-backbone", "features"))
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    {"backbone": export_backbone, "verify-backbone": verify_backbone, "features": write_features}[a.cmd](a.smoke)


if __name__ == "__main__":
    main()

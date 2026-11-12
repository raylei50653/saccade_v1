"""Fixed-weight YOLO26m ONNX exports; four existing HR reference inputs.

Primary raw6 exports preserve one2many head for same-F04 system comparisons.
Additional official-default export uses original one2one/end2end decoded head.
No training, tuning, deployment change or synthetic scan status outputs.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time

sys.dont_write_bytecode = True
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
MAIN = Path("/home/ray/developer/ai/saccade_v1")
DEV = Path("/home/ray/developer/ai/saccade_v1_dev")
ROOT = MAIN / "build/yolo26m-trt-comparison-20261111"
SOURCE = Path("/home/ray/.cache/saccade-v1/t01/src")
BASE = Path("/home/ray/developer/ai/dataset/yolo26m.pt")
TEACHER = DEV / "build-hr/HR01-UA2-20261102/teacher/attempt-001/epoch_0004.ckpt"
BASE_SHA = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
TEACHER_SHA = "08be5abae55d26b762e262948ded7d3deb11295aa0509ee04ad3582a2c9774cd"
REFS = Path("/home/ray/.cache/saccade-v1/hr-deploy/refs")
NAMES = [f"{kind}_{level}" for kind in ("cls", "reg") for level in ("p3", "p4", "p5")]
sys.path[:0] = [str(MAIN / "training/t01"), str(DEV / "training/hr"), str(SOURCE / "src"), str(SOURCE)]
import numpy as np
import torch
import onnx
import onnxruntime as ort
from t01_export import Backbone
import t01_eval


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda: f.read(4 << 20), b""):
            h.update(x)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


class RawHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = Backbone(model)
        self.head = model.model[-1]
        assert self.head.nc == 80 and self.head.reg_max == 1 and self.head.nl == 3

    def forward(self, images):
        features = self.backbone(images)
        cls = tuple(self.head.cv3[i](features[i]) for i in range(3))
        reg = tuple(self.head.cv2[i](features[i]) for i in range(3))
        return (*cls, *reg)


def load_model(arm):
    from ultralytics import YOLO
    assert sha(BASE) == BASE_SHA
    model = YOLO(str(BASE)).model.float().eval()
    if arm == "finetuned":
        assert sha(TEACHER) == TEACHER_SHA
        checkpoint = torch.load(TEACHER, map_location="cpu", weights_only=False)
        state = {k.removeprefix("yolo_model.").replace("._orig_mod.", "."): v
                 for k, v in checkpoint["model"].items() if k.startswith("yolo_model.")}
        model.load_state_dict(state, strict=True)
    if arm == "official-default":
        head = model.model[-1]
        assert head.end2end
        head.export = True
        head.format = "onnx"
        head.dynamic = False
        head.max_det = 300
        return model.requires_grad_(False)
    return RawHead(model).eval().requires_grad_(False)


def metrics(actual, expected):
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    return {"shape": list(a.shape), "finite": bool(np.isfinite(a).all() and np.isfinite(b).all()),
            "max_abs": float(np.abs(a-b).max()), "mean_abs": float(np.abs(a-b).mean()),
            "rel_l2": float(np.linalg.norm((a-b).ravel()) / max(np.linalg.norm(b.ravel()), 1e-30))}


def compare_decoded(actual, expected):
    from scipy.optimize import linear_sum_assignment
    a, b = actual[0].astype(np.float64), expected[0].astype(np.float64)
    # Match the same decoded detections despite close-score TopK order changes.
    cost = np.abs(a[:, None, :4] - b[None, :, :4]).mean(2) / 1280
    cost += np.abs(a[:, None, 4] - b[None, :, 4])
    cost += (a[:, None, 5] != b[None, :, 5]) * 10
    i, j = linear_sum_assignment(cost)
    reordered = a[i]
    expected_matched = b[j]
    return {"raw_ordered": metrics(actual, expected), "matched": metrics(reordered, expected_matched),
            "class_mismatches": int((reordered[:, 5] != expected_matched[:, 5]).sum()),
            "max_box_abs_pixels": float(np.abs(reordered[:, :4] - expected_matched[:, :4]).max()),
            "max_score_abs": float(np.abs(reordered[:, 4] - expected_matched[:, 4]).max())}


def run(arm):
    started = time.monotonic()
    out = ROOT / "artifacts" / arm
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.onnx"
    assert not path.exists(), f"refuse overwrite {path}"
    model = load_model(arm)
    names = ["output0"] if arm == "official-default" else NAMES
    with torch.no_grad():
        model(torch.zeros(1, 3, 768, 1280))
        torch.onnx.export(model, (torch.zeros(1, 3, 768, 1280),), str(path), export_params=True,
                          opset_version=20, do_constant_folding=False, training=torch.onnx.TrainingMode.EVAL,
                          input_names=["images"], output_names=names, dynamo=False)
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph, full_check=True)
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.save(graph, str(path))
    describe = lambda v: {"name": v.name, "dtype": v.type.tensor_type.elem_type,
                           "shape": [d.dim_value if d.HasField("dim_value") else d.dim_param
                                     for d in v.type.tensor_type.shape.dim]}
    export = {"arm": arm, "onnx": str(path), "onnx_sha256": sha(path), "onnx_bytes": path.stat().st_size,
              "inputs": [describe(v) for v in graph.graph.input], "outputs": [describe(v) for v in graph.graph.output],
              "operators": dict(sorted(__import__('collections').Counter(n.op_type for n in graph.graph.node).items())),
              "opset": 20, "ir_version": graph.ir_version, "scan_status_outputs": False,
              "base": {"path": str(BASE), "sha256": BASE_SHA},
              "teacher": {"path": str(TEACHER), "sha256": TEACHER_SHA, "state": "model primary EMA"} if arm == "finetuned" else None,
              "policy": "original one2one/end2end decoded Top300, no NMS" if arm == "official-default"
                        else "one2many raw cls logits / reg ltrb, P3/P4/P5; downstream native F04 unchanged",
              "versions": {n: importlib.metadata.version(n) for n in ("torch", "onnx", "onnxruntime", "ultralytics")}}
    write(out / "export.json", export)
    print(json.dumps({"event": "EXPORTED", "arm": arm, "onnx_sha256": export["onnx_sha256"],
                      "inputs": export["inputs"], "outputs": export["outputs"]}), flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    precision = t01_eval.strict_fp32()
    model = model.cuda()
    refs = out / "refs"
    refs.mkdir()
    rows = []
    with torch.no_grad():
        for i, fid in enumerate((1, 218, 437, 654)):
            source = REFS / f"case-{i:03d}.f32"
            x = np.fromfile(source, dtype="<f4").reshape(1, 3, 768, 1280)
            source_bytes = source.read_bytes()
            (refs / f"input-{i}.f32").write_bytes(source_bytes)
            outputs = model(torch.from_numpy(x).cuda())
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
            pt = [v.cpu().contiguous().numpy() for v in outputs]
            actual = session.run(names, {"images": x})
            with (refs / f"pytorch-{i}.f32").open("wb") as stream:
                for v in pt:
                    stream.write(v.astype("<f4").tobytes())
            with (refs / f"onnx-{i}.f32").open("wb") as stream:
                for v in actual:
                    stream.write(v.astype("<f4").tobytes())
            checks = {name: metrics(a, b) for name, a, b in zip(names, actual, pt, strict=True)}
            decoded = compare_decoded(actual[0], pt[0]) if arm == "official-default" else None
            passed = all(v["finite"] and v["rel_l2"] <= 1e-4 for v in checks.values()) if decoded is None else (
                decoded["matched"]["finite"] and decoded["matched"]["rel_l2"] <= 1e-4 and decoded["class_mismatches"] == 0)
            row = {"sequence": "MOT17-10-SDP", "fid": fid, "input_sha256": sha(source),
                   "pytorch_outputs_sha256": sha(refs / f"pytorch-{i}.f32"),
                   "onnx_outputs_sha256": sha(refs / f"onnx-{i}.f32"), "outputs": checks,
                   "decoded_set_comparison": decoded, "pass": passed}
            rows.append(row)
            print(json.dumps({"event": "PT_ORT_VERIFIED", "arm": arm, "fid": fid, "pass": passed,
                              "max_rel_l2": max(v["rel_l2"] for v in checks.values())}), flush=True)
    verification = {"status": "PASS" if all(r["pass"] for r in rows) else "FAIL", "arm": arm,
                    "onnx_sha256": sha(path), "provider": session.get_providers(), "precision": precision,
                    "input_shape": [1, 3, 768, 1280], "output_names": names,
                    "output_shapes": [list(v.shape) for v in pt], "frames": rows,
                    "tolerance": {"FP32_PT_vs_ORT_rel_l2": 1e-4},
                    "scope": "four fixed HR inputs; FP32 numerical export check, not quality or deployment acceptance",
                    "wall_seconds": time.monotonic() - started}
    write(out / "onnx-verify.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError(f"{arm} PT vs ONNX verification failed")
    del model
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["official", "finetuned", "official-default"], required=True)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(0)
    run(args.arm)


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
from collections import Counter
import json
import numpy as np
from export_models import ROOT, sha, metrics, compare_decoded, write

p = argparse.ArgumentParser()
p.add_argument("arm", choices=["official", "finetuned", "official-default"])
args = p.parse_args()
out = ROOT / "artifacts" / args.arm
verification = json.loads((out / "onnx-verify.json").read_text())
native = json.loads((out / "trt-verify/summary.json").read_text())
assert native["status"] == "PASS" and native["verify_only"] and native["ordinary_repeat_and_graph_bits_equal"]
assert native["engine_sha256"] == sha(out / "model.engine")
shapes = verification["output_shapes"]
sizes = [int(np.prod(s)) for s in shapes]
names = verification["output_names"]
rows = []
for i, fid in enumerate((1, 218, 437, 654)):
    actual = np.fromfile(out / "trt-verify" / f"input-{i}.f32", dtype="<f4")
    reference = np.fromfile(out / "refs" / f"pytorch-{i}.f32", dtype="<f4")
    assert actual.size == reference.size == sum(sizes)
    offset = 0
    checks = {}
    decoded = None
    for name, shape, size in zip(names, shapes, sizes, strict=True):
        a = actual[offset:offset+size].reshape(shape)
        b = reference[offset:offset+size].reshape(shape)
        checks[name] = metrics(a, b)
        if args.arm == "official-default":
            decoded = compare_decoded(a, b)
        offset += size
    # Primary raw6 FP16 numerical check: all tensors finite and <=1% relative L2.
    # Decoded default TopK additionally records order/set/class changes; finite
    # output + full raw numerical metrics are retained without claiming equality.
    passed = all(x["finite"] and x["rel_l2"] <= .01 for x in checks.values()) if decoded is None else all(x["finite"] for x in checks.values())
    rows.append({"fid": fid, "outputs": checks, "decoded_set_comparison": decoded,
                 "raw_reference_sha256": sha(out / "refs" / f"pytorch-{i}.f32"),
                 "trt_outputs_sha256": sha(out / "trt-verify" / f"input-{i}.f32"), "pass": passed})
layers = json.loads((out / "layers.json").read_text())
formats = Counter()
def visit(value):
    if isinstance(value, dict):
        if "Format/Datatype" in value:
            formats[value["Format/Datatype"]] += 1
        for v in value.values(): visit(v)
    elif isinstance(value, list):
        for v in value: visit(v)
visit(layers)
report = {"status": "PASS" if all(r["pass"] for r in rows) else "FAIL", "arm": args.arm,
          "engine_sha256": sha(out / "model.engine"), "onnx_sha256": sha(out / "model.onnx"),
          "trt_ordinary_repeat_graph_bits_equal": True,
          "scan_status_count": native["scan_status_count"], "fp16_internal_actual_formats": dict(formats),
          "engine_layer_count": len(layers.get("Layers", [])), "frames": rows,
          "limits": {"raw6_fp16_rel_l2": .01, "default_decoded": "finite required; TopK ordering/set metrics reported, no equivalence claim"},
          "scope": "four fixed inputs, numerical implementation check; separate from tracker quality and performance",
          "max_raw_rel_l2": max(v["rel_l2"] for r in rows for v in r["outputs"].values())}
write(out / "trt-verify.json", report)
print(json.dumps({"status": report["status"], "arm": args.arm, "max_raw_rel_l2": report["max_raw_rel_l2"],
                  "engine_layer_count": report["engine_layer_count"], "formats": dict(formats)}))
assert report["status"] == "PASS"

"""Binary64 reference and per-element FP32 error scales of the T01 YOLO26m backbone/neck
(docs/BACKBONE_POLICY.md, t01-backbone-fp32/1), for the native TensorRT backbone and the F03 model.

    venv-export/bin/python t01_backbone_ref.py inputs
    venv-export/bin/python t01_backbone_ref.py reference
    venv-export/bin/python t01_backbone_ref.py precheck
    venv-export/bin/python t01_backbone_ref.py compare DUMP_DIR
    venv-export/bin/python t01_backbone_ref.py diagnose MODEL_DUMP_DIR CHAIN_DUMP_DIR

`inputs` writes the fixed model inputs (BACKBONE_POLICY §4): three MOT17 frames
through the T01 input contract (PyTorch fp32-strict, read back) and three
synthetic tensors, as FP32 little-endian NCHW frames back to back.
`reference` evaluates layers 0-22 of the selected teacher (e12) in binary64 on
the CPU, with a functional forward written from the module definitions (no
ONNX, no TensorRT, no CUDA), together with the worst-case bound E and the
probabilistic scale sigma of every element under the operation model of
BACKBONE_POLICY §2-§3. It checks itself against the Ultralytics modules run in
float64 and checks sigma <= E. `compare` judges the TensorRT dumps of
`backbone_trt run` element by element (|trt - ref| <= 16 sigma + 2^-110), adds
the pre-fixed T01 tolerances against PyTorch FP32 and the training cache, and
evaluates the semantic variants of §6, each of which must be detected.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import t01_precision
from t01_common import read_json, sha256_file, write_json
from t01_export import FEATURE_FRAMES, OUTPUT_NAMES, Backbone, frames, load_teacher_yolo, rel_l2, setup, strict_fp32

POLICY = "t01-backbone-fp32/1"
# Operation model (BACKBONE_POLICY §2); the shared constants are HEAD_POLICY's.
U = 2.0**-24            # FP32 unit roundoff
ETA = 2.0**-150         # absolute error of one FP32 rounding in the subnormal range
EPS_T = 2.0**-17        # relative error of a TensorRT FP32 exp, sigmoid or SiLU result
SILU_FLOOR = 2.0**-125  # SiLU absolute error per unit |x| where sigmoid underflows
SILU_LIP = 1.1          # >= max |SiLU'(x)|
FOLD = 5.0              # BatchNorm folded in FP32: <= 5 u of the per-output magnitude (§2)
SOFTMAX_EXP = 8.0       # transcendental evaluations per softmax output (numerator and sum, online rescaling)
SOFTMAX_FLOOR = 2.0**-126
LAMBDA = 16.0
FLOOR = 2.0**-110
LEVELS = ("p3", "p4", "p5")
SHAPES = ((256, 80, 80), (512, 40, 40), (512, 20, 20))
INPUT_SHAPE = (3, 640, 640)
SYNTHETIC = ("zeros", "ones", "random/1")
VARIANT_INPUTS = ("MOT17-02-SDP/1", "random/1")
VARIANTS = ("bgr_input", "no_bn_eps", "bilinear_upsample", "concat_reversed", "flipped_kernels", "sppf_no_shortcut",
            "sppf_kernel3", "attention_no_scale", "softmax_key_axis", "attention_no_pe", "chunk_swapped", "untrained_weights")
THREADS = 16


def gamma(n: float) -> float:
    return n * U / (1 - n * U)


@dataclass
class Ref:
    """Binary64 value, worst-case bound and probabilistic scale of one tensor."""

    v: torch.Tensor
    e: torch.Tensor
    s: torch.Tensor

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Ref:
        return Ref(fn(self.v), fn(self.e), fn(self.s))


def exact(x: torch.Tensor) -> Ref:
    z = torch.zeros_like(x)
    return Ref(x, z, z.clone())


def cat(xs: list[Ref], dim: int = 1) -> Ref:
    return Ref(torch.cat([x.v for x in xs], dim), torch.cat([x.e for x in xs], dim), torch.cat([x.s for x in xs], dim))


@dataclass
class Ctx:
    variant: str | None = None
    counts: dict[str, int] = field(default_factory=dict)

    def count(self, op: str) -> None:
        self.counts[op] = self.counts.get(op, 0) + 1


# ---- operations (BACKBONE_POLICY §3) ----

def conv_bn(ctx: Ctx, x: Ref, m: nn.Module, act: bool) -> Ref:
    """Conv2d (no bias) + eval BatchNorm (+ SiLU): one FP32 dot product per output with the folded parameters
    w' = w s, b' = beta - mu s, s = gamma / sqrt(var + eps), plus the fold term FOLD u (sum |w' x| + |beta| + |mu s|)."""
    conv, bn = m.conv, m.bn
    assert isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d) and conv.bias is None
    assert conv.dilation == (1, 1) and conv.padding_mode == "zeros"
    eps = 0.0 if ctx.variant == "no_bn_eps" else float(np.float32(bn.eps))
    s = bn.weight.double() / torch.sqrt(bn.running_var.double() + eps)
    w = conv.weight.double()
    if ctx.variant == "flipped_kernels" and w.shape[-1] > 1:
        w = torch.flip(w, (-2, -1))
    wf = w * s.view(-1, 1, 1, 1)
    mu_s = bn.running_mean.double() * s
    bf = bn.bias.double() - mu_s
    args = (conv.stride, conv.padding, conv.dilation, conv.groups)
    v = F.conv2d(x.v, wf, bf, *args)
    aw = wf.abs()
    mag = F.conv2d(x.v.abs(), aw, None, *args)
    m_out = mag + bf.abs().view(1, -1, 1, 1)
    m_fold = mag + (bn.bias.double().abs() + mu_s.abs()).view(1, -1, 1, 1)
    terms = float(w.shape[1] * w.shape[2] * w.shape[3] + 1)
    g = gamma(terms)
    e = (1 + g) * F.conv2d(x.e, aw, None, *args) + g * m_out + 2 * terms * ETA + FOLD * U * m_fold
    var = F.conv2d(x.s * x.s, wf * wf, None, *args)
    sig = torch.sqrt(var + 2 * terms * U * U * m_out * m_out) + FOLD * U * m_fold
    ctx.count("conv")
    y = Ref(v, e, sig)
    if act:
        assert isinstance(m.act, nn.SiLU), type(m.act)
        return silu(ctx, y)
    assert isinstance(m.act, nn.Identity), type(m.act)
    return y


def conv_module(ctx: Ctx, x: Ref, m: nn.Module) -> Ref:
    assert type(m).__name__ == "Conv", type(m)
    return conv_bn(ctx, x, m, act=isinstance(m.act, nn.SiLU))


def silu(ctx: Ctx, x: Ref) -> Ref:
    """HEAD_POLICY §3 SiLU (head_ref::silu)."""
    v = x.v
    sg = torch.sigmoid(v)
    out = v * sg
    slope = sg * (1 + v * (1 - sg))
    e = SILU_LIP * x.e
    ctx.count("silu")
    return Ref(out, e + EPS_T * (out.abs() + e) + SILU_FLOOR * (v.abs() + x.e),
               torch.hypot(slope * x.s, EPS_T * out) + SILU_FLOOR * v.abs())


def add(ctx: Ctx, a: Ref, b: Ref) -> Ref:
    """HEAD_POLICY §3 sum (head_ref::add): operands may share sources, scales add linearly."""
    v = a.v + b.v
    ctx.count("add")
    return Ref(v, (1 + U) * (a.e + b.e) + U * v.abs(), a.s + b.s + U * v.abs())


def scale(ctx: Ctx, x: Ref, c: float) -> Ref:
    v = c * x.v
    ctx.count("scale")
    return Ref(v, abs(c) * x.e + U * (v.abs() + abs(c) * x.e) + ETA, abs(c) * x.s + U * v.abs())


def maxpool(ctx: Ctx, x: Ref, k: int) -> Ref:
    """Exact: |max x' - max x| <= max |x' - x| over the window, so bound and scale are the window maxima."""
    ctx.count("maxpool")
    return x.map(lambda t: F.max_pool2d(t, k, 1, k // 2))


def upsample(ctx: Ctx, x: Ref, m: nn.Module) -> Ref:
    assert isinstance(m, nn.Upsample) and m.mode == "nearest" and m.scale_factor == 2.0 and m.size is None
    ctx.count("upsample")
    if ctx.variant == "bilinear_upsample":
        return x.map(lambda t: F.interpolate(t, scale_factor=2.0, mode="bilinear", align_corners=False))
    return x.map(lambda t: F.interpolate(t, scale_factor=2.0, mode="nearest"))


def matmul(ctx: Ctx, a: Ref, b: Ref) -> Ref:
    """y = sum_k a_k b_k with both operands carrying errors: the per-k terms |b| s_a + |a| s_b add linearly
    (shared sources), distinct k in quadrature; own rounding as an FP32 dot product of K terms."""
    v = a.v @ b.v
    aa, ab = a.v.abs(), b.v.abs()
    mag = aa @ ab
    k = float(a.v.shape[-1])
    g = gamma(k)
    e = (1 + g) * (a.e @ ab + aa @ b.e + a.e @ b.e) + g * mag + 2 * k * ETA
    var = (a.s * a.s) @ (b.v * b.v) + (a.v * a.v) @ (b.s * b.s) + 2 * ((aa * a.s) @ (ab * b.s))
    ctx.count("matmul")
    return Ref(v, e, torch.sqrt(var + 2 * k * U * U * mag * mag))


def softmax(ctx: Ctx, z: Ref, dim: int) -> Ref:
    """y_i = exp(z_i - m) / sum_j exp(z_j - m). Inputs, first order: dy_i = y_i (dz_i - sum_j y_j dz_j), bounded
    linearly. Own rounding, relative to y_i: SOFTMAX_EXP eps_t, u |z - m| from the argument, the sum's roundings
    (sqrt(N) u for sigma, N u for the bound) and the division; plus 2^-126 absolute for flushed results."""
    y = torch.softmax(z.v, dim)
    d = (z.v - z.v.amax(dim, keepdim=True)).abs()
    n = float(z.v.shape[dim])
    arg = U * (d + (y * d).sum(dim, keepdim=True))
    s_in = y * (z.s + (y * z.s).sum(dim, keepdim=True))
    e_in = y * (z.e + (y * z.e).sum(dim, keepdim=True))
    ctx.count("softmax")
    return Ref(y, e_in + (SOFTMAX_EXP * EPS_T + arg + (n + 1) * U) * y + SOFTMAX_FLOOR,
               s_in + (SOFTMAX_EXP * EPS_T + arg + (math.sqrt(n) + 1) * U) * y + SOFTMAX_FLOOR)


# ---- modules (Ultralytics 8.4.37 definitions; layer types of YOLO26m 0-22) ----

def bottleneck(ctx: Ctx, x: Ref, m: nn.Module) -> Ref:
    y = conv_module(ctx, conv_module(ctx, x, m.cv1), m.cv2)
    return add(ctx, x, y) if m.add else y


def attention(ctx: Ctx, x: Ref, m: nn.Module) -> Ref:
    b, c, h, w = x.v.shape
    n = h * w
    nh, kd, hd = m.num_heads, m.key_dim, m.head_dim
    qkv = conv_bn(ctx, x, m.qkv, act=False).map(lambda t: t.view(b, nh, 2 * kd + hd, n))
    q = qkv.map(lambda t: t[:, :, :kd])
    k = qkv.map(lambda t: t[:, :, kd:2 * kd])
    v = qkv.map(lambda t: t[:, :, 2 * kd:])
    logits = matmul(ctx, q.map(lambda t: t.transpose(-2, -1)), k)
    s = 1.0 if ctx.variant == "attention_no_scale" else float(np.float32(m.scale))
    p = softmax(ctx, scale(ctx, logits, s), -2 if ctx.variant == "softmax_key_axis" else -1)
    out = matmul(ctx, v, p.map(lambda t: t.transpose(-2, -1))).map(lambda t: t.reshape(b, c, h, w))
    if ctx.variant != "attention_no_pe":
        out = add(ctx, out, conv_bn(ctx, v.map(lambda t: t.reshape(b, c, h, w)), m.pe, act=False))
    return conv_bn(ctx, out, m.proj, act=False)


def psablock(ctx: Ctx, x: Ref, m: nn.Module) -> Ref:
    a = attention(ctx, x, m.attn)
    x = add(ctx, x, a) if m.add else a
    f = conv_module(ctx, conv_module(ctx, x, m.ffn[0]), m.ffn[1])
    return add(ctx, x, f) if m.add else f


def run_block(ctx: Ctx, x: Ref, m: nn.Module) -> Ref:
    name = type(m).__name__
    if name == "Bottleneck":
        return bottleneck(ctx, x, m)
    if name == "PSABlock":
        return psablock(ctx, x, m)
    if name == "C3k":
        inner = conv_module(ctx, x, m.cv1)
        for sub in m.m:
            inner = run_block(ctx, inner, sub)
        return conv_module(ctx, cat([inner, conv_module(ctx, x, m.cv2)]), m.cv3)
    if name == "Sequential":
        for sub in m:
            x = run_block(ctx, x, sub)
        return x
    raise SystemExit(f"no reference for block {name}")


def layer(ctx: Ctx, i: int, x: Any, m: nn.Module) -> Ref:
    name = type(m).__name__
    if name == "Conv":
        return conv_module(ctx, x, m)
    if name == "C3k2":
        y0 = conv_module(ctx, x, m.cv1)
        half = y0.v.shape[1] // 2
        parts = [y0.map(lambda t: t[:, :half]), y0.map(lambda t: t[:, half:])]
        if ctx.variant == "chunk_swapped":
            parts.reverse()
        for sub in m.m:
            parts.append(run_block(ctx, parts[-1], sub))
        return conv_module(ctx, cat(parts), m.cv2)
    if name == "SPPF":
        k = 3 if ctx.variant == "sppf_kernel3" else m.m.kernel_size
        assert m.m.stride == 1 and m.m.padding == m.m.kernel_size // 2 and not m.m.ceil_mode
        ys = [conv_module(ctx, x, m.cv1)]
        for _ in range(m.n):
            ys.append(maxpool(ctx, ys[-1], k))
        y = conv_module(ctx, cat(ys), m.cv2)
        return add(ctx, y, x) if getattr(m, "add", False) and ctx.variant != "sppf_no_shortcut" else y
    if name == "C2PSA":
        y0 = conv_module(ctx, x, m.cv1)
        a, b = y0.map(lambda t: t[:, :m.c]), y0.map(lambda t: t[:, m.c:])
        for sub in m.m:
            b = run_block(ctx, b, sub)
        return conv_module(ctx, cat([a, b]), m.cv2)
    if name == "Upsample":
        return upsample(ctx, x, m)
    if name == "Concat":
        assert m.d == 1
        parts = list(x)
        if ctx.variant == "concat_reversed" and i == 12:
            parts.reverse()
        return cat(parts)
    raise SystemExit(f"no reference for layer {i} {name}")


def reference(model: nn.Module, image: torch.Tensor, variant: str | None = None) -> tuple[list[Ref], dict[str, int]]:
    """Layers 0-22 with the model's own routing (as t01_export.Backbone); image (1,3,640,640) float64."""
    ctx = Ctx(variant)
    if variant == "bgr_input":
        image = image[:, [2, 1, 0]]
    saved: list[Ref | None] = []
    x: Any = exact(image)
    keep = set(model.save)
    for i in range(23):
        m = model.model[i]
        if m.f != -1:
            x = saved[m.f] if isinstance(m.f, int) else [x if j == -1 else saved[j] for j in m.f]
        x = layer(ctx, i, x, m)
        saved.append(x if i in keep else None)
    return [saved[16], saved[19], saved[22]], ctx.counts  # type: ignore[list-item]


# ---- files ----

def input_frames(out: Path) -> tuple[dict[str, Any], np.ndarray]:
    rec = read_json(out / "f03-inputs.json")
    path = Path(rec["file"])
    if sha256_file(path) != rec["sha256"]:
        raise SystemExit(f"{path} differs from its record")
    data = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(len(rec["labels"]), *INPUT_SHAPE)
    return rec, data


def write_inputs(smoke: bool) -> None:
    recipe, runs, out = setup(smoke)
    precision = strict_fp32()
    device = torch.device("cuda")
    real = frames(recipe, FEATURE_FRAMES, device).cpu().numpy().astype("<f4")
    rng = np.random.Generator(np.random.PCG64(1))
    synthetic = {"zeros": np.zeros(INPUT_SHAPE, "<f4"), "ones": np.ones(INPUT_SHAPE, "<f4"),
                 "random/1": rng.random(INPUT_SHAPE, dtype=np.float32).astype("<f4")}
    labels = [f"{s}/{f}" for s, f in FEATURE_FRAMES] + list(SYNTHETIC)
    blob = np.ascontiguousarray(np.concatenate([real] + [synthetic[k][None] for k in SYNTHETIC]))
    path = out / "f03-inputs.f32"
    path.write_bytes(blob.tobytes())
    write_json(out / "f03-inputs.json", {
        "policy": POLICY, "file": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "labels": labels,
        "layout": "per input: images [1,3,640,640] FP32 little-endian NCHW, RGB in [0,1]",
        "real_frames": "T01 input contract t01-input/1 (nvJPEG hardware decode, /255, antialiased bilinear stretch), PyTorch fp32-strict",
        "synthetic": {"zeros": "0", "ones": "1", "random/1": "numpy PCG64(1) Generator.random(float32), uniform [0,1)"},
        "per_input_sha256": {lab: sha256_file_bytes(blob[i].tobytes()) for i, lab in enumerate(labels)},
        "compute_precision": precision, "compute_precision_end": t01_precision.checked_readback("fp32-strict"),
    })
    print(f"{path} sha256={sha256_file(path)} inputs={len(labels)}")


def sha256_file_bytes(data: bytes) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(data).hexdigest()


def float64_module(model: nn.Module) -> nn.Module:
    """The Ultralytics modules in float64 with the FP32 constants of the deployed network (BN eps, attention scale)."""
    import copy  # noqa: PLC0415

    m = copy.deepcopy(model).double().eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eps = float(np.float32(mod.eps))
        if type(mod).__name__ == "Attention":
            mod.scale = float(np.float32(mod.scale))
    return m


def write_reference(smoke: bool) -> None:
    torch.set_num_threads(THREADS)
    recipe, runs, out = setup(smoke)
    exp = read_json(out / "backbone-export.json")
    teacher, info = load_teacher_yolo(recipe, Path(exp["teacher_checkpoint"]))
    rec, data = input_frames(out)
    module64 = Backbone(float64_module(teacher)).eval()
    path = out / "backbone-reference.f64"
    rows = []
    with torch.no_grad(), path.open("wb") as handle:
        for i, label in enumerate(rec["labels"]):
            t0 = time.time()
            image = torch.from_numpy(data[i:i + 1].astype(np.float64))
            refs, counts = reference(teacher, image)
            ref_s = time.time() - t0
            direct = module64(image)
            row: dict[str, Any] = {"label": label, "reference_seconds": round(ref_s, 2), "operations": counts, "levels": {}}
            for k, (lv, r) in enumerate(zip(LEVELS, refs)):
                assert tuple(r.v.shape[1:]) == SHAPES[k]
                for t in (r.v, r.e, r.s):
                    handle.write(np.ascontiguousarray(t.numpy().astype("<f8")).tobytes())
                scale_ = float(direct[k].abs().max())
                rel = r.s / r.v.abs().clamp_min(1e-300)
                row["levels"][lv] = {
                    "self_check_max_abs_vs_module_float64": float((r.v - direct[k]).abs().max()),
                    "self_check_relative_to_max": float((r.v - direct[k]).abs().max()) / max(scale_, 1e-300),
                    "sigma_le_bound": bool((r.s <= r.e).all()),
                    "finite": bool(torch.isfinite(r.v).all() and torch.isfinite(r.s).all()),
                    "max_abs_value": scale_, "max_sigma": float(r.s.max()), "median_sigma": float(r.s.median()),
                    "median_sigma_over_abs_value": float(rel.median()), "max_bound": float(r.e.max()),
                    "bound_infinite": int(torch.isinf(r.e).sum()),
                }
            rows.append(row)
            print(label, f"{ref_s:.1f}s", {lv: f"self {v['self_check_relative_to_max']:.1e} sigma<=E {v['sigma_le_bound']} "
                                               f"med sigma/|v| {v['median_sigma_over_abs_value']:.1e}" for lv, v in row["levels"].items()},
                  flush=True)
    ok = all(lv["self_check_relative_to_max"] <= 1e-12 and lv["sigma_le_bound"] and lv["finite"] for r in rows for lv in r["levels"].values())
    structure = onnx_structure(Path(exp["onnx"]), rows[0]["operations"])
    ok = ok and structure["match"] and all(r["operations"] == rows[0]["operations"] for r in rows)
    write_json(out / "backbone-reference.json", {
        "policy": POLICY, "file": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path),
        "layout": "per input, per level p3/p4/p5: value, bound, sigma, each float64 little-endian NCHW [1,C,H,W]",
        "inputs_sha256": rec["sha256"], "labels": rec["labels"], "teacher_sha256": info["teacher_sha256"],
        "torch": torch.__version__, "threads": THREADS, "self_checks_pass": ok, "onnx_structure": structure, "per_input": rows,
        "constants": {"u": U, "eta": ETA, "eps_t": EPS_T, "silu_floor": SILU_FLOOR, "silu_lipschitz": SILU_LIP, "fold_units": FOLD,
                      "softmax_exp": SOFTMAX_EXP, "softmax_floor": SOFTMAX_FLOOR, "lambda": LAMBDA, "floor": FLOOR},
    })
    print(f"{path} sha256={sha256_file(path)} self_checks={'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("reference self-checks failed")


def onnx_structure(onnx_path: Path, counts: dict[str, int]) -> dict[str, Any]:
    """The reference's tensor operations against the deployed graph's node counts (a structural cross-check;
    the remaining ONNX nodes are shape arithmetic, reshapes, splits and concatenations)."""
    import onnx  # noqa: PLC0415

    nodes: dict[str, int] = {}
    for node in onnx.load(str(onnx_path), load_external_data=False).graph.node:
        nodes[node.op_type] = nodes.get(node.op_type, 0) + 1
    pairs = {"conv": ("Conv", "BatchNormalization"), "silu": ("Sigmoid",), "matmul": ("MatMul",), "softmax": ("Softmax",),
             "maxpool": ("MaxPool",), "upsample": ("Resize",)}
    rows = {op: {"reference": counts.get(op, 0), "onnx": {t: nodes.get(t, 0) for t in types}} for op, types in pairs.items()}
    match = all(all(n == r["reference"] for n in r["onnx"].values()) for r in rows.values())
    return {"onnx_sha256": sha256_file(onnx_path), "pairs": rows, "onnx_nodes": nodes, "match": match}


def read_reference(out: Path) -> tuple[dict[str, Any], np.memmap]:
    rec = read_json(out / "backbone-reference.json")
    path = Path(rec["file"])
    if sha256_file(path) != rec["sha256"]:
        raise SystemExit(f"{path} differs from its record")
    return rec, np.memmap(path, dtype="<f8", mode="r")


def judge(trt: np.ndarray, value: np.ndarray, sigma: np.ndarray, bound: np.ndarray) -> dict[str, Any]:
    """BACKBONE_POLICY §5 for one tensor; same report fields as HEAD_POLICY §4."""
    t = trt.astype(np.float64)
    err = np.abs(t - value)
    tol = LAMBDA * sigma + FLOOR
    ratio = err / tol
    worst = int(np.argmax(ratio))
    rel_ok = np.abs(value) >= LAMBDA * sigma
    c, h, w = np.unravel_index(worst, value.shape[1:])
    return {"count": int(value.size), "failures": int((err > tol).sum()), "nonfinite": int((~np.isfinite(t)).sum()),
            "max_abs": float(err.max()), "max_rel": float((err[rel_ok] / np.abs(value[rel_ok])).max()) if rel_ok.any() else 0.0,
            "rel_excluded": int((~rel_ok).sum()), "max_ratio": float(ratio.max()),
            "max_bound_ratio": float((err / (2 * bound)).max()),
            "worst": {"c": int(c), "h": int(h), "w": int(w), "trt": float(t.ravel()[worst]),
                      "trt_bits": f"{int(np.float32(t.ravel()[worst]).view(np.uint32)):08x}", "reference": float(value.ravel()[worst]),
                      "tolerance": float(tol.ravel()[worst])}}


def split_dump(path: Path) -> list[np.ndarray]:
    raw = np.frombuffer(path.read_bytes(), dtype="<f4")
    out, pos = [], 0
    for s in SHAPES:
        n = int(np.prod(s))
        out.append(raw[pos:pos + n].reshape(1, *s))
        pos += n
    if pos != raw.size:
        raise SystemExit(f"{path}: {raw.size} values, expected {pos}")
    return out


def level_refs(ref: np.memmap, i: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(value, bound, sigma) of p3/p4/p5 for input i."""
    per_level = [int(np.prod(s)) for s in SHAPES]
    pos = i * 3 * sum(per_level)
    out = []
    for k, n in enumerate(per_level):
        out.append(tuple(np.asarray(ref[pos + j * n:pos + (j + 1) * n]).reshape(1, *SHAPES[k]) for j in range(3)))
        pos += 3 * n
    return out  # type: ignore[return-value]


def variant_refs(teacher: nn.Module, base: nn.Module, data: np.ndarray, labels: list[str]) -> dict[str, dict[str, list[Ref]]]:
    out: dict[str, dict[str, list[Ref]]] = {}
    with torch.no_grad():
        for variant in VARIANTS:
            out[variant] = {}
            for label in VARIANT_INPUTS:
                image = torch.from_numpy(data[labels.index(label):labels.index(label) + 1].astype(np.float64))
                out[variant][label] = reference(base if variant == "untrained_weights" else teacher, image,
                                                None if variant == "untrained_weights" else variant)[0]
    return out


def sigmoid_probe() -> list[dict[str, float]]:
    """ONNX Runtime CPU Sigmoid against the exact value on a fixed grid."""
    import onnxruntime as ort  # noqa: PLC0415
    from onnx import TensorProto, helper  # noqa: PLC0415

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])
    model = helper.make_model(helper.make_graph([helper.make_node("Sigmoid", ["x"], ["y"])], "sigmoid", [x], [y]),
                              opset_imports=[helper.make_opsetid("", 20)])
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    xs = np.array([-30, -20, -18, -17, -16, -15, -12, -8, -4, -1, 0, 1, 4, 8, 16], np.float32)
    got = sess.run(None, {"x": xs})[0].astype(np.float64)
    exact_ = 1 / (1 + np.exp(-xs.astype(np.float64)))
    return [{"x": float(a), "onnxruntime": float(g), "exact": float(e), "rel_err": float(abs(g - e) / e)} for a, g, e in zip(xs, got, exact_)]


def precheck(smoke: bool) -> None:
    """CPU only, before any GPU comparison (§5): two FP32 CPU implementations inside the policy band, and every
    semantic variant outside it."""
    torch.set_num_threads(THREADS)
    from t01_export import ort_session  # noqa: PLC0415

    recipe, runs, out = setup(smoke)
    exp = read_json(out / "backbone-export.json")
    rec_in, data = input_frames(out)
    rec_ref, ref = read_reference(out)
    if rec_ref["inputs_sha256"] != rec_in["sha256"]:
        raise SystemExit("the reference was computed for other inputs")
    teacher, _ = load_teacher_yolo(recipe, Path(exp["teacher_checkpoint"]))
    base, _ = load_teacher_yolo(recipe, None)
    sess, ort_info = ort_session(Path(exp["onnx"]))
    net = Backbone(teacher).eval()
    report: dict[str, Any] = {"policy": POLICY, "inputs_sha256": rec_in["sha256"], "reference_sha256": rec_ref["sha256"],
                              "onnx_sha256": exp["onnx_sha256"], "onnxruntime": ort_info, "torch_threads": THREADS,
                              "implementations": {}, "variants": {}}
    with torch.no_grad():
        for impl in ("pytorch_fp32_cpu", "onnxruntime_fp32_cpu"):
            rows = []
            for i, label in enumerate(rec_in["labels"]):
                x = data[i:i + 1].copy()
                got = ([t.numpy() for t in net(torch.from_numpy(x))] if impl == "pytorch_fp32_cpu"
                       else sess.run(list(OUTPUT_NAMES), {"images": x}))
                levels = {}
                for k, lv in enumerate(LEVELS):
                    value, bound, sigma = level_refs(ref, i)[k]
                    levels[lv] = judge(got[k], value, sigma, bound)
                    outside = np.abs(got[k].astype(np.float64) - value) > LAMBDA * sigma + FLOOR
                    levels[lv]["outside_max_abs_reference"] = float(np.abs(value[outside]).max()) if outside.any() else None
                rows.append({"label": label, "levels": levels})
            report["implementations"][impl] = {"max_ratio": max(lv["max_ratio"] for r in rows for lv in r["levels"].values()),
                                               "failures": sum(lv["failures"] for r in rows for lv in r["levels"].values()),
                                               "inputs": rows}
            print(impl, f"max ratio {report['implementations'][impl]['max_ratio']:.4f}",
                  f"failures {report['implementations'][impl]['failures']}", flush=True)
    for variant, per_input in variant_refs(teacher, base, data, rec_in["labels"]).items():
        entry = {}
        for label, refs in per_input.items():
            lv_refs = level_refs(ref, rec_in["labels"].index(label))
            entry[label] = {lv: {"exceeding": int((np.abs(refs[k].v.numpy() - lv_refs[k][0]) > LAMBDA * lv_refs[k][2] + FLOOR).sum()),
                                 "max_ratio": float((np.abs(refs[k].v.numpy() - lv_refs[k][0]) / (LAMBDA * lv_refs[k][2] + FLOOR)).max()),
                                 "rel_l2": rel_l2(refs[k].v.numpy(), lv_refs[k][0])} for k, lv in enumerate(LEVELS)}
        detected = any(lv["exceeding"] > 0 for r in entry.values() for lv in r.values())
        report["variants"][variant] = {"detected": detected, "inputs": entry}
        weakest = min((lv["exceeding"] / int(np.prod(SHAPES[k])), lab, name)
                      for lab, r in entry.items() for k, (name, lv) in enumerate(r.items()))
        print("variant", variant, "DETECTED" if detected else "MISSED", f"weakest exceeding fraction {weakest[0]:.3g} ({weakest[1]} {weakest[2]})",
              flush=True)
    missed = [v for v, r in report["variants"].items() if not r["detected"]]
    # Criterion: the accurate FP32 CPU implementation (libm transcendental functions). ONNX Runtime's MLAS
    # sigmoid has a small absolute but large relative error for x < -8 (probe below), outside eps_t: its
    # result is recorded, not required (BACKBONE_POLICY §5).
    report["onnxruntime_sigmoid_probe"] = sigmoid_probe()
    inside = report["implementations"]["pytorch_fp32_cpu"]["failures"] == 0
    report["variants_missed"] = missed
    report["result"] = "PASS" if inside and not missed else "FAIL"
    write_json(out / "backbone-precheck.json", report)
    print("precheck", report["result"], "missed", missed or "none")
    if report["result"] != "PASS":
        raise SystemExit(1)


def compare(smoke: bool, dump_dir: Path) -> None:
    torch.set_num_threads(THREADS)
    recipe, runs, out = setup(smoke)
    precision = strict_fp32()
    exp = read_json(out / "backbone-export.json")
    rec_in, data = input_frames(out)
    rec_ref, ref = read_reference(out)
    if rec_ref["inputs_sha256"] != rec_in["sha256"]:
        raise SystemExit("the reference was computed for other inputs")
    per_level = [int(np.prod(s)) for s in SHAPES]
    per_input = 3 * sum(per_level)
    report: dict[str, Any] = {"policy": POLICY, "dump_dir": str(dump_dir), "inputs_sha256": rec_in["sha256"],
                              "reference_sha256": rec_ref["sha256"], "onnx_sha256": exp["onnx_sha256"],
                              "pytorch_compute_precision": precision, "inputs": [], "secondary": [], "variants": {}}
    trt_all: dict[str, list[np.ndarray]] = {}
    passed = True
    for i, label in enumerate(rec_in["labels"]):
        dump = dump_dir / f"input-{i}.f32"
        trt = split_dump(dump)
        trt_all[label] = trt
        row: dict[str, Any] = {"label": label, "dump": str(dump), "dump_sha256": sha256_file(dump), "levels": {}}
        pos = i * per_input
        for k, lv in enumerate(LEVELS):
            n = per_level[k]
            value, bound, sigma = (np.asarray(ref[pos + j * n:pos + (j + 1) * n]).reshape(1, *SHAPES[k]) for j in range(3))
            pos += 3 * n
            row["levels"][lv] = judge(trt[k], value, sigma, bound)
            passed = passed and row["levels"][lv]["failures"] == 0 and row["levels"][lv]["nonfinite"] == 0
        report["inputs"].append(row)
        print(label, {lv: f"fail {v['failures']} ratio {v['max_ratio']:.4f} abs {v['max_abs']:.2e}" for lv, v in row["levels"].items()},
              flush=True)
    # Secondary, pre-fixed T01 tolerances (t01_verify_features.TOLERANCES).
    from t01_verify_features import TOLERANCES  # noqa: PLC0415

    teacher, _ = load_teacher_yolo(recipe, Path(exp["teacher_checkpoint"]))
    device = torch.device("cuda")
    net = Backbone(teacher.to(device)).eval()
    caches = {"MOT17-02-SDP": "cache_eval", "MOT17-10-SDP": "cache_val", "MOT17-13-SDP": "cache_train"}
    with torch.no_grad():
        for i, (seq, fid) in enumerate(FEATURE_FRAMES):
            label = f"{seq}/{fid}"
            pt = [t.cpu().numpy() for t in net(torch.from_numpy(data[i:i + 1].copy()).to(device))]
            cached = torch.load(runs / caches[seq] / "cache" / seq / f"{fid:06d}.pt", map_location="cpu", weights_only=True)
            row = {"label": label, "cache": caches[seq], "levels": {}}
            for k, lv in enumerate(LEVELS):
                c = cached[lv].float().numpy()[None]
                row["levels"][lv] = {"trt_vs_pytorch_fp32_rel_l2": rel_l2(trt_all[label][k], pt[k]),
                                     "cache_vs_trt_rel_l2": rel_l2(c, trt_all[label][k])}
            report["secondary"].append(row)
    worst_pt = max(lv["trt_vs_pytorch_fp32_rel_l2"] for r in report["secondary"] for lv in r["levels"].values())
    worst_cache = max(lv["cache_vs_trt_rel_l2"] for r in report["secondary"] for lv in r["levels"].values())
    report["secondary_checks"] = {
        "fp32_implementation": {"worst": worst_pt, "tol": TOLERANCES["fp32_implementation_rel_l2"],
                                "pass": worst_pt <= TOLERANCES["fp32_implementation_rel_l2"]},
        "train_vs_deploy": {"worst": worst_cache, "tol": TOLERANCES["train_vs_deploy_rel_l2"],
                            "pass": worst_cache <= TOLERANCES["train_vs_deploy_rel_l2"]},
    }
    passed = passed and all(c["pass"] for c in report["secondary_checks"].values())
    report["pytorch_compute_precision_end"] = t01_precision.checked_readback("fp32-strict")
    # Semantic variants (§6): each must exceed the tolerance somewhere against TensorRT as well.
    base, _ = load_teacher_yolo(recipe, None)
    for variant, per_input in variant_refs(teacher.cpu(), base, data, rec_in["labels"]).items():
        entry = {}
        for label, refs in per_input.items():
            lv_refs = level_refs(ref, rec_in["labels"].index(label))
            entry[label] = {lv: {"vs_reference_exceeding": int((np.abs(refs[k].v.numpy() - lv_refs[k][0]) > LAMBDA * lv_refs[k][2] + FLOOR).sum()),
                                 "vs_trt_exceeding": int((np.abs(trt_all[label][k].astype(np.float64) - refs[k].v.numpy())
                                                          > LAMBDA * refs[k].s.numpy() + FLOOR).sum())} for k, lv in enumerate(LEVELS)}
        report["variants"][variant] = {
            "detected_vs_reference": any(lv["vs_reference_exceeding"] > 0 for r in entry.values() for lv in r.values()),
            "detected_vs_trt": any(lv["vs_trt_exceeding"] > 0 for r in entry.values() for lv in r.values()), "inputs": entry}
        print("variant", variant, report["variants"][variant]["detected_vs_reference"], report["variants"][variant]["detected_vs_trt"], flush=True)
    missed = [v for v, r in report["variants"].items() if not (r["detected_vs_reference"] and r["detected_vs_trt"])]
    report["variants_missed"] = missed
    passed = passed and not missed
    report["result"] = "PASS" if passed else "FAIL"
    write_json(out / "backbone-compare.json", report)
    print("secondary", {k: f"{v['worst']:.2e} <= {v['tol']:.2e} {v['pass']}" for k, v in report["secondary_checks"].items()})
    print("variants missed", missed or "none")
    print("result", report["result"])
    if not passed:
        raise SystemExit(1)


HEAD_OUT_SHAPES = ((80, 80, 80), (80, 40, 40), (80, 20, 20), (4, 80, 80), (4, 40, 40), (4, 20, 20))
HEAD_OUT_NAMES = ("cls_p3", "cls_p4", "cls_p5", "reg_p3", "reg_p4", "reg_p5")


def split_head_dump(path: Path) -> list[np.ndarray]:
    raw = np.frombuffer(path.read_bytes(), dtype="<f4")
    out, pos = [], 0
    for shape in HEAD_OUT_SHAPES:
        n = int(np.prod(shape))
        out.append(raw[pos:pos + n].reshape(shape))
        pos += n
    if pos != raw.size:
        raise SystemExit(f"{path}: {raw.size} values, expected {pos}")
    return out


def diagnose(smoke: bool, model_dump: Path, chain_dump: Path) -> None:
    """Diagnostics of §7, not acceptance: the F03 engine's six outputs against the training-side PyTorch model
    (backbone fp32-strict + T1 head) and against the two-stage chain (backbone engine -> head engine)."""
    import t01_eval  # noqa: PLC0415

    recipe, runs, out = setup(smoke)
    precision = strict_fp32()
    exp = read_json(out / "backbone-export.json")
    rec_in, data = input_frames(out)
    sel = read_json(runs / "t1" / "selection.json")
    device = torch.device("cuda")
    teacher, _ = load_teacher_yolo(recipe, Path(exp["teacher_checkpoint"]))
    backbone = Backbone(teacher.to(device)).eval()
    head, _ = t01_eval.build_student(Path(sel["checkpoint"]), device)
    report: dict[str, Any] = {"policy": POLICY.replace("backbone", "f03"), "t1_sha256": sel["sha256"], "inputs_sha256": rec_in["sha256"],
                              "pytorch_compute_precision": precision, "inputs": []}

    def decoded(arrays: list[np.ndarray]) -> dict[str, torch.Tensor]:
        c, r = t01_eval.flatten_levels([torch.from_numpy(np.array(a))[None].to(device) for a in arrays[:3]],
                                       [torch.from_numpy(np.array(a))[None].to(device) for a in arrays[3:]])
        return t01_eval.decode(c, r)[0]

    with torch.no_grad():
        for i, label in enumerate(rec_in["labels"]):
            cls, reg = head(list(backbone(torch.from_numpy(data[i:i + 1].copy()).to(device))))
            pt = [t.cpu().numpy()[0] for t in (*cls, *reg)]
            model = split_head_dump(model_dump / f"input-{i}.f32")
            chain = split_head_dump(chain_dump / f"features-{i}.f32")
            row: dict[str, Any] = {"label": label, "model_dump_sha256": sha256_file(model_dump / f"input-{i}.f32"),
                                   "chain_dump_sha256": sha256_file(chain_dump / f"features-{i}.f32"), "outputs": {}}
            for name, a_pt, a_m, a_c in zip(HEAD_OUT_NAMES, pt, model, chain):
                row["outputs"][name] = {"model_vs_pytorch_rel_l2": rel_l2(a_m, a_pt),
                                        "model_vs_pytorch_max_abs": float(np.abs(a_m.astype(np.float64) - a_pt).max()),
                                        "model_vs_chain_rel_l2": rel_l2(a_m, a_c),
                                        "model_vs_chain_bit_identical": bool(np.array_equal(a_m.view(np.uint32), a_c.view(np.uint32)))}
            d_pt, d_m = decoded(pt), decoded(model)
            row["detections"] = {"pytorch": int(d_pt["boxes"].shape[0]), "model": int(d_m["boxes"].shape[0])}
            if d_pt["boxes"].shape == d_m["boxes"].shape and d_pt["boxes"].numel():
                row["detections"].update({"same_order_max_box_abs_px": float((d_pt["boxes"] - d_m["boxes"]).abs().max()),
                                          "same_order_max_score_abs": float((d_pt["scores"] - d_m["scores"]).abs().max())})
            report["inputs"].append(row)
            print(label, f"model vs pytorch rel L2 <= {max(v['model_vs_pytorch_rel_l2'] for v in row['outputs'].values()):.2e}",
                  f"model vs chain rel L2 <= {max(v['model_vs_chain_rel_l2'] for v in row['outputs'].values()):.2e}", row["detections"], flush=True)
    report["pytorch_compute_precision_end"] = t01_precision.checked_readback("fp32-strict")
    write_json(out / "f03-diagnostics.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=("inputs", "reference", "precheck", "compare", "diagnose"))
    parser.add_argument("paths", nargs="*", type=Path, help="compare: DUMP_DIR; diagnose: MODEL_DUMP_DIR CHAIN_DUMP_DIR")
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    if a.cmd == "diagnose":
        if len(a.paths) != 2:
            parser.error("diagnose needs MODEL_DUMP_DIR CHAIN_DUMP_DIR")
        diagnose(a.smoke, *a.paths)
        return
    if a.cmd == "compare":
        if len(a.paths) != 1:
            parser.error("compare needs DUMP_DIR")
        compare(a.smoke, a.paths[0])
        return
    if a.paths:
        parser.error(f"{a.cmd} takes no paths")
    {"inputs": write_inputs, "reference": write_reference, "precheck": precheck}[a.cmd](a.smoke)


if __name__ == "__main__":
    sys.exit(main())

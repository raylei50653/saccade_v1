"""T01 compute-precision policies: set, read back and check inside the running process.

A policy fixes the FP32 math mode of cuDNN convolutions and cuBLAS matmuls.
`readback` reads the torch flags (legacy and fp32_precision APIs), the
NVIDIA_TF32_OVERRIDE environment variable and, with probe=True, measures the
math actually used: a conv2d and a matmul against float64 on the GPU. FP32
math gives ~2e-6 (conv) / ~2e-7 (matmul) relative error, TF32 ~3e-4, so the
3e-5 threshold separates them by an order of magnitude on either side.

Storage dtypes (fp16 cache files, fp32 checkpoints) are recorded separately by
the callers; this module only covers how values are computed.
"""

from __future__ import annotations

import os
from typing import Any

PROBE_TF32_THRESHOLD = 3e-5

POLICIES: dict[str, dict[str, Any]] = {
    # Feature producers (caches, GT1 live features) and evaluation/export references:
    # the same FP32 math as the deployment engines (S04: FP32, TF32 cleared).
    "fp32-strict": {
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "nvidia_tf32_override": "0",
        "conv_math": "fp32",
        "matmul_math": "fp32",
    },
    # The original recipe's setting (PyTorch 2.11 defaults made explicit): TF32 cuDNN
    # convolutions, FP32 matmuls. The teacher also trains under bf16 autocast.
    "torch-default": {
        "cudnn_allow_tf32": True,
        "cuda_matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "nvidia_tf32_override": None,
        "conv_math": "tf32",
        "matmul_math": "fp32",
    },
}
_PROBED = ("conv_math", "matmul_math")


def apply(policy: str) -> None:
    """Set the policy in this process. Call before the first cuDNN/cuBLAS FP32 op."""
    import torch  # noqa: PLC0415

    want = POLICIES[policy]
    if want["nvidia_tf32_override"] is None:
        os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
    else:
        os.environ["NVIDIA_TF32_OVERRIDE"] = want["nvidia_tf32_override"]
    torch.backends.cudnn.allow_tf32 = want["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = want["cuda_matmul_allow_tf32"]
    torch.set_float32_matmul_precision(want["float32_matmul_precision"])


def _probe() -> dict[str, Any]:
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(4, 64, 32, 32, device="cuda", generator=g)
    w = torch.randn(64, 64, 3, 3, device="cuda", generator=g)
    a = torch.randn(256, 512, device="cuda", generator=g)
    b = torch.randn(512, 256, device="cuda", generator=g)
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        ref_c = F.conv2d(x.double(), w.double(), padding=1)
        ref_m = a.double() @ b.double()
        conv = float((F.conv2d(x, w, padding=1).double() - ref_c).norm() / ref_c.norm())
        mm = float(((a @ b).double() - ref_m).norm() / ref_m.norm())
    return {
        "conv_rel_err_vs_fp64": conv,
        "matmul_rel_err_vs_fp64": mm,
        "conv_math": "tf32" if conv > PROBE_TF32_THRESHOLD else "fp32",
        "matmul_math": "tf32" if mm > PROBE_TF32_THRESHOLD else "fp32",
    }


def readback(probe: bool = True) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    rb: dict[str, Any] = {
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "fp32_precision_api": {
            "generic": torch.backends.fp32_precision,
            "cudnn": torch.backends.cudnn.fp32_precision,
            "cudnn_conv": torch.backends.cudnn.conv.fp32_precision,
            "cuda_matmul": torch.backends.cuda.matmul.fp32_precision,
        },
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if probe:
        rb.update(_probe())
    return rb


def check(policy: str, rb: dict[str, Any]) -> list[str]:
    """Mismatches between a readback and the policy (probe keys only if probed)."""
    return [f"{k}: read {rb[k]!r}, policy {v!r}" for k, v in POLICIES[policy].items()
            if (k not in _PROBED or k in rb) and rb.get(k) != v]


def canonical(policy: str) -> dict[str, Any]:
    """What manifests and consumers compare: the policy name and its fixed settings."""
    return {"policy": policy, **POLICIES[policy]}


def checked_readback(policy: str | None = None, probe: bool = True) -> dict[str, Any]:
    """Read back in this process and raise unless it matches the policy (default: $SACCADE_T01_PRECISION)."""
    policy = policy or os.environ["SACCADE_T01_PRECISION"]
    rb = readback(probe)
    errors = check(policy, rb)
    if errors:
        raise RuntimeError(f"T01 precision policy {policy} not in effect: {errors}")
    return {"policy": policy, **rb}

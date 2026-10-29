"""DualMamba: Dual-Dynamics Mamba block for YOLOv26 NMS-free dual-head detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class DualMambaBlock(nn.Module):
    """Dual-head-aware Mamba block with sharp + smooth parallel branches.

    Sharp branch  (small d_state, large expand): high-freq, discriminative.
                  Designed to serve one-to-one matching head.
    Smooth branch (large d_state, small expand): low-freq, dense.
                  Designed to serve one-to-many matching head.
    Outputs a single fused feature via learnable per-token gating.
    """

    def __init__(self, c1, c2=None,
                 d_state_sharp=8, d_state_smooth=32,
                 d_conv=4,
                 expand_sharp=2, expand_smooth=1):
        super().__init__()
        c2 = c1 if c2 is None else c2
        assert c1 == c2, f"DualMamba requires c1 == c2, got {c1} vs {c2}"

        self.norm = nn.LayerNorm(c1)

        # Sharp branch (serves one-to-one head)
        self.mamba_sharp_fwd = Mamba(d_model=c1, d_state=d_state_sharp,
                                     d_conv=d_conv, expand=expand_sharp)
        self.mamba_sharp_bwd = Mamba(d_model=c1, d_state=d_state_sharp,
                                     d_conv=d_conv, expand=expand_sharp)

        # Smooth branch (serves one-to-many head)
        self.mamba_smooth_fwd = Mamba(d_model=c1, d_state=d_state_smooth,
                                      d_conv=d_conv, expand=expand_smooth)
        self.mamba_smooth_bwd = Mamba(d_model=c1, d_state=d_state_smooth,
                                      d_conv=d_conv, expand=expand_smooth)

        # Per-token learnable gating
        self.gate = nn.Sequential(
            nn.Linear(c1, max(c1 // 4, 16)),
            nn.GELU(),
            nn.Linear(max(c1 // 4, 16), 2),
        )

        self.proj = nn.Linear(c1, c1)

    def forward(self, x):
        # CPU bypass for ultralytics dummy forward during model build
        if not x.is_cuda:
            return x

        B, C, H, W = x.shape
        z = x.flatten(2).transpose(1, 2)         # (B, HW, C)
        zn = self.norm(z)

        # Sharp branch
        y_sharp = self.mamba_sharp_fwd(zn) + self.mamba_sharp_bwd(zn.flip(1)).flip(1)

        # Smooth branch
        y_smooth = self.mamba_smooth_fwd(zn) + self.mamba_smooth_bwd(zn.flip(1)).flip(1)

        # Per-token softmax gating
        gate = F.softmax(self.gate(zn), dim=-1)  # (B, HW, 2)
        w_sharp, w_smooth = gate[..., 0:1], gate[..., 1:2]

        y = w_sharp * y_sharp + w_smooth * y_smooth
        y = self.proj(y) + z                     # residual

        return y.transpose(1, 2).reshape(B, C, H, W)
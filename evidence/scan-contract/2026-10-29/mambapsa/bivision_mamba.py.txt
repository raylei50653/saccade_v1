"""BiVisionMamba block for YOLOv26 neck."""
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class BiVisionMamba(nn.Module):
    """Bidirectional Vision Mamba block. Keeps channel count, adds global context."""

    def __init__(self, c1, c2=None, d_state=16, d_conv=4, expand=2):
        super().__init__()
        c2 = c1 if c2 is None else c2
        assert c1 == c2, f"BiVisionMamba expects c1 == c2, got {c1} vs {c2}"
        self.norm = nn.LayerNorm(c1)
        self.mamba_fwd = Mamba(d_model=c1, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=c1, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(c1, c1)

    def forward(self, x):
        # Ultralytics 在 model build 時會用 CPU dummy input 推 stride，
        # Mamba CUDA kernel 不吃 CPU tensor，這時直接做 identity 通過即可
        # （shape 不變，stride 推算正確；實際訓練/inference 一定在 GPU 上）
        if not x.is_cuda:
            return x

        B, C, H, W = x.shape
        z = x.flatten(2).transpose(1, 2)         # (B, HW, C)
        zn = self.norm(z)
        y_f = self.mamba_fwd(zn)
        y_b = self.mamba_bwd(zn.flip(1)).flip(1)
        y = self.proj(y_f + y_b) + z             # residual
        return y.transpose(1, 2).reshape(B, C, H, W)
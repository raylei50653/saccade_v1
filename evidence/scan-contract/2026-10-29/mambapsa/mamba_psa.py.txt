"""MambaPSA: Drop-in replacement for v26's C2PSA using lightweight Mamba."""
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from ultralytics.nn.modules.conv import Conv


class MambaCore(nn.Module):
    """Lightweight Mamba block (mono-direction, d_state=8, expand=1) for
    use inside MambaPSA. Params-neutral replacement for PSA attention.
    """

    def __init__(self, c, d_state=8, d_conv=4, expand=1):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.mamba = Mamba(d_model=c, d_state=d_state,
                           d_conv=d_conv, expand=expand)
        self.proj = nn.Linear(c, c)

    def forward(self, x):
        # x: (B, C, H, W). CPU bypass for ultralytics dummy forward.
        if not x.is_cuda:
            return x
        B, C, H, W = x.shape
        z = x.flatten(2).transpose(1, 2)        # (B, HW, C)
        y = self.mamba(self.norm(z))
        y = self.proj(y) + z                    # residual
        return y.transpose(1, 2).reshape(B, C, H, W)


class MambaPSA(nn.Module):
    """Mamba-based Position-Sensitive Aggregation.

    Drop-in replacement for ultralytics.nn.modules.block.C2PSA.
    Preserves CSP topology; replaces PSA attention with lightweight Mamba.

    Args:
        c1, c2: input/output channels (must equal)
        e: expansion ratio for hidden channels (default 0.5)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2, f"MambaPSA requires c1 == c2, got {c1} vs {c2}"
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, 2 * c_, 1, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        # n stacked Mamba cores (n=1 by default, matching common C2PSA usage)
        self.m = nn.Sequential(*[MambaCore(c_) for _ in range(n)])

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))
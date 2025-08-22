from __future__ import annotations
import math, itertools
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedProjector(torch.nn.Module):
    """
    Map arbitrary features -> [0,1] with learnable scale/bias and a smooth squash.
    y = sigmoid(exp(log_gain) * x + bias) * (1-2*eps) + eps
    """
    def __init__(self, channels: int, eps: float = 1e-3):
        super().__init__()
        self.log_gain = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.bias     = torch.nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps      = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z  = torch.exp(self.log_gain) * x + self.bias
        y  = torch.sigmoid(z)
        if self.eps > 0:
            y = y * (1 - 2*self.eps) + self.eps
        return y


################################################################################
# 1×1 MLP (implemented with 1×1 convs)                                         #
################################################################################

class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)
    
def create_mlp(
    in_ch: int,
    out_ch: int,
    n_layers: int,
    hidden: int = 16,
    act: nn.Module = Sine(),
) -> nn.Module:
    """MLP realised with 1x1 convolutions (same weights everywhere)."""
    if n_layers == 0:
        return nn.Identity()

    layers: List[nn.Module] = []

    cur_in = in_ch
    for i in range(n_layers):
        cur_out = out_ch if i == n_layers - 1 else hidden
        layers.append(nn.Conv2d(cur_in, cur_out, 1))
        if i != n_layers - 1:
            layers.append(act)
        cur_in = cur_out

    return nn.Sequential(*layers)

# ------------------------------------------------------------------
# A compact UNet that preserves resolution end-to-end
# (You can replace this with your larger UNet implementation)
# ------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class SmallUNet(nn.Module):
    """
    Robust to arbitrary H×W:
      - Encoder downsamples with ceil_mode=True (keeps last row/col when odd).
      - Decoder upsamples to the *exact* target size (no scale_factor ambiguities).
      - Skip path stays at full res; we only 1x1-project channels (no stride).
    """
    def __init__(self, in_ch, out_ch, base=32):
        super().__init__()
        F = base
        # encoder
        self.enc1 = ConvBlock(in_ch, F)             # -> [B,F,H,W]
        self.pool = nn.AvgPool2d(2, ceil_mode=True) # -> [B,F,ceil(H/2),ceil(W/2)]
        self.enc2 = ConvBlock(F, 2*F)               # -> [B,2F,···]

        # bottleneck
        self.bot  = ConvBlock(2*F, 2*F)             # -> [B,2F,···]

        # skip projection (no spatial change)
        self.skip_proj = nn.Conv2d(F, F, kernel_size=1)

        # decoder
        self.dec1 = ConvBlock(2*F + F, F)           # concat(upsampled, skip)
        self.out  = nn.Conv2d(F, out_ch, 3, padding=1)

    def forward(self, x):
        s1 = self.enc1(x)                                  # [B,F,H,W]
        x2 = self.pool(s1)                                 # [B,F,ceil(H/2),ceil(W/2)]
        x2 = self.enc2(x2)                                 # [B,2F,···]
        b  = self.bot(x2)                                  # [B,2F,···]

        # upsample decoder activations to match s1 exactly
        up   = torch.nn.functional.interpolate(
            b, size=s1.shape[2:], mode='bilinear', align_corners=False
        )                                                  # [B,2F,H,W]

        skip = self.skip_proj(s1)                          # [B,F,H,W]
        x    = torch.cat([up, skip], dim=1)                # [B,2F+F,H,W]
        x    = self.dec1(x)                                # [B,F,H,W]
        return self.out(x)                                 # [B,out_ch,H,W]
    
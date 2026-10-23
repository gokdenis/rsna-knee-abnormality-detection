"""2.5D window encoder with per-finding attention pooling over a study's windows."""

import numpy as np
import timm
import torch
import torch.nn as nn

from .preprocess import LAYOUTS

N_SLOTS = 5


def window_centers(layout: str = "v1") -> tuple[np.ndarray, np.ndarray]:
    """Centre slice index and slot id of every 3-slice window that stays inside one slot."""
    lay = LAYOUTS[layout]
    centers, slots = [], []
    for s, ((_, _, k), off) in enumerate(zip(lay.slots, lay.offsets)):
        for c in range(off + 1, off + k - 1):
            centers.append(c)
            slots.append(s)
    return np.array(centers), np.array(slots)


class KneeMIL(nn.Module):
    """Shared 2D backbone on 3-slice windows; one attention distribution per finding.

    Input: x [B, K, 3, H, W], slot [B, K] (long), valid [B, K] (bool).
    Output: logits [B, n_labels].
    """

    def __init__(self, arch: str, n_labels: int = 12, pretrained: bool = True, drop: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg")
        dim = self.backbone.num_features
        self.slot_embed = nn.Embedding(N_SLOTS, dim)
        nn.init.zeros_(self.slot_embed.weight)
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n_labels))
        self.cls_w = nn.Parameter(torch.zeros(n_labels, dim))
        self.cls_b = nn.Parameter(torch.zeros(n_labels))
        nn.init.trunc_normal_(self.cls_w, std=0.02)

    def forward(self, x, slot, valid):
        b, k = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1)).view(b, k, -1)
        h = self.norm(feats + self.slot_embed(slot))
        scores = self.att(h).float()                                   # [B, K, L]
        scores = scores.masked_fill(~valid[..., None], float("-inf"))
        weights = torch.softmax(scores, dim=1)                          # over windows
        pooled = torch.einsum("bkl,bkd->bld", weights, h.float())       # [B, L, D]
        return (pooled * self.cls_w).sum(-1) + self.cls_b

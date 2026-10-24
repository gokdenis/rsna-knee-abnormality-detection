"""Inference for the public Raptor CoAtNet weights (dreaddevelopment/raptor-knee-finespacing).

The weights were trained on competition data only. Their preprocessing differs from ours and
is reproduced here exactly as published: 80 slices over 2-98% of each series, 140 mm crop
(capped at the image size), 336 px, windows over the whole stack, CoAtNet at 384 px.
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydicom.pixel_data_handlers.util import apply_modality_lut

IMG = 336
RES = 384
CROP_MM = 140.0
SPAN = (0.02, 0.98)
SLOTS = [("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15), ("Coronal", 0, 10), ("Axial", -1, 15)]
MAXS = sum(k for _, _, k in SLOTS)  # 80
K_EVAL = 78
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, dim, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, dim))
        self.clsb = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        b, k = x.shape[:2]
        h = self.norm(self.backbone(x.flatten(0, 1)).view(b, k, -1))
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def load_raptor(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k")
    bb = timm.create_model(arch, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    model = RaptorClassifier(bb, bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval()


def _order(series_dir):
    recs, spacings = [], []
    for f in glob.glob(series_dir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            spacings.append(ps)
            recs.append((pos, f, ps))
        except Exception:
            recs.append((0.0, f, 0.5))
    recs.sort(key=lambda r: r[0])
    return [(f, ps) for _, f, ps in recs], float(np.median(spacings)) if spacings else 0.5


def _read(f):
    try:
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
            a = a.max() - a
        return a
    except Exception:
        return None


def _crop(a, ps):
    h, w = a.shape
    c = min(int(round(CROP_MM / max(ps, 1e-3))), min(h, w))
    y0, x0 = (h - c) // 2, (w - c) // 2
    return cv2.resize(a[y0:y0 + c, x0:x0 + c], (IMG, IMG), interpolation=cv2.INTER_AREA)


def _pick(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_raptor_study(study_dir, rows, threads=8):
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx, used = 0, set()
    with ThreadPoolExecutor(threads) as pool:
        for plane, fluid, k in SLOTS:
            r = _pick(rows, plane, fluid, used)
            if r is None:
                idx += k
                continue
            used.add(r["SeriesInstanceUID"])
            files, med_ps = _order(os.path.join(study_dir, r["SeriesInstanceUID"]))
            if not files:
                idx += k
                continue
            n = len(files)
            lo, hi = int(n * SPAN[0]), int(n * SPAN[1]) - 1
            hi = max(hi, lo)
            picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
            sel = [files[min(p, n - 1)] for p in picks]
            arrs = list(pool.map(_read, [f for f, _ in sel]))
            valid = [a for a in arrs if a is not None]
            if valid:
                loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
            else:
                loq, hiq = 0.0, 1.0
            for a, (_, ps) in zip(arrs, sel):
                if idx >= MAXS:
                    break
                if a is not None:
                    aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
                    vol[idx] = (_crop(aw, ps if ps > 0 else med_ps) * 255).astype(np.uint8)
                idx += 1
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


def raptor_centers(mask):
    """K_EVAL window centres evenly spaced over the filled part of the stack (as published)."""
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, MAXS - 2))]
    return np.array([cs[i] for i in np.linspace(0, len(cs) - 1, K_EVAL).round().astype(int)])


def raptor_windows(vol: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """vol: uint8 [MAXS, IMG, IMG] on the target device -> normalised [1, K_EVAL, 3, RES, RES]."""
    c = torch.as_tensor(raptor_centers(mask), device=vol.device)
    tri = torch.stack([vol[c - 1], vol[c], vol[c + 1]], dim=1).float() / 255.0
    x = F.interpolate(tri, size=(RES, RES), mode="bilinear", align_corners=False)
    x = (x - MEAN.to(vol.device)) / STD.to(vol.device)
    return x[None]

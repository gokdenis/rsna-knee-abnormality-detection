"""The 96-slice CoAtNet layout of nartaa's public OAI checkpoint (0.949), for fine-tuning and inference.

Reproduces the published preprocessing exactly (rsna-knee-0949-anatomical-mirror): five fixed series
slots (26 + 22 sagittal, 18 + 12 coronal, 18 axial = 96 slices) over 2-98% of each series, 2-98
percentile windowing per series, 140 mm centre crop (capped at the image size) resized to 384 px,
3-slice windows centre-cropped to 320 px for a CoAtNet-RMLP-2 with per-finding attention pooling.

The series picking is factored out so that OAI studies (no train_series.csv, different sequence
names) can fill the same slots.
"""

import glob
import io
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import pydicom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from pydicom.pixel_data_handlers.util import apply_modality_lut

IMG = 384          # stored slice size
RES = 320          # model input: centre crop of IMG (random crop when training)
CROP_MM = 140.0
SPAN = (0.02, 0.98)
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18), ("Coronal", 0, 12), ("Axial", -1, 18)]
SLOT_SIZES = [k for _, _, k in SLOTS]
SAGITTAL_SLOTS = (0, 1)
MAXS = sum(SLOT_SIZES)  # 96
K_EVAL = 94
ARCH = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
ARCH_224 = "coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- model

class CoatMIL(nn.Module):
    """CoAtNet backbone + per-finding attention pooling over windows (same parameter names as the release)."""

    def __init__(self, backbone, dim, n=12, drop=0.2):
        super().__init__()
        self.att_temp = 1.0  # attention softmax temperature over windows (1.0 = published model)
        self.backbone = backbone
        self.norm = nn.LayerNorm(dim)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)

    def forward(self, x):
        b, k = x.shape[:2]
        flat = x.flatten(0, 1)
        f = torch.cat([self.backbone(chunk) for chunk in flat.split(16)], dim=0)
        h = self.norm(f.view(b, k, -1))
        a = torch.softmax(self.att(h) / self.att_temp, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


@dataclass(frozen=True)
class View:
    """How a model sees the stored 384 px windows: crop `crop` px (centre at eval), resize to `res`."""
    name: str
    res: int
    crop: int
    mirror_tta: bool
    ck_res: int  # the "res" value stored in checkpoints of this view


VIEWS = {
    "r320": View("r320", RES, RES, True, 384),                         # the 0.949 checkpoint
    "c08r224": View("c08r224", 224, int(round(IMG * 0.8)), False, 224),  # the 0.945 checkpoint (0.8 crop -> 224)
}


def view_of(ck):
    return VIEWS["c08r224"] if int(ck.get("res", 384)) == 224 else VIEWS["r320"]


def build_model(arch=ARCH, pretrained=False, res=RES):
    bb = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=3, img_size=res, global_pool="avg")
    return CoatMIL(bb, bb.num_features)


def load_model_view(path, device="cpu"):
    """Model and its View from a release-format checkpoint ({"model", "arch", "res"})."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    view = view_of(ck)
    model = build_model(ck.get("arch", ARCH), res=view.res)
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device), view


def load_model(path, device="cpu"):
    return load_model_view(path, device)[0]


# ---------------------------------------------------------------- DICOM -> volume

def order_series(series_dir, pattern="*.dcm"):
    """Files of one series sorted along the slice normal, with their pixel spacing, and the median spacing."""
    recs, spacings = [], []
    for f in glob.glob(os.path.join(series_dir, pattern)):
        if os.path.isdir(f):
            continue
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
        if isinstance(f, (bytes, bytearray)):  # in-memory DICOM: a fresh stream per read (a slice may be picked twice)
            f = io.BytesIO(f)
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


def build_volume(slot_series, threads=8):
    """slot_series: one entry per SLOTS item, each None or (files_sorted_with_spacing, median_spacing).

    Returns uint8 [MAXS, IMG, IMG] and the filled-slice mask, as the published build_study does.
    """
    vol = np.zeros((MAXS, IMG, IMG), np.uint8)
    idx = 0
    with ThreadPoolExecutor(threads) as pool:
        for (_, _, k), entry in zip(SLOTS, slot_series):
            if entry is None or not entry[0]:
                idx += k
                continue
            files, med_ps = entry
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


def pick_competition_series(rows):
    """Series row per slot from train_series.csv rows, as published (prefer the slot's fluid flag)."""
    out, used = [], set()
    for plane, fluid, _ in SLOTS:
        cands = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid] if fluid in (0, 1) else []
        r = (pref or cands or [None])[0]
        if r is not None:
            used.add(r["SeriesInstanceUID"])
        out.append(r)
    return out


def pick_alternative_series(rows):
    """Second view of a study: per slot an unused series with the same plane / fluid / fat flags as the primary pick
    (first in CSV order), else the primary itself. None when no slot has such an alternative."""
    prim = pick_competition_series(rows)
    used = {r["SeriesInstanceUID"] for r in prim if r is not None}
    out, any_alt = [], False
    for r in prim:
        alt = None
        if r is not None:
            alt = next((c for c in rows if c["SeriesInstanceUID"] not in used and
                        all(str(c.get(k)) == str(r.get(k)) for k in ("Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"))),
                       None)
        if alt is not None:
            used.add(alt["SeriesInstanceUID"])
            any_alt = True
        out.append(alt or r)
    return out if any_alt else None


def build_competition_study_alt(study_dir, rows, threads=8):
    """Volume from the alternative picks (pick_alternative_series), or None when the study has no alternative."""
    picks = pick_alternative_series(rows)
    if picks is None:
        return None
    slot_series = [None if r is None else order_series(os.path.join(study_dir, r["SeriesInstanceUID"])) for r in picks]
    return build_volume(slot_series, threads)


def build_competition_study(study_dir, rows, threads=8):
    slot_series = []
    for r in pick_competition_series(rows):
        slot_series.append(None if r is None else order_series(os.path.join(study_dir, r["SeriesInstanceUID"])))
    return build_volume(slot_series, threads)


# ---------------------------------------------------------------- windows

def slot_of_slice():
    return np.repeat(np.arange(len(SLOT_SIZES)), SLOT_SIZES)


WINDOW_FILTER = "none"  # evaluation experiment: "slot" drops windows spanning two slots, "slot+empty" also empty slices


def eval_centers(mask, k=K_EVAL):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = list(range(lo + 1, hi)) or [max(1, min((lo + hi) // 2, MAXS - 2))]
    out = np.array([cs[i] for i in np.linspace(0, len(cs) - 1, k).round().astype(int)])
    if WINDOW_FILTER != "none":
        slot = slot_of_slice()
        keep = slot[out - 1] == slot[out + 1]
        if WINDOW_FILTER == "slot+empty":
            keep &= (mask[out - 1] > 0) & (mask[out] > 0) & (mask[out + 1] > 0)
        if keep.sum() >= 3:
            out = out[keep]
    return out


def train_centers(mask, k, rng):
    valid = np.where(mask > 0)[0]
    lo, hi = (int(valid.min()), int(valid.max())) if len(valid) >= 3 else (0, 2)
    pool = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([1])
    return np.sort(rng.choice(pool, size=k, replace=len(pool) < k))


def mirror(tri, centers):
    """True left/right mirror of uint8 windows [..., K, 3, H, W]: sagittal windows reverse their slice
    channels, coronal/axial windows flip the width (the published anatomical-mirror TTA)."""
    sag = torch.from_numpy(np.isin(slot_of_slice()[np.asarray(centers)], SAGITTAL_SLOTS)).to(tri.device)
    sag = sag.view(*([1] * (tri.dim() - 4)), -1, 1, 1, 1)
    return torch.where(sag, tri.flip(-3), tri.flip(-1))


def to_input(tri_uint8, device, crop_xy=None, view=VIEWS["r320"]):
    """uint8 [B, K, 3, IMG, IMG] -> normalised float [B, K, 3, view.res, view.res].

    Crops view.crop px (centred unless crop_xy=(y0, x0)), resizes to view.res when they differ (before the
    normalisation, as the published 224 script does), then applies the ImageNet normalisation.
    """
    c = view.crop
    y0, x0 = crop_xy if crop_xy is not None else ((IMG - c) // 2, (IMG - c) // 2)
    x = tri_uint8[..., y0:y0 + c, x0:x0 + c].to(device, non_blocking=True)
    b, k = x.shape[:2]
    x = x.flatten(0, 1).float() / 255.0
    if view.res != c:
        x = F.interpolate(x, size=(view.res, view.res), mode="bilinear", align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    return x.view(b, k, 3, view.res, view.res)

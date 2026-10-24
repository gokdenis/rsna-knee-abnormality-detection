"""DICOM study -> fixed-size uint8 slice stack.

The same function builds the training cache and the test-time input, so training and
inference see identical pixels.

Layout: five series slots, filled in a fixed order. Each slot takes `k` slices spread evenly
over the central SPAN of the physically sorted series, windowed by series percentiles and
cropped to a CROP_MM square around the image centre, then resized to IMG pixels.
"""

import glob
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

from dataclasses import dataclass

CROP_MM = 140.0
PCT = (2.0, 98.0)


@dataclass(frozen=True)
class Layout:
    img: int
    span: tuple
    slots: tuple  # (plane, preferred Fluid_Sensitive or None, slices)
    jpeg: int = 0  # >0: slices are stored as JPEG at this quality (and round-tripped at test time)

    @property
    def n_slices(self) -> int:
        return sum(k for _, _, k in self.slots)

    @property
    def offsets(self) -> np.ndarray:
        return np.cumsum([0] + [k for _, _, k in self.slots])[:-1]


LAYOUTS = {
    # 64 slices over the central 88% of each series.
    "v1": Layout(224, (0.06, 0.94), (("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
                                      ("Coronal", 0, 8), ("Axial", None, 12))),
    # Denser and wider: 80 slices over 2-98% (Raptor's best-measured sampling).
    "v2": Layout(224, (0.02, 0.98), (("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15),
                                      ("Coronal", 0, 10), ("Axial", None, 15))),
    # v2 sampling at 320 px (0.44 mm/px), JPEG-stored to fit Kaggle's output limit.
    "v3": Layout(320, (0.02, 0.98), (("Sagittal", 1, 22), ("Sagittal", 0, 18), ("Coronal", 1, 15),
                                      ("Coronal", 0, 10), ("Axial", None, 15)), jpeg=95),
}


def encode_slice(sl: np.ndarray, quality: int) -> bytes:
    return cv2.imencode(".jpg", sl, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def decode_slice(buf) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_GRAYSCALE)


def jpeg_roundtrip(vol: np.ndarray, quality: int) -> np.ndarray:
    """Apply the cache's JPEG encoding to a freshly built stack so test pixels match training."""
    return np.stack([decode_slice(encode_slice(s, quality)) for s in vol])

# Defaults (v1) kept as module constants for checkpoints that predate layouts.
IMG = LAYOUTS["v1"].img
SLOTS = list(LAYOUTS["v1"].slots)
N_SLICES = LAYOUTS["v1"].n_slices
SLOT_OFFSETS = LAYOUTS["v1"].offsets


def _slice_position(ds) -> float:
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is not None and ipp is not None and len(iop) == 6:
        normal = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
        return float(np.dot(np.asarray(ipp, float), normal))
    loc = getattr(ds, "SliceLocation", None)
    if loc is not None:
        return float(loc)
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def _read_header(path: str):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        spacing = getattr(ds, "PixelSpacing", None)
        return _slice_position(ds), path, float(spacing[0]) if spacing is not None else 0.0
    except Exception:
        return None


def _read_pixels(path: str):
    try:
        ds = pydicom.dcmread(path)
        arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32)
        if arr.ndim == 3:  # multi-frame or colour: keep the first plane
            arr = arr[0] if arr.shape[0] < arr.shape[-1] else arr[..., 0]
        if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
            arr = arr.max() - arr
        return arr
    except Exception:
        return None


def _crop_resize(arr: np.ndarray, spacing: float, img: int) -> np.ndarray:
    """Centre square of CROP_MM (zero-padded if the field of view is smaller), resized to IMG."""
    h, w = arr.shape
    side = int(round(CROP_MM / spacing)) if spacing > 0 else min(h, w)
    side = max(side, 8)
    out = np.zeros((side, side), np.float32)
    sy, sx = max(0, (h - side) // 2), max(0, (w - side) // 2)
    dy, dx = max(0, (side - h) // 2), max(0, (side - w) // 2)
    ch, cw = min(h, side), min(w, side)
    out[dy:dy + ch, dx:dx + cw] = arr[sy:sy + ch, sx:sx + cw]
    return cv2.resize(out, (img, img), interpolation=cv2.INTER_AREA)


def pick_series(series_rows: list[dict], series_sizes: dict[str, int], slots=SLOTS) -> list[dict | None]:
    """Assign one series per slot: right plane, preferred fluid flag, most slices; no reuse."""
    used, picks = set(), []
    for plane, fluid, _ in slots:
        cands = [r for r in series_rows
                 if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
        if fluid is not None:
            preferred = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
            cands = preferred or cands
        cands.sort(key=lambda r: -series_sizes.get(r["SeriesInstanceUID"], 0))
        choice = cands[0] if cands else None
        if choice is not None:
            used.add(choice["SeriesInstanceUID"])
        picks.append(choice)
    return picks


def _memo(fn, cache):
    """Wrap a per-file reader so repeated reads of the same path (across layouts) hit memory."""
    if cache is None:
        return fn

    def wrapped(path):
        key = (fn.__name__, path)
        if key not in cache:
            cache[key] = fn(path)
        return cache[key]
    return wrapped


def build_study(study_dir: str, series_rows: list[dict], threads: int = 8, layout: str = "v1",
                apply_jpeg: bool = True, cache: dict | None = None):
    """Return (vol uint8 [n_slices, img, img], mask uint8 [n_slices]) for one study.

    For JPEG layouts the stack is JPEG round-tripped once (test time). The cache builder passes
    apply_jpeg=False and encodes the original pixels itself, so both sides decode the same bytes.
    Pass the same `cache` dict when building several layouts of one study to read each file once.
    """
    read_header, read_pixels = _memo(_read_header, cache), _memo(_read_pixels, cache)
    lay = LAYOUTS[layout]
    vol = np.zeros((lay.n_slices, lay.img, lay.img), np.uint8)
    mask = np.zeros(lay.n_slices, np.uint8)
    files = {r["SeriesInstanceUID"]: sorted(glob.glob(os.path.join(study_dir, r["SeriesInstanceUID"], "*.dcm")))
             for r in series_rows}
    picks = pick_series(series_rows, {k: len(v) for k, v in files.items()}, lay.slots)

    with ThreadPoolExecutor(threads) as pool:
        for (_, _, k), offset, row in zip(lay.slots, lay.offsets, picks):
            if row is None or not files[row["SeriesInstanceUID"]]:
                continue
            headers = [h for h in pool.map(read_header, files[row["SeriesInstanceUID"]]) if h]
            if not headers:
                continue
            headers.sort(key=lambda h: h[0])
            n = len(headers)
            lo, hi = int(n * lay.span[0]), max(int(n * lay.span[0]), int(n * lay.span[1]) - 1)
            idx = np.linspace(lo, hi, k).round().astype(int)
            uniq = sorted(set(idx.tolist()))
            pixels = dict(zip(uniq, pool.map(read_pixels, [headers[i][1] for i in uniq])))
            valid = [a for a in pixels.values() if a is not None]
            if not valid:
                continue
            sample = np.concatenate([a.ravel()[::7] for a in valid])
            p_lo, p_hi = np.percentile(sample, PCT)
            spacings = [headers[i][2] for i in uniq if headers[i][2] > 0]
            fallback_spacing = float(np.median(spacings)) if spacings else 0.0
            for j, i in enumerate(idx):
                arr = pixels[i]
                if arr is None:
                    continue
                arr = np.clip((arr - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
                spacing = headers[i][2] or fallback_spacing
                vol[offset + j] = (_crop_resize(arr, spacing, lay.img) * 255).astype(np.uint8)
                mask[offset + j] = 1
    if lay.jpeg and apply_jpeg:
        vol = jpeg_roundtrip(vol, lay.jpeg)
    return vol, mask

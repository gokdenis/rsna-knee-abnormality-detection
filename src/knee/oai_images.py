"""Find OAI knee MRI series on disk and map them to the coat96 slots.

OAI image releases store one folder per series (DICOM files, often without an extension). The
sequence and the knee side are read from the DICOM SeriesDescription, e.g. "SAG_IW_TSE_RIGHT",
"SAG_3D_DESS_WE_LEFT", "COR_MPR_RIGHT", "AX_MPR_LEFT", "COR_IW_TSE_RIGHT", "COR_T1_3D_FLASH_RIGHT".

Slot mapping (competition slot -> closest OAI sequence, available for both knees):
    Sagittal, fluid-sensitive  -> SAG IW TSE FS
    Sagittal, other            -> SAG 3D DESS WE
    Coronal, fluid-sensitive   -> COR IW TSE
    Coronal, other             -> COR MPR (DESS reformat), else COR T1 3D FLASH
    Axial                      -> AX MPR (DESS reformat)
"""

import io
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pydicom

from . import coat96

SLOT_SEQUENCES = [["SAG_IW_TSE"], ["SAG_DESS"], ["COR_IW_TSE"], ["COR_MPR", "COR_FLASH"], ["AX_MPR"]]
assert len(SLOT_SEQUENCES) == len(coat96.SLOTS)


def classify(description: str):
    """(sequence, side) from a SeriesDescription; sequence None when the series is not used."""
    d = description.upper().replace(" ", "_")
    side = "R" if "RIGHT" in d else ("L" if "LEFT" in d else None)
    if d.startswith("SAG") and "DESS" in d:
        seq = "SAG_DESS"
    elif d.startswith("SAG") and "IW" in d and "TSE" in d:
        seq = "SAG_IW_TSE"
    elif d.startswith("COR") and "IW" in d and "TSE" in d:
        seq = "COR_IW_TSE"
    elif d.startswith("COR") and "MPR" in d:
        seq = "COR_MPR"
    elif d.startswith("COR") and "FLASH" in d:
        seq = "COR_FLASH"
    elif d.startswith("AX") and "MPR" in d:
        seq = "AX_MPR"
    else:
        seq = None
    return seq, side


def _first_file(folder):
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and not name.startswith("."):
            return path
    return None


def _describe(folder):
    try:
        f = _first_file(folder)
    except OSError:  # folder vanished (still being unpacked)
        return None
    if f is None:
        return None
    try:
        h = pydicom.dcmread(f, stop_before_pixels=True)
    except Exception:
        return None
    desc = str(getattr(h, "SeriesDescription", ""))
    seq, side = classify(desc)
    return {"folder": folder, "ID": str(getattr(h, "PatientID", "")).strip(), "description": desc,
            "sequence": seq, "SIDE": side, "date": str(getattr(h, "StudyDate", "")),
            "n_files": sum(1 for n in os.listdir(folder) if not n.startswith("."))}


def scan(root, threads=32):
    """One row per series folder under root (folders that directly contain files)."""
    folders = [d for d, _, files in os.walk(root)
               if not d.endswith(".tmp") and any(not f.startswith(".") for f in files)]
    with ThreadPoolExecutor(threads) as pool:
        rows = [r for r in pool.map(_describe, folders) if r is not None]
    return pd.DataFrame(rows)


def knee_series(series: pd.DataFrame):
    """{(ID, SIDE): [folder or None per slot]} using the slot priority above (largest series wins ties)."""
    out = {}
    usable = series.dropna(subset=["sequence", "SIDE"]).sort_values("n_files", ascending=False)
    for (pid, side), grp in usable.groupby(["ID", "SIDE"]):
        by_seq = grp.groupby("sequence")["folder"].first().to_dict()
        out[(pid, side)] = [next((by_seq[s] for s in seqs if s in by_seq), None) for seqs in SLOT_SEQUENCES]
    return out


def build_knee(slot_folders, threads=8):
    slot_series = [None if f is None else coat96.order_series(f, pattern="*") for f in slot_folders]
    return coat96.build_volume(slot_series, threads)


def order_series_tar(tar_path):
    """Like coat96.order_series for a series stored as one .tar.gz (read in memory, nothing unpacked)."""
    import numpy as np
    with tarfile.open(tar_path) as tar:
        blobs = [tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()]
    recs, spacings = [], []
    for b in blobs:
        try:
            h = pydicom.dcmread(io.BytesIO(b), stop_before_pixels=True)
            iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None)
            ps = float(ps[0]) if ps is not None else 0.5
            spacings.append(ps)
            recs.append((pos, b, ps))
        except Exception:
            recs.append((0.0, b, 0.5))
    recs.sort(key=lambda r: r[0])
    # raw bytes, not a shared BytesIO: build_volume may pick the same slice twice and reads in threads
    return [(b, ps) for _, b, ps in recs], float(np.median(spacings)) if spacings else 0.5


def build_knee_tar(slot_tars, threads=4):
    """Volume of one knee from one .tar.gz per slot (None for a missing slot)."""
    slot_series = [None if t is None else order_series_tar(t) for t in slot_tars]
    return coat96.build_volume(slot_series, threads)

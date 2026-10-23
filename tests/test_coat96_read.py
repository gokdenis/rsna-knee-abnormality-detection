"""Regression: an in-memory DICOM slice picked several times (short series) must fill every pick.

Run: .venv/bin/python -I tests/test_coat96_read.py
"""

import io
import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import coat96  # noqa: E402


def _dicom_bytes():
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows = ds.Columns = 64
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation, ds.SamplesPerPixel = 15, 0, 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelSpacing = [0.5, 0.5]
    ds.PixelData = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) + 1).tobytes()
    buf = io.BytesIO()
    pydicom.dcmwrite(buf, ds, enforce_file_format=True)
    return buf.getvalue()


def test_repeated_in_memory_slice_fills_every_pick():
    entry = ([(_dicom_bytes(), 0.5)], 0.5)
    _, mask = coat96.build_volume([entry] + [None] * (len(coat96.SLOTS) - 1))
    k = coat96.SLOTS[0][2]
    assert mask[:k].sum() == k, mask[:k]


if __name__ == "__main__":
    test_repeated_in_memory_slice_fills_every_pick()
    print("ok")

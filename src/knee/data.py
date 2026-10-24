"""Study-level dataset of 2.5D windows read from the uint8 slice cache."""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import window_centers
from .preprocess import LAYOUTS, decode_slice

CENTERS, CENTER_SLOTS = window_centers()  # v1 defaults


def study_windows(vol: np.ndarray, mask: np.ndarray, pick: np.ndarray, layout: str = "v1"):
    """Windows for the chosen window indices: uint8 [K, 3, H, W], slot ids [K], valid flags [K]."""
    centers, center_slots = window_centers(layout)
    c = centers[pick]
    x = np.stack([vol[c - 1], vol[c], vol[c + 1]], axis=1)
    valid = (mask[c - 1] & mask[c] & mask[c + 1]).astype(bool)
    return x, center_slots[pick], valid


class SliceStore:
    """Read slices of a study from either cache format.

    npy:  vols.npy  uint8 [N, S, H, W] (memory-mapped)
    jpeg: blobs.bin concatenated JPEG slices + offsets.npy int64 [N, S + 1] byte offsets
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.jpeg = os.path.exists(os.path.join(cache_dir, "blobs.bin"))
        self._data = None

    def _open(self):
        if self._data is None:
            if self.jpeg:
                self._data = (np.memmap(os.path.join(self.cache_dir, "blobs.bin"), np.uint8, mode="r"),
                              np.load(os.path.join(self.cache_dir, "offsets.npy")))
            else:
                self._data = np.load(os.path.join(self.cache_dir, "vols.npy"), mmap_mode="r")

    def study(self, row: int, needed: np.ndarray, n_slices: int, img: int) -> np.ndarray:
        """uint8 [n_slices, img, img]; only `needed` slices are guaranteed to be filled."""
        self._open()
        if not self.jpeg:
            return self._data[row]
        blobs, offsets = self._data
        vol = np.zeros((n_slices, img, img), np.uint8)
        for i in needed:
            a, b = offsets[row, i], offsets[row, i + 1]
            if b > a:
                vol[i] = decode_slice(blobs[a:b])
        return vol


class StudyDataset(Dataset):
    """Returns (windows uint8 [K,3,H,W], slot [K], valid [K], target [L]).

    Train mode samples k windows among the valid ones; eval mode returns every window.
    """

    def __init__(self, cache_dir: str, masks, rows, targets, k: int | None, train: bool, layout: str = "v1"):
        self.store, self.masks = SliceStore(cache_dir), masks
        self.layout = layout
        self.lay = LAYOUTS[layout]
        self.centers, _ = window_centers(layout)
        self.rows, self.targets = np.asarray(rows), np.asarray(targets, np.float32)
        self.k, self.train = k, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        mask = self.masks[row]
        all_idx = np.arange(len(self.centers))
        if self.train:
            c = self.centers
            ok = np.flatnonzero(mask[c - 1] & mask[c] & mask[c + 1])
            pool = ok if len(ok) else all_idx
            # torch's RNG differs per worker and epoch; a fixed numpy RNG would repeat across workers
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            pick = rng.choice(pool, size=self.k, replace=len(pool) < self.k)
        else:
            pick = all_idx
        c = self.centers[pick]
        needed = np.unique(np.concatenate([c - 1, c, c + 1]))
        vol = self.store.study(row, needed, self.lay.n_slices, self.lay.img)
        x, slot, valid = study_windows(vol, mask, pick, self.layout)
        if not valid.any():
            valid[:] = True  # empty study: attend uniformly over blanks rather than produce NaN
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(slot).long(),
                torch.from_numpy(valid), torch.from_numpy(self.targets[i]))


def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """Light per-window affine + intensity jitter on float input [N, 3, H, W] in [0, 1]. No flips."""
    n = x.shape[0]
    dev = x.device
    angle = (torch.rand(n, device=dev) - 0.5) * (2 * np.pi * 10 / 360)
    scale = 1 + (torch.rand(n, device=dev) - 0.5) * 0.2
    shift = (torch.rand(n, 2, device=dev) - 0.5) * 0.12
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1),
                         torch.stack([sin, cos, shift[:, 1]], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    gain = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.3
    gamma = torch.exp((torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4)
    return (x.clamp(0, 1) ** gamma * gain).clamp(0, 1)

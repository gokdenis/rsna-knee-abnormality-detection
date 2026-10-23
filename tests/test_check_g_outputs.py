"""Synthetic positive / negative cases for scripts/check_g_outputs.py (no GPU, no Kaggle).

Run: .venv/bin/python -I tests/test_check_g_outputs.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knee.constants import LABELS  # noqa: E402

MEMBERS = ["g1", "g2", "g3", "g5", "g6"]
STEMS = ["coat96_g123_soup", "coat96_soup5_swa"]


def make(d, members=MEMBERS, stems=STEMS, json_stems=None, rows=4, preds=None, labels=LABELS):
    up = d / "upload"
    up.mkdir(parents=True)
    for t in set(MEMBERS) | {"g0"}:
        torch.save({"model": {}}, up / f"coat96_{t}_swa.pt")
    torch.save({"model": {}, "swa_of": [f"/workspace/runs/{t}/coat96_{t}_swa.pt" for t in members]},
               up / "coat96_soup5_swa.pt")
    (d / "logs").mkdir()
    res = {f"/workspace/runs/x/{s}.pt": {"gold": 0.92, "oai": 0.93} for s in (stems if json_stems is None else json_stems)}
    json.dump(res, open(d / "logs/eval_soup5.json", "w"))
    z = {"labels": np.array(labels)}
    for s in ("gold", "oai"):
        z[f"{s}__id"] = np.array([f"id{i}" for i in range(rows)])
        z[f"{s}__target"] = np.full((rows, len(LABELS)), np.nan, np.float32)
        for st in stems:
            z[f"{s}__{st}"] = np.full((rows, len(LABELS)), 0.5, np.float32) if preds is None else preds(rows)
    np.savez(d / "logs/eval_soup5_preds.npz", **z)


def run(d):
    cmd = [sys.executable, "-I", str(ROOT / "scripts/check_g_outputs.py"), "--dir", str(d), "--soup", "coat96_soup5_swa.pt",
           "--members", *MEMBERS, "--files", "g0", "--eval", "logs/eval_soup5.json=" + ",".join(STEMS)]
    return subprocess.run(cmd, capture_output=True, text=True).returncode


CASES = {
    "full positive": (dict(), 0),
    "empty json": (dict(json_stems=[]), 1),
    "soup5 evaluation missing": (dict(stems=["coat96_g123_soup"]), 1),
    "zero rows": (dict(rows=0), 1),
    "all-NaN predictions": (dict(preds=lambda n: np.full((n, len(LABELS)), np.nan, np.float32)), 1),
    "wrong soup members": (dict(members=["g1", "g2", "g3", "g4", "g6"]), 1),
    "label order differs": (dict(labels=LABELS[::-1]), 1),
}

if __name__ == "__main__":
    bad = 0
    for name, (kw, want) in CASES.items():
        with tempfile.TemporaryDirectory() as t:
            make(Path(t), **kw)
            got = run(Path(t))
        print(f"{'ok  ' if got == want else 'FAIL'} {name}: exit {got}, expected {want}")
        bad += got != want
    sys.exit(1 if bad else 0)

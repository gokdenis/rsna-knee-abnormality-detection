"""Gate for a finished pod job: a new soup / checkpoint set may be used only if every check passes.

Checks (local copies in --dir, e.g. ~/knee_g_seeds):
  - the soup checkpoint's swa_of list is exactly the expected members (by run tag, no missing / extra / repeated);
  - every expected checkpoint file is present locally (and, with --kaggle, in the dataset by name and size);
  - each evaluator JSON holds exactly the expected checkpoints (given per file, independent of the output) with finite
    AUCs, and its NPZ has the competition label order, non-empty ids / targets and one finite [0, 1] prediction array
    per expected checkpoint.
Exit code 0 = usable, 1 = do not use.

Example:
    python -I scripts/check_g_outputs.py --dir ~/knee_g_seeds --soup coat96_soup5_swa.pt --members g1 g2 g3 g5 g6 \
        --files g0 g1 g2 g3 g123soup h1 g4 g5 g6 soup5 --kaggle \\
        --eval logs/eval_soup5.json=coat96_g123_soup,coat96_soup5_swa \\
               logs/eval_all.json=raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa,coat96_g0_swa,coat96_g1_swa,coat96_g2_swa,coat96_g3_swa,coat96_g123_soup
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.constants import LABELS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--soup", required=True, help="soup file name inside <dir>/upload")
    ap.add_argument("--members", nargs="+", required=True, help="run tags that must form the soup, e.g. g1 g2 g3 g5 g6")
    ap.add_argument("--files", nargs="+", required=True, help="run tags whose coat96_<tag>_swa.pt must be backed up")
    ap.add_argument("--eval", nargs="*", default=[],
                    help="path=stem1,stem2,...: evaluator JSON (relative to --dir), NPZ next to it, and the exact checkpoint stems it must hold")
    ap.add_argument("--kaggle", action="store_true", help="also compare names and sizes with the Kaggle dataset")
    args = ap.parse_args()
    d, ok = Path(args.dir).expanduser(), True

    def fail(msg):
        nonlocal ok
        ok = False
        print("FAIL", msg)

    up = d / "upload"
    swa_of = torch.load(up / args.soup, map_location="cpu", weights_only=False).get("swa_of", [])
    tags = [Path(p).name.removeprefix("coat96_").removesuffix("_swa.pt") for p in swa_of]
    print("soup members:", tags)
    if sorted(tags) != sorted(args.members) or len(set(tags)) != len(tags):
        fail(f"soup members {tags} != expected {args.members}")

    # required backups: the listed runs, the soup members and the soup itself (so a narrow --files cannot skip them)
    want = {f"coat96_{t}_swa.pt" for t in set(args.files) | set(args.members)} | {args.soup}
    have = {p.name: p.stat().st_size for p in up.glob("*.pt")}
    for name in sorted(want - set(have)):
        fail(f"missing local checkpoint {name}")
    if args.kaggle:
        out = subprocess.run([str(Path(__file__).resolve().parents[1] / ".venv/bin/kaggle"), "datasets", "files",
                              "gkdnzz31/knee-coat96-oai-g"], capture_output=True, text=True).stdout
        remote = {ln.split()[0]: int(ln.split()[1]) for ln in out.splitlines() if ln.split() and ln.split()[0].endswith(".pt")}
        for name in sorted(want):
            if remote.get(name) != have.get(name):
                fail(f"kaggle {name}: remote size {remote.get(name)} vs local {have.get(name)}")

    for spec in args.eval:
        rel, _, exp = spec.partition("=")
        expected = sorted(filter(None, exp.split(",")))
        if not expected:
            fail(f"{rel}: no expected checkpoints given (use path=stem1,stem2,...)")
            continue
        js = d / rel
        if not js.exists():
            fail(f"missing {rel}")
            continue
        res = json.load(open(js))
        cks = {Path(k).stem: k for k in res if k != "mean_prob"}
        if sorted(cks) != expected:
            fail(f"{rel}: evaluated {sorted(cks)} != expected {expected}")
        for stem, k in cks.items():
            for s in ("gold", "oai"):
                v = res[k].get(s)
                if not isinstance(v, float) or not np.isfinite(v):
                    fail(f"{rel}: {stem} has no finite {s} AUC")
        print(Path(rel).name, {stem: {s: round(res[k][s], 4) for s in ("gold", "oai") if isinstance(res[k].get(s), float)}
                               for stem, k in cks.items()})
        npz = js.with_name(js.stem + "_preds.npz")
        if not npz.exists():
            fail(f"missing {npz.name}")
            continue
        z = np.load(npz, allow_pickle=False)
        if list(z["labels"]) != list(LABELS):
            fail(f"{npz.name}: label names / order differ")
        for s in ("gold", "oai"):
            if f"{s}__id" not in z or f"{s}__target" not in z:
                fail(f"{npz.name}: no {s} ids / targets")
                continue
            n = len(z[f"{s}__id"])
            if n == 0:
                fail(f"{npz.name}: {s} has no rows")
            if z[f"{s}__target"].shape != (n, len(LABELS)):  # targets may hold NaN (masked labels)
                fail(f"{npz.name}: {s} target shape {z[s + '__target'].shape}")
            for stem in expected:
                key = f"{s}__{stem}"
                if key not in z or z[key].shape != (n, len(LABELS)):
                    fail(f"{npz.name}: bad or missing {key}")
                elif not (np.isfinite(z[key]).all() and (z[key] >= 0).all() and (z[key] <= 1).all()):
                    fail(f"{npz.name}: {key} has values outside finite [0, 1]")
            print(f"  {npz.name} {s}: {n} rows, {len(expected)} checkpoints")
    print("USABLE" if ok else "DO NOT USE")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

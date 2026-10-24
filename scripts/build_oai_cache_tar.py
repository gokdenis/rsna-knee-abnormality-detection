"""Build a coat96 cache for OAI knees straight from the downloaded series .tar.gz files (no unpacking).

Uses the selection table written by oai_select.py (links.csv: ID, SIDE, sequence, image_file) to know which
tarball holds which sequence, so no DICOM header scan is needed. Same output format as build_oai_cache.py.

Usage:
    python -I scripts/build_oai_cache_tar.py --links /workspace/oai/V00/links_p0.txt \
        --table /workspace/oai/V00/links.csv --raw /workspace/oai/V00p0/raw --visit V00 --out /workspace/cache-oai-V00p0
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import coat96  # noqa: E402
from knee.oai_images import SLOT_SEQUENCES, build_knee_tar  # noqa: E402
from knee.preprocess import encode_slice  # noqa: E402

JPEG = 95


def local_path(raw, url):
    """s3://bucket/submission_x/00m/... -> <raw>/image03/00m/... (downloadcmd's layout)."""
    return os.path.join(raw, "image03", url.split("/", 4)[4])


def work(item):
    key, tars = item
    t0 = time.time()
    try:
        vol, mask = build_knee_tar(tars, threads=4)
        return key, [encode_slice(s, JPEG) if m else b"" for s, m in zip(vol, mask)], mask, time.time() - t0, None
    except Exception as exc:
        return key, [b""] * coat96.MAXS, np.zeros(coat96.MAXS, np.uint8), time.time() - t0, repr(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--links", required=True, help="the part's link list (only these files are used)")
    ap.add_argument("--table", required=True, help="links.csv from oai_select.py")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--visit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wanted = {l.strip() for l in open(args.links) if l.strip()}
    table = pd.read_csv(args.table, dtype={"ID": str})
    table = table[table["image_file"].isin(wanted)].copy()
    table["path"] = table["image_file"].map(lambda u: local_path(args.raw, u))
    present = table["path"].map(os.path.isfile)
    print(f"{args.visit}: files listed {len(table)}, present {int(present.sum())}", flush=True)
    table = table[present]
    items = []
    for (pid, side), grp in sorted(table.groupby(["ID", "SIDE"])):
        by_seq = dict(zip(grp["sequence"], grp["path"]))
        items.append(((pid, side), [next((by_seq[s] for s in seqs if s in by_seq), None) for seqs in SLOT_SEQUENCES]))
    n = len(items)
    blobs = open(out / "blobs.bin", "wb")
    offsets = np.zeros((n, coat96.MAXS + 1), np.int64)
    masks = np.zeros((n, coat96.MAXS), np.uint8)
    row_of = {k: i for i, (k, _) in enumerate(items)}
    pos, failures, t_start = 0, {}, time.time()
    with Pool(args.workers) as pool:
        for done, (key, slices, mask, _, err) in enumerate(pool.imap_unordered(work, items, chunksize=2), 1):
            row = row_of[key]
            masks[row] = mask
            offsets[row, 0] = pos
            for j, buf in enumerate(slices):
                blobs.write(buf)
                pos += len(buf)
                offsets[row, j + 1] = pos
            if err:
                failures["-".join(key)] = err
            if done % 100 == 0 or done == n:
                el = time.time() - t_start
                print(f"{done}/{n} | {el / 60:.1f} min | eta {el / done * (n - done) / 60:.1f} min | "
                      f"failures {len(failures)} | {pos / 1e9:.1f} GB", flush=True)
    blobs.close()
    np.save(out / "offsets.npy", offsets)
    np.save(out / "masks.npy", masks)
    np.save(out / "ids.npy", np.array([f"OAI-{pid}-{side}-{args.visit}" for (pid, side), _ in items]))
    pd.DataFrame([{"row": i, "ID": pid, "SIDE": side, "VISIT": args.visit, **{f"slot{j}": f for j, f in enumerate(ts)}}
                  for i, ((pid, side), ts) in enumerate(items)]).to_csv(out / "index.csv", index=False)
    starts = np.cumsum([0] + coat96.SLOT_SIZES[:-1])
    stats = {"n": n, "gb": pos / 1e9, "minutes": (time.time() - t_start) / 60,
             "filled_slices_mean": float(masks.sum(1).mean()) if n else 0.0,
             "slot_fill": masks[:, starts].mean(0).tolist() if n else [], "failures": failures}
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "failures"}, indent=1))


if __name__ == "__main__":
    main()

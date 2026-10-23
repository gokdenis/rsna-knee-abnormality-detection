"""Build a coat96 JPEG slice cache for MOAKS-labelled OAI knees from an extracted OAI image tree.

Writes the same format as the competition caches (blobs.bin, offsets.npy, masks.npy, ids.npy) plus
series.csv (every series found) and index.csv (cache row -> ID, SIDE, VISIT, slot folders).
Cache ids are "OAI-<ID>-<SIDE>-<VISIT>", matching scripts/train_coat96.py.

Usage:
    python -I scripts/build_oai_cache.py --root /workspace/oai/V00 --visit V00 \
        --labels /workspace/new_knee/data/oai/oai_labels.csv --out /workspace/cache-oai-V00
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import coat96  # noqa: E402
from knee.oai_images import build_knee, knee_series, scan  # noqa: E402
from knee.preprocess import encode_slice  # noqa: E402

JPEG = 95


def work(item):
    key, folders = item
    t0 = time.time()
    try:
        vol, mask = build_knee(folders, threads=4)
        return key, [encode_slice(s, JPEG) if m else b"" for s, m in zip(vol, mask)], mask, time.time() - t0, None
    except Exception as exc:
        return key, [b""] * coat96.MAXS, np.zeros(coat96.MAXS, np.uint8), time.time() - t0, repr(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--visit", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the first N knees (smoke test)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    series = scan(args.root)
    series.to_csv(out / "series.csv", index=False)
    print(f"series {len(series)} in {(time.time() - t0) / 60:.1f} min; sequences:\n"
          f"{series['sequence'].value_counts(dropna=False).to_string()}", flush=True)

    labels = pd.read_csv(args.labels, dtype={"ID": str})
    wanted = set(map(tuple, labels.loc[labels["VISIT"] == args.visit, ["ID", "SIDE"]].values))
    knees = {k: v for k, v in knee_series(series).items() if k in wanted}
    items = sorted(knees.items())
    if args.limit:
        items = items[:args.limit]
    n = len(items)
    print(f"labelled knees for {args.visit}: {len(wanted)}; with images: {n}", flush=True)

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
    pd.DataFrame([{"row": i, "ID": pid, "SIDE": side, "VISIT": args.visit, **{f"slot{j}": f for j, f in enumerate(fs)}}
                  for i, ((pid, side), fs) in enumerate(items)]).to_csv(out / "index.csv", index=False)
    stats = {"n": n, "gb": pos / 1e9, "minutes": (time.time() - t_start) / 60,
             "filled_slices_mean": float(masks.sum(1).mean()) if n else 0.0,
             "slot_fill": masks.reshape(n, -1)[:, np.cumsum([0] + coat96.SLOT_SIZES[:-1])].mean(0).tolist() if n else [],
             "failures": failures}
    json.dump(stats, open(out / "stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "failures"}, indent=1))


if __name__ == "__main__":
    main()

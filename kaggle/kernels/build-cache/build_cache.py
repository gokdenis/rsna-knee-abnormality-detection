"""Build the training slice cache from the competition DICOMs (CPU only).

Outputs in /kaggle/working:
    vols.npy   uint8 [N, N_SLICES, IMG, IMG]
    masks.npy  uint8 [N, N_SLICES]   1 where a slice was filled
    ids.npy    str   [N]             StudyInstanceUID in row order
    stats.json timing and failure counts
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

SMOKE = False  # True: only a handful of studies, to time the pipeline
SMOKE_N = 40
WORKERS = os.cpu_count() or 4
LAYOUT = globals().get("LAYOUT", "v1")
SHARD, N_SHARDS = globals().get("SHARD", 0), globals().get("N_SHARDS", 1)  # every N_SHARDS-th study from SHARD


sys.path.insert(0, code_dir())
from knee.preprocess import LAYOUTS, build_study, encode_slice  # noqa: E402

if LAYOUT == "raptor":  # the public Raptor sampling (80 x 336), stored as JPEG
    from knee import raptor as _raptor
    IMG, N_SLICES, JPEG = _raptor.IMG, _raptor.MAXS, 95
elif LAYOUT in ("coat96", "coat96alt"):  # nartaa's OAI checkpoint layout (96 x 384), stored as JPEG; alt = second view
    from knee import coat96 as _coat96
    IMG, N_SLICES, JPEG = _coat96.IMG, _coat96.MAXS, 95
else:
    IMG, N_SLICES, JPEG = LAYOUTS[LAYOUT].img, LAYOUTS[LAYOUT].n_slices, LAYOUTS[LAYOUT].jpeg

ROOT = comp_root()
SERIES = pd.read_csv(os.path.join(ROOT, "train_series.csv"), dtype=str)
SERIES_BY_STUDY = {k: v.to_dict("records") for k, v in SERIES.groupby("StudyInstanceUID")}


def work(study_id):
    t0 = time.time()
    try:
        if LAYOUT == "coat96alt":  # alternative series of the same type per slot (studies that have one)
            vol, mask = _coat96.build_competition_study_alt(os.path.join(ROOT, "train_series", study_id),
                                                            SERIES_BY_STUDY.get(study_id, []), threads=6)
        elif LAYOUT == "coat96":
            vol, mask = _coat96.build_competition_study(os.path.join(ROOT, "train_series", study_id),
                                                        SERIES_BY_STUDY.get(study_id, []), threads=6)
        elif LAYOUT == "raptor":
            vol, mask = _raptor.build_raptor_study(os.path.join(ROOT, "train_series", study_id),
                                                   SERIES_BY_STUDY.get(study_id, []), threads=6)
        else:
            vol, mask = build_study(os.path.join(ROOT, "train_series", study_id),
                                    SERIES_BY_STUDY.get(study_id, []), threads=6, layout=LAYOUT, apply_jpeg=False)
        if JPEG:  # encode in the worker; empty slices are stored as zero-length entries
            vol = [encode_slice(s, JPEG) if m else b"" for s, m in zip(vol, mask)]
        return study_id, vol, mask, time.time() - t0, None
    except Exception as exc:  # never lose a row; record the failure
        empty = [b""] * N_SLICES if JPEG else np.zeros((N_SLICES, IMG, IMG), np.uint8)
        return study_id, empty, np.zeros(N_SLICES, np.uint8), time.time() - t0, repr(exc)


def main():
    ids = pd.read_csv(os.path.join(ROOT, "train.csv"), usecols=["StudyInstanceUID"])
    ids = ids["StudyInstanceUID"].astype(str).tolist()
    ids = ids[SHARD::N_SHARDS]
    if LAYOUT == "coat96alt":
        ids = [s for s in ids if _coat96.pick_alternative_series(SERIES_BY_STUDY.get(s, [])) is not None]
    if SMOKE:
        ids = ids[:SMOKE_N]
    n = len(ids)
    print(f"studies {n} | workers {WORKERS} | layout {LAYOUT} | slices {N_SLICES} | img {IMG}", flush=True)

    if JPEG:
        blobs = open("/kaggle/working/blobs.bin", "wb")
        offsets = np.zeros((n, N_SLICES + 1), np.int64)
        pos = 0
    else:
        vols = np.lib.format.open_memmap("/kaggle/working/vols.npy", mode="w+",
                                         dtype=np.uint8, shape=(n, N_SLICES, IMG, IMG))
    masks = np.zeros((n, N_SLICES), np.uint8)
    row_of = {sid: i for i, sid in enumerate(ids)}
    failures, times, t_start = {}, [], time.time()

    with Pool(WORKERS) as pool:
        for done, (sid, vol, mask, dt, err) in enumerate(pool.imap_unordered(work, ids, chunksize=2), 1):
            row = row_of[sid]
            masks[row] = mask
            if JPEG:
                offsets[row, 0] = pos
                for j, buf in enumerate(vol):
                    blobs.write(buf)
                    pos += len(buf)
                    offsets[row, j + 1] = pos
            else:
                vols[row] = vol
            times.append(dt)
            if err:
                failures[sid] = err
            if done % 100 == 0 or done == n:
                elapsed = time.time() - t_start
                print(f"{done}/{n} | {elapsed / 60:.1f} min | eta {elapsed / done * (n - done) / 60:.1f} min"
                      f" | failures {len(failures)}", flush=True)

    if JPEG:
        blobs.close()
        np.save("/kaggle/working/offsets.npy", offsets)
        print(f"blobs {pos / 1e9:.2f} GB", flush=True)
    else:
        vols.flush()
    np.save("/kaggle/working/masks.npy", masks)
    np.save("/kaggle/working/ids.npy", np.array(ids))
    stats = {
        "layout": LAYOUT, "n": n, "minutes": (time.time() - t_start) / 60, "sec_per_study_mean": float(np.mean(times)),
        "filled_slices_mean": float(masks.sum(1).mean()), "empty_studies": int((masks.sum(1) == 0).sum()),
        "failures": failures,
    }
    json.dump(stats, open("/kaggle/working/stats.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if k != "failures"}, indent=1))
    print("failures:", list(failures.items())[:10])


if __name__ == "__main__":
    main()

"""Unpack downloaded OAI series tarballs into one folder per series and delete the tarballs.

Runs alongside downloadcmd: every pass unpacks the finished .tar.gz files (not *.partial), and with
--follow it keeps polling until the download log reports completion.

Usage:
    python -I scripts/oai_extract.py --raw /workspace/oai/V00/raw --out /workspace/oai/V00/img \
        --follow /workspace/logs/dl_V00.log
"""

import argparse
import os
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def unpack(path):
    dest = Path(OUT) / Path(path).name.replace(".tar.gz", "")
    tmp = dest.with_name(dest.name + ".tmp")
    try:
        with tarfile.open(path) as tar:
            members = [m for m in tar.getmembers() if m.isfile() and not os.path.isabs(m.name) and ".." not in m.name]
            tmp.mkdir(parents=True, exist_ok=True)
            for m in members:
                m.name = os.path.basename(m.name)
                tar.extract(m, tmp, filter="data")
        tmp.rename(dest)
        os.remove(path)
        return None
    except Exception as exc:
        os.rename(path, path + ".bad")  # keep it out of the next pass
        return f"{path}: {exc!r}"


def finished(raw):
    return [str(p) for p in Path(raw).rglob("*.tar.gz")]


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--follow", default=None, help="downloadcmd log; keep polling until it says it finished")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    OUT = args.out
    Path(OUT).mkdir(parents=True, exist_ok=True)
    done, errors = 0, []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(OUT,)) as pool:
        while True:
            log_done = args.follow is None or "Finished processing all download requests" in Path(args.follow).read_text(errors="ignore")
            todo = finished(args.raw)
            for err in pool.map(unpack, todo, chunksize=8):
                if err:
                    errors.append(err)
            done += len(todo)
            print(f"unpacked {done} (errors {len(errors)})", flush=True)
            if log_done and not todo:
                break
            if not todo:
                time.sleep(30)
    for e in errors[:20]:
        print("ERROR", e)


def _init(out):
    global OUT
    OUT = out


OUT = None

if __name__ == "__main__":
    main()

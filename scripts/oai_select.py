"""Pick the OAI image files to download: the coat96 series of MOAKS-labelled knees for one visit.

Reads the package's image03.txt (series description per file) and the package file listing (sizes),
writes an S3 link list for `downloadcmd -t` and prints the expected size.

Usage:
    python -I scripts/oai_select.py --image03 /workspace/oai/V00/image03.txt \
        --sizes /root/NDA/nda-tools/downloadcmd/packages/<V00_PACKAGE_ID>/package_file_metadata_<V00_PACKAGE_ID>.txt \
        --labels data/oai/oai_labels.csv --visit V00 --out /workspace/oai/V00/links.txt
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.oai_images import SLOT_SEQUENCES, classify  # noqa: E402

NEEDED = {s for seqs in SLOT_SEQUENCES for s in seqs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image03", required=True)
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--visit", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip", nargs="*", default=["COR_FLASH"], help="sequences not to download")
    ap.add_argument("--parts", type=int, default=1, help="also write links_p<i>.txt, splitting the knees into parts")
    args = ap.parse_args()

    m = pd.read_csv(args.image03, sep="\t", skiprows=[1], dtype=str, low_memory=False)
    sizes = pd.read_csv(args.sizes, dtype=str)
    size_of = dict(zip(sizes["NDA_S3_URL"], sizes["FILE_SIZE"].astype(float)))
    cls = m["image_description"].fillna("").map(classify)
    m["sequence"], m["SIDE"] = cls.str[0], cls.str[1]
    m["ID"] = m["src_subject_id"].astype(str).str.strip()
    m = m[m["sequence"].isin(NEEDED - set(args.skip)) & m["SIDE"].notna()]

    labels = pd.read_csv(args.labels, dtype={"ID": str})
    wanted = labels.loc[labels["VISIT"] == args.visit, ["ID", "SIDE"]].drop_duplicates()
    pick = m.merge(wanted, on=["ID", "SIDE"])
    pick = pick.assign(gb=pick["image_file"].map(size_of) / 1e9)
    # one file per knee and sequence (the largest when a series was repeated)
    pick = pick.sort_values("gb", ascending=False).drop_duplicates(["ID", "SIDE", "sequence"])
    Path(args.out).write_text("\n".join(pick["image_file"]) + "\n")
    pick[["ID", "SIDE", "sequence", "image_description", "image_file", "gb"]].to_csv(
        Path(args.out).with_suffix(".csv"), index=False)
    knees = pick[["ID", "SIDE"]].drop_duplicates()
    if args.parts > 1:  # whole knees per part, so each part can be cached on its own (limited disk)
        part_of = {k: i % args.parts for i, k in enumerate(map(tuple, knees.values))}
        part = pick[["ID", "SIDE"]].apply(tuple, axis=1).map(part_of)
        for i in range(args.parts):
            Path(args.out).with_name(f"links_p{i}.txt").write_text("\n".join(pick.loc[part == i, "image_file"]) + "\n")
    print(f"{args.visit}: labelled knees {len(wanted)}, with images {len(knees)}, files {len(pick)}, "
          f"{pick['gb'].sum():.1f} GB")
    print(pick.groupby("sequence")["gb"].agg(["count", "sum"]).round(1).to_string())


if __name__ == "__main__":
    main()

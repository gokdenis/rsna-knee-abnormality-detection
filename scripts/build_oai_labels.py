"""Build soft competition labels for OAI knees from MOAKS readings.

Output: data/oai/oai_labels.csv with ID, SIDE (R/L), VISIT (V00/V01), 12 soft labels (NaN = masked).

Usage:
    .venv/bin/python scripts/build_oai_labels.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knee.constants import LABELS  # noqa: E402
from knee.oai_labels import collapse, map_moaks, resolve_within_grade  # noqa: E402

SQ = ROOT / "data/oai/raw/extracted/Semi-Quant Scoring_ASCII"
# Visits with MOAKS readings whose image packages are on the NDA account (package ids in docs/PLAN.md).
TABLES = {"V00": "kMRI_SQ_MOAKS_BICL00.txt", "V01": "kMRI_SQ_MOAKS_BICL01.txt",
          "V03": "kMRI_SQ_MOAKS_BICL03.txt", "V06": "kMRI_SQ_MOAKS_BICL06.txt"}
# Visits read by MOAKS but without downloaded images: used only as reading history for change codes.
HISTORY_ONLY = {"V05": "kMRI_SQ_MOAKS_BICL05.txt"}
# The 12-month export stores only the cartilage area grade (no full-thickness digit).
NO_FULL_THICKNESS = {"V01"}


def main() -> None:
    history = {**TABLES, **HISTORY_ONLY}
    raw = {visit: pd.read_csv(SQ / name, sep="|", dtype=str, encoding="latin-1") for visit, name in history.items()}
    raw, stats = resolve_within_grade(raw)
    print("within-grade change codes (+/-0.5):", stats)
    parts = [map_moaks(raw[visit], visit, with_cartilage=visit not in NO_FULL_THICKNESS) for visit in TABLES]
    labels = collapse(pd.concat(parts, ignore_index=True))
    out = ROOT / "data/oai/oai_labels.csv"
    labels.to_csv(out, index=False)
    print(f"Wrote {out}: {len(labels)} knee-visits "
          f"({labels[['ID', 'SIDE']].drop_duplicates().shape[0]} knees, {labels['ID'].nunique()} subjects)")
    summary = pd.DataFrame({"available": labels[LABELS].notna().sum(),
                            "positive_rate": (labels[LABELS] >= 0.5).sum() / labels[LABELS].notna().sum()})
    print(summary.round(3).to_string())
    print(labels.groupby("VISIT").size().to_string())


if __name__ == "__main__":
    main()

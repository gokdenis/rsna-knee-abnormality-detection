"""Map OAI MOAKS readings to the competition's 12 findings.

Each knee/visit becomes one row of soft targets in [0, 1]; findings MOAKS does not score
(or that are missing for a read project) are NaN and must be masked in the loss.

Mapping follows the competition's labeling criteria as closely as MOAKS allows:
- OA (medial / lateral / patellofemoral): MOAKS cartilage codes "A.F" give the share of the
  subregion with any cartilage loss (A: 0-3) and with full-thickness loss (F: 0-3). Positive when a
  subregion shows >=10% full-thickness loss, or 10-75%+ area with some full-thickness loss.
- Meniscus: any tear type (radial/horizontal/vertical/complex), maceration or root tear in any horn.
  Intrasubstance signal alone is negative.
- ACL: complete tear = 1, partial = 0.5 (competition counts only high-grade tears).
- Effusion / Synovitis (Hoffa): moderate or severe = 1, mild = small soft value.
- Baker's: popliteal cyst present (MOAKS does not grade size here) = 0.7.
- MCL, Contusion (MOAKS BMLs are mostly degenerative), Fracture: not mapped.

Follow-up visits use two change codes (MOAKS data dictionary, kMRI_SQ_MOAKS_BICL_Descrip.pdf, 1.6.1):
+0.5 / -0.5 mean "same grade as the previous visit, with a within-grade worsening / improvement". They are
not grades; `resolve_within_grade` replaces them with the knee's previous absolute grade (same reading
project first) and marks cells it cannot resolve, which make the compartment uncertain unless another
subregion is clearly positive.
"""

import re

import numpy as np
import pandas as pd

from .constants import LABELS

# MOAKS cartilage subregions per compartment (femur anterior = trochlea belongs to PF).
COMPARTMENTS = {
    "Medial OA": ["FMC", "FMP", "TMA", "TMC", "TMP"],
    "Lateral OA": ["FLC", "FLP", "TLA", "TLC", "TLP"],
    "PF OA": ["PM", "PL", "FMA", "FLA"],
}
MENISCUS_HORNS = {"Medial Meniscus": ["MA", "MB", "MP"], "Lateral Meniscus": ["LA", "LB", "LP"]}
# radial, horizontal, vertical, complex, partial / progressive partial (follow-up only) / complete maceration
TEAR_CODES = {2, 3, 4, 5, 6, 7, 8}
UNRESOLVED = "UNRESOLVED"  # a within-grade change code without a usable previous grade


def code(value) -> float:
    """Numeric code of a MOAKS cell ('2.2: 10-75% area, ...' or a bare '2'); NaN when missing/invalid."""
    if not isinstance(value, str):
        return np.nan
    m = re.match(r"\s*(-?[0-9]+(?:\.[0-9]+)?)\s*(?::|$)", value)
    if not m:
        return np.nan
    c = float(m.group(1))
    return np.nan if c < 0 else c


def cartilage_score(values) -> float:
    """Soft OA target for one compartment from its subregion cartilage codes.

    Unresolved change codes leave the compartment unknown (NaN) unless the known subregions already make
    it clearly positive.
    """
    unresolved = any(v == UNRESOLVED for v in values)
    codes = [code(v) for v in values if v != UNRESOLVED]
    codes = [c for c in codes if not np.isnan(c)]
    if not codes:
        return np.nan
    score = _cartilage_score(codes)
    return np.nan if unresolved and score < 1.0 else score


def _cartilage_score(codes) -> float:
    area = [int(c) for c in codes]
    full = [int(round((c - int(c)) * 10)) for c in codes]
    if any(f >= 2 for f in full) or any(a >= 2 and f >= 1 for a, f in zip(area, full)):
        return 1.0
    if any(a == 3 for a in area):
        return 0.5
    if any(a == 2 for a in area):
        return 0.2
    return 0.0


def meniscus_score(horn_values, root_value) -> float:
    codes = [code(v) for v in horn_values]
    root = code(root_value)
    known = [c for c in codes if not np.isnan(c)]
    if not known and np.isnan(root):
        return np.nan
    if any(int(c) in TEAR_CODES for c in known) or root == 1:
        return 1.0
    return 0.0


def graded(value, mild: float) -> float:
    c = code(value)
    if np.isnan(c):
        return np.nan
    return 1.0 if c >= 2 else (mild if c == 1 else 0.0)


def _is_change_code(value) -> bool:
    return isinstance(value, str) and re.match(r"\s*-?0\.5\s*(?::|$)", value) is not None


def _side(value) -> str:
    """'1: Right' / '1' -> 'R', '2: Left' / '2' -> 'L' (the 12-month export uses bare digits)."""
    v = str(value)
    return "R" if ("Right" in v or v.strip() == "1") else "L"


def resolve_within_grade(tables: dict, order=("V00", "V01", "V03", "V05", "V06"), area_only=("V01",)):
    """Replace +/-0.5 cartilage change codes with the grade they refer to, or mark them UNRESOLVED.

    A change code means "same grade as this reader's previous reading". The reference is the latest earlier
    reading of the same knee (ID + normalised side) and subregion in the same reading project (READPRJ);
    other projects are never used. If that reading is itself a change code it was resolved first (visits
    are processed in order). If it comes from an area-only export (V01: no full-thickness digit) the grade
    is only used when the area grade is 0 or 1, where the full-thickness grade cannot change the OA target;
    otherwise, or without a reference reading, the cell is UNRESOLVED.

    tables: {visit: raw MOAKS table}; visits used only as history (e.g. V05) can be included.
    Returns (new tables, per-visit counts by outcome).
    """
    out, stats = {}, {}
    subs = [sub for subs_ in COMPARTMENTS.values() for sub in subs_]
    for i, visit in enumerate(order):
        if visit not in tables:
            continue
        df = tables[visit].copy()
        df["_side"] = df["SIDE"].map(_side)
        counts = {"same_project": 0, "area_only_low": 0, "unresolved_no_reference": 0,
                  "unresolved_area_only": 0, "unresolved_reference": 0}
        earlier = [v for v in reversed(order[:i]) if v in out]
        for sub in subs:
            col = f"{visit}MCM{sub}"
            if col not in df:
                continue
            for r in df.index[df[col].map(_is_change_code)]:
                key = (df.at[r, "ID"], df.at[r, "_side"], df.at[r, "READPRJ"])
                ref = None
                for prev in earlier:
                    pt, pcol = out[prev], f"{prev}MCM{sub}"
                    if pcol not in pt:
                        continue
                    hit = pt[(pt["ID"] == key[0]) & (pt["_side"] == key[1]) & (pt["READPRJ"] == key[2])][pcol]
                    hit = [h for h in hit if isinstance(h, str) and (h == UNRESOLVED or not np.isnan(code(h)))]
                    if hit:
                        ref = (prev, hit[0])
                        break
                if ref is None:
                    df.at[r, col] = UNRESOLVED
                    counts["unresolved_no_reference"] += 1
                elif ref[1] == UNRESOLVED:
                    df.at[r, col] = UNRESOLVED
                    counts["unresolved_reference"] += 1
                elif ref[0] in area_only:
                    area = int(code(ref[1]))
                    if area <= 1:
                        df.at[r, col] = str(area)
                        counts["area_only_low"] += 1
                    else:
                        df.at[r, col] = UNRESOLVED
                        counts["unresolved_area_only"] += 1
                else:
                    df.at[r, col] = ref[1]
                    counts["same_project"] += 1
        out[visit] = df
        stats[visit] = counts
    return {v: t.drop(columns="_side") for v, t in out.items()}, stats


def map_moaks(df: pd.DataFrame, visit: str, with_cartilage: bool = True) -> pd.DataFrame:
    """Rows of a kMRI_SQ_MOAKS_BICLxx table -> ID, SIDE, VISIT, 12 soft labels (NaN = masked).

    Set with_cartilage=False for exports whose cartilage cells lack the full-thickness digit
    (the 12-month release stores only the area grade), so OA stays masked there.
    """
    p = f"{visit}M"
    side = df["SIDE"].astype(str).map(lambda s: "R" if ("Right" in s or s.strip() == "1") else "L")
    out = pd.DataFrame({"ID": df["ID"].astype(str), "SIDE": side,
                        "VISIT": visit, "READPRJ": df.get("READPRJ")})
    for col in LABELS:
        out[col] = np.nan
    for label, subs in COMPARTMENTS.items():
        if not with_cartilage:
            continue
        cols = [f"{p}CM{s}" for s in subs if f"{p}CM{s}" in df]
        out[label] = [cartilage_score(r) for r in df[cols].itertuples(index=False)]
    for label, horns in MENISCUS_HORNS.items():
        side = horns[0][0]
        tear_cols = [f"{p}MT{h}" for h in horns]
        root_col = f"{p}MRT{side}"
        out[label] = [meniscus_score(r[:-1], r[-1]) for r in df[tear_cols + [root_col]].itertuples(index=False)]
    acl = df[f"{p}ACLTR"].map(code)
    out["ACL"] = acl.map(lambda c: np.nan if np.isnan(c) else {0: 0.0, 1: 0.5, 2: 1.0}.get(int(c), np.nan))
    out["Effusion"] = df[f"{p}EFFWK"].map(lambda v: graded(v, 0.2))
    out["Synovitis"] = df[f"{p}SYIC"].map(lambda v: graded(v, 0.3))
    out["Baker's"] = df[f"{p}POPCYS"].map(lambda v: np.nan if np.isnan(code(v)) else 0.7 * code(v))
    return out


def collapse(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per knee/visit: the mean over reads of each finding (soft consensus; NaN when no read scored it).

    Repeated reads come from different reading projects and disagree on part of the knees (e.g. ~15% of
    medial meniscus groups); averaging keeps that uncertainty instead of picking one read.
    """
    return rows.groupby(["ID", "SIDE", "VISIT"], as_index=False)[LABELS].mean()

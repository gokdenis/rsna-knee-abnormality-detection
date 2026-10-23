"""Regression tests for the MOAKS change-code resolution and the OA / meniscus mapping."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee.oai_labels import UNRESOLVED, cartilage_score, meniscus_score, resolve_within_grade  # noqa: E402


def table(visit, rows):
    """rows: (ID, SIDE, READPRJ, FMC value)."""
    return pd.DataFrame([{"ID": i, "SIDE": s, "READPRJ": p, f"{visit}MCMFMC": v} for i, s, p, v in rows])


def resolve(tables):
    out, _ = resolve_within_grade(tables)
    return {v: list(t[f"{v}MCMFMC"]) for v, t in out.items()}


def test_same_project_reference():
    got = resolve({"V00": table("V00", [("1", "1: Right", "22", "2.2: x")]),
                   "V03": table("V03", [("1", "1: Right", "22", "0.5: Within-grade change")])})
    assert got["V03"] == ["2.2: x"]


def test_no_reference_is_unresolved():
    got = resolve({"V03": table("V03", [("1", "1: Right", "22", "0.5: Within-grade change")])})
    assert got["V03"] == [UNRESOLVED]


def test_other_project_is_not_used():
    got = resolve({"V00": table("V00", [("1", "1: Right", "65", "3.3: x")]),
                   "V03": table("V03", [("1", "1: Right", "22", "0.5: Within-grade change")])})
    assert got["V03"] == [UNRESOLVED]


def test_area_only_intermediate_visit():
    # V01 (area only, bare-digit side) is the previous reading: area 2 cannot fix the full-thickness grade
    t00 = table("V00", [("1", "1: Right", "22", "2.0: x"), ("2", "2: Left", "22", "1.0: x")])
    t01 = table("V01", [("1", "1", "22", "2"), ("2", "2", "22", "1")])
    t03 = table("V03", [("1", "1: Right", "22", "0.5: Within-grade change"),
                        ("2", "2: Left", "22", "-0.5: Within-grade change")])
    got = resolve({"V00": t00, "V01": t01, "V03": t03})
    assert got["V03"] == [UNRESOLVED, "1"]


def test_chain_through_resolved_change_code():
    got = resolve({"V00": table("V00", [("1", "1: Right", "22", "3.2: x")]),
                   "V03": table("V03", [("1", "1: Right", "22", "0.5: Within-grade change")]),
                   "V06": table("V06", [("1", "1: Right", "22", "0.5: Within-grade change")])})
    assert got["V06"] == ["3.2: x"]


def test_unresolved_compartment_masked_unless_clearly_positive():
    assert np.isnan(cartilage_score([UNRESOLVED, "0: none", "1.0: x"]))
    assert cartilage_score([UNRESOLVED, "2.2: x"]) == 1.0


def test_meniscus_code_7_is_a_tear():
    assert meniscus_score(["7: progressive partial maceration", "0", "0"], "0") == 1.0
    assert meniscus_score(["1: intrasubstance signal", "0", "0"], "0") == 0.0

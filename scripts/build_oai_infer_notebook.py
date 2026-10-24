"""Add our fine-tuned coat96 checkpoints as an extra stage to the public 0.957 (+224) notebook.

Each new stage reuses one of nartaa's published inference scripts of the base unchanged except: our
checkpoint list as ARMS, no hash pinning of the release checkpoint, and its own output files:
    --files     -> 384 script (mirror TTA)  -> ours96_raw.csv / ours96_sub.csv
    --files224  -> 224 script (0.8 crop)    -> ours224_raw.csv / ours224_sub.csv
The blend cell mixes our rank into the Raptor slot (0.65 / 0.35 between our 384 and 224 stages, as the base
does for its own two Raptor models):
    rr = (1 - A96) * rr + A96 * rank(ours)
or, in three-way mode (--b3), the final rank keeps a share B3 for our stages:
    final = (1 - B3) * final + B3 * rank(ours)
A stage runs only if the 9 h clock allows; any failure leaves the base submission unchanged.

Usage:
    python scripts/build_oai_infer_notebook.py --base kaggle/kernels/knee-h957-r224/knee-h957-r224.ipynb \
        --dataset gkdnzz31/knee-coat96-oai --files coat96_a_swa.pt --a96 0.5 --slug knee-oai96-blend
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KINDS = {  # base script prefix, our script/outputs, arch and res of our checkpoints, release hash to drop
    "384": dict(script="/tmp/raptor_run.py", ours="ours96", raptor="raptor",
                arch="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", res=384, release="7e5315dad125"),
    "384alt": dict(script="/tmp/raptor_run.py", ours="ours96alt", raptor="raptor",
                   arch="coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", res=384, release="7e5315dad125"),
    "r384alt": dict(script="/tmp/raptor_run.py", ours="raptoralt", raptor="raptor"),
    "r224alt": dict(script="/tmp/raptor224_run.py", ours="raptor224alt", raptor="raptor224"),
    "224": dict(script="/tmp/raptor224_run.py", ours="ours224", raptor="raptor224",
                arch="coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", res=224, release="3394fdd7e668"),
}


def src(cell):
    return "".join(cell["source"])


def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


READ_LINE = "        vol, mask = _READER(sid, _SER, _TSDIR)\n"
JPEG_LINES = READ_LINE + (
    "        import cv2 as _cv2   # [ours] JPEG q95 round-trip: our fine-tunes were trained on q95 JPEG slices\n"
    "        for _j in np.where(mask > 0)[0]:\n"
    "            vol[_j] = _cv2.imdecode(_cv2.imencode('.jpg', vol[_j], [_cv2.IMWRITE_JPEG_QUALITY, 95])[1],"
    " _cv2.IMREAD_GRAYSCALE)\n")


SER_LINE = '    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}\n'
ALT_SER = (
    "    # [ours] second view: per slot an unused series with the same plane / fluid / fat flags as the primary pick\n"
    "    # (first in table order), else the primary; only the chosen series are passed, in slot order, so the\n"
    "    # reader's own picker selects them. Studies without any alternative get no series and are skipped.\n"
    "    def _alt_rows(rows):\n"
    "        prim, used = [], set()\n"
    "        for plane, fluid, _k in SLOTS:\n"
    "            r = _pick_series_for_slot(rows, plane, fluid, used)\n"
    "            if r is not None:\n"
    "                used.add(r['SeriesInstanceUID'])\n"
    "            prim.append(r)\n"
    "        out, anyalt = [], False\n"
    "        for r in prim:\n"
    "            a = None if r is None else next((c for c in rows if c['SeriesInstanceUID'] not in used and all(\n"
    "                str(c.get(q)) == str(r.get(q)) for q in ('Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression'))), None)\n"
    "            if a is not None:\n"
    "                used.add(a['SeriesInstanceUID']); anyalt = True\n"
    "            if (a or r) is not None:\n"
    "                out.append(a or r)\n"
    "        return out if anyalt else []\n"
    "    SER = {k: _alt_rows(v) for k, v in SER.items()}\n"
    "    print('[ours] second view: studies with an alternative', sum(1 for v in SER.values() if v), flush=True)\n")


def ours_script(base_script, files, kind="384", jpeg=True):
    k = KINDS[kind]
    s = base_script.replace(f"%%writefile {k['script']}", f"%%writefile /tmp/{k['ours']}_run.py", 1)
    # drop the release checkpoint pinning header (weight lookup + sha256 of the release file)
    s = re.sub(r"_weight_name = .*?\"Checkpoint hash differs\"\n", "", s, count=1, flags=re.S)
    s = re.sub(r"    assert observed_sha == '[0-9a-f]{64}', \"Bound checkpoint bytes differ\"\n", "", s, count=1)
    s = re.sub(r"    assert int\(ck.get\(\"res\", res_default\)\) == 384, \"Expected original 384 checkpoint\"\n", "", s,
               count=1)
    arms = ",\n".join(f'    {{"file": "{f}", "arch": "{k["arch"]}", "res": {k["res"]}, "w": 1.0}}' for f in files)
    s, n = re.subn(r"ARMS = \[.*?\n\]", "ARMS = [   # our coat96 fine-tunes (OAI MOAKS + competition)\n" + arms + ",\n]",
                   s, count=1, flags=re.S)
    assert n == 1, "ARMS block not found"
    for suffix in ("raw", "sub"):
        a, b = f"/kaggle/working/{k['raptor']}_{suffix}.csv", f"/kaggle/working/{k['ours']}_{suffix}.csv"
        assert a in s, a
        s = s.replace(a, b)
    assert k["release"] not in s, "release hash still pinned"
    if jpeg:
        assert s.count(READ_LINE) == 1, "volume read line not found"
        s = s.replace(READ_LINE, JPEG_LINES)
    if kind == "384alt":
        assert s.count(SER_LINE) == 1, "series table line not found"
        s = s.replace(SER_LINE, SER_LINE + ALT_SER)
    return add_status(s, k["ours"])


R224_MIRROR = (
    "\nSLOT_SIZES = [26, 22, 18, 12, 18]   # [ours] anatomical mirror TTA for the 224 model, same rule as the 384 script\n"
    "SAGITTAL_SLOTS = (0, 1)\n\n\n"
    "def _slot_index(D):\n"
    "    b = np.cumsum([0] + SLOT_SIZES)\n"
    "    if b[-1] != D: raise RuntimeError(f'slot sizes sum to {b[-1]}, volume has {D} slices')\n"
    "    s_ = np.searchsorted(b, np.arange(D), side='right') - 1\n"
    "    return s_, None\n\n\n"
    "def _mirror_tri(tri, cs, D):\n"
    "    \"\"\"L/R mirror of (k,3,H,W) uint8 triplets before cropping and normalisation: sagittal windows reverse the slice\n"
    "    channels, coronal / axial windows flip the width (as _mirror_tri in the 384 script).\"\"\"\n"
    "    s_, _ = _slot_index(D)\n"
    "    sag = torch.from_numpy(np.isin(s_[np.asarray(cs, dtype=np.int64)], SAGITTAL_SLOTS)).to(tri.device).view(-1, 1, 1, 1)\n"
    "    return torch.where(sag, tri.flip(1), tri.flip(-1))\n\n\n")


def add_r224_mirror(s):
    """Patch the published 224 script: plain + anatomical-mirror windows, mean of the two probabilities (GPU path)."""
    a = "def gpu_windows(vol, mask, k, res, dev, norm=NORM):\n"
    assert s.count(a) == 1, "224 gpu_windows not found"
    s = s.replace(a, R224_MIRROR + "def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):\n")
    a = "    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8\n"
    assert s.count(a) == 1, "224 triplet line not found"
    s = s.replace(a, a + "    if mirror: tri = _mirror_tri(tri, cs, D)   # [ours]\n")
    a = "                        arm_probs[a][i] = infer_probs(model, xw, d)\n"
    assert s.count(a) == 1, "224 inference line not found"
    s = s.replace(a, "                        arm_probs[a][i] = 0.5 * (infer_probs(model, xw, d) + infer_probs(\n"
                     "                            model, gpu_windows(vol, mask, K_EVAL, res, d, mirror=True), d))   # [ours] mirror TTA\n")
    return s


def compact_script(cell):
    """A %%writefile cell stored as zlib + base64 and written at run time: same file content, much smaller notebook
    (Kaggle rejects notebooks above ~1 MB)."""
    import base64
    import zlib
    head, body = cell.split("\n", 1)
    assert head.startswith("%%writefile ")
    path = head.split(" ", 1)[1].strip()
    blob = base64.b64encode(zlib.compress(body.encode(), 9)).decode()
    return (f"# [ours] writes {path}: the published script with the [ours] patches, stored zlib + base64 so the notebook\n"
            f"# stays under Kaggle's size limit (decoded text is identical to the uncompressed build)\n"
            f"import base64 as _b64, zlib as _zlib\n"
            f"open({path!r}, 'w').write(_zlib.decompress(_b64.b64decode({blob!r})).decode())\n"
            f"print('wrote {path}', flush=True)\n")


def raptor_alt_script(base_script, kind):
    """The published Raptor script (same checkpoint and pinning) run on the alternative series, own output files."""
    k = KINDS[kind]
    s = base_script.replace(f"%%writefile {k['script']}", f"%%writefile /tmp/{k['ours']}_run.py", 1)
    for suffix in ("raw", "sub"):
        a, b = f"/kaggle/working/{k['raptor']}_{suffix}.csv", f"/kaggle/working/{k['ours']}_{suffix}.csv"
        assert a in s, a
        s = s.replace(a, b)
    assert s.count(SER_LINE) == 1, "series table line not found"
    s = s.replace(SER_LINE, SER_LINE + ALT_SER)
    return add_status(s, k["ours"])


def add_status(s, ours):
    """Explicit per-study success: studies without usable slices are not inferred, and every study that went
    through all arms without an exception is flagged in <ours>_status.csv (read by the blend)."""
    edits = [
        ("    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]\n",
         "    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]\n"
         "    _ok = np.zeros(N, bool)   # [ours] explicit per-study success\n"),
        ("            sid, vol, mask = item; i = pos[sid]\n",
         "            sid, vol, mask = item; i = pos[sid]\n"
         "            if int(np.asarray(mask).sum()) < 3:   # [ours] preprocessing failed / no usable slices: no prediction\n"
         "                print(f\"  [gpu {d}] study {i} {sid[:16]} SKIPPED (no usable slices)\", flush=True)\n"
         "                with lock:\n"
         "                    done_n[0] += 1\n"
         "                q.task_done()\n"
         "                continue\n"),
        ("            except Exception as e:\n                print(f\"  [gpu {d}] study {i} {sid[:16]} FALLBACK",
         "                _ok[i] = True   # [ours] every arm finished for this study\n"
         "            except Exception as e:\n                print(f\"  [gpu {d}] study {i} {sid[:16]} FALLBACK"),
        ("    q.join()\n    for th in threads: th.join()\n",
         "    q.join()\n    for th in threads: th.join()\n"
         f"    pd.DataFrame({{'StudyInstanceUID': test_ids, 'ok': _ok}}).to_csv('/kaggle/working/{ours}_status.csv', index=False)\n"
         "    print('[ours] studies ok', int(_ok.sum()), 'of', N, flush=True)\n"),
    ]
    for a, b in edits:
        assert s.count(a) == 1, a[:60]
        s = s.replace(a, b)
    return s


RUN_CELL = '''# Our coat96 fine-tunes (OAI MOAKS + competition labels), {name} stage. Runs only if the 9 h clock allows;
# any failure or timeout leaves the blend exactly as the base recipe.
import os, sys, time, subprocess
{flag} = False
_left = 9 * 3600 - (time.time() - NB_T0) - 30 * 60          # keep 30 min for the blend and the margin
if _left < 40 * 60:
    print(f"{flag}_SKIPPED only {{_left / 60:.0f}} min left", flush=True)
else:
    _t = time.time()
    try:
        _rc = subprocess.run([sys.executable, '-u', '/tmp/{ours}_run.py'], timeout=min(_left, 3 * 3600)).returncode
        {flag} = _rc == 0 and os.path.isfile('/kaggle/working/{ours}_sub.csv')
        print(f"{tag}_STAGE rc={{_rc}} ok={{{flag}}} seconds {{time.time() - _t:.0f}}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"{tag}_TIMEOUT after {{time.time() - _t:.0f}}s - blending without it", flush=True)
'''
FLAGS = {"384": ("O96_OK", "O96"), "384alt": ("OALT_OK", "OALT"), "224": ("O224_OK", "O224"),
         "r384alt": ("RALT_OK", "RALT"), "r224alt": ("R224ALT_OK", "R224ALT")}

LOAD_O96 = '''A96 = {a96}   # share of the Raptor slot given to our coat96 fine-tunes; fixed before any run
A96_C = {a96_c}   # per-finding overrides of A96
B3 = {b3}   # three-way mode: share of the final rank given to our stages, taken from both base sides; fixed before any run
B3_C = {b3_c}   # per-finding overrides of B3


def _load_ours(ok, name):
    """(ranks frame, per-study failure mask) of one of our stages, or (None, None)."""
    if not ok:
        return None, None
    try:
        _r = pd.read_csv(WORK / (name + '_sub.csv'), dtype={{'StudyInstanceUID': str}})
        assert sorted(_r['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_r.columns)
        _r = _r.set_index('StudyInstanceUID').loc[test_ids, labels]
        assert np.isfinite(_r.values).all()
        _w = pd.read_csv(WORK / (name + '_raw.csv'), dtype={{'StudyInstanceUID': str}})
        _w = _w.set_index('StudyInstanceUID').loc[test_ids, labels]
        _st = pd.read_csv(WORK / (name + '_status.csv'), dtype={{'StudyInstanceUID': str}}).set_index('StudyInstanceUID')
        # explicit success flag from our stage, plus the script's 0.5 fallback rows as a second guard
        _fail = ~_st.loc[test_ids, 'ok'].astype(bool).values | (_w.values == 0.5).all(axis=1)
        print(name.upper() + '_STUDIES', len(test_ids), 'failed', int(_fail.sum()), flush=True)
        if _fail.mean() > 0.03:
            raise RuntimeError('%s failed on %d of %d studies' % (name, _fail.sum(), len(test_ids)))
        return _r, _fail
    except Exception as _e:
        print(name.upper() + '_UNUSABLE', repr(_e), flush=True)
        return None, None


r96, f96 = _load_ours(globals().get('O96_OK', False), 'ours96')
r224o, f224o = _load_ours(globals().get('O224_OK', False), 'ours224')
for _name, _r in {required}:   # --require: the stage must still be usable after the CSV / mask checks
    if (r96 if _name == 'ours96' else r224o) is None:
        raise RuntimeError(_name + ' required but unusable after validation')
if r96 is not None and globals().get('OALT_OK', False):
    # second view from alternative series of the same type: mean of the two probabilities where it scored
    try:
        _rd = lambda f: pd.read_csv(WORK / f, dtype={{'StudyInstanceUID': str}}).set_index('StudyInstanceUID')
        _wa, _w96 = _rd('ours96alt_raw.csv').loc[test_ids, labels], _rd('ours96_raw.csv').loc[test_ids, labels]
        _oka = (_rd('ours96alt_status.csv').loc[test_ids, 'ok'].astype(bool).values
                & ~(_wa.values == 0.5).all(axis=1) & ~f96 & np.isfinite(_wa.values).all(axis=1))
        _avg = _w96.values.astype(np.float64).copy()
        _avg[_oka] = 0.5 * (_w96.values[_oka] + _wa.values[_oka])
        r96 = pd.DataFrame(_avg, index=_w96.index, columns=labels)   # re-ranked over its non-failed rows below
        print('OURS96ALT_STUDIES', len(test_ids), 'second view', int(_oka.sum()), flush=True)
        ALT_N = int(_oka.sum())
    except Exception as _e:
        print('OURS96ALT_UNUSABLE', repr(_e), flush=True)
RAW_ALT = {{}}   # second view for the public Raptor models: mean of the two raw probabilities where it scored
for _nm, _base, _flag in (('raptoralt', 'raw', 'RALT_OK'), ('raptor224alt', 'raw224', 'R224ALT_OK')):
    if globals().get(_flag, False):
        try:
            _rd2 = lambda f: pd.read_csv(WORK / f, dtype={{'StudyInstanceUID': str}}).set_index('StudyInstanceUID')
            _wa = _rd2(_nm + '_raw.csv').loc[test_ids, labels]
            _oka = (_rd2(_nm + '_status.csv').loc[test_ids, 'ok'].astype(bool).values
                    & ~(_wa.values == 0.5).all(axis=1) & np.isfinite(_wa.values).all(axis=1))
            _b = globals()[_base].values.astype(np.float64).copy()
            _b[_oka] = 0.5 * (_b[_oka] + _wa.values[_oka])
            RAW_ALT[_base] = pd.DataFrame(_b, index=globals()[_base].index, columns=labels)
            print(_nm.upper() + '_STUDIES', len(test_ids), 'second view', int(_oka.sum()), flush=True)
        except Exception as _e:
            print(_nm.upper() + '_UNUSABLE', repr(_e), flush=True)
if {alt_required} and globals().get('ALT_N', 0) < 0.1 * len(test_ids):
    raise RuntimeError('second view required but used on fewer than 10% of the studies')
n = len(test_ids)
'''

MIX_O96 = '''    # our coat96 fine-tunes share the Raptor slot (384 / 224 at 0.65 / 0.35 when both ran); each is ranked over
    # its own non-failed studies, and a study it failed on keeps the base Raptor rank
    _num, _den = np.zeros(n), np.zeros(n)
    for w_, r_, f_ in ((0.65, r96, f96), (0.35, r224o, f224o)):
        if r_ is not None:
            ok_ = ~rap_fail & ~f_
            v_ = pct_on(r_[c].values, ok_)
            _num[ok_] += w_ * v_[ok_]
            _den[ok_] += w_
    has_ = _den > 0
    if _den.any() and A96_C.get(c, A96) > 0:
        a_ = A96_C.get(c, A96)
        rr = rr.copy()
        rr[has_] = (1.0 - a_) * rr[has_] + a_ * _num[has_] / _den[has_]
    wr = W_RAPTOR_C.get(c, W_RAPTOR)
'''


MIX_B3 = '''    # three-way mode: our stages keep their own share of the final rank (taken from both base sides) on the
    # studies they scored; failed studies keep the base blend
    b3_ = B3_C.get(c, B3)
    if b3_ > 0 and has_.any():
        b = b.copy()
        b[has_] = (1.0 - b3_) * b[has_] + b3_ * _num[has_] / _den[has_]
    out[:, j] = b
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", nargs="+", required=True, help="Kaggle dataset(s) holding our checkpoints")
    ap.add_argument("--files", nargs="*", default=[], help="our 384 checkpoints")
    ap.add_argument("--files224", nargs="*", default=[], help="our 224 (0.8 crop) checkpoints")
    ap.add_argument("--a96", type=float, required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--require", action="store_true",
                    help="raise if our stage is skipped or fails (diagnostic: shows on the LB as a notebook error)")
    ap.add_argument("--no-jpeg", action="store_true", help="skip the q95 JPEG round-trip of the test volumes")
    ap.add_argument("--a96-per-label", default="", help='e.g. "Effusion=1.0,Synovitis=1.0,ACL=0.5"')
    ap.add_argument("--b3", type=float, default=0.0, help="three-way share of the final rank for our stages")
    ap.add_argument("--b3-per-label", default="")
    ap.add_argument("--alt", action="store_true", help="add the alternative-series second view to the 384 stage")
    ap.add_argument("--raptor-alt", action="store_true", help="second view for the public Raptor 384 and 224 models too")
    ap.add_argument("--r224-mirror", action="store_true", help="anatomical-mirror TTA for the public 224 model")
    ap.add_argument("--compact-alt", action="store_true", help="store the second-view stage scripts compressed")
    args = ap.parse_args()

    per_label = {}
    for item in filter(None, args.a96_per_label.split(",")):
        k, v = item.rsplit("=", 1)
        per_label[k.strip()] = float(v)
    b3_label = {}
    for item in filter(None, args.b3_per_label.split(",")):
        k, v = item.rsplit("=", 1)
        b3_label[k.strip()] = float(v)
    base = Path(args.base)
    nb = json.loads(base.read_text())
    cells = nb["cells"]
    assert args.files or args.files224, "no checkpoints given"
    i_blend = next(i for i, c in enumerate(cells) if "W_RAPTOR, W_OURS" in src(c))
    if args.r224_mirror:   # patch the published 224 script before any stage is derived from it
        i224 = next(i for i, c in enumerate(cells) if src(c).startswith("%%writefile /tmp/raptor224_run.py"))
        cells[i224]["source"] = add_r224_mirror(src(cells[i224]))
    blend = src(cells[i_blend])
    required = [(KINDS[k]["ours"], None) for k, f in (("384", args.files), ("224", args.files224)) if f and args.require]
    loop_old = "    r384, r224 = pct_on(raw[c].values, ~r384_fail), pct_on(raw224[c].values, ~r224_fail)"
    if args.raptor_alt:
        assert blend.count(loop_old) == 1, "raptor rank line not found"
        blend = blend.replace(loop_old, "    r384, r224 = pct_on(RAW_ALT.get('raw', raw)[c].values, ~r384_fail), "
                                        "pct_on(RAW_ALT.get('raw224', raw224)[c].values, ~r224_fail)   # [ours] second view")
    for old, new in (("n = len(test_ids)\n", LOAD_O96.format(a96=args.a96, a96_c=per_label, b3=args.b3, b3_c=b3_label, required=required,
                                                     alt_required=bool(args.alt and args.require))),
                     ("    wr = W_RAPTOR_C.get(c, W_RAPTOR)\n", MIX_O96),
                     ("    out[:, j] = b\n", MIX_B3)):
        assert blend.count(old) == 1, old
        blend = blend.replace(old, new)
    blend = blend.replace("'mode': 'blend',", "'mode': 'blend', 'a96': A96, 'a96_c': A96_C, 'o96_used': r96 is not None, "
                                              "'o224_used': r224o is not None, 'b3': B3, 'b3_c': B3_C,")
    cells[i_blend]["source"] = blend
    new = []
    for kind, files in (("384", args.files), ("384alt", args.files if args.alt else []), ("224", args.files224),
                        ("r384alt", ["release"] if args.raptor_alt else []), ("r224alt", ["release"] if args.raptor_alt else [])):
        if not files:
            continue
        k = KINDS[kind]
        flag, tag = FLAGS[kind]
        i_script = next(i for i, c in enumerate(cells) if src(c).startswith(f"%%writefile {k['script']}"))
        run_cell = RUN_CELL.format(name=kind, flag=flag, ours=k["ours"], tag=tag)
        if args.require:
            run_cell += f"if not {flag}:\n    raise RuntimeError('{tag} stage required but not completed (skipped, timed out or failed)')\n"
        script = (raptor_alt_script(src(cells[i_script]), kind) if kind.startswith("r") and kind.endswith("alt")
                  else ours_script(src(cells[i_script]), files, kind, jpeg=not args.no_jpeg))
        if args.compact_alt and kind.endswith("alt"):
            script = compact_script(script)
        new += [code_cell(script), code_cell(run_cell)]
    nb["cells"] = cells[:i_blend] + new + cells[i_blend:]

    out = ROOT / "kaggle/kernels" / args.slug
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((base.parent / "kernel-metadata.json").read_text())
    meta.update(id=f"gkdnzz31/{args.slug}", title=args.slug, code_file=f"{args.slug}.ipynb")
    meta["dataset_sources"] = list(dict.fromkeys(meta["dataset_sources"] + args.dataset))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} (stages inserted before cell {i_blend}; A96={args.a96}; 384 {args.files}; 224 {args.files224})")


if __name__ == "__main__":
    main()

"""Build a Kaggle notebook that runs a public stack notebook unchanged, then our models, then blends.

Cells: <public notebook cells> -> save its submission -> free GPU -> run our inference script in a
subprocess -> per-label rank blend (stack weight W_STACK). If our part fails, the stack's
submission is kept.

Usage:
    .venv/bin/python scripts/build_stack_notebook.py --public outputs/public/fast2xt4 \
        --slug knee-stack-blend --our-runs knee-train-pl2-v2-cns knee-train-pl2-v3 --w-stack 0.7
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "gkdnzz31"


def code_cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", nargs="+", required=True, help="folder(s) with a pulled public notebook + metadata")
    ap.add_argument("--w-public", type=float, nargs="*", default=None,
                    help="relative weights of the public notebooks (default equal)")
    ap.add_argument("--slug", required=True)
    ap.add_argument("--our-runs", nargs="*", default=[], help="our training kernels to attach (none: public blend only)")
    ap.add_argument("--weights", default="{}", help="JSON group weights for our part")
    ap.add_argument("--raptor-ft", nargs="*", default=[], help="fine-tuned Raptor checkpoint names")
    ap.add_argument("--w-stack", type=float, default=0.7)
    args = ap.parse_args()

    publics = []
    for folder in args.public:
        pub = Path(folder)
        meta_i = json.loads((pub / "kernel-metadata.json").read_text())
        publics.append((meta_i, json.loads((pub / meta_i["code_file"]).read_text())))
    meta, nb = publics[0]
    w_public = args.w_public or [1.0] * len(publics)
    assert len(w_public) == len(publics)

    our_script = "\n".join([
        (ROOT / "kaggle/kaggle_paths.py").read_text(),
        f"INCLUDE_GROUPS = {args.our_runs!r}",
        'OUT_CSV = "/kaggle/working/ours.csv"',
        f"GROUP_WEIGHTS = {json.loads(args.weights)!r}",
        f"RAPTOR_FT_FILES = {args.raptor_ft!r}",
        (ROOT / "kaggle/kernels/infer/infer.py").read_text(),
    ])
    def save_and_free(i):
        return code_cell(
            f"# --- keep public notebook {i}'s submission and release GPU memory ---\n"
            "import gc, os, shutil, torch\n"
            f"if os.path.exists('/kaggle/working/submission.csv'):\n"
            f"    shutil.move('/kaggle/working/submission.csv', '/kaggle/working/public_{i}.csv')\n"
            "for _name, _val in list(globals().items()):\n"
            "    if isinstance(_val, (torch.nn.Module, torch.Tensor)) and not _name.startswith('_'):\n"
            "        del globals()[_name]\n"
            "gc.collect(); torch.cuda.empty_cache()\n"
        )

    cells = []
    for i, (_, nb_i) in enumerate(publics):
        cells += nb_i["cells"] + [save_and_free(i)]
    cells += [
        code_cell(
            "# --- our models (separate process so the stack's state cannot interfere) ---\n"
            ""
            f"open('/kaggle/working/our_infer.py', 'w').write({our_script!r})\n"
            "import subprocess, sys\n"
            "_res = subprocess.run([sys.executable, '/kaggle/working/our_infer.py'], capture_output=True, text=True)\n"
            "print(_res.stdout[-4000:]); print(_res.stderr[-4000:])\n"
        ),
        code_cell(
            "# --- per-label rank blend of the public notebooks, then with ours (skipping any that failed) ---\n"
            "import os, numpy as np, pandas as pd\n"
            f"W_STACK = {args.w_stack}\n"
            f"W_PUBLIC = {w_public!r}\n"
            "parts = [(w, pd.read_csv(f'/kaggle/working/public_{i}.csv')) for i, w in enumerate(W_PUBLIC)\n"
            "         if os.path.exists(f'/kaggle/working/public_{i}.csv')]\n"
            "print('public parts available:', len(parts), 'of', len(W_PUBLIC))\n"
            "base_ids = parts[0][1]['StudyInstanceUID']\n"
            "cols = list(parts[0][1].columns[1:])\n"
            "def ranks(df):\n"
            "    df = df.set_index('StudyInstanceUID').loc[base_ids]\n"
            "    return np.stack([df[c].rank(pct=True).values for c in cols], 1)\n"
            "wsum = sum(w for w, _ in parts)\n"
            "stack = sum(w * ranks(df) for w, df in parts) / wsum\n"
            "try:\n"
            "    final = W_STACK * stack + (1 - W_STACK) * ranks(pd.read_csv('/kaggle/working/ours.csv'))\n"
            "    print('blended with our models, W_STACK =', W_STACK)\n"
            "except Exception as exc:\n"
            "    final = stack\n"
            "    print('our part unavailable, using the public blend only:', repr(exc))\n"
            "assert np.isfinite(final).all()\n"
            "out = pd.DataFrame(final, columns=cols); out.insert(0, 'StudyInstanceUID', base_ids.values)\n"
            "out.to_csv('/kaggle/working/submission.csv', index=False)\n"
            "print(out.head())\n"
        ),
    ]
    if not args.our_runs:  # public-only blend: drop the cell that runs our models
        cells = [c for c in cells if "our_infer.py" not in "".join(c["source"])]
    nb["cells"] = cells

    out_dir = ROOT / "outputs/kernels" / args.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notebook.ipynb").write_text(json.dumps(nb))
    datasets = [d for m, _ in publics for d in m.get("dataset_sources", []) if d] + [f"{USER}/knee-code"]
    if args.raptor_ft:
        datasets.append("dreaddevelopment/raptor-knee-finespacing")
    new_meta = {
        "id": f"{USER}/{args.slug}", "title": args.slug, "code_file": "notebook.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": sorted(set(datasets)),
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": sorted(set([k for m, _ in publics for k in m.get("kernel_sources", []) if k] +
                                     [f"{USER}/{r}" for r in args.our_runs])),
        "model_sources": sorted(set(x for m, _ in publics for x in m.get("model_sources", []) if x)),
    }
    (out_dir / "kernel-metadata.json").write_text(json.dumps(new_meta, indent=2))
    print(out_dir, "| cells", len(cells), "| datasets", len(new_meta["dataset_sources"]),
          "| kernels", new_meta["kernel_sources"])


if __name__ == "__main__":
    main()

"""Weight interpolation of two checkpoints of the same model: alpha * A + (1 - alpha) * B (WiSE-FT style).

Usage: python3 scripts/interp_ckpt.py --a soup.pt --b release.pt --alpha 0.75 --out wise75.pt
"""

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ca, cb = (torch.load(p, map_location="cpu", weights_only=False) for p in (args.a, args.b))
    ma, mb = ca.get("model", ca), cb.get("model", cb)
    assert ma.keys() == mb.keys(), "different parameter names"
    if ca.get("arch") and cb.get("arch"):
        assert ca["arch"] == cb["arch"], "different architectures"
    assert int(ca.get("res", 384)) == int(cb.get("res", 384)), "different resolutions"
    out = {}
    for k, v in ma.items():
        assert v.shape == mb[k].shape, k
        out[k] = (args.alpha * v.float() + (1 - args.alpha) * mb[k].float()).to(v.dtype) if v.is_floating_point() else v
    torch.save({"model": out, "arch": ca.get("arch"), "res": ca.get("res", 384),
                "interp": {"a": args.a, "b": args.b, "alpha": args.alpha}}, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()

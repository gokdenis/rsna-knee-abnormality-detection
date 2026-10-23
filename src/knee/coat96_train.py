"""Fine-tune the coat96 CoAtNet (nartaa's OAI checkpoint) on competition + OAI MOAKS targets.

Targets may be NaN (finding not scored for that study, e.g. MCL/Contusion/Fracture for OAI knees);
those entries are masked out of the loss. Checkpoints are saved in the release format
({"model", "arch", "res": 384}) so the public inference script can load them unchanged.

Validation: the 58 competition gold studies (never trained on) and a subject-level OAI holdout
scored against MOAKS targets (>= 0.5 counts as positive).
"""

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler

from . import coat96
from .constants import LABELS
from .data import SliceStore

# many small worker tensors are kept during evaluation; file descriptors run out otherwise
torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class Coat96Config:
    epochs: int = 4
    batch_size: int = 2
    k_train: int = 16
    lr_backbone: float = 2e-5
    lr_head: float = 2e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.03
    oai_weight: float = 1.0        # loss weight of OAI studies (scales their gradient; loss / number of known cells)
    oai_fraction: float = 1.0      # share of the OAI training knees drawn each epoch
    oai_holdout: float = 0.05      # share of OAI subjects kept out for validation
    holdout_seed: int = 2026       # fixed, independent of `seed`, so every run is scored on the same OAI subjects
    epoch_size: int = 0            # studies drawn per epoch (0 = every training study once); a control run
                                   # without OAI uses the OAI run's epoch size to match the optimizer steps
    mirror_p: float = 0.5
    grad_ckpt: bool = False        # 48 GB fits batch 4 x 16 windows without it (~30% faster)
    workers: int = 6
    eval_every: int = 0            # extra gold evaluations every N steps (0 = only per epoch)
    time_budget_min: float = 600.0
    seed: int = 42
    tag: str = "a"


class CachedStudies(Dataset):
    """(uint8 windows [K, 3, IMG, IMG], centres [K], targets [12] with NaN, loss weight)."""

    def __init__(self, cache_dir, rows, targets, weight, k, train, alt=None, alt_p=0.5):
        self.cache_dir = cache_dir
        self.masks = np.load(f"{cache_dir}/masks.npy")
        self.store = SliceStore(cache_dir)
        self.rows, self.targets = np.asarray(rows), np.asarray(targets, np.float32)
        self.weight, self.k, self.train = float(weight), k, train
        # alt = (cache_dir, rows aligned with `rows`, -1 = none): in training, a study with an alternative-series view
        # uses it with probability alt_p instead of its primary view (same number of samples and steps)
        self.alt = None
        if alt is not None:
            self.alt = (np.load(f"{alt[0]}/masks.npy"), SliceStore(alt[0]), np.asarray(alt[1]))
            self.alt_p = float(alt_p)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        masks, store, row = self.masks, self.store, self.rows[i]
        if self.train:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            if self.alt is not None and self.alt[2][i] >= 0 and rng.random() < self.alt_p:
                masks, store, row = self.alt[0], self.alt[1], self.alt[2][i]
            c = coat96.train_centers(masks[row], self.k, rng)
        else:
            c = coat96.eval_centers(masks[row])
        needed = np.unique(np.concatenate([c - 1, c, c + 1]))
        vol = store.study(row, needed, coat96.MAXS, coat96.IMG)
        x = np.stack([vol[c - 1], vol[c], vol[c + 1]], axis=1)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(c), torch.from_numpy(self.targets[i]),
                torch.tensor(self.weight))


def _rows(cache_dir, ids):
    row_of = {s: i for i, s in enumerate(np.load(f"{cache_dir}/ids.npy", allow_pickle=True).astype(str))}
    return np.array([row_of.get(s, -1) for s in ids])


def competition_sets(cache_dirs, labels):
    """Train and gold datasets per competition cache shard. labels: StudyInstanceUID, is_gold, LABELS."""
    train, gold = [], []
    for d in cache_dirs:
        rows = _rows(d, labels["StudyInstanceUID"].astype(str))
        part = labels.assign(row=rows)[rows >= 0]
        train.append((d, part[part["is_gold"] == 0]))
        gold.append((d, part[part["is_gold"] == 1]))
    return train, gold


def oai_sets(cache_dirs, oai_labels, holdout, seed):
    """Train and holdout (subject-level split) datasets per OAI cache. oai_labels: ID, SIDE, VISIT, LABELS."""
    lab = oai_labels.assign(key="OAI-" + oai_labels["ID"].astype(str) + "-" + oai_labels["SIDE"] + "-" + oai_labels["VISIT"])
    subjects = np.array(sorted(lab["ID"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(subjects, size=int(round(holdout * len(subjects))), replace=False))
    train, val = [], []
    for d in cache_dirs:
        rows = _rows(d, lab["key"])
        part = lab.assign(row=rows)[rows >= 0]
        is_held = part["ID"].astype(str).isin(held)
        train.append((d, part[~is_held]))
        val.append((d, part[is_held]))
    return train, val


def masked_auc(y, p):
    """Per-label AUC over rows with a clear target; NaN when a label has one class only.

    Targets in (0.4, 0.6) are left out: reader ties after the consensus (0.5), partial ACL tears (0.5).
    The same mask applies to every run, so runs stay comparable.
    """
    out = {}
    for j, name in enumerate(LABELS):
        known = ~np.isnan(y[:, j]) & ~((y[:, j] > 0.4) & (y[:, j] < 0.6))
        yy = (y[known, j] >= 0.5).astype(int)
        out[name] = roc_auc_score(yy, p[known, j]) if known.any() and 0 < yy.sum() < len(yy) else float("nan")
    return out


@torch.no_grad()
def predict(model, datasets, device, workers, view=coat96.VIEWS["r320"]):
    model.eval()
    ys, ps = [], []
    for ds in datasets:
        if len(ds) == 0:
            continue
        for x, c, y, _ in DataLoader(ds, batch_size=1, num_workers=workers):
            p = 0
            flips = (False, True) if view.mirror_tta else (False,)  # anatomical-mirror TTA where published
            for flip in flips:
                xx = coat96.mirror(x, c[0].numpy()) if flip else x
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = p + torch.sigmoid(model(coat96.to_input(xx, device, view=view)).float()) / len(flips)
            ps.append(p.cpu().numpy())
            ys.append(y.numpy().copy())  # do not keep the worker's shared-memory tensor alive
    return (np.concatenate(ys), np.concatenate(ps)) if ys else (np.zeros((0, 12)), np.zeros((0, 12)))


def _augment(x, c, cfg, rng, view):
    """Random crop offset (shared in the batch) and per-sample anatomical mirror."""
    if cfg.mirror_p > 0:
        flip = torch.from_numpy(rng.random(len(x)) < cfg.mirror_p)
        for i in torch.where(flip)[0].tolist():
            x[i] = coat96.mirror(x[i], c[i].numpy())
    m = coat96.IMG - view.crop
    return x, (int(rng.integers(0, m + 1)), int(rng.integers(0, m + 1)))


def finetune(cfg: Coat96Config, weights, comp_caches, comp_labels, oai_caches, oai_labels, out_dir, device="cuda",
             comp_extra=(), comp_alt=None):
    """comp_extra: caches with extra views of competition studies (e.g. alternative series); train rows only, so the
    gold58 validation is unchanged."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    comp_train, comp_gold = competition_sets(comp_caches, comp_labels)
    if comp_extra:
        comp_train = comp_train + competition_sets(list(comp_extra), comp_labels)[0]
    oai_train, oai_val = oai_sets(oai_caches, oai_labels, cfg.oai_holdout, cfg.holdout_seed) if oai_caches else ([], [])

    def make(parts, weight, train):
        return [CachedStudies(d, p["row"], p[LABELS].values, weight, cfg.k_train if train else None, train)
                for d, p in parts if len(p)]

    gold_sets, oai_val_sets = make(comp_gold, 1.0, False), make(oai_val, 1.0, False)
    if comp_alt:  # view swap: same rows and steps as without it
        comp_sets = [CachedStudies(d, p["row"], p[LABELS].values, 1.0, cfg.k_train, True,
                                   alt=(comp_alt, _rows(comp_alt, p["StudyInstanceUID"].astype(str))))
                     for d, p in comp_train if len(p)]
        print(f"[{cfg.tag}] view swap from {comp_alt}: "
              f"{sum(int((ds.alt[2] >= 0).sum()) for ds in comp_sets)} train studies have an alternative", flush=True)
    else:
        comp_sets = make(comp_train, 1.0, True)
    n_comp = sum(len(d) for d in comp_sets)
    n_oai = sum(len(p) for _, p in oai_train)
    print(f"[{cfg.tag}] train: competition {n_comp}, OAI {n_oai} (x{cfg.oai_fraction} per epoch); "
          f"val: gold {sum(len(d) for d in gold_sets)}, OAI {sum(len(d) for d in oai_val_sets)}", flush=True)

    model, view = coat96.load_model_view(weights, device)
    arch = torch.load(weights, map_location="cpu", weights_only=False).get("arch", coat96.ARCH)
    print(f"[{cfg.tag}] start {weights} | {arch} | view {view.name}", flush=True)
    model.backbone.set_grad_checkpointing(cfg.grad_ckpt)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
                             {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)
    steps_per_epoch = (cfg.epoch_size or (n_comp + int(cfg.oai_fraction * n_oai))) // cfg.batch_size
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))

    def evaluate(label):
        res = {}
        y, p = predict(model, gold_sets, device, cfg.workers, view)
        if len(y):
            g = masked_auc(y, p)
            res["gold_auc"] = float(np.nanmean(list(g.values())))
            res["gold_per_label"] = g
        y, p = predict(model, oai_val_sets, device, cfg.workers, view)
        if len(y):
            o = masked_auc(y, p)
            res["oai_auc"] = float(np.nanmean(list(o.values())))
            res["oai_per_label"] = o
        print(f"[{cfg.tag}] {label}: gold {res.get('gold_auc', float('nan')):.4f} "
              f"oai {res.get('oai_auc', float('nan')):.4f}", flush=True)
        if "oai_per_label" in res:
            print("   oai per label: " + " ".join(f"{k[:6]} {v:.3f}" for k, v in res["oai_per_label"].items()
                                                  if not np.isnan(v)), flush=True)
        model.train()
        return res

    history, t_start, step = [{"epoch": -1, **evaluate("before fine-tuning")}], time.time(), 0
    for epoch in range(cfg.epochs):
        epoch_sets = list(comp_sets)
        for d, p in oai_train:
            if len(p):
                take = p.sample(frac=cfg.oai_fraction, random_state=int(rng.integers(1 << 30))) if cfg.oai_fraction < 1 else p
                epoch_sets.append(CachedStudies(d, take["row"], take[LABELS].values, cfg.oai_weight, cfg.k_train, True))
        pool = ConcatDataset(epoch_sets)
        sampler = (RandomSampler(pool, replacement=len(pool) < cfg.epoch_size, num_samples=cfg.epoch_size)
                   if cfg.epoch_size else RandomSampler(pool))
        loader = DataLoader(pool, batch_size=cfg.batch_size, sampler=sampler, drop_last=True,
                            num_workers=cfg.workers, pin_memory=True, persistent_workers=False)
        model.train()
        t0, losses = time.time(), []
        for x, c, y, w in loader:
            x, crop = _augment(x, c, cfg, rng, view)
            y, w = y.to(device), w.to(device)
            known = ~torch.isnan(y)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(coat96.to_input(x, device, crop, view))
            loss_el = F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(y), reduction="none")
            # weighted sum over known cells / number of known cells: a source weight (oai_weight) scales that
            # source's gradient directly instead of cancelling out in a weighted mean
            wt = known.float() * w[:, None]
            loss = (loss_el * wt).sum() / known.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step % 200 == 0:
                print(f"[{cfg.tag}] ep {epoch} step {step} loss {np.mean(losses[-200:]):.4f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if cfg.eval_every and step % cfg.eval_every == 0:
                history.append({"epoch": epoch, "step": step, **evaluate(f"step {step}")})
        minutes = (time.time() - t0) / 60
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "minutes": minutes, **evaluate(f"epoch {epoch}")})
        torch.save({"model": model.state_dict(), "arch": arch, "res": view.ck_res, "config": asdict(cfg), "epoch": epoch},
                   f"{out_dir}/coat96_{cfg.tag}_ep{epoch}.pt")
        json.dump(history, open(f"{out_dir}/coat96_{cfg.tag}_log.json", "w"), indent=1)
        if (time.time() - t_start) / 60 + minutes > cfg.time_budget_min:
            print(f"[{cfg.tag}] time budget reached", flush=True)
            break
    return history


@torch.no_grad()
def predict_caches(weights, cache_dirs, device="cuda", workers=6):
    """Predictions (with the view's published TTA) of a checkpoint for every study of the given caches -> DataFrame (id, LABELS)."""
    model, view = coat96.load_model_view(weights, device)
    frames = []
    for d in cache_dirs:
        ids = np.load(f"{d}/ids.npy", allow_pickle=True).astype(str)
        ds = CachedStudies(d, np.arange(len(ids)), np.full((len(ids), len(LABELS)), np.nan), 1.0, None, False)
        _, p = predict(model, [ds], device, workers, view)
        frames.append(pd.DataFrame(p, columns=LABELS).assign(id=ids))
    return pd.concat(frames, ignore_index=True)[["id"] + LABELS]


def swa(paths, out_path):
    """Average the weights of several epoch checkpoints (the release used SWA over three epochs)."""
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    avg = {}
    for k, v in cks[0]["model"].items():
        avg[k] = (sum(c["model"][k].float() for c in cks) / len(cks)).to(v.dtype) if v.is_floating_point() else v
    torch.save({"model": avg, "arch": cks[0].get("arch", coat96.ARCH), "res": cks[0].get("res", 384),
                "swa_of": list(map(str, paths))},
               out_path)

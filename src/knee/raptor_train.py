"""Fine-tune the public Raptor CoAtNet on our pseudo-labels, using the Raptor-format JPEG cache.

The cache stores the published Raptor sampling (80 slices x 336 px) as JPEG, so the fine-tuned
model must see JPEG round-tripped stacks at test time as well (see infer.py).
"""

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .constants import ID_COL, LABELS
from .data import SliceStore
from .raptor import IMG, MAXS, MEAN, RES, STD, load_raptor, raptor_centers
from .train import macro_auc


@dataclass
class RaptorFTConfig:
    epochs: int = 4
    batch_size: int = 2
    k_train: int = 12
    lr_backbone: float = 2e-5
    lr_head: float = 2e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    workers: int = 2
    time_budget_min: float = 600.0
    seed: int = 42
    tag: str = "a"


class RaptorStudies(Dataset):
    def __init__(self, cache_dir, masks, rows, targets, k, train):
        self.store, self.masks = SliceStore(cache_dir), masks
        self.rows, self.targets = np.asarray(rows), np.asarray(targets, np.float32)
        self.k, self.train = k, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row, mask = self.rows[i], self.masks[self.rows[i]]
        if self.train:
            valid = np.where(mask > 0)[0]
            lo, hi = (int(valid.min()), int(valid.max())) if len(valid) >= 3 else (0, 2)
            pool = np.arange(lo + 1, hi) if hi - lo >= 2 else np.array([1])
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,))))
            c = rng.choice(pool, size=self.k, replace=len(pool) < self.k)
        else:
            c = raptor_centers(mask)
        needed = np.unique(np.concatenate([c - 1, c, c + 1]))
        vol = self.store.study(row, needed, MAXS, IMG)
        x = np.stack([vol[c - 1], vol[c], vol[c + 1]], axis=1)  # [K, 3, IMG, IMG] uint8
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.targets[i])


def to_raptor_input(x_uint8, device):
    b, k = x_uint8.shape[:2]
    x = x_uint8.to(device, non_blocking=True).flatten(0, 1).float() / 255.0
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    return x.view(b, k, *x.shape[1:])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(to_raptor_input(x, device)).float()).cpu().numpy())
    return np.concatenate(out)


def finetune(cfg: RaptorFTConfig, weights: str, cache_dir: str, labels: pd.DataFrame, out_dir: str, device: str):
    torch.manual_seed(cfg.seed)
    ids = np.load(f"{cache_dir}/ids.npy", allow_pickle=True).astype(str)
    masks = np.load(f"{cache_dir}/masks.npy")
    row_of = {sid: i for i, sid in enumerate(ids)}
    labels = labels[labels[ID_COL].isin(row_of)].reset_index(drop=True)
    labels["row"] = labels[ID_COL].map(row_of)
    trn, gold = labels[labels["is_gold"] == 0], labels[labels["is_gold"] == 1]

    dl_trn = DataLoader(RaptorStudies(cache_dir, masks, trn["row"], trn[LABELS], cfg.k_train, True),
                        batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.workers,
                        pin_memory=True, persistent_workers=True)
    dl_gold = DataLoader(RaptorStudies(cache_dir, masks, gold["row"], gold[LABELS], None, False),
                         batch_size=1, num_workers=cfg.workers)

    model = load_raptor(weights, device)
    model.backbone.set_grad_checkpointing(True)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
                             {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(dl_trn)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    scaler = torch.amp.GradScaler("cuda")

    y_gold = gold[LABELS].values.astype(int)
    base = macro_auc(y_gold, predict(model, dl_gold, device))[0]
    print(f"[{cfg.tag}] gold auc before fine-tuning {base:.4f}", flush=True)
    history, t_start = [{"epoch": -1, "gold_auc": base}], time.time()
    for epoch in range(cfg.epochs):
        model.train()
        t0, losses = time.time(), []
        for step, (x, y) in enumerate(dl_trn):
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(to_raptor_input(x, device))
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
            if step % 200 == 0:
                print(f"[{cfg.tag}] ep {epoch} step {step}/{len(dl_trn)} loss {np.mean(losses):.4f} "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        gold_auc = macro_auc(y_gold, predict(model, dl_gold, device))[0]
        minutes = (time.time() - t0) / 60
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "gold_auc": gold_auc, "minutes": minutes})
        print(f"[{cfg.tag}] ep {epoch} loss {np.mean(losses):.4f} gold_auc {gold_auc:.4f} {minutes:.1f} min", flush=True)
        # Gold is too small to select on safely: keep every epoch, decide later.
        torch.save({"model": model.state_dict(), "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
                    "config": asdict(cfg), "epoch": epoch}, f"{out_dir}/raptorft_{cfg.tag}_ep{epoch}.pt")
        if (time.time() - t_start) / 60 + minutes > cfg.time_budget_min:
            print(f"[{cfg.tag}] time budget reached", flush=True)
            break
    json.dump(history, open(f"{out_dir}/raptorft_{cfg.tag}_log.json", "w"), indent=1)

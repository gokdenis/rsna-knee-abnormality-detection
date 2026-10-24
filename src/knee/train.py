"""One-fold training loop: soft-target BCE, fp16 autocast, cosine schedule, best-epoch checkpoint."""

import json
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .constants import ID_COL, LABELS
from .data import StudyDataset, gpu_augment
from .model import KneeMIL

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass
class Config:
    arch: str = "convnext_tiny.fb_in22k_ft_in1k"
    epochs: int = 12
    batch_size: int = 4          # studies per step
    k_train: int = 12            # windows sampled per study during training
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    workers: int = 2
    max_train_studies: int = 0   # >0: subsample for a smoke run
    pretrained: bool = True
    layout: str = "v1"           # must match the cache the run trains on
    time_budget_min: float = 200.0  # per fold; stop before an epoch that would overrun it
    seed: int = 42


def to_input(x_uint8: torch.Tensor, device, augment: bool) -> torch.Tensor:
    """uint8 [B, K, 3, H, W] -> normalised float [B, K, 3, H, W] on device."""
    b, k = x_uint8.shape[:2]
    x = x_uint8.to(device, non_blocking=True).flatten(0, 1).float() / 255.0
    if augment:
        x = gpu_augment(x)
    x = (x - MEAN.to(device)) / STD.to(device)
    return x.view(b, k, *x.shape[1:])


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, list[float]]:
    aucs = []
    for j in range(y_true.shape[1]):
        y = y_true[:, j]
        aucs.append(roc_auc_score(y, y_pred[:, j]) if 0 < y.sum() < len(y) else float("nan"))
    return float(np.nanmean(aucs)), aucs


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, slot, valid, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            logits = model(to_input(x, device, False), slot.to(device), valid.to(device))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def train_fold(cfg: Config, fold: int, cache_dir: str, labels: pd.DataFrame, out_dir: str, device: str):
    torch.manual_seed(cfg.seed + fold)
    np.random.seed(cfg.seed + fold)
    ids = np.load(f"{cache_dir}/ids.npy", allow_pickle=True).astype(str)
    masks = np.load(f"{cache_dir}/masks.npy")
    row_of = {sid: i for i, sid in enumerate(ids)}
    labels = labels[labels[ID_COL].isin(row_of)].reset_index(drop=True)
    labels["row"] = labels[ID_COL].map(row_of)

    trn = labels[(labels["fold"] != fold) & (labels["is_gold"] == 0)]
    val = labels[labels["fold"] == fold]
    if cfg.max_train_studies:
        trn = trn.sample(min(cfg.max_train_studies, len(trn)), random_state=0)
        val = val.head(max(50, cfg.max_train_studies // 4))

    dl_trn = DataLoader(StudyDataset(cache_dir, masks, trn["row"], trn[LABELS], cfg.k_train, True, cfg.layout),
                        batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0)
    dl_val = DataLoader(StudyDataset(cache_dir, masks, val["row"], val[LABELS], None, False, cfg.layout),
                        batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0)

    model = KneeMIL(cfg.arch, len(LABELS), pretrained=cfg.pretrained).to(device)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
                             {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.weight_decay)
    total = cfg.epochs * len(dl_trn)
    warm = max(1, int(cfg.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    y_val_bin = (val[LABELS].values > 0.5).astype(int)
    gold_mask = val["is_gold"].values == 1
    best, history = -1.0, []
    fold_start = time.time()
    for epoch in range(cfg.epochs):
        if history and (time.time() - fold_start) / 60 + history[-1]["minutes"] > cfg.time_budget_min:
            print(f"[fold {fold}] time budget reached after {epoch} epochs", flush=True)
            break
        model.train()
        t0, losses = time.time(), []
        for x, slot, valid, y in dl_trn:
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(to_input(x, device, True), slot.to(device), valid.to(device))
            loss = F.binary_cross_entropy_with_logits(logits.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())

        pred = predict(model, dl_val, device)
        auc, per = macro_auc(y_val_bin[~gold_mask], pred[~gold_mask])
        gold_auc = macro_auc(y_val_bin[gold_mask], pred[gold_mask])[0] if gold_mask.sum() > 3 else float("nan")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auc": auc,
                        "gold_auc": gold_auc, "minutes": (time.time() - t0) / 60})
        print(f"[fold {fold}] ep {epoch} loss {np.mean(losses):.4f} val_auc {auc:.4f} "
              f"gold_auc {gold_auc:.4f} {(time.time() - t0) / 60:.1f} min", flush=True)
        if auc > best:
            best = auc
            torch.save({"model": model.state_dict(), "arch": cfg.arch, "labels": LABELS,
                        "config": asdict(cfg), "fold": fold, "epoch": epoch},
                       f"{out_dir}/fold{fold}.pt")
            oof = pd.DataFrame(pred, columns=LABELS)
            oof.insert(0, ID_COL, val[ID_COL].values)
            oof.to_csv(f"{out_dir}/oof_fold{fold}.csv", index=False)

    json.dump({"best_val_auc": best, "history": history}, open(f"{out_dir}/log_fold{fold}.json", "w"), indent=1)
    return best

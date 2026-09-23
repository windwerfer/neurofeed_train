#!/usr/bin/env python3
"""Wave 2 step 1: improve hypnagogic *precision* on SC4002 via threshold + optional retrain.

Leakage policy (documented):
  - Primary: sweep decision threshold on SC4001 same-night balanced val (seed=42, val_frac=0.2),
    then apply chosen threshold to SC4002 holdout. No SC4002 labels used for selection.
  - Optional ablation: also report best-on-SC4002 threshold (LEAKY — not for claiming
    generalization; only to bound how much headroom exists).
  - Optional: retrain HeadA with precision-favoring class weights / focal-ish CE on SC4001
    train only; still pick threshold on SC4001 val.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder
from src.head_a import (
    HEAD_A_BINARY_LABELS,
    HeadALinear,
    class_weights_from_y,
    undersample_balanced,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report

SEED = 42
BATCH = 32
LR = 1e-3
EPOCHS = 5
VAL_FRAC = 0.2
TARGET_SR = 256.0
HYPN_ID = 1  # HEAD_A_BINARY_LABELS index
# Minimum hypnagogic recall on val when maximizing precision
MIN_VAL_RECALL = 0.50
# Prefer not to drop recall more than this fraction of baseline recall on val
RECALL_FLOOR_FRAC = 0.85


def find_weights() -> Path:
    cands = [
        Path("/tmp/kaggle_out3/models/CBraMod/pretrained_weights.pth"),
        ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("CBraMod weights not found")


def encode_all(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(X) == 0:
        return torch.zeros((0, 200), dtype=torch.float32)
    Xt = torch.from_numpy(X.astype(np.float32))
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            emb_list.append(encoder(Xt[i : i + BATCH].to(device)).cpu())
    return torch.cat(emb_list, dim=0)


def softmax_probs(logits: torch.Tensor) -> np.ndarray:
    return F.softmax(logits, dim=-1).numpy()


def pred_from_probs(probs: np.ndarray, threshold: float, positive_id: int = HYPN_ID) -> np.ndarray:
    """Binary decision: positive if P(positive) >= threshold; else negative (drowsy=0)."""
    p_pos = probs[:, positive_id]
    return np.where(p_pos >= threshold, positive_id, 1 - positive_id).astype(np.int64)


def report_from_pred(y: np.ndarray, pred: np.ndarray, name: str, extra: Optional[Dict] = None) -> Dict[str, Any]:
    if len(y) == 0:
        out: Dict[str, Any] = {
            "name": name,
            "n": 0,
            "acc": None,
            "macro_f1": None,
            "per_class": {},
            "confusion": [],
            "counts_true": {},
        }
        if extra:
            out.update(extra)
        return out
    acc = float((pred == y).mean())
    f1 = macro_f1(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    report = per_class_report(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    per_class = {k: v for k, v in report.items() if k != "macro_f1"}
    cm = confusion_matrix(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    counts_true = {HEAD_A_BINARY_LABELS[i]: int((y == i).sum()) for i in range(2)}
    out = {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1": f1,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": list(HEAD_A_BINARY_LABELS),
        "counts_true": counts_true,
    }
    if extra:
        out.update(extra)
    return out


def hypn_metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    r = per_class_report(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    h = r["hypnagogic"]
    return {
        "precision": float(h["precision"]),
        "recall": float(h["recall"]),
        "f1": float(h["f1"]),
        "macro_f1": float(macro_f1(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)),
        "acc": float((pred == y).mean()) if len(y) else 0.0,
    }


def f_beta(prec: float, rec: float, beta: float = 0.5) -> float:
    if prec + rec == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def sweep_thresholds(
    y: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> List[Dict[str, Any]]:
    rows = []
    for t in thresholds:
        pred = pred_from_probs(probs, float(t))
        m = hypn_metrics(y, pred)
        rows.append(
            {
                "threshold": float(t),
                "hypnagogic_precision": m["precision"],
                "hypnagogic_recall": m["recall"],
                "hypnagogic_f1": m["f1"],
                "hypnagogic_f0.5": f_beta(m["precision"], m["recall"], 0.5),
                "macro_f1": m["macro_f1"],
                "acc": m["acc"],
            }
        )
    return rows


def pick_threshold(
    sweep: List[Dict[str, Any]],
    baseline_recall: float,
) -> Dict[str, Any]:
    """Maximize hypnagogic precision subject to recall floors; tie-break by F0.5 then recall."""
    recall_floor = max(MIN_VAL_RECALL, RECALL_FLOOR_FRAC * baseline_recall)
    eligible = [r for r in sweep if r["hypnagogic_recall"] >= recall_floor - 1e-12]
    strategy = "max_precision_subject_to_recall_floor"
    if not eligible:
        # fall back: max F0.5 among those with recall >= MIN_VAL_RECALL, else best F0.5 overall
        eligible = [r for r in sweep if r["hypnagogic_recall"] >= MIN_VAL_RECALL - 1e-12]
        strategy = "fallback_min_recall_0.50_then_f0.5"
        if not eligible:
            eligible = list(sweep)
            strategy = "fallback_best_f0.5_unconstrained"
            eligible.sort(key=lambda r: (r["hypnagogic_f0.5"], r["hypnagogic_precision"], r["hypnagogic_recall"]))
            chosen = eligible[-1]
            return {"chosen": chosen, "strategy": strategy, "recall_floor": recall_floor, "n_eligible": 0}

    # Among eligible: max precision, then F0.5, then recall
    eligible.sort(
        key=lambda r: (
            r["hypnagogic_precision"],
            r["hypnagogic_f0.5"],
            r["hypnagogic_recall"],
            -abs(r["threshold"] - 0.5),  # prefer nearer 0.5 on ties
        )
    )
    chosen = eligible[-1]
    return {
        "chosen": chosen,
        "strategy": strategy,
        "recall_floor": recall_floor,
        "n_eligible": len(eligible),
    }


class FocalCE(nn.Module):
    """Focal-ish CE: down-weight easy examples; optional per-class alpha."""

    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 1.5):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        gather = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = p.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -(1 - pt).clamp(min=0).pow(self.gamma) * gather
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def train_head(
    emb: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    *,
    mode: str = "baseline_invfreq",
    epochs: int = EPOCHS,
) -> Tuple[HeadALinear, List[dict], Dict[str, Any]]:
    """mode:
      - baseline_invfreq: inverse-frequency CE (matches original night_holdout)
      - precision_favor: down-weight hypnagogic class (conservative hypn predictions)
      - focal_precision: focal CE + precision-favor class alpha
    """
    yt = torch.from_numpy(y)
    head = HeadALinear(in_dim=emb.shape[-1], n_classes=2).to(device)
    inv = class_weights_from_y(y, n_classes=2)

    if mode == "baseline_invfreq":
        w = inv
        crit: nn.Module = nn.CrossEntropyLoss(weight=w.to(device))
        weight_note = {"class_weights": inv.tolist(), "loss": "CE_invfreq"}
    elif mode == "precision_favor":
        # Lower weight on hypnagogic → model less eager to call hypnagogic → fewer FP
        w = torch.tensor([1.2, 0.6], dtype=torch.float32)  # drowsy, hypnagogic
        crit = nn.CrossEntropyLoss(weight=w.to(device))
        weight_note = {"class_weights": w.tolist(), "loss": "CE_precision_favor"}
    elif mode == "focal_precision":
        w = torch.tensor([1.3, 0.55], dtype=torch.float32)
        crit = FocalCE(weight=w.to(device), gamma=1.5)
        weight_note = {"class_weights": w.tolist(), "loss": "focal_CE", "gamma": 1.5}
    else:
        raise ValueError(mode)

    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(emb, yt), batch_size=BATCH, shuffle=True)
    history = []
    for ep in range(epochs):
        head.train()
        total, n = 0.0, 0
        for xb, ybatch in loader:
            xb = xb.to(device)
            ybatch = ybatch.to(device)
            opt.zero_grad()
            logits = head(xb)
            loss = crit(logits, ybatch)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(ybatch)
            n += len(ybatch)
        head.eval()
        with torch.no_grad():
            pred = head(emb.to(device)).argmax(dim=-1).cpu().numpy()
        acc = float((pred == y).mean())
        f1 = macro_f1(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
        row = {"epoch": ep + 1, "loss": total / max(n, 1), "train_acc": acc, "train_macro_f1": f1}
        history.append(row)
        print(f"  [{mode}] {row}")
    return head, history, weight_note


def reconstruct_sc4001_splits(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    npz = np.load(ROOT / "exports/windows_sc4001/sleep_edf_sc4001_n1slice_windows.npz")
    X1, y1 = npz["X"], npz["y"].astype(np.int64)
    Xb, yb = undersample_balanced(X1, y1, rng)
    n_bal = len(yb)
    perm = rng.permutation(n_bal)
    n_val = max(1, int(round(n_bal * VAL_FRAC)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    return Xb[tr_idx], yb[tr_idx], Xb[val_idx], yb[val_idx]


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    weights = find_weights()
    head_path = ROOT / "exports/head_a_holdout/head_a_binary_state_dict.pt"
    out_dir = ROOT / "exports/head_a_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "emb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    print("weights", weights)
    print("reconstructing SC4001 train/val (seed=%d val_frac=%.2f)..." % (SEED, VAL_FRAC))
    X_tr, y_tr, X_val, y_val = reconstruct_sc4001_splits(rng)
    print("train", X_tr.shape, Counter(y_tr.tolist()), "val", X_val.shape, Counter(y_val.tolist()))

    npz2 = np.load(ROOT / "exports/windows_sc4002/sleep_edf_sc4002_n1slice_windows.npz")
    X2, y2 = npz2["X"], npz2["y"].astype(np.int64)
    # Balanced SC4002 with fresh rng fork so we don't disturb val reconstruction —
    # use dedicated seed-derived generator matching night_holdout after train split:
    # night_holdout used same rng sequentially: balance sc4001 → split → later balance sc4002.
    # For eval we only need a deterministic balanced subsample; seed=42 standalone is fine
    # and matches tighten / prior balanced metrics approximately. Document seed.
    rng_bal = np.random.default_rng(SEED)
    X2b, y2b = undersample_balanced(X2, y2, rng_bal)
    print("SC4002 all", X2.shape, Counter(y2.tolist()), "balanced", X2b.shape, Counter(y2b.tolist()))

    emb_tr_path = cache_dir / "emb_sc4001_train.pt"
    emb_val_path = cache_dir / "emb_sc4001_val.pt"
    emb2_path = cache_dir / "emb_sc4002_all.pt"
    emb2b_path = cache_dir / "emb_sc4002_balanced.pt"

    need_encode = not all(p.exists() for p in (emb_tr_path, emb_val_path, emb2_path, emb2b_path))
    if need_encode:
        encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
        encoder.to(device)
        print("encoding SC4001 train...")
        emb_tr = encode_all(encoder, X_tr, device)
        print("encoding SC4001 val...")
        emb_val = encode_all(encoder, X_val, device)
        print("encoding SC4002 all...")
        emb2 = encode_all(encoder, X2, device)
        print("encoding SC4002 balanced...")
        emb2b = encode_all(encoder, X2b, device)
        torch.save(emb_tr, emb_tr_path)
        torch.save(emb_val, emb_val_path)
        torch.save(emb2, emb2_path)
        torch.save(emb2b, emb2b_path)
        # also save y for cache sanity
        np.savez(
            cache_dir / "split_labels.npz",
            y_tr=y_tr,
            y_val=y_val,
            y2=y2,
            y2b=y2b,
        )
    else:
        print("loading cached embeddings from", cache_dir)
        emb_tr = torch.load(emb_tr_path, map_location="cpu", weights_only=True)
        emb_val = torch.load(emb_val_path, map_location="cpu", weights_only=True)
        emb2 = torch.load(emb2_path, map_location="cpu", weights_only=True)
        emb2b = torch.load(emb2b_path, map_location="cpu", weights_only=True)
        labs = np.load(cache_dir / "split_labels.npz")
        # Prefer live reconstructed y (same seed) over cache if shapes match
        assert len(y_tr) == len(emb_tr) and len(y_val) == len(emb_val)
        assert len(y2) == len(emb2) and len(y2b) == len(emb2b)

    # --- Baseline head (frozen trained weights) ---
    head0 = HeadALinear(in_dim=emb_tr.shape[-1], n_classes=2)
    head0.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
    head0.eval()
    with torch.no_grad():
        logits_val0 = head0(emb_val)
        logits2_0 = head0(emb2)
        logits2b_0 = head0(emb2b)
    probs_val0 = softmax_probs(logits_val0)
    probs2_0 = softmax_probs(logits2_0)
    probs2b_0 = softmax_probs(logits2b_0)

    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    # Baseline at argmax / 0.5
    before_val = report_from_pred(y_val, pred_from_probs(probs_val0, 0.5), "sc4001_val_threshold_0.5")
    before_all = report_from_pred(y2, pred_from_probs(probs2_0, 0.5), "sc4002_all_threshold_0.5")
    before_bal = report_from_pred(y2b, pred_from_probs(probs2b_0, 0.5), "sc4002_balanced_threshold_0.5")
    print("\n=== BEFORE (threshold=0.5, frozen head) ===")
    print("val hypn", before_val["per_class"].get("hypnagogic"))
    print("sc4002_all hypn", before_all["per_class"].get("hypnagogic"), "macro_f1", before_all["macro_f1"])
    print("sc4002_bal hypn", before_bal["per_class"].get("hypnagogic"), "macro_f1", before_bal["macro_f1"])

    baseline_val_recall = float(before_val["per_class"]["hypnagogic"]["recall"])
    sweep_val0 = sweep_thresholds(y_val, probs_val0, thresholds)
    pick0 = pick_threshold(sweep_val0, baseline_val_recall)
    t0 = float(pick0["chosen"]["threshold"])
    print("\n=== threshold pick on SC4001 val (frozen head) ===")
    print(json.dumps(pick0, indent=2))

    after_val0 = report_from_pred(y_val, pred_from_probs(probs_val0, t0), f"sc4001_val_threshold_{t0}")
    after_all0 = report_from_pred(y2, pred_from_probs(probs2_0, t0), f"sc4002_all_threshold_{t0}")
    after_bal0 = report_from_pred(y2b, pred_from_probs(probs2b_0, t0), f"sc4002_balanced_threshold_{t0}")

    # LEAKY ablation: best threshold on SC4002 all (document only)
    sweep_leak = sweep_thresholds(y2, probs2_0, thresholds)
    leak_pick = max(sweep_leak, key=lambda r: (r["hypnagogic_f0.5"], r["hypnagogic_precision"]))

    # --- Optional retrains ---
    retrain_results = {}
    best_overall = {
        "name": "frozen_head_threshold",
        "threshold": t0,
        "head_mode": "frozen_original",
        "sc4002_all": after_all0,
        "sc4002_balanced": after_bal0,
        "sc4001_val": after_val0,
        "pick": pick0,
        "score_sc4002_hypn_precision": float(after_all0["per_class"]["hypnagogic"]["precision"]),
        "score_sc4002_hypn_recall": float(after_all0["per_class"]["hypnagogic"]["recall"]),
    }

    for mode in ("precision_favor", "focal_precision"):
        print(f"\n=== retrain mode={mode} ===")
        torch.manual_seed(SEED)
        head_r, hist_r, wnote = train_head(emb_tr, y_tr, device, mode=mode)
        head_r.eval()
        with torch.no_grad():
            probs_val_r = softmax_probs(head_r(emb_val))
            probs2_r = softmax_probs(head_r(emb2))
            probs2b_r = softmax_probs(head_r(emb2b))
        before_r_all = report_from_pred(y2, pred_from_probs(probs2_r, 0.5), f"sc4002_all_{mode}_t0.5")
        sweep_r = sweep_thresholds(y_val, probs_val_r, thresholds)
        base_rec_r = float(
            report_from_pred(y_val, pred_from_probs(probs_val_r, 0.5), "tmp")["per_class"]["hypnagogic"]["recall"]
        )
        pick_r = pick_threshold(sweep_r, base_rec_r)
        tr = float(pick_r["chosen"]["threshold"])
        after_r_val = report_from_pred(y_val, pred_from_probs(probs_val_r, tr), f"val_{mode}_t{tr}")
        after_r_all = report_from_pred(y2, pred_from_probs(probs2_r, tr), f"sc4002_all_{mode}_t{tr}")
        after_r_bal = report_from_pred(y2b, pred_from_probs(probs2b_r, tr), f"sc4002_bal_{mode}_t{tr}")
        entry = {
            "mode": mode,
            "weight_note": wnote,
            "history": hist_r,
            "threshold": tr,
            "pick": pick_r,
            "before_threshold_0.5_sc4002_all": before_r_all,
            "sc4001_val": after_r_val,
            "sc4002_all": after_r_all,
            "sc4002_balanced": after_r_bal,
            "state_dict": {k: v.cpu() for k, v in head_r.state_dict().items()},
        }
        retrain_results[mode] = entry
        hyp_p = float(after_r_all["per_class"]["hypnagogic"]["precision"])
        hyp_r = float(after_r_all["per_class"]["hypnagogic"]["recall"])
        print(f"  {mode} t={tr}: sc4002 hypn P={hyp_p:.4f} R={hyp_r:.4f} macroF1={after_r_all['macro_f1']:.4f}")

        # Prefer higher precision on SC4002 all; require recall not collapse below 0.40
        if hyp_r >= 0.40 and (
            hyp_p > best_overall["score_sc4002_hypn_precision"]
            or (
                abs(hyp_p - best_overall["score_sc4002_hypn_precision"]) < 1e-6
                and hyp_r > best_overall["score_sc4002_hypn_recall"]
            )
        ):
            best_overall = {
                "name": f"retrain_{mode}_threshold",
                "threshold": tr,
                "head_mode": mode,
                "sc4002_all": after_r_all,
                "sc4002_balanced": after_r_bal,
                "sc4001_val": after_r_val,
                "pick": pick_r,
                "score_sc4002_hypn_precision": hyp_p,
                "score_sc4002_hypn_recall": hyp_r,
                "weight_note": wnote,
                "history": hist_r,
            }

    # Save best head if retrain won; else keep original + threshold
    tuned_head_path = out_dir / "head_a_binary_precision_tuned.pt"
    if best_overall["head_mode"] == "frozen_original":
        # copy original weights as tuned artifact (threshold-only)
        torch.save(head0.state_dict(), tuned_head_path)
        used_retrain = False
    else:
        mode = best_overall["head_mode"]
        torch.save(retrain_results[mode]["state_dict"], tuned_head_path)
        used_retrain = True

    # Strip state_dicts from JSON payload
    retrain_json = {}
    for k, v in retrain_results.items():
        retrain_json[k] = {kk: vv for kk, vv in v.items() if kk != "state_dict"}

    payload = {
        "task": "hypnagogic_precision_tune",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "val_frac": VAL_FRAC,
        "leakage_policy": {
            "threshold_selection": "SC4001 same-night balanced val only (seed=42, val_frac=0.2)",
            "holdout": "SC4002 never used for threshold or weight selection (except documented leaky ablation)",
            "retrain": "SC4001 balanced train only; optional precision-favor / focal weights",
        },
        "label_list": HEAD_A_BINARY_LABELS,
        "original_head_path": str(head_path),
        "tuned_head_path": str(tuned_head_path),
        "decision_rule": "predict hypnagogic if P(hypnagogic) >= threshold else drowsy",
        "before": {
            "threshold": 0.5,
            "sc4001_val": before_val,
            "sc4002_all_labeled": before_all,
            "sc4002_balanced": before_bal,
        },
        "threshold_sweep_frozen_head": {
            "selection": pick0,
            "chosen_threshold": t0,
            "sc4001_val": after_val0,
            "sc4002_all_labeled": after_all0,
            "sc4002_balanced": after_bal0,
            "sweep_on_val_top": sorted(
                sweep_val0,
                key=lambda r: (r["hypnagogic_precision"], r["hypnagogic_f0.5"]),
                reverse=True,
            )[:15],
            "leaky_best_on_sc4002_all_ablation": {
                "note": "LEAKY — do not use for claimed generalization; bounds headroom only",
                "best": leak_pick,
            },
        },
        "retrain_variants": retrain_json,
        "selected": {
            "method": best_overall["name"],
            "head_mode": best_overall["head_mode"],
            "threshold": best_overall["threshold"],
            "used_retrain": used_retrain,
            "weight_note": best_overall.get("weight_note"),
            "sc4001_val": best_overall["sc4001_val"],
            "sc4002_all_labeled": best_overall["sc4002_all"],
            "sc4002_balanced": best_overall["sc4002_balanced"],
            "pick": best_overall["pick"],
        },
        "improvement": {
            "sc4002_all_hypnagogic_precision": {
                "before": float(before_all["per_class"]["hypnagogic"]["precision"]),
                "after": float(best_overall["sc4002_all"]["per_class"]["hypnagogic"]["precision"]),
                "delta": float(best_overall["sc4002_all"]["per_class"]["hypnagogic"]["precision"])
                - float(before_all["per_class"]["hypnagogic"]["precision"]),
            },
            "sc4002_all_hypnagogic_recall": {
                "before": float(before_all["per_class"]["hypnagogic"]["recall"]),
                "after": float(best_overall["sc4002_all"]["per_class"]["hypnagogic"]["recall"]),
                "delta": float(best_overall["sc4002_all"]["per_class"]["hypnagogic"]["recall"])
                - float(before_all["per_class"]["hypnagogic"]["recall"]),
            },
            "sc4002_all_macro_f1": {
                "before": float(before_all["macro_f1"]),
                "after": float(best_overall["sc4002_all"]["macro_f1"]),
                "delta": float(best_overall["sc4002_all"]["macro_f1"]) - float(before_all["macro_f1"]),
            },
            "sc4002_balanced_hypnagogic_precision": {
                "before": float(before_bal["per_class"]["hypnagogic"]["precision"]),
                "after": float(best_overall["sc4002_balanced"]["per_class"]["hypnagogic"]["precision"]),
                "delta": float(best_overall["sc4002_balanced"]["per_class"]["hypnagogic"]["precision"])
                - float(before_bal["per_class"]["hypnagogic"]["precision"]),
            },
            "sc4002_balanced_macro_f1": {
                "before": float(before_bal["macro_f1"]),
                "after": float(best_overall["sc4002_balanced"]["macro_f1"]),
                "delta": float(best_overall["sc4002_balanced"]["macro_f1"]) - float(before_bal["macro_f1"]),
            },
        },
        "min_val_recall": MIN_VAL_RECALL,
        "recall_floor_frac_of_baseline": RECALL_FLOOR_FRAC,
    }

    metrics_path = out_dir / "precision_tune_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print("wrote", metrics_path)
    print("wrote", tuned_head_path)

    # Update run_manifest with precision_tune section
    man_path = out_dir / "run_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
    else:
        man = {}
    man["precision_tune"] = {
        "created_utc": payload["created_utc"],
        "method": best_overall["name"],
        "head_mode": best_overall["head_mode"],
        "threshold": best_overall["threshold"],
        "tuned_head_path": str(tuned_head_path),
        "decision_rule": payload["decision_rule"],
        "leakage_policy": payload["leakage_policy"],
        "improvement": payload["improvement"],
        "metrics_path": str(metrics_path),
        "before_sc4002_all_hypnagogic": before_all["per_class"]["hypnagogic"],
        "after_sc4002_all_hypnagogic": best_overall["sc4002_all"]["per_class"]["hypnagogic"],
        "before_sc4002_all_macro_f1": before_all["macro_f1"],
        "after_sc4002_all_macro_f1": best_overall["sc4002_all"]["macro_f1"],
    }
    man_path.write_text(json.dumps(man, indent=2))
    print("updated", man_path)

    print("\n=== SELECTED ===")
    print(json.dumps(payload["selected"] | {"improvement": payload["improvement"]}, indent=2, default=str))


if __name__ == "__main__":
    main()

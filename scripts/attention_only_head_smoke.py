#!/usr/bin/env python3
"""Attention-only binary Head A smoke on OpenNeuro ds003969 (Muse-proximal).

Train: sub001 + sub002 (htr) — per-subject undersample_balanced then concat.
Holdout: sub025 (ctr) — full / balanced / stride×4 metrics.
Frozen CBraMod encoder (CPU OK) + HeadALinear(n_classes=2).
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder
from src.head_a import (
    ATTENTION_LABELS,
    HeadALinear,
    class_weights_from_y,
    undersample_balanced,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report

SEED = 42
EPOCHS = 8
BATCH = 32
LR = 1e-3
VAL_FRAC = 0.15

TRAIN_TAGS = ["sub001", "sub002"]  # htr
HOLDOUT_TAG = "sub025"  # ctr

OUT_DIR = ROOT / "exports" / "attention_only_head_smoke"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_weights() -> Path:
    cands = [
        ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth",
        Path("/tmp/kaggle_out3/models/CBraMod/pretrained_weights.pth"),
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("CBraMod weights not found")


def load_attention(tag: str) -> Dict[str, Any]:
    d = ROOT / f"exports/windows_ds003969/{tag}"
    npz = d / f"ds003969_{tag}_attention_windows.npz"
    man_path = d / f"ds003969_{tag}_attention_manifest.json"
    if not npz.exists():
        npz = next(d.glob("*_attention_windows.npz"))
    if not man_path.exists():
        man_path = next(d.glob("*_attention_manifest.json"))
    data = np.load(npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)  # 0=concentration, 1=mind_wandering
    assert set(np.unique(y)).issubset({0, 1}), f"unexpected labels in {tag}: {np.unique(y)}"
    starts = (
        data["starts"].astype(np.int64)
        if "starts" in data.files
        else np.arange(len(y), dtype=np.int64)
    )
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    counts = {ATTENTION_LABELS[i]: int((y == i).sum()) for i in range(2)}
    return {
        "tag": tag,
        "group": man.get("group"),
        "X": X,
        "y": y,
        "starts": starts,
        "counts": counts,
        "npz_path": npz,
        "npz_sha256": sha256(npz),
        "manifest": man,
    }


def balance_train(
    packs: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Per-subject undersample_balanced then concat."""
    Xs, ys = [], []
    detail: Dict[str, Any] = {}
    for p in packs:
        present = sorted(set(int(c) for c in np.unique(p["y"])))
        if len(present) < 2:
            detail[p["tag"]] = {
                "skipped": True,
                "reason": "single_class",
                "counts": p["counts"],
            }
            continue
        Xb, yb = undersample_balanced(p["X"], p["y"], rng)
        Xs.append(Xb)
        ys.append(yb)
        detail[p["tag"]] = {
            "skipped": False,
            "group": p.get("group"),
            "n_total": int(len(yb)),
            "per_class": {ATTENTION_LABELS[i]: int((yb == i).sum()) for i in present},
            "counts_full": p["counts"],
        }
    if not Xs:
        raise RuntimeError("no train subjects with both attention classes")
    return np.concatenate(Xs), np.concatenate(ys), detail


def stride4(X: np.ndarray, y: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(starts)
    idx = order[::4]
    return X[idx], y[idx]


def encode_all(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(X) == 0:
        return torch.zeros((0, 200), dtype=torch.float32)
    Xt = torch.from_numpy(X.astype(np.float32))
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            emb_list.append(encoder(Xt[i : i + BATCH].to(device)).cpu())
    return torch.cat(emb_list, dim=0)


def eval_split(
    head: nn.Module,
    emb: torch.Tensor,
    y: np.ndarray,
    name: str,
) -> Dict[str, Any]:
    head.eval()
    if len(y) == 0:
        return {
            "name": name,
            "n": 0,
            "acc": None,
            "macro_f1": None,
            "per_class": {},
            "confusion": [],
            "counts_true": {},
            "note": "empty split",
        }
    with torch.no_grad():
        pred = head(emb).argmax(dim=-1).numpy()
    acc = float((pred == y).mean())
    f1 = macro_f1(y.tolist(), pred.tolist(), ATTENTION_LABELS)
    report = per_class_report(y.tolist(), pred.tolist(), ATTENTION_LABELS)
    per_class = {k: v for k, v in report.items() if k != "macro_f1"}
    cm = confusion_matrix(y.tolist(), pred.tolist(), ATTENTION_LABELS)
    return {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1": f1,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": list(ATTENTION_LABELS),
        "counts_true": {ATTENTION_LABELS[i]: int((y == i).sum()) for i in range(2)},
    }


def train_head(
    emb: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[HeadALinear, List[Dict[str, Any]], Dict[str, Any]]:
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(VAL_FRAC * n)))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    emb_tr, y_tr = emb[tr_idx], y[tr_idx]
    emb_va, y_va = emb[va_idx], y[va_idx]

    head = HeadALinear(in_dim=emb.shape[-1], n_classes=2).to(device)
    w = class_weights_from_y(y_tr, n_classes=2).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(emb_tr, torch.from_numpy(y_tr)),
        batch_size=BATCH,
        shuffle=True,
    )

    history: List[Dict[str, Any]] = []
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    for ep in range(EPOCHS):
        head.train()
        total, n_seen = 0.0, 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = head(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n_seen += len(yb)
        head.eval()
        with torch.no_grad():
            pred_tr = head(emb_tr.to(device)).argmax(dim=-1).cpu().numpy()
            pred_va = head(emb_va.to(device)).argmax(dim=-1).cpu().numpy()
        row = {
            "epoch": ep + 1,
            "loss": total / max(n_seen, 1),
            "train_acc": float((pred_tr == y_tr).mean()),
            "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), ATTENTION_LABELS),
            "val_acc": float((pred_va == y_va).mean()),
            "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), ATTENTION_LABELS),
        }
        history.append(row)
        print(row, flush=True)
        if row["val_macro_f1"] >= best_val_f1:
            best_val_f1 = float(row["val_macro_f1"])
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
        print(f"restored best val epoch={best_epoch} val_macro_f1={best_val_f1:.4f}", flush=True)

    # recompute train metrics at best checkpoint
    head.eval()
    with torch.no_grad():
        pred_tr = head(emb_tr.to(device)).argmax(dim=-1).cpu().numpy()
        pred_va = head(emb_va.to(device)).argmax(dim=-1).cpu().numpy()
    splits = {
        "train_n": int(len(y_tr)),
        "val_n": int(len(y_va)),
        "best_epoch": best_epoch,
        "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), ATTENTION_LABELS),
        "val_acc": float((pred_va == y_va).mean()),
        "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), ATTENTION_LABELS),
        "train_acc": float((pred_tr == y_tr).mean()),
        "last_epoch_val_macro_f1": history[-1]["val_macro_f1"] if history else None,
    }
    return head, history, splits


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    train_packs = [load_attention(t) for t in TRAIN_TAGS]
    hold = load_attention(HOLDOUT_TAG)

    X_tr, y_tr, bal_detail = balance_train(train_packs, rng)
    print("train balanced", X_tr.shape, Counter(y_tr.tolist()), bal_detail, flush=True)
    print(
        "holdout",
        hold["tag"],
        hold.get("group"),
        hold["X"].shape,
        hold["counts"],
        flush=True,
    )

    weights = find_weights()
    encoder = FrozenCBraModEncoder(weights, source_sr=256.0, pool="mean")
    notes = encoder.adapter_notes()
    encoder.to(device)
    print("encoder", notes["native_input"], "<-", notes["fed_input"], flush=True)

    print("encoding train…", flush=True)
    emb_tr = encode_all(encoder, X_tr, device)

    X_h_full, y_h_full = hold["X"], hold["y"]
    X_h_bal, y_h_bal = undersample_balanced(hold["X"], hold["y"], rng)
    X_h_s4, y_h_s4 = stride4(hold["X"], hold["y"], hold["starts"])

    print("encoding holdout full / balanced / stride4…", flush=True)
    emb_h_full = encode_all(encoder, X_h_full, device)
    emb_h_bal = encode_all(encoder, X_h_bal, device)
    emb_h_s4 = encode_all(encoder, X_h_s4, device)

    head, history, splits = train_head(emb_tr, y_tr, device, rng)

    evals = {
        "train_balanced": eval_split(head, emb_tr, y_tr, "train_balanced"),
        "holdout_sub025_full": eval_split(head, emb_h_full, y_h_full, "holdout_sub025_full"),
        "holdout_sub025_balanced": eval_split(head, emb_h_bal, y_h_bal, "holdout_sub025_balanced"),
        "holdout_sub025_stride4": eval_split(head, emb_h_s4, y_h_s4, "holdout_sub025_stride4"),
    }
    for k, v in evals.items():
        print(k, {kk: v[kk] for kk in ("n", "acc", "macro_f1")}, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    head_path = OUT_DIR / "head_a_attention_binary_ds003969.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": list(ATTENTION_LABELS),
            "in_dim": int(emb_tr.shape[-1]),
            "n_classes": 2,
            "head": "HeadALinear",
            "dataset": "ds003969",
            "train_tags": TRAIN_TAGS,
            "holdout_tag": HOLDOUT_TAG,
            "task": "binary_concentration_vs_mind_wandering",
        },
        head_path,
    )

    headline = {
        "val_macro_f1": splits["val_macro_f1"],
        "val_acc": splits["val_acc"],
        "holdout_full_macro_f1": evals["holdout_sub025_full"]["macro_f1"],
        "holdout_balanced_macro_f1": evals["holdout_sub025_balanced"]["macro_f1"],
        "holdout_stride4_macro_f1": evals["holdout_sub025_stride4"]["macro_f1"],
        "holdout_full_acc": evals["holdout_sub025_full"]["acc"],
        "holdout_balanced_acc": evals["holdout_sub025_balanced"]["acc"],
        "holdout_stride4_acc": evals["holdout_sub025_stride4"]["acc"],
    }

    metrics = {
        "task": "attention_only_binary_ds003969",
        "label_list": list(ATTENTION_LABELS),
        "train_tags": TRAIN_TAGS,
        "holdout_tag": HOLDOUT_TAG,
        "train_balance": bal_detail,
        "holdout_counts_full": hold["counts"],
        "splits": splits,
        "history": history,
        "evals": evals,
        "headline": headline,
    }
    metrics_path = OUT_DIR / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    takeaway = (
        f"Attention-only binary on Muse-proximal ds003969: val F1={headline['val_macro_f1']:.3f}; "
        f"sub025 holdout full/bal/stride4 F1="
        f"{headline['holdout_full_macro_f1']:.3f}/"
        f"{headline['holdout_balanced_macro_f1']:.3f}/"
        f"{headline['holdout_stride4_macro_f1']:.3f}."
    )
    if headline["holdout_balanced_macro_f1"] is not None:
        if headline["holdout_balanced_macro_f1"] < 0.55:
            takeaway += (
                " Holdout near chance — subject transfer still weak; need more subjects "
                "or stronger labels before shipping attention head."
            )
        elif headline["holdout_balanced_macro_f1"] < 0.70:
            takeaway += (
                " Modest subject-holdout signal on Muse AF7/AF8; better than ds001787 joint "
                "collapse but not pack-ready yet."
            )
        else:
            takeaway += (
                " Useful Muse-proximal attention signal under subject holdout — candidate "
                "for a separate attention head family."
            )

    manifest = {
        "task": "attention_only_head_smoke",
        "dataset": "ds003969",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "label_list": list(ATTENTION_LABELS),
        "label_rule": "block-level: med*→concentration; think*→mind_wandering (protocol proxy)",
        "train_tags": TRAIN_TAGS,
        "train_groups": {p["tag"]: p.get("group") for p in train_packs},
        "holdout_tag": HOLDOUT_TAG,
        "holdout_group": hold.get("group"),
        "train_balance": bal_detail,
        "npz_sha256": {
            p["tag"]: p["npz_sha256"] for p in train_packs + [hold]
        },
        "encoder_name": "CBraMod",
        "encoder_weights_path": str(weights),
        "encoder_weights_sha256": notes["weights_sha256"],
        "encoder_adapter": notes,
        "head": "HeadALinear",
        "n_classes": 2,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "history": history,
        "headline": headline,
        "metrics_ref": str(metrics_path.relative_to(ROOT)),
        "head_path": str(head_path.relative_to(ROOT)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_attribution": {
            "ds003969": "OpenNeuro CC0-1.0 — Muse meditation / mind-wandering blocks",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
        "caveats": [
            "protocol-proxy labels (med vs think blocks), weaker than probe ratings",
            "subject holdout is one ctr subject (sub025); train is two htr subjects",
            "overlapping windows; stride×4 reported for temporal thinning",
            "montage: native AF7/AF8 + TP7/TP8→TP9/TP10 proxies vs CBraMod TUEG pretrain",
        ],
    }
    man_path = OUT_DIR / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))

    summary = {
        "step": "attention_only_head_smoke",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "train_tags": TRAIN_TAGS,
        "holdout_tag": HOLDOUT_TAG,
        "headline": headline,
        "takeaway": takeaway,
        "artifacts": {
            "script": "scripts/attention_only_head_smoke.py",
            "docs": "docs/attention_only_head_smoke.md",
            "metrics": str(metrics_path.relative_to(ROOT)),
            "manifest": str(man_path.relative_to(ROOT)),
            "head": str(head_path.relative_to(ROOT)),
        },
    }
    (OUT_DIR / "step_summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

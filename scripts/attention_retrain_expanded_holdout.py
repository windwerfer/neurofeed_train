#!/usr/bin/env python3
"""Attention-only binary Head A retrain on expanded ds001787 pool.

Train: expert tags with both classes (sub001/002/003/004 ses01).
Primary holdout: sub019_ses01 (novice, both classes).
Secondary holdout: sub013_ses01 (novice, both classes).
Probes: sub014/sub017 (single-class concentration) — accuracy only.
Frozen CBraMod encoder (CPU) + HeadALinear(n_classes=2).
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
EPOCHS = 10
BATCH = 32
LR = 1e-3
VAL_FRAC = 0.15

TRAIN_TAGS = [
    "sub001_ses01",
    "sub002_ses01",
    "sub003_ses01",
    "sub004_ses01",
]
HOLDOUT_PRIMARY = "sub019_ses01"
HOLDOUT_SECONDARY = "sub013_ses01"
PROBE_TAGS = ["sub014_ses01", "sub017_ses01"]

OUT_DIR = ROOT / "exports" / "attention_retrain_expanded_holdout"
STEP_NAME = "attention_retrain_expanded_holdout"


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
    d = ROOT / f"exports/windows_ds001787/{tag}"
    npz = d / f"ds001787_{tag}_attention_windows.npz"
    man_path = d / f"ds001787_{tag}_attention_manifest.json"
    if not npz.exists():
        npz = next(d.glob("*_attention_windows.npz"))
    if not man_path.exists():
        man_path = next(d.glob("*_attention_manifest.json"))
    data = np.load(npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
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
    present = sorted(set(int(c) for c in np.unique(y)))
    if len(present) < 2:
        return {
            "name": name,
            "n": int(len(y)),
            "acc": acc,
            "macro_f1": None,
            "per_class": {},
            "confusion": [],
            "counts_true": {ATTENTION_LABELS[i]: int((y == i).sum()) for i in range(2)},
            "note": "single_class_probe",
            "pred_counts": {ATTENTION_LABELS[i]: int((pred == i).sum()) for i in range(2)},
        }
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
    hold_p = load_attention(HOLDOUT_PRIMARY)
    hold_s = load_attention(HOLDOUT_SECONDARY)
    probes = [load_attention(t) for t in PROBE_TAGS]

    X_tr, y_tr, bal_detail = balance_train(train_packs, rng)
    print("train balanced", X_tr.shape, Counter(y_tr.tolist()), bal_detail, flush=True)
    print("holdout primary", hold_p["tag"], hold_p.get("group"), hold_p["counts"], flush=True)
    print("holdout secondary", hold_s["tag"], hold_s.get("group"), hold_s["counts"], flush=True)

    weights = find_weights()
    encoder = FrozenCBraModEncoder(weights, source_sr=256.0, pool="mean")
    notes = encoder.adapter_notes()
    encoder.to(device)
    print("encoder", notes["native_input"], "<-", notes["fed_input"], flush=True)

    print("encoding train…", flush=True)
    emb_tr = encode_all(encoder, X_tr, device)

    def pack_hold(hold: Dict[str, Any]) -> Dict[str, Any]:
        Xf, yf = hold["X"], hold["y"]
        present = sorted(set(int(c) for c in np.unique(yf)))
        if len(present) >= 2:
            Xb, yb = undersample_balanced(Xf, yf, rng)
        else:
            Xb, yb = Xf[:0], yf[:0]
        Xs4, ys4 = stride4(Xf, yf, hold["starts"])
        return {
            "full": (Xf, yf, encode_all(encoder, Xf, device)),
            "balanced": (Xb, yb, encode_all(encoder, Xb, device)),
            "stride4": (Xs4, ys4, encode_all(encoder, Xs4, device)),
        }

    print("encoding holdouts…", flush=True)
    hp = pack_hold(hold_p)
    hs = pack_hold(hold_s)

    head, history, splits = train_head(emb_tr, y_tr, device, rng)

    evals: Dict[str, Any] = {
        "train_balanced": eval_split(head, emb_tr, y_tr, "train_balanced"),
        f"holdout_{HOLDOUT_PRIMARY}_full": eval_split(
            head, hp["full"][2], hp["full"][1], f"holdout_{HOLDOUT_PRIMARY}_full"
        ),
        f"holdout_{HOLDOUT_PRIMARY}_balanced": eval_split(
            head, hp["balanced"][2], hp["balanced"][1], f"holdout_{HOLDOUT_PRIMARY}_balanced"
        ),
        f"holdout_{HOLDOUT_PRIMARY}_stride4": eval_split(
            head, hp["stride4"][2], hp["stride4"][1], f"holdout_{HOLDOUT_PRIMARY}_stride4"
        ),
        f"holdout_{HOLDOUT_SECONDARY}_full": eval_split(
            head, hs["full"][2], hs["full"][1], f"holdout_{HOLDOUT_SECONDARY}_full"
        ),
        f"holdout_{HOLDOUT_SECONDARY}_balanced": eval_split(
            head, hs["balanced"][2], hs["balanced"][1], f"holdout_{HOLDOUT_SECONDARY}_balanced"
        ),
        f"holdout_{HOLDOUT_SECONDARY}_stride4": eval_split(
            head, hs["stride4"][2], hs["stride4"][1], f"holdout_{HOLDOUT_SECONDARY}_stride4"
        ),
    }

    for pr in probes:
        emb_pr = encode_all(encoder, pr["X"], device)
        evals[f"probe_{pr['tag']}_full"] = eval_split(
            head, emb_pr, pr["y"], f"probe_{pr['tag']}_full"
        )

    for k, v in evals.items():
        print(k, {kk: v.get(kk) for kk in ("n", "acc", "macro_f1", "note")}, flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    head_path = OUT_DIR / "head_a_attention_binary_ds001787_expanded.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": list(ATTENTION_LABELS),
            "in_dim": int(emb_tr.shape[-1]),
            "n_classes": 2,
            "head": "HeadALinear",
            "dataset": "ds001787",
            "train_tags": TRAIN_TAGS,
            "holdout_primary": HOLDOUT_PRIMARY,
            "holdout_secondary": HOLDOUT_SECONDARY,
            "probe_tags": PROBE_TAGS,
            "task": "binary_concentration_vs_mind_wandering",
        },
        head_path,
    )

    def f1(key: str):
        return evals[key]["macro_f1"]

    headline = {
        "val_macro_f1": splits["val_macro_f1"],
        "val_acc": splits["val_acc"],
        "holdout_primary_full_macro_f1": f1(f"holdout_{HOLDOUT_PRIMARY}_full"),
        "holdout_primary_balanced_macro_f1": f1(f"holdout_{HOLDOUT_PRIMARY}_balanced"),
        "holdout_primary_stride4_macro_f1": f1(f"holdout_{HOLDOUT_PRIMARY}_stride4"),
        "holdout_secondary_full_macro_f1": f1(f"holdout_{HOLDOUT_SECONDARY}_full"),
        "holdout_secondary_balanced_macro_f1": f1(f"holdout_{HOLDOUT_SECONDARY}_balanced"),
        "holdout_secondary_stride4_macro_f1": f1(f"holdout_{HOLDOUT_SECONDARY}_stride4"),
        "probe_sub014_acc": evals["probe_sub014_ses01_full"]["acc"],
        "probe_sub017_acc": evals["probe_sub017_ses01_full"]["acc"],
    }

    prior_joint_sub013_bal = 0.3333333333333333
    prior_ds003969_holdout_bal = 0.3333333333333333

    bal_p = headline["holdout_primary_balanced_macro_f1"]
    bal_s = headline["holdout_secondary_balanced_macro_f1"]
    takeaway = (
        f"Expanded ds001787 attention retrain (4 expert train → novice holdouts): "
        f"val F1={headline['val_macro_f1']:.3f}; "
        f"sub019 bal/full/s4 F1={bal_p:.3f}/{headline['holdout_primary_full_macro_f1']:.3f}/"
        f"{headline['holdout_primary_stride4_macro_f1']:.3f}; "
        f"sub013 bal/full/s4 F1={bal_s:.3f}/{headline['holdout_secondary_full_macro_f1']:.3f}/"
        f"{headline['holdout_secondary_stride4_macro_f1']:.3f}."
    )
    ship = False
    if bal_p is not None and bal_s is not None:
        if min(bal_p, bal_s) >= 0.70:
            takeaway += " Both novice holdouts ≥0.70 — candidate attention head for pack (still Muse-proximal, not personal cal)."
            ship = True
        elif min(bal_p, bal_s) >= 0.55:
            takeaway += (
                f" Above prior chance (~{prior_joint_sub013_bal:.2f} joint / "
                f"{prior_ds003969_holdout_bal:.2f} ds003969) but not ship-ready; "
                "consider more subjects or ds003969 mix before packaging."
            )
        else:
            takeaway += (
                " Holdouts still near chance — subject transfer remains weak; "
                "do not ship attention head; next: more_ds003969_subjects or personal Muse cal."
            )

    metrics = {
        "task": "attention_retrain_expanded_holdout_ds001787",
        "label_list": list(ATTENTION_LABELS),
        "train_tags": TRAIN_TAGS,
        "holdout_primary": HOLDOUT_PRIMARY,
        "holdout_secondary": HOLDOUT_SECONDARY,
        "probe_tags": PROBE_TAGS,
        "train_balance": bal_detail,
        "holdout_primary_counts": hold_p["counts"],
        "holdout_secondary_counts": hold_s["counts"],
        "splits": splits,
        "history": history,
        "evals": evals,
        "headline": headline,
        "ship_candidate": ship,
        "priors": {
            "joint_4way_sub013_balanced_macro_f1": prior_joint_sub013_bal,
            "ds003969_sub025_balanced_macro_f1": prior_ds003969_holdout_bal,
        },
    }
    metrics_path = OUT_DIR / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    encoder_sha = sha256(weights)
    run_manifest = {
        "step": STEP_NAME,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "ds001787",
        "license_spdx": "CC0-1.0",
        "encoder": {
            "family": "CBraMod",
            "weights_path": str(weights.relative_to(ROOT)),
            "sha256": encoder_sha,
            "frozen": True,
            "source_sr": 256.0,
            "adapter": notes,
        },
        "head_path": str(head_path.relative_to(ROOT)),
        "train_tags": TRAIN_TAGS,
        "holdout_primary": HOLDOUT_PRIMARY,
        "holdout_secondary": HOLDOUT_SECONDARY,
        "probe_tags": PROBE_TAGS,
        "headline": headline,
        "ship_candidate": ship,
        "takeaway": takeaway,
        "hard_reject": "Do not map events value 2/4 to classes",
        "label_rule": "Q1>Q2→concentration; Q1<Q2→mind_wandering; ties dropped",
        "channel_proxy": "AF7/AF8/TP9/TP10 with TP9/TP10←P9/P10 from BioSemi64",
        "npz_sha256": {
            p["tag"]: p["npz_sha256"] for p in train_packs + [hold_p, hold_s] + probes
        },
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))

    docs = f"""# {STEP_NAME}

**Step:** `{STEP_NAME}`  
**Completed (UTC):** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}  
**Dataset:** OpenNeuro `ds001787` — SPDX **CC0-1.0**

## Split

| Role | Tags |
|------|------|
| Train (experts, both classes, per-subject undersample) | {', '.join(TRAIN_TAGS)} |
| Primary holdout (novice) | {HOLDOUT_PRIMARY} |
| Secondary holdout (novice) | {HOLDOUT_SECONDARY} |
| Single-class probes | {', '.join(PROBE_TAGS)} |

## Headline metrics

| Metric | Value |
|--------|------:|
| val macro-F1 | {headline['val_macro_f1']:.4f} |
| sub019 balanced macro-F1 | {bal_p:.4f} |
| sub019 full macro-F1 | {headline['holdout_primary_full_macro_f1']:.4f} |
| sub019 stride×4 macro-F1 | {headline['holdout_primary_stride4_macro_f1']:.4f} |
| sub013 balanced macro-F1 | {bal_s:.4f} |
| sub013 full macro-F1 | {headline['holdout_secondary_full_macro_f1']:.4f} |
| sub013 stride×4 macro-F1 | {headline['holdout_secondary_stride4_macro_f1']:.4f} |
| probe sub014 acc | {headline['probe_sub014_acc']:.4f} |
| probe sub017 acc | {headline['probe_sub017_acc']:.4f} |
| ship_candidate | {ship} |

## Takeaway

{takeaway}

## Artifacts

- Script: `scripts/attention_retrain_expanded_holdout.py`
- Head: `{head_path.relative_to(ROOT)}`
- Metrics: `exports/{STEP_NAME}/metrics_summary.json`
- Manifest: `exports/{STEP_NAME}/run_manifest.json`

## Out of scope

- No Kaggle reversion this step (windows already versioned)
- No joint 4-way retrain; no Head B/C training
- Packaging only if ship_candidate (see next invent: repack if true)
"""
    (ROOT / "docs" / f"{STEP_NAME}.md").write_text(docs)

    step_summary = {
        "step": STEP_NAME,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "train_tags": TRAIN_TAGS,
        "holdout_primary": HOLDOUT_PRIMARY,
        "holdout_secondary": HOLDOUT_SECONDARY,
        "probe_tags": PROBE_TAGS,
        "metrics": headline,
        "ship_candidate": ship,
        "takeaway": takeaway,
        "artifacts": {
            "script": f"scripts/{STEP_NAME}.py",
            "docs": f"docs/{STEP_NAME}.md",
            "metrics": f"exports/{STEP_NAME}/metrics_summary.json",
            "manifest": f"exports/{STEP_NAME}/run_manifest.json",
            "head": str(head_path.relative_to(ROOT)),
        },
    }
    (OUT_DIR / "step_summary.json").write_text(json.dumps(step_summary, indent=2))
    (ROOT / "exports" / f"{STEP_NAME}_summary.json").write_text(
        json.dumps(step_summary, indent=2)
    )
    print("TAKEAWAY:", takeaway, flush=True)
    print("ship_candidate:", ship, flush=True)


if __name__ == "__main__":
    main()

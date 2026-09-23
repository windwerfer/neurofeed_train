#!/usr/bin/env python3
"""Wave 2 step: expand multi-night train pool with SC4021 + SC4041.

Protocol:
  Train nights: SC4001 + SC4011 + SC4021 + SC4041 (four cassette subjects; equal per-night balance)
  Primary holdout: SC4002 (same holdout as prior multi-night / wave-1 for comparison)
  Secondary holdout: SC4031 (unseen subject)

Compares against prior 2-night multi-night holdout metrics.
Reports overlapping + non-overlap (stride×4 / greedy) metrics.
"""
from __future__ import annotations

import json
import hashlib
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
    HEAD_A_BINARY_LABELS,
    HeadALinear,
    class_weights_from_y,
    undersample_balanced,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report

SEED = 42
WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0
EPOCHS = 8
BATCH = 32
LR = 1e-3
VAL_FRAC = 0.2
TRAIN_NIGHTS = ["sc4001", "sc4011", "sc4021", "sc4041"]
PRIMARY_HOLDOUT = "sc4002"
SECONDARY_HOLDOUT = "sc4031"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_weights() -> Path:
    cands = [
        Path("/tmp/kaggle_out3/models/CBraMod/pretrained_weights.pth"),
        ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("CBraMod weights not found")


def load_night(tag: str) -> Dict[str, Any]:
    d = ROOT / f"exports/windows_{tag}"
    npz = d / f"sleep_edf_{tag}_n1slice_windows.npz"
    man = d / f"sleep_edf_{tag}_n1slice_manifest.json"
    data = np.load(npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    starts = data["starts"].astype(np.int64) if "starts" in data.files else np.arange(len(y), dtype=np.int64)
    manifest = json.loads(man.read_text())
    counts = {
        HEAD_A_BINARY_LABELS[i]: int((y == i).sum()) for i in range(len(HEAD_A_BINARY_LABELS))
    }
    return {
        "tag": tag,
        "X": X,
        "y": y,
        "starts": starts,
        "manifest": manifest,
        "counts": counts,
        "npz_path": npz,
        "man_path": man,
        "npz_sha256": sha256(npz),
    }


def encode_all(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(X) == 0:
        return torch.zeros((0, 200), dtype=torch.float32)
    Xt = torch.from_numpy(X.astype(np.float32))
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            emb_list.append(encoder(Xt[i : i + BATCH].to(device)).cpu())
    return torch.cat(emb_list, dim=0)


def eval_pred(y: np.ndarray, pred: np.ndarray, name: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if len(y) == 0:
        out: Dict[str, Any] = {
            "name": name,
            "n": 0,
            "acc": None,
            "macro_f1": None,
            "per_class": {},
            "confusion": [],
            "counts_true": {},
            "note": "empty",
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
    print(f"\n=== {name} === n={out['n']} acc={acc:.4f} macro_f1={f1:.4f}")
    print("per_class:", json.dumps(per_class, indent=2))
    print("confusion:", cm.tolist())
    return out


def nonoverlap_indices(starts: np.ndarray, window_samples: int) -> np.ndarray:
    order = np.argsort(starts)
    kept: List[int] = []
    last_end = -10**18
    for i in order:
        s = int(starts[i])
        if s >= last_end:
            kept.append(int(i))
            last_end = s + window_samples
    return np.asarray(kept, dtype=np.int64)


def undersample_balanced_idx(y: np.ndarray, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y_sub = y[idx]
    classes, counts = np.unique(y_sub, return_counts=True)
    if len(classes) < 2:
        return idx
    m = int(counts.min())
    picks = []
    for c in classes:
        cand = idx[y_sub == c]
        picks.append(rng.choice(cand, size=m, replace=False))
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


def train_head(emb: torch.Tensor, y: np.ndarray, device: torch.device) -> Tuple[HeadALinear, List[dict]]:
    yt = torch.from_numpy(y)
    head = HeadALinear(in_dim=emb.shape[-1], n_classes=2).to(device)
    w = class_weights_from_y(y, n_classes=2)
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(emb, yt), batch_size=BATCH, shuffle=True)
    history = []
    for ep in range(EPOCHS):
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
        print(row)
    return head, history


def equal_night_balance(
    nights: List[Dict[str, Any]], rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Per-night undersample, then equalize nights to the smallest night's per-class count."""
    per_night = []
    for n in nights:
        Xb, yb = undersample_balanced(n["X"], n["y"], rng)
        per_class = int(len(yb) // 2)
        per_night.append({"tag": n["tag"], "X": Xb, "y": yb, "per_class": per_class})
        print(f"  balanced {n['tag']}: {Xb.shape} per_class={per_class} counts={dict(Counter(yb.tolist()))}")
    target = min(p["per_class"] for p in per_night)
    Xs, ys, meta = [], [], {}
    for p in per_night:
        idxs = []
        for c in range(2):
            cand = np.where(p["y"] == c)[0]
            pick = rng.choice(cand, size=target, replace=False)
            idxs.append(pick)
        idxs = np.concatenate(idxs)
        rng.shuffle(idxs)
        Xs.append(p["X"][idxs])
        ys.append(p["y"][idxs])
        meta[p["tag"]] = {"per_class_used": target, "n_total": int(len(idxs))}
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(y))
    return X[perm], y[perm], meta


def evaluate_night(
    head: nn.Module,
    encoder: FrozenCBraModEncoder,
    night: Dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
    prefix: str,
) -> Dict[str, Any]:
    X, y, starts = night["X"], night["y"], night["starts"]
    print(f"encoding {prefix} all ({len(y)})...")
    emb = encode_all(encoder, X, device)
    head.eval()
    with torch.no_grad():
        pred_all = head(emb.to(device)).argmax(dim=-1).cpu().numpy()

    out: Dict[str, Any] = {
        "tag": night["tag"],
        "raw_counts": night["counts"],
        "all_labeled_overlap": eval_pred(y, pred_all, f"{prefix}_all_labeled_overlap"),
    }

    # balanced overlap
    if min(night["counts"].values()) > 0:
        Xb, yb = undersample_balanced(X, y, rng)
        emb_b = encode_all(encoder, Xb, device)
        with torch.no_grad():
            pred_b = head(emb_b.to(device)).argmax(dim=-1).cpu().numpy()
        out["balanced_overlap"] = eval_pred(yb, pred_b, f"{prefix}_balanced_overlap")
    else:
        out["balanced_overlap"] = eval_pred(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), f"{prefix}_balanced_overlap")

    win_samp = int(round(WINDOW_SEC * TARGET_SR))
    stride = int(round(WINDOW_SEC / HOP_SEC))  # 4

    idx_s = np.arange(0, len(y), stride, dtype=np.int64)
    out["all_labeled_stride4"] = eval_pred(y[idx_s], pred_all[idx_s], f"{prefix}_all_labeled_stride4", {"stride": stride, "offset": 0})

    idx_g = nonoverlap_indices(starts, win_samp)
    out["all_labeled_greedy"] = eval_pred(y[idx_g], pred_all[idx_g], f"{prefix}_all_labeled_greedy")

    if min(night["counts"].values()) > 0:
        idx_gb = undersample_balanced_idx(y, idx_g, rng)
        out["balanced_greedy"] = eval_pred(y[idx_gb], pred_all[idx_gb], f"{prefix}_balanced_greedy")
        idx_sb = undersample_balanced_idx(y, idx_s, rng)
        out["balanced_stride4"] = eval_pred(y[idx_sb], pred_all[idx_sb], f"{prefix}_balanced_stride4")
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    weights = find_weights()
    print("weights", weights)

    nights = {t: load_night(t) for t in TRAIN_NIGHTS + [PRIMARY_HOLDOUT, SECONDARY_HOLDOUT]}
    for t, n in nights.items():
        print(f"loaded {t}: X={n['X'].shape} counts={n['counts']}")

    train_list = [nights[t] for t in TRAIN_NIGHTS]
    print("equal-night balance...")
    X_all, y_all, night_meta = equal_night_balance(train_list, rng)
    print("combined train pool", X_all.shape, Counter(y_all.tolist()), night_meta)

    n_bal = len(y_all)
    perm = rng.permutation(n_bal)
    n_val = max(1, int(round(n_bal * VAL_FRAC)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    print(f"split train {X_tr.shape} val {X_val.shape}")

    device = torch.device("cpu")
    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
    notes = encoder.adapter_notes()
    encoder.to(device)

    print("encoding train...")
    emb_tr = encode_all(encoder, X_tr, device)
    print("encoding val...")
    emb_val = encode_all(encoder, X_val, device)

    print("training head...")
    head, history = train_head(emb_tr, y_tr, device)

    head.eval()
    with torch.no_grad():
        pred_tr = head(emb_tr.to(device)).argmax(dim=-1).cpu().numpy()
        pred_val = head(emb_val.to(device)).argmax(dim=-1).cpu().numpy()

    metrics: Dict[str, Any] = {
        "train_balanced": eval_pred(y_tr, pred_tr, "train_balanced_multi_night_pool"),
        "val_same_pool": eval_pred(y_val, pred_val, "val_same_pool"),
        "holdout_primary_sc4002": evaluate_night(
            head, encoder, nights[PRIMARY_HOLDOUT], device, rng, "sc4002"
        ),
        "holdout_secondary_sc4031": evaluate_night(
            head, encoder, nights[SECONDARY_HOLDOUT], device, rng, "sc4031"
        ),
    }

    run_dir = ROOT / "exports" / "head_a_multi_night_pool"
    run_dir.mkdir(parents=True, exist_ok=True)
    head_path = run_dir / "head_a_binary_multi_night_pool.pt"
    torch.save(head.state_dict(), head_path)

    # Compare to prior single-night and 2-night multi-night if present
    prior = {}
    prior_path = ROOT / "exports/head_a_holdout/metrics_summary.json"
    if prior_path.exists():
        prior_raw = json.loads(prior_path.read_text())
        m = prior_raw.get("metrics", {})
        prior = {
            "sc4002_all_labeled_macro_f1": (m.get("sc4002_all_labeled") or {}).get("macro_f1"),
            "sc4002_balanced_macro_f1": (m.get("sc4002_balanced") or {}).get("macro_f1"),
            "source": str(prior_path),
        }
    tighten_path = ROOT / "exports/head_a_holdout/tighten_eval_metrics.json"
    if tighten_path.exists():
        t = json.loads(tighten_path.read_text())
        for key in ("stride4_offset0", "nonoverlap_stride4", "all_labeled_stride4", "metrics"):
            if key in t:
                prior[f"tighten_{key}"] = t[key]
                break
        prior["tighten_source"] = str(tighten_path)

    prior_2night = {}
    prior_2_path = ROOT / "exports/head_a_multi_night/metrics_summary.json"
    if prior_2_path.exists():
        p2 = json.loads(prior_2_path.read_text())
        prior_2night = {
            "train_nights": p2.get("train_nights"),
            "sc4002_stride4_macro_f1": (p2.get("metrics", {}).get("sc4002", {}).get("all_labeled_stride4") or {}).get("macro_f1"),
            "sc4002_balanced_greedy_macro_f1": (p2.get("metrics", {}).get("sc4002", {}).get("balanced_greedy") or {}).get("macro_f1"),
            "sc4031_stride4_macro_f1": (p2.get("metrics", {}).get("sc4031", {}).get("all_labeled_stride4") or {}).get("macro_f1"),
            "sc4031_balanced_greedy_macro_f1": (p2.get("metrics", {}).get("sc4031", {}).get("balanced_greedy") or {}).get("macro_f1"),
            "source": str(prior_2_path),
        }
        # also from overnight_state / step_summary if present
        ss = ROOT / "exports/head_a_multi_night/step_summary.json"
        if ss.exists():
            s2 = json.loads(ss.read_text())
            prior_2night.update({
                "sc4002_stride4_macro_f1": s2.get("sc4002_stride4_macro_f1", prior_2night.get("sc4002_stride4_macro_f1")),
                "sc4002_balanced_greedy_macro_f1": s2.get("sc4002_balanced_greedy_macro_f1", prior_2night.get("sc4002_balanced_greedy_macro_f1")),
                "sc4031_stride4_macro_f1": s2.get("sc4031_stride4_macro_f1", prior_2night.get("sc4031_stride4_macro_f1")),
                "sc4031_balanced_greedy_macro_f1": s2.get("sc4031_balanced_greedy_macro_f1", prior_2night.get("sc4031_balanced_greedy_macro_f1")),
            })

    primary = metrics["holdout_primary_sc4002"]
    secondary = metrics["holdout_secondary_sc4031"]
    comparison = {
        "prior_single_night_train_sc4001_holdout_sc4002": prior,
        "prior_2night_multi": prior_2night,
        "multi_night_train": TRAIN_NIGHTS,
        "sc4002_all_overlap_macro_f1": primary["all_labeled_overlap"]["macro_f1"],
        "sc4002_all_stride4_macro_f1": primary["all_labeled_stride4"]["macro_f1"],
        "sc4002_balanced_greedy_macro_f1": (primary.get("balanced_greedy") or {}).get("macro_f1"),
        "sc4031_all_stride4_macro_f1": secondary["all_labeled_stride4"]["macro_f1"],
        "sc4031_balanced_greedy_macro_f1": (secondary.get("balanced_greedy") or {}).get("macro_f1"),
    }

    run_manifest = {
        "task": "binary_drowsy_vs_hypnagogic_multi_night_pool_sc4021_sc4041",
        "train_nights": [t.upper() for t in TRAIN_NIGHTS],
        "primary_holdout": PRIMARY_HOLDOUT.upper(),
        "secondary_holdout": SECONDARY_HOLDOUT.upper(),
        "equal_night_balance": night_meta,
        "protocol_note": (
            "Per-night undersample then equalize to min night per-class across four "
            "subjects (SC4001/4011/4021/4041); holdouts SC4002 + SC4031 unchanged."
        ),
        "slice_recipe": {
            "around_stage": "stage 1",
            "pre_sec": 1200,
            "post_sec": 2400,
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "majority_frac": 0.7,
        },
        "val_frac": VAL_FRAC,
        "random_seed": SEED,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "encoder_name": "CBraMod",
        "encoder_weights_path": str(weights),
        "encoder_weights_sha256": notes["weights_sha256"],
        "encoder_adapter": notes,
        "head": "HeadALinear",
        "label_list": HEAD_A_BINARY_LABELS,
        "history": history,
        "metrics": metrics,
        "comparison_to_single_night": comparison,
        "night_npz_sha256": {t: nights[t]["npz_sha256"] for t in nights},
        "head_path": str(head_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_attribution": {
            "Sleep-EDF": "PhysioNet ODC-By — attribute PhysioNet / Sleep-EDF Expanded",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
        "caveats": [
            "Montage mismatch vs TUEG/CBraMod pretrain (Fpz-Cz/Pz-Oz proxy → Muse AF7/AF8/TP9/TP10).",
            "Overlapping hop-0.5s windows correlate; prefer stride4/greedy metrics.",
            "Labels are W→drowsy and N1→hypnagogic proxies only.",
        ],
    }

    man_path = run_dir / "run_manifest.json"
    man_path.write_text(json.dumps(run_manifest, indent=2))
    metrics_path = run_dir / "metrics_summary.json"
    metrics_path.write_text(
        json.dumps(
            {
                "train_nights": [t.upper() for t in TRAIN_NIGHTS],
                "primary_holdout": PRIMARY_HOLDOUT.upper(),
                "secondary_holdout": SECONDARY_HOLDOUT.upper(),
                "equal_night_balance": night_meta,
                "history": history,
                "metrics": {
                    "train_balanced": {
                        "n": metrics["train_balanced"]["n"],
                        "acc": metrics["train_balanced"]["acc"],
                        "macro_f1": metrics["train_balanced"]["macro_f1"],
                    },
                    "val_same_pool": {
                        "n": metrics["val_same_pool"]["n"],
                        "acc": metrics["val_same_pool"]["acc"],
                        "macro_f1": metrics["val_same_pool"]["macro_f1"],
                    },
                    "sc4002": {
                        k: {
                            "n": v.get("n"),
                            "acc": v.get("acc"),
                            "macro_f1": v.get("macro_f1"),
                            "counts_true": v.get("counts_true"),
                            "per_class": v.get("per_class"),
                        }
                        for k, v in primary.items()
                        if isinstance(v, dict) and "n" in v
                    },
                    "sc4031": {
                        k: {
                            "n": v.get("n"),
                            "acc": v.get("acc"),
                            "macro_f1": v.get("macro_f1"),
                            "counts_true": v.get("counts_true"),
                            "per_class": v.get("per_class"),
                        }
                        for k, v in secondary.items()
                        if isinstance(v, dict) and "n" in v
                    },
                },
                "comparison_to_single_night": comparison,
                "created_utc": run_manifest["created_utc"],
            },
            indent=2,
        )
    )

    docs = ROOT / "docs" / "multi_night_pool_sc4021_sc4041.md"
    sc4002_s4 = primary["all_labeled_stride4"]
    sc4002_bg = primary.get("balanced_greedy") or {}
    sc4031_s4 = secondary["all_labeled_stride4"]
    sc4031_bg = secondary.get("balanced_greedy") or {}
    docs.write_text(
        f"""# Multi-night pool (SC4021+SC4041) — Head A binary (drowsy vs hypnagogic)

Generated: {run_manifest['created_utc']} (UTC). User TZ Asia/Bangkok (UTC+7).

## Protocol

- **Train:** SC4001 + SC4011 + SC4021 + SC4041 (four cassette subjects). Per-night undersample, then equalize to the smallest night's per-class count so abundant-N1 nights do not dominate.
- **Primary holdout:** SC4002 (same holdout night as wave-1 single-night train→holdout, for comparison).
- **Secondary holdout:** SC4031 (unseen subject).
- Encoder: frozen CBraMod. Head: `HeadALinear`. Recipe: first N1 ±20/40 min; window 2 s hop 0.5 s; majority 0.7; W→drowsy, N1→hypnagogic.
- Prefer **stride×4 / greedy non-overlap** scores over hop-0.5 overlap.

Script: `scripts/multi_night_pool_sc4021_sc4041.py`. Artifacts under `exports/head_a_multi_night_pool/`.

## Equal-night train pool

```json
{json.dumps(night_meta, indent=2)}
```

Train pool after equalize: n={n_bal} (val_frac={VAL_FRAC}). Final train n={len(y_tr)}, val n={len(y_val)}.

## Headline holdout metrics

| Split | n | accuracy | macro-F1 |
|-------|---|----------|----------|
| Train balanced (multi-night) | {metrics['train_balanced']['n']} | {metrics['train_balanced']['acc']:.4f} | {metrics['train_balanced']['macro_f1']:.4f} |
| Val same pool | {metrics['val_same_pool']['n']} | {metrics['val_same_pool']['acc']:.4f} | {metrics['val_same_pool']['macro_f1']:.4f} |
| SC4002 all stride×4 | {sc4002_s4['n']} | {sc4002_s4['acc']:.4f} | {sc4002_s4['macro_f1']:.4f} |
| SC4002 balanced greedy | {sc4002_bg.get('n')} | {sc4002_bg.get('acc')} | {sc4002_bg.get('macro_f1')} |
| SC4031 all stride×4 | {sc4031_s4['n']} | {sc4031_s4['acc']:.4f} | {sc4031_s4['macro_f1']:.4f} |
| SC4031 balanced greedy | {sc4031_bg.get('n')} | {sc4031_bg.get('acc')} | {sc4031_bg.get('macro_f1')} |

## vs prior single-night and 2-night multi

Prior single-night overlap all-labeled macro-F1 ≈ {prior.get('sc4002_all_labeled_macro_f1')}; balanced ≈ {prior.get('sc4002_balanced_macro_f1')}.
Prior 2-night (SC4001+SC4011) SC4002 stride×4 ≈ {prior_2night.get('sc4002_stride4_macro_f1')}; SC4031 stride×4 ≈ {prior_2night.get('sc4031_stride4_macro_f1')}.
4-night pool SC4002 all-overlap macro-F1 = {primary['all_labeled_overlap']['macro_f1']:.4f}; stride×4 = {sc4002_s4['macro_f1']:.4f}.

## Per-class (SC4002 stride×4)

```json
{json.dumps(sc4002_s4.get('per_class'), indent=2)}
```

Confusion: `{sc4002_s4.get('confusion')}`

## Per-class (SC4031 stride×4)

```json
{json.dumps(sc4031_s4.get('per_class'), indent=2)}
```

Confusion: `{sc4031_s4.get('confusion')}`

## Caveats

- Montage mismatch vs CBraMod TUEG pretrain; Muse-proxy channels only.
- Overlap hop scores inflate effective n; trust stride/greedy.
- Still binary drowsy/hypnagogic proxies — concentration / mind_wandering need attention datasets next.

## Artifacts

- Head: `exports/head_a_multi_night_pool/head_a_binary_multi_night_pool.pt`
- Metrics: `exports/head_a_multi_night_pool/metrics_summary.json`
- Full run: `exports/head_a_multi_night_pool/run_manifest.json`
"""
    )

    summary = {
        "train_nights": [t.upper() for t in TRAIN_NIGHTS],
        "primary_holdout": PRIMARY_HOLDOUT.upper(),
        "secondary_holdout": SECONDARY_HOLDOUT.upper(),
        "equal_night_balance": night_meta,
        "sc4002_stride4_macro_f1": sc4002_s4["macro_f1"],
        "sc4002_balanced_greedy_macro_f1": sc4002_bg.get("macro_f1"),
        "sc4031_stride4_macro_f1": sc4031_s4["macro_f1"],
        "sc4031_balanced_greedy_macro_f1": sc4031_bg.get("macro_f1"),
        "train_macro_f1": metrics["train_balanced"]["macro_f1"],
        "val_macro_f1": metrics["val_same_pool"]["macro_f1"],
        "comparison": comparison,
        "artifacts": {
            "script": "scripts/multi_night_pool_sc4021_sc4041.py",
            "docs": "docs/multi_night_pool_sc4021_sc4041.md",
            "metrics": "exports/head_a_multi_night_pool/metrics_summary.json",
            "run_manifest": "exports/head_a_multi_night_pool/run_manifest.json",
            "head": "exports/head_a_multi_night_pool/head_a_binary_multi_night_pool.pt",
        },
        "created_utc": run_manifest["created_utc"],
    }
    (run_dir / "step_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== STEP SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("wrote", man_path)
    print("wrote", metrics_path)
    print("wrote", docs)
    print("wrote", head_path)


if __name__ == "__main__":
    main()

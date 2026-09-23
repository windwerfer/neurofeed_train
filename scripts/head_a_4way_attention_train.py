#!/usr/bin/env python3
"""Wave 3: joint 4-way Head A — Sleep-EDF vigilance + ds001787 attention.

Train with alternating present-masks (vigilance batches mask attention logits;
attention batches mask vigilance). Subject/night holdouts kept separate.
"""
from __future__ import annotations

import json
import hashlib
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder
from src.head_a import (
    HEAD_A_4WAY_LABELS,
    HeadALinear,
    binary_ids_to_4way,
    class_weights_from_y,
    group_ids,
    masked_cross_entropy,
    present_class_ids,
    predict_present,
    undersample_balanced,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report

SEED = 42
EPOCHS = 8
BATCH = 32
LR = 1e-3
VAL_FRAC = 0.15
PER_CLASS_PER_NIGHT = 298  # match multi_night_pool equal-night balance

VIG_TRAIN_NIGHTS = ["sc4001", "sc4011", "sc4021", "sc4041"]
VIG_HOLDOUT_PRIMARY = "sc4002"
VIG_HOLDOUT_SECONDARY = "sc4031"
ATT_TRAIN_TAGS = ["sub001_ses01", "sub002_ses01"]  # expert, both classes
ATT_HOLDOUT_TAG = "sub013_ses01"  # novice, both classes
ATT_PROBE_TAG = "sub017_ses01"  # all-concentration pathology — probe only


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


def load_night_4way(tag: str) -> Dict[str, Any]:
    d = ROOT / f"exports/windows_{tag}"
    npz = d / f"sleep_edf_{tag}_n1slice_windows.npz"
    man = d / f"sleep_edf_{tag}_n1slice_manifest.json"
    data = np.load(npz)
    X = data["X"].astype(np.float32)
    y = binary_ids_to_4way(data["y"].astype(np.int64))
    starts = data["starts"].astype(np.int64) if "starts" in data.files else np.arange(len(y), dtype=np.int64)
    counts = {HEAD_A_4WAY_LABELS[i]: int((y == i).sum()) for i in range(4)}
    return {
        "tag": tag,
        "domain": "vigilance",
        "X": X,
        "y": y,
        "starts": starts,
        "counts": counts,
        "npz_path": npz,
        "npz_sha256": sha256(npz),
        "manifest": json.loads(man.read_text()) if man.exists() else {},
    }


def load_attention(tag: str) -> Dict[str, Any]:
    d = ROOT / f"exports/windows_ds001787/{tag}"
    npz = next(d.glob("*_attention_windows.npz"))
    man_path = next(d.glob("*_attention_manifest.json"))
    data = np.load(npz)
    X = data["X"].astype(np.float32)
    y_att = data["y"].astype(np.int64)  # 0=concentration, 1=mind_wandering
    # Map attention-local ids → 4-way ids 0/1
    y = y_att.copy()
    assert set(np.unique(y_att)).issubset({0, 1}), f"unexpected att ids in {tag}"
    starts = data["starts"].astype(np.int64) if "starts" in data.files else np.arange(len(y), dtype=np.int64)
    counts = {HEAD_A_4WAY_LABELS[i]: int((y == i).sum()) for i in range(4)}
    man = json.loads(man_path.read_text())
    return {
        "tag": tag,
        "domain": "attention",
        "group": man.get("group"),
        "X": X,
        "y": y,
        "starts": starts,
        "counts": counts,
        "npz_path": npz,
        "npz_sha256": sha256(npz),
        "manifest": man,
    }


def equal_night_balance(
    packs: List[Dict[str, Any]],
    per_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Undersample each night to per_class for drowsy+hypnagogic (ids 2,3)."""
    Xs, ys = [], []
    detail = {}
    for p in packs:
        y = p["y"]
        X = p["X"]
        idxs = []
        used = {}
        for c in (2, 3):
            cand = np.where(y == c)[0]
            n = min(per_class, len(cand))
            pick = rng.choice(cand, size=n, replace=False)
            idxs.append(pick)
            used[HEAD_A_4WAY_LABELS[c]] = int(n)
        idxs = np.concatenate(idxs)
        rng.shuffle(idxs)
        Xs.append(X[idxs])
        ys.append(y[idxs])
        detail[p["tag"]] = {"per_class_used": used, "n_total": int(len(idxs))}
    return np.concatenate(Xs), np.concatenate(ys), detail


def balance_attention(
    packs: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Per-subject undersample to that subject's min(concentration, mind_wandering)."""
    Xs, ys = [], []
    detail = {}
    for p in packs:
        Xb, yb = undersample_balanced(p["X"], p["y"], rng)
        # After balance only 0/1 remain; if a subject lacks a class, undersample_balanced
        # uses min of present — skip subjects with <2 classes for train balance
        present = present_class_ids(p["y"], 4)
        if len(present) < 2:
            detail[p["tag"]] = {"skipped": True, "reason": "single_class", "counts": p["counts"]}
            continue
        Xs.append(Xb)
        ys.append(yb)
        detail[p["tag"]] = {
            "skipped": False,
            "n_total": int(len(yb)),
            "per_class": {HEAD_A_4WAY_LABELS[i]: int((yb == i).sum()) for i in present},
            "counts_full": p["counts"],
        }
    if not Xs:
        raise RuntimeError("no attention subjects with both classes for train")
    return np.concatenate(Xs), np.concatenate(ys), detail


def encode_all(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(X) == 0:
        return torch.zeros((0, 200), dtype=torch.float32)
    Xt = torch.from_numpy(X.astype(np.float32))
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            emb_list.append(encoder(Xt[i : i + BATCH].to(device)).cpu())
    return torch.cat(emb_list, dim=0)


def present_macro_f1(y: np.ndarray, pred: np.ndarray, present: List[int]) -> float:
    names = [HEAD_A_4WAY_LABELS[i] for i in present]
    remap = {c: i for i, c in enumerate(present)}
    yt = np.asarray([remap[int(v)] for v in y], dtype=np.int64)
    yp = np.asarray([remap[int(v)] for v in pred], dtype=np.int64)
    return macro_f1(yt.tolist(), yp.tolist(), names)


def eval_split(
    head: nn.Module,
    emb: torch.Tensor,
    y: np.ndarray,
    present: List[int],
    name: str,
) -> Dict[str, Any]:
    head.eval()
    with torch.no_grad():
        logits = head(emb)
        pred = predict_present(logits, present).numpy()
    names_full = HEAD_A_4WAY_LABELS
    report_full = per_class_report(y.tolist(), pred.tolist(), names_full)
    present_names = [names_full[i] for i in present]
    f1_present = present_macro_f1(y, pred, present) if len(y) else None
    acc = float((pred == y).mean()) if len(y) else None
    cm = confusion_matrix(y.tolist(), pred.tolist(), names_full).tolist()
    return {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1_present": f1_present,
        "present_classes": present_names,
        "per_class": {k: report_full[k] for k in present_names if k in report_full},
        "confusion_4way": cm,
        "counts_true": {names_full[i]: int((y == i).sum()) for i in range(4)},
    }


def stride4(X: np.ndarray, y: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(starts)
    idx = order[::4]
    return X[idx], y[idx]


def train_alternating(
    emb_vig: torch.Tensor,
    y_vig: np.ndarray,
    emb_att: torch.Tensor,
    y_att: np.ndarray,
    device: torch.device,
) -> Tuple[HeadALinear, List[Dict[str, Any]], Dict[str, Any]]:
    att_ids, vig_ids = group_ids()
    rng = np.random.default_rng(SEED)

    def split_emb(emb: torch.Tensor, y: np.ndarray):
        n = len(y)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = max(1, int(round(VAL_FRAC * n)))
        va, tr = idx[:n_val], idx[n_val:]
        return emb[tr], y[tr], emb[va], y[va]

    emb_v_tr, y_v_tr, emb_v_va, y_v_va = split_emb(emb_vig, y_vig)
    emb_a_tr, y_a_tr, emb_a_va, y_a_va = split_emb(emb_att, y_att)

    head = HeadALinear(in_dim=emb_vig.shape[-1], n_classes=4).to(device)
    # Class weights from combined train labels (full 4-way)
    y_all_tr = np.concatenate([y_v_tr, y_a_tr])
    w = class_weights_from_y(y_all_tr, n_classes=4).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)

    vig_loader = DataLoader(
        TensorDataset(emb_v_tr, torch.from_numpy(y_v_tr)),
        batch_size=BATCH,
        shuffle=True,
    )
    att_loader = DataLoader(
        TensorDataset(emb_a_tr, torch.from_numpy(y_a_tr)),
        batch_size=BATCH,
        shuffle=True,
    )

    history: List[Dict[str, Any]] = []
    for ep in range(EPOCHS):
        head.train()
        total, n_seen = 0.0, 0
        # Zip loaders; restart the shorter one
        vig_it = iter(vig_loader)
        att_it = iter(att_loader)
        n_steps = max(len(vig_loader), len(att_loader))
        for _ in range(n_steps):
            try:
                xb_v, yb_v = next(vig_it)
            except StopIteration:
                vig_it = iter(vig_loader)
                xb_v, yb_v = next(vig_it)
            try:
                xb_a, yb_a = next(att_it)
            except StopIteration:
                att_it = iter(att_loader)
                xb_a, yb_a = next(att_it)

            for xb, yb, present in (
                (xb_v, yb_v, vig_ids),
                (xb_a, yb_a, att_ids),
            ):
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = head(xb)
                loss = masked_cross_entropy(logits, yb, present, weight=w)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(yb)
                n_seen += len(yb)

        head.eval()
        with torch.no_grad():
            tr_v = predict_present(head(emb_v_tr.to(device)), vig_ids).cpu().numpy()
            va_v = predict_present(head(emb_v_va.to(device)), vig_ids).cpu().numpy()
            tr_a = predict_present(head(emb_a_tr.to(device)), att_ids).cpu().numpy()
            va_a = predict_present(head(emb_a_va.to(device)), att_ids).cpu().numpy()
        row = {
            "epoch": ep + 1,
            "loss": total / max(n_seen, 1),
            "train_vig_macro_f1": present_macro_f1(y_v_tr, tr_v, vig_ids),
            "val_vig_macro_f1": present_macro_f1(y_v_va, va_v, vig_ids),
            "train_att_macro_f1": present_macro_f1(y_a_tr, tr_a, att_ids),
            "val_att_macro_f1": present_macro_f1(y_a_va, va_a, att_ids),
        }
        history.append(row)
        print(row)

    splits = {
        "vig_train_n": int(len(y_v_tr)),
        "vig_val_n": int(len(y_v_va)),
        "att_train_n": int(len(y_a_tr)),
        "att_val_n": int(len(y_a_va)),
        "vig_val_macro_f1": history[-1]["val_vig_macro_f1"] if history else None,
        "att_val_macro_f1": history[-1]["val_att_macro_f1"] if history else None,
    }
    return head, history, splits


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    att_ids, vig_ids = group_ids()

    vig_train_packs = [load_night_4way(t) for t in VIG_TRAIN_NIGHTS]
    vig_h1 = load_night_4way(VIG_HOLDOUT_PRIMARY)
    vig_h2 = load_night_4way(VIG_HOLDOUT_SECONDARY)
    att_train_packs = [load_attention(t) for t in ATT_TRAIN_TAGS]
    att_hold = load_attention(ATT_HOLDOUT_TAG)
    att_probe = load_attention(ATT_PROBE_TAG)

    X_v, y_v, vig_bal = equal_night_balance(vig_train_packs, PER_CLASS_PER_NIGHT, rng)
    X_a, y_a, att_bal = balance_attention(att_train_packs, rng)
    print("vig balanced", X_v.shape, Counter(y_v.tolist()), vig_bal)
    print("att balanced", X_a.shape, Counter(y_a.tolist()), att_bal)

    Xh1, yh1 = stride4(vig_h1["X"], vig_h1["y"], vig_h1["starts"])
    Xh2, yh2 = stride4(vig_h2["X"], vig_h2["y"], vig_h2["starts"])

    weights = find_weights()
    encoder = FrozenCBraModEncoder(weights, source_sr=256.0, pool="mean")
    notes = encoder.adapter_notes()
    encoder.to(device)
    print("encoder", notes["native_input"], "<-", notes["fed_input"])

    print("encoding train vig…", flush=True)
    emb_v = encode_all(encoder, X_v, device)
    print("encoding train att…", flush=True)
    emb_a = encode_all(encoder, X_a, device)
    print("encoding holdouts…", flush=True)
    emb_h1 = encode_all(encoder, Xh1, device)
    emb_h2 = encode_all(encoder, Xh2, device)
    emb_ah = encode_all(encoder, att_hold["X"], device)
    # Balance holdout attention for clearer F1 (also report full)
    X_ah_b, y_ah_b = undersample_balanced(att_hold["X"], att_hold["y"], rng)
    emb_ah_b = encode_all(encoder, X_ah_b, device)
    emb_probe = encode_all(encoder, att_probe["X"], device)

    head, history, splits = train_alternating(emb_v, y_v, emb_a, y_a, device)

    evals = {
        "train_vig_balanced": eval_split(head, emb_v, y_v, vig_ids, "train_vig_balanced"),
        "train_att_balanced": eval_split(head, emb_a, y_a, att_ids, "train_att_balanced"),
        "holdout_sc4002_stride4": eval_split(head, emb_h1, yh1, vig_ids, "holdout_sc4002_stride4"),
        "holdout_sc4031_stride4": eval_split(head, emb_h2, yh2, vig_ids, "holdout_sc4031_stride4"),
        "holdout_att_sub013_full": eval_split(head, emb_ah, att_hold["y"], att_ids, "holdout_att_sub013_full"),
        "holdout_att_sub013_balanced": eval_split(head, emb_ah_b, y_ah_b, att_ids, "holdout_att_sub013_balanced"),
        "probe_att_sub017_concentration_only": eval_split(
            head, emb_probe, att_probe["y"], att_ids, "probe_att_sub017_concentration_only"
        ),
    }
    for k, v in evals.items():
        print(k, {kk: v[kk] for kk in ("n", "acc", "macro_f1_present")})

    out_dir = ROOT / "exports" / "head_a_4way_attention_train"
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / "head_a_4way_joint.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": HEAD_A_4WAY_LABELS,
            "present_at_train": HEAD_A_4WAY_LABELS,
            "train_schedule": "alternating_masked_vigilance_attention_batches",
            "in_dim": int(emb_v.shape[-1]),
            "n_classes": 4,
            "head": "HeadALinear",
            "mask": "masked_cross_entropy_present_subset",
            "vig_train_nights": VIG_TRAIN_NIGHTS,
            "att_train_tags": ATT_TRAIN_TAGS,
            "att_holdout_tag": ATT_HOLDOUT_TAG,
        },
        head_path,
    )

    metrics = {
        "train_schedule": "alternating_masked_batches",
        "vig_train_nights": VIG_TRAIN_NIGHTS,
        "vig_balance": vig_bal,
        "att_train_tags": ATT_TRAIN_TAGS,
        "att_balance": att_bal,
        "att_holdout_tag": ATT_HOLDOUT_TAG,
        "att_probe_tag": ATT_PROBE_TAG,
        "splits": splits,
        "history": history,
        "evals": evals,
        "headline": {
            "val_vig_macro_f1": splits["vig_val_macro_f1"],
            "val_att_macro_f1": splits["att_val_macro_f1"],
            "sc4002_stride4_macro_f1": evals["holdout_sc4002_stride4"]["macro_f1_present"],
            "sc4031_stride4_macro_f1": evals["holdout_sc4031_stride4"]["macro_f1_present"],
            "att_sub013_full_macro_f1": evals["holdout_att_sub013_full"]["macro_f1_present"],
            "att_sub013_balanced_macro_f1": evals["holdout_att_sub013_balanced"]["macro_f1_present"],
            "att_sub017_probe_acc": evals["probe_att_sub017_concentration_only"]["acc"],
        },
    }
    metrics_path = out_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    manifest = {
        "task": "head_a_4way_attention_train",
        "label_list": HEAD_A_4WAY_LABELS,
        "train_schedule": {
            "type": "alternating_masked_batches",
            "vigilance_present": [HEAD_A_4WAY_LABELS[i] for i in vig_ids],
            "attention_present": [HEAD_A_4WAY_LABELS[i] for i in att_ids],
            "rationale": "domain batches never train the other domain's logits via masked CE",
        },
        "vigilance": {
            "train_nights": VIG_TRAIN_NIGHTS,
            "balance": vig_bal,
            "holdout_primary": VIG_HOLDOUT_PRIMARY,
            "holdout_secondary": VIG_HOLDOUT_SECONDARY,
            "npz_sha256": {p["tag"]: p["npz_sha256"] for p in vig_train_packs + [vig_h1, vig_h2]},
        },
        "attention": {
            "train_tags": ATT_TRAIN_TAGS,
            "holdout_tag": ATT_HOLDOUT_TAG,
            "probe_tag": ATT_PROBE_TAG,
            "balance": att_bal,
            "label_rule": "Q1>Q2→concentration; Q1<Q2→mind_wandering; ties dropped",
            "hard_reject": "Do not map events value 2/4 to classes",
            "npz_sha256": {
                p["tag"]: p["npz_sha256"]
                for p in att_train_packs + [att_hold, att_probe]
            },
        },
        "encoder_name": "CBraMod",
        "encoder_weights_path": str(weights),
        "encoder_weights_sha256": notes["weights_sha256"],
        "encoder_adapter": notes,
        "head": "HeadALinear",
        "n_classes": 4,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "per_class_per_night": PER_CLASS_PER_NIGHT,
        "history": history,
        "metrics_ref": str(metrics_path.relative_to(ROOT)),
        "head_path": str(head_path.relative_to(ROOT)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_attribution": {
            "Sleep-EDF": "PhysioNet ODC-By — attribute PhysioNet / Sleep-EDF Expanded",
            "ds001787": "OpenNeuro CC0-1.0 — Brandmeyer & Delorme meditation MW",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
        "caveats": [
            "montage mismatch: Sleep-EDF vs CBraMod TUEG pretrain vs BioSemi→Muse-proxy ds001787",
            "attention holdout is one novice subject (sub013); sub017 all-concentration excluded from train",
            "overlapping 2s windows; vigilance eval uses stride×4",
        ],
    }
    man_path = out_dir / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))

    summary = {
        "step": "head_a_4way_attention_train",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "vig_train_nights": VIG_TRAIN_NIGHTS,
        "att_train_tags": ATT_TRAIN_TAGS,
        "att_holdout_tag": ATT_HOLDOUT_TAG,
        "headline": metrics["headline"],
        "takeaway": (
            "Joint 4-way Head A with alternating masked domain batches. "
            f"SC4002 stride4 vig F1={metrics['headline']['sc4002_stride4_macro_f1']:.3f}; "
            f"att sub013 balanced F1={metrics['headline']['att_sub013_balanced_macro_f1']:.3f}."
        ),
        "artifacts": {
            "script": "scripts/head_a_4way_attention_train.py",
            "docs": "docs/head_a_4way_attention_train.md",
            "metrics": str(metrics_path.relative_to(ROOT)),
            "manifest": str(man_path.relative_to(ROOT)),
            "head": str(head_path.relative_to(ROOT)),
        },
    }
    (out_dir / "step_summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

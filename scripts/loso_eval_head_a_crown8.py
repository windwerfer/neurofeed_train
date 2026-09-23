#!/usr/bin/env python3
"""LOSO Head A on Crown8 attention packs (C=8). Parallel to muse4 loso_eval_head_a.

Usage:
  uv run python scripts/loso_eval_head_a_crown8.py --dry-run
  uv run python scripts/loso_eval_head_a_crown8.py --max-folds 3
  uv run python scripts/loso_eval_head_a_crown8.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder
from src.channel_map import CROWN8_CHANNELS
from src.head_a import ATTENTION_LABELS, HeadALinear, class_weights_from_y
from src.metrics import per_class_report

SEED = 42
EPOCHS = 5
BATCH = 32
LR = 1e-3
ATT_CORPORA = ["attention_ds001787_crown8", "attention_ds003969_crown8"]
OUT_DIR = ROOT / "exports" / "loso_head_a_attention_crown8"
MUSE4_REF_F1 = 0.36098056607840057  # exports/loso_head_a_attention/metrics_summary.json


def find_weights() -> Path:
    cands = [
        ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth",
        Path("/tmp/kaggle_out3/models/CBraMod/pretrained_weights.pth"),
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("CBraMod weights not found")


def subject_for_recording(corpus: str, recording_id: str) -> str:
    if corpus.startswith("attention_ds001787"):
        core = recording_id.split("_")[0]
        num = core.replace("sub", "")
        sid = f"sub-{num.zfill(3) if num.isdigit() else num}"
        return f"ds001787/{sid}"
    if corpus.startswith("attention_ds003969"):
        num = recording_id.replace("sub", "")
        sid = f"sub-{num.zfill(3) if num.isdigit() else num}"
        return f"ds003969/{sid}"
    return f"{corpus}/{recording_id}"


def load_attention_packs(qc_only: bool = True) -> List[Dict[str, Any]]:
    packs: List[Dict[str, Any]] = []
    for corpus in ATT_CORPORA:
        win_dir = ROOT / "datasets" / corpus / "windows"
        if not win_dir.exists():
            print(f"WARN missing {win_dir}", flush=True)
            continue
        for npz_path in sorted(win_dir.glob("*_windows.npz")):
            rid = npz_path.name.replace("_windows.npz", "")
            data = np.load(npz_path, allow_pickle=True)
            X = data["X"].astype(np.float32)
            if X.ndim != 3 or X.shape[1] != 8:
                raise ValueError(f"{npz_path} expected (N,8,T), got {X.shape}")
            y_local = data["y"].astype(np.int64)
            names = [str(n) for n in data["label_names"].tolist()]
            table = {i: ATTENTION_LABELS.index(n) for i, n in enumerate(names) if n in ATTENTION_LABELS}
            mask = np.asarray([int(v) in table for v in y_local], dtype=bool)
            if qc_only:
                qc_path = win_dir / f"{rid}_qc.npz"
                if qc_path.exists():
                    mask &= np.load(qc_path)["qc_pass"].astype(bool)
            if not mask.any():
                continue
            y = np.asarray([table[int(v)] for v in y_local[mask]], dtype=np.int64)
            packs.append({
                "corpus": corpus,
                "recording_id": rid,
                "subject_id": subject_for_recording(corpus, rid),
                "montage_id": "crown8",
                "X": X[mask],
                "y": y,
            })
    return packs


def encode(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    outs = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            outs.append(encoder(torch.from_numpy(X[i : i + BATCH]).to(device)).cpu())
    return torch.cat(outs, dim=0)


def train_eval_fold_emb(
    emb_tr: torch.Tensor,
    y_tr: np.ndarray,
    emb_te: torch.Tensor,
    y_te: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    idx0 = np.where(y_tr == 0)[0]
    idx1 = np.where(y_tr == 1)[0]
    n = min(len(idx0), len(idx1))
    if n == 0:
        return {
            "n_train_balanced": 0,
            "n_test": int(len(y_te)),
            "test_counts": {ATTENTION_LABELS[i]: int((y_te == i).sum()) for i in range(2)},
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "per_class": {},
            "skip_reason": "no both-class train",
        }
    pick0 = rng.choice(idx0, size=n, replace=False)
    pick1 = rng.choice(idx1, size=n, replace=False)
    bal = np.concatenate([pick0, pick1])
    rng.shuffle(bal)
    emb_b = emb_tr[bal]
    yb = y_tr[bal]
    head = HeadALinear(in_dim=emb_b.shape[-1], n_classes=2).to(device)
    w = class_weights_from_y(yb, n_classes=2).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(emb_b, torch.from_numpy(yb)), batch_size=BATCH, shuffle=True)
    for _ in range(EPOCHS):
        head.train()
        for xb, ybatch in loader:
            opt.zero_grad()
            loss = crit(head(xb.to(device)), ybatch.to(device))
            loss.backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        pred = head(emb_te.to(device)).argmax(dim=-1).cpu().numpy()
    report = per_class_report(y_te.tolist(), pred.tolist(), ATTENTION_LABELS)
    return {
        "n_train_balanced": int(len(yb)),
        "n_test": int(len(y_te)),
        "test_counts": {ATTENTION_LABELS[i]: int((y_te == i).sum()) for i in range(2)},
        "macro_f1": float(report["macro_f1"]["f1"]),
        "accuracy": float((pred == y_te).mean()) if len(y_te) else 0.0,
        "per_class": {k: v for k, v in report.items() if k != "macro_f1"},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--min-test-windows", type=int, default=20)
    ap.add_argument("--require-both-classes", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    packs = load_attention_packs(qc_only=True)
    by_subj: Dict[str, List[Dict[str, Any]]] = {}
    for p in packs:
        by_subj.setdefault(p["subject_id"], []).append(p)
    subjects = sorted(by_subj)
    print(f"loaded {len(packs)} recordings / {len(subjects)} subjects (crown8)", flush=True)

    fold_subjects = []
    skipped = []
    for sid in subjects:
        y = np.concatenate([p["y"] for p in by_subj[sid]])
        counts = Counter(y.tolist())
        if len(y) < args.min_test_windows:
            skipped.append({"subject": sid, "reason": "too_few_windows", "n": int(len(y))})
            continue
        if args.require_both_classes and len(counts) < 2:
            skipped.append({"subject": sid, "reason": "single_class", "counts": {str(k): v for k, v in counts.items()}})
            continue
        fold_subjects.append(sid)
    if args.max_folds > 0:
        fold_subjects = fold_subjects[: args.max_folds]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = {
        "step": "loso_eval_head_a_crown8",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "montage_id": "crown8",
        "channels": list(CROWN8_CHANNELS),
        "n_subjects_available": len(subjects),
        "n_folds": len(fold_subjects),
        "fold_subjects": fold_subjects,
        "skipped": skipped,
        "id_policy": "corpus-qualified (ds001787/sub-XXX | ds003969/sub-XXX)",
        "labels": list(ATTENTION_LABELS),
        "qc": "qc_pass windows only",
        "muse4_loso_macro_f1_ref": MUSE4_REF_F1,
        "interpretation_rule": (
            "If crown8 >> chance (~0.5 balanced / muse4~0.36) → montage mattered; "
            "if crown8 also ~chance → label/task mismatch is prime suspect."
        ),
        "out_dir": str(OUT_DIR),
    }
    (OUT_DIR / "fold_plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    if args.dry_run or not fold_subjects:
        print(json.dumps(plan, indent=2))
        print(f"wrote {OUT_DIR / 'fold_plan.json'} (dry_run or empty)")
        return

    device = torch.device("cpu")
    encoder = FrozenCBraModEncoder(find_weights(), source_sr=256.0, pool="mean").to(device)
    emb_by_subj: Dict[str, torch.Tensor] = {}
    y_by_subj: Dict[str, np.ndarray] = {}
    for sid in subjects:
        X = np.concatenate([p["X"] for p in by_subj[sid]])
        y = np.concatenate([p["y"] for p in by_subj[sid]])
        print(f"encode {sid} n={len(y)} C={X.shape[1]}", flush=True)
        emb_by_subj[sid] = encode(encoder, X, device)
        y_by_subj[sid] = y
    del encoder
    for plist in by_subj.values():
        for p in plist:
            p.pop("X", None)

    folds = []
    for i, sid in enumerate(fold_subjects, 1):
        emb_te = emb_by_subj[sid]
        y_te = y_by_subj[sid]
        tr_sids = [s for s in subjects if s != sid]
        emb_tr = torch.cat([emb_by_subj[s] for s in tr_sids], dim=0)
        y_tr = np.concatenate([y_by_subj[s] for s in tr_sids])
        print(f"fold {i}/{len(fold_subjects)} holdout={sid} n_te={len(y_te)} n_tr={len(y_tr)}", flush=True)
        m = train_eval_fold_emb(emb_tr, y_tr, emb_te, y_te, device, rng)
        m["holdout_subject"] = sid
        folds.append(m)
        print(f"  macro_f1={m['macro_f1']:.3f} acc={m['accuracy']:.3f}", flush=True)

    f1s = [f["macro_f1"] for f in folds]
    mean_f1 = float(np.mean(f1s)) if f1s else None
    std_f1 = float(np.std(f1s)) if f1s else None
    if mean_f1 is None:
        verdict = "no_folds"
    elif mean_f1 >= 0.55:
        verdict = "crown8_above_chance_montage_likely_mattered"
    elif mean_f1 <= 0.45:
        verdict = "crown8_near_or_below_chance_label_task_mismatch_prime_suspect"
    else:
        verdict = "crown8_ambiguous_near_chance"
    ship = bool(mean_f1 is not None and mean_f1 > 0.60 and (mean_f1 - (std_f1 or 0)) > 0.55)
    summary = {
        "step": "loso_eval_head_a_crown8",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "montage_id": "crown8",
        "channels": list(CROWN8_CHANNELS),
        "ship_candidate": ship,
        "verdict": verdict,
        "n_subjects_available": len(subjects),
        "n_folds": len(folds),
        "macro_f1_mean": mean_f1,
        "macro_f1_std": std_f1,
        "accuracy_mean": float(np.mean([f["accuracy"] for f in folds])) if folds else None,
        "muse4_loso_macro_f1_ref": MUSE4_REF_F1,
        "delta_vs_muse4": (mean_f1 - MUSE4_REF_F1) if mean_f1 is not None else None,
        "folds": folds,
        "chance_note": "Binary chance ≈ 0.5 macro-F1 if balanced; muse4 ref ≈ 0.36.",
        "labels": list(ATTENTION_LABELS),
        "encoder": "FrozenCBraMod + HeadALinear (C=8)",
        "qc": "qc_pass only",
    }
    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        json.dumps({"holdout": f["holdout_subject"], "macro_f1": f["macro_f1"], "accuracy": f["accuracy"], "n_test": f["n_test"]})
        for f in folds
    ]
    (OUT_DIR / "folds.jsonl").write_text("\n".join(lines) + "\n")
    print(
        f"DONE n_folds={len(folds)} macro_f1_mean={mean_f1:.4f}±{std_f1:.4f} "
        f"muse4_ref={MUSE4_REF_F1:.4f} delta={summary['delta_vs_muse4']:.4f} verdict={verdict}",
        flush=True,
    )
    print(f"wrote {OUT_DIR / 'metrics_summary.json'}", flush=True)


if __name__ == "__main__":
    main()

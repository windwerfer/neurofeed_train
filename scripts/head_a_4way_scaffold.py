#!/usr/bin/env python3
"""Wave 2: scaffold 4-way Head A with missing-class mask; smoke on available labels.

Available today: Sleep-EDF vigilance only (drowsy / hypnagogic). Attention classes
(concentration / mind_wandering) stay in the head but are masked out of CE/softmax
until ALLOW corpora are ingested.
"""
from __future__ import annotations

import json
import hashlib
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
from src.head_a import (
    HEAD_A_4WAY_LABELS,
    HEAD_A_BINARY_LABELS,
    HeadALinear,
    binary_ids_to_4way,
    class_weights_from_y,
    masked_cross_entropy,
    present_class_ids,
    predict_present,
    undersample_balanced,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report

SEED = 42
EPOCHS = 6
BATCH = 32
LR = 1e-3
VAL_FRAC = 0.2
TRAIN_NIGHT = "sc4001"
HOLDOUT_NIGHT = "sc4002"


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


def load_night_binary(tag: str) -> Dict[str, Any]:
    d = ROOT / f"exports/windows_{tag}"
    npz = d / f"sleep_edf_{tag}_n1slice_windows.npz"
    man = d / f"sleep_edf_{tag}_n1slice_manifest.json"
    data = np.load(npz)
    X = data["X"].astype(np.float32)
    y_bin = data["y"].astype(np.int64)
    y = binary_ids_to_4way(y_bin)
    starts = data["starts"].astype(np.int64) if "starts" in data.files else np.arange(len(y), dtype=np.int64)
    manifest = json.loads(man.read_text()) if man.exists() else {}
    counts = {HEAD_A_4WAY_LABELS[i]: int((y == i).sum()) for i in range(len(HEAD_A_4WAY_LABELS))}
    return {
        "tag": tag,
        "X": X,
        "y": y,
        "y_bin": y_bin,
        "starts": starts,
        "manifest": manifest,
        "counts": counts,
        "npz_path": npz,
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


def present_macro_f1(y: np.ndarray, pred: np.ndarray, present: List[int]) -> float:
    """Macro-F1 only over present classes (names subset of HEAD_A_4WAY_LABELS)."""
    names = [HEAD_A_4WAY_LABELS[i] for i in present]
    # Remap to dense 0..len(present)-1 for metrics helpers
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
    f1_present = present_macro_f1(y, pred, present)
    acc = float((pred == y).mean()) if len(y) else None
    cm = confusion_matrix(y.tolist(), pred.tolist(), names_full).tolist()
    return {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1_present": f1_present,
        "macro_f1_4way_naive": float(report_full["macro_f1"]["f1"]),
        "present_classes": present_names,
        "per_class": {k: report_full[k] for k in present_names},
        "confusion_4way": cm,
        "counts_true": {names_full[i]: int((y == i).sum()) for i in range(4)},
        "note": "macro_f1_present ignores missing attention classes; naive 4-way includes zeros for them",
    }


def train_masked(
    emb: torch.Tensor,
    y: np.ndarray,
    present: List[int],
    device: torch.device,
) -> tuple[HeadALinear, List[Dict[str, Any]], np.ndarray, np.ndarray]:
    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    n_val = max(1, int(round(VAL_FRAC * n)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    emb_tr, y_tr = emb[tr_idx], y[tr_idx]
    emb_va, y_va = emb[val_idx], y[val_idx]

    head = HeadALinear(in_dim=emb.shape[-1], n_classes=4).to(device)
    # Weights over full 4-way vector; masked_ce subsets them
    w = class_weights_from_y(y_tr, n_classes=4).to(device)
    # Zero weight on missing classes (defensive; mask already drops them)
    for i in range(4):
        if i not in present:
            w[i] = 0.0
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(emb_tr, torch.from_numpy(y_tr)),
        batch_size=BATCH,
        shuffle=True,
    )
    history: List[Dict[str, Any]] = []
    for ep in range(EPOCHS):
        head.train()
        total, n_seen = 0.0, 0
        for xb, yb in loader:
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
            tr_pred = predict_present(head(emb_tr.to(device)), present).cpu().numpy()
            va_pred = predict_present(head(emb_va.to(device)), present).cpu().numpy()
        row = {
            "epoch": ep + 1,
            "loss": total / max(n_seen, 1),
            "train_acc": float((tr_pred == y_tr).mean()),
            "train_macro_f1_present": present_macro_f1(y_tr, tr_pred, present),
            "val_acc": float((va_pred == y_va).mean()),
            "val_macro_f1_present": present_macro_f1(y_va, va_pred, present),
        }
        history.append(row)
        print(row)
    return head, history, tr_idx, val_idx


def unit_check_mask(device: torch.device) -> Dict[str, Any]:
    """Sanity: missing-class columns get ~0 grad when masked; present columns move."""
    torch.manual_seed(0)
    head = HeadALinear(in_dim=8, n_classes=4, dropout=0.0).to(device)
    x = torch.randn(16, 8, device=device)
    # only classes 2 and 3
    y = torch.tensor([2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3], device=device)
    present = [2, 3]
    opt = torch.optim.SGD(head.parameters(), lr=0.1)
    opt.zero_grad()
    logits = head(x)
    loss = masked_cross_entropy(logits, y, present)
    loss.backward()
    # Linear weight grad shape [4, 8]
    w_grad = head.net[1].weight.grad.detach().cpu().numpy()
    present_norm = float(np.linalg.norm(w_grad[present]))
    missing_norm = float(np.linalg.norm(w_grad[[0, 1]]))
    return {
        "present_grad_norm": present_norm,
        "missing_grad_norm": missing_norm,
        "missing_near_zero": missing_norm < 1e-8,
        "present_moves": present_norm > 1e-6,
        "ok": missing_norm < 1e-8 and present_norm > 1e-6,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")

    mask_check = unit_check_mask(device)
    print("mask_unit_check", mask_check)
    if not mask_check["ok"]:
        raise RuntimeError(f"missing-class mask unit check failed: {mask_check}")

    train_night = load_night_binary(TRAIN_NIGHT)
    hold_night = load_night_binary(HOLDOUT_NIGHT)
    print("train counts", train_night["counts"])
    print("holdout counts", hold_night["counts"])

    present = present_class_ids(train_night["y"], n_classes=4)
    assert present == [2, 3], f"expected vigilance-only present={present}"

    # Balance train night on present classes
    Xb, yb = undersample_balanced(train_night["X"], train_night["y"], rng)
    print("balanced", Xb.shape, Counter(yb.tolist()))

    weights = find_weights()
    encoder = FrozenCBraModEncoder(weights, source_sr=256.0, pool="mean")
    notes = encoder.adapter_notes()
    encoder.to(device)
    print("encoder", notes["native_input"], "<-", notes["fed_input"])

    emb_tr = encode_all(encoder, Xb, device)
    # Holdout: stride×4 non-overlap approx via starts if available
    Xh, yh, sh = hold_night["X"], hold_night["y"], hold_night["starts"]
    # Keep every 4th window by start order for lighter CPU eval
    order = np.argsort(sh)
    stride_idx = order[::4]
    Xh_s, yh_s = Xh[stride_idx], yh[stride_idx]
    emb_h = encode_all(encoder, Xh_s, device)

    head, history, tr_idx, val_idx = train_masked(emb_tr, yb, present, device)

    train_eval = eval_split(head, emb_tr, yb, present, "train_balanced_sc4001")
    hold_eval = eval_split(head, emb_h, yh_s, present, "holdout_sc4002_stride4")

    # Compare against binary 2-way head shape expectations (plumbing only)
    print("train_eval", {k: train_eval[k] for k in ("acc", "macro_f1_present", "macro_f1_4way_naive")})
    print("hold_eval", {k: hold_eval[k] for k in ("acc", "macro_f1_present", "macro_f1_4way_naive")})

    out_dir = ROOT / "exports" / "head_a_4way_scaffold"
    out_dir.mkdir(parents=True, exist_ok=True)
    head_path = out_dir / "head_a_4way_masked_smoke.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": HEAD_A_4WAY_LABELS,
            "present_at_train": [HEAD_A_4WAY_LABELS[i] for i in present],
            "in_dim": int(emb_tr.shape[-1]),
            "n_classes": 4,
            "head": "HeadALinear",
            "mask": "masked_cross_entropy_present_subset",
        },
        head_path,
    )

    metrics = {
        "mask_unit_check": mask_check,
        "present_classes": [HEAD_A_4WAY_LABELS[i] for i in present],
        "missing_classes": [n for i, n in enumerate(HEAD_A_4WAY_LABELS) if i not in present],
        "train_night": TRAIN_NIGHT,
        "holdout_night": HOLDOUT_NIGHT,
        "balanced_n_per_class": int(len(yb) // max(len(present), 1)),
        "history": history,
        "train_eval": train_eval,
        "holdout_eval": hold_eval,
        "final_val_macro_f1_present": history[-1]["val_macro_f1_present"] if history else None,
        "holdout_macro_f1_present": hold_eval["macro_f1_present"],
    }
    metrics_path = out_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    manifest = {
        "task": "head_a_4way_scaffold_smoke",
        "label_list": HEAD_A_4WAY_LABELS,
        "binary_label_list": HEAD_A_BINARY_LABELS,
        "present_at_train": [HEAD_A_4WAY_LABELS[i] for i in present],
        "missing_at_train": [n for i, n in enumerate(HEAD_A_4WAY_LABELS) if i not in present],
        "mask": {
            "train_loss": "masked_cross_entropy over present class subset",
            "predict": "argmax restricted to present classes",
            "rationale": "attention-only and vigilance-only corpora can share one 4-way head without training missing logits",
        },
        "train_night": {
            "tag": TRAIN_NIGHT,
            "counts_full": train_night["counts"],
            "npz_sha256": train_night["npz_sha256"],
        },
        "holdout_night": {
            "tag": HOLDOUT_NIGHT,
            "counts_full": hold_night["counts"],
            "npz_sha256": hold_night["npz_sha256"],
            "eval_stride": 4,
            "n_eval": int(len(yh_s)),
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
        "history": history,
        "metrics_ref": str(metrics_path.relative_to(ROOT)),
        "head_path": str(head_path.relative_to(ROOT)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_attribution": {
            "Sleep-EDF": "PhysioNet ODC-By — attribute PhysioNet / Sleep-EDF Expanded",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
        "next": [
            "ingest small ds001787 Muse-proxy windows after response codebook confirm",
            "joint train with attention present-mask batches alternating vigilance batches",
        ],
    }
    man_path = out_dir / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))

    summary = {
        "step": "head_a_4way_scaffold",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "mask_unit_check_ok": mask_check["ok"],
        "present_classes": metrics["present_classes"],
        "missing_classes": metrics["missing_classes"],
        "train_macro_f1_present": train_eval["macro_f1_present"],
        "val_macro_f1_present": history[-1]["val_macro_f1_present"],
        "holdout_sc4002_stride4_macro_f1_present": hold_eval["macro_f1_present"],
        "holdout_acc": hold_eval["acc"],
        "artifacts": {
            "script": "scripts/head_a_4way_scaffold.py",
            "docs": "docs/head_a_4way_scaffold.md",
            "metrics": str(metrics_path.relative_to(ROOT)),
            "manifest": str(man_path.relative_to(ROOT)),
            "head": str(head_path.relative_to(ROOT)),
            "src": "src/head_a.py",
        },
    }
    (out_dir / "step_summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

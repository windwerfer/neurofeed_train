#!/usr/bin/env python3
"""Overnight step 3 (next_index=2): stride-aware / non-overlapping holdout eval on SC4002."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder
from src.head_a import HEAD_A_BINARY_LABELS, HeadALinear
from src.metrics import confusion_matrix, macro_f1, per_class_report

SEED = 42
BATCH = 32
WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0


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


def report_split(y: np.ndarray, pred: np.ndarray, name: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if len(y) == 0:
        out = {"name": name, "n": 0, "acc": None, "macro_f1": None, "per_class": {}, "confusion": [], "counts_true": {}}
        if extra:
            out.update(extra)
        return out
    acc = float((pred == y).mean())
    f1 = macro_f1(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    report = per_class_report(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    per_class = {k: v for k, v in report.items() if k != "macro_f1"}
    cm = confusion_matrix(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    counts_true = {HEAD_A_BINARY_LABELS[i]: int((y == i).sum()) for i in range(2)}
    out: Dict[str, Any] = {
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
    """Greedy left-to-right: keep a window only if it does not overlap the previous kept one."""
    order = np.argsort(starts)
    kept: List[int] = []
    last_end = -10**18
    for i in order:
        s = int(starts[i])
        if s >= last_end:
            kept.append(int(i))
            last_end = s + window_samples
    return np.asarray(kept, dtype=np.int64)


def stride_indices(n: int, stride: int, offset: int = 0) -> np.ndarray:
    return np.arange(offset, n, stride, dtype=np.int64)


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


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    npz_path = ROOT / "exports/windows_sc4002/sleep_edf_sc4002_n1slice_windows.npz"
    head_path = ROOT / "exports/head_a_holdout/head_a_binary_state_dict.pt"
    weights = find_weights()

    data = np.load(npz_path)
    X = data["X"]
    y = data["y"].astype(np.int64)
    starts = data["starts"].astype(np.int64)

    window_samples = int(round(WINDOW_SEC * TARGET_SR))  # 512
    hop_samples = int(round(HOP_SEC * TARGET_SR))  # 128
    stride = max(1, window_samples // hop_samples)  # 4 → non-overlap by fixed stride

    device = torch.device("cpu")
    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
    encoder.to(device)
    print("encoding SC4002...", X.shape)
    emb = encode_all(encoder, X, device)

    head = HeadALinear(in_dim=emb.shape[-1], n_classes=2)
    head.load_state_dict(torch.load(head_path, map_location="cpu"))
    head.eval()
    with torch.no_grad():
        pred_all = head(emb).argmax(dim=-1).numpy()

    # 1) Overlapping (original scoring)
    m_overlap = report_split(
        y,
        pred_all,
        "sc4002_overlap_hop0.5s",
        extra={"scoring": "overlap", "hop_sec": HOP_SEC, "window_sec": WINDOW_SEC},
    )

    # 2) Fixed stride subsample (every 4th window from offset 0) — non-overlapping grid
    idx_stride = stride_indices(len(y), stride, offset=0)
    m_stride = report_split(
        y[idx_stride],
        pred_all[idx_stride],
        "sc4002_nonoverlap_stride4",
        extra={
            "scoring": "fixed_stride",
            "stride_windows": stride,
            "effective_hop_sec": WINDOW_SEC,
            "n_selected": int(len(idx_stride)),
            "offset": 0,
        },
    )

    # Also report mean over offsets 0..stride-1 for stability
    offset_rows = []
    for off in range(stride):
        idx = stride_indices(len(y), stride, offset=off)
        row = report_split(y[idx], pred_all[idx], f"sc4002_stride4_offset{off}")
        offset_rows.append(
            {
                "offset": off,
                "n": row["n"],
                "acc": row["acc"],
                "macro_f1": row["macro_f1"],
                "per_class": row["per_class"],
                "counts_true": row["counts_true"],
            }
        )
    mean_acc = float(np.mean([r["acc"] for r in offset_rows]))
    mean_f1 = float(np.mean([r["macro_f1"] for r in offset_rows]))

    # 3) Greedy non-overlap by starts (handles any gaps)
    idx_greedy = nonoverlap_indices(starts, window_samples)
    m_greedy = report_split(
        y[idx_greedy],
        pred_all[idx_greedy],
        "sc4002_nonoverlap_greedy",
        extra={
            "scoring": "greedy_nonoverlap",
            "window_samples": window_samples,
            "n_selected": int(len(idx_greedy)),
        },
    )

    # 4) Balanced non-overlap (undersample on greedy set)
    idx_bal = undersample_balanced_idx(y, idx_greedy, rng)
    m_bal = report_split(
        y[idx_bal],
        pred_all[idx_bal],
        "sc4002_nonoverlap_greedy_balanced",
        extra={"scoring": "greedy_nonoverlap_balanced", "seed": SEED, "n_selected": int(len(idx_bal))},
    )

    # 5) Balanced stride offset-0
    idx_stride_bal = undersample_balanced_idx(y, idx_stride, rng)
    m_stride_bal = report_split(
        y[idx_stride_bal],
        pred_all[idx_stride_bal],
        "sc4002_nonoverlap_stride4_balanced",
        extra={"scoring": "fixed_stride_balanced", "seed": SEED, "n_selected": int(len(idx_stride_bal))},
    )

    out_dir = ROOT / "exports" / "head_a_holdout"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "tighten_holdout_eval_stride_aware",
        "holdout_night": "SC4002",
        "head_path": str(head_path),
        "windows_npz": str(npz_path),
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "stride_windows": stride,
        "rationale": (
            "Training windows used hop=0.5s on 2s windows (4× overlap). "
            "Overlapping scoring inflates n and correlates neighbors. "
            "Non-overlapping / stride-aware scoring keeps windows ≥2s apart so metrics "
            "are closer to independent samples."
        ),
        "metrics": {
            "overlap_hop0.5s": m_overlap,
            "nonoverlap_stride4": m_stride,
            "nonoverlap_stride4_offset_mean": {
                "mean_acc": mean_acc,
                "mean_macro_f1": mean_f1,
                "offsets": offset_rows,
            },
            "nonoverlap_greedy": m_greedy,
            "nonoverlap_greedy_balanced": m_bal,
            "nonoverlap_stride4_balanced": m_stride_bal,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "label_list": HEAD_A_BINARY_LABELS,
    }
    metrics_path = out_dir / "tighten_eval_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))
    print("wrote", metrics_path)
    print("stride-offset mean acc", mean_acc, "macro_f1", mean_f1)


if __name__ == "__main__":
    main()

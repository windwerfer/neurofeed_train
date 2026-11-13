"""Tiny Head A classifiers on frozen encoder embeddings.

LEGACY 4-way + binary helpers (masked CE). Product architecture is
**multi-head** — see ``src.heads`` and ``docs/head_a_multihead.md``
(decision 2026-09-07): A-vig / A-med / A-eng separate Linear heads.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Smoke binary: drowsy vs hypnagogic (Sleep-EDF W / N1 proxies)
HEAD_A_BINARY_LABELS: List[str] = ["drowsy", "hypnagogic"]

# Locked 4-way Head A (attention + vigilance)
HEAD_A_4WAY_LABELS: List[str] = [
    "concentration",
    "mind_wandering",
    "drowsy",
    "hypnagogic",
]

ATTENTION_LABELS: List[str] = ["concentration", "mind_wandering"]
VIGILANCE_LABELS: List[str] = ["drowsy", "hypnagogic"]


class HeadALinear(nn.Module):
    def __init__(self, in_dim: int = 200, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_classes = int(n_classes)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeadAMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 200,
        n_classes: int = 2,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def labels_to_ids(labels: Sequence[str], label_list: Sequence[str]) -> np.ndarray:
    table = {n: i for i, n in enumerate(label_list)}
    return np.asarray([table[str(v)] for v in labels], dtype=np.int64)


def binary_ids_to_4way(y_binary: np.ndarray) -> np.ndarray:
    """Map HEAD_A_BINARY_LABELS ids → HEAD_A_4WAY_LABELS ids (drowsy=2, hypnagogic=3)."""
    y_binary = np.asarray(y_binary, dtype=np.int64)
    out = np.empty_like(y_binary)
    # binary 0 drowsy → 2; binary 1 hypnagogic → 3
    out[y_binary == 0] = HEAD_A_4WAY_LABELS.index("drowsy")
    out[y_binary == 1] = HEAD_A_4WAY_LABELS.index("hypnagogic")
    if np.any((y_binary != 0) & (y_binary != 1)):
        bad = np.unique(y_binary[(y_binary != 0) & (y_binary != 1)])
        raise ValueError(f"unexpected binary ids: {bad.tolist()}")
    return out


def undersample_balanced(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Undersample majority classes to min class count (smoke clarity)."""
    classes, counts = np.unique(y, return_counts=True)
    m = int(counts.min())
    idxs = []
    for c in classes:
        cand = np.nonzero(y == c)[0]
        pick = rng.choice(cand, size=m, replace=False)
        idxs.append(pick)
    idxs = np.concatenate(idxs)
    rng.shuffle(idxs)
    return X[idxs], y[idxs]


def class_weights_from_y(y: np.ndarray, n_classes: Optional[int] = None) -> torch.Tensor:
    """Inverse-frequency weights for CrossEntropyLoss (full class vector)."""
    if n_classes is None:
        n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)


def present_class_ids(y: np.ndarray, n_classes: int) -> List[int]:
    """Sorted unique class ids that appear in y (and lie in [0, n_classes))."""
    present = sorted({int(c) for c in np.unique(y) if 0 <= int(c) < n_classes})
    return present


def class_presence_mask(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Bool mask [n_classes] — True for classes present in y."""
    mask = torch.zeros(n_classes, dtype=torch.bool)
    for c in present_class_ids(y, n_classes):
        mask[c] = True
    return mask


def masked_cross_entropy(
    logits: torch.Tensor,
    y: torch.Tensor,
    present: Sequence[int],
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CE over a subset of logits so missing classes never enter the softmax.

    Used when a batch / corpus only has some of the 4 Head A labels (e.g. Sleep-EDF
    vigilance only). Attention logits stay untrained on vigilance-only batches and
    vice versa.
    """
    if len(present) == 0:
        return logits.sum() * 0.0
    present_t = torch.as_tensor(list(present), device=logits.device, dtype=torch.long)
    # Remap global ids → positions within present subset
    remap = torch.full((logits.shape[-1],), -1, device=logits.device, dtype=torch.long)
    remap[present_t] = torch.arange(len(present), device=logits.device, dtype=torch.long)
    y_sub = remap[y]
    if (y_sub < 0).any():
        raise ValueError("y contains class ids outside the present mask")
    logits_sub = logits.index_select(dim=-1, index=present_t)
    w_sub = None
    if weight is not None:
        w_sub = weight.index_select(dim=0, index=present_t)
    return F.cross_entropy(logits_sub, y_sub, weight=w_sub)


def predict_present(
    logits: torch.Tensor,
    present: Sequence[int],
) -> torch.Tensor:
    """Argmax restricted to present classes (global ids)."""
    present_t = torch.as_tensor(list(present), device=logits.device, dtype=torch.long)
    sub = logits.index_select(dim=-1, index=present_t)
    local = sub.argmax(dim=-1)
    return present_t[local]


def group_ids(label_list: Sequence[str] = HEAD_A_4WAY_LABELS) -> Tuple[List[int], List[int]]:
    """Return (attention_ids, vigilance_ids) into label_list."""
    att = [label_list.index(n) for n in ATTENTION_LABELS if n in label_list]
    vig = [label_list.index(n) for n in VIGILANCE_LABELS if n in label_list]
    return att, vig


def find_weights() -> Path:
    """Locate CBraMod weights relative to this repo (same root as training scripts)."""
    from src.safe_paths import first_existing_cbramod_weights

    return first_existing_cbramod_weights(Path(__file__).resolve().parents[1])


def cap_balanced(
    X: np.ndarray,
    y: np.ndarray,
    max_n: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample a (typically already balanced) set to at most ``max_n`` rows."""
    if max_n <= 0 or len(y) <= max_n:
        return X, y
    per = max_n // 2
    picks = []
    for c in (0, 1):
        cand = np.nonzero(y == c)[0]
        picks.append(rng.choice(cand, size=min(per, len(cand)), replace=False))
    sel = np.concatenate(picks)
    rng.shuffle(sel)
    return X[sel], y[sel]


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
            cand = np.nonzero(p["y"] == c)[0]
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


def train_eval_fold_emb(
    emb_tr: torch.Tensor,
    y_tr: np.ndarray,
    emb_te: torch.Tensor,
    y_te: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    *,
    epochs: int = 5,
    batch: int = 32,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """Train a binary Head A on balanced LOSO fold embeddings and evaluate."""
    from torch.utils.data import DataLoader, TensorDataset

    from src.metrics import per_class_report

    idx0 = np.nonzero(y_tr == 0)[0]
    idx1 = np.nonzero(y_tr == 1)[0]
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
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(emb_b, torch.from_numpy(yb)), batch_size=batch, shuffle=True)
    for _ in range(epochs):
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


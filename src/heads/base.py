"""Shared building blocks for Head A multi-head (tiny Linear/MLP on frozen emb).

Encode contract (shared frozen encoder — REVE and/or CBraMod):

    emb = encoder.encode(windows)   # windows: [B, C, T] float; emb: [B, D]
    logits = head(emb)              # [B, n_classes] raw logits (no softmax)

Notes:
- Encoder stays frozen; only head parameters train.
- Each Head A-* has its own CE / label set — do not fuse into one exclusive softmax.
- Montages: pass muse4 or crown8 windows consistently; never mix in one batch.
- Default in_dim=200 matches CBraMod pooled emb; REVE may differ — set in_dim at construct.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class TinyLinearHead(nn.Module):
    """Dropout + Linear → logits [B, n_classes]."""

    def __init__(self, in_dim: int = 200, n_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_classes = int(n_classes)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.in_dim, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyMLPHead(nn.Module):
    """Linear → GELU → Dropout → Linear → logits [B, n_classes]."""

    def __init__(
        self,
        in_dim: int = 200,
        n_classes: int = 2,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.n_classes = int(n_classes)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def labels_to_ids(labels: Sequence[str], label_list: Sequence[str]) -> np.ndarray:
    table = {n: i for i, n in enumerate(label_list)}
    return np.asarray([table[str(v)] for v in labels], dtype=np.int64)


def class_weights_from_y(y: np.ndarray, n_classes: Optional[int] = None) -> torch.Tensor:
    if n_classes is None:
        n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)


def undersample_balanced(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    m = int(counts.min())
    idxs: List[np.ndarray] = []
    for c in classes:
        cand = np.where(y == c)[0]
        pick = rng.choice(cand, size=m, replace=False)
        idxs.append(pick)
    idxs_arr = np.concatenate(idxs)
    rng.shuffle(idxs_arr)
    return X[idxs_arr], y[idxs_arr]

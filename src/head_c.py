"""Tiny Head C classifiers on frozen encoder embeddings (sleep stage smoke).

Head C is NOT a priority product — smoke / pipeline verification only.
Labels come from Sleep-EDF hypnogram → stage_coarse (or stage_raw 5-way).
Muse-proxy transfer for deep/REM is expected to be weak.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

# Preferred smoke task: drop unknown
HEAD_C_COARSE_LABELS: List[str] = ["wake", "light", "deep", "rem"]

# Optional 5-way raw (N3 merges stage 3+4)
HEAD_C_RAW5_LABELS: List[str] = ["W", "N1", "N2", "N3", "REM"]

STAGE_RAW_TO_RAW5 = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


class HeadCLinear(nn.Module):
    def __init__(self, in_dim: int = 200, n_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_classes = int(n_classes)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeadCMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 200,
        n_classes: int = 4,
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


def stage_raw_to_raw5(stage: str) -> Optional[str]:
    return STAGE_RAW_TO_RAW5.get(stage.strip())


def undersample_balanced(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(y, return_counts=True)
    m = int(counts.min())
    idxs = []
    for c in classes:
        cand = np.where(y == c)[0]
        pick = rng.choice(cand, size=m, replace=False)
        idxs.append(pick)
    idxs = np.concatenate(idxs)
    rng.shuffle(idxs)
    return X[idxs], y[idxs]


def class_weights_from_y(y: np.ndarray, n_classes: Optional[int] = None) -> torch.Tensor:
    if n_classes is None:
        n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w, dtype=torch.float32)


def cap_per_class(
    X: np.ndarray,
    y: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
    starts: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Cap each class to at most per_class examples (no undersample of minorities)."""
    idxs: List[np.ndarray] = []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        n = min(int(per_class), len(cand))
        if n == 0:
            continue
        idxs.append(rng.choice(cand, size=n, replace=False))
    if not idxs:
        empty_st = np.zeros(0, dtype=np.int64) if starts is not None else None
        return X[:0], y[:0], empty_st
    pick = np.concatenate(idxs)
    rng.shuffle(pick)
    st_out = starts[pick] if starts is not None else None
    return X[pick], y[pick], st_out

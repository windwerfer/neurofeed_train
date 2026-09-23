"""Tiny Head A classifiers on frozen encoder embeddings.

LEGACY 4-way + binary helpers (masked CE). Product architecture is
**multi-head** — see ``src.heads`` and ``docs/head_a_multihead.md``
(decision 2026-09-07): A-vig / A-med / A-eng separate Linear heads.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

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
) -> tuple[np.ndarray, np.ndarray]:
    """Undersample majority classes to min class count (smoke clarity)."""
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

"""Head A-med depth: meditation_depth_low vs meditation_depth_high (in-session UX).

Binary remap of Brandmeyer & Delorme ds001787 probe Q1 (0–3 meditation depth).
Not rest vs meditation; not Q1-vs-Q2 concentration/MW.
See docs/head_a_med_depth_smoke.md, docs/head_a_multihead.md.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .base import TinyLinearHead, TinyMLPHead, labels_to_ids

HEAD_A_DEPTH_LABELS: List[str] = [
    "meditation_depth_low",
    "meditation_depth_high",
]

# Extreme filter on Q1 (0–3): low ≤1, high ≥2 (integer scale → no mid band).
Q1_LOW_MAX = 1
Q1_HIGH_MIN = 2


def q1_to_depth_label(q1: Optional[int]) -> Optional[str]:
    """Map probe Q1 rating → depth class; None if missing or outside extremes rule."""
    if q1 is None:
        return None
    q = int(q1)
    if q <= Q1_LOW_MAX:
        return "meditation_depth_low"
    if q >= Q1_HIGH_MIN:
        return "meditation_depth_high"
    return None


class HeadADepthLinear(TinyLinearHead):
    """Logits [B, 2] — meditation_depth_low / meditation_depth_high."""

    def __init__(self, in_dim: int = 200, dropout: float = 0.1):
        super().__init__(in_dim=in_dim, n_classes=2, dropout=dropout)
        self.label_list: List[str] = list(HEAD_A_DEPTH_LABELS)


class HeadADepthMLP(TinyMLPHead):
    def __init__(
        self,
        in_dim: int = 200,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_dim=in_dim, n_classes=2, hidden=hidden, dropout=dropout
        )
        self.label_list: List[str] = list(HEAD_A_DEPTH_LABELS)


def depth_labels_to_ids(labels: Sequence[str]):
    return labels_to_ids(labels, HEAD_A_DEPTH_LABELS)

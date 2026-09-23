"""Head A-med: rest vs meditation (beginners UX; ship second).

Logits [B, 2]. Not mind_wandering / concentration.
See docs/head_a_multihead.md.
"""

from __future__ import annotations

from typing import List, Sequence

from .base import TinyLinearHead, TinyMLPHead, labels_to_ids

HEAD_A_MED_LABELS: List[str] = ["rest", "meditation"]


class HeadAMedLinear(TinyLinearHead):
    """Logits [B, 2] — rest / meditation."""

    def __init__(self, in_dim: int = 200, dropout: float = 0.1):
        super().__init__(in_dim=in_dim, n_classes=2, dropout=dropout)
        self.label_list: List[str] = list(HEAD_A_MED_LABELS)


class HeadAMedMLP(TinyMLPHead):
    def __init__(
        self,
        in_dim: int = 200,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_dim=in_dim, n_classes=2, hidden=hidden, dropout=dropout
        )
        self.label_list: List[str] = list(HEAD_A_MED_LABELS)


def med_labels_to_ids(labels: Sequence[str]):
    return labels_to_ids(labels, HEAD_A_MED_LABELS)

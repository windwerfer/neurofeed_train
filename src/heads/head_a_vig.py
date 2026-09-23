"""Head A-vig: vigilance / sleep-onset (ship path).

Labels: drowsy / hypnagogic (+ optional awake).
Train/eval: Sleep-EDF vigilance_sleep_edf (muse4 proxy).
See docs/head_a_multihead.md.
"""

from __future__ import annotations

from typing import List, Sequence

from .base import TinyLinearHead, TinyMLPHead, labels_to_ids

HEAD_A_VIG_LABELS_2: List[str] = ["drowsy", "hypnagogic"]
HEAD_A_VIG_LABELS_3: List[str] = ["awake", "drowsy", "hypnagogic"]

# Default ship = 2-way
HEAD_A_VIG_LABELS: List[str] = list(HEAD_A_VIG_LABELS_2)


class HeadAVigLinear(TinyLinearHead):
    """Logits [B, 2] or [B, 3] depending on n_classes / include_awake."""

    def __init__(
        self,
        in_dim: int = 200,
        include_awake: bool = False,
        dropout: float = 0.1,
        n_classes: int | None = None,
    ):
        if n_classes is None:
            n_classes = 3 if include_awake else 2
        super().__init__(in_dim=in_dim, n_classes=n_classes, dropout=dropout)
        self.include_awake = bool(include_awake)
        self.label_list: List[str] = (
            list(HEAD_A_VIG_LABELS_3) if self.n_classes == 3 else list(HEAD_A_VIG_LABELS_2)
        )


class HeadAVigMLP(TinyMLPHead):
    def __init__(
        self,
        in_dim: int = 200,
        include_awake: bool = False,
        hidden: int = 64,
        dropout: float = 0.1,
        n_classes: int | None = None,
    ):
        if n_classes is None:
            n_classes = 3 if include_awake else 2
        super().__init__(
            in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout
        )
        self.include_awake = bool(include_awake)
        self.label_list: List[str] = (
            list(HEAD_A_VIG_LABELS_3) if self.n_classes == 3 else list(HEAD_A_VIG_LABELS_2)
        )


def vig_labels_to_ids(
    labels: Sequence[str],
    include_awake: bool = False,
):
    lab = HEAD_A_VIG_LABELS_3 if include_awake else HEAD_A_VIG_LABELS_2
    return labels_to_ids(labels, lab)

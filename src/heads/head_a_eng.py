"""Head A-eng: engagement / cognitive load.

Corpus expanded 2026-09-08 toward ~120 unique subjects — see
docs/head_a_eng_corpus_expansion.md. Smoke: docs/head_a_eng_smoke.md.
Fit/misfit evaluated on CBraMod+REVE — ship_candidate False (order/domain confounds). STEW raw still
IEEE-gated; HF processed CC-BY-4.0 used. Not a free byproduct of A-vig/A-med.

Logits [B, 2].
"""

from __future__ import annotations

from typing import List, Sequence

from .base import TinyLinearHead, TinyMLPHead, labels_to_ids

HEAD_A_ENG_LABELS: List[str] = ["low_engagement", "high_engagement"]

# Honest status flag for trainers / exporters
ENG_TRAIN_STATUS = "trained_dual_misfit_ship_false"  # ~133 unique persons; see docs/head_a_eng_corpus_expansion.md
ENG_CORPORA = ("ds007169", "ds007262", "eegmat", "stew", "ds007554")  # local windows ready
ENG_DEFERRED_CORPORA = ENG_CORPORA  # back-compat alias


class HeadAEngLinear(TinyLinearHead):
    """Logits [B, 2] — interface ready; do not claim trained weights yet."""

    def __init__(self, in_dim: int = 200, dropout: float = 0.1):
        super().__init__(in_dim=in_dim, n_classes=2, dropout=dropout)
        self.label_list: List[str] = list(HEAD_A_ENG_LABELS)
        self.train_status = ENG_TRAIN_STATUS


class HeadAEngMLP(TinyMLPHead):
    def __init__(
        self,
        in_dim: int = 200,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_dim=in_dim, n_classes=2, hidden=hidden, dropout=dropout
        )
        self.label_list: List[str] = list(HEAD_A_ENG_LABELS)
        self.train_status = ENG_TRAIN_STATUS


def eng_labels_to_ids(labels: Sequence[str]):
    return labels_to_ids(labels, HEAD_A_ENG_LABELS)

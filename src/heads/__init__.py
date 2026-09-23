"""Head A multi-head package (2026-09-07 lock).

Shared frozen encoder → separate tiny heads:
  - head_a_vig: drowsy / hypnagogic (+ optional awake) — ship first
  - head_a_med: rest / meditation — ship second (see also depth operationalization)
  - head_a_depth: meditation_depth_low / high (Q1 extremes; preferred A-med live UX)
  - head_a_eng: low/high engagement — scaffold; deferred STEW/ds007169 ingest

Encode API: emb = encoder.encode(windows); logits = head(emb).
Docs: docs/head_a_multihead.md
"""

from .base import TinyLinearHead, TinyMLPHead, labels_to_ids
from .head_a_vig import (
    HEAD_A_VIG_LABELS,
    HEAD_A_VIG_LABELS_2,
    HEAD_A_VIG_LABELS_3,
    HeadAVigLinear,
    HeadAVigMLP,
)
from .head_a_med import HEAD_A_MED_LABELS, HeadAMedLinear, HeadAMedMLP
from .head_a_depth import (
    HEAD_A_DEPTH_LABELS,
    Q1_HIGH_MIN,
    Q1_LOW_MAX,
    HeadADepthLinear,
    HeadADepthMLP,
    q1_to_depth_label,
)
from .head_a_eng import (
    ENG_CORPORA,
    ENG_DEFERRED_CORPORA,
    ENG_TRAIN_STATUS,
    HEAD_A_ENG_LABELS,
    HeadAEngLinear,
    HeadAEngMLP,
)

__all__ = [
    "TinyLinearHead",
    "TinyMLPHead",
    "labels_to_ids",
    "HEAD_A_VIG_LABELS",
    "HEAD_A_VIG_LABELS_2",
    "HEAD_A_VIG_LABELS_3",
    "HeadAVigLinear",
    "HeadAVigMLP",
    "HEAD_A_MED_LABELS",
    "HeadAMedLinear",
    "HeadAMedMLP",
    "HEAD_A_DEPTH_LABELS",
    "Q1_LOW_MAX",
    "Q1_HIGH_MIN",
    "HeadADepthLinear",
    "HeadADepthMLP",
    "q1_to_depth_label",
    "HEAD_A_ENG_LABELS",
    "ENG_TRAIN_STATUS",
    "ENG_CORPORA",
    "ENG_DEFERRED_CORPORA",
    "HeadAEngLinear",
    "HeadAEngMLP",
]

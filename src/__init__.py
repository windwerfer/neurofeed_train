"""Muse EEG heads helpers (Kaggle dataset windwerfer/muse-eeg-heads-src)."""

from .channel_map import MUSE_CHANNELS
from .metrics import HEAD_A_LABELS, HEAD_B_LABELS

__all__ = [
    "MUSE_CHANNELS",
    "HEAD_A_LABELS",
    "HEAD_B_LABELS",
]

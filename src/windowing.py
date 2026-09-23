"""Sliding-window utilities for Muse EEG tensors."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np


def sliding_windows(
    data: np.ndarray,
    sample_rate: float,
    window_sec: float = 2.0,
    hop_sec: float = 0.5,
    axis: int = -1,
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Yield (start_sample, window) along time axis.

    data: array with time on `axis` (default last), shape (..., n_times).
    """
    n_times = data.shape[axis]
    win = int(round(window_sec * sample_rate))
    hop = int(round(hop_sec * sample_rate))
    if win <= 0 or hop <= 0:
        raise ValueError("window_sec and hop_sec must yield positive sample counts")
    start = 0
    while start + win <= n_times:
        sl = [slice(None)] * data.ndim
        sl[axis] = slice(start, start + win)
        yield start, data[tuple(sl)]
        start += hop


def stack_windows(
    data: np.ndarray,
    sample_rate: float,
    window_sec: float = 2.0,
    hop_sec: float = 0.5,
    max_windows: Optional[int] = None,
) -> np.ndarray:
    """Return array of shape (n_windows, *non_time_dims, win_samples)."""
    windows = []
    for i, (_, w) in enumerate(
        sliding_windows(data, sample_rate, window_sec, hop_sec, axis=-1)
    ):
        windows.append(w)
        if max_windows is not None and i + 1 >= max_windows:
            break
    if not windows:
        raise ValueError("No windows fit; check duration vs window_sec")
    return np.stack(windows, axis=0)


def label_per_window(
    event_samples: np.ndarray,
    event_labels: np.ndarray,
    n_times: int,
    sample_rate: float,
    window_sec: float = 2.0,
    hop_sec: float = 0.5,
    default_label: Optional[str] = None,
) -> list:
    """
    Assign each window the label of the nearest event inside the window,
    else default_label (or None).
    """
    win = int(round(window_sec * sample_rate))
    hop = int(round(hop_sec * sample_rate))
    labels = []
    start = 0
    events = np.asarray(event_samples)
    labs = np.asarray(event_labels)
    while start + win <= n_times:
        end = start + win
        inside = (events >= start) & (events < end)
        if inside.any():
            # pick earliest event in window
            idxs = np.where(inside)[0]
            labels.append(labs[idxs[0]].item() if hasattr(labs[idxs[0]], "item") else labs[idxs[0]])
        else:
            labels.append(default_label)
        start += hop
    return labels

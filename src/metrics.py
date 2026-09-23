"""Classification metrics for Head A / Head B."""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np


def _as_ids(y: Sequence, label_list: Sequence[str]) -> np.ndarray:
    if len(y) == 0:
        return np.array([], dtype=int)
    if isinstance(y[0], (int, np.integer)):
        return np.asarray(y, dtype=int)
    table = {n: i for i, n in enumerate(label_list)}
    return np.asarray([table[v] for v in y], dtype=int)


def confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    label_list: Sequence[str],
) -> np.ndarray:
    n = len(label_list)
    yt = _as_ids(y_true, label_list)
    yp = _as_ids(y_pred, label_list)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(yt, yp):
        if 0 <= t < n and 0 <= p < n:
            cm[t, p] += 1
    return cm


def macro_f1(
    y_true: Sequence,
    y_pred: Sequence,
    label_list: Sequence[str],
) -> float:
    cm = confusion_matrix(y_true, y_pred, label_list)
    f1s = []
    for i in range(len(label_list)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_report(
    y_true: Sequence,
    y_pred: Sequence,
    label_list: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    cm = confusion_matrix(y_true, y_pred, label_list)
    report: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(label_list):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        support = float(cm[i, :].sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        report[name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": support,
        }
    report["macro_f1"] = {"f1": macro_f1(y_true, y_pred, label_list)}  # type: ignore
    return report


HEAD_A_LABELS: List[str] = [
    "concentration",
    "mind_wandering",
    "drowsy",
    "hypnagogic",
]

HEAD_B_LABELS: List[str] = [
    "blink",
    "double_blink",
    "jaw",
    "double_jaw",
    "clean",
]

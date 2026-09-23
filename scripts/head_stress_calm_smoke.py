#!/usr/bin/env python3
"""Small stress vs calm CBraMod chance-check smoke (CPU, muse4 proxy).

Gated behind order-aware A-eng PASS (docs/head_a_eng_smoke.md).

Corpus: PhysioNet eegmat v1.0.0 — EEG During Mental Arithmetic Tasks
(ODC-By-1.0). Rest/background (_1) → calm; mental arithmetic (_2) → stress.
Paper frames serial subtraction as a standardized stress-inducing protocol;
this is NOT meditation calm vs psychosocial stress — honest adjacent UX proxy.

Size-bounded: n=8 subjects ≈ 40 MB (vs aborted PhysioNet MATB ~606 MB/sub).
Montage: muse4 proxy F7/F8/T3/T4 → AF7/AF8/TP9/TP10 (23-ch 10–20 professional).
No Kaggle.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from zoneinfo import ZoneInfo

import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder  # noqa: E402
from src.heads.base import (  # noqa: E402
    TinyLinearHead,
    class_weights_from_y,
    undersample_balanced,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report  # noqa: E402
from src.windowing import sliding_windows  # noqa: E402

SEED = 42
EPOCHS = 8
BATCH = 32
LR = 1e-3
VAL_FRAC = 0.15
WINDOW_SEC = 2.0
HOP_SEC = 1.0
TARGET_SR = 256.0
EDGE_TRIM_SEC = 2.0
MAX_WINDOWS_PER_COND = 50
SEGMENT_SEC = 60.0  # use last 60s rest + full arithmetic (~60s)
PAUSE_SEC = 1.0

SRC_CHS = ["F7", "F8", "T3", "T4"]
MUSE_CHS = ["AF7", "AF8", "TP9", "TP10"]
CH_MAP = dict(zip(SRC_CHS, MUSE_CHS))

LABELS = ["calm", "stress"]
# Subject00..07; holdout last
SUBJECT_IDS = [f"{i:02d}" for i in range(8)]
HOLDOUT_ID = "07"
TRAIN_IDS = [s for s in SUBJECT_IDS if s != HOLDOUT_ID]

DATASET_ID = "eegmat"
DATASET_NAME = (
    "PhysioNet EEG During Mental Arithmetic Tasks / eegmat "
    "(23-ch 10–20 professional)"
)
DATASET_DOI = "doi:10.13026/C2JQ1P"
LICENSE_SPDX = "ODC-By-1.0"
S3_ROOT = "https://physionet-open.s3.amazonaws.com/eegmat/1.0.0"
HTTP_ROOT = "https://physionet.org/files/eegmat/1.0.0"

CACHE = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data" / DATASET_ID
OUT_DIR = ROOT / "exports" / "head_stress_calm_smoke"
WINDOWS_DIR = OUT_DIR / "windows"
DOC_PATH = ROOT / "docs" / "head_stress_calm_smoke.md"
VERSION_NOTE = "stress_calm_eegmat_n8_2026-09-08"

SHIP_F1_MIN = 0.55
CHANCE_F1 = 0.5


class HeadStressCalmLinear(TinyLinearHead):
    def __init__(self, in_dim: int = 200, dropout: float = 0.1):
        super().__init__(in_dim=in_dim, n_classes=2, dropout=dropout)
        self.label_list = list(LABELS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pause(msg: str) -> None:
    print(f"[pace] {msg} (sleep {PAUSE_SEC}s)", flush=True)
    time.sleep(PAUSE_SEC)


def download(url: str, dest: Path, expected_size: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"[skip] {dest.name} ({dest.stat().st_size} B)", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get] {dest.name} ← {url}", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    if expected_size is not None and tmp.stat().st_size != expected_size:
        raise RuntimeError(
            f"size mismatch {dest.name}: got {tmp.stat().st_size} want {expected_size}"
        )
    tmp.replace(dest)


def ensure_meta() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in ("README.txt", "subject-info.csv", "RECORDS", "SHA256SUMS.txt"):
        dest = CACHE / name
        if not dest.exists():
            pause(f"meta {name}")
            # prefer CDN/S3 then physionet
            try:
                download(f"{S3_ROOT}/{name}", dest)
            except Exception:
                download(f"{HTTP_ROOT}/{name}", dest)


def ensure_subject(sid: str) -> Dict[str, Path]:
    paths = {}
    for cond, label in (("1", "calm"), ("2", "stress")):
        fname = f"Subject{sid}_{cond}.edf"
        dest = CACHE / fname
        if not dest.exists():
            pause(f"dl {fname}")
            try:
                download(f"{S3_ROOT}/{fname}", dest)
            except Exception:
                download(f"{HTTP_ROOT}/{fname}", dest)
        paths[label] = dest
    return paths


def _norm_ch(name: str) -> str:
    n = name.upper().replace("EEG", "").strip()
    n = n.replace(" ", "")
    return n


def load_muse4_segment(edf: Path, which: str) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Load muse4 proxy; take last SEGMENT_SEC for rest, full clip (≤SEGMENT_SEC) for stress."""
    raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
    available = {_norm_ch(c): c for c in raw.ch_names}
    picks = []
    for src in SRC_CHS:
        if src not in available:
            raise KeyError(f"{edf.name}: missing {src}; have {list(available)}")
        picks.append(available[src])
    raw.pick(picks)
    rename = {available[s]: CH_MAP[s] for s in SRC_CHS}
    raw.rename_channels(rename)
    raw.reorder_channels(MUSE_CHS)
    if abs(raw.info["sfreq"] - TARGET_SR) > 1e-3:
        raw.resample(TARGET_SR, verbose="ERROR")
    data = raw.get_data().astype(np.float64)  # SI volts
    # physical µV-scale; still z-score for CBraMod transfer consistency
    raw_mean = data.mean(axis=1, keepdims=True)
    raw_std = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
    data = ((data - raw_mean) / raw_std).astype(np.float32)
    data = np.clip(data, -15.0, 15.0)

    sfreq = float(raw.info["sfreq"])
    n = data.shape[1]
    need = int(round(SEGMENT_SEC * sfreq))
    if which == "calm":
        # last 60 s of background rest
        if n > need:
            data = data[:, n - need :]
            t0 = (n - need) / sfreq
        else:
            t0 = 0.0
    else:
        # arithmetic already ~60 s; truncate to SEGMENT_SEC if longer
        if n > need:
            data = data[:, :need]
        t0 = 0.0

    meta = {
        "sfreq": sfreq,
        "n_times": int(data.shape[1]),
        "duration_sec": float(data.shape[1] / sfreq),
        "segment_start_sec_in_file": float(t0),
        "ch_names": list(MUSE_CHS),
        "src_channels": list(SRC_CHS),
        "channel_map": dict(CH_MAP),
        "edf": str(edf.relative_to(ROOT)),
        "scale_note": "per_channel_zscore_clip15_physical_volts",
        "which": which,
    }
    return data, sfreq, meta


def window_condition(
    data: np.ndarray, sfreq: float, label: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    trim = int(round(EDGE_TRIM_SEC * sfreq))
    if data.shape[1] <= 2 * trim + int(round(WINDOW_SEC * sfreq)):
        trim = max(0, int(data.shape[1] * 0.05))
    cropped = data[:, trim : data.shape[1] - trim if trim else data.shape[1]]
    wins, starts = [], []
    for start, w in sliding_windows(cropped, sfreq, WINDOW_SEC, HOP_SEC, axis=-1):
        wins.append(w)
        starts.append(trim + start)
    if not wins:
        raise ValueError(f"no windows for {label}")
    X = np.stack(wins, axis=0).astype(np.float32)
    starts_arr = np.asarray(starts, dtype=np.int64)
    if len(X) > MAX_WINDOWS_PER_COND:
        idx = np.unique(np.linspace(0, len(X) - 1, MAX_WINDOWS_PER_COND).astype(int))
        X = X[idx]
        starts_arr = starts_arr[idx]
    y = np.full(len(X), LABELS.index(label), dtype=np.int64)
    return X, y, starts_arr


def build_subject(sid: str, paths: Dict[str, Path]) -> Dict[str, Any]:
    Xs, ys, sts, cond_meta = [], [], [], {}
    for label in LABELS:
        data, sfreq, meta = load_muse4_segment(paths[label], which=label)
        X, y, starts = window_condition(data, sfreq, label)
        Xs.append(X)
        ys.append(y)
        sts.append(starts)
        cond_meta[label] = {**meta, "n_windows": int(len(y))}
        print(
            f"  Subject{sid} {label}: dur={meta['duration_sec']:.1f}s windows={len(y)}",
            flush=True,
        )
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    starts = np.concatenate(sts)
    counts = {LABELS[i]: int((y == i).sum()) for i in range(2)}
    pack = {
        "subject": f"Subject{sid}",
        "subject_id": sid,
        "X": X,
        "y": y,
        "starts": starts,
        "counts": counts,
        "conditions": cond_meta,
        "montage": "muse4",
        "channels": list(MUSE_CHS),
        "channel_map": dict(CH_MAP),
        "label_mapping": {
            "rest_background_1": "calm",
            "mental_arithmetic_2": "stress",
        },
        "protocol_note": (
            "Rest (_1) always precedes arithmetic (_2); duration-matched ~60s "
            "segments (last 60s rest). Residual order confound possible."
        ),
    }
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    npz = WINDOWS_DIR / f"Subject{sid}_stresscalm_windows.npz"
    man = WINDOWS_DIR / f"Subject{sid}_stresscalm_manifest.json"
    np.savez_compressed(
        npz, X=X, y=y, starts=starts, label_names=np.asarray(LABELS)
    )
    man.write_text(
        json.dumps({k: v for k, v in pack.items() if k not in ("X", "y", "starts")}, indent=2)
    )
    pack["npz_path"] = npz
    pack["npz_sha256"] = sha256(npz)
    return pack


def find_weights() -> Path:
    p = ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def encode_all(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(X) == 0:
        return torch.zeros((0, 200), dtype=torch.float32)
    Xt = torch.from_numpy(X.astype(np.float32))
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            emb_list.append(encoder(Xt[i : i + BATCH].to(device)).cpu())
    return torch.cat(emb_list, dim=0)


def balance_train(
    packs: List[Dict[str, Any]], rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    Xs, ys = [], []
    detail: Dict[str, Any] = {}
    for p in packs:
        present = sorted(set(int(c) for c in np.unique(p["y"])))
        if len(present) < 2:
            detail[p["subject"]] = {"skipped": True, "counts": p["counts"]}
            continue
        Xb, yb = undersample_balanced(p["X"], p["y"], rng)
        Xs.append(Xb)
        ys.append(yb)
        detail[p["subject"]] = {
            "skipped": False,
            "n_total": int(len(yb)),
            "per_class": {LABELS[i]: int((yb == i).sum()) for i in present},
            "counts_full": p["counts"],
        }
    if not Xs:
        raise RuntimeError("no train subjects with both classes")
    return np.concatenate(Xs), np.concatenate(ys), detail


def stride4(X: np.ndarray, y: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(starts)
    idx = order[::4]
    return X[idx], y[idx]


def eval_split(head: nn.Module, emb: torch.Tensor, y: np.ndarray, name: str) -> Dict[str, Any]:
    head.eval()
    if len(y) == 0:
        return {"name": name, "n": 0, "acc": None, "macro_f1": None, "note": "empty"}
    with torch.no_grad():
        pred = head(emb).argmax(dim=-1).numpy()
    acc = float((pred == y).mean())
    f1 = macro_f1(y.tolist(), pred.tolist(), LABELS)
    report = per_class_report(y.tolist(), pred.tolist(), LABELS)
    per_class = {k: v for k, v in report.items() if k != "macro_f1"}
    cm = confusion_matrix(y.tolist(), pred.tolist(), LABELS)
    return {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1": f1,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": list(LABELS),
        "counts_true": {LABELS[i]: int((y == i).sum()) for i in range(2)},
    }


def train_head(
    emb: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[HeadStressCalmLinear, List[Dict[str, Any]], Dict[str, Any]]:
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(VAL_FRAC * n)))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    emb_tr, y_tr = emb[tr_idx], y[tr_idx]
    emb_va, y_va = emb[va_idx], y[va_idx]

    head = HeadStressCalmLinear(in_dim=emb.shape[-1]).to(device)
    w = class_weights_from_y(y_tr, n_classes=2).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(emb_tr, torch.from_numpy(y_tr)),
        batch_size=BATCH,
        shuffle=True,
    )

    history: List[Dict[str, Any]] = []
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    for ep in range(EPOCHS):
        head.train()
        total, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n_seen += len(yb)
        head.eval()
        with torch.no_grad():
            pred_tr = head(emb_tr.to(device)).argmax(dim=-1).cpu().numpy()
            pred_va = head(emb_va.to(device)).argmax(dim=-1).cpu().numpy()
        row = {
            "epoch": ep + 1,
            "loss": total / max(n_seen, 1),
            "train_acc": float((pred_tr == y_tr).mean()),
            "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), LABELS),
            "val_acc": float((pred_va == y_va).mean()),
            "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), LABELS),
        }
        history.append(row)
        print(row, flush=True)
        if row["val_macro_f1"] >= best_val_f1:
            best_val_f1 = float(row["val_macro_f1"])
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
        print(f"restored best val epoch={best_epoch} val_macro_f1={best_val_f1:.4f}", flush=True)

    head.eval()
    with torch.no_grad():
        pred_tr = head(emb_tr.to(device)).argmax(dim=-1).cpu().numpy()
        pred_va = head(emb_va.to(device)).argmax(dim=-1).cpu().numpy()
    splits = {
        "train_n": int(len(y_tr)),
        "val_n": int(len(y_va)),
        "best_epoch": best_epoch,
        "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), LABELS),
        "val_acc": float((pred_va == y_va).mean()),
        "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), LABELS),
        "train_acc": float((pred_tr == y_tr).mean()),
    }
    return head, history, splits


def recommendation(holdout_bal_f1: float) -> Dict[str, Any]:
    delta = holdout_bal_f1 - CHANCE_F1
    if holdout_bal_f1 >= 0.60:
        rec = "keep_exploring"
        note = (
            "Holdout bal macro-F1 clearly above chance on rest vs arithmetic "
            "(stress-induction proxy). Residual: rest always precedes task; "
            "not meditation calm. Do NOT ship public stress/calm yet — expand n, "
            "order-aware / counterbalanced stress corpora (SAM-40 if size OK later)."
        )
    elif holdout_bal_f1 >= 0.55:
        rec = "keep_exploring_weak"
        note = (
            "Weak lift vs chance — continue research / personal Muse cal; do not ship."
        )
    elif holdout_bal_f1 >= 0.45:
        rec = "personal_cal_only"
        note = (
            "Near chance on this smoke — drop as public stress/calm head; "
            "personal Muse cal only if product wants stress UX."
        )
    else:
        rec = "drop_public"
        note = "At/below chance — drop public stress/calm for now."
    return {
        "recommendation": rec,
        "ship_candidate": False,
        "holdout_balanced_macro_f1": holdout_bal_f1,
        "chance_macro_f1": CHANCE_F1,
        "delta_vs_chance": delta,
        "note": note,
        "order_confound_rest_before_task": True,
    }


def write_doc(summary: Dict[str, Any]) -> None:
    h = summary["headline"]
    rec = summary["recommendation"]
    lines = [
        "# Stress/calm smoke (CBraMod, CPU)",
        "",
        f"**Completed (Asia/Bangkok):** {summary['completed_bkk']}",
        f"**UTC:** {summary['completed_utc']}",
        f"**Version:** `{VERSION_NOTE}`",
        "",
        "## Gate",
        "",
        "Ran only after **order-aware A-eng PASS** "
        f"(primary mid 2vs3 holdout bal F1="
        f"{summary.get('eng_orderaware_primary_f1', 'see eng doc')} > 0.55). "
        "See `docs/head_a_eng_smoke.md` § Order-aware re-smoke.",
        "",
        "## Goal",
        "",
        "Frozen **CBraMod** chance-check for **stress vs calm** "
        "(`calm` vs `stress`) on a **size-bounded** Muse-mappable open corpus.",
        "",
        "## Data pick",
        "",
        "| Candidate | License | Why not / why used |",
        "|-----------|---------|-------------------|",
        "| PhysioNet MATB `neuro-stress-resilience-hci` | ODbL | "
        "~606 MB/subject; AF7/AF8 fNIRS — aborted earlier |",
        "| SAM-40 Emotiv Flex | CC-BY-4.0 | Single 760 MB RAR — over smoke bound |",
        "| alkabbany Muse-S | CC-BY-4.0 | Google Drive; n=5 |",
        "| STEW | CC-BY-4.0 | IEEE login for raw |",
        f"| **{DATASET_NAME}** | **{LICENSE_SPDX}** | **USED** — "
        "~5 MB/subject; rest vs arithmetic |",
        "",
        f"**Dataset bracket tag:** `{DATASET_NAME}`",
        "",
        "## Label mapping (honest)",
        "",
        "| Source | Label | Notes |",
        "|--------|-------|-------|",
        "| `SubjectXX_1.edf` background rest | `calm` | Last 60 s of rest file |",
        "| `SubjectXX_2.edf` mental arithmetic | `stress` | ~60 s; standardized "
        "stress-induction protocol (serial subtraction) |",
        "",
        "Product UX caveat: rest≠meditation calm; arithmetic stress≠psychosocial stress.",
        "",
        "## Montage",
        "",
        "- **muse4 only** (never mixed with crown8)",
        "- Proxy: `F7→AF7`, `F8→AF8`, `T3→TP9`, `T4→TP10`",
        "- Resample 500 Hz → 256 Hz; 2 s / 1 s hop; edge trim 2 s; "
        "cap 50 windows/condition",
        "- Per-channel z-score + clip±15 before encode",
        "",
        "## Split",
        "",
        f"- Train: Subject{', Subject'.join(TRAIN_IDS)} (n={len(TRAIN_IDS)})",
        f"- Holdout: Subject{HOLDOUT_ID}",
        "- Per-subject undersample_balanced; 15% val; best-by-val",
        "",
        "## Metrics (headline)",
        "",
        "| Split | n | accuracy | macro-F1 |",
        "|-------|--:|---------:|---------:|",
        f"| Train (balanced) | {h['train_n']} | {h['train_acc']:.3f} | {h['train_macro_f1']:.3f} |",
        f"| Val | {h['val_n']} | {h['val_acc']:.3f} | {h['val_macro_f1']:.3f} |",
        f"| Holdout full | {h['holdout_full_n']} | {h['holdout_full_acc']:.3f} | {h['holdout_full_macro_f1']:.3f} |",
        f"| Holdout balanced | {h['holdout_balanced_n']} | {h['holdout_balanced_acc']:.3f} | {h['holdout_balanced_macro_f1']:.3f} |",
        f"| Holdout stride×4 | {h['holdout_stride4_n']} | {h['holdout_stride4_acc']:.3f} | {h['holdout_stride4_macro_f1']:.3f} |",
        "",
        f"**Chance macro-F1 (balanced binary):** `{CHANCE_F1}`",
        f"**Δ holdout-balanced vs chance:** `{h['holdout_balanced_macro_f1'] - CHANCE_F1:+.3f}`",
        f"**ship_candidate:** `{summary['ship_candidate']}`",
        "",
        "## Confounds",
        "",
        "1. **Order:** rest always before arithmetic — time/drift may contribute.",
        "2. Cognitive load / stress-induction ≠ meditation calm UX.",
        "3. Single holdout subject; overlapping windows; muse4 proxy.",
        "",
        "## Recommendation",
        "",
        f"- **Code:** `{rec['recommendation']}`",
        f"- **Note:** {rec['note']}",
        "",
        "## Provenance",
        "",
        f"- Dataset: `{DATASET_NAME}` — SPDX `{LICENSE_SPDX}` — `{DATASET_DOI}`",
        "- Encoder: CBraMod Apache-2.0 `pretrained_weights.pth`",
        "- Montage: **muse4** only",
        "- Script: `scripts/head_stress_calm_smoke.py`",
        "- Exports: `exports/head_stress_calm_smoke/`",
        "- No Kaggle push",
        "",
        "## Takeaway",
        "",
        summary["takeaway"],
        "",
    ]
    DOC_PATH.write_text("\n".join(lines))


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Stress/calm smoke (CBraMod CPU) ===", flush=True)
    print("dataset", DATASET_NAME, flush=True)
    print("subjects", SUBJECT_IDS, "holdout", HOLDOUT_ID, flush=True)

    # gate note from orderaware step summary if present
    oa = ROOT / "exports/head_a_eng_orderaware/step_summary.json"
    eng_primary = None
    if oa.exists():
        oa_sum = json.loads(oa.read_text())
        eng_primary = oa_sum.get("primary_holdout_balanced_macro_f1")
        if not oa_sum.get("proceed_stress"):
            raise RuntimeError("Order-aware eng did not pass — refusing stress download")
        print("gate OK eng primary F1", eng_primary, flush=True)

    ensure_meta()
    packs: List[Dict[str, Any]] = []
    for sid in SUBJECT_IDS:
        print(f"\n--- Subject{sid} ---", flush=True)
        paths = ensure_subject(sid)
        packs.append(build_subject(sid, paths))

    train_packs = [p for p in packs if p["subject_id"] in TRAIN_IDS]
    hold = next(p for p in packs if p["subject_id"] == HOLDOUT_ID)

    X_tr, y_tr, bal_detail = balance_train(train_packs, rng)
    print("train balanced", X_tr.shape, Counter(y_tr.tolist()), flush=True)
    print("holdout", hold["subject"], hold["X"].shape, hold["counts"], flush=True)

    weights = find_weights()
    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
    encoder.to(device)

    emb_tr = encode_all(encoder, X_tr, device)
    X_h_bal, y_h_bal = undersample_balanced(hold["X"], hold["y"], rng)
    X_h_s4, y_h_s4 = stride4(hold["X"], hold["y"], hold["starts"])
    emb_h_full = encode_all(encoder, hold["X"], device)
    emb_h_bal = encode_all(encoder, X_h_bal, device)
    emb_h_s4 = encode_all(encoder, X_h_s4, device)

    head, history, split_info = train_head(emb_tr, y_tr, device, rng)
    ev_tr = eval_split(head, emb_tr, y_tr, "train_balanced")
    ev_full = eval_split(head, emb_h_full, hold["y"], "holdout_full")
    ev_bal = eval_split(head, emb_h_bal, y_h_bal, "holdout_balanced")
    ev_s4 = eval_split(head, emb_h_s4, y_h_s4, "holdout_stride4")

    print("EVAL holdout bal", ev_bal, flush=True)

    pt_path = OUT_DIR / "head_stress_calm_eegmat.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": LABELS,
            "in_dim": int(emb_tr.shape[-1]),
            "montage": "muse4",
            "dataset": DATASET_NAME,
            "holdout": f"Subject{HOLDOUT_ID}",
            "best_epoch": split_info["best_epoch"],
        },
        pt_path,
    )

    rec = recommendation(float(ev_bal["macro_f1"]))
    bkk = ZoneInfo("Asia/Bangkok")
    now_utc = datetime.now(timezone.utc)
    now_bkk = now_utc.astimezone(bkk)

    headline = {
        "train_n": split_info["train_n"],
        "train_acc": split_info["train_acc"],
        "train_macro_f1": split_info["train_macro_f1"],
        "val_n": split_info["val_n"],
        "val_acc": split_info["val_acc"],
        "val_macro_f1": split_info["val_macro_f1"],
        "holdout_full_n": ev_full["n"],
        "holdout_full_acc": ev_full["acc"],
        "holdout_full_macro_f1": ev_full["macro_f1"],
        "holdout_balanced_n": ev_bal["n"],
        "holdout_balanced_acc": ev_bal["acc"],
        "holdout_balanced_macro_f1": ev_bal["macro_f1"],
        "holdout_stride4_n": ev_s4["n"],
        "holdout_stride4_acc": ev_s4["acc"],
        "holdout_stride4_macro_f1": ev_s4["macro_f1"],
    }

    takeaway = (
        f"Stress/calm smoke: {DATASET_NAME}; muse4 proxy F7/F8/T3/T4; "
        f"n={len(SUBJECT_IDS)} holdout Subject{HOLDOUT_ID}; "
        f"holdout bal macro-F1={ev_bal['macro_f1']:.3f} vs chance {CHANCE_F1} "
        f"(Δ{ev_bal['macro_f1']-CHANCE_F1:+.3f}); ship_candidate=False; "
        f"rec={rec['recommendation']}."
    )

    summary = {
        "step": "head_stress_calm_smoke",
        "version": VERSION_NOTE,
        "gated_by": "head_a_eng_orderaware PASS",
        "eng_orderaware_primary_f1": eng_primary,
        "dataset": DATASET_NAME,
        "dataset_id": DATASET_ID,
        "dataset_doi": DATASET_DOI,
        "license_spdx": LICENSE_SPDX,
        "montage": "muse4",
        "channel_map": dict(CH_MAP),
        "labels": LABELS,
        "label_mapping": {
            "rest_background_1": "calm",
            "mental_arithmetic_2": "stress",
        },
        "subjects": [f"Subject{s}" for s in SUBJECT_IDS],
        "train_subjects": [f"Subject{s}" for s in TRAIN_IDS],
        "holdout": f"Subject{HOLDOUT_ID}",
        "n_subjects": len(SUBJECT_IDS),
        "headline": headline,
        "evals": {
            "train_balanced": ev_tr,
            "holdout_full": ev_full,
            "holdout_balanced": ev_bal,
            "holdout_stride4": ev_s4,
        },
        "history": history,
        "split_info": split_info,
        "balance_detail": bal_detail,
        "ship_candidate": False,
        "recommendation": rec,
        "weights_sha256": sha256(weights),
        "head_pt": str(pt_path.relative_to(ROOT)),
        "head_pt_sha256": sha256(pt_path),
        "completed_utc": now_utc.isoformat(),
        "completed_bkk": now_bkk.strftime("%Y-%m-%d %H:%M:%S ICT"),
        "takeaway": takeaway,
        "no_kaggle_push": True,
        "confounds": [
            "Rest always precedes arithmetic — temporal confound possible",
            "Cognitive-load stress induction ≠ meditation calm UX",
            "Single holdout subject; overlapping windows",
        ],
    }

    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/head_stress_calm_smoke.py",
                "version": VERSION_NOTE,
                "dataset": DATASET_NAME,
                "subjects": SUBJECT_IDS,
                "holdout": HOLDOUT_ID,
                "seed": SEED,
                "epochs": EPOCHS,
                "montage": "muse4",
            },
            indent=2,
        )
    )
    (OUT_DIR / "step_summary.json").write_text(
        json.dumps(
            {
                "step": "head_stress_calm_smoke",
                "dataset": DATASET_NAME,
                "n_subjects": len(SUBJECT_IDS),
                "holdout": f"Subject{HOLDOUT_ID}",
                "holdout_balanced_macro_f1": ev_bal["macro_f1"],
                "chance_macro_f1": CHANCE_F1,
                "ship_candidate": False,
                "recommendation": rec["recommendation"],
            },
            indent=2,
        )
    )
    write_doc(summary)
    print("\n=== DONE stress/calm ===", flush=True)
    print(takeaway, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Order-aware A-eng re-smoke on OpenNeuro ds007169 (muse4 proxy, frozen CBraMod CPU).

Prior smoke (docs/head_a_eng_smoke.md) got holdout bal macro-F1 ≈ 0.969 on
1-back→low / 4-back→high, but protocol is strictly L1→L2→L3→L4 with no overlap —
temporal/drift confound risk.

This script breaks the shortcut with honest order-aware checks:

1. PRIMARY — mid difficulties: 2-back→low vs 3-back→high (adjacent blocks)
2. SECONDARY — edge-matched mid: late half of 2-back vs early half of 3-back
3. SECONDARY — edge-matched extremes: late half of 1-back vs early half of 4-back
   (still large gap via L2+L3; residual risk documented)
4. CONTROL — predict time bin (early vs late main-session) instead of load
5. CONTROL — within-block early vs late on pooled 2-back windows only

Pass criterion to proceed to stress/calm: PRIMARY holdout bal macro-F1 > 0.55.
If ~chance, STOP — do not download stress.

Montage: muse4 only. Same subjects/holdout/seed/windowing as prior smoke.
No Kaggle.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder  # noqa: E402
from src.heads.base import class_weights_from_y, undersample_balanced  # noqa: E402
from src.heads.head_a_eng import HEAD_A_ENG_LABELS, HeadAEngLinear  # noqa: E402
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
EDGE_TRIM_SEC = 5.0
MAX_WINDOWS_PER_BLOCK = 80
PAUSE_SEC = 0.5  # raw already cached; short pause if any meta fetch

SRC_CHS = ["F7", "F8", "T3", "T4"]
MUSE_CHS = ["AF7", "AF8", "TP9", "TP10"]
CH_MAP = dict(zip(SRC_CHS, MUSE_CHS))

SUBJECTS = [
    "sub-001",
    "sub-002",
    "sub-003",
    "sub-004",
    "sub-005",
    "sub-006",
    "sub-007",
    "sub-008",
]
HOLDOUT = "sub-008"
TRAIN_SUBJECTS = [s for s in SUBJECTS if s != HOLDOUT]

DATASET_ID = "ds007169"
DATASET_NAME = (
    "Multimodal Cognitive Workload n-back / ds007169 "
    "(19-ch 10–20 mobile EEG, professional)"
)
DATASET_DOI = "doi:10.18112/openneuro.ds007169.v1.0.5"
LICENSE_SPDX = "CC0-1.0"
S3_ROOT = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}"

CACHE = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data" / DATASET_ID
RAW = CACHE / "raw"
OUT_DIR = ROOT / "exports" / "head_a_eng_orderaware"
WINDOWS_DIR = OUT_DIR / "windows"
DOC_PATH = ROOT / "docs" / "head_a_eng_smoke.md"
VERSION_NOTE = "eng_orderaware_ds007169_n8_2026-09-08"

SHIP_F1_MIN = 0.55
CHANCE_F1 = 0.5
# time-bin labels reuse eng head interface (binary)
TIME_LABELS = ["early_session", "late_session"]
WITHIN_LABELS = ["early_within_block", "late_within_block"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pause(msg: str) -> None:
    print(f"[pace] {msg} (sleep {PAUSE_SEC}s)", flush=True)
    time.sleep(PAUSE_SEC)


def list_s3(prefix: str) -> List[Tuple[str, int]]:
    url = (
        f"https://s3.amazonaws.com/openneuro.org"
        f"?prefix={prefix}&delimiter=/&max-keys=1000"
    )
    data = urllib.request.urlopen(url, timeout=60).read()
    root = ET.fromstring(data)
    out: List[Tuple[str, int]] = []
    for el in root:
        if el.tag.split("}")[-1] != "Contents":
            continue
        key = next(c for c in el if c.tag.split("}")[-1] == "Key").text
        size = int(next(c for c in el if c.tag.split("}")[-1] == "Size").text)
        out.append((key, size))
    return out


def download(url: str, dest: Path, expected_size: Optional[int] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"[skip] {dest.name} ({dest.stat().st_size} B)", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get] {dest.name} ← {url[:110]}…", flush=True)
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
    for name in (
        "README",
        "participants.tsv",
        "dataset_description.json",
        "task-nback_eeg.json",
        "task-nback_events.json",
    ):
        dest = CACHE / name
        if not dest.exists():
            pause(f"meta {name}")
            download(f"{S3_ROOT}/{name}", dest)


def download_subject(sub: str) -> Dict[str, Path]:
    prefix = f"{DATASET_ID}/{sub}/eeg/"
    dest_dir = RAW / sub / "eeg"
    # Prefer already-cached files (no network if complete)
    wanted_suf = (
        "channels.tsv",
        "eeg.eeg",
        "eeg.json",
        "eeg.vhdr",
        "eeg.vmrk",
        "events.tsv",
    )
    paths: Dict[str, Path] = {}
    if dest_dir.exists():
        for p in dest_dir.iterdir():
            if p.name.endswith("_eeg.vhdr"):
                paths["vhdr"] = p
            elif p.name.endswith("_events.tsv"):
                paths["events"] = p
            elif p.name.endswith("_channels.tsv"):
                paths["channels"] = p
        if "vhdr" in paths and "events" in paths:
            print(f"[cache] {sub} raw present", flush=True)
            return paths

    pause(f"list {sub}")
    files = list_s3(prefix)
    for key, size in files:
        fname = key.split("/")[-1]
        if not any(fname.endswith(suf) for suf in wanted_suf):
            continue
        dest = RAW / sub / "eeg" / fname
        pause(f"dl {fname} ({size/1e6:.1f}MB)")
        download(f"https://s3.amazonaws.com/openneuro.org/{key}", dest, expected_size=size)
        if fname.endswith("_eeg.vhdr"):
            paths["vhdr"] = dest
        elif fname.endswith("_events.tsv"):
            paths["events"] = dest
        elif fname.endswith("_channels.tsv"):
            paths["channels"] = dest
    missing = [k for k in ("vhdr", "events") if k not in paths]
    if missing:
        raise FileNotFoundError(f"{sub} missing: {missing}")
    return paths


def main_nback_blocks_all(events_path: Path) -> List[Dict[str, Any]]:
    """Contiguous non-tutorial blocks for levels 1–4."""
    with events_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    trials = [
        r
        for r in rows
        if r["trial_type"].endswith("-back")
        and str(r.get("istutorial", "false")).lower() == "false"
        and r.get("nback_level") in {"1", "2", "3", "4"}
    ]
    blocks: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for r in trials:
        lvl = r["nback_level"]
        onset = float(r["onset"])
        if cur is None or cur["level"] != lvl:
            if cur is not None:
                blocks.append(cur)
            cur = {
                "level": lvl,
                "start": onset,
                "end": onset,
                "n_trials": 1,
            }
        else:
            cur["end"] = onset
            cur["n_trials"] += 1
    if cur is not None:
        blocks.append(cur)
    for b in blocks:
        b["end"] = float(b["end"]) + 1.7
        b["duration_sec"] = float(b["end"] - b["start"])
    return blocks


def load_muse4(vhdr: Path) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
    available = {c.upper(): c for c in raw.ch_names}
    picks = []
    for src in SRC_CHS:
        key = src.upper()
        if key not in available:
            raise KeyError(f"{vhdr.name}: missing {src}; have {raw.ch_names}")
        picks.append(available[key])
    raw.pick(picks)
    rename = {available[s.upper()]: CH_MAP[s] for s in SRC_CHS}
    raw.rename_channels(rename)
    raw.reorder_channels(MUSE_CHS)
    if abs(raw.info["sfreq"] - TARGET_SR) > 1e-3:
        raw.resample(TARGET_SR, verbose="ERROR")
    data = raw.get_data().astype(np.float64)
    raw_mean = data.mean(axis=1, keepdims=True)
    raw_std = data.std(axis=1, keepdims=True)
    raw_std = np.maximum(raw_std, 1e-12)
    data = ((data - raw_mean) / raw_std).astype(np.float32)
    data = np.clip(data, -15.0, 15.0)
    meta = {
        "sfreq": float(raw.info["sfreq"]),
        "n_times": int(data.shape[1]),
        "duration_sec": float(data.shape[1] / raw.info["sfreq"]),
        "ch_names": list(MUSE_CHS),
        "src_channels": list(SRC_CHS),
        "channel_map": dict(CH_MAP),
        "vhdr": str(vhdr.relative_to(ROOT)),
        "scale_note": "per_channel_zscore_clip15_nonphysical_brainvision_floats",
        "raw_mean_before_z": raw_mean.ravel().astype(float).tolist(),
        "raw_std_before_z": raw_std.ravel().astype(float).tolist(),
    }
    return data, float(raw.info["sfreq"]), meta


def window_block(
    data: np.ndarray,
    sfreq: float,
    start_sec: float,
    end_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    s0 = max(0, int(round(start_sec * sfreq)))
    s1 = min(data.shape[1], int(round(end_sec * sfreq)))
    seg = data[:, s0:s1]
    trim = int(round(EDGE_TRIM_SEC * sfreq))
    if seg.shape[1] <= 2 * trim + int(round(WINDOW_SEC * sfreq)):
        trim = max(0, int(seg.shape[1] * 0.05))
    cropped = seg[:, trim : seg.shape[1] - trim if trim else seg.shape[1]]
    wins, starts = [], []
    for start, w in sliding_windows(cropped, sfreq, WINDOW_SEC, HOP_SEC, axis=-1):
        wins.append(w)
        starts.append(s0 + trim + start)
    if not wins:
        raise ValueError(f"no windows span={end_sec-start_sec:.1f}s")
    X = np.stack(wins, axis=0).astype(np.float32)
    starts_arr = np.asarray(starts, dtype=np.int64)
    if len(X) > MAX_WINDOWS_PER_BLOCK:
        idx = np.unique(np.linspace(0, len(X) - 1, MAX_WINDOWS_PER_BLOCK).astype(int))
        X = X[idx]
        starts_arr = starts_arr[idx]
    return X, starts_arr


def build_subject_pack(sub: str, paths: Dict[str, Path]) -> Dict[str, Any]:
    data, sfreq, meta = load_muse4(paths["vhdr"])
    blocks = main_nback_blocks_all(paths["events"])
    by_level: Dict[str, Dict[str, Any]] = {}
    for b in blocks:
        X, starts = window_block(data, sfreq, b["start"], b["end"])
        by_level[b["level"]] = {
            **b,
            "X": X,
            "starts": starts,
            "n_windows": int(len(X)),
            "t_mid": float((b["start"] + b["end"]) / 2.0),
        }
        print(
            f"  {sub} L{b['level']}: span={b['duration_sec']:.1f}s "
            f"windows={len(X)} t=[{b['start']:.1f},{b['end']:.1f}]",
            flush=True,
        )
    missing = [lv for lv in ("1", "2", "3", "4") if lv not in by_level]
    if missing:
        raise RuntimeError(f"{sub} missing main levels {missing}")
    # session mid for early/late time-bin control (main experiment only)
    t0 = min(by_level[lv]["start"] for lv in by_level)
    t1 = max(by_level[lv]["end"] for lv in by_level)
    session_mid = 0.5 * (t0 + t1)
    pack = {
        "subject": sub,
        "by_level": by_level,
        "session_t0": float(t0),
        "session_t1": float(t1),
        "session_mid": float(session_mid),
        "montage": "muse4",
        "channels": list(MUSE_CHS),
        "src_channels": list(SRC_CHS),
        "channel_map": dict(CH_MAP),
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "target_sr": TARGET_SR,
        "edge_trim_sec": EDGE_TRIM_SEC,
        "max_windows_per_block": MAX_WINDOWS_PER_BLOCK,
        "recording": meta,
        "protocol_note": "strict L1→L2→L3→L4; no counterbalancing",
    }
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    # save compact per-level arrays
    npz = WINDOWS_DIR / f"{sub}_orderaware_windows.npz"
    save_kw = {}
    man_levels = {}
    for lv, info in by_level.items():
        save_kw[f"X_L{lv}"] = info["X"]
        save_kw[f"starts_L{lv}"] = info["starts"]
        man_levels[lv] = {
            k: v for k, v in info.items() if k not in ("X", "starts")
        }
    np.savez_compressed(npz, **save_kw)
    man = WINDOWS_DIR / f"{sub}_orderaware_manifest.json"
    man.write_text(
        json.dumps(
            {
                "subject": sub,
                "levels": man_levels,
                "session_t0": pack["session_t0"],
                "session_t1": pack["session_t1"],
                "session_mid": pack["session_mid"],
                "montage": "muse4",
                "channels": list(MUSE_CHS),
                "channel_map": dict(CH_MAP),
                "protocol_note": pack["protocol_note"],
                "recording": meta,
            },
            indent=2,
        )
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


def half_slice(X: np.ndarray, starts: np.ndarray, which: str) -> Tuple[np.ndarray, np.ndarray]:
    """Take early or late half of windows ordered by start sample."""
    order = np.argsort(starts)
    n = len(order)
    mid = n // 2
    if which == "early":
        idx = order[:mid] if mid > 0 else order[: max(1, n)]
    elif which == "late":
        idx = order[mid:] if mid < n else order[-max(1, n) :]
    else:
        raise ValueError(which)
    return X[idx], starts[idx]


def make_binary_from_levels(
    pack: Dict[str, Any],
    low_level: str,
    high_level: str,
    low_half: Optional[str] = None,
    high_half: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build X,y,starts for low vs high engagement from specified levels/halves."""
    lo = pack["by_level"][low_level]
    hi = pack["by_level"][high_level]
    if low_half is None:
        X_lo, st_lo = lo["X"], lo["starts"]
    else:
        X_lo, st_lo = half_slice(lo["X"], lo["starts"], low_half)
    if high_half is None:
        X_hi, st_hi = hi["X"], hi["starts"]
    else:
        X_hi, st_hi = half_slice(hi["X"], hi["starts"], high_half)
    X = np.concatenate([X_lo, X_hi], axis=0)
    y = np.concatenate(
        [
            np.zeros(len(X_lo), dtype=np.int64),  # low_engagement
            np.ones(len(X_hi), dtype=np.int64),  # high_engagement
        ]
    )
    starts = np.concatenate([st_lo, st_hi], axis=0)
    return X, y, starts


def make_timebin(pack: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label all main-level windows by early vs late session (ignore load)."""
    mid = pack["session_mid"]
    Xs, ys, sts = [], [], []
    for lv in ("1", "2", "3", "4"):
        info = pack["by_level"][lv]
        X, starts = info["X"], info["starts"]
        # convert start sample → seconds
        sfreq = pack["recording"]["sfreq"]
        t_sec = starts.astype(np.float64) / float(sfreq)
        early = t_sec < mid
        late = ~early
        if early.any():
            Xs.append(X[early])
            ys.append(np.zeros(int(early.sum()), dtype=np.int64))  # early_session
            sts.append(starts[early])
        if late.any():
            Xs.append(X[late])
            ys.append(np.ones(int(late.sum()), dtype=np.int64))  # late_session
            sts.append(starts[late])
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sts)


def make_within_block_early_late(
    pack: Dict[str, Any], level: str = "2"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Within a single level: early half vs late half (pure time, no load change)."""
    info = pack["by_level"][level]
    X_e, st_e = half_slice(info["X"], info["starts"], "early")
    X_l, st_l = half_slice(info["X"], info["starts"], "late")
    X = np.concatenate([X_e, X_l], axis=0)
    y = np.concatenate(
        [
            np.zeros(len(X_e), dtype=np.int64),
            np.ones(len(X_l), dtype=np.int64),
        ]
    )
    starts = np.concatenate([st_e, st_l], axis=0)
    return X, y, starts


def balance_xy(
    packs_xy: List[Tuple[str, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    Xs, ys = [], []
    detail: Dict[str, Any] = {}
    for sub, X, y in packs_xy:
        present = sorted(set(int(c) for c in np.unique(y)))
        if len(present) < 2:
            detail[sub] = {
                "skipped": True,
                "counts": {int(k): int(v) for k, v in Counter(y.tolist()).items()},
            }
            continue
        Xb, yb = undersample_balanced(X, y, rng)
        Xs.append(Xb)
        ys.append(yb)
        detail[sub] = {
            "skipped": False,
            "n_total": int(len(yb)),
            "per_class": {int(i): int((yb == i).sum()) for i in present},
            "counts_full": {int(k): int(v) for k, v in Counter(y.tolist()).items()},
        }
    if not Xs:
        raise RuntimeError("no train subjects with both classes")
    return np.concatenate(Xs), np.concatenate(ys), detail


def stride4(X: np.ndarray, y: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(starts)
    idx = order[::4]
    return X[idx], y[idx]


def eval_split(
    head: nn.Module,
    emb: torch.Tensor,
    y: np.ndarray,
    name: str,
    labels: Sequence[str],
) -> Dict[str, Any]:
    head.eval()
    if len(y) == 0:
        return {"name": name, "n": 0, "acc": None, "macro_f1": None, "note": "empty"}
    with torch.no_grad():
        pred = head(emb).argmax(dim=-1).numpy()
    acc = float((pred == y).mean())
    f1 = macro_f1(y.tolist(), pred.tolist(), list(labels))
    report = per_class_report(y.tolist(), pred.tolist(), list(labels))
    per_class = {k: v for k, v in report.items() if k != "macro_f1"}
    cm = confusion_matrix(y.tolist(), pred.tolist(), list(labels))
    return {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1": f1,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": list(labels),
        "counts_true": {labels[i]: int((y == i).sum()) for i in range(len(labels))},
    }


def train_head(
    emb: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    labels: Sequence[str],
) -> Tuple[HeadAEngLinear, List[Dict[str, Any]], Dict[str, Any]]:
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(VAL_FRAC * n)))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    emb_tr, y_tr = emb[tr_idx], y[tr_idx]
    emb_va, y_va = emb[va_idx], y[va_idx]

    head = HeadAEngLinear(in_dim=emb.shape[-1]).to(device)
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
            "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), list(labels)),
            "val_acc": float((pred_va == y_va).mean()),
            "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), list(labels)),
        }
        history.append(row)
        print(f"    {row}", flush=True)
        if row["val_macro_f1"] >= best_val_f1:
            best_val_f1 = float(row["val_macro_f1"])
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
        print(f"    restored best val epoch={best_epoch} val_macro_f1={best_val_f1:.4f}", flush=True)

    head.eval()
    with torch.no_grad():
        pred_tr = head(emb_tr.to(device)).argmax(dim=-1).cpu().numpy()
        pred_va = head(emb_va.to(device)).argmax(dim=-1).cpu().numpy()
    splits = {
        "train_n": int(len(y_tr)),
        "val_n": int(len(y_va)),
        "best_epoch": best_epoch,
        "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), list(labels)),
        "val_acc": float((pred_va == y_va).mean()),
        "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), list(labels)),
        "train_acc": float((pred_tr == y_tr).mean()),
    }
    return head, history, splits


def run_check(
    name: str,
    description: str,
    packs: List[Dict[str, Any]],
    encoder: FrozenCBraModEncoder,
    device: torch.device,
    rng: np.random.Generator,
    builder,
    labels: Sequence[str],
    is_primary: bool = False,
) -> Dict[str, Any]:
    print(f"\n===== CHECK: {name} =====", flush=True)
    print(description, flush=True)
    train_xy = []
    for p in packs:
        if p["subject"] not in TRAIN_SUBJECTS:
            continue
        X, y, starts = builder(p)
        train_xy.append((p["subject"], X, y))
    hold = next(p for p in packs if p["subject"] == HOLDOUT)
    X_h, y_h, st_h = builder(hold)

    X_tr, y_tr, bal_detail = balance_xy(train_xy, rng)
    print(f"  train balanced {X_tr.shape} {Counter(y_tr.tolist())}", flush=True)
    print(f"  holdout {hold['subject']} {X_h.shape} {Counter(y_h.tolist())}", flush=True)

    emb_tr = encode_all(encoder, X_tr, device)
    X_h_bal, y_h_bal = undersample_balanced(X_h, y_h, rng)
    X_h_s4, y_h_s4 = stride4(X_h, y_h, st_h)
    emb_h_full = encode_all(encoder, X_h, device)
    emb_h_bal = encode_all(encoder, X_h_bal, device)
    emb_h_s4 = encode_all(encoder, X_h_s4, device)

    # fresh RNG stream per check for val split reproducibility relative to name
    # stable per-check seed (avoid PYTHONHASHSEED nondeterminism)
    name_seed = SEED + sum(ord(c) for c in name) * 17
    local_rng = np.random.default_rng(name_seed)
    head, history, split_info = train_head(emb_tr, y_tr, device, local_rng, labels)

    ev_tr = eval_split(head, emb_tr, y_tr, "train_balanced", labels)
    ev_full = eval_split(head, emb_h_full, y_h, "holdout_full", labels)
    ev_bal = eval_split(head, emb_h_bal, y_h_bal, "holdout_balanced", labels)
    ev_s4 = eval_split(head, emb_h_s4, y_h_s4, "holdout_stride4", labels)

    print(f"  EVAL holdout bal macro-F1={ev_bal['macro_f1']:.4f} n={ev_bal['n']}", flush=True)

    pt_path = OUT_DIR / f"head_{name}.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": list(labels),
            "in_dim": int(emb_tr.shape[-1]),
            "montage": "muse4",
            "dataset": DATASET_NAME,
            "holdout": HOLDOUT,
            "check": name,
            "best_epoch": split_info["best_epoch"],
        },
        pt_path,
    )

    f1 = float(ev_bal["macro_f1"])
    return {
        "name": name,
        "description": description,
        "is_primary": is_primary,
        "labels": list(labels),
        "headline": {
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
            "delta_vs_chance": f1 - CHANCE_F1,
            "beats_chance_bar": f1 > SHIP_F1_MIN,
        },
        "evals": {
            "train_balanced": ev_tr,
            "holdout_full": ev_full,
            "holdout_balanced": ev_bal,
            "holdout_stride4": ev_s4,
        },
        "history": history,
        "split_info": split_info,
        "balance_detail": bal_detail,
        "head_pt": str(pt_path.relative_to(ROOT)),
        "head_pt_sha256": sha256(pt_path),
    }


def decide(primary_f1: float, checks: Dict[str, Any]) -> Dict[str, Any]:
    mid = checks["mid_2v3"]["headline"]["holdout_balanced_macro_f1"]
    edge_mid = checks["edge_2late_3early"]["headline"]["holdout_balanced_macro_f1"]
    edge_ext = checks["edge_1late_4early"]["headline"]["holdout_balanced_macro_f1"]
    timebin = checks["timebin_control"]["headline"]["holdout_balanced_macro_f1"]
    within = checks["within_L2_early_late"]["headline"]["holdout_balanced_macro_f1"]

    pass_bar = primary_f1 > SHIP_F1_MIN
    # Confound signals: time easy AND mid collapses, or mid high only when temporally far
    time_easy = timebin >= 0.70 or within >= 0.65
    mid_ok = mid > SHIP_F1_MIN
    edge_mid_ok = edge_mid > SHIP_F1_MIN

    if pass_bar and mid_ok and edge_mid_ok:
        verdict = "PASS_orderaware"
        note = (
            f"Primary mid 2vs3 holdout bal F1={mid:.3f} and edge-matched 2-late/3-early "
            f"F1={edge_mid:.3f} both > {SHIP_F1_MIN}. Load signal survives adjacent-block "
            f"and near-time-matched tests. Time-bin control F1={timebin:.3f}, "
            f"within-L2 early/late F1={within:.3f}. Residual risk: protocol still "
            f"strictly ordered (no counterbalance); n=1 holdout subject."
        )
        proceed_stress = True
    elif pass_bar and mid_ok and not edge_mid_ok:
        verdict = "PASS_weak_residual_time"
        note = (
            f"Primary mid 2vs3 F1={mid:.3f} > {SHIP_F1_MIN}, but edge-matched "
            f"2-late/3-early F1={edge_mid:.3f} is weaker — residual time/order risk. "
            f"Time-bin F1={timebin:.3f}, within-L2 F1={within:.3f}. Proceed to small "
            f"stress smoke with caution; do not ship A-eng."
        )
        proceed_stress = True
    elif not pass_bar:
        verdict = "FAIL_near_chance"
        note = (
            f"Primary mid 2vs3 holdout bal F1={mid:.3f} not clearly above "
            f"{SHIP_F1_MIN} (chance {CHANCE_F1}). Prior 1vs4 F1≈0.969 was likely "
            f"inflated by block order / session time. Time-bin control F1={timebin:.3f}, "
            f"within-L2 F1={within:.3f}, edge 1-late/4-early F1={edge_ext:.3f}. "
            f"STOP — do not download stress. Next: STEW rest-vs-SIMKAP if access, "
            f"or order-counterbalanced load corpus; personal Muse cal optional."
        )
        proceed_stress = False
    else:
        verdict = "MIXED"
        note = (
            f"Mixed order-aware results (mid={mid:.3f}, edge_mid={edge_mid:.3f}, "
            f"timebin={timebin:.3f}). Honest read: residual confound likely. "
            f"proceed_stress={pass_bar}."
        )
        proceed_stress = pass_bar

    if time_easy and not mid_ok:
        note += " Time controls easy while load collapses → confound confirmed."

    return {
        "verdict": verdict,
        "proceed_stress": proceed_stress,
        "pass_criterion": f"primary mid_2v3 holdout bal macro-F1 > {SHIP_F1_MIN}",
        "primary_holdout_balanced_macro_f1": primary_f1,
        "chance_macro_f1": CHANCE_F1,
        "delta_vs_chance": primary_f1 - CHANCE_F1,
        "ship_candidate": False,  # never ship from this n=8 ordered protocol alone
        "note": note,
        "time_control_easy": time_easy,
        "recommendation": (
            "proceed_small_stress_smoke" if proceed_stress else "stop_skip_stress"
        ),
    }


def update_doc(summary: Dict[str, Any]) -> None:
    """Append / refresh order-aware section in docs/head_a_eng_smoke.md."""
    checks = summary["checks"]
    dec = summary["decision"]
    prior = DOC_PATH.read_text() if DOC_PATH.exists() else ""

    # Strip previous order-aware append if re-run
    marker = "\n## Order-aware re-smoke"
    if marker in prior:
        prior = prior.split(marker)[0].rstrip() + "\n"

    rows = []
    for key in (
        "mid_2v3",
        "edge_2late_3early",
        "edge_1late_4early",
        "timebin_control",
        "within_L2_early_late",
    ):
        c = checks[key]
        h = c["headline"]
        flag = "PRIMARY" if c["is_primary"] else "ctrl/sec"
        rows.append(
            f"| `{c['name']}` ({flag}) | {h['holdout_balanced_n']} | "
            f"{h['holdout_balanced_acc']:.3f} | {h['holdout_balanced_macro_f1']:.3f} | "
            f"{h['delta_vs_chance']:+.3f} |"
        )

    section = [
        "",
        "## Order-aware re-smoke",
        "",
        f"**Completed (Asia/Bangkok):** {summary['completed_bkk']}",
        f"**UTC:** {summary['completed_utc']}",
        f"**Version:** `{VERSION_NOTE}`",
        f"**Exports:** `exports/head_a_eng_orderaware/`",
        f"**Script:** `scripts/head_a_eng_orderaware.py`",
        "",
        "### Why",
        "",
        "Prior 1-back→low / 4-back→high holdout bal macro-F1 ≈ 0.969, but main protocol "
        "is **strictly L1→L2→L3→L4** (no counterbalance / no temporal overlap). "
        "Order-aware checks break the time shortcut.",
        "",
        f"**Dataset:** `{DATASET_NAME}` — SPDX `{LICENSE_SPDX}`",
        "",
        "### Design",
        "",
        "| Check | What | Shortcut broken? |",
        "|-------|------|------------------|",
        "| `mid_2v3` **PRIMARY** | 2-back→low vs 3-back→high (full adjacent blocks) | "
        "Uses mid difficulties; ~minutes closer than 1vs4 |",
        "| `edge_2late_3early` | Late half 2-back vs early half 3-back | "
        "Nearly time-matched across instruction boundary |",
        "| `edge_1late_4early` | Late half 1-back vs early half 4-back | "
        "Closest edges of extremes; **residual gap** via L2+L3 (~5+ min) |",
        "| `timebin_control` | Predict early vs late main-session (ignore load) | "
        "If easy, time is discriminative |",
        "| `within_L2_early_late` | Early vs late half within 2-back only | "
        "Pure within-block drift (no load change) |",
        "",
        "### Results (holdout bal)",
        "",
        "| Check | n | accuracy | macro-F1 | Δ vs chance |",
        "|-------|--:|---------:|---------:|------------:|",
        *rows,
        "",
        f"**Pass bar (proceed to stress):** primary `mid_2v3` bal F1 **> {SHIP_F1_MIN}**",
        f"**Primary F1:** `{dec['primary_holdout_balanced_macro_f1']:.3f}` "
        f"(Δ `{dec['delta_vs_chance']:+.3f}`)",
        f"**Verdict:** `{dec['verdict']}`",
        f"**Proceed to stress/calm smoke?** `{dec['proceed_stress']}`",
        f"**ship_candidate:** `{dec['ship_candidate']}` (still False — ordered protocol / n=8)",
        "",
        "### Decision note",
        "",
        dec["note"],
        "",
        "### Residual risk (honest)",
        "",
        "- Protocol never counterbalances n-back order — strongest honest tests still "
        "have weak residual time structure.",
        "- Single holdout subject; overlapping windows; muse4 proxy F7/F8/T3/T4; "
        "non-physical BrainVision floats → z-score.",
        "- Engagement/load ≠ meditation stress/calm UX.",
        "",
        "### Takeaway (order-aware)",
        "",
        summary["takeaway"],
        "",
    ]
    DOC_PATH.write_text(prior.rstrip() + "\n" + "\n".join(section))


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Head A-eng ORDER-AWARE re-smoke (CBraMod CPU) ===", flush=True)
    print("dataset", DATASET_NAME, flush=True)
    print("subjects", SUBJECTS, "holdout", HOLDOUT, flush=True)

    ensure_meta()
    packs: List[Dict[str, Any]] = []
    for sub in SUBJECTS:
        print(f"\n--- {sub} ---", flush=True)
        paths = download_subject(sub)
        packs.append(build_subject_pack(sub, paths))

    weights = find_weights()
    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
    notes = encoder.adapter_notes()
    encoder.to(device)
    print("encoder", notes["native_input"], "<-", notes["fed_input"], flush=True)

    checks_spec = [
        (
            "mid_2v3",
            "PRIMARY: 2-back→low_engagement vs 3-back→high_engagement (adjacent mid levels)",
            lambda p: make_binary_from_levels(p, "2", "3"),
            HEAD_A_ENG_LABELS,
            True,
        ),
        (
            "edge_2late_3early",
            "Near time-match: late half of 2-back vs early half of 3-back",
            lambda p: make_binary_from_levels(p, "2", "3", "late", "early"),
            HEAD_A_ENG_LABELS,
            False,
        ),
        (
            "edge_1late_4early",
            "Closest edges of extremes (residual L2+L3 gap): late 1-back vs early 4-back",
            lambda p: make_binary_from_levels(p, "1", "4", "late", "early"),
            HEAD_A_ENG_LABELS,
            False,
        ),
        (
            "timebin_control",
            "CONTROL: predict early vs late main-session time bin (ignore load)",
            make_timebin,
            TIME_LABELS,
            False,
        ),
        (
            "within_L2_early_late",
            "CONTROL: early vs late half within 2-back only (no load change)",
            make_within_block_early_late,
            WITHIN_LABELS,
            False,
        ),
    ]

    checks: Dict[str, Any] = {}
    for name, desc, builder, labels, is_primary in checks_spec:
        checks[name] = run_check(
            name, desc, packs, encoder, device, rng, builder, labels, is_primary
        )

    primary_f1 = float(checks["mid_2v3"]["headline"]["holdout_balanced_macro_f1"])
    decision = decide(primary_f1, checks)

    bkk = ZoneInfo("Asia/Bangkok")
    now_utc = datetime.now(timezone.utc)
    now_bkk = now_utc.astimezone(bkk)

    takeaway = (
        f"ORDER-AWARE eng on {DATASET_NAME}: primary mid 2vs3 holdout bal "
        f"macro-F1={primary_f1:.3f} (bar>{SHIP_F1_MIN}); "
        f"edge 2late/3early={checks['edge_2late_3early']['headline']['holdout_balanced_macro_f1']:.3f}; "
        f"timebin={checks['timebin_control']['headline']['holdout_balanced_macro_f1']:.3f}; "
        f"within-L2={checks['within_L2_early_late']['headline']['holdout_balanced_macro_f1']:.3f}; "
        f"verdict={decision['verdict']}; proceed_stress={decision['proceed_stress']}."
    )

    summary = {
        "step": "head_a_eng_orderaware",
        "version": VERSION_NOTE,
        "dataset": DATASET_NAME,
        "dataset_id": DATASET_ID,
        "dataset_doi": DATASET_DOI,
        "license_spdx": LICENSE_SPDX,
        "montage": "muse4",
        "channel_map": dict(CH_MAP),
        "subjects": SUBJECTS,
        "train_subjects": TRAIN_SUBJECTS,
        "holdout": HOLDOUT,
        "n_subjects": len(SUBJECTS),
        "protocol": "strict L1→L2→L3→L4 main; tutorials excluded",
        "prior_smoke": {
            "path": "exports/head_a_eng_smoke/",
            "label_mapping": {"1-back": "low_engagement", "4-back": "high_engagement"},
            "holdout_balanced_macro_f1": 0.9687194525904204,
            "confound": "1-back always precedes 4-back",
        },
        "pass_criterion": decision["pass_criterion"],
        "checks": checks,
        "decision": decision,
        "weights_sha256": sha256(weights),
        "completed_utc": now_utc.isoformat(),
        "completed_bkk": now_bkk.strftime("%Y-%m-%d %H:%M:%S ICT"),
        "takeaway": takeaway,
        "no_kaggle_push": True,
    }

    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/head_a_eng_orderaware.py",
                "version": VERSION_NOTE,
                "dataset": DATASET_NAME,
                "subjects": SUBJECTS,
                "holdout": HOLDOUT,
                "seed": SEED,
                "epochs": EPOCHS,
                "checks": list(checks.keys()),
                "primary_check": "mid_2v3",
                "montage": "muse4",
            },
            indent=2,
        )
    )
    (OUT_DIR / "step_summary.json").write_text(
        json.dumps(
            {
                "step": "head_a_eng_orderaware",
                "dataset": DATASET_NAME,
                "primary_check": "mid_2v3",
                "primary_holdout_balanced_macro_f1": primary_f1,
                "chance_macro_f1": CHANCE_F1,
                "pass_bar": SHIP_F1_MIN,
                "verdict": decision["verdict"],
                "proceed_stress": decision["proceed_stress"],
                "ship_candidate": False,
                "recommendation": decision["recommendation"],
            },
            indent=2,
        )
    )
    update_doc(summary)

    print("\n=== DONE order-aware ===", flush=True)
    print(takeaway, flush=True)
    print("decision", json.dumps(decision, indent=2), flush=True)
    print("doc", DOC_PATH, flush=True)
    print("PROCEED_STRESS", decision["proceed_stress"], flush=True)


if __name__ == "__main__":
    main()

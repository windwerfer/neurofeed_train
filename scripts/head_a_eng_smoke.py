#!/usr/bin/env python3
"""Head A-eng CPU chance-check smoke: frozen CBraMod on OpenNeuro ds007169.

SWITCHED from stress/calm: PhysioNet neuro-stress-resilience-hci (MATB) is open
(ODbL) with Resting/Stress/Recovery phases and professional 32-ch g.tec EEG, but
each subject CSV is ~606 MB (~4.8 GB for n=8) and AF7/AF8 are fNIRS emitters not
EEG — download paced at ~0.25 MB/s (~40 min/subject) so aborted for smoke bound.
Native Muse stress/relax (alkabbany; CC-BY-4.0) is Google-Drive only + n=5.
STEW is IEEE-login gated for raw files.

Fallback (this script): OpenNeuro ds007169 n-back workload (CC0) — 19-ch
professional mobile EEG @ 250 Hz; muse4 proxy F7/F8/T3/T4 → AF7/AF8/TP9/TP10.
Honest labels: 1-back → low_engagement, 4-back → high_engagement (extremes).

Montage: muse4 only. Never mix crown8.
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
from typing import Any, Dict, List, Optional, Tuple
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
PAUSE_SEC = 1.5

# muse4 proxy from 19-ch 10–20 (no native AF7/AF8/TP9/TP10)
SRC_CHS = ["F7", "F8", "T3", "T4"]
MUSE_CHS = ["AF7", "AF8", "TP9", "TP10"]
CH_MAP = dict(zip(SRC_CHS, MUSE_CHS))

# 8 analysis_included subjects; holdout last for subject-out chance check
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

# extreme n-back bins only (drop 2/3)
LEVEL_TO_LABEL = {
    "1": "low_engagement",
    "4": "high_engagement",
}

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
OUT_DIR = ROOT / "exports" / "head_a_eng_smoke"
WINDOWS_DIR = OUT_DIR / "windows"
DOC_PATH = ROOT / "docs" / "head_a_eng_smoke.md"
VERSION_NOTE = "eng_fallback_ds007169_n8_zscore_2026-09-08"

# ship bar: holdout balanced macro-F1 clearly above chance
SHIP_F1_MIN = 0.55
CHANCE_F1 = 0.5


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
    pause(f"list {sub}")
    files = list_s3(prefix)
    wanted = (
        "channels.tsv",
        "eeg.eeg",
        "eeg.json",
        "eeg.vhdr",
        "eeg.vmrk",
        "events.tsv",
    )
    paths: Dict[str, Path] = {}
    for key, size in files:
        fname = key.split("/")[-1]
        if not any(fname.endswith(suf) for suf in wanted):
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


def main_nback_blocks(events_path: Path) -> List[Dict[str, Any]]:
    """Contiguous non-tutorial blocks for levels 1 and 4 only."""
    with events_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    trials = [
        r
        for r in rows
        if r["trial_type"].endswith("-back")
        and str(r.get("istutorial", "false")).lower() == "false"
        and r.get("nback_level") in LEVEL_TO_LABEL
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
                "label": LEVEL_TO_LABEL[lvl],
                "start": onset,
                "end": onset,
                "n_trials": 1,
            }
        else:
            cur["end"] = onset
            cur["n_trials"] += 1
    if cur is not None:
        blocks.append(cur)
    # extend end by ~1 ISI so last trial has content (~1.7s)
    for b in blocks:
        b["end"] = float(b["end"]) + 1.7
        b["duration_sec"] = float(b["end"] - b["start"])
    return blocks


def load_muse4(vhdr: Path) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
    # channel names may be FP1/FP2; pick F7/F8/T3/T4
    available = {c.upper(): c for c in raw.ch_names}
    picks = []
    for src in SRC_CHS:
        key = src.upper()
        if key not in available:
            raise KeyError(f"{vhdr.name}: missing {src}; have {raw.ch_names}")
        picks.append(available[key])
    raw.pick(picks)
    # rename to muse4 order
    rename = {available[s.upper()]: CH_MAP[s] for s in SRC_CHS}
    raw.rename_channels(rename)
    raw.reorder_channels(MUSE_CHS)
    if abs(raw.info["sfreq"] - TARGET_SR) > 1e-3:
        raw.resample(TARGET_SR, verbose="ERROR")
    data = raw.get_data().astype(np.float64)  # (4, T)
    # ds007169 BrainVision floats are non-physical (~1e12 ADC-ish). MNE SI
    # values remain huge (~1e5–1e6). Per-channel z-score (honest; documented)
    # so CBraMod sees O(1) amplitudes comparable to other smokes.
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
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        raise ValueError(f"no windows label={label} span={end_sec-start_sec:.1f}s")
    X = np.stack(wins, axis=0).astype(np.float32)
    starts_arr = np.asarray(starts, dtype=np.int64)
    if len(X) > MAX_WINDOWS_PER_BLOCK:
        idx = np.unique(np.linspace(0, len(X) - 1, MAX_WINDOWS_PER_BLOCK).astype(int))
        X = X[idx]
        starts_arr = starts_arr[idx]
    y = np.full(len(X), HEAD_A_ENG_LABELS.index(label), dtype=np.int64)
    return X, y, starts_arr


def build_subject_pack(
    sub: str,
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    data, sfreq, meta = load_muse4(paths["vhdr"])
    blocks = main_nback_blocks(paths["events"])
    Xs, ys, starts_all, block_meta = [], [], [], []
    for b in blocks:
        X, y, starts = window_block(data, sfreq, b["start"], b["end"], b["label"])
        Xs.append(X)
        ys.append(y)
        starts_all.append(starts)
        block_meta.append({**b, "n_windows": int(len(y))})
        print(
            f"  {sub} L{b['level']}→{b['label']}: "
            f"span={b['duration_sec']:.1f}s windows={len(y)}",
            flush=True,
        )
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    starts = np.concatenate(starts_all, axis=0)
    counts = {HEAD_A_ENG_LABELS[i]: int((y == i).sum()) for i in range(2)}
    pack = {
        "subject": sub,
        "X": X,
        "y": y,
        "starts": starts,
        "counts": counts,
        "blocks": block_meta,
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
    }
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    npz = WINDOWS_DIR / f"{sub}_aeng_windows.npz"
    man = WINDOWS_DIR / f"{sub}_aeng_manifest.json"
    np.savez_compressed(
        npz,
        X=X,
        y=y,
        starts=starts,
        label_names=np.asarray(HEAD_A_ENG_LABELS),
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
    packs: List[Dict[str, Any]],
    rng: np.random.Generator,
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
            "per_class": {HEAD_A_ENG_LABELS[i]: int((yb == i).sum()) for i in present},
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
    f1 = macro_f1(y.tolist(), pred.tolist(), HEAD_A_ENG_LABELS)
    report = per_class_report(y.tolist(), pred.tolist(), HEAD_A_ENG_LABELS)
    per_class = {k: v for k, v in report.items() if k != "macro_f1"}
    cm = confusion_matrix(y.tolist(), pred.tolist(), HEAD_A_ENG_LABELS)
    return {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1": f1,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": list(HEAD_A_ENG_LABELS),
        "counts_true": {HEAD_A_ENG_LABELS[i]: int((y == i).sum()) for i in range(2)},
    }


def train_head(
    emb: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
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
            "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), HEAD_A_ENG_LABELS),
            "val_acc": float((pred_va == y_va).mean()),
            "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), HEAD_A_ENG_LABELS),
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
        "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), HEAD_A_ENG_LABELS),
        "val_acc": float((pred_va == y_va).mean()),
        "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), HEAD_A_ENG_LABELS),
        "train_acc": float((pred_tr == y_tr).mean()),
    }
    return head, history, splits


def recommendation(holdout_bal_f1: float) -> Dict[str, Any]:
    """ship_candidate always False here: 1-back always precedes 4-back (order confound)."""
    delta = holdout_bal_f1 - CHANCE_F1
    if holdout_bal_f1 >= 0.60:
        rec = "keep_exploring"
        note = (
            "Holdout bal macro-F1 >> chance, but main 1-back always precedes 4-back — "
            "possible temporal/drift confound. Keep exploring (STEW / order-aware); "
            "do NOT ship public A-eng yet. Stress/calm still needs lighter corpus."
        )
    elif holdout_bal_f1 >= 0.55:
        rec = "keep_exploring_weak"
        note = (
            "Weak lift vs chance on subject holdout — continue research / personal-cal; "
            "do not ship. Block-order confound also applies."
        )
    elif holdout_bal_f1 >= 0.45:
        rec = "personal_cal_only"
        note = (
            "Near chance — drop as public A-eng; personal Muse cal only if product wants "
            "load UX. Stress/calm still untested at smoke scale."
        )
    else:
        rec = "drop_public"
        note = (
            "At/below chance — drop public engagement head; optional personal-cal only. "
            "Stress/calm still untested (PhysioNet too heavy for this smoke)."
        )
    return {
        "recommendation": rec,
        "ship_candidate": False,  # veto: block-order confound on ds007169 extremes
        "holdout_balanced_macro_f1": holdout_bal_f1,
        "chance_macro_f1": CHANCE_F1,
        "delta_vs_chance": delta,
        "note": note,
        "run_engagement_next": False,
        "run_stress_next_if_lighter_data": True,
        "block_order_confound": True,
    }


def write_doc(summary: Dict[str, Any]) -> None:
    h = summary["headline"]
    rec = summary["recommendation"]
    lines = [
        "# Head A-eng smoke (CBraMod, CPU) — engagement fallback",
        "",
        f"**Completed (Asia/Bangkok):** {summary['completed_bkk']}",
        f"**UTC:** {summary['completed_utc']}",
        f"**Version:** `{VERSION_NOTE}`",
        "",
        "## Goal",
        "",
        "Frozen **CBraMod** chance-check for stress/calm Head exploration; **switched** to "
        "**A-eng** (`low_engagement` vs `high_engagement`) after stress corpora proved "
        "too heavy/gated for a size-bounded smoke.",
        "",
        "## Data pick (first viable → fallback)",
        "",
        "| Candidate | License | Why not / why used |",
        "|-----------|---------|-------------------|",
        "| PhysioNet `neuro-stress-resilience-hci` MATB | ODbL v1.0 | "
        "~606 MB/subject CSV; AF7/AF8 are fNIRS not EEG; ~0.25 MB/s → aborted |",
        "| alkabbany Muse-S stress/relax | CC-BY-4.0 | Google Drive only; n=5 under target |",
        "| STEW Emotiv 14-ch | CC-BY-4.0 | IEEE login for raw; HF mirror processed-only |",
        f"| **{DATASET_NAME}** | **{LICENSE_SPDX}** | **USED** — ~23 MB EEG/subject; "
        "objective 1–4 back |",
        "",
        f"**Dataset bracket tag:** `{DATASET_NAME}`",
        "",
        "## Label mapping (honest)",
        "",
        "| Source | A-eng label | Notes |",
        "|--------|-------------|-------|",
        "| main-experiment `1-back` block | `low_engagement` | Extreme low load bin |",
        "| main-experiment `4-back` block | `high_engagement` | Extreme high load bin |",
        "| `2-back` / `3-back` | **excluded** | Mid bins dropped for cleaner binary |",
        "| tutorial trials (`istutorial=true`) | **excluded** | Protocol practice |",
        "| Resting / Stress (PhysioNet) | **not used** | Download aborted (size) |",
        "",
        "## Montage",
        "",
        "- **muse4 only** (never mixed with crown8)",
        "- Proxy map: `F7→AF7`, `F8→AF8`, `T3→TP9`, `T4→TP10`",
        "- Resample 250 Hz → 256 Hz; 2 s windows / 1 s hop; edge trim 5 s; "
        "cap 80 windows/block",
        "",
        "## Split",
        "",
        f"- Train: `{', '.join(TRAIN_SUBJECTS)}` (n={len(TRAIN_SUBJECTS)})",
        f"- Holdout: `{HOLDOUT}`",
        "- Per-subject `undersample_balanced` then concat; internal 15% val; best-by-val",
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
        f"**ship_candidate:** `{summary['ship_candidate']}` (bar: holdout bal F1 ≥ {SHIP_F1_MIN})",
        "",
        "## Recommendation",
        "",
        f"- **Code:** `{rec['recommendation']}`",
        f"- **Note:** {rec['note']}",
        f"- **Run engagement next?** `{rec['run_engagement_next']}` (this run already is eng)",
        f"- **Lighter stress corpus later?** `{rec['run_stress_next_if_lighter_data']}`",
        "",
        "## Provenance",
        "",
        f"- Dataset: `{DATASET_NAME}` — SPDX `{LICENSE_SPDX}` — `{DATASET_DOI}`",
        "- Encoder: CBraMod Apache-2.0 `pretrained_weights.pth` (local cache)",
        "- Montage: **muse4** only",
        "- Script: `scripts/head_a_eng_smoke.py`",
        "- Exports: `exports/head_a_eng_smoke/`",
        "- No Kaggle push",
        "",
        "## Caveats",
        "",
        "- Engagement/load ≠ meditation stress/calm UX; product mapping is adjacent only.",
        "- F7/F8/T3/T4 are proxies for Muse AF7/AF8/TP9/TP10 — domain gap expected.",
        "- BrainVision floats are non-physical (~1e12); applied per-channel z-score + clip±15 before encode.",
        "- Overlapping windows; modest n=8; CBraMod TUEG → 4-ch transfer gap.",
        "- Stress/calm public head still **untested** at smoke scale.",
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

    print("=== Head A-eng smoke (CBraMod CPU) — engagement fallback ===", flush=True)
    print("dataset", DATASET_NAME, flush=True)
    print("labels", HEAD_A_ENG_LABELS, flush=True)
    print("subjects", SUBJECTS, "holdout", HOLDOUT, flush=True)

    ensure_meta()
    packs: List[Dict[str, Any]] = []
    for sub in SUBJECTS:
        print(f"\n--- {sub} ---", flush=True)
        paths = download_subject(sub)
        packs.append(build_subject_pack(sub, paths))

    train_packs = [p for p in packs if p["subject"] in TRAIN_SUBJECTS]
    hold = next(p for p in packs if p["subject"] == HOLDOUT)

    X_tr, y_tr, bal_detail = balance_train(train_packs, rng)
    print("train balanced", X_tr.shape, Counter(y_tr.tolist()), bal_detail, flush=True)
    print("holdout", hold["subject"], hold["X"].shape, hold["counts"], flush=True)

    weights = find_weights()
    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
    notes = encoder.adapter_notes()
    encoder.to(device)
    print("encoder", notes["native_input"], "<-", notes["fed_input"], flush=True)

    print("encoding train…", flush=True)
    emb_tr = encode_all(encoder, X_tr, device)

    X_h_full, y_h_full = hold["X"], hold["y"]
    X_h_bal, y_h_bal = undersample_balanced(hold["X"], hold["y"], rng)
    X_h_s4, y_h_s4 = stride4(hold["X"], hold["y"], hold["starts"])

    print("encoding holdout…", flush=True)
    emb_h_full = encode_all(encoder, X_h_full, device)
    emb_h_bal = encode_all(encoder, X_h_bal, device)
    emb_h_s4 = encode_all(encoder, X_h_s4, device)

    head, history, split_info = train_head(emb_tr, y_tr, device, rng)

    ev_tr = eval_split(head, emb_tr, y_tr, "train_balanced")
    ev_full = eval_split(head, emb_h_full, y_h_full, "holdout_full")
    ev_bal = eval_split(head, emb_h_bal, y_h_bal, "holdout_balanced")
    ev_s4 = eval_split(head, emb_h_s4, y_h_s4, "holdout_stride4")

    print("EVAL train", ev_tr, flush=True)
    print("EVAL holdout full", ev_full, flush=True)
    print("EVAL holdout bal", ev_bal, flush=True)
    print("EVAL holdout s4", ev_s4, flush=True)

    pt_path = OUT_DIR / "head_a_eng_ds007169.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": HEAD_A_ENG_LABELS,
            "in_dim": int(emb_tr.shape[-1]),
            "montage": "muse4",
            "dataset": DATASET_NAME,
            "holdout": HOLDOUT,
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
        f"SWITCHED stress→engagement: {DATASET_NAME}; muse4 proxy F7/F8/T3/T4; "
        f"n={len(SUBJECTS)} holdout {HOLDOUT}; holdout bal macro-F1="
        f"{ev_bal['macro_f1']:.3f} vs chance {CHANCE_F1} "
        f"(Δ{ev_bal['macro_f1']-CHANCE_F1:+.3f}); ship_candidate={rec['ship_candidate']}; "
        f"rec={rec['recommendation']}."
    )

    summary = {
        "step": "head_a_eng_smoke",
        "version": VERSION_NOTE,
        "switched_from": "stress_calm",
        "switch_reason": (
            "PhysioNet MATB stress ~606MB/subject too heavy/slow for smoke; "
            "alkabbany Muse stress Drive-gated n=5; STEW IEEE-gated"
        ),
        "dataset": DATASET_NAME,
        "dataset_id": DATASET_ID,
        "dataset_doi": DATASET_DOI,
        "license_spdx": LICENSE_SPDX,
        "montage": "muse4",
        "channel_map": dict(CH_MAP),
        "labels": HEAD_A_ENG_LABELS,
        "label_mapping": {f"{k}-back": v for k, v in LEVEL_TO_LABEL.items()},
        "subjects": SUBJECTS,
        "train_subjects": TRAIN_SUBJECTS,
        "holdout": HOLDOUT,
        "n_subjects": len(SUBJECTS),
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
        "ship_candidate": rec["ship_candidate"],
        "recommendation": rec,
        "weights_sha256": sha256(weights),
        "head_pt": str(pt_path.relative_to(ROOT)),
        "head_pt_sha256": sha256(pt_path),
        "completed_utc": now_utc.isoformat(),
        "completed_bkk": now_bkk.strftime("%Y-%m-%d %H:%M:%S ICT"),
        "takeaway": takeaway,
        "no_kaggle_push": True,
    }

    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/head_a_eng_smoke.py",
                "version": VERSION_NOTE,
                "dataset": DATASET_NAME,
                "subjects": SUBJECTS,
                "holdout": HOLDOUT,
                "seed": SEED,
                "epochs": EPOCHS,
                "window_sec": WINDOW_SEC,
                "hop_sec": HOP_SEC,
                "target_sr": TARGET_SR,
                "montage": "muse4",
                "channel_map": dict(CH_MAP),
            },
            indent=2,
        )
    )
    (OUT_DIR / "step_summary.json").write_text(
        json.dumps(
            {
                "step": "head_a_eng_smoke",
                "dataset": DATASET_NAME,
                "n_subjects": len(SUBJECTS),
                "holdout": HOLDOUT,
                "holdout_balanced_macro_f1": ev_bal["macro_f1"],
                "chance_macro_f1": CHANCE_F1,
                "ship_candidate": rec["ship_candidate"],
                "recommendation": rec["recommendation"],
                "switched_from_stress": True,
                "run_engagement_next": False,
            },
            indent=2,
        )
    )

    write_doc(summary)
    print("\n=== DONE ===", flush=True)
    print(takeaway, flush=True)
    print("doc", DOC_PATH, flush=True)


if __name__ == "__main__":
    main()

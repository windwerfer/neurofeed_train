#!/usr/bin/env python3
"""Head A-med CPU smoke: frozen CBraMod + HeadAMed on OpenNeuro ds003816.

Honest labels (CC0):
  PreResting  → rest
  LKMSelf     → meditation  (Loving-Kindness Meditation to Self)

Critical exclusions (documented, not used):
  - ds003969 think* is NOT rest
  - Sleep-EDF W is NOT rest for A-med
  - Local ds001787/ds003969 lack honest rest blocks

Montage: muse4 only (native AF7/AF8/TP9/TP10). Never mix crown8.
"""
from __future__ import annotations

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

import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder  # noqa: E402
from src.heads.base import class_weights_from_y, undersample_balanced  # noqa: E402
from src.heads.head_a_med import HEAD_A_MED_LABELS, HeadAMedLinear  # noqa: E402
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
EDGE_TRIM_SEC = 10.0
MAX_WINDOWS_PER_BLOCK = 80
PAUSE_SEC = 2.0
MUSE_CHS = ["AF7", "AF8", "TP9", "TP10"]

# n=12: prior n=6 lightest + next 6 lightest unused st with PreResting+LKMSelf ≥5MB each.
# Same holdout as n=6 for fair comparison. New ~406MB + prior ~162MB cached.
SUBJECTS = [
    # prior n=6 (lightest)
    "sub-09st",
    "sub-22st",
    "sub-48st",
    "sub-26st",
    "sub-19st",
    "sub-23st",
    # next 6 lightest unused ≥5MB/task (from scripts/_list_ds003816_st_sizes.py)
    "sub-16st",
    "sub-24st",
    "sub-36st",
    "sub-02st",
    "sub-05st",
    "sub-31st",
]
HOLDOUT = "sub-23st"  # same as n=6 smoke for fair holdout compare
TRAIN_SUBJECTS = [s for s in SUBJECTS if s != HOLDOUT]
N6_SUBJECTS = [
    "sub-09st",
    "sub-22st",
    "sub-48st",
    "sub-26st",
    "sub-19st",
    "sub-23st",
]
N6_HOLDOUT_MACRO_F1 = {
    "full": 0.358,
    "balanced": 0.333,
    "stride4": 0.357,
    "val": 0.364,
}
VERSION_NOTE = "n12_double_subjects_2026-09-07"

TASK_LABEL = {
    "PreResting": "rest",
    "LKMSelf": "meditation",
}

DATASET_ID = "ds003816"
DATASET_DOI = "doi:10.18112/openneuro.ds003816.v1.0.1"
LICENSE_SPDX = "CC0-1.0"
S3_ROOT = f"https://s3.amazonaws.com/openneuro.org/{DATASET_ID}"

CACHE = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data" / DATASET_ID
RAW = CACHE / "raw"
OUT_DIR = ROOT / "exports" / "head_a_med_smoke_n12"
WINDOWS_DIR = OUT_DIR / "windows"
DOC_PATH = ROOT / "docs" / "head_a_med_smoke.md"


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
        print(f"[skip] {dest.relative_to(CACHE)} ({dest.stat().st_size} B)", flush=True)
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
    for name in ("README", "participants.tsv", "dataset_description.json"):
        dest = CACHE / name
        if not dest.exists():
            pause(f"meta {name}")
            download(f"{S3_ROOT}/{name}", dest)


def download_subject(sub: str) -> Dict[str, Path]:
    """Download PreResting + LKMSelf BrainVision trio (+ channels) for ses-01."""
    prefix = f"{DATASET_ID}/{sub}/ses-01/eeg/"
    pause(f"list {sub}")
    files = list_s3(prefix)
    wanted_suffixes = (
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
        task = None
        for t in TASK_LABEL:
            if f"_task-{t}_" in fname:
                task = t
                break
        if task is None:
            continue
        if not any(fname.endswith(suf) for suf in wanted_suffixes):
            continue
        rel = Path(sub) / "ses-01" / "eeg" / fname
        dest = RAW / rel
        pause(f"dl {fname} ({size/1e6:.1f}MB)")
        download(f"https://s3.amazonaws.com/openneuro.org/{key}", dest, expected_size=size)
        if fname.endswith("_eeg.vhdr"):
            paths[task] = dest
    missing = [t for t in TASK_LABEL if t not in paths]
    if missing:
        raise FileNotFoundError(f"{sub} missing tasks: {missing}")
    return paths


def load_muse4_block(vhdr: Path) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Load BrainVision → (4, T) @ TARGET_SR muse4."""
    raw = mne.io.read_raw_brainvision(str(vhdr), preload=True, verbose="ERROR")
    raw.pick(MUSE_CHS)
    # Ensure order
    raw.reorder_channels(MUSE_CHS)
    if abs(raw.info["sfreq"] - TARGET_SR) > 1e-3:
        raw.resample(TARGET_SR, verbose="ERROR")
    data = raw.get_data().astype(np.float32)  # (4, T) volts
    # Scale to microvolts-ish for encoder stability (Sleep-EDF path uses similar magnitudes)
    # BrainVision is typically µV already when MNE loads; keep as-is.
    meta = {
        "sfreq": float(raw.info["sfreq"]),
        "n_times": int(data.shape[1]),
        "duration_sec": float(data.shape[1] / raw.info["sfreq"]),
        "ch_names": list(MUSE_CHS),
        "vhdr": str(vhdr.relative_to(ROOT)),
    }
    return data, float(raw.info["sfreq"]), meta


def window_block(
    data: np.ndarray,
    sfreq: float,
    label: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X (N,4,T), y ids, starts — edge-trimmed, capped."""
    trim = int(round(EDGE_TRIM_SEC * sfreq))
    if data.shape[1] <= 2 * trim + int(round(WINDOW_SEC * sfreq)):
        # short recording: soft trim
        trim = max(0, int(data.shape[1] * 0.05))
    cropped = data[:, trim : data.shape[1] - trim if trim else data.shape[1]]
    wins = []
    starts = []
    for start, w in sliding_windows(cropped, sfreq, WINDOW_SEC, HOP_SEC, axis=-1):
        wins.append(w)
        starts.append(start + trim)
    if not wins:
        raise ValueError(f"no windows for label={label} shape={data.shape}")
    X = np.stack(wins, axis=0).astype(np.float32)
    starts_arr = np.asarray(starts, dtype=np.int64)
    if len(X) > MAX_WINDOWS_PER_BLOCK:
        # evenly spaced subsample for smoke (less correlated than random clump)
        idx = np.linspace(0, len(X) - 1, MAX_WINDOWS_PER_BLOCK).astype(int)
        idx = np.unique(idx)
        X = X[idx]
        starts_arr = starts_arr[idx]
    y = np.full(len(X), HEAD_A_MED_LABELS.index(label), dtype=np.int64)
    return X, y, starts_arr


def build_subject_pack(
    sub: str,
    vhdrs: Dict[str, Path],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    Xs, ys, starts_all, block_meta = [], [], [], {}
    for task, label in TASK_LABEL.items():
        data, sfreq, meta = load_muse4_block(vhdrs[task])
        X, y, starts = window_block(data, sfreq, label, rng)
        Xs.append(X)
        ys.append(y)
        starts_all.append(starts)
        block_meta[task] = {
            **meta,
            "label": label,
            "n_windows": int(len(y)),
            "vhdr_sha256": sha256(vhdrs[task]),
        }
        print(
            f"  {sub} {task}→{label}: dur={meta['duration_sec']:.1f}s "
            f"windows={len(y)} shape={X.shape}",
            flush=True,
        )
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    starts = np.concatenate(starts_all, axis=0)
    counts = {HEAD_A_MED_LABELS[i]: int((y == i).sum()) for i in range(2)}
    pack = {
        "subject": sub,
        "X": X,
        "y": y,
        "starts": starts,
        "counts": counts,
        "blocks": block_meta,
        "montage": "muse4",
        "channels": list(MUSE_CHS),
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "target_sr": TARGET_SR,
        "edge_trim_sec": EDGE_TRIM_SEC,
        "max_windows_per_block": MAX_WINDOWS_PER_BLOCK,
    }
    WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
    npz = WINDOWS_DIR / f"{sub}_amed_windows.npz"
    man = WINDOWS_DIR / f"{sub}_amed_manifest.json"
    np.savez_compressed(
        npz,
        X=X,
        y=y,
        starts=starts,
        label_names=np.asarray(HEAD_A_MED_LABELS),
    )
    man.write_text(
        json.dumps(
            {k: v for k, v in pack.items() if k not in ("X", "y", "starts")},
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
            "per_class": {HEAD_A_MED_LABELS[i]: int((yb == i).sum()) for i in present},
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
    f1 = macro_f1(y.tolist(), pred.tolist(), HEAD_A_MED_LABELS)
    report = per_class_report(y.tolist(), pred.tolist(), HEAD_A_MED_LABELS)
    per_class = {k: v for k, v in report.items() if k != "macro_f1"}
    cm = confusion_matrix(y.tolist(), pred.tolist(), HEAD_A_MED_LABELS)
    return {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1": f1,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": list(HEAD_A_MED_LABELS),
        "counts_true": {HEAD_A_MED_LABELS[i]: int((y == i).sum()) for i in range(2)},
    }


def train_head(
    emb: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[HeadAMedLinear, List[Dict[str, Any]], Dict[str, Any]]:
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(VAL_FRAC * n)))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    emb_tr, y_tr = emb[tr_idx], y[tr_idx]
    emb_va, y_va = emb[va_idx], y[va_idx]

    head = HeadAMedLinear(in_dim=emb.shape[-1]).to(device)
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
            "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), HEAD_A_MED_LABELS),
            "val_acc": float((pred_va == y_va).mean()),
            "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), HEAD_A_MED_LABELS),
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
        "val_macro_f1": macro_f1(y_va.tolist(), pred_va.tolist(), HEAD_A_MED_LABELS),
        "val_acc": float((pred_va == y_va).mean()),
        "train_macro_f1": macro_f1(y_tr.tolist(), pred_tr.tolist(), HEAD_A_MED_LABELS),
        "train_acc": float((pred_tr == y_tr).mean()),
    }
    return head, history, splits


def write_doc(summary: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    h = summary["headline"]
    lines = [
        "# Head A-med smoke (CBraMod, CPU)",
        "",
        f"**Completed (Asia/Bangkok):** {summary['completed_bkk']}",
        f"**UTC:** {summary['completed_utc']}",
        f"**Version:** `{VERSION_NOTE}` — n=12 double-subjects (n=6 kept in exports/head_a_med_smoke/)",
        "",
        "## Goal",
        "",
        "Frozen **CBraMod** + **HeadAMed** (`rest` vs `meditation`) local CPU smoke, "
        "muse4 only, subject holdout.",
        "",
        "## Data strategy",
        "",
        "1. Searched local `ds001787` / `ds003969` / cache for honest rest↔meditation — "
        "**no usable rest**: ds003969 only has `med*` + `think*` (think ≠ rest); "
        "ds001787 is probe-rated meditation task without baseline rest blocks; "
        "Sleep-EDF W is A-vig drowsy proxy — **not** used.",
        "2. Downloaded a **small** open slice: OpenNeuro **ds003816** (CC0) — "
        "Loving-Kindness Meditation study with explicit **PreResting** and **LKMSelf** "
        "blocks; native AF7/AF8/TP9/TP10.",
        "3. **n=6 (prior):** six lightest `st` subjects with both tasks ≥5 MB (~162 MB).",
        f"4. **n=12 (this run):** same six + next six lightest unused `st` "
        f"(`{', '.join(s for s in SUBJECTS if s not in N6_SUBJECTS)}`; ~406 MB new).",
        "",
        "## Label mapping (honest)",
        "",
        "| Source task | A-med label | Notes |",
        "|-------------|-------------|-------|",
        "| `PreResting` | `rest` | Eyes-closed pre-session resting state (protocol) |",
        "| `LKMSelf` | `meditation` | Radiating LKM to Self (eyes closed) |",
        "| ds003969 `think*` | **excluded** | Instructed thinking ≠ rest |",
        "| Sleep-EDF W | **excluded** | A-vig drowsy proxy, not med-rest |",
        "| HF EEGMeditation | **not used** | Gated; prefer ungated CC0 |",
        "",
        "## Split",
        "",
        f"- Train: `{', '.join(TRAIN_SUBJECTS)}` (n={len(TRAIN_SUBJECTS)})",
        f"- Holdout: `{HOLDOUT}` (same as n=6 for fair compare)",
        "- Per-subject undersample_balanced then concat; internal 15% val for early pick.",
        "",
        "## Metrics (headline) — n=12",
        "",
        f"| Split | n | accuracy | macro-F1 |",
        f"|-------|--:|---------:|---------:|",
        f"| Train (balanced) | {h['train_n']} | {h['train_acc']:.3f} | {h['train_macro_f1']:.3f} |",
        f"| Val | {h['val_n']} | {h['val_acc']:.3f} | {h['val_macro_f1']:.3f} |",
        f"| Holdout full | {h['holdout_full_n']} | {h['holdout_full_acc']:.3f} | {h['holdout_full_macro_f1']:.3f} |",
        f"| Holdout balanced | {h['holdout_balanced_n']} | {h['holdout_balanced_acc']:.3f} | {h['holdout_balanced_macro_f1']:.3f} |",
        f"| Holdout stride×4 | {h['holdout_stride4_n']} | {h['holdout_stride4_acc']:.3f} | {h['holdout_stride4_macro_f1']:.3f} |",
        "",
        f"**ship_candidate:** `{summary['ship_candidate']}`",
        "",
        "## n=6 vs n=12 comparison",
        "",
        "| Metric | n=6 | n=12 | \u0394 |",
        "|--------|----:|-----:|--:|",
        f"| Val macro-F1 | {summary.get('n6_baseline', N6_HOLDOUT_MACRO_F1)['val']:.3f} | {h['val_macro_f1']:.3f} | {h['val_macro_f1'] - summary.get('n6_baseline', N6_HOLDOUT_MACRO_F1)['val']:+.3f} |",
        f"| Holdout full macro-F1 | {summary.get('n6_baseline', N6_HOLDOUT_MACRO_F1)['full']:.3f} | {h['holdout_full_macro_f1']:.3f} | {h['holdout_full_macro_f1'] - summary.get('n6_baseline', N6_HOLDOUT_MACRO_F1)['full']:+.3f} |",
        f"| Holdout balanced macro-F1 | {summary.get('n6_baseline', N6_HOLDOUT_MACRO_F1)['balanced']:.3f} | {h['holdout_balanced_macro_f1']:.3f} | {h['holdout_balanced_macro_f1'] - summary.get('n6_baseline', N6_HOLDOUT_MACRO_F1)['balanced']:+.3f} |",
        f"| Holdout stride\u00d74 macro-F1 | {summary.get('n6_baseline', N6_HOLDOUT_MACRO_F1)['stride4']:.3f} | {h['holdout_stride4_macro_f1']:.3f} | {h['holdout_stride4_macro_f1'] - summary.get('n6_baseline', N6_HOLDOUT_MACRO_F1)['stride4']:+.3f} |",
        f"| Train subjects | 5 | {len(TRAIN_SUBJECTS)} | +{len(TRAIN_SUBJECTS) - 5} |",
        f"| Holdout subject | `{HOLDOUT}` | `{HOLDOUT}` | same |",
        "",
        f"**Meaningful improvement vs n=6?** `{summary.get('improved_vs_n6')}` "
        "(criterion: holdout balanced macro-F1 lift \u2265 +0.05 and \u2265 0.45 absolute).",
        "",
        "## Provenance",
        "",
        f"- Dataset: OpenNeuro `{DATASET_ID}` — SPDX `{LICENSE_SPDX}` — `{DATASET_DOI}`",
        "- Encoder: CBraMod Apache-2.0 pretrained_weights (local cache)",
        "- Montage: **muse4** only (never mixed with crown8)",
        f"- Script: `scripts/head_a_med_smoke.py`",
        "- Exports n=6: `exports/head_a_med_smoke/`",
        "- Exports n=12: `exports/head_a_med_smoke_n12/`",
        "- Alt pairs doc: `docs/head_a_med_alt_label_pairs.md`",
        "",
        "## Caveats",
        "",
        "- PreResting is protocol resting state (not personal Muse cal); LKM ≠ breath-focus "
        "meditation used in app UX — proxy only.",
        "- Still modest n subjects; overlapping windows; domain gap CBraMod TUEG → 4-ch Muse proxy.",
        "- No Kaggle push; REVE not trained for this smoke.",
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

    print("=== Head A-med smoke (CBraMod CPU) ===", flush=True)
    print("label_list", HEAD_A_MED_LABELS, flush=True)
    print("subjects", SUBJECTS, "holdout", HOLDOUT, flush=True)

    ensure_meta()
    packs: List[Dict[str, Any]] = []
    for sub in SUBJECTS:
        print(f"\n--- {sub} ---", flush=True)
        vhdrs = download_subject(sub)
        packs.append(build_subject_pack(sub, vhdrs, rng))

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

    head, history, splits = train_head(emb_tr, y_tr, device, rng)

    evals = {
        "train_balanced": eval_split(head, emb_tr, y_tr, "train_balanced"),
        f"holdout_{HOLDOUT}_full": eval_split(head, emb_h_full, y_h_full, f"holdout_{HOLDOUT}_full"),
        f"holdout_{HOLDOUT}_balanced": eval_split(
            head, emb_h_bal, y_h_bal, f"holdout_{HOLDOUT}_balanced"
        ),
        f"holdout_{HOLDOUT}_stride4": eval_split(
            head, emb_h_s4, y_h_s4, f"holdout_{HOLDOUT}_stride4"
        ),
    }
    for k, v in evals.items():
        print(k, {kk: v[kk] for kk in ("n", "acc", "macro_f1")}, flush=True)

    head_path = OUT_DIR / "head_a_med_linear_ds003816.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": list(HEAD_A_MED_LABELS),
            "in_dim": int(emb_tr.shape[-1]),
            "n_classes": 2,
            "head": "HeadAMedLinear",
            "dataset": DATASET_ID,
            "train_subjects": TRAIN_SUBJECTS,
            "holdout_subject": HOLDOUT,
            "task": "binary_rest_vs_meditation",
            "montage": "muse4",
        },
        head_path,
    )

    hold_full = evals[f"holdout_{HOLDOUT}_full"]
    hold_bal = evals[f"holdout_{HOLDOUT}_balanced"]
    hold_s4 = evals[f"holdout_{HOLDOUT}_stride4"]
    train_ev = evals["train_balanced"]

    headline = {
        "train_n": train_ev["n"],
        "train_acc": train_ev["acc"],
        "train_macro_f1": train_ev["macro_f1"],
        "val_n": splits["val_n"],
        "val_acc": splits["val_acc"],
        "val_macro_f1": splits["val_macro_f1"],
        "holdout_full_n": hold_full["n"],
        "holdout_full_acc": hold_full["acc"],
        "holdout_full_macro_f1": hold_full["macro_f1"],
        "holdout_balanced_n": hold_bal["n"],
        "holdout_balanced_acc": hold_bal["acc"],
        "holdout_balanced_macro_f1": hold_bal["macro_f1"],
        "holdout_stride4_n": hold_s4["n"],
        "holdout_stride4_acc": hold_s4["acc"],
        "holdout_stride4_macro_f1": hold_s4["macro_f1"],
        "n_subjects_total": len(SUBJECTS),
        "n_subjects_train": len(TRAIN_SUBJECTS),
        "n_windows_train_balanced": int(len(y_tr)),
        "n_windows_holdout_full": int(len(y_h_full)),
    }

    # Ship gate: require clearly strong subject-holdout (balanced macro-F1 ≥ 0.70)
    ship = bool(
        hold_bal["macro_f1"] is not None
        and hold_bal["macro_f1"] >= 0.70
        and hold_full["macro_f1"] is not None
        and hold_full["macro_f1"] >= 0.65
    )

    n6_base = dict(N6_HOLDOUT_MACRO_F1)
    bal_f1 = hold_bal["macro_f1"] if hold_bal["macro_f1"] is not None else 0.0
    improved = bool(
        bal_f1 >= 0.45 and (bal_f1 - n6_base["balanced"]) >= 0.05
    )
    takeaway = (
        f"A-med n=12 smoke on ds003816 muse4: val F1={headline['val_macro_f1']:.3f}; "
        f"holdout {HOLDOUT} full/bal/stride4 F1="
        f"{headline['holdout_full_macro_f1']:.3f}/"
        f"{headline['holdout_balanced_macro_f1']:.3f}/"
        f"{headline['holdout_stride4_macro_f1']:.3f} "
        f"(n6 bal was {n6_base['balanced']:.3f}; improved={improved})."
    )
    if hold_bal["macro_f1"] is not None:
        if not improved and hold_bal["macro_f1"] < 0.55:
            takeaway += (
                " No meaningful lift vs n=6 — rest↔med still near chance; "
                "see docs/head_a_med_alt_label_pairs.md for next label pair."
            )
        elif hold_bal["macro_f1"] < 0.55:
            takeaway += (
                " Holdout near chance — rest↔med transfer weak on this smoke; "
                "need more subjects / personal Muse cal before shipping A-med."
            )
        elif hold_bal["macro_f1"] < 0.70:
            takeaway += (
                " Modest holdout signal — useful plumbing check, not pack-ready."
            )
        else:
            takeaway += (
                " Strong subject-holdout on this small smoke — still treat as "
                "provisional until larger LOSO."
            )

    now = datetime.now(timezone.utc)
    # Asia/Bangkok = UTC+7
    bkk = (now.timestamp() + 7 * 3600)
    from datetime import timedelta

    completed_bkk = (now + timedelta(hours=7)).strftime("%Y-%m-%d %H:%M:%S ICT")

    metrics = {
        "task": "head_a_med_smoke_ds003816_n12",
        "version": VERSION_NOTE,
        "n6_baseline": n6_base,
        "improved_vs_n6": improved,
        "label_list": list(HEAD_A_MED_LABELS),
        "label_map": TASK_LABEL,
        "train_subjects": TRAIN_SUBJECTS,
        "holdout_subject": HOLDOUT,
        "train_balance": bal_detail,
        "holdout_counts_full": hold["counts"],
        "subject_counts_full": {p["subject"]: p["counts"] for p in packs},
        "splits": splits,
        "history": history,
        "evals": evals,
        "headline": headline,
        "ship_candidate": ship,
    }
    metrics_path = OUT_DIR / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    manifest = {
        "task": "head_a_med_smoke_n12",
        "version": VERSION_NOTE,
        "n6_baseline": n6_base,
        "improved_vs_n6": improved,
        "dataset": DATASET_ID,
        "doi": DATASET_DOI,
        "license_spdx": LICENSE_SPDX,
        "label_list": list(HEAD_A_MED_LABELS),
        "label_rule": "PreResting→rest; LKMSelf→meditation (protocol blocks)",
        "label_honesty": {
            "rest_source": "ds003816 task-PreResting",
            "meditation_source": "ds003816 task-LKMSelf (Loving Kindness Meditation to Self)",
            "excluded": [
                "ds003969 think* (instructed thinking ≠ rest)",
                "Sleep-EDF W (A-vig drowsy proxy)",
                "HF EEGMeditation (gated; not used)",
            ],
            "local_search": "ds001787/ds003969 lacked honest rest blocks for A-med",
        },
        "train_subjects": TRAIN_SUBJECTS,
        "holdout_subject": HOLDOUT,
        "montage": "muse4",
        "channels": MUSE_CHS,
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "target_sr": TARGET_SR,
        "edge_trim_sec": EDGE_TRIM_SEC,
        "max_windows_per_block": MAX_WINDOWS_PER_BLOCK,
        "train_balance": bal_detail,
        "npz_sha256": {p["subject"]: p["npz_sha256"] for p in packs},
        "encoder_name": "CBraMod",
        "encoder_weights_path": str(weights.relative_to(ROOT)),
        "encoder_weights_sha256": notes["weights_sha256"],
        "encoder_adapter": notes,
        "head": "HeadAMedLinear",
        "n_classes": 2,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "seed": SEED,
        "device": "cpu",
        "history": history,
        "headline": headline,
        "ship_candidate": ship,
        "metrics_ref": str(metrics_path.relative_to(ROOT)),
        "head_path": str(head_path.relative_to(ROOT)),
        "created_utc": now.isoformat(),
        "created_bkk": completed_bkk,
        "license_attribution": {
            "ds003816": "OpenNeuro CC0 — LKM / PreResting blocks (Sun / Wong / Gao)",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
        "caveats": [
            "protocol resting + LKM proxy — not personal Muse breath-focus cal",
            "subject count (11 train / 1 holdout; n=12 total)",
            "overlapping windows; stride×4 reported",
            "CBraMod TUEG domain gap vs 4-ch Muse proxy",
            "no Kaggle push; REVE not used",
        ],
    }
    man_path = OUT_DIR / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))

    summary = {
        "step": "head_a_med_smoke_n12",
        "version": VERSION_NOTE,
        "completed_utc": now.isoformat(),
        "completed_bkk": completed_bkk,
        "train_subjects": TRAIN_SUBJECTS,
        "holdout_subject": HOLDOUT,
        "n_subjects_total": len(SUBJECTS),
        "n6_baseline": n6_base,
        "improved_vs_n6": improved,
        "headline": headline,
        "ship_candidate": ship,
        "takeaway": takeaway,
        "label_sources": {
            "rest": "ds003816 PreResting",
            "meditation": "ds003816 LKMSelf",
        },
        "artifacts": {
            "script": "scripts/head_a_med_smoke.py",
            "docs": "docs/head_a_med_smoke.md",
            "docs_alt_pairs": "docs/head_a_med_alt_label_pairs.md",
            "metrics": str(metrics_path.relative_to(ROOT)),
            "manifest": str(man_path.relative_to(ROOT)),
            "head": str(head_path.relative_to(ROOT)),
            "windows": str(WINDOWS_DIR.relative_to(ROOT)),
        },
    }
    (OUT_DIR / "step_summary.json").write_text(json.dumps(summary, indent=2))
    write_doc(summary, manifest)
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print("wrote", DOC_PATH, flush=True)


if __name__ == "__main__":
    main()

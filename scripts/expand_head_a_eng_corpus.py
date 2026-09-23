#!/usr/bin/env python3
"""Expand Head A-eng engagement/load corpus toward ~120 unique subjects.

Open / ship-eligible sources (muse4 only; never mix crown8):
  - OpenNeuro ds007169 n-back (CC0, professional 19-ch) — 1-back→low, 4-back→high
  - OpenNeuro ds007262 arithmetic (CC0, professional 19-ch; SAME Barras cohort)
      low difficulty 0.6-1.5 → low; high 5.1-6.9 → high (randomized order)
  - PhysioNet EEGMAT (ODC-By-1.0, professional 23-ch) — rest→low, arithmetic→high
  - STEW via HF monster-monash/STEW processed windows (CC-BY-4.0, Emotiv hobbyist)
      rating-bin y: 0→low, 1→high — raw IEEE DataPort still login-walled
  - OpenNeuro ds007554 CMx7-MM (CC0, professional 32-ch)
      PassiveMotor→low, NbackArithmetic→high (subjective CL ratings agree)

Unique-person accounting: Barras ds007169+ds007262 share IDs → count once.
Target ~120 unique people for fit-vs-misfit decision. No full REVE/CBraMod train
required here; optional tiny chance-check is separate.

Resume-friendly. Prefer uv/.venv; pace OpenNeuro downloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.heads.head_a_eng import HEAD_A_ENG_LABELS  # noqa: E402
from src.windowing import sliding_windows  # noqa: E402

SEED = 42
WINDOW_SEC = 2.0
HOP_SEC = 1.0
TARGET_SR = 256.0
EDGE_TRIM_SEC = 5.0
MAX_WINDOWS_PER_BLOCK = 80
PAUSE_SEC = 1.0
MUSE_CHS = ["AF7", "AF8", "TP9", "TP10"]

CACHE = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data"
EXPORTS = ROOT / "exports"
CORPUS = ROOT / "datasets/engagement_a_eng"
WINDOWS_ROOT = EXPORTS / "windows_aeng"
PROGRESS = EXPORTS / "head_a_eng_expand"
DOC_PATH = ROOT / "docs/head_a_eng_corpus_expansion.md"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pause(msg: str, sec: float = PAUSE_SEC) -> None:
    print(f"[pace] {msg} (sleep {sec}s)", flush=True)
    if sec > 0:
        time.sleep(sec)


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


def download(url: str, dest: Path, expected_size: Optional[int] = None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        if expected_size is None or dest.stat().st_size == expected_size:
            return False
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    print(f"[get] {dest.name} ← {url[:100]}…", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "muse-eeg-heads/a-eng-expand"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r, tmp.open("wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"empty download {dest.name}")
        if expected_size is not None and tmp.stat().st_size != expected_size:
            got = tmp.stat().st_size
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"size mismatch {dest.name}: got {got} want {expected_size}")
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return True


def zscore_clip(data: np.ndarray) -> np.ndarray:
    mean = data.mean(axis=1, keepdims=True)
    std = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
    return np.clip(((data - mean) / std).astype(np.float32), -15.0, 15.0)


def _norm_ch(name: str) -> str:
    n = name.upper().strip()
    for prefix in ("EEG ", "EEG-", "EEG_"):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n.replace(".", "").replace(" ", "")


def pick_muse4(
    raw: mne.io.BaseRaw,
    src_chs: Sequence[str],
    aliases: Optional[Dict[str, Sequence[str]]] = None,
) -> mne.io.BaseRaw:
    """Pick 4 channels and rename to Muse order."""
    available = {_norm_ch(c): c for c in raw.ch_names}
    aliases = aliases or {}
    picks = []
    rename = {}
    for src, muse in zip(src_chs, MUSE_CHS):
        candidates = [src] + list(aliases.get(src, ()))
        found = None
        for cand in candidates:
            key = _norm_ch(cand)
            if key in available:
                found = available[key]
                break
        if found is None:
            raise KeyError(f"missing {src} (tried {candidates}); have {raw.ch_names}")
        picks.append(found)
        rename[found] = muse
    raw = raw.copy().pick(picks)
    raw.rename_channels(rename)
    raw.reorder_channels(list(MUSE_CHS))
    if abs(raw.info["sfreq"] - TARGET_SR) > 1e-3:
        raw.resample(TARGET_SR, verbose="ERROR")
    return raw


def window_span(
    data: np.ndarray,
    sfreq: float,
    start_sec: float,
    end_sec: float,
    label: str,
    edge_trim: float = EDGE_TRIM_SEC,
    max_windows: int = MAX_WINDOWS_PER_BLOCK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s0 = max(0, int(round(start_sec * sfreq)))
    s1 = min(data.shape[1], int(round(end_sec * sfreq)))
    seg = data[:, s0:s1]
    trim = int(round(edge_trim * sfreq))
    if seg.shape[1] <= 2 * trim + int(round(WINDOW_SEC * sfreq)):
        trim = max(0, int(seg.shape[1] * 0.05))
    cropped = seg[:, trim : seg.shape[1] - trim if trim else seg.shape[1]]
    wins, starts = [], []
    for start, w in sliding_windows(cropped, sfreq, WINDOW_SEC, HOP_SEC, axis=-1):
        wins.append(w)
        starts.append(s0 + trim + start)
    if not wins:
        return (
            np.zeros((0, 4, int(WINDOW_SEC * sfreq)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    X = np.stack(wins, axis=0).astype(np.float32)
    starts_arr = np.asarray(starts, dtype=np.int64)
    if len(X) > max_windows:
        idx = np.unique(np.linspace(0, len(X) - 1, max_windows).astype(int))
        X = X[idx]
        starts_arr = starts_arr[idx]
    y = np.full(len(X), HEAD_A_ENG_LABELS.index(label), dtype=np.int64)
    return X, y, starts_arr


def save_pack(
    out_dir: Path,
    subject_key: str,
    X: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{subject_key}_aeng_windows.npz"
    man = out_dir / f"{subject_key}_aeng_manifest.json"
    np.savez_compressed(
        npz,
        X=X,
        y=y,
        starts=starts,
        label_names=np.asarray(HEAD_A_ENG_LABELS),
    )
    counts = {HEAD_A_ENG_LABELS[i]: int((y == i).sum()) for i in range(2)}
    payload = {
        **meta,
        "subject_key": subject_key,
        "counts": counts,
        "n_windows": int(len(y)),
        "npz_path": str(npz.relative_to(ROOT)),
        "npz_sha256": sha256(npz),
        "montage": "muse4",
        "channels": list(MUSE_CHS),
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "target_sr": TARGET_SR,
        "labels": list(HEAD_A_ENG_LABELS),
        "completed_utc": utc_now(),
    }
    man.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def both_classes(y: np.ndarray) -> bool:
    return len(set(int(v) for v in np.unique(y))) >= 2


# ---------------------------------------------------------------------------
# ds007169
# ---------------------------------------------------------------------------

DS7169 = "ds007169"
DS7169_SRC = ["F7", "F8", "T3", "T4"]
DS7169_LEVEL = {"1": "low_engagement", "4": "high_engagement"}


def ensure_ds007169_meta(raw_root: Path) -> None:
    s3 = f"https://s3.amazonaws.com/openneuro.org/{DS7169}"
    for name in (
        "README",
        "participants.tsv",
        "dataset_description.json",
        "task-nback_eeg.json",
        "task-nback_events.json",
    ):
        dest = raw_root.parent / name if raw_root.name == "raw" else raw_root / name
        # store meta beside raw/
        dest = CACHE / DS7169 / name
        if not dest.exists():
            pause(f"meta {DS7169}/{name}")
            download(f"{s3}/{name}", dest)


def ds007169_subjects() -> List[str]:
    part = CACHE / DS7169 / "participants.tsv"
    if not part.exists():
        ensure_ds007169_meta(CACHE / DS7169 / "raw")
    rows = list(csv.DictReader(part.open(), delimiter="\t"))
    return [
        r["participant_id"]
        for r in rows
        if str(r.get("analysis_included", "true")).lower() == "true"
    ]


def download_ds007169_subject(sub: str) -> Dict[str, Path]:
    raw = CACHE / DS7169 / "raw"
    prefix = f"{DS7169}/{sub}/eeg/"
    pause(f"list {sub}")
    files = list_s3(prefix)
    wanted = ("channels.tsv", "eeg.eeg", "eeg.json", "eeg.vhdr", "eeg.vmrk", "events.tsv")
    paths: Dict[str, Path] = {}
    for key, size in files:
        fname = key.split("/")[-1]
        if not any(fname.endswith(suf) for suf in wanted):
            continue
        dest = raw / sub / "eeg" / fname
        if download(
            f"https://s3.amazonaws.com/openneuro.org/{key}", dest, expected_size=size
        ):
            pause(f"dl {fname}")
        if fname.endswith("_eeg.vhdr"):
            paths["vhdr"] = dest
        elif fname.endswith("_events.tsv"):
            paths["events"] = dest
    if "vhdr" not in paths or "events" not in paths:
        raise FileNotFoundError(f"{sub} missing vhdr/events")
    return paths


def nback_blocks(events_path: Path) -> List[Dict[str, Any]]:
    with events_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    trials = [
        r
        for r in rows
        if r["trial_type"].endswith("-back")
        and str(r.get("istutorial", "false")).lower() == "false"
        and r.get("nback_level") in DS7169_LEVEL
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
                "label": DS7169_LEVEL[lvl],
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


def ingest_ds007169(force: bool = False) -> List[Dict[str, Any]]:
    ensure_ds007169_meta(CACHE / DS7169 / "raw")
    out_dir = WINDOWS_ROOT / "ds007169"
    results = []
    for sub in ds007169_subjects():
        subject_key = f"ds007169_{sub}"
        man = out_dir / f"{subject_key}_aeng_manifest.json"
        if man.exists() and not force:
            results.append(json.loads(man.read_text()))
            print(f"[skip] {subject_key}", flush=True)
            continue
        paths = download_ds007169_subject(sub)
        raw = mne.io.read_raw_brainvision(str(paths["vhdr"]), preload=True, verbose="ERROR")
        raw = pick_muse4(raw, DS7169_SRC)
        data = zscore_clip(raw.get_data().astype(np.float64))
        sfreq = float(raw.info["sfreq"])
        Xs, ys, starts_all, block_meta = [], [], [], []
        for b in nback_blocks(paths["events"]):
            X, y, starts = window_span(data, sfreq, b["start"], b["end"], b["label"])
            if len(y) == 0:
                continue
            Xs.append(X)
            ys.append(y)
            starts_all.append(starts)
            block_meta.append({**b, "n_windows": int(len(y))})
        if not Xs:
            print(f"[warn] {subject_key} no windows", flush=True)
            continue
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        starts = np.concatenate(starts_all)
        if not both_classes(y):
            print(f"[warn] {subject_key} missing class", flush=True)
            continue
        meta = {
            "source": "ds007169",
            "dataset_tag": "Multimodal Cognitive Workload n-back / ds007169 (19-ch 10–20 mobile EEG, professional)",
            "license_spdx": "CC0-1.0",
            "device_class": "professional",
            "ch_count_source": 19,
            "src_channels": list(DS7169_SRC),
            "channel_map": dict(zip(DS7169_SRC, MUSE_CHS)),
            "unique_person_id": f"barras_{sub}",
            "original_subject": sub,
            "label_rule": "main 1-back→low_engagement; 4-back→high_engagement; drop 2/3 + tutorial",
            "order_confound": "strict L1→L2→L3→L4; 1-back always before 4-back",
            "scale_note": "per_channel_zscore_clip15_nonphysical_brainvision_floats",
            "blocks": block_meta,
            "research_only": False,
        }
        payload = save_pack(out_dir, subject_key, X, y, starts, meta)
        results.append(payload)
        print(f"[ok] {subject_key} n={len(y)} {payload['counts']}", flush=True)
    return results


# ---------------------------------------------------------------------------
# ds007262 — same Barras cohort; randomized arithmetic difficulty
# ---------------------------------------------------------------------------

DS7262 = "ds007262"
DS7262_LOW = {"0.6-1.5"}
DS7262_HIGH = {"5.1-6.0", "6.0-6.9"}


def ensure_ds007262_meta() -> None:
    s3 = f"https://s3.amazonaws.com/openneuro.org/{DS7262}"
    for name in (
        "README.md",
        "participants.tsv",
        "dataset_description.json",
        "task-arithmetic_eeg.json",
        "task-arithmetic_events.json",
    ):
        dest = CACHE / DS7262 / name
        if not dest.exists():
            pause(f"meta {DS7262}/{name}")
            try:
                download(f"{s3}/{name}", dest)
            except Exception as e:
                print(f"[warn] meta {name}: {e}", flush=True)


def ds007262_subjects() -> List[str]:
    ensure_ds007262_meta()
    part = CACHE / DS7262 / "participants.tsv"
    rows = list(csv.DictReader(part.open(), delimiter="\t"))
    return [
        r["participant_id"]
        for r in rows
        if str(r.get("analysis_included", "true")).lower() == "true"
    ]


def download_ds007262_subject(sub: str) -> Dict[str, Path]:
    raw = CACHE / DS7262 / "raw"
    prefix = f"{DS7262}/{sub}/eeg/"
    pause(f"list {sub}")
    files = list_s3(prefix)
    paths: Dict[str, Path] = {}
    wanted = ("channels.tsv", "eeg.eeg", "eeg.json", "eeg.vhdr", "eeg.vmrk", "events.tsv")
    for key, size in files:
        fname = key.split("/")[-1]
        if not any(fname.endswith(suf) for suf in wanted):
            continue
        dest = raw / sub / "eeg" / fname
        if download(
            f"https://s3.amazonaws.com/openneuro.org/{key}", dest, expected_size=size
        ):
            pause(f"dl {fname}")
        if fname.endswith("_eeg.vhdr"):
            paths["vhdr"] = dest
        elif fname.endswith("_events.tsv"):
            paths["events"] = dest
    if "vhdr" not in paths or "events" not in paths:
        raise FileNotFoundError(f"{sub} missing vhdr/events")
    return paths


def arithmetic_blocks(events_path: Path) -> List[Dict[str, Any]]:
    """Group non-tutorial trials by difficulty_range; extremes only."""
    with events_path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    # main phase only: after started_arithmetic, before finished
    in_main = False
    trials = []
    for r in rows:
        tt = r.get("trial_type", "")
        if tt == "started_arithmetic":
            in_main = True
            continue
        if tt == "finished_arithmetic":
            in_main = False
            continue
        if not in_main:
            continue
        dr = r.get("difficulty_range", "n/a")
        if dr in DS7262_LOW or dr in DS7262_HIGH:
            trials.append(r)
    # each trial is ~6s presentation; window around onset
    blocks = []
    for r in trials:
        dr = r["difficulty_range"]
        label = "low_engagement" if dr in DS7262_LOW else "high_engagement"
        onset = float(r["onset"])
        blocks.append(
            {
                "difficulty_range": dr,
                "label": label,
                "start": onset,
                "end": onset + 6.0,
                "duration_sec": 6.0,
                "n_trials": 1,
            }
        )
    return blocks


def ingest_ds007262(force: bool = False) -> List[Dict[str, Any]]:
    out_dir = WINDOWS_ROOT / "ds007262"
    results = []
    for sub in ds007262_subjects():
        subject_key = f"ds007262_{sub}"
        man = out_dir / f"{subject_key}_aeng_manifest.json"
        if man.exists() and not force:
            results.append(json.loads(man.read_text()))
            print(f"[skip] {subject_key}", flush=True)
            continue
        paths = download_ds007262_subject(sub)
        raw = mne.io.read_raw_brainvision(str(paths["vhdr"]), preload=True, verbose="ERROR")
        raw = pick_muse4(raw, DS7169_SRC)
        data = zscore_clip(raw.get_data().astype(np.float64))
        sfreq = float(raw.info["sfreq"])
        blocks = arithmetic_blocks(paths["events"])
        # merge contiguous same-label trials into spans for denser windows
        merged: List[Dict[str, Any]] = []
        for b in sorted(blocks, key=lambda x: x["start"]):
            if merged and merged[-1]["label"] == b["label"] and b["start"] - merged[-1]["end"] < 1.0:
                merged[-1]["end"] = b["end"]
                merged[-1]["n_trials"] += 1
                merged[-1]["duration_sec"] = merged[-1]["end"] - merged[-1]["start"]
            else:
                merged.append(dict(b))
        Xs, ys, starts_all, block_meta = [], [], [], []
        for b in merged:
            X, y, starts = window_span(
                data, sfreq, b["start"], b["end"], b["label"], edge_trim=0.5, max_windows=40
            )
            if len(y) == 0:
                continue
            Xs.append(X)
            ys.append(y)
            starts_all.append(starts)
            block_meta.append({**b, "n_windows": int(len(y))})
        if not Xs:
            print(f"[warn] {subject_key} no windows", flush=True)
            continue
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        starts = np.concatenate(starts_all)
        if not both_classes(y):
            print(f"[warn] {subject_key} missing class counts={Counter(y.tolist())}", flush=True)
            continue
        meta = {
            "source": "ds007262",
            "dataset_tag": "Cognitive Workload 8-level arithmetic / ds007262 (19-ch 10–20 mobile EEG, professional)",
            "license_spdx": "CC0-1.0",
            "device_class": "professional",
            "ch_count_source": 19,
            "src_channels": list(DS7169_SRC),
            "channel_map": dict(zip(DS7169_SRC, MUSE_CHS)),
            "unique_person_id": f"barras_{sub}",
            "original_subject": sub,
            "label_rule": "difficulty 0.6-1.5→low; 5.1-6.9→high (drop mid); main phase only",
            "order_confound": "difficulty randomized across trials (better than ds007169)",
            "scale_note": "per_channel_zscore_clip15_nonphysical_brainvision_floats",
            "same_cohort_as": "ds007169",
            "blocks": block_meta,
            "research_only": False,
        }
        payload = save_pack(out_dir, subject_key, X, y, starts, meta)
        results.append(payload)
        print(f"[ok] {subject_key} n={len(y)} {payload['counts']}", flush=True)
    return results


# ---------------------------------------------------------------------------
# EEGMAT
# ---------------------------------------------------------------------------

EEGMAT_DIR = CACHE / "eegmat"
EEGMAT_SRC = ["F7", "F8", "T3", "T4"]


def eegmat_subjects() -> List[str]:
    edfs = sorted(EEGMAT_DIR.glob("Subject*_1.edf"))
    return [p.name.replace("_1.edf", "") for p in edfs]


def ingest_eegmat(force: bool = False) -> List[Dict[str, Any]]:
    out_dir = WINDOWS_ROOT / "eegmat"
    results = []
    for sub in eegmat_subjects():
        subject_key = f"eegmat_{sub}"
        man = out_dir / f"{subject_key}_aeng_manifest.json"
        if man.exists() and not force:
            results.append(json.loads(man.read_text()))
            print(f"[skip] {subject_key}", flush=True)
            continue
        rest_p = EEGMAT_DIR / f"{sub}_1.edf"
        task_p = EEGMAT_DIR / f"{sub}_2.edf"
        if not rest_p.exists() or not task_p.exists():
            print(f"[warn] {subject_key} missing edf", flush=True)
            continue
        Xs, ys, starts_all, block_meta = [], [], [], []
        offset = 0
        for path, label, cond in (
            (rest_p, "low_engagement", "rest"),
            (task_p, "high_engagement", "arithmetic"),
        ):
            raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
            raw = pick_muse4(raw, EEGMAT_SRC)
            data = zscore_clip(raw.get_data().astype(np.float64))
            sfreq = float(raw.info["sfreq"])
            # use last 60s of rest (files may be longer); full task (~60s)
            dur = data.shape[1] / sfreq
            start = max(0.0, dur - 60.0) if cond == "rest" else 0.0
            end = dur
            X, y, starts = window_span(
                data, sfreq, start, end, label, edge_trim=2.0, max_windows=50
            )
            if len(y) == 0:
                continue
            starts = starts + offset
            offset += data.shape[1]
            Xs.append(X)
            ys.append(y)
            starts_all.append(starts)
            block_meta.append(
                {
                    "condition": cond,
                    "label": label,
                    "file": path.name,
                    "start": start,
                    "end": end,
                    "n_windows": int(len(y)),
                }
            )
        if not Xs:
            continue
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        starts = np.concatenate(starts_all)
        if not both_classes(y):
            print(f"[warn] {subject_key} missing class", flush=True)
            continue
        meta = {
            "source": "eegmat",
            "dataset_tag": "PhysioNet EEG During Mental Arithmetic Tasks / eegmat (23-ch 10–20 professional)",
            "license_spdx": "ODC-By-1.0",
            "device_class": "professional",
            "ch_count_source": 23,
            "src_channels": list(EEGMAT_SRC),
            "channel_map": dict(zip(EEGMAT_SRC, MUSE_CHS)),
            "unique_person_id": f"eegmat_{sub}",
            "original_subject": sub,
            "label_rule": "SubjectXX_1 rest→low_engagement; SubjectXX_2 arithmetic→high_engagement",
            "order_confound": "rest always precedes arithmetic",
            "scale_note": "per_channel_zscore_clip15",
            "blocks": block_meta,
            "research_only": False,
        }
        payload = save_pack(out_dir, subject_key, X, y, starts, meta)
        results.append(payload)
        print(f"[ok] {subject_key} n={len(y)} {payload['counts']}", flush=True)
    return results


# ---------------------------------------------------------------------------
# STEW (HF processed — CC-BY-4.0; raw IEEE still gated)
# ---------------------------------------------------------------------------

STEW_DIR = CACHE / "STEW"
# Emotiv EPOC 14-ch order typically used in STEW literature
STEW_CHS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]
STEW_PICK = ["AF3", "AF4", "T7", "T8"]  # → AF7, AF8, TP9, TP10
STEW_MAP = {"AF3": "AF7", "AF4": "AF8", "T7": "TP9", "T8": "TP10"}


def ingest_stew(force: bool = False) -> List[Dict[str, Any]]:
    x_path = STEW_DIR / "STEW_X.npy"
    y_path = STEW_DIR / "STEW_y.npy"
    sid_path = STEW_DIR / "STEW_subject_id.csv"
    if not x_path.exists() or not y_path.exists():
        print("[block] STEW_X/y.npy missing — HF download not ready; skip STEW", flush=True)
        return []
    out_dir = WINDOWS_ROOT / "stew"
    # load arrays (mmap X)
    print("[stew] loading arrays…", flush=True)
    X_all = np.load(x_path, mmap_mode="r")  # (N, 14, 256) expected
    y_all = np.load(y_path)
    # subject ids — CSV may lack header
    sid_raw = np.loadtxt(sid_path, delimiter=",", dtype=str)
    if sid_raw.ndim > 1:
        sid_raw = sid_raw[:, 0]
    # first row may be header-like leftover from bad parse; align lengths
    if len(sid_raw) == len(y_all) + 1:
        sid_raw = sid_raw[1:]
    if len(sid_raw) != len(y_all):
        # try pandas-less: read without assuming header
        with sid_path.open() as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines and not lines[0].replace(",", "").replace("-", "").isdigit():
            # has header
            lines = lines[1:]
        sid_raw = np.asarray([ln.split(",")[0] for ln in lines])
    if len(y_all) == len(X_all) - 0 and len(sid_raw) == len(y_all) - 0:
        pass
    # y may be length N or N-1 from csv quirks
    n = min(len(X_all), len(y_all), len(sid_raw))
    X_all = X_all[:n]
    y_all = y_all[:n]
    sid_raw = sid_raw[:n]
    print(f"[stew] n={n} shape={getattr(X_all, 'shape', None)} unique_sid={len(set(sid_raw.tolist()))}", flush=True)

    pick_idx = [STEW_CHS.index(c) for c in STEW_PICK]
    results = []
    by_sub: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(sid_raw):
        by_sub[str(s)].append(i)

    for sub, idxs in sorted(by_sub.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0]):
        subject_key = f"stew_sub-{int(sub):03d}" if str(sub).isdigit() else f"stew_{sub}"
        man = out_dir / f"{subject_key}_aeng_manifest.json"
        if man.exists() and not force:
            results.append(json.loads(man.read_text()))
            print(f"[skip] {subject_key}", flush=True)
            continue
        idxs = np.asarray(idxs, dtype=int)
        # cap windows per class for balance / size
        y_sub = y_all[idxs].astype(int)
        # STEW HF: 0=low, 1=high (rating ≤4 vs >4)
        low_ix = idxs[y_sub == 0]
        high_ix = idxs[y_sub == 1]
        rng = np.random.default_rng(SEED + (int(sub) if str(sub).isdigit() else 0))
        max_per = 80
        if len(low_ix) > max_per:
            low_ix = rng.choice(low_ix, size=max_per, replace=False)
        if len(high_ix) > max_per:
            high_ix = rng.choice(high_ix, size=max_per, replace=False)
        if len(low_ix) == 0 or len(high_ix) == 0:
            print(f"[warn] {subject_key} missing class L={len(low_ix)} H={len(high_ix)}", flush=True)
            continue
        keep = np.concatenate([low_ix, high_ix])
        keep.sort()
        X14 = np.asarray(X_all[keep], dtype=np.float32)  # (n, 14, 256)
        # resample 128→256 by linear upsample along time
        X4 = X14[:, pick_idx, :]  # (n, 4, 256_at_128Hz) — length 256 @ 128Hz = 2s
        # upsample time 256→512 then we want 2s@256Hz=512 samples; STEW already 2s@128=256
        t_old = np.linspace(0, 1, X4.shape[-1], endpoint=False)
        t_new = np.linspace(0, 1, int(WINDOW_SEC * TARGET_SR), endpoint=False)
        X_up = np.stack(
            [np.stack([np.interp(t_new, t_old, X4[i, c]) for c in range(4)]) for i in range(len(X4))],
            axis=0,
        ).astype(np.float32)
        # per-window zscore
        for i in range(len(X_up)):
            X_up[i] = zscore_clip(X_up[i].astype(np.float64))
        y = np.asarray(y_all[keep], dtype=np.int64)
        # map: ensure 0=low 1=high matching HEAD labels
        starts = np.arange(len(y), dtype=np.int64)
        meta = {
            "source": "stew",
            "dataset_tag": "STEW Simultaneous Task EEG Workload (Emotiv EPOC 14-ch, hobbyist) — HF processed MONSTER mirror",
            "license_spdx": "CC-BY-4.0",
            "device_class": "hobbyist",
            "ch_count_source": 14,
            "src_channels": list(STEW_PICK),
            "channel_map": dict(STEW_MAP),
            "unique_person_id": f"stew_{sub}",
            "original_subject": str(sub),
            "label_rule": "HF binary: rating≤4→low_engagement(0); rating>4→high_engagement(1)",
            "order_confound": "rest then SIMKAP within subject; windows pre-segmented 2s@128Hz",
            "scale_note": "per_window_zscore_clip15; upsampled 128→256 Hz",
            "raw_access": "IEEE DataPort login for raw; used HF processed CC-BY-4.0 mirror",
            "research_only": False,  # CC-BY ship-OK with attribution; note raw gated
            "n_windows_source_before_cap": int(len(idxs)),
        }
        payload = save_pack(out_dir, subject_key, X_up, y, starts, meta)
        results.append(payload)
        print(f"[ok] {subject_key} n={len(y)} {payload['counts']}", flush=True)
    return results


# ---------------------------------------------------------------------------
# ds007554 — PassiveMotor low vs NbackArithmetic high
# ---------------------------------------------------------------------------

DS7554 = "ds007554"
DS7554_SRC = ["F7", "F8", "T7", "T8"]  # T7/T8 present (not T3/T4)
DS7554_ALIASES = {"T7": ("T3",), "T8": ("T4",)}
DS7554_LOW_TASK = "passivemotor"
DS7554_HIGH_TASK = "nbackarithmetic"


def ensure_ds007554_meta() -> None:
    s3 = f"https://s3.amazonaws.com/openneuro.org/{DS7554}"
    for name in (
        "README",
        "participants.tsv",
        "dataset_description.json",
        "phenotype/cognitive_load.tsv",
    ):
        dest = CACHE / DS7554 / name
        if not dest.exists():
            pause(f"meta {DS7554}/{name}")
            try:
                download(f"{s3}/{name}", dest)
            except Exception as e:
                print(f"[warn] {name}: {e}", flush=True)


def ds007554_subjects() -> List[str]:
    ensure_ds007554_meta()
    part = CACHE / DS7554 / "participants.tsv"
    rows = list(csv.DictReader(part.open(), delimiter="\t"))
    return [r["participant_id"] for r in rows]


def download_ds007554_task(sub: str, ses: str, task: str) -> Optional[Path]:
    """Download one EDF + events for subject/session/task. Returns edf path."""
    prefix = f"{DS7554}/{sub}/{ses}/eeg/"
    files = list_s3(prefix)
    edf = None
    for key, size in files:
        fname = key.split("/")[-1]
        if f"task-{task}_eeg.edf" in fname:
            dest = CACHE / DS7554 / "raw" / sub / ses / "eeg" / fname
            if download(
                f"https://s3.amazonaws.com/openneuro.org/{key}", dest, expected_size=size
            ):
                pause(f"dl {fname}", 0.5)
            edf = dest
        elif f"task-{task}_events.tsv" in fname:
            dest = CACHE / DS7554 / "raw" / sub / ses / "eeg" / fname
            download(
                f"https://s3.amazonaws.com/openneuro.org/{key}", dest, expected_size=size
            )
        elif f"task-{task}_channels.tsv" in fname:
            dest = CACHE / DS7554 / "raw" / sub / ses / "eeg" / fname
            download(
                f"https://s3.amazonaws.com/openneuro.org/{key}", dest, expected_size=size
            )
    return edf


def _ds7554_available_tasks(sub: str, ses: str) -> List[str]:
    prefix = f"{DS7554}/{sub}/{ses}/eeg/"
    try:
        files = list_s3(prefix)
    except Exception as e:
        print(f"[warn] list {sub}/{ses}: {e}", flush=True)
        return []
    tasks = []
    for key, _size in files:
        fname = key.split("/")[-1]
        if "_task-" in fname and fname.endswith("_eeg.edf"):
            # sub-XXX_ses-YY_task-NAME_eeg.edf
            mid = fname.split("_task-", 1)[1]
            task = mid[: -len("_eeg.edf")]
            tasks.append(task)
    return sorted(set(tasks))


def ingest_ds007554(
    force: bool = False,
    sessions: Sequence[str] = ("ses-01", "ses-02", "ses-03"),
) -> List[Dict[str, Any]]:
    """Prefer PassiveMotor vs NbackArithmetic; fall back to motorimagery / nback / mentalarithmetic."""
    low_pref = ("passivemotor", "motorimagery")
    high_pref = ("nbackarithmetic", "mentalarithmetic", "nback")
    out_dir = WINDOWS_ROOT / "ds007554"
    results = []
    for sub in ds007554_subjects():
        subject_key = f"ds007554_{sub}"
        man = out_dir / f"{subject_key}_aeng_manifest.json"
        if man.exists() and not force:
            results.append(json.loads(man.read_text()))
            print(f"[skip] {subject_key}", flush=True)
            continue
        Xs, ys, starts_all, block_meta = [], [], [], []
        offset = 0
        used_sessions = []
        got_low = got_high = False
        for ses in sessions:
            if got_low and got_high:
                break
            pause(f"list {sub}/{ses}")
            avail = _ds7554_available_tasks(sub, ses)
            if not avail:
                continue
            picks: List[Tuple[str, str]] = []
            if not got_low:
                for t in low_pref:
                    if t in avail:
                        picks.append((t, "low_engagement"))
                        break
            if not got_high:
                for t in high_pref:
                    if t in avail:
                        picks.append((t, "high_engagement"))
                        break
            if not picks:
                continue
            for task, label in picks:
                edf = download_ds007554_task(sub, ses, task)
                if edf is None or not edf.exists():
                    print(f"[warn] missing {sub}/{ses}/{task}", flush=True)
                    continue
                raw = mne.io.read_raw_edf(str(edf), preload=True, verbose="ERROR")
                try:
                    raw = pick_muse4(raw, DS7554_SRC, aliases=DS7554_ALIASES)
                except KeyError as e:
                    print(f"[warn] {sub} ch: {e}", flush=True)
                    continue
                data = zscore_clip(raw.get_data().astype(np.float64))
                sfreq = float(raw.info["sfreq"])
                dur = data.shape[1] / sfreq
                start = 25.0 if dur > 40 else 0.0
                X, y, starts = window_span(
                    data, sfreq, start, dur, label, edge_trim=2.0, max_windows=60
                )
                if len(y) == 0:
                    continue
                starts = starts + offset
                offset += data.shape[1]
                Xs.append(X)
                ys.append(y)
                starts_all.append(starts)
                block_meta.append(
                    {
                        "session": ses,
                        "task": task,
                        "label": label,
                        "n_windows": int(len(y)),
                        "duration_sec": dur,
                    }
                )
                used_sessions.append(ses)
                if label == "low_engagement":
                    got_low = True
                else:
                    got_high = True
        if not Xs:
            print(f"[warn] {subject_key} no windows", flush=True)
            continue
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        starts = np.concatenate(starts_all)
        if not both_classes(y):
            print(f"[warn] {subject_key} missing class {Counter(y.tolist())}", flush=True)
            continue
        meta = {
            "source": "ds007554",
            "dataset_tag": "CMx7-MM hierarchical cognitive-motor / ds007554 (32-ch EEG, professional)",
            "license_spdx": "CC0-1.0",
            "device_class": "professional",
            "ch_count_source": 32,
            "src_channels": list(DS7554_SRC),
            "channel_map": dict(zip(DS7554_SRC, MUSE_CHS)),
            "unique_person_id": f"ds007554_{sub}",
            "original_subject": sub,
            "label_rule": (
                "low: passivemotor|motorimagery; high: nbackarithmetic|mentalarithmetic|nback "
                "(prefer extremes; CL ratings typically agree)"
            ),
            "order_confound": "task order within session not fully counterbalanced; multi-session fallback",
            "scale_note": "per_channel_zscore_clip15",
            "sessions_used": sorted(set(used_sessions)),
            "blocks": block_meta,
            "research_only": False,
        }
        payload = save_pack(out_dir, subject_key, X, y, starts, meta)
        results.append(payload)
        print(f"[ok] {subject_key} n={len(y)} {payload['counts']}", flush=True)
    return results


# ---------------------------------------------------------------------------
# finalize splits + docs
# ---------------------------------------------------------------------------


def discover_all() -> List[Dict[str, Any]]:
    items = []
    for man in sorted(WINDOWS_ROOT.glob("*/*_aeng_manifest.json")):
        m = json.loads(man.read_text())
        items.append(m)
    return items


def write_splits(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Subject-wise splits on unique_person_id (Barras shared across tasks)."""
    by_person: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_person[it["unique_person_id"]].append(it)
    persons = sorted(by_person.keys())
    rng = np.random.default_rng(SEED)
    order = list(persons)
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, int(round(0.15 * n)))
    n_val = max(1, int(round(0.15 * n)))
    test = sorted(order[:n_test])
    val = sorted(order[n_test : n_test + n_val])
    train = sorted(order[n_test + n_val :])
    CORPUS.mkdir(parents=True, exist_ok=True)
    (CORPUS / "windows").mkdir(exist_ok=True)
    (CORPUS / "splits").mkdir(exist_ok=True)
    # symlink windows
    for it in items:
        src = ROOT / it["npz_path"]
        dst = CORPUS / "windows" / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        man_src = src.with_name(src.name.replace("_windows.npz", "_manifest.json"))
        # actual manifest name pattern
        man_src = src.parent / src.name.replace("_aeng_windows.npz", "_aeng_manifest.json")
        man_dst = CORPUS / "windows" / man_src.name
        if man_dst.exists() or man_dst.is_symlink():
            man_dst.unlink()
        if man_src.exists():
            man_dst.symlink_to(man_src.resolve())
    split_obj = {
        "corpus": "engagement_a_eng",
        "policy": "fixed_subject_json",
        "policy_note": (
            "Subject-wise on unique_person_id. Barras ds007169+ds007262 share person IDs "
            "so both tasks stay in the same split (no identity leakage)."
        ),
        "created_utc": utc_now(),
        "seed": SEED,
        "splits": {"train": train, "val": val, "test": test},
        "n_unique_persons": n,
        "n_window_packs": len(items),
        "persons": {
            p: {
                "packs": [i["subject_key"] for i in by_person[p]],
                "sources": sorted({i["source"] for i in by_person[p]}),
            }
            for p in persons
        },
    }
    split_path = CORPUS / "splits" / "splits.json"
    split_path.write_text(json.dumps(split_obj, indent=2) + "\n")
    return split_obj


def summarize(items: List[Dict[str, Any]], split_obj: Dict[str, Any]) -> Dict[str, Any]:
    by_source: Dict[str, Any] = {}
    persons = set()
    for it in items:
        persons.add(it["unique_person_id"])
        s = it["source"]
        d = by_source.setdefault(
            s,
            {
                "n_packs": 0,
                "n_windows": 0,
                "persons": set(),
                "license_spdx": it.get("license_spdx"),
                "device_class": it.get("device_class"),
                "dataset_tag": it.get("dataset_tag"),
                "research_only": it.get("research_only", False),
                "order_confound": it.get("order_confound"),
            },
        )
        d["n_packs"] += 1
        d["n_windows"] += int(it.get("n_windows") or 0)
        d["persons"].add(it["unique_person_id"])
    for s, d in by_source.items():
        d["persons"] = sorted(d["persons"])
        d["n_persons"] = len(d["persons"])
    summary = {
        "step": "expand_head_a_eng_corpus",
        "completed_utc": utc_now(),
        "target_unique_subjects": 120,
        "unique_persons_reached": len(persons),
        "remaining": max(0, 120 - len(persons)),
        "n_window_packs": len(items),
        "total_windows": sum(int(i.get("n_windows") or 0) for i in items),
        "by_source": by_source,
        "splits": {
            "train": len(split_obj["splits"]["train"]),
            "val": len(split_obj["splits"]["val"]),
            "test": len(split_obj["splits"]["test"]),
        },
        "blockers": [],
        "montage": "muse4_only",
    }
    return summary


def write_doc(summary: Dict[str, Any]) -> None:
    lines = [
        "# Head A-eng corpus expansion (~120 unique subjects)",
        "",
        f"**Status:** {'DONE enough for fit/misfit' if summary['unique_persons_reached'] >= 100 else 'IN PROGRESS'} — "
        f"**{summary['unique_persons_reached']}** unique persons / target ~120",
        f"**Date:** {summary['completed_utc']}",
        f"**Windows:** {summary['total_windows']} across {summary['n_window_packs']} packs",
        f"**Goal:** grow labeled low vs high engagement windows toward **~120 unique subjects** "
        "for A-eng fit vs misfit (no full REVE/CBraMod train required in this step).",
        "",
        "## ALLOW sources ingested",
        "",
        "| Source | License | Persons | Packs | Windows | Device | Order / confound |",
        "|--------|---------|--------:|------:|--------:|--------|------------------|",
    ]
    for src, d in sorted(summary["by_source"].items()):
        lines.append(
            f"| `{src}` | {d.get('license_spdx')} | {d['n_persons']} | {d['n_packs']} | "
            f"{d['n_windows']} | {d.get('device_class')} (ch tag in manifest) | {d.get('order_confound','')} |"
        )
    lines += [
        "",
        "### Dataset tags (ch, professional|hobbyist)",
        "",
    ]
    for src, d in sorted(summary["by_source"].items()):
        lines.append(f"- **{src}:** {d.get('dataset_tag')}")
    lines += [
        "",
        "## Label mapping (honest)",
        "",
        "| Source | low_engagement | high_engagement |",
        "|--------|----------------|-----------------|",
        "| ds007169 | main 1-back | main 4-back |",
        "| ds007262 | difficulty 0.6–1.5 | difficulty 5.1–6.9 |",
        "| eegmat | rest (`_1`) | mental arithmetic (`_2`) |",
        "| stew | rating ≤4 (HF y=0) | rating >4 (HF y=1) |",
        "| ds007554 | PassiveMotor | NbackArithmetic |",
        "",
        "## Unique-person policy",
        "",
        "- Barras **ds007169** and **ds007262** share participant IDs/demographics → "
        "`unique_person_id = barras_{sub}`; both task packs stay in the **same** split.",
        "- Other corpora use `{source}_{id}`.",
        "- Splits: subject-wise ~70/15/15 on unique_person_id (`datasets/engagement_a_eng/splits/splits.json`).",
        "",
        "## Montage",
        "",
        "- **muse4 only** — never mixed with crown8.",
        "- Proxies: F7/F8/T3|T7/T4|T8 or Emotiv AF3/AF4/T7/T8 → AF7/AF8/TP9/TP10.",
        "- 2 s windows / 1 s hop @ 256 Hz; per-channel (or per-window) z-score + clip±15.",
        "",
        "## Blockers / notes",
        "",
        "- **STEW raw** remains IEEE DataPort login-walled; this corpus uses the "
        "**HF monster-monash/STEW processed** CC-BY-4.0 mirror (2 s @ 128 Hz windows).",
        "- **UNIVERSE** Muse-S (CC-BY-4.0, n=24) is ~20 GB — deferred (size); would top up if needed.",
        "- ds007169 order confound (L1→L4) remains; ds007262 randomized difficulty is the cleaner Barras signal.",
        "- EEGMAT rest→arith is ordered (time confound).",
        "",
        "## Scripts / paths",
        "",
        "| Path | Role |",
        "|------|------|",
        "| `scripts/expand_head_a_eng_corpus.py` | download + window + splits |",
        "| `exports/windows_aeng/{source}/` | per-source npz + manifests |",
        "| `datasets/engagement_a_eng/` | corpus symlinks + splits |",
        "| `exports/head_a_eng_expand/` | progress / summary JSON |",
        "",
        "## Next (out of scope here)",
        "",
        "Optional cheap CBraMod chance-check on expanded set; full A-eng train for fit vs misfit.",
        "",
        "## Final counts",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Unique persons | **{summary['unique_persons_reached']}** (target ~120) |",
        f"| Window packs | {summary['n_window_packs']} |",
        f"| Windows | {summary['total_windows']} |",
        f"| Splits | train {summary['splits']['train']} / val {summary['splits']['val']} / test {summary['splits']['test']} |",
        "",
    ]
    if summary.get("blockers"):
        lines.append("**Blockers:**")
        for b in summary["blockers"]:
            lines.append(f"- {b}")
        lines.append("")
    DOC_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC_PATH}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--sources",
        default="ds007169,ds007262,eegmat,stew,ds007554",
        help="comma list",
    )
    ap.add_argument("--skip-finalize", action="store_true")
    args = ap.parse_args()
    PROGRESS.mkdir(parents=True, exist_ok=True)
    WINDOWS_ROOT.mkdir(parents=True, exist_ok=True)
    wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
    all_items: List[Dict[str, Any]] = []
    blockers: List[str] = []

    if "ds007169" in wanted:
        print("=== ds007169 ===", flush=True)
        all_items.extend(ingest_ds007169(force=args.force))
    if "ds007262" in wanted:
        print("=== ds007262 ===", flush=True)
        all_items.extend(ingest_ds007262(force=args.force))
    if "eegmat" in wanted:
        print("=== eegmat ===", flush=True)
        all_items.extend(ingest_eegmat(force=args.force))
    if "stew" in wanted:
        print("=== stew ===", flush=True)
        stew_items = ingest_stew(force=args.force)
        if not stew_items and not list((WINDOWS_ROOT / "stew").glob("*_manifest.json")):
            blockers.append(
                "STEW: HF processed arrays not on disk yet (raw IEEE still gated); retry after STEW_X.npy download"
            )
        all_items.extend(stew_items)
    if "ds007554" in wanted:
        print("=== ds007554 ===", flush=True)
        all_items.extend(ingest_ds007554(force=args.force))

    # re-discover in case of skips / prior
    if not args.skip_finalize:
        items = discover_all()
        split_obj = write_splits(items)
        summary = summarize(items, split_obj)
        summary["blockers"] = blockers
        (PROGRESS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_doc(summary)
        print(json.dumps({k: summary[k] for k in (
            "unique_persons_reached", "remaining", "n_window_packs", "total_windows",
            "splits", "blockers",
        )}, indent=2))
        print("by_source persons:", {s: d["n_persons"] for s, d in summary["by_source"].items()})


if __name__ == "__main__":
    main()

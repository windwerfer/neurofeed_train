#!/usr/bin/env python3
"""Wave2 step ds001787_attention_ingest: Muse-proxy attention windows from OpenNeuro ds001787."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.head_a import ATTENTION_LABELS, labels_to_ids  # noqa: E402
from src.windowing import stack_windows  # noqa: E402

SEED = 42
WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0
EPOCH_PRE = 10.05  # seconds before Q1 onset
EPOCH_POST = 0.05  # end just before Q1
VALUE_TO_RATING = {1: 0, 2: 1, 4: 2, 8: 3}
# BioSemi64 has AF7/AF8 + P9/P10 (no TP9/TP10); P9/P10 ≈ Muse mastoid TP9/TP10
MUSE_PREFERRED = ["AF7", "AF8", "P9", "P10"]
MUSE_FALLBACK = ["AF7", "AF8", "TP7", "TP8"]
MUSE_OUTPUT_NAMES = ["AF7", "AF8", "TP9", "TP10"]  # export under Muse names

# 2 expert + 2 novice, ses-01 only (focused bootstrap)
SUBJECTS: List[Dict[str, str]] = [
    {"sub": "001", "ses": "01", "group": "expert", "log_candidates": ["sub01_info.txt", "sub01_2_info.txt"]},
    {"sub": "002", "ses": "01", "group": "expert", "log_candidates": ["sub02__info.txt", "sub02_2_info.txt"]},
    {"sub": "013", "ses": "01", "group": "novice", "log_candidates": ["sub13_info.txt", "sub13_2_info.txt"]},
    {"sub": "017", "ses": "01", "group": "novice", "log_candidates": ["sub17_info.txt", "sub17_2_info.txt"]},
]

CACHE = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data/ds001787"
RAW = CACHE / "raw"
LOG_DIR = CACHE / "code/MW_Current_TextFileBIDS"
WINDOWS_PKG = ROOT / "kaggle_datasets/muse-eeg-heads-windows"
EXPORT = ROOT / "exports/windows_ds001787"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_behavioral_log(path: Path) -> List[Dict[str, Any]]:
    """Parse MW_Current_TextFileBIDS info.txt → probe ratings (Q1/Q2/Q3)."""
    text = path.read_text(errors="replace")
    probes: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    ask_re = re.compile(r"MW question asked at time\s+([0-9.]+)")
    key_re = re.compile(
        r"Key\s+(\d+)\s+\(code\s+(\d+)\)\s+pressed at time\s+([0-9.]+)\s+seconds,\s+status\s+(\d+)"
    )
    for line in text.splitlines():
        m = ask_re.search(line)
        if m:
            if current is not None:
                probes.append(current)
            current = {
                "log_time": float(m.group(1)),
                "q1": None,
                "q2": None,
                "q3": None,
            }
            continue
        m = key_re.search(line)
        if m and current is not None:
            rating = int(m.group(2))  # code is 0..3
            status = int(m.group(4))
            if status == 1:
                current["q1"] = rating
            elif status == 2:
                current["q2"] = rating
            elif status == 3:
                current["q3"] = rating
    if current is not None:
        probes.append(current)
    return probes


def label_from_q1_q2(q1: Optional[int], q2: Optional[int]) -> Optional[str]:
    if q1 is None or q2 is None:
        return None
    if q1 > q2:
        return "concentration"
    if q1 < q2:
        return "mind_wandering"
    return None  # tie drop


def parse_events_probes(events_path: Path) -> List[Dict[str, Any]]:
    """Ordered responses after each value=128 → Q1/Q2/Q3 ratings."""
    df = pd.read_csv(events_path, sep="\t")
    probes: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    resp_i = 0
    for _, row in df.iterrows():
        val = int(row["value"]) if pd.notna(row["value"]) else None
        onset = float(row["onset"])
        sample = int(float(row["sample"])) if pd.notna(row["sample"]) else int(round(onset * TARGET_SR))
        if val == 128:
            if current is not None:
                probes.append(current)
            current = {
                "q1_onset_sec": onset,
                "q1_onset_sample": sample,
                "q1": None,
                "q2": None,
                "q3": None,
                "source": "events",
            }
            resp_i = 0
            continue
        if current is None:
            continue
        if val in VALUE_TO_RATING:
            rating = VALUE_TO_RATING[val]
            resp_i += 1
            if resp_i == 1:
                current["q1"] = rating
            elif resp_i == 2:
                current["q2"] = rating
            elif resp_i == 3:
                current["q3"] = rating
        # value 16 = involuntary; ignore for Q assignment
    if current is not None:
        probes.append(current)
    return probes


def merge_log_ratings(
    event_probes: List[Dict[str, Any]],
    log_probes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    """Prefer log Q1/Q2 when lengths match (order); else keep events."""
    if not log_probes:
        return event_probes, "events_only"
    if len(log_probes) != len(event_probes):
        # try nearest-time match with median offset
        if len(log_probes) < 3 or len(event_probes) < 3:
            return event_probes, "events_only_log_len_mismatch"
        offsets = []
        n = min(len(log_probes), len(event_probes))
        for i in range(n):
            offsets.append(event_probes[i]["q1_onset_sec"] - log_probes[i]["log_time"])
        med = float(np.median(offsets))
        used = set()
        merged = []
        for ep in event_probes:
            best_j, best_d = None, 1e9
            for j, lp in enumerate(log_probes):
                if j in used:
                    continue
                d = abs((lp["log_time"] + med) - ep["q1_onset_sec"])
                if d < best_d:
                    best_d, best_j = d, j
            row = dict(ep)
            if best_j is not None and best_d < 5.0:
                used.add(best_j)
                lp = log_probes[best_j]
                row["q1"] = lp["q1"]
                row["q2"] = lp["q2"]
                row["q3"] = lp["q3"]
                row["source"] = "log_aligned"
                row["align_abs_err_sec"] = best_d
            merged.append(row)
        return merged, "events_plus_log_nearest"
    # equal length: trust log ratings, keep EEG onsets from events
    merged = []
    for ep, lp in zip(event_probes, log_probes):
        row = dict(ep)
        row["q1"] = lp["q1"]
        row["q2"] = lp["q2"]
        row["q3"] = lp["q3"]
        row["source"] = "log_order"
        merged.append(row)
    return merged, "events_onset_log_ratings"


def pick_log_file(candidates: Sequence[str], ses: str) -> Optional[Path]:
    """Map ses-01 → first session log, ses-02 → *_2_info when present."""
    existing = [LOG_DIR / c for c in candidates if (LOG_DIR / c).exists()]
    if not existing:
        return None
    if ses == "01":
        # prefer non-_2 / non-_3
        for p in existing:
            if re.search(r"sub\d+_info\.txt$", p.name) or re.search(r"sub\d+__info\.txt$", p.name):
                return p
        return existing[0]
    if ses == "02":
        for p in existing:
            if "_2_info" in p.name:
                return p
    return existing[0]


def biosemi_ab_to_1020(raw):
    """Rename A1..A32,B1..B32 to biosemi64 10-20 labels in place."""
    import mne

    montage = mne.channels.make_standard_montage("biosemi64")
    ab = [f"A{i}" for i in range(1, 33)] + [f"B{i}" for i in range(1, 33)]
    if all(c in raw.ch_names for c in ab[:8]):
        mapping = {old: new for old, new in zip(ab, montage.ch_names)}
        # only rename present EEG AB channels
        mapping = {k: v for k, v in mapping.items() if k in raw.ch_names}
        raw.rename_channels(mapping)
        raw.set_montage(montage, on_missing="ignore")
    return raw


def resolve_muse_channels(ch_names: Sequence[str]) -> Tuple[List[str], List[int], str, List[str]]:
    norm_map: Dict[str, int] = {}
    for i, c in enumerate(ch_names):
        n = c.strip()
        if n.upper().startswith("EEG"):
            n = n[3:].lstrip("- ").strip()
        norm_map[n.upper()] = i
        norm_map[c.upper()] = i

    for strategy, wanted in (
        ("af7_af8_p9_p10", MUSE_PREFERRED),
        ("af7_af8_tp7_tp8", MUSE_FALLBACK),
        ("exact_tp9_tp10", ["AF7", "AF8", "TP9", "TP10"]),
    ):
        idxs = []
        ok = True
        for w in wanted:
            if w.upper() not in norm_map:
                ok = False
                break
            idxs.append(norm_map[w.upper()])
        if ok:
            return list(wanted), idxs, strategy, list(MUSE_OUTPUT_NAMES)
    raise KeyError(f"Cannot map Muse channels from {list(ch_names)[:24]}...")


def load_muse_proxy(bdf: Path) -> Tuple[np.ndarray, float, List[str], str]:
    import mne

    raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
    # Keep EEG AB + drop EXG/Status etc. after rename
    raw = biosemi_ab_to_1020(raw)
    eeg_names = [c for c in raw.ch_names if c in set(mne.channels.make_standard_montage("biosemi64").ch_names)]
    if eeg_names:
        raw.pick(eeg_names)
    else:
        raw.pick_types(eeg=True, exclude="bads")
    names_src, idxs, strategy, out_names = resolve_muse_channels(raw.ch_names)
    data = raw.get_data(picks=idxs)  # (4, n_times) volts
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - TARGET_SR) > 1e-3:
        raw2 = raw.copy().pick(idxs)
        raw2.resample(TARGET_SR, npad="auto")
        data = raw2.get_data()
        sfreq = TARGET_SR
    # Export under Muse names regardless of source labels
    return data.astype(np.float32), sfreq, out_names, f"{strategy}->{'/'.join(out_names)} via {names_src}"


def extract_windows_for_recording(
    data: np.ndarray,
    sfreq: float,
    probes: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Return X, y, starts, probe_ids + drop counts."""
    win = int(round(WINDOW_SEC * sfreq))
    hop = int(round(HOP_SEC * sfreq))
    n_times = data.shape[1]
    Xs: List[np.ndarray] = []
    ys: List[str] = []
    starts: List[int] = []
    probe_ids: List[int] = []
    drops = Counter()
    for pi, pr in enumerate(probes):
        lab = label_from_q1_q2(pr.get("q1"), pr.get("q2"))
        if lab is None:
            if pr.get("q1") is None or pr.get("q2") is None:
                drops["incomplete"] += 1
            else:
                drops["tie"] += 1
            continue
        onset = float(pr["q1_onset_sec"])
        t0 = onset - EPOCH_PRE
        t1 = onset - EPOCH_POST
        s0 = int(round(t0 * sfreq))
        s1 = int(round(t1 * sfreq))
        if s0 < 0 or s1 > n_times or (s1 - s0) < win:
            drops["oob"] += 1
            continue
        segment = data[:, s0:s1]
        # sliding windows within pre-Q1 epoch
        local_starts = []
        pos = 0
        while pos + win <= segment.shape[1]:
            Xs.append(segment[:, pos : pos + win])
            ys.append(lab)
            starts.append(s0 + pos)
            probe_ids.append(pi)
            local_starts.append(pos)
            pos += hop
        if not local_starts:
            drops["no_window"] += 1
    if not Xs:
        empty = np.zeros((0, 4, win), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), dict(drops)
    X = np.stack(Xs, axis=0).astype(np.float32)
    y = labels_to_ids(ys, ATTENTION_LABELS)
    return X, y, np.asarray(starts, dtype=np.int64), np.asarray(probe_ids, dtype=np.int64), dict(drops)


def process_subject(spec: Dict[str, str]) -> Dict[str, Any]:
    sub, ses = spec["sub"], spec["ses"]
    eeg_dir = RAW / f"sub-{sub}" / f"ses-{ses}" / "eeg"
    bdf = eeg_dir / f"sub-{sub}_ses-{ses}_task-meditation_eeg.bdf"
    events = eeg_dir / f"sub-{sub}_ses-{ses}_task-meditation_events.tsv"
    if not bdf.exists() or not events.exists():
        raise FileNotFoundError(f"missing {bdf} or {events}")

    event_probes = parse_events_probes(events)
    log_path = pick_log_file(spec["log_candidates"], ses)
    log_probes = parse_behavioral_log(log_path) if log_path else []
    probes, merge_mode = merge_log_ratings(event_probes, log_probes)

    data, sfreq, ch_names, ch_strategy = load_muse_proxy(bdf)
    X, y, starts, probe_ids, drops = extract_windows_for_recording(data, sfreq, probes)
    counts = {
        ATTENTION_LABELS[i]: int((y == i).sum()) for i in range(len(ATTENTION_LABELS))
    }
    tag = f"sub{sub}_ses{ses}"
    out_dir = EXPORT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"ds001787_{tag}_attention_windows.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        starts=starts,
        probe_ids=probe_ids,
        label_names=np.asarray(ATTENTION_LABELS),
        channels=np.asarray(ch_names),
    )
    labeled_probes = sum(
        1 for p in probes if label_from_q1_q2(p.get("q1"), p.get("q2")) is not None
    )
    manifest = {
        "dataset": "ds001787",
        "doi": "doi:10.18112/openneuro.ds001787.v1.1.1",
        "license_spdx": "CC0-1.0",
        "subject": f"sub-{sub}",
        "session": f"ses-{ses}",
        "group": spec["group"],
        "tag": tag,
        "bdf": str(bdf),
        "events": str(events),
        "log_file": str(log_path) if log_path else None,
        "merge_mode": merge_mode,
        "n_event_probes": len(event_probes),
        "n_log_probes": len(log_probes),
        "n_labeled_probes": labeled_probes,
        "drops": drops,
        "n_windows_per_label": counts,
        "n_windows_total": int(X.shape[0]),
        "window_shape": list(X.shape),
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "epoch_pre_q1_sec": EPOCH_PRE,
        "epoch_end_before_q1_sec": EPOCH_POST,
        "target_sr": sfreq,
        "channels": ch_names,
        "channel_strategy": ch_strategy,
        "channel_note": "BioSemi64 A/B renamed; export AF7/AF8/TP9/TP10 where TP9/TP10←P9/P10 (fallback TP7/TP8); not native Muse geometry",
        "label_rule": "Q1>Q2→concentration; Q1<Q2→mind_wandering; tie/incomplete drop",
        "hard_reject": "Do NOT map events value 2→concentration / 4→mind_wandering",
        "npz_path": str(npz_path),
        "npz_sha256": sha256(npz_path),
        "bdf_sha256": sha256(bdf),
        "events_sha256": sha256(events),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "step": "ds001787_attention_ingest",
    }
    man_path = out_dir / f"ds001787_{tag}_attention_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # copy into private windows package
    for src in (npz_path, man_path):
        shutil.copy2(src, WINDOWS_PKG / src.name)
    print(f"[{tag}] windows={X.shape} counts={counts} drops={drops} ch={ch_strategy} merge={merge_mode}")
    return {
        "tag": tag,
        "subject": f"sub-{sub}",
        "session": f"ses-{ses}",
        "group": spec["group"],
        "counts": counts,
        "n_windows_total": int(X.shape[0]),
        "n_labeled_probes": labeled_probes,
        "drops": drops,
        "merge_mode": merge_mode,
        "channel_strategy": ch_strategy,
        "channels": ch_names,
        "npz_sha256": manifest["npz_sha256"],
        "bdf_sha256": manifest["bdf_sha256"],
        "npz": str(npz_path),
        "manifest": str(man_path),
    }


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_DIR.exists() or not any(LOG_DIR.glob("*.txt")):
        zpath = CACHE / "code/MW_Current_TextFileBIDS.zip"
        if zpath.exists():
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(CACHE / "code")

    results: Dict[str, Any] = {
        "step": "ds001787_attention_ingest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": [],
        "why": "cache 2 expert + 2 novice Muse-proxy attention windows (pre-Q1) for Head A 4-way bootstrap",
        "doi": "doi:10.18112/openneuro.ds001787.v1.1.1",
        "license_spdx": "CC0-1.0",
    }
    totals = Counter()
    for spec in SUBJECTS:
        info = process_subject(spec)
        results["subjects"].append(info)
        for k, v in info["counts"].items():
            totals[k] += v

    results["totals"] = dict(totals)
    results["n_windows_total"] = int(sum(totals.values()))

    provenance = {
        "step": "ds001787_attention_ingest",
        "source": "https://openneuro.org/datasets/ds001787",
        "doi": "doi:10.18112/openneuro.ds001787.v1.1.1",
        "license_spdx": "CC0-1.0",
        "added_utc": datetime.now(timezone.utc).isoformat(),
        "recordings": [],
    }
    for spec in SUBJECTS:
        sub, ses = spec["sub"], spec["ses"]
        eeg_dir = RAW / f"sub-{sub}" / f"ses-{ses}" / "eeg"
        for name in (
            f"sub-{sub}_ses-{ses}_task-meditation_eeg.bdf",
            f"sub-{sub}_ses-{ses}_task-meditation_events.tsv",
            f"sub-{sub}_ses-{ses}_task-meditation_eeg.json",
        ):
            p = eeg_dir / name
            if not p.exists():
                continue
            provenance["recordings"].append(
                {
                    "name": name,
                    "subject": f"sub-{sub}",
                    "session": f"ses-{ses}",
                    "sha256": sha256(p),
                    "bytes": p.stat().st_size,
                    "url": f"https://s3.amazonaws.com/openneuro.org/ds001787/sub-{sub}/ses-{ses}/eeg/{name}",
                }
            )
    prov_path = CACHE / "provenance_attention_ingest.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    results["provenance_path"] = str(prov_path)

    # update windows package rollup (merge with existing sleep-edf rollup if present)
    rollup_path = WINDOWS_PKG / "manifest.json"
    rollup: Dict[str, Any] = {}
    if rollup_path.exists():
        try:
            rollup = json.loads(rollup_path.read_text())
        except json.JSONDecodeError:
            rollup = {}
    rollup["ds001787_attention"] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": [s["tag"] for s in results["subjects"]],
        "totals": results["totals"],
        "n_windows_total": results["n_windows_total"],
        "recipe": {
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "epoch_pre_q1_sec": EPOCH_PRE,
            "epoch_end_before_q1_sec": EPOCH_POST,
            "target_sr": TARGET_SR,
            "label_rule": "Q1>Q2 concentration; Q1<Q2 mind_wandering",
        },
        "files": [],
    }
    for s in results["subjects"]:
        tag = s["tag"]
        rollup["ds001787_attention"]["files"].extend(
            [
                f"ds001787_{tag}_attention_windows.npz",
                f"ds001787_{tag}_attention_manifest.json",
            ]
        )
        rollup["ds001787_attention"][tag] = {
            "n_windows_per_label": s["counts"],
            "n_windows_total": s["n_windows_total"],
            "npz_sha256": s["npz_sha256"],
            "group": s["group"],
            "channel_strategy": s["channel_strategy"],
        }
    rollup["updated_utc"] = datetime.now(timezone.utc).isoformat()
    rollup_path.write_text(json.dumps(rollup, indent=2) + "\n")

    summary_path = ROOT / "exports/ds001787_attention_ingest_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("totals", "n_windows_total", "why")}, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

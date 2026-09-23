#!/usr/bin/env python3
"""Add HMC sleep-staging subjects (CC-BY-4.0) to Head C corpus.

Muse4 proxy from professional 4-ch EEG:
  AF7=EEG F4-M1, AF8=EEG C4-M1, TP9=EEG C3-M2, TP10=EEG O2-M1
Same N1-slice recipe as Sleep-EDF expand (around first N1, 20/40 min, 2s/0.5s).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edf_io import read_edf_header, read_edf_signals
from src.head_a import HEAD_A_BINARY_LABELS, labels_to_ids
from src.sleep_edf import (
    STAGE_TO_COARSE,
    STAGE_TO_HEAD_A,
    bandpass_fft,
    first_stage_onset,
    hypnogram_to_stage_series,
    majority_stage_in_window,
    resample_poly,
    stage_to_coarse,
    windows_with_head_a_labels,
)

WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0
PRE_SEC = 20 * 60
POST_SEC = 40 * 60
MAJORITY = 0.7

HMC_DIR = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data/hmc-sleep-staging/recordings"
EXPORTS = ROOT / "exports"
CORPUS = ROOT / "datasets/vigilance_sleep_edf"
PROGRESS = EXPORTS / "head_c_expand"

HMC_EEG = ["EEG F4-M1", "EEG C4-M1", "EEG C3-M2", "EEG O2-M1"]
HMC_PROXY_NOTE = (
    "HMC professional PSG (F4-M1/C4-M1/C3-M2/O2-M1 @ 256 Hz) mapped to Muse order "
    "AF7=F4-M1, AF8=C4-M1, TP9=C3-M2, TP10=O2-M1. "
    "License: CC-BY-4.0 PhysioNet hmc-sleep-staging 1.1."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hmc_scoring(txt_path: Path) -> List[Tuple[float, float, str]]:
    """Return (onset_sec, duration_sec, annotation_text) for Sleep stage* rows."""
    events: List[Tuple[float, float, str]] = []
    with txt_path.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        # normalize keys
        for row in reader:
            # handle possible BOM / spacing
            keys = {k.strip(): k for k in row.keys()}
            onset_k = keys.get("Recording onset") or keys.get("Recording onset".lower())
            dur_k = keys.get("Duration")
            ann_k = keys.get("Annotation")
            if not ann_k:
                # fallback positional via values
                vals = list(row.values())
                if len(vals) < 5:
                    continue
                onset, dur, ann = vals[2], vals[3], vals[4]
            else:
                onset = row[onset_k] if onset_k else row.get("Recording onset")
                dur = row[dur_k] if dur_k else row.get("Duration")
                ann = row[ann_k]
            ann = (ann or "").strip()
            if not ann.startswith("Sleep stage"):
                continue
            try:
                onset_f = float(onset)
                dur_f = float(dur) if dur not in (None, "") else 30.0
            except (TypeError, ValueError):
                continue
            events.append((onset_f, dur_f, ann))
    return events


def hmc_to_muse_proxy(eeg_4ch: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """(4,T) F4,C4,C3,O2 → (4,T) AF7,AF8,TP9,TP10."""
    if eeg_4ch.shape[0] != 4:
        raise ValueError(f"expected 4 EEG channels, got {eeg_4ch.shape}")
    # input order matches HMC_EEG
    f4, c4, c3, o2 = eeg_4ch
    out = np.stack([f4, c4, c3, o2], axis=0)
    return out.astype(np.float32), ["AF7", "AF8", "TP9", "TP10"]


def load_hmc_recording(
    psg: Path,
    scoring_txt: Path,
    target_sr: float = TARGET_SR,
    around_stage: str = "stage N1",
    pre_sec: float = PRE_SEC,
    post_sec: float = POST_SEC,
) -> Dict[str, Any]:
    events = parse_hmc_scoring(scoring_txt)
    info = read_edf_header(psg)
    rec_dur = float(info.record_duration)
    onset = first_stage_onset(events, around_stage)
    if onset is None:
        raise ValueError(f"No N1 in {scoring_txt.name}")
    start_sec = max(0.0, onset - pre_sec)
    duration_sec = pre_sec + post_sec
    start_rec = int(start_sec // rec_dur)
    n_rec = int(max(1, round(duration_sec / rec_dur)))
    n_rec = min(n_rec, info.n_records - start_rec)

    eeg, chs, sfreq = read_edf_signals(
        psg,
        labels=HMC_EEG,
        max_records=start_rec + n_rec,
    )
    if start_rec > 0:
        spr = int(round(sfreq * rec_dur))
        eeg = eeg[:, start_rec * spr : (start_rec + n_rec) * spr]
    eeg = bandpass_fft(eeg, sfreq, l_freq=1.0, h_freq=45.0)
    eeg = resample_poly(eeg, sfreq, target_sr)
    x, muse_order = hmc_to_muse_proxy(eeg)
    t0 = start_rec * rec_dur
    shifted = [(o - t0, d, t) for o, d, t in events]
    stages = hypnogram_to_stage_series(shifted, n_times=x.shape[-1], sfreq=target_sr)
    return {
        "data": x,
        "ch_names": muse_order,
        "sfreq": float(target_sr),
        "stages": stages,
        "events": events,
        "slice_start_sec": float(t0),
        "source_channels": list(chs),
        "source_sfreq": float(sfreq),
        "proxy_note": HMC_PROXY_NOTE,
    }


def build_hmc_windows(sid: str) -> Dict[str, Any]:
    psg = HMC_DIR / f"{sid}.edf"
    scoring = HMC_DIR / f"{sid}_sleepscoring.txt"
    if not psg.exists() or not scoring.exists():
        raise FileNotFoundError(f"missing {psg.name} or {scoring.name}")
    tag = sid.lower()
    out_dir = EXPORTS / f"windows_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"hmc_{tag}_n1slice_windows.npz"
    man_path = out_dir / f"hmc_{tag}_n1slice_manifest.json"
    if npz_path.exists() and man_path.exists():
        man = json.loads(man_path.read_text())
        if man.get("head_c_labels"):
            print(f"[{tag}] skip existing n={man.get('n_windows_total')}", flush=True)
            return {"skipped": True, "manifest": man, "npz_path": npz_path, "man_path": man_path, "psg": psg, "hyp": scoring}

    rec = load_hmc_recording(psg, scoring)
    X, labels, keep = windows_with_head_a_labels(
        rec["data"],
        rec["stages"],
        sfreq=TARGET_SR,
        window_sec=WINDOW_SEC,
        hop_sec=HOP_SEC,
        majority_frac=MAJORITY,
    )
    mask = [lab in HEAD_A_BINARY_LABELS for lab in labels]
    X = X[np.asarray(mask)] if len(labels) else X
    labels = [lab for lab, m in zip(labels, mask) if m]
    y = labels_to_ids(labels, HEAD_A_BINARY_LABELS) if labels else np.zeros((0,), dtype=np.int64)
    hop = int(round(HOP_SEC * TARGET_SR))
    starts = np.asarray([keep[i] * hop for i, m in enumerate(mask) if m], dtype=np.int64)
    win = int(round(WINDOW_SEC * TARGET_SR))
    stage_raw = [majority_stage_in_window(rec["stages"], int(st), win) for st in starts]
    stage_coarse = [stage_to_coarse(r) for r in stage_raw]
    counts = dict(Counter(labels))
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y,
        starts=starts,
        label_names=np.asarray(HEAD_A_BINARY_LABELS),
        stage_raw=np.asarray(stage_raw, dtype=object),
        stage_coarse=np.asarray(stage_coarse, dtype=object),
    )
    manifest = {
        "psg_file": psg.name,
        "hypno_file": scoring.name,
        "psg_path": str(psg),
        "hypno_path": str(scoring),
        "tag": tag,
        "recording_id": sid,
        "subject_id": sid,
        "study": "hmc",
        "source": "hmc",
        "night": 1,
        "slice_start_sec": rec["slice_start_sec"],
        "pre_sec": PRE_SEC,
        "post_sec": POST_SEC,
        "around_stage": "stage N1",
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "majority_frac": MAJORITY,
        "target_sr": TARGET_SR,
        "channel_proxy_note": HMC_PROXY_NOTE,
        "montage": {
            "kind": "muse4_proxy",
            "ch_count": 4,
            "device_class": "professional",
            "source_channels": HMC_EEG,
            "mapped_to": ["AF7", "AF8", "TP9", "TP10"],
        },
        "stage_to_label_map": {k: v for k, v in STAGE_TO_HEAD_A.items()},
        "stage_to_coarse_map": dict(STAGE_TO_COARSE),
        "n_windows_per_label": counts,
        "n_windows_total": int(X.shape[0]),
        "window_shape": list(X.shape),
        "stage_raw_counts": dict(Counter(stage_raw)),
        "stage_coarse_counts": dict(Counter(stage_coarse)),
        "head_c_labels": True,
        "npz_path": str(npz_path),
        "npz_sha256": sha256(npz_path),
        "psg_sha256": sha256(psg),
        "hypno_sha256": sha256(scoring),
        "label_list": list(HEAD_A_BINARY_LABELS),
        "created_utc": utc_now(),
        "license_attribution": {
            "HMC": "PhysioNet CC-BY-4.0 — Haaglanden Medisch Centrum sleep staging v1.1",
        },
        "source_url_base": "https://physionet.org/files/hmc-sleep-staging/1.1/",
        "step": "expand_head_c_hmc",
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{tag}] slice={rec['slice_start_sec']} windows={X.shape} counts={counts}", flush=True)
    return {"skipped": False, "manifest": manifest, "npz_path": npz_path, "man_path": man_path, "psg": psg, "hyp": scoring}


def link_into_corpus(sid: str, npz: Path, man: Path, psg: Path, hyp: Path) -> None:
    win_dir = CORPUS / "windows"
    raw_dir = CORPUS / "raw"
    win_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _relink(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    _relink(npz, win_dir / f"{sid}_windows.npz")
    _relink(man, win_dir / f"{sid}_manifest.json")
    _relink(psg, raw_dir / psg.name)
    _relink(hyp, raw_dir / hyp.name)


def list_local_hmc_ids() -> List[str]:
    ids = []
    for p in sorted(HMC_DIR.glob("SN*.edf")):
        if "_sleepscoring" in p.name:
            continue
        sid = p.stem
        if (HMC_DIR / f"{sid}_sleepscoring.txt").exists():
            ids.append(sid)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()
    ids = args.ids or list_local_hmc_ids()
    if args.limit is not None:
        ids = ids[: args.limit]
    PROGRESS.mkdir(parents=True, exist_ok=True)
    results = []
    for i, sid in enumerate(ids, 1):
        print(f"\n=== HMC [{i}/{len(ids)}] {sid} ===", flush=True)
        try:
            built = build_hmc_windows(sid)
            link_into_corpus(sid, built["npz_path"], built["man_path"], built["psg"], built["hyp"])
            results.append({"subject_id": sid, "recording_id": sid, "ok": True, "n_windows": built["manifest"].get("n_windows_total")})
        except Exception as e:
            print(f"[ERROR] {sid}: {type(e).__name__}: {e}", flush=True)
            results.append({"subject_id": sid, "recording_id": sid, "ok": False, "error": f"{type(e).__name__}: {e}"})
    summary = {
        "step": "expand_head_c_hmc",
        "completed_utc": utc_now(),
        "ok": sum(1 for r in results if r.get("ok")),
        "failed": [r for r in results if not r.get("ok")],
        "results": results,
        "montage": "(ch_count=4, professional) muse4 proxy HMC F4/C4/C3/O2",
        "license": "CC-BY-4.0",
    }
    out = EXPORTS / "expand_head_c_hmc_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"ok": summary["ok"], "failed_n": len(summary["failed"])}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

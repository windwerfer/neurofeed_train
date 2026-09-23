#!/usr/bin/env python3
"""Build HMC Head A-vig window packs for crown2_strong and crown4_hmc montages.

Same N1-slice recipe as expand_head_c_hmc.py / Muse HMC path:
  2 s @ 256 Hz, hop 0.5 s, around first N1, pre 20 min / post 40 min, majority 0.7.
  Labels: W→drowsy, N1→hypnagogic.

Montages (true C; no zero-pad to 8):
  crown2_strong: C3, C4 ← EEG C3-M2, EEG C4-M1
  crown4_hmc:    C3, C4, F6, PO4 ← C3-M2, C4-M1, F4-M1 (≈F6), O2-M1 (≈PO4)

Writes:
  datasets/vigilance_hmc_crown2/{windows,splits,README.md,ATTRIBUTION.md}
  datasets/vigilance_hmc_crown4/{windows,splits,README.md,ATTRIBUTION.md}

Subject-wise 70/15/15 splits are identical across both packs (seed=42).
"""
from __future__ import annotations

import os

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.edf_io import read_edf_header, read_edf_signals
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

# Avoid importing src.head_a (pulls torch); labels match Head A-vig binary.
HEAD_A_BINARY_LABELS = ["drowsy", "hypnagogic"]


def labels_to_ids(labels, label_list):
    table = {n: i for i, n in enumerate(label_list)}
    return np.asarray([table[x] for x in labels], dtype=np.int64)


WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0
PRE_SEC = 20 * 60
POST_SEC = 40 * 60
MAJORITY = 0.7
SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 remainder

HMC_DIR = Path(os.environ.get("HMC_RECORDINGS_DIR", str(ROOT / "data/hmc-sleep-staging/recordings")))
EXPORTS = ROOT / "exports" / "hmc_crown_vig"

MONTAGES: Dict[str, Dict[str, Any]] = {
    "crown2_strong": {
        "pack_name": "vigilance_hmc_crown2",
        "kind": "crown2_strong",
        "ch_names": ["C3", "C4"],
        "hmc_labels": ["EEG C3-M2", "EEG C4-M1"],
        "approx_note": "True central pair from HMC PSG; no fabricated channels.",
    },
    "crown4_hmc": {
        "pack_name": "vigilance_hmc_crown4",
        "kind": "crown4_hmc",
        "ch_names": ["C3", "C4", "F6", "PO4"],
        "hmc_labels": ["EEG C3-M2", "EEG C4-M1", "EEG F4-M1", "EEG O2-M1"],
        "approx_note": (
            "C3/C4 exact; F6≈EEG F4-M1 (nearest frontal-right in HMC 4-EEG set); "
            "PO4≈EEG O2-M1 (nearest posterior-right). Honest approximation, not Muse Crown8."
        ),
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hmc_scoring(txt_path: Path) -> List[Tuple[float, float, str]]:
    events: List[Tuple[float, float, str]] = []
    with txt_path.open(newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            keys = {k.strip(): k for k in row.keys()}
            onset_k = keys.get("Recording onset") or keys.get("Recording onset".lower())
            dur_k = keys.get("Duration")
            ann_k = keys.get("Annotation")
            if not ann_k:
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


def list_local_hmc_ids() -> List[str]:
    ids = []
    for p in sorted(HMC_DIR.glob("SN*.edf")):
        if "_sleepscoring" in p.name:
            continue
        sid = p.stem
        if (HMC_DIR / f"{sid}_sleepscoring.txt").exists():
            ids.append(sid)
    return ids


def load_hmc_montage(
    psg: Path,
    scoring_txt: Path,
    hmc_labels: Sequence[str],
    ch_names: Sequence[str],
) -> Dict[str, Any]:
    events = parse_hmc_scoring(scoring_txt)
    info = read_edf_header(psg)
    rec_dur = float(info.record_duration)
    onset = first_stage_onset(events, "stage N1")
    if onset is None:
        raise ValueError(f"No N1 in {scoring_txt.name}")
    start_sec = max(0.0, onset - PRE_SEC)
    duration_sec = PRE_SEC + POST_SEC
    start_rec = int(start_sec // rec_dur)
    n_rec = int(max(1, round(duration_sec / rec_dur)))
    n_rec = min(n_rec, info.n_records - start_rec)

    eeg, chs, sfreq = read_edf_signals(
        psg,
        labels=list(hmc_labels),
        max_records=start_rec + n_rec,
    )
    if eeg.shape[0] != len(hmc_labels):
        raise ValueError(f"expected {len(hmc_labels)} ch, got {eeg.shape[0]} from {chs}")
    if start_rec > 0:
        spr = int(round(sfreq * rec_dur))
        eeg = eeg[:, start_rec * spr : (start_rec + n_rec) * spr]
    eeg = bandpass_fft(eeg, sfreq, l_freq=1.0, h_freq=45.0)
    eeg = resample_poly(eeg, sfreq, TARGET_SR)
    x = eeg.astype(np.float32)
    t0 = start_rec * rec_dur
    shifted = [(o - t0, d, t) for o, d, t in events]
    stages = hypnogram_to_stage_series(shifted, n_times=x.shape[-1], sfreq=TARGET_SR)
    return {
        "data": x,
        "ch_names": list(ch_names),
        "sfreq": float(TARGET_SR),
        "stages": stages,
        "events": events,
        "slice_start_sec": float(t0),
        "source_channels": list(chs),
        "source_sfreq": float(sfreq),
    }


def build_windows_for_sid(sid: str, montage_key: str, force: bool = False) -> Dict[str, Any]:
    cfg = MONTAGES[montage_key]
    pack = ROOT / "datasets" / cfg["pack_name"]
    win_dir = pack / "windows"
    win_dir.mkdir(parents=True, exist_ok=True)
    psg = HMC_DIR / f"{sid}.edf"
    scoring = HMC_DIR / f"{sid}_sleepscoring.txt"
    if not psg.exists() or not scoring.exists():
        raise FileNotFoundError(f"missing {psg.name} or {scoring.name}")

    npz_path = win_dir / f"{sid}_windows.npz"
    man_path = win_dir / f"{sid}_manifest.json"
    if npz_path.exists() and man_path.exists() and not force:
        man = json.loads(man_path.read_text())
        if man.get("montage", {}).get("kind") == cfg["kind"] and man.get("n_windows_total", 0) > 0:
            return {"skipped": True, "manifest": man, "npz_path": npz_path, "man_path": man_path}

    rec = load_hmc_montage(psg, scoring, cfg["hmc_labels"], cfg["ch_names"])
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

    if X.ndim != 3 or X.shape[1] != len(cfg["ch_names"]):
        raise RuntimeError(f"{sid} bad shape {getattr(X, 'shape', None)} expected C={len(cfg['ch_names'])}")

    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y,
        starts=starts,
        label_names=np.asarray(HEAD_A_BINARY_LABELS),
        stage_raw=np.asarray(stage_raw, dtype=object),
        stage_coarse=np.asarray(stage_coarse, dtype=object),
        ch_names=np.asarray(cfg["ch_names"]),
    )
    manifest = {
        "psg_file": psg.name,
        "hypno_file": scoring.name,
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
        "montage": {
            "kind": cfg["kind"],
            "ch_count": len(cfg["ch_names"]),
            "ch_names": list(cfg["ch_names"]),
            "source_channels": list(cfg["hmc_labels"]),
            "mapped_from": list(rec["source_channels"]),
            "approx_note": cfg["approx_note"],
            "device_class": "professional",
        },
        "stage_to_label_map": {k: v for k, v in STAGE_TO_HEAD_A.items()},
        "stage_to_coarse_map": dict(STAGE_TO_COARSE),
        "n_windows_per_label": counts,
        "n_windows_total": int(X.shape[0]),
        "window_shape": list(X.shape),
        "stage_raw_counts": dict(Counter(stage_raw)),
        "stage_coarse_counts": dict(Counter(stage_coarse)),
        "label_list": list(HEAD_A_BINARY_LABELS),
        "npz_path": str(npz_path),
        "npz_sha256": sha256(npz_path),
        "psg_sha256": sha256(psg),
        "hypno_sha256": sha256(scoring),
        "created_utc": utc_now(),
        "license_attribution": {
            "HMC": "PhysioNet CC-BY-4.0 — Haaglanden Medisch Centrum sleep staging v1.1",
        },
        "source_url_base": "https://physionet.org/files/hmc-sleep-staging/1.1/",
        "step": "expand_hmc_crown_vig",
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[{cfg['kind']}|{sid}] shape={X.shape} counts={counts} slice={rec['slice_start_sec']:.1f}s",
        flush=True,
    )
    return {"skipped": False, "manifest": manifest, "npz_path": npz_path, "man_path": man_path}


def make_subject_splits(subject_ids: Sequence[str], seed: int = SEED) -> Dict[str, List[str]]:
    rng = np.random.default_rng(seed)
    ids = sorted(subject_ids)
    perm = rng.permutation(len(ids))
    ordered = [ids[i] for i in perm]
    n = len(ordered)
    n_train = int(round(n * TRAIN_FRAC))
    n_val = int(round(n * VAL_FRAC))
    # ensure at least 1 in val/test when n>=3
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)
    n_test = n - n_train - n_val
    train = sorted(ordered[:n_train])
    val = sorted(ordered[n_train : n_train + n_val])
    test = sorted(ordered[n_train + n_val :])
    assert len(train) + len(val) + len(test) == n
    assert len(set(train) & set(val) & set(test)) == 0 or (
        not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    )
    overlap = (set(train) & set(val)) | (set(train) & set(test)) | (set(val) & set(test))
    if overlap:
        raise RuntimeError(f"split overlap: {overlap}")
    return {"train": train, "val": val, "test": test}


def write_pack_docs(pack: Path, montage_key: str, splits: Dict[str, List[str]], ok_ids: List[str]) -> None:
    cfg = MONTAGES[montage_key]
    attr = (
        "# Attribution\n\n"
        "Haaglanden Medisch Centrum sleep staging dataset (hmc-sleep-staging) v1.1,\n"
        "PhysioNet (https://physionet.org/content/hmc-sleep-staging/1.1/).\n"
        "License: **CC-BY-4.0**.\n\n"
        "Citation: Alvarez-Estevez & Rijsman (or PhysioNet recommended citation for this version).\n"
        "Do not redistribute without license terms; this pack is for private lab use.\n"
    )
    (pack / "ATTRIBUTION.md").write_text(attr)
    readme = f"""# {cfg['pack_name']}

Head A-vig (drowsy vs hypnagogic) N1-slice windows from PhysioNet HMC v1.1.

## Montage: `{cfg['kind']}`
- Channels ({len(cfg['ch_names'])}): {', '.join(cfg['ch_names'])}
- HMC source labels: {', '.join(cfg['hmc_labels'])}
- Note: {cfg['approx_note']}
- **True C={len(cfg['ch_names'])} tensors** — no zero-pad to Crown8.

## Window recipe
- 2.0 s @ 256 Hz, hop 0.5 s
- Slice: first N1 ± (20 min pre / 40 min post)
- Majority stage frac ≥ 0.7; keep only drowsy (W) / hypnagogic (N1)

## Splits
Subject-wise ~70/15/15 (seed={SEED}), shared with sibling crown pack.
- train: {len(splits['train'])} subjects
- val: {len(splits['val'])} subjects
- test: {len(splits['test'])} subjects
- windows built for: {len(ok_ids)} subjects/nights

## License
See ATTRIBUTION.md (CC-BY-4.0). Private lab; do not publish to HF/public GitHub yet.
"""
    (pack / "README.md").write_text(readme)


def write_splits(pack: Path, splits: Dict[str, List[str]], ok_ids: List[str], montage_key: str) -> None:
    cfg = MONTAGES[montage_key]
    split_dir = pack / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for sp, subs in splits.items():
        (split_dir / f"{sp}_subjects.json").write_text(
            json.dumps({"split": sp, "subjects": subs, "n": len(subs)}, indent=2) + "\n"
        )
    recordings = {}
    for sid in ok_ids:
        man_path = pack / "windows" / f"{sid}_manifest.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else {}
        recordings[sid] = {
            "subject_id": sid,
            "recording_id": sid,
            "source": "hmc",
            "n_windows": man.get("n_windows_total"),
            "window_shape": man.get("window_shape"),
        }
    subjects_meta = {
        sid: {"subject_id": sid, "recordings": [sid], "source": "hmc"} for sid in ok_ids
    }
    sid_to_split = {}
    for sp, subs in splits.items():
        for s in subs:
            sid_to_split[s] = sp
    policy = {
        "corpus": cfg["pack_name"],
        "policy": "subject_wise_70_15_15",
        "seed": SEED,
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "created_utc": utc_now(),
        "subject_key_rule": "HMC SN### id == subject_id (1 night / subject)",
        "montage": {
            "kind": cfg["kind"],
            "ch_count": len(cfg["ch_names"]),
            "ch_names": cfg["ch_names"],
            "source_channels": cfg["hmc_labels"],
        },
        "recordings": recordings,
        "subjects": subjects_meta,
        "splits": {sp: {"subjects": splits[sp], "n": len(splits[sp])} for sp in ("train", "val", "test")},
        "counts": {
            "n_subjects_ok": len(ok_ids),
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
        },
        "subject_to_split": sid_to_split,
    }
    (split_dir / "split_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    (split_dir / "subjects.json").write_text(
        json.dumps({"subjects": sorted(ok_ids), "n": len(ok_ids)}, indent=2) + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--montages", nargs="*", default=["crown2_strong", "crown4_hmc"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-missing-n1", action="store_true", default=True)
    args = ap.parse_args()

    ids = args.ids or list_local_hmc_ids()
    if args.limit is not None:
        ids = ids[: args.limit]
    EXPORTS.mkdir(parents=True, exist_ok=True)

    # Build both montages for the same ID list; splits from successfully built IDs
    # Prefer intersection of both montages succeeding.
    per_montage_ok: Dict[str, List[str]] = {m: [] for m in args.montages}
    failures: List[Dict[str, Any]] = []

    for montage_key in args.montages:
        if montage_key not in MONTAGES:
            raise SystemExit(f"unknown montage {montage_key}")
        print(f"\n===== building {montage_key} for {len(ids)} subjects =====", flush=True)
        for i, sid in enumerate(ids, 1):
            print(f"\n--- [{montage_key}] {i}/{len(ids)} {sid} ---", flush=True)
            try:
                built = build_windows_for_sid(sid, montage_key, force=args.force)
                n = built["manifest"].get("n_windows_total", 0)
                if n <= 0:
                    raise ValueError("zero windows after labeling")
                per_montage_ok[montage_key].append(sid)
            except Exception as e:
                print(f"[ERROR] {montage_key}/{sid}: {type(e).__name__}: {e}", flush=True)
                failures.append({"montage": montage_key, "subject_id": sid, "error": f"{type(e).__name__}: {e}"})

    # Common subjects that succeeded for ALL requested montages → frozen shared splits
    common = None
    for m in args.montages:
        s = set(per_montage_ok[m])
        common = s if common is None else (common & s)
    common_ids = sorted(common or [])
    if not common_ids:
        raise SystemExit("no subjects succeeded for all montages")

    splits = make_subject_splits(common_ids, seed=SEED)
    # Drop any subject not in common from being written into sibling packs' split files
    for montage_key in args.montages:
        cfg = MONTAGES[montage_key]
        pack = ROOT / "datasets" / cfg["pack_name"]
        write_splits(pack, splits, common_ids, montage_key)
        write_pack_docs(pack, montage_key, splits, common_ids)

    summary = {
        "step": "expand_hmc_crown_vig",
        "completed_utc": utc_now(),
        "n_ids_attempted": len(ids),
        "per_montage_ok": {m: len(v) for m, v in per_montage_ok.items()},
        "common_subjects": len(common_ids),
        "splits": {k: len(v) for k, v in splits.items()},
        "failures": failures,
        "montages": {m: MONTAGES[m] for m in args.montages},
        "recipe": {
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "target_sr": TARGET_SR,
            "pre_sec": PRE_SEC,
            "post_sec": POST_SEC,
            "majority": MAJORITY,
            "labels": list(HEAD_A_BINARY_LABELS),
        },
    }
    out = EXPORTS / "expand_hmc_crown_vig_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("common_subjects", "splits", "per_montage_ok")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

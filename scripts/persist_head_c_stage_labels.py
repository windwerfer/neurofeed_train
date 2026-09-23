#!/usr/bin/env python3
"""Backfill stage_raw + stage_coarse onto existing Sleep-EDF Head A window exports."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sleep_edf import (
    STAGE_TO_COARSE,
    load_sleep_edf_recording,
    majority_stage_in_window,
    stage_to_coarse,
)

NIGHTS = ["SC4001", "SC4002", "SC4011", "SC4021", "SC4031", "SC4041"]
CASSETTE = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data/sleep-edfx-pilot/sleep-cassette"
WINDOWS_PKG = ROOT / "kaggle_datasets/muse-eeg-heads-windows"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def backfill_night(tag: str) -> Dict[str, Any]:
    out_dir = ROOT / f"exports/windows_{tag}"
    npz_path = out_dir / f"sleep_edf_{tag}_n1slice_windows.npz"
    man_path = out_dir / f"sleep_edf_{tag}_n1slice_manifest.json"
    man = json.loads(man_path.read_text())
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]
    y = z["y"]
    starts = z["starts"]
    label_names = z["label_names"]

    psg = CASSETTE / man["psg_file"]
    hyp = CASSETTE / man["hypno_file"]
    if not psg.exists() or not hyp.exists():
        raise FileNotFoundError(f"missing {psg} or {hyp}")

    target_sr = float(man["target_sr"])
    window_sec = float(man["window_sec"])
    pre_sec = float(man["pre_sec"])
    post_sec = float(man["post_sec"])
    slice_start = float(man["slice_start_sec"])
    win = int(round(window_sec * target_sr))

    rec = load_sleep_edf_recording(
        psg,
        hyp,
        target_sr=target_sr,
        start_sec=slice_start,
        duration_sec=pre_sec + post_sec,
    )
    stages = rec["stages"]
    # Prefer exact manifest slice_start; warn if loader drifted a record
    if abs(float(rec["slice_start_sec"]) - slice_start) > 1e-6:
        print(
            f"[{tag}] slice_start_sec drift: manifest={slice_start} "
            f"loader={rec['slice_start_sec']}"
        )

    stage_raw: List[str] = []
    stage_coarse: List[str] = []
    mismatch_head_a = 0
    for i, st in enumerate(starts):
        st_i = int(st)
        if st_i + win > len(stages):
            raise ValueError(
                f"[{tag}] start {st_i}+{win} exceeds stages len {len(stages)} "
                f"(n_windows={len(starts)})"
            )
        raw = majority_stage_in_window(stages, st_i, win)
        coarse = stage_to_coarse(raw)
        stage_raw.append(raw)
        stage_coarse.append(coarse)
        # Consistency check vs Head A binary ids
        lab = str(label_names[int(y[i])])
        if lab == "drowsy" and raw != "Sleep stage W":
            mismatch_head_a += 1
        elif lab == "hypnagogic" and raw != "Sleep stage 1":
            mismatch_head_a += 1

    stage_raw_arr = np.asarray(stage_raw, dtype=object)
    stage_coarse_arr = np.asarray(stage_coarse, dtype=object)

    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y,
        starts=starts,
        label_names=label_names,
        stage_raw=stage_raw_arr,
        stage_coarse=stage_coarse_arr,
    )
    new_sha = sha256(npz_path)

    man["stage_raw_counts"] = dict(Counter(stage_raw))
    man["stage_coarse_counts"] = dict(Counter(stage_coarse))
    man["stage_to_coarse_map"] = dict(STAGE_TO_COARSE)
    man["head_c_labels"] = True
    man["head_c_backfill_utc"] = datetime.now(timezone.utc).isoformat()
    man["head_a_stage_mismatch_windows"] = int(mismatch_head_a)
    man["npz_sha256"] = new_sha
    man["npz_path"] = str(npz_path)
    man_path.write_text(json.dumps(man, indent=2) + "\n")

    # Mirror into Kaggle package
    WINDOWS_PKG.mkdir(parents=True, exist_ok=True)
    shutil.copy2(npz_path, WINDOWS_PKG / npz_path.name)
    shutil.copy2(man_path, WINDOWS_PKG / man_path.name)

    print(
        f"[{tag}] n={len(starts)} raw={dict(Counter(stage_raw))} "
        f"coarse={dict(Counter(stage_coarse))} head_a_mismatch={mismatch_head_a}"
    )
    return {
        "tag": tag,
        "n_windows": int(len(starts)),
        "stage_raw_counts": dict(Counter(stage_raw)),
        "stage_coarse_counts": dict(Counter(stage_coarse)),
        "head_a_stage_mismatch_windows": int(mismatch_head_a),
        "npz_sha256": new_sha,
        "slice_start_sec": slice_start,
        "loader_slice_start_sec": float(rec["slice_start_sec"]),
    }


def main() -> None:
    results: Dict[str, Any] = {
        "step": "persist_head_c_stage_labels",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "nights": [],
        "policy": "docs/head_c_labels_policy.md",
        "note": "Backfill only; Head A mapping unchanged; no Head C training.",
    }
    for night in NIGHTS:
        tag = night.lower()
        results["nights"].append(backfill_night(tag))

    # Rollup manifest for windows package
    rollup: Dict[str, Any] = {
        "dataset": "muse-eeg-heads-windows",
        "nights": NIGHTS,
        "files": [],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "head_c_labels": True,
        "recipe": {
            "around_stage": "stage 1",
            "pre_sec": 1200,
            "post_sec": 2400,
            "window_sec": 2.0,
            "hop_sec": 0.5,
            "majority_frac": 0.7,
            "target_sr": 256.0,
            "labels": {"Sleep stage W": "drowsy", "Sleep stage 1": "hypnagogic"},
            "stage_to_coarse": dict(STAGE_TO_COARSE),
        },
    }
    for night in NIGHTS:
        tag = night.lower()
        npz = WINDOWS_PKG / f"sleep_edf_{tag}_n1slice_windows.npz"
        man_p = WINDOWS_PKG / f"sleep_edf_{tag}_n1slice_manifest.json"
        man_data = json.loads(man_p.read_text())
        rollup["files"].extend([npz.name, man_p.name])
        rollup[tag] = {
            "n_windows_per_label": man_data.get("n_windows_per_label"),
            "n_windows_total": man_data.get("n_windows_total"),
            "slice_start_sec": man_data.get("slice_start_sec"),
            "npz_sha256": man_data.get("npz_sha256"),
            "stage_raw_counts": man_data.get("stage_raw_counts"),
            "stage_coarse_counts": man_data.get("stage_coarse_counts"),
        }
    (WINDOWS_PKG / "manifest.json").write_text(json.dumps(rollup, indent=2) + "\n")

    # README note
    readme = WINDOWS_PKG / "README.md"
    if readme.exists():
        txt = readme.read_text()
        if "stage_raw" not in txt:
            txt = (
                txt.rstrip()
                + "\n\n## Head C labels (dataset only)\n\n"
                + "Each npz includes `stage_raw` (hypnogram text) and "
                + "`stage_coarse` (`wake`/`light`/`deep`/`rem`/`unknown`). "
                + "Head A labels unchanged (W→drowsy, N1→hypnagogic). "
                + "No Head C training in this package yet.\n"
            )
            readme.write_text(txt)

    summary_path = ROOT / "exports/persist_head_c_stage_labels_summary.json"
    results["kaggle_package"] = str(WINDOWS_PKG)
    results["total_windows"] = sum(n["n_windows"] for n in results["nights"])
    results["total_head_a_mismatch"] = sum(
        n["head_a_stage_mismatch_windows"] for n in results["nights"]
    )
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("total_windows", "total_head_a_mismatch")}, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

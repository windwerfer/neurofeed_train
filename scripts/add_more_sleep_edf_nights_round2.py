#!/usr/bin/env python3
"""Wave2 step more_sleep_edf_nights_round2: add SC4021 + SC4041 windows + provenance."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sleep_edf import (
    PROXY_NOTE,
    STAGE_TO_HEAD_A,
    load_sleep_edf_recording,
    windows_with_head_a_labels,
)
from src.head_a import HEAD_A_BINARY_LABELS, labels_to_ids

SEED = 42
WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0
PRE_SEC = 20 * 60
POST_SEC = 40 * 60
MAJORITY = 0.7

NIGHTS = [
    ("SC4021", "SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf"),
    ("SC4041", "SC4041E0-PSG.edf", "SC4041EC-Hypnogram.edf"),
]

NIGHTS_ALL = ["SC4001", "SC4002", "SC4011", "SC4021", "SC4031", "SC4041"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_night_windows(psg: Path, hyp: Path, out_dir: Path, tag: str) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = load_sleep_edf_recording(
        psg,
        hyp,
        target_sr=TARGET_SR,
        around_stage="stage 1",
        pre_sec=PRE_SEC,
        post_sec=POST_SEC,
    )
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
    counts = dict(Counter(labels))
    npz_path = out_dir / f"sleep_edf_{tag}_n1slice_windows.npz"
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y,
        starts=starts,
        label_names=np.asarray(HEAD_A_BINARY_LABELS),
    )
    manifest = {
        "psg_file": psg.name,
        "hypno_file": hyp.name,
        "psg_path": str(psg),
        "hypno_path": str(hyp),
        "tag": tag,
        "slice_start_sec": rec["slice_start_sec"],
        "pre_sec": PRE_SEC,
        "post_sec": POST_SEC,
        "around_stage": "stage 1",
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "majority_frac": MAJORITY,
        "target_sr": TARGET_SR,
        "channel_proxy_note": PROXY_NOTE,
        "stage_to_label_map": {k: v for k, v in STAGE_TO_HEAD_A.items()},
        "n_windows_per_label": counts,
        "n_windows_total": int(X.shape[0]),
        "window_shape": list(X.shape),
        "npz_path": str(npz_path),
        "npz_sha256": sha256(npz_path),
        "psg_sha256": sha256(psg),
        "hypno_sha256": sha256(hyp),
        "label_list": list(HEAD_A_BINARY_LABELS),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_attribution": {
            "Sleep-EDF": "PhysioNet ODC-By — attribute PhysioNet / Sleep-EDF Expanded",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
        "source_url_base": "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/",
        "step": "more_sleep_edf_nights_round2",
    }
    man_path = out_dir / f"sleep_edf_{tag}_n1slice_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[{tag}] slice_start_sec={rec['slice_start_sec']} windows={X.shape} counts={counts}")
    return {
        "manifest": manifest,
        "npz_path": npz_path,
        "man_path": man_path,
        "counts": counts,
    }


def main() -> None:
    cassette = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data/sleep-edfx-pilot/sleep-cassette"
    windows_pkg = ROOT / "kaggle_datasets/muse-eeg-heads-windows"
    results: Dict[str, Any] = {
        "step": "more_sleep_edf_nights_round2",
        "nights": [],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "why": "two more cassette subjects (02, 04) for broader multi-subject pool",
    }

    for subject, psg_name, hyp_name in NIGHTS:
        tag = subject.lower()
        psg = cassette / psg_name
        hyp = cassette / hyp_name
        if not psg.exists() or not hyp.exists():
            raise FileNotFoundError(f"Missing {psg} or {hyp}")
        out_dir = ROOT / f"exports/windows_{tag}"
        built = build_night_windows(psg, hyp, out_dir, tag)
        for src in (built["npz_path"], built["man_path"]):
            shutil.copy2(src, windows_pkg / src.name)
        results["nights"].append(
            {
                "subject": subject,
                "tag": tag,
                "counts": built["counts"],
                "n_windows_total": built["manifest"]["n_windows_total"],
                "slice_start_sec": built["manifest"]["slice_start_sec"],
                "npz_sha256": built["manifest"]["npz_sha256"],
                "psg_sha256": built["manifest"]["psg_sha256"],
                "hypno_sha256": built["manifest"]["hypno_sha256"],
                "npz": str(built["npz_path"]),
                "manifest": str(built["man_path"]),
            }
        )

    provenance = {
        "step": "more_sleep_edf_nights_round2",
        "source": "https://physionet.org/content/sleep-edfx/1.0.0/",
        "license": "ODC-By",
        "added_utc": datetime.now(timezone.utc).isoformat(),
        "files": [],
    }
    for subject, psg_name, hyp_name in NIGHTS:
        for name in (psg_name, hyp_name):
            p = cassette / name
            provenance["files"].append(
                {
                    "name": name,
                    "subject": subject,
                    "sha256": sha256(p),
                    "bytes": p.stat().st_size,
                    "url": f"https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/{name}",
                }
            )
    prov_path = cassette.parent / "provenance_more_nights_round2.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    results["provenance_path"] = str(prov_path)

    rollup: Dict[str, Any] = {
        "dataset": "muse-eeg-heads-windows",
        "nights": NIGHTS_ALL,
        "files": [],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "recipe": {
            "around_stage": "stage 1",
            "pre_sec": PRE_SEC,
            "post_sec": POST_SEC,
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "majority_frac": MAJORITY,
            "target_sr": TARGET_SR,
            "labels": {"Sleep stage W": "drowsy", "Sleep stage 1": "hypnagogic"},
        },
    }
    for subj in NIGHTS_ALL:
        tag = subj.lower()
        npz = windows_pkg / f"sleep_edf_{tag}_n1slice_windows.npz"
        man = windows_pkg / f"sleep_edf_{tag}_n1slice_manifest.json"
        if not npz.exists() or not man.exists():
            continue
        man_data = json.loads(man.read_text())
        rollup["files"].extend([npz.name, man.name])
        rollup[tag] = {
            "n_windows_per_label": man_data.get("n_windows_per_label"),
            "n_windows_total": man_data.get("n_windows_total"),
            "slice_start_sec": man_data.get("slice_start_sec"),
            "npz_sha256": man_data.get("npz_sha256") or sha256(npz),
        }
    (windows_pkg / "manifest.json").write_text(json.dumps(rollup, indent=2) + "\n")
    results["rollup"] = {"nights": rollup["nights"], "updated_utc": rollup["updated_utc"]}
    results["per_night"] = {
        k: rollup[k] for k in ("sc4001", "sc4002", "sc4011", "sc4021", "sc4031", "sc4041") if k in rollup
    }

    out_summary = ROOT / "exports/more_sleep_edf_nights_round2_summary.json"
    out_summary.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()

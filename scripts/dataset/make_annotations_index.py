#!/usr/bin/env python3
"""Build windows_index.csv + recording_manifest.json per corpus from linked windows."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"


def subject_for_sleep(recording_id: str) -> str:
    # SC4001 -> SC400
    return recording_id[:5]


def subject_for_ds001787(recording_id: str) -> str:
    # sub001_ses01 -> sub-001
    core = recording_id.split("_")[0]  # sub001
    num = core.replace("sub", "")
    return f"sub-{num.zfill(3) if num.isdigit() else num}"


def subject_for_ds003969(recording_id: str) -> str:
    # sub001 -> sub-001
    num = recording_id.replace("sub", "")
    return f"sub-{num.zfill(3) if num.isdigit() else num}"


def load_split_map(corpus_dir: Path) -> dict[str, str]:
    """subject_id -> split name"""
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        p = corpus_dir / "splits" / f"{split}_subjects.json"
        if not p.exists():
            continue
        for sid in json.loads(p.read_text())["subjects"]:
            out[sid] = split
    return out


def index_sleep(corpus_dir: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    recordings: list[dict] = []
    split_map = load_split_map(corpus_dir)
    win_dir = corpus_dir / "windows"
    for npz_path in sorted(win_dir.glob("*_windows.npz")):
        rid = npz_path.name.replace("_windows.npz", "")
        sid = subject_for_sleep(rid)
        man_path = win_dir / f"{rid}_manifest.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else {}
        data = np.load(npz_path, allow_pickle=True)
        y = data["y"]
        starts = data["starts"]
        label_names = list(data["label_names"])
        stage_raw = data["stage_raw"] if "stage_raw" in data.files else None
        stage_coarse = data["stage_coarse"] if "stage_coarse" in data.files else None
        n = int(y.shape[0])
        slice_start = float(man.get("slice_start_sec", 0.0))
        target_sr = float(man.get("target_sr", 256.0))
        for i in range(n):
            lab = label_names[int(y[i])] if int(y[i]) < len(label_names) else str(y[i])
            row = {
                "corpus": "vigilance_sleep_edf",
                "subject_id": sid,
                "recording_id": rid,
                "window_index": i,
                "split": split_map.get(sid, ""),
                "y_head_a": lab,
                "stage_raw": str(stage_raw[i]) if stage_raw is not None else "",
                "stage_coarse": str(stage_coarse[i]) if stage_coarse is not None else "",
                "slice_start_sec": slice_start,
                "window_start_sample": int(starts[i]),
                "window_start_sec": slice_start + (int(starts[i]) / target_sr),
                "provenance": "physionet-sleep-edf",
                "license_spdx": "ODbL-1.0",
                "npz_relpath": str(npz_path.relative_to(ROOT)),
            }
            rows.append(row)
        recordings.append(
            {
                "recording_id": rid,
                "subject_id": sid,
                "split": split_map.get(sid, ""),
                "n_windows": n,
                "label_counts": {
                    label_names[k]: int((y == k).sum()) for k in range(len(label_names))
                },
                "npz_sha256": man.get("npz_sha256"),
                "slice_start_sec": slice_start,
                "window_sec": man.get("window_sec"),
                "hop_sec": man.get("hop_sec"),
                "channels_note": man.get("channel_proxy_note"),
                "manifest_relpath": str(man_path.relative_to(ROOT))
                if man_path.exists()
                else None,
                "npz_relpath": str(npz_path.relative_to(ROOT)),
            }
        )
    return rows, recordings


CORPUS_META = {
    "vigilance_sleep_edf": {
        "provenance": "physionet-sleep-edf",
        "license_spdx": "Open Data Commons Open Database License v1.0",
        "doi": "",
    },
    "attention_ds001787": {
        "provenance": "openneuro-ds001787",
        "license_spdx": "CC0-1.0",
        "doi": "doi:10.18112/openneuro.ds001787.v1.1.1",
    },
    "attention_ds003969": {
        "provenance": "openneuro-ds003969",
        "license_spdx": "CC0-1.0",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
    },
}


def index_attention(
    corpus: str, corpus_dir: Path, subject_fn
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    recordings: list[dict] = []
    split_map = load_split_map(corpus_dir)
    meta = CORPUS_META.get(corpus, {})
    win_dir = corpus_dir / "windows"
    for npz_path in sorted(win_dir.glob("*_windows.npz")):
        rid = npz_path.name.replace("_windows.npz", "")
        sid = subject_fn(rid)
        man_path = win_dir / f"{rid}_manifest.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else {}
        data = np.load(npz_path, allow_pickle=True)
        y = data["y"]
        starts = data["starts"]
        label_names = list(data["label_names"])
        n = int(y.shape[0])
        provenance = man.get("doi") or meta.get("doi") or meta.get("provenance", "")
        license_spdx = man.get("license_spdx") or meta.get("license_spdx", "")
        for i in range(n):
            lab = label_names[int(y[i])] if int(y[i]) < len(label_names) else str(y[i])
            rows.append(
                {
                    "corpus": corpus,
                    "subject_id": sid,
                    "recording_id": rid,
                    "window_index": i,
                    "split": split_map.get(sid, ""),
                    "y_head_a": lab,
                    "stage_raw": "",
                    "stage_coarse": "",
                    "slice_start_sec": 0.0,
                    "window_start_sample": int(starts[i]),
                    "window_start_sec": "",
                    "provenance": provenance,
                    "license_spdx": license_spdx,
                    "npz_relpath": str(npz_path.relative_to(ROOT)),
                }
            )
        recordings.append(
            {
                "recording_id": rid,
                "subject_id": sid,
                "split": split_map.get(sid, ""),
                "n_windows": n,
                "label_counts": {
                    label_names[k]: int((y == k).sum()) for k in range(len(label_names))
                },
                "npz_sha256": man.get("npz_sha256"),
                "channels": man.get("channels")
                or (list(data["channels"]) if "channels" in data.files else None),
                "channel_strategy": man.get("channel_strategy"),
                "provenance": provenance,
                "license_spdx": license_spdx,
                "manifest_relpath": str(man_path.relative_to(ROOT))
                if man_path.exists()
                else None,
                "npz_relpath": str(npz_path.relative_to(ROOT)),
            }
        )
    return rows, recordings


def write_index(corpus_dir: Path, rows: list[dict], recordings: list[dict]) -> None:
    ann = corpus_dir / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    csv_path = ann / "windows_index.csv"
    fields = [
        "corpus",
        "subject_id",
        "recording_id",
        "window_index",
        "split",
        "y_head_a",
        "stage_raw",
        "stage_coarse",
        "slice_start_sec",
        "window_start_sample",
        "window_start_sec",
        "provenance",
        "license_spdx",
        "npz_relpath",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_windows": len(rows),
        "n_recordings": len(recordings),
        "recordings": recordings,
        "windows_index": str(csv_path.relative_to(ROOT)),
    }
    (ann / "recording_manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(
        f"{corpus_dir.name}: {len(recordings)} recordings, {len(rows)} windows → "
        f"{csv_path.relative_to(ROOT)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATASETS)
    args = ap.parse_args()
    root: Path = args.root

    rows, recs = index_sleep(root / "vigilance_sleep_edf")
    write_index(root / "vigilance_sleep_edf", rows, recs)

    rows, recs = index_attention(
        "attention_ds001787", root / "attention_ds001787", subject_for_ds001787
    )
    write_index(root / "attention_ds001787", rows, recs)

    rows, recs = index_attention(
        "attention_ds003969", root / "attention_ds003969", subject_for_ds003969
    )
    write_index(root / "attention_ds003969", rows, recs)


if __name__ == "__main__":
    main()

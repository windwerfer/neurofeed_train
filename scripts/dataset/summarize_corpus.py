#!/usr/bin/env python3
"""Print per-corpus recording/window/split counts."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

CORPORA = (
    "vigilance_sleep_edf",
    "attention_ds001787",
    "attention_ds003969",
)


def summarize(corpus_dir: Path) -> None:
    name = corpus_dir.name
    print(f"\n=== {name} ===")
    splits_dir = corpus_dir / "splits"
    for split in ("train", "val", "test"):
        p = splits_dir / f"{split}_subjects.json"
        if p.exists():
            d = json.loads(p.read_text())
            print(
                f"  {split}: subjects={d['subjects']} recordings={d.get('recordings', [])}"
            )

    idx = corpus_dir / "annotations" / "windows_index.csv"
    if not idx.exists():
        print("  (no windows_index.csv yet)")
        return
    by_split: Counter = Counter()
    by_label: Counter = Counter()
    by_split_label: dict[str, Counter] = defaultdict(Counter)
    subjects: set[str] = set()
    recordings: set[str] = set()
    with idx.open() as f:
        for row in csv.DictReader(f):
            by_split[row["split"] or "?"] += 1
            by_label[row["y_head_a"]] += 1
            by_split_label[row["split"] or "?"][row["y_head_a"]] += 1
            subjects.add(row["subject_id"])
            recordings.add(row["recording_id"])
    print(f"  subjects={len(subjects)} recordings={len(recordings)} windows={sum(by_split.values())}")
    print(f"  windows_by_split: {dict(by_split)}")
    print(f"  windows_by_label: {dict(by_label)}")
    for sp in ("train", "val", "test", "?"):
        if sp in by_split_label:
            print(f"  {sp} labels: {dict(by_split_label[sp])}")

    raw = list((corpus_dir / "raw").glob("*"))
    raw = [p for p in raw if p.name != ".gitkeep"]
    wins = list((corpus_dir / "windows").glob("*.npz"))
    print(f"  raw_files={len(raw)} window_npz={len(wins)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATASETS)
    args = ap.parse_args()
    root: Path = args.root
    print(f"corpus summary under {root}")
    for c in CORPORA:
        summarize(root / c)


if __name__ == "__main__":
    main()

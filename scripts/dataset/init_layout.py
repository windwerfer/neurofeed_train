#!/usr/bin/env python3
"""Create local-first Muse EEG dataset curation layout under datasets/."""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

CORPORA = (
    "vigilance_sleep_edf",
    "attention_ds001787",
    "attention_ds003969",
)

SUBDIRS = ("raw", "windows", "annotations", "splits")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATASETS)
    args = ap.parse_args()
    root: Path = args.root

    created: list[str] = []
    for corpus in CORPORA:
        for sub in SUBDIRS:
            d = root / corpus / sub
            d.mkdir(parents=True, exist_ok=True)
            keep = d / ".gitkeep"
            if not keep.exists():
                keep.write_text("")
                created.append(str(keep.relative_to(root.parent)))
        (root / corpus).mkdir(parents=True, exist_ok=True)

    common = root / "common"
    common.mkdir(parents=True, exist_ok=True)

    print(f"layout ready under {root}")
    for c in CORPORA:
        print(f"  {c}/{{raw,windows,annotations,splits}}")
    if created:
        print(f"created {len(created)} .gitkeep files")


if __name__ == "__main__":
    main()

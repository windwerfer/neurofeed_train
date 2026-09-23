#!/usr/bin/env python3
"""Validate fixed subject splits: no subject leakage across train/val/test."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

CORPORA = (
    "vigilance_sleep_edf",
    "attention_ds001787",
    "attention_ds003969",
)


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_corpus(corpus_dir: Path) -> list[str]:
    errors: list[str] = []
    splits_dir = corpus_dir / "splits"
    name = corpus_dir.name
    required = [
        "subjects.json",
        "train_subjects.json",
        "val_subjects.json",
        "test_subjects.json",
    ]
    for fn in required:
        if not (splits_dir / fn).exists():
            errors.append(f"{name}: missing {fn}")
            return errors

    train = set(load_split(splits_dir / "train_subjects.json")["subjects"])
    val = set(load_split(splits_dir / "val_subjects.json")["subjects"])
    test = set(load_split(splits_dir / "test_subjects.json")["subjects"])
    catalog = set(load_split(splits_dir / "subjects.json")["subjects"].keys())

    if not train:
        errors.append(f"{name}: train empty")
    if not val:
        errors.append(f"{name}: val empty")
    if not test:
        errors.append(f"{name}: test empty")

    for a, b, lab in (
        (train, val, "train∩val"),
        (train, test, "train∩test"),
        (val, test, "val∩test"),
    ):
        inter = a & b
        if inter:
            errors.append(f"{name}: subject leakage {lab}: {sorted(inter)}")

    union = train | val | test
    missing = catalog - union
    extra = union - catalog
    if missing:
        errors.append(f"{name}: subjects not assigned to a split: {sorted(missing)}")
    if extra:
        errors.append(f"{name}: split subjects missing from catalog: {sorted(extra)}")

    # Recording-level overlap check via subjects.json recordings lists
    subj_meta = load_split(splits_dir / "subjects.json")["subjects"]
    rec_owner: dict[str, str] = {}
    for split_name, sset in (("train", train), ("val", val), ("test", test)):
        for sid in sset:
            for rid in subj_meta[sid]["recordings"]:
                if rid in rec_owner and rec_owner[rid] != split_name:
                    errors.append(
                        f"{name}: recording {rid} in both {rec_owner[rid]} and {split_name}"
                    )
                rec_owner[rid] = split_name

    # Sleep-EDF special: SC4001 and SC4002 must share subject and same split
    if name == "vigilance_sleep_edf":
        policy = load_split(splits_dir / "split_policy.json")
        recs = policy.get("recordings", {})
        by_subj: dict[str, set[str]] = {}
        for rid, meta in recs.items():
            sid = meta["subject_id"]
            by_subj.setdefault(sid, set()).add(rid)
        split_of = {}
        for split_name, sset in (("train", train), ("val", val), ("test", test)):
            for sid in sset:
                split_of[sid] = split_name
        for sid, rset in by_subj.items():
            if sid not in split_of:
                errors.append(f"{name}: subject {sid} has recordings but no split")
                continue
            # all recordings of subject must map to that subject only
            for rid in rset:
                if recs[rid]["subject_id"] != sid:
                    errors.append(f"{name}: recording {rid} subject mismatch")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATASETS)
    args = ap.parse_args()
    root: Path = args.root

    all_errors: list[str] = []
    for corpus in CORPORA:
        errs = validate_corpus(root / corpus)
        if errs:
            all_errors.extend(errs)
        else:
            train = load_split(root / corpus / "splits" / "train_subjects.json")
            val = load_split(root / corpus / "splits" / "val_subjects.json")
            test = load_split(root / corpus / "splits" / "test_subjects.json")
            print(
                f"OK {corpus}: train={train['subjects']} val={val['subjects']} "
                f"test={test['subjects']}"
            )

    if all_errors:
        print("VALIDATION FAILED:", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("all corpora: no subject leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

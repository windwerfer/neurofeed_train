#!/usr/bin/env python3
"""Batch A toward Head A ~40 subjects: +8 Muse-proximal ds003969 (CC0).

Local-first; paced OpenNeuro downloads; no Kaggle upload.
New subjects → train split (frozen test holdouts untouched).
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ds003969_attention_ingest as base  # noqa: E402

STEP = "expand_attention_batch40_ds003969"

# Prefer Muse-proximal volume: 4 htr + 4 ctr not yet ingested
SUBJECTS: List[Dict[str, str]] = [
    {"sub": "007", "group": "htr", "first_session": "meditation", "tree": "925b13b4590eeb7df6c2a5a782a22eed8a9f9499"},
    {"sub": "008", "group": "htr", "first_session": "thinking", "tree": "fd0bd5d241de27f1bb1a0c957ab8db684d5fef4e"},
    {"sub": "009", "group": "htr", "first_session": "meditation", "tree": "04032b8756ca8dbaadcdabb86af9becfed467c40"},
    {"sub": "010", "group": "htr", "first_session": "thinking", "tree": "127c093465f11282bff24eabf25eb3412869e865"},
    {"sub": "030", "group": "ctr", "first_session": "meditation", "tree": "ddd14c8b6f0dce52953abcf7d24ae08e09bb016f"},
    {"sub": "031", "group": "ctr", "first_session": "thinking", "tree": "b22654ae91925c5115fcacf51e951d9d264db906"},
    {"sub": "032", "group": "ctr", "first_session": "meditation", "tree": "e735761a82f4d6b3a6dd0a91e86fc61664b9add1"},
    {"sub": "033", "group": "ctr", "first_session": "thinking", "tree": "35771a8d96ee156e132ee89c7595e4a408df0087"},
]

PRIOR = [
    "sub001", "sub002", "sub003", "sub004", "sub005", "sub006",
    "sub025", "sub026", "sub027", "sub028", "sub029",
]

# Frozen honest holdouts — never move into train/val
FROZEN_TEST = ["sub-025", "sub-027"]
FROZEN_VAL = ["sub-026", "sub-028"]


def migrate_into_datasets(tag: str) -> None:
    """Symlink export windows into datasets/attention_ds003969/windows/."""
    win_dir = ROOT / "datasets/attention_ds003969/windows"
    win_dir.mkdir(parents=True, exist_ok=True)
    src_npz = base.EXPORT / tag / f"ds003969_{tag}_attention_windows.npz"
    src_man = base.EXPORT / tag / f"ds003969_{tag}_attention_manifest.json"
    for src, name in (
        (src_npz, f"{tag}_windows.npz"),
        (src_man, f"{tag}_manifest.json"),
    ):
        dst = win_dir / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def update_splits(new_tags: List[str]) -> None:
    """Append new subjects to train; keep frozen val/test."""
    split_dir = ROOT / "datasets/attention_ds003969/splits"
    subjects_path = split_dir / "subjects.json"
    subjects_doc = json.loads(subjects_path.read_text())
    subjects = subjects_doc.setdefault("subjects", {})

    for tag in new_tags:
        num = tag.replace("sub", "")
        sid = f"sub-{num.zfill(3)}"
        spec = next(s for s in SUBJECTS if s["sub"] == num.zfill(3) or s["sub"] == num)
        subjects[sid] = {
            "group": spec["group"],
            "recordings": [tag],
            "notes": f"batch40 {STEP}",
        }

    train = sorted(
        sid
        for sid in subjects
        if sid not in FROZEN_TEST and sid not in FROZEN_VAL
    )
    val = list(FROZEN_VAL)
    test = list(FROZEN_TEST)

    created = datetime.now(timezone.utc).isoformat()
    subjects_doc["created_utc"] = created
    subjects_path.write_text(json.dumps(subjects_doc, indent=2) + "\n")

    for split, ids in (("train", train), ("val", val), ("test", test)):
        (split_dir / f"{split}_subjects.json").write_text(
            json.dumps(
                {
                    "corpus": "attention_ds003969",
                    "split": split,
                    "policy": "fixed_subject_json",
                    "subjects": ids,
                    "created_utc": created,
                    "note": f"updated by {STEP}; frozen test={FROZEN_TEST}",
                },
                indent=2,
            )
            + "\n"
        )

    policy = {
        "corpus": "attention_ds003969",
        "policy": "fixed_subject_json",
        "updated_utc": created,
        "updated_by": STEP,
        "splits": {"train": train, "val": val, "test": test},
        "frozen_test": FROZEN_TEST,
        "frozen_val": FROZEN_VAL,
    }
    (split_dir / "split_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(f"splits train={len(train)} val={len(val)} test={len(test)}", flush=True)


def main() -> None:
    base.CACHE.mkdir(parents=True, exist_ok=True)
    base.EXPORT.mkdir(parents=True, exist_ok=True)
    base.WINDOWS_PKG.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "step": STEP,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_subjects": list(PRIOR),
        "subjects": [],
        "errors": [],
        "why": "Batch A toward ~40 Head A attention subjects; Muse-proximal ds003969 CC0",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "plan": "docs/head_a_40_subjects_plan.md",
        "label_rule": "med*→concentration; think*→mind_wandering",
        "muse_proxy": "AF7/AF8 native; TP7/TP8→TP9/TP10",
    }
    totals = Counter()
    provenance = {
        "step": STEP,
        "source": "https://openneuro.org/datasets/ds003969",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "added_utc": datetime.now(timezone.utc).isoformat(),
        "recordings": [],
    }
    new_tags: List[str] = []

    for spec in SUBJECTS:
        try:
            recs = base.fetch_subject(spec)
            info = base.process_subject(spec, recs)
            man_path = Path(info["manifest"])
            man = json.loads(man_path.read_text())
            man["step"] = STEP
            man_path.write_text(json.dumps(man, indent=2) + "\n")
            shutil.copy2(man_path, base.WINDOWS_PKG / man_path.name)
            shutil.copy2(Path(info["npz"]), base.WINDOWS_PKG / Path(info["npz"]).name)
            migrate_into_datasets(info["tag"])
            results["subjects"].append(info)
            new_tags.append(info["tag"])
            for k, v in info["counts"].items():
                totals[k] += v
            for rec in recs:
                provenance["recordings"].append(
                    {
                        "name": rec["bdf"].name,
                        "subject": f"sub-{spec['sub']}",
                        "task": rec["task"],
                        "label": rec["label"],
                        "sha256": base.sha256(rec["bdf"]),
                        "bytes": rec["bdf_bytes"],
                        "url": rec["bdf_url"],
                    }
                )
            print(
                f"done {info['tag']}: {info['counts']} total={info['n_windows_total']}",
                flush=True,
            )
        except Exception as e:
            err = {"sub": spec["sub"], "error": repr(e)}
            results["errors"].append(err)
            print(f"ERROR sub-{spec['sub']}: {e!r}", flush=True)

    results["totals_new"] = dict(totals)
    results["n_windows_new"] = int(sum(totals.values()))
    results["new_tags"] = new_tags

    if new_tags:
        update_splits(new_tags)

    pool_tags = PRIOR + new_tags
    pool_totals = Counter()
    for tag in pool_tags:
        man_p = base.EXPORT / tag / f"ds003969_{tag}_attention_manifest.json"
        if not man_p.exists():
            continue
        man = json.loads(man_p.read_text())
        for k, v in man["n_windows_per_label"].items():
            pool_totals[k] += int(v)
    results["pool_subjects"] = pool_tags
    results["pool_totals"] = dict(pool_totals)
    results["n_windows_pool"] = int(sum(pool_totals.values()))
    results["n_subjects_pool"] = len(pool_tags)
    results["gap_to_40_attention_approx"] = max(
        0, 40 - (12 + len(pool_tags))
    )  # ds001787 12 + this corpus

    prov_path = base.CACHE / f"provenance_{STEP}.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    results["provenance_path"] = str(prov_path)

    summary_path = ROOT / "exports" / f"{STEP}_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "totals_new": results["totals_new"],
                "n_windows_new": results["n_windows_new"],
                "new_tags": new_tags,
                "n_subjects_pool_ds003969": results["n_subjects_pool"],
                "errors": results["errors"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

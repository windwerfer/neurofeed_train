#!/usr/bin/env python3
"""Close Head A attention gap to ≥40: +5 Muse-proximal ds003969 (CC0).

Local-first; paced OpenNeuro downloads; no Kaggle; no HF EEGMeditation.
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

STEP = "expand_attention_batch40_close_ds003969"

# +3 htr (011–013) + 2 ctr (034–035) → closes gap of 5 from 35→40
SUBJECTS: List[Dict[str, str]] = [
    {"sub": "011", "group": "htr", "first_session": "meditation", "tree": "d4895982261cee34a6fe70cc0ef0e9692640811a"},
    {"sub": "012", "group": "htr", "first_session": "thinking", "tree": "97efd258456b21a00e1bcdff397821ba6ee5754e"},
    {"sub": "013", "group": "htr", "first_session": "meditation", "tree": "49e4525ee9c6a775575d76f43064ce1a64b0a926"},
    {"sub": "034", "group": "ctr", "first_session": "meditation", "tree": "f3bd1f84b376db0090829ff7ecc493df765404c9"},
    {"sub": "035", "group": "ctr", "first_session": "thinking", "tree": "e3165ba223eb0009464b58bae811d919b3422ff8"},
]

PRIOR = [
    "sub001", "sub002", "sub003", "sub004", "sub005", "sub006",
    "sub007", "sub008", "sub009", "sub010",
    "sub025", "sub026", "sub027", "sub028", "sub029",
    "sub030", "sub031", "sub032", "sub033",
]

FROZEN_TEST = ["sub-025", "sub-027"]
FROZEN_VAL = ["sub-026", "sub-028"]


def migrate_into_datasets(tag: str) -> None:
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
            "notes": f"batch40-close {STEP}",
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
        "why": "Close gap of 5 to ≥40 Head A attention subjects; Muse-proximal ds003969 CC0 only (no HF)",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "plan": "docs/head_a_40_subjects_plan.md",
        "label_rule": "med*→concentration; think*→mind_wandering",
        "muse_proxy": "AF7/AF8 native; TP7/TP8→TP9/TP10",
        "id_collision_note": (
            "ds001787 and ds003969 both use sub-00x IDs; LOSO must use corpus-qualified "
            "IDs (ds001787/sub-XXX vs ds003969/sub-XXX)."
        ),
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
    # ds001787 currently 16 + this corpus pool
    results["attention_union_approx"] = 16 + len(pool_tags)
    results["gap_to_40_attention"] = max(0, 40 - (16 + len(pool_tags)))

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
                "attention_union_approx": results["attention_union_approx"],
                "errors": results["errors"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

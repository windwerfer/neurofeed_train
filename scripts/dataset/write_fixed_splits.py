#!/usr/bin/env python3
"""Write FIXED subject-split JSONs (policy A) for each corpus.

Sleep-EDF: stable subject keys SC4sss (cassette subject), recordings SC4sssn.
Prior night-holdout put SC4001 in train and SC4002 in test (same subject 00) —
policy A puts entire subject SC400 in test so there is no subject leakage.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

POLICY = "fixed_subject_json"
POLICY_NOTE = (
    "Option A: fixed subject-level train/val/test JSON. "
    "No subject appears in more than one split. "
    "Prior Sleep-EDF night-holdout (train SC4001 / test SC4002) leaked subject 00; "
    "policy A assigns subject SC400 entirely to test."
)


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def sleep_edf_splits() -> dict:
    """Legacy v0 5-subject splits. Prefer scripts/finalize_head_c_corpus.py after expansion."""
    # recording_id -> subject_id (SC4sss from SC4sssn)
    recordings = {
        "SC4001": {"subject_id": "SC400", "night": 1, "tag": "sc4001"},
        "SC4002": {"subject_id": "SC400", "night": 2, "tag": "sc4002"},
        "SC4011": {"subject_id": "SC401", "night": 1, "tag": "sc4011"},
        "SC4021": {"subject_id": "SC402", "night": 1, "tag": "sc4021"},
        "SC4031": {"subject_id": "SC403", "night": 1, "tag": "sc4031"},
        "SC4041": {"subject_id": "SC404", "night": 1, "tag": "sc4041"},
    }
    subjects = {
        "SC400": {
            "recordings": ["SC4001", "SC4002"],
            "notes": "prior primary holdout night SC4002; both nights held out for subject purity",
        },
        "SC401": {"recordings": ["SC4011"], "notes": ""},
        "SC402": {"recordings": ["SC4021"], "notes": ""},
        "SC403": {
            "recordings": ["SC4031"],
            "notes": "prior secondary holdout",
        },
        "SC404": {"recordings": ["SC4041"], "notes": ""},
    }
    train = ["SC401", "SC402", "SC404"]
    val = ["SC403"]
    test = ["SC400"]
    return {
        "corpus": "vigilance_sleep_edf",
        "policy": POLICY,
        "policy_note": POLICY_NOTE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject_key_rule": "SC4sss from Sleep-EDF cassette id SC4sssn (sss=subject, n=night)",
        "recordings": recordings,
        "subjects": subjects,
        "splits": {
            "train": train,
            "val": val,
            "test": test,
        },
        "prior_mapping": {
            "primary_holdout_night": "SC4002",
            "secondary_holdout_night": "SC4031",
            "train_nights_prior": ["SC4001", "SC4011", "SC4021", "SC4041"],
        },
    }


def ds001787_splits() -> dict:
    # Prefer failed holdouts in test (honest). Val uses subjects with BOTH classes
    # (prior val sub-014/017 were concentration-only after ingest).
    subjects = {
        "sub-001": {"group": "expert", "recordings": ["sub001_ses01"]},
        "sub-002": {"group": "expert", "recordings": ["sub002_ses01"]},
        "sub-003": {"group": "expert", "recordings": ["sub003_ses01"]},
        "sub-004": {"group": "expert", "recordings": ["sub004_ses01"]},
        "sub-005": {"group": "expert", "recordings": ["sub005_ses01"]},
        "sub-006": {
            "group": "expert",
            "recordings": ["sub006_ses01"],
            "notes": "val: both classes present",
        },
        "sub-013": {
            "group": "novice",
            "recordings": ["sub013_ses01"],
            "notes": "prior secondary holdout (near chance)",
        },
        "sub-014": {
            "group": "novice",
            "recordings": ["sub014_ses01"],
            "notes": "conc-only after ingest; kept in train",
        },
        "sub-015": {
            "group": "novice",
            "recordings": ["sub015_ses01"],
            "notes": "val: balanced C/MW",
        },
        "sub-017": {
            "group": "novice",
            "recordings": ["sub017_ses01"],
            "notes": "conc-only after ingest; kept in train",
        },
        "sub-019": {
            "group": "novice",
            "recordings": ["sub019_ses01"],
            "notes": "prior primary holdout (near chance)",
        },
        "sub-020": {"group": "novice", "recordings": ["sub020_ses01"]},
    }
    train = [
        "sub-001",
        "sub-002",
        "sub-003",
        "sub-004",
        "sub-005",
        "sub-014",
        "sub-017",
        "sub-020",
    ]
    val = ["sub-006", "sub-015"]
    test = ["sub-019", "sub-013"]  # frozen honest holdouts
    return {
        "corpus": "attention_ds001787",
        "policy": POLICY,
        "policy_note": POLICY_NOTE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "doi": "doi:10.18112/openneuro.ds001787.v1.1.1",
        "license_spdx": "CC0-1.0",
        "subjects": subjects,
        "splits": {"train": train, "val": val, "test": test},
        "val_balance_note": (
            "Moved conc-only sub-014/017 out of val; val=sub-006+sub-015 both have C and MW. "
            "Test holdouts unchanged for honesty."
        ),
        "prior_mapping": {
            "train_tags": [
                "sub001_ses01",
                "sub002_ses01",
                "sub003_ses01",
                "sub004_ses01",
                "sub005_ses01",
                "sub014_ses01",
                "sub017_ses01",
                "sub020_ses01",
            ],
            "holdout_primary": "sub019_ses01",
            "holdout_secondary": "sub013_ses01",
            "val_tags": ["sub006_ses01", "sub015_ses01"],
        },
    }


def ds003969_splits() -> dict:
    # Prior failed holdout sub025 (ctr) stays in test; expand train/val with new subjects.
    subjects = {
        "sub-001": {"group": "htr", "recordings": ["sub001"]},
        "sub-002": {"group": "htr", "recordings": ["sub002"]},
        "sub-003": {"group": "htr", "recordings": ["sub003"]},
        "sub-004": {"group": "htr", "recordings": ["sub004"]},
        "sub-005": {"group": "htr", "recordings": ["sub005"]},
        "sub-006": {"group": "htr", "recordings": ["sub006"]},
        "sub-025": {
            "group": "ctr",
            "recordings": ["sub025"],
            "notes": "prior holdout (near chance on attention_only_head_smoke)",
        },
        "sub-026": {"group": "ctr", "recordings": ["sub026"]},
        "sub-027": {
            "group": "ctr",
            "recordings": ["sub027"],
            "notes": "held in test with sub-025 for multi-subject holdout",
        },
        "sub-028": {"group": "ctr", "recordings": ["sub028"]},
        "sub-029": {"group": "ctr", "recordings": ["sub029"]},
    }
    train = ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-029"]
    val = ["sub-026", "sub-028"]  # 2 ctr subjects, both balanced C/MW
    test = ["sub-025", "sub-027"]  # frozen honest holdouts
    return {
        "corpus": "attention_ds003969",
        "policy": POLICY,
        "policy_note": POLICY_NOTE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "subjects": subjects,
        "splits": {"train": train, "val": val, "test": test},
        "val_balance_note": (
            "Val expanded to sub-026+sub-028 (both 400/400). "
            "Test holdouts sub-025/027 unchanged. New sub-029 in train with htr."
        ),
        "prior_mapping": {
            "train_tags_smoke": ["sub001", "sub002"],
            "holdout_tag_smoke": "sub025",
            "val_tags": ["sub026", "sub028"],
        },
    }


def emit_corpus(base: Path, full: dict) -> None:
    splits_dir = base / "splits"
    subjects_doc = {
        "corpus": full["corpus"],
        "policy": POLICY,
        "subjects": full["subjects"],
        "recordings": full.get("recordings"),
        "created_utc": full["created_utc"],
    }
    _write(splits_dir / "subjects.json", subjects_doc)
    for split_name in ("train", "val", "test"):
        subj = full["splits"][split_name]
        recs: list[str] = []
        for sid in subj:
            recs.extend(full["subjects"][sid]["recordings"])
        payload = {
            "corpus": full["corpus"],
            "split": split_name,
            "policy": POLICY,
            "subjects": subj,
            "recordings": recs,
            "created_utc": full["created_utc"],
        }
        _write(splits_dir / f"{split_name}_subjects.json", payload)
    _write(splits_dir / "split_policy.json", full)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATASETS)
    args = ap.parse_args()
    root: Path = args.root

    emit_corpus(root / "vigilance_sleep_edf", sleep_edf_splits())
    emit_corpus(root / "attention_ds001787", ds001787_splits())
    emit_corpus(root / "attention_ds003969", ds003969_splits())
    print("fixed subject splits written (policy A)")


if __name__ == "__main__":
    main()

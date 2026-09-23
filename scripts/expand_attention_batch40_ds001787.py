#!/usr/bin/env python3
"""Batch B toward Head A ~40 subjects: more ds001787 probe-label subjects (CC0).

Local-first; paced downloads; frozen test holdouts untouched.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ds001787_attention_ingest as base  # noqa: E402

STEP = "expand_attention_batch40_ds001787"
OPENNEURO_BASE = "https://s3.amazonaws.com/openneuro.org/ds001787"
PAUSE_SEC = 4.0

SUBJECTS: List[Dict[str, str]] = [
    {"sub": "007", "ses": "01", "group": "expert", "log_candidates": ["sub07_info.txt", "sub07_2_info.txt"]},
    {"sub": "008", "ses": "01", "group": "expert", "log_candidates": ["sub08_info.txt", "sub08_2_info.txt"]},
    {"sub": "016", "ses": "01", "group": "novice", "log_candidates": ["sub16_info.txt", "sub16_2_info.txt"]},
    {"sub": "021", "ses": "01", "group": "novice", "log_candidates": ["sub21_info.txt", "sub21_2_info.txt"]},
]

PRIOR = [
    "sub001_ses01", "sub002_ses01", "sub003_ses01", "sub004_ses01",
    "sub005_ses01", "sub006_ses01", "sub013_ses01", "sub014_ses01",
    "sub015_ses01", "sub017_ses01", "sub019_ses01", "sub020_ses01",
]

FROZEN_TEST = ["sub-019", "sub-013"]
FROZEN_VAL = ["sub-006", "sub-015"]


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip exists {dest.name} ({dest.stat().st_size} bytes)", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"GET {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "muse-eeg-heads/continuum"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.rename(dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes)", flush=True)
    time.sleep(PAUSE_SEC)


def ensure_subject_files(spec: Dict[str, str]) -> None:
    sub, ses = spec["sub"], spec["ses"]
    eeg_dir = base.RAW / f"sub-{sub}" / f"ses-{ses}" / "eeg"
    for name in (
        f"sub-{sub}_ses-{ses}_task-meditation_eeg.bdf",
        f"sub-{sub}_ses-{ses}_task-meditation_events.tsv",
        f"sub-{sub}_ses-{ses}_task-meditation_eeg.json",
    ):
        url = f"{OPENNEURO_BASE}/sub-{sub}/ses-{ses}/eeg/{name}"
        download_file(url, eeg_dir / name)
    # behavioral logs if missing
    base.LOG_DIR.mkdir(parents=True, exist_ok=True)
    for cand in spec["log_candidates"]:
        dest = base.LOG_DIR / cand
        if dest.exists():
            continue
        # logs live under code/ in OpenNeuro — try common path
        url = f"{OPENNEURO_BASE}/code/MW_Current_TextFileBIDS/{cand}"
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"log miss {cand}: {e!r}", flush=True)


def migrate_into_datasets(tag: str) -> None:
    win_dir = ROOT / "datasets/attention_ds001787/windows"
    win_dir.mkdir(parents=True, exist_ok=True)
    src_npz = base.EXPORT / tag / f"ds001787_{tag}_attention_windows.npz"
    src_man = base.EXPORT / tag / f"ds001787_{tag}_attention_manifest.json"
    for src, name in (
        (src_npz, f"{tag}_windows.npz"),
        (src_man, f"{tag}_manifest.json"),
    ):
        dst = win_dir / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def update_splits(new_tags: List[str]) -> None:
    split_dir = ROOT / "datasets/attention_ds001787/splits"
    subjects_path = split_dir / "subjects.json"
    subjects_doc = json.loads(subjects_path.read_text())
    subjects = subjects_doc.setdefault("subjects", {})

    for tag in new_tags:
        # tag like sub007_ses01
        core = tag.split("_")[0]  # sub007
        num = core.replace("sub", "")
        sid = f"sub-{num.zfill(3)}"
        spec = next(s for s in SUBJECTS if s["sub"] == num.zfill(3) or s["sub"] == num)
        subjects[sid] = {
            "group": spec["group"],
            "recordings": [tag],
            "notes": f"batch40 {STEP}",
        }

    train = sorted(
        sid for sid in subjects if sid not in FROZEN_TEST and sid not in FROZEN_VAL
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
                    "corpus": "attention_ds001787",
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
    (split_dir / "split_policy.json").write_text(
        json.dumps(
            {
                "corpus": "attention_ds001787",
                "policy": "fixed_subject_json",
                "updated_utc": created,
                "updated_by": STEP,
                "splits": {"train": train, "val": val, "test": test},
                "frozen_test": FROZEN_TEST,
                "frozen_val": FROZEN_VAL,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"splits train={len(train)} val={len(val)} test={len(test)}", flush=True)


def main() -> None:
    base.LOG_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "step": STEP,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_subjects": list(PRIOR),
        "subjects": [],
        "errors": [],
        "why": "Batch B toward ~40 Head A attention subjects; ds001787 probe labels CC0",
        "doi": "doi:10.18112/openneuro.ds001787.v1.1.1",
        "license_spdx": "CC0-1.0",
        "plan": "docs/head_a_40_subjects_plan.md",
        "hard_reject": "Do not map events value 2/4 to classes",
        "label_rule": "Q1>Q2→concentration; Q1<Q2→mind_wandering; ties dropped",
    }
    for spec in SUBJECTS:
        try:
            ensure_subject_files(spec)
        except Exception as e:
            results["errors"].append({"sub": spec["sub"], "phase": "download", "error": repr(e)})
            print(f"DOWNLOAD ERROR sub-{spec['sub']}: {e!r}", flush=True)

    totals = Counter()
    new_tags: List[str] = []
    for spec in SUBJECTS:
        try:
            info = base.process_subject(spec)
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
            print(f"done {info['tag']}: {info['counts']}", flush=True)
        except Exception as e:
            results["errors"].append({"sub": spec["sub"], "phase": "process", "error": repr(e)})
            print(f"PROCESS ERROR sub-{spec['sub']}: {e!r}", flush=True)

    results["totals_new"] = dict(totals)
    results["n_windows_new"] = int(sum(totals.values()))
    results["new_tags"] = new_tags
    if new_tags:
        update_splits(new_tags)

    summary_path = ROOT / "exports" / f"{STEP}_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"totals_new": results["totals_new"], "new_tags": new_tags, "errors": results["errors"]}, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

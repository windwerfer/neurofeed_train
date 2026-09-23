#!/usr/bin/env python3
"""Wave2 invent step: add 4 more ds001787 subjects (2 expert + 2 novice) for attention pool."""
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

# Reuse ingest helpers
import ds001787_attention_ingest as base  # noqa: E402

SEED = 42
# New subjects only (ses-01). Avoid overlap with prior 001/002/013/017.
SUBJECTS: List[Dict[str, str]] = [
    {"sub": "003", "ses": "01", "group": "expert", "log_candidates": ["sub03_info.txt", "sub03_2_info.txt"]},
    {"sub": "004", "ses": "01", "group": "expert", "log_candidates": ["sub04_info.txt", "sub04_2_info.txt"]},
    {"sub": "014", "ses": "01", "group": "novice", "log_candidates": ["sub14_info.txt", "sub14_2_info.txt"]},
    {"sub": "019", "ses": "01", "group": "novice", "log_candidates": ["sub19_info.txt", "sub19_2_info.txt"]},
]

OPENNEURO_BASE = "https://s3.amazonaws.com/openneuro.org/ds001787"
PAUSE_SEC = 4.0


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"skip exists {dest.name} ({dest.stat().st_size} bytes)")
        return
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "muse-eeg-heads/continuum"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.rename(dest)
    print(f"saved {dest} ({dest.stat().st_size} bytes)")
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


def main() -> None:
    base.LOG_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "step": "more_ds001787_subjects",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_subjects": ["sub001_ses01", "sub002_ses01", "sub013_ses01", "sub017_ses01"],
        "subjects": [],
        "why": (
            "Expand ds001787 attention pool with 2 more expert + 2 more novice "
            "(subject holdout was near chance; need more subjects before retrain)"
        ),
        "doi": "doi:10.18112/openneuro.ds001787.v1.1.1",
        "license_spdx": "CC0-1.0",
        "hard_reject": "Do not map events value 2/4 to classes",
        "label_rule": "Q1>Q2→concentration; Q1<Q2→mind_wandering; ties dropped",
    }
    for spec in SUBJECTS:
        ensure_subject_files(spec)

    totals = Counter()
    for spec in SUBJECTS:
        info = base.process_subject(spec)
        # stamp step name in subject export manifests already written — rewrite step field
        man_path = Path(info["manifest"])
        man = json.loads(man_path.read_text())
        man["step"] = "more_ds001787_subjects"
        man_path.write_text(json.dumps(man, indent=2) + "\n")
        shutil.copy2(man_path, base.WINDOWS_PKG / man_path.name)
        results["subjects"].append(info)
        for k, v in info["counts"].items():
            totals[k] += v

    results["totals_new"] = dict(totals)
    results["n_windows_new"] = int(sum(totals.values()))

    # Pool totals = prior exports + new
    pool_tags = results["prior_subjects"] + [s["tag"] for s in results["subjects"]]
    pool_totals = Counter()
    for tag in pool_tags:
        man = json.loads(
            (base.EXPORT / tag / f"ds001787_{tag}_attention_manifest.json").read_text()
        )
        for k, v in man["n_windows_per_label"].items():
            pool_totals[k] += int(v)
    results["pool_subjects"] = pool_tags
    results["pool_totals"] = dict(pool_totals)
    results["n_windows_pool"] = int(sum(pool_totals.values()))

    # Append provenance for new recordings only
    prov_path = base.CACHE / "provenance_more_ds001787_subjects.json"
    provenance = {
        "step": "more_ds001787_subjects",
        "source": "https://openneuro.org/datasets/ds001787",
        "doi": "doi:10.18112/openneuro.ds001787.v1.1.1",
        "license_spdx": "CC0-1.0",
        "added_utc": datetime.now(timezone.utc).isoformat(),
        "recordings": [],
    }
    for spec in SUBJECTS:
        sub, ses = spec["sub"], spec["ses"]
        eeg_dir = base.RAW / f"sub-{sub}" / f"ses-{ses}" / "eeg"
        for name in (
            f"sub-{sub}_ses-{ses}_task-meditation_eeg.bdf",
            f"sub-{sub}_ses-{ses}_task-meditation_events.tsv",
            f"sub-{sub}_ses-{ses}_task-meditation_eeg.json",
        ):
            p = eeg_dir / name
            if not p.exists():
                continue
            provenance["recordings"].append(
                {
                    "name": name,
                    "subject": f"sub-{sub}",
                    "session": f"ses-{ses}",
                    "sha256": base.sha256(p),
                    "bytes": p.stat().st_size,
                    "url": f"{OPENNEURO_BASE}/sub-{sub}/ses-{ses}/eeg/{name}",
                }
            )
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    results["provenance_path"] = str(prov_path)

    # Merge rollup for all pool subjects
    rollup_path = base.WINDOWS_PKG / "manifest.json"
    rollup: Dict[str, Any] = {}
    if rollup_path.exists():
        try:
            rollup = json.loads(rollup_path.read_text())
        except json.JSONDecodeError:
            rollup = {}
    files = []
    for tag in pool_tags:
        files.extend(
            [
                f"ds001787_{tag}_attention_windows.npz",
                f"ds001787_{tag}_attention_manifest.json",
            ]
        )
    rollup["ds001787_attention"] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": pool_tags,
        "totals": dict(pool_totals),
        "n_windows_total": int(sum(pool_totals.values())),
        "recipe": {
            "window_sec": base.WINDOW_SEC,
            "hop_sec": base.HOP_SEC,
            "epoch_pre_q1_sec": base.EPOCH_PRE,
            "epoch_end_before_q1_sec": base.EPOCH_POST,
            "target_sr": base.TARGET_SR,
            "label_rule": "Q1>Q2 concentration; Q1<Q2 mind_wandering",
        },
        "files": files,
        "round": "more_ds001787_subjects",
    }
    for s in results["subjects"]:
        tag = s["tag"]
        rollup["ds001787_attention"][tag] = {
            "n_windows_per_label": s["counts"],
            "n_windows_total": s["n_windows_total"],
            "npz_sha256": s["npz_sha256"],
            "group": s["group"],
            "channel_strategy": s["channel_strategy"],
        }
    # keep prior tag entries if present in manifests
    for tag in results["prior_subjects"]:
        man = json.loads(
            (base.EXPORT / tag / f"ds001787_{tag}_attention_manifest.json").read_text()
        )
        rollup["ds001787_attention"][tag] = {
            "n_windows_per_label": man["n_windows_per_label"],
            "n_windows_total": man["n_windows_total"],
            "npz_sha256": man["npz_sha256"],
            "group": man.get("group"),
            "channel_strategy": man.get("channel_strategy"),
        }
    rollup["updated_utc"] = datetime.now(timezone.utc).isoformat()
    rollup_path.write_text(json.dumps(rollup, indent=2) + "\n")

    summary_path = ROOT / "exports/more_ds001787_subjects_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("totals_new", "n_windows_new", "pool_totals", "n_windows_pool", "why")}, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

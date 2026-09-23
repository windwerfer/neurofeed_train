#!/usr/bin/env python3
"""Expand ds001787 attention corpus with more ALLOW subjects (local-first, no Kaggle)."""
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

# Round 3: prefer subjects with both classes in behavioral logs (for val balance).
SUBJECTS: List[Dict[str, str]] = [
    {"sub": "005", "ses": "01", "group": "expert", "log_candidates": ["sub05_info.txt", "sub05_2_info.txt"]},
    {"sub": "006", "ses": "01", "group": "expert", "log_candidates": ["sub06_info.txt", "sub06_2_info.txt"]},
    {"sub": "015", "ses": "01", "group": "novice", "log_candidates": ["sub15_info.txt"]},
    {"sub": "020", "ses": "01", "group": "novice", "log_candidates": ["sub20_info.txt"]},
]

PRIOR = [
    "sub001_ses01",
    "sub002_ses01",
    "sub003_ses01",
    "sub004_ses01",
    "sub013_ses01",
    "sub014_ses01",
    "sub017_ses01",
    "sub019_ses01",
]

OPENNEURO_BASE = "https://s3.amazonaws.com/openneuro.org/ds001787"
PAUSE_SEC = 4.0
STEP = "expand_attention_ds001787"


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


def main() -> None:
    base.LOG_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "step": STEP,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_subjects": list(PRIOR),
        "subjects": [],
        "why": (
            "Complete local attention corpora for concentration/mind_wandering; "
            "add subjects with both classes so val is not concentration-only"
        ),
        "doi": "doi:10.18112/openneuro.ds001787.v1.1.1",
        "license_spdx": "CC0-1.0",
        "hard_reject": "Do not map events value 2/4 to classes",
        "label_rule": "Q1>Q2→concentration; Q1<Q2→mind_wandering; ties dropped",
        "muse_proxy": "AF7/AF8 + P9/P10→TP9/TP10 (BioSemi64)",
    }
    for spec in SUBJECTS:
        ensure_subject_files(spec)

    totals = Counter()
    for spec in SUBJECTS:
        info = base.process_subject(spec)
        man_path = Path(info["manifest"])
        man = json.loads(man_path.read_text())
        man["step"] = STEP
        man_path.write_text(json.dumps(man, indent=2) + "\n")
        shutil.copy2(man_path, base.WINDOWS_PKG / man_path.name)
        shutil.copy2(Path(info["npz"]), base.WINDOWS_PKG / Path(info["npz"]).name)
        results["subjects"].append(info)
        for k, v in info["counts"].items():
            totals[k] += v
        print(
            f"done {info['tag']}: {info['counts']} total={info['n_windows_total']}",
            flush=True,
        )

    results["totals_new"] = dict(totals)
    results["n_windows_new"] = int(sum(totals.values()))

    pool_tags = PRIOR + [s["tag"] for s in results["subjects"]]
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

    prov_path = base.CACHE / f"provenance_{STEP}.json"
    provenance = {
        "step": STEP,
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

    summary_path = ROOT / f"exports/{STEP}_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: results[k]
                for k in (
                    "totals_new",
                    "n_windows_new",
                    "pool_totals",
                    "n_windows_pool",
                    "pool_subjects",
                )
            },
            indent=2,
        )
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

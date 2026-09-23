#!/usr/bin/env python3
"""Invented continuum step: add 4 more ds003969 subjects (2 htr + 2 ctr) for attention pool."""
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

# New subjects only — avoid overlap with prior 001/002/025.
SUBJECTS: List[Dict[str, str]] = [
    {
        "sub": "003",
        "group": "htr",
        "first_session": "meditation",
        "tree": "46edec2e4113231cae44612ac965f8cc1d8c5668",
    },
    {
        "sub": "004",
        "group": "htr",
        "first_session": "thinking",
        "tree": "96bd18f5d5793fd2eb876c77a5df2c4fab5c288a",
    },
    {
        "sub": "026",
        "group": "ctr",
        "first_session": "meditation",
        "tree": "e67c6986869bc71a6a2465452cb6692f15485185",
    },
    {
        "sub": "027",
        "group": "ctr",
        "first_session": "thinking",
        "tree": "cc66c4b71ed4ec2f04da2e7e7a56ac973194f23a",
    },
]

PRIOR = ["sub001", "sub002", "sub025"]


def main() -> None:
    base.CACHE.mkdir(parents=True, exist_ok=True)
    base.EXPORT.mkdir(parents=True, exist_ok=True)
    base.WINDOWS_PKG.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "step": "more_ds003969_subjects",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_subjects": list(PRIOR),
        "subjects": [],
        "why": (
            "Expand Muse-proximal ds003969 attention pool after subject-holdout near chance "
            "(attention_only + ds001787 expanded retrain); add 2 htr + 2 ctr"
        ),
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "tasks": list(base.TASKS),
        "label_rule": "med*→concentration; think*→mind_wandering",
    }
    totals = Counter()
    provenance = {
        "step": "more_ds003969_subjects",
        "source": "https://openneuro.org/datasets/ds003969",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "added_utc": datetime.now(timezone.utc).isoformat(),
        "recordings": [],
    }

    for spec in SUBJECTS:
        recs = base.fetch_subject(spec)
        info = base.process_subject(spec, recs)
        man_path = Path(info["manifest"])
        man = json.loads(man_path.read_text())
        man["step"] = "more_ds003969_subjects"
        man_path.write_text(json.dumps(man, indent=2) + "\n")
        shutil.copy2(man_path, base.WINDOWS_PKG / man_path.name)
        npz_path = Path(info["npz"])
        shutil.copy2(npz_path, base.WINDOWS_PKG / npz_path.name)
        results["subjects"].append(info)
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

    results["totals_new"] = dict(totals)
    results["n_windows_new"] = int(sum(totals.values()))

    pool_tags = PRIOR + [s["tag"] for s in results["subjects"]]
    pool_totals = Counter()
    for tag in pool_tags:
        man = json.loads(
            (base.EXPORT / tag / f"ds003969_{tag}_attention_manifest.json").read_text()
        )
        for k, v in man["n_windows_per_label"].items():
            pool_totals[k] += int(v)
    results["pool_subjects"] = pool_tags
    results["pool_totals"] = dict(pool_totals)
    results["n_windows_pool"] = int(sum(pool_totals.values()))

    prov_path = base.CACHE / "provenance_more_ds003969_subjects.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    results["provenance_path"] = str(prov_path)

    rollup_path = base.WINDOWS_PKG / "manifest.json"
    rollup: Dict[str, Any] = {}
    if rollup_path.exists():
        try:
            rollup = json.loads(rollup_path.read_text())
        except json.JSONDecodeError:
            rollup = {}
    entry = rollup.setdefault("ds003969_attention", {})
    entry["updated_utc"] = datetime.now(timezone.utc).isoformat()
    entry["subjects"] = pool_tags
    entry["totals"] = results["pool_totals"]
    entry["n_windows_total"] = results["n_windows_pool"]
    entry["recipe"] = {
        "window_sec": base.WINDOW_SEC,
        "hop_sec": base.HOP_SEC,
        "edge_trim_sec": base.EDGE_TRIM_SEC,
        "max_windows_per_block": base.MAX_WINDOWS_PER_BLOCK,
        "target_sr": base.TARGET_SR,
        "tasks": list(base.TASKS),
        "label_rule": "med* concentration; think* mind_wandering",
        "channel_proxy": "AF7/AF8/TP7/TP8→AF7/AF8/TP9/TP10",
    }
    files = entry.setdefault("files", [])
    for s in results["subjects"]:
        tag = s["tag"]
        for name in (
            f"ds003969_{tag}_attention_windows.npz",
            f"ds003969_{tag}_attention_manifest.json",
        ):
            if name not in files:
                files.append(name)
        entry[tag] = {
            "n_windows_per_label": s["counts"],
            "n_windows_total": s["n_windows_total"],
            "npz_sha256": s["npz_sha256"],
            "group": s["group"],
            "channel_strategy": s["channel_strategy"],
        }
    rollup["updated_utc"] = datetime.now(timezone.utc).isoformat()
    rollup_path.write_text(json.dumps(rollup, indent=2) + "\n")

    summary_path = ROOT / "exports/more_ds003969_subjects_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(
        json.dumps(
            {
                "totals_new": results["totals_new"],
                "n_windows_new": results["n_windows_new"],
                "pool_totals": results["pool_totals"],
                "n_windows_pool": results["n_windows_pool"],
                "subjects": [s["tag"] for s in results["subjects"]],
            },
            indent=2,
        )
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

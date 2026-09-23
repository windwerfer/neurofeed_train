#!/usr/bin/env python3
"""Expand ds003969 attention corpus with more ALLOW subjects (local-first, no Kaggle)."""
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

SUBJECTS: List[Dict[str, str]] = [
    {
        "sub": "005",
        "group": "htr",
        "first_session": "meditation",
        "tree": "c7a43e9fabf21a66995f2dff207ee1ba68efae61",
    },
    {
        "sub": "006",
        "group": "htr",
        "first_session": "thinking",
        "tree": "4d216de17d4c5495dc9361333f5f86c48a734887",
    },
    {
        "sub": "028",
        "group": "ctr",
        "first_session": "meditation",
        "tree": "5f03248af280d35207cb19ebefd1944b404a554b",
    },
    {
        "sub": "029",
        "group": "ctr",
        "first_session": "thinking",
        "tree": "8bc9ddb5c4cefcfc625abd8f52d4147c5ab488d6",
    },
]

PRIOR = ["sub001", "sub002", "sub003", "sub004", "sub025", "sub026", "sub027"]
STEP = "expand_attention_ds003969"


def main() -> None:
    base.CACHE.mkdir(parents=True, exist_ok=True)
    base.EXPORT.mkdir(parents=True, exist_ok=True)
    base.WINDOWS_PKG.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "step": STEP,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_subjects": list(PRIOR),
        "subjects": [],
        "why": (
            "Complete local attention corpora; add 2 htr + 2 ctr for stronger "
            "subject holdouts (prior attention holdouts near chance)"
        ),
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "tasks": list(base.TASKS),
        "label_rule": "med*→concentration; think*→mind_wandering",
        "muse_proxy": "AF7/AF8 native; TP7/TP8→TP9/TP10 proxies",
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

    for spec in SUBJECTS:
        recs = base.fetch_subject(spec)
        info = base.process_subject(spec, recs)
        man_path = Path(info["manifest"])
        man = json.loads(man_path.read_text())
        man["step"] = STEP
        man_path.write_text(json.dumps(man, indent=2) + "\n")
        shutil.copy2(man_path, base.WINDOWS_PKG / man_path.name)
        shutil.copy2(Path(info["npz"]), base.WINDOWS_PKG / Path(info["npz"]).name)
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
            (base.EXPORT / tag / f"ds003969_{tag}_attention_manifest.json").read_text()
        )
        for k, v in man["n_windows_per_label"].items():
            pool_totals[k] += int(v)
    results["pool_subjects"] = pool_tags
    results["pool_totals"] = dict(pool_totals)
    results["n_windows_pool"] = int(sum(pool_totals.values()))

    prov_path = base.CACHE / f"provenance_{STEP}.json"
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

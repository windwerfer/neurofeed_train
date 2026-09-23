#!/usr/bin/env python3
"""Discover all Head C window manifests, rewrite fixed subject splits, validate."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util

def _load_expand():
    spec = importlib.util.spec_from_file_location(
        "expand_head_c_sleep_corpus",
        ROOT / "scripts" / "expand_head_c_sleep_corpus.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

EXPORTS = ROOT / "exports"
CORPUS = ROOT / "datasets/vigilance_sleep_edf"


def discover() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for man_path in sorted(EXPORTS.glob("windows_*/**/*_n1slice_manifest.json")):
        m = json.loads(man_path.read_text())
        if not m.get("head_c_labels"):
            continue
        rid = m.get("recording_id") or m["tag"].upper()
        sid = m.get("subject_id")
        if not sid:
            if rid.startswith("SC") and len(rid) >= 5:
                sid = "SC4" + rid[3:5]
            elif rid.startswith("ST") and len(rid) >= 5:
                sid = "ST7" + rid[3:5]
            else:
                sid = rid
        items.append(
            {
                "subject_id": sid,
                "recording_id": rid,
                "tag": m["tag"],
                "night": str(m.get("night") or (rid[-1] if rid[-1].isdigit() else "1")),
                "study": m.get("study"),
                "source": m.get("source", "sleep_edf"),
                "ok": True,
                "n_windows": m.get("n_windows_total"),
                "manifest": str(man_path),
            }
        )
    # de-dupe by recording_id
    by_rid = {i["recording_id"]: i for i in items}
    return list(by_rid.values())


def main() -> None:
    items = discover()
    policy = _load_expand().write_splits(items)
    # validate
    rc = subprocess.call([sys.executable, str(ROOT / "scripts/dataset/validate_splits.py")])
    summary = {
        "step": "finalize_head_c_corpus",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "counts": policy["counts"],
        "vs_target_120": {
            "unique_subjects": policy["counts"]["n_subjects"],
            "remaining": max(0, 120 - policy["counts"]["n_subjects"]),
            "ready_to_train": policy["counts"]["n_subjects"] >= 100,
        },
        "by_source": {},
        "validate_rc": rc,
        "montage_report": policy.get("montage_report"),
    }
    for it in items:
        src = it.get("source") or "unknown"
        summary["by_source"].setdefault(src, {"subjects": set(), "recordings": 0, "windows": 0})
        summary["by_source"][src]["subjects"].add(it["subject_id"])
        summary["by_source"][src]["recordings"] += 1
        summary["by_source"][src]["windows"] += int(it.get("n_windows") or 0)
    for src, d in summary["by_source"].items():
        d["subjects"] = sorted(d["subjects"])
        d["n_subjects"] = len(d["subjects"])
    out = EXPORTS / "finalize_head_c_corpus_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("counts", "vs_target_120", "by_source", "validate_rc")}, indent=2))
    print(f"wrote {out}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stamp montage_id=muse4 on attention window manifests (never crown; no Sleep-EDF invent)."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONTAGE_ID = "muse4"
MODEL_ORDER = ["AF7", "AF8", "TP9", "TP10"]
CORPORA = {
    "attention_ds003969": ROOT / "exports" / "windows_ds003969",
    "attention_ds001787": ROOT / "exports" / "windows_ds001787",
}

def stamp_manifest(path: Path) -> bool:
    doc = json.loads(path.read_text())
    changed = False
    if doc.get("montage_id") != MONTAGE_ID:
        doc["montage_id"] = MONTAGE_ID
        changed = True
    if doc.get("channel_order_model") != MODEL_ORDER:
        doc["channel_order_model"] = list(MODEL_ORDER)
        changed = True
    # ensure policy note
    note = "muse4 only; never mix with crown8 in one example"
    if doc.get("montage_policy") != note:
        doc["montage_policy"] = note
        changed = True
    if changed:
        doc["montage_stamped_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(doc, indent=2) + "\n")
    return changed

def main() -> None:
    summary = {"step": "stamp_montage_muse4", "created_utc": datetime.now(timezone.utc).isoformat(), "stamped": [], "skipped": []}
    for corpus, win_root in CORPORA.items():
        if not win_root.exists():
            continue
        for man in sorted(win_root.rglob("*_attention_manifest.json")):
            if stamp_manifest(man):
                summary["stamped"].append(str(man.relative_to(ROOT)))
            else:
                summary["skipped"].append(str(man.relative_to(ROOT)))
        # also datasets/*/windows/*_manifest.json symlinks targets already stamped
        ds_win = ROOT / "datasets" / corpus / "windows"
        for man in sorted(ds_win.glob("*_manifest.json")):
            # dataset manifests may be thinner copies — stamp if json
            try:
                if stamp_manifest(man.resolve() if man.is_symlink() else man):
                    summary["stamped"].append(str(man.relative_to(ROOT)))
            except Exception as e:
                summary.setdefault("errors", []).append({"path": str(man), "error": repr(e)})
    out = ROOT / "exports" / "stamp_montage_muse4_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"stamped={len(summary['stamped'])} skipped={len(summary['skipped'])} -> {out}")

if __name__ == "__main__":
    main()

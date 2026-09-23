#!/usr/bin/env python3
"""Sync muse4 attention windows + QC + annotations + splits into Kaggle pack.

Updates kaggle_datasets/muse-eeg-heads-windows (version bump via kaggle CLI separately).
Does NOT upload raw BDFs. Resolves symlinks to real npz bytes.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "kaggle_datasets" / "muse-eeg-heads-windows"
CORPORA = {
    "attention_ds001787": {
        "prefix": "ds001787",
        "flat_windows": "{prefix}_{rid}_attention_windows.npz",
        "flat_qc": "{prefix}_{rid}_attention_qc.npz",
        "flat_manifest": "{prefix}_{rid}_attention_manifest.json",
    },
    "attention_ds003969": {
        "prefix": "ds003969",
        "flat_windows": "{prefix}_{rid}_attention_windows.npz",
        "flat_qc": "{prefix}_{rid}_attention_qc.npz",
        "flat_manifest": "{prefix}_{rid}_attention_manifest.json",
    },
}


def copy_resolved(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def main() -> None:
    PACK.mkdir(parents=True, exist_ok=True)
    copied = {"windows": 0, "qc": 0, "manifests": 0}
    for corpus, meta in CORPORA.items():
        win_dir = ROOT / "datasets" / corpus / "windows"
        prefix = meta["prefix"]
        for npz in sorted(win_dir.glob("*_windows.npz")):
            rid = npz.name.replace("_windows.npz", "")
            dst_w = PACK / meta["flat_windows"].format(prefix=prefix, rid=rid)
            copy_resolved(npz, dst_w)
            copied["windows"] += 1
            qc = win_dir / f"{rid}_qc.npz"
            if qc.exists():
                dst_q = PACK / meta["flat_qc"].format(prefix=prefix, rid=rid)
                copy_resolved(qc, dst_q)
                copied["qc"] += 1
            man = win_dir / f"{rid}_manifest.json"
            if man.exists():
                dst_m = PACK / meta["flat_manifest"].format(prefix=prefix, rid=rid)
                copy_resolved(man, dst_m)
                copied["manifests"] += 1

        # annotations + splits (nested under pack for clarity)
        for sub in ("annotations", "splits"):
            src = ROOT / "datasets" / corpus / sub
            if not src.exists():
                continue
            dst = PACK / "attention" / corpus / sub
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".gitkeep"))

    # common montages
    common_dst = PACK / "common"
    common_dst.mkdir(exist_ok=True)
    for name in ("montages.json", "label_maps.json", "schema_windows.md"):
        src = ROOT / "datasets" / "common" / name
        if src.exists():
            shutil.copy2(src, common_dst / name)

    readme = f"""# muse-eeg-heads-windows

Private Muse-proxy windows for windwerfer/muse-eeg-heads.

## Contents

- Sleep-EDF N1-slice windows (SC4001–SC4041) — vigilance / Head A smoke
- **Attention muse4** OpenNeuro ds001787 + ds003969 windows + QC masks
- `attention/<corpus>/annotations|splits/` — indexes for LOSO
- `common/montages.json` — muse4/crown8 positions

## Attention naming

- `ds001787_<rid>_attention_windows.npz` + `_qc.npz` + `_manifest.json`
- `ds003969_<rid>_attention_windows.npz` + `_qc.npz` + `_manifest.json`
- Channels AF7,AF8,TP9,TP10 @ 256 Hz, 2 s (T=512). `montage_id=muse4`.

## License

- Sleep-EDF: PhysioNet ODC-By
- OpenNeuro attention: per corpus ATTRIBUTION (typically CC0)
- Private dataset — do not make public. No raw BDFs.

Synced UTC: {datetime.now(timezone.utc).isoformat()}
Copied: {json.dumps(copied)}
"""
    (PACK / "README.md").write_text(readme)
    meta_path = PACK / "dataset-metadata.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "title": "muse-eeg-heads-windows",
                    "id": "windwerfer/muse-eeg-heads-windows",
                    "licenses": [{"name": "other"}],
                    "isPrivate": True,
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps({"pack": str(PACK), "copied": copied}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Link/copy existing window exports + raw EDFs into datasets/ layout.

Default: symlink windows npz+manifest and raw Sleep-EDF PSG/hypnogram files.
Does not duplicate large binaries.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
EXPORTS = ROOT / "exports"
CACHE_SLEEP = (
    ROOT
    / "kaggle_datasets"
    / "muse-eeg-heads-cache"
    / "data"
    / "sleep-edfx-pilot"
    / "sleep-cassette"
)
TMP_SLEEP = Path("/tmp/kaggle_out3/data/sleep-edfx-pilot/sleep-cassette")

SLEEP_NIGHTS = {
    "SC4001": "sc4001",
    "SC4002": "sc4002",
    "SC4011": "sc4011",
    "SC4021": "sc4021",
    "SC4031": "sc4031",
    "SC4041": "sc4041",
}

# Hypnogram letter varies (EC vs EH)
HYPN_O_CANDIDATES = {
    "SC4001": ["SC4001EC-Hypnogram.edf"],
    "SC4002": ["SC4002EC-Hypnogram.edf"],
    "SC4011": ["SC4011EH-Hypnogram.edf", "SC4011EC-Hypnogram.edf"],
    "SC4021": ["SC4021EH-Hypnogram.edf", "SC4021EC-Hypnogram.edf"],
    "SC4031": ["SC4031EC-Hypnogram.edf"],
    "SC4041": ["SC4041EC-Hypnogram.edf"],
}


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copied"
    dst.symlink_to(src.resolve())
    return "symlinked"


def find_sleep_raw(name: str) -> Path | None:
    for base in (CACHE_SLEEP, TMP_SLEEP):
        p = base / name
        if p.exists():
            return p
    return None


def migrate_sleep(mode: str) -> list[dict]:
    out: list[dict] = []
    win_dir = DATASETS / "vigilance_sleep_edf" / "windows"
    raw_dir = DATASETS / "vigilance_sleep_edf" / "raw"
    for rid, tag in SLEEP_NIGHTS.items():
        src_dir = EXPORTS / f"windows_{tag}"
        npz = src_dir / f"sleep_edf_{tag}_n1slice_windows.npz"
        man = src_dir / f"sleep_edf_{tag}_n1slice_manifest.json"
        entry: dict = {"recording_id": rid, "tag": tag}
        if npz.exists():
            dest = win_dir / f"{rid}_windows.npz"
            entry["windows"] = {
                "src": str(npz),
                "dst": str(dest.relative_to(ROOT)),
                "action": link_or_copy(npz, dest, mode),
            }
        else:
            entry["windows"] = {"missing": str(npz)}
        if man.exists():
            dest_m = win_dir / f"{rid}_manifest.json"
            entry["manifest"] = {
                "src": str(man),
                "dst": str(dest_m.relative_to(ROOT)),
                "action": link_or_copy(man, dest_m, mode),
            }
        psg_name = f"{rid}E0-PSG.edf"
        psg = find_sleep_raw(psg_name)
        if psg:
            dest_p = raw_dir / psg_name
            entry["psg"] = {
                "src": str(psg),
                "dst": str(dest_p.relative_to(ROOT)),
                "action": link_or_copy(psg, dest_p, mode),
            }
        else:
            entry["psg"] = {"missing": psg_name}
        hyp = None
        for cand in HYPN_O_CANDIDATES.get(rid, []):
            hyp = find_sleep_raw(cand)
            if hyp:
                dest_h = raw_dir / cand
                entry["hypnogram"] = {
                    "src": str(hyp),
                    "dst": str(dest_h.relative_to(ROOT)),
                    "action": link_or_copy(hyp, dest_h, mode),
                }
                break
        if hyp is None:
            entry["hypnogram"] = {"missing": HYPN_O_CANDIDATES.get(rid)}
        out.append(entry)
    return out


def migrate_attention(
    corpus: str, export_subdir: str, file_prefix: str, tags: list[str], mode: str
) -> list[dict]:
    out: list[dict] = []
    win_dir = DATASETS / corpus / "windows"
    src_root = EXPORTS / export_subdir
    for tag in tags:
        src_dir = src_root / tag
        npz = src_dir / f"{file_prefix}_{tag}_attention_windows.npz"
        man = src_dir / f"{file_prefix}_{tag}_attention_manifest.json"
        entry: dict = {"recording_id": tag}
        if npz.exists():
            dest = win_dir / f"{tag}_windows.npz"
            entry["windows"] = {
                "src": str(npz),
                "dst": str(dest.relative_to(ROOT)),
                "action": link_or_copy(npz, dest, mode),
            }
        else:
            entry["windows"] = {"missing": str(npz)}
        if man.exists():
            dest_m = win_dir / f"{tag}_manifest.json"
            entry["manifest"] = {
                "src": str(man),
                "dst": str(dest_m.relative_to(ROOT)),
                "action": link_or_copy(man, dest_m, mode),
            }
        out.append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    args = ap.parse_args()

    ds1787_tags = sorted(
        p.name for p in (EXPORTS / "windows_ds001787").iterdir() if p.is_dir()
    )
    ds3969_tags = sorted(
        p.name for p in (EXPORTS / "windows_ds003969").iterdir() if p.is_dir()
    )

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "vigilance_sleep_edf": migrate_sleep(args.mode),
        "attention_ds001787": migrate_attention(
            "attention_ds001787",
            "windows_ds001787",
            "ds001787",
            ds1787_tags,
            args.mode,
        ),
        "attention_ds003969": migrate_attention(
            "attention_ds003969",
            "windows_ds003969",
            "ds003969",
            ds3969_tags,
            args.mode,
        ),
    }
    out_path = DATASETS / "migration_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(ROOT)}")

    def count_ok(rows: list[dict]) -> tuple[int, int]:
        ok = sum(1 for r in rows if "missing" not in r.get("windows", {}))
        return ok, len(rows)

    for k in ("vigilance_sleep_edf", "attention_ds001787", "attention_ds003969"):
        ok, n = count_ok(report[k])
        print(f"  {k}: {ok}/{n} windows linked")


if __name__ == "__main__":
    main()

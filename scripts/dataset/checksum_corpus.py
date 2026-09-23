#!/usr/bin/env python3
"""Compute SHA256 for raw + windows files; write checksums.json per corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"

CORPORA = (
    "vigilance_sleep_edf",
    "attention_ds001787",
    "attention_ds003969",
)

BINARY_GLOBS = (
    "raw/**/*",
    "windows/**/*.npz",
    "windows/**/*.edf",
    "windows/**/*.bdf",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def resolve_path(path: Path) -> Path:
    if path.is_symlink():
        return path.resolve()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATASETS)
    ap.add_argument(
        "--verify-manifest",
        action="store_true",
        help="compare to npz_sha256 in sibling *_manifest.json when present",
    )
    args = ap.parse_args()
    root: Path = args.root

    for corpus in CORPORA:
        cdir = root / corpus
        entries: list[dict] = []
        seen: set[Path] = set()
        for pattern in ("raw/*", "windows/*"):
            for path in sorted(cdir.glob(pattern)):
                if not path.is_file() and not path.is_symlink():
                    continue
                if path.name == ".gitkeep":
                    continue
                if path.suffix.lower() not in {
                    ".edf",
                    ".bdf",
                    ".npz",
                    ".json",
                }:
                    # still hash manifests alongside binaries for completeness of windows/
                    if path.suffix.lower() != ".json":
                        continue
                # Only checksum binaries + skip re-hashing large json repeatedly? include json manifests
                if path.suffix.lower() == ".json" and "manifest" not in path.name:
                    continue
                rp = resolve_path(path)
                if rp in seen:
                    continue
                seen.add(rp)
                digest = sha256_file(rp)
                ent = {
                    "relpath": str(path.relative_to(ROOT)),
                    "resolved": str(rp),
                    "bytes": rp.stat().st_size,
                    "sha256": digest,
                    "symlink": path.is_symlink(),
                }
                if args.verify_manifest and path.suffix == ".npz":
                    man = path.with_name(
                        path.name.replace("_windows.npz", "_manifest.json")
                    )
                    if man.exists():
                        m = json.loads(man.read_text())
                        expected = m.get("npz_sha256")
                        ent["manifest_sha256"] = expected
                        ent["sha256_match"] = (
                            expected == digest if expected else None
                        )
                entries.append(ent)
        out = {
            "corpus": corpus,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_files": len(entries),
            "files": entries,
        }
        out_path = cdir / "annotations" / "checksums.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2) + "\n")
        matched = sum(1 for e in entries if e.get("sha256_match") is True)
        checked = sum(1 for e in entries if "sha256_match" in e)
        print(
            f"{corpus}: {len(entries)} files checksummed"
            + (f"; npz match {matched}/{checked}" if args.verify_manifest else "")
        )


if __name__ == "__main__":
    main()

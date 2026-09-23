#!/usr/bin/env python3
"""Expand Head C sleep-stage corpus toward ~120 unique subjects.

Sources (ALLOW / open):
  - PhysioNet Sleep-EDF Expanded cassette (~78) + telemetry (~22) — ODC-By
  - PhysioNet HMC sleep staging (optional top-up) — CC-BY-4.0

Recipe (reuse Sleep-EDF N1-slice pipeline):
  around first N1 / stage 1; pre 20 min / post 40 min; window 2 s / hop 0.5 s;
  majority 0.7; Muse4 proxy; Head A W→drowsy N1→hypnagogic; stage_raw+stage_coarse.

Resume-friendly: skips existing raw EDFs and existing windows npz.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.head_a import HEAD_A_BINARY_LABELS, labels_to_ids
from src.sleep_edf import (
    PROXY_NOTE,
    STAGE_TO_COARSE,
    STAGE_TO_HEAD_A,
    load_sleep_edf_recording,
    majority_stage_in_window,
    stage_to_coarse,
    windows_with_head_a_labels,
)

SEED = 42
WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0
PRE_SEC = 20 * 60
POST_SEC = 40 * 60
MAJORITY = 0.7
PAUSE_SEC = 1.5
BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"
HMC_BASE_URL = "https://physionet.org/files/hmc-sleep-staging/1.1/"

CACHE_ROOT = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data"
SLEEP_ROOT = CACHE_ROOT / "sleep-edfx-pilot"
CASS_DIR = SLEEP_ROOT / "sleep-cassette"
TELEM_DIR = SLEEP_ROOT / "sleep-telemetry"
HMC_DIR = CACHE_ROOT / "hmc-sleep-staging/recordings"
WINDOWS_PKG = ROOT / "kaggle_datasets/muse-eeg-heads-windows"
CORPUS = ROOT / "datasets/vigilance_sleep_edf"
EXPORTS = ROOT / "exports"
PROGRESS = EXPORTS / "head_c_expand"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, pause: float = PAUSE_SEC) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return False
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"[dl] {url} -> {dest.name}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "muse-eeg-heads/head-c-expand"})
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)
    tmp.replace(dest)
    if pause > 0:
        time.sleep(pause)
    return True


def load_sha_map(sha_path: Path) -> Dict[str, str]:
    """relpath -> sha256"""
    out: Dict[str, str] = {}
    if not sha_path.exists():
        return out
    for line in sha_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[-1]] = parts[0]
    return out


def hypnogram_index(sha_map: Dict[str, str]) -> Dict[str, str]:
    """recording prefix SC4001 / ST7011 -> relative hypnogram path"""
    idx: Dict[str, str] = {}
    for rel in sha_map:
        if "Hypnogram" not in rel:
            continue
        name = Path(rel).name
        idx[name[:6]] = rel
    return idx


def inventory_sleep_edf(
    records_path: Path, sha_map: Dict[str, str], night1_only: bool = True
) -> List[Dict[str, Any]]:
    """One preferred night per unique subject (night 1 if present else night 2)."""
    hyp_idx = hypnogram_index(sha_map)
    by_subj: Dict[str, List[Tuple[str, str, str]]] = {}
    # (night, study, psg_rel)
    for line in records_path.read_text().splitlines():
        line = line.strip()
        if not line.endswith("PSG.edf"):
            continue
        name = Path(line).name
        if name.startswith("SC"):
            m = re.match(r"SC4(\d{2})(\d)", name)
            if not m:
                continue
            sid, night = m.group(1), m.group(2)
            subj = f"SC4{sid}"
            study = "cassette"
        elif name.startswith("ST"):
            m = re.match(r"ST7(\d{2})(\d)", name)
            if not m:
                continue
            sid, night = m.group(1), m.group(2)
            subj = f"ST7{sid}"
            study = "telemetry"
        else:
            continue
        by_subj.setdefault(subj, []).append((night, study, line))

    items: List[Dict[str, Any]] = []
    for subj, nights in sorted(by_subj.items()):
        nights_sorted = sorted(nights, key=lambda t: t[0])
        chosen = None
        if night1_only:
            for n, study, rel in nights_sorted:
                if n == "1":
                    chosen = (n, study, rel)
                    break
            if chosen is None:
                chosen = nights_sorted[0]  # e.g. subj 36/52 missing night1
        else:
            # keep all nights as separate recordings later — here still one pass list
            for n, study, rel in nights_sorted:
                prefix = Path(rel).name[:6]
                hyp_rel = hyp_idx.get(prefix)
                if not hyp_rel:
                    raise KeyError(f"no hypnogram for {prefix}")
                rid = prefix  # SC4001 / ST7011
                items.append(
                    {
                        "subject_id": subj,
                        "recording_id": rid,
                        "tag": rid.lower(),
                        "study": study,
                        "night": n,
                        "psg_rel": rel,
                        "hyp_rel": hyp_rel,
                        "source": "sleep_edf",
                    }
                )
            continue
        n, study, rel = chosen
        prefix = Path(rel).name[:6]
        hyp_rel = hyp_idx.get(prefix)
        if not hyp_rel:
            raise KeyError(f"no hypnogram for {prefix}")
        items.append(
            {
                "subject_id": subj,
                "recording_id": prefix,
                "tag": prefix.lower(),
                "study": study,
                "night": n,
                "psg_rel": rel,
                "hyp_rel": hyp_rel,
                "source": "sleep_edf",
            }
        )
    return items


def ensure_sleep_pair(item: Dict[str, Any]) -> Tuple[Path, Path]:
    dest_dir = CASS_DIR if item["study"] == "cassette" else TELEM_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    psg = dest_dir / Path(item["psg_rel"]).name
    hyp = dest_dir / Path(item["hyp_rel"]).name
    download(BASE_URL + item["psg_rel"], psg)
    download(BASE_URL + item["hyp_rel"], hyp)
    return psg, hyp


def build_night_windows(psg: Path, hyp: Path, out_dir: Path, tag: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"sleep_edf_{tag}_n1slice_windows.npz"
    man_path = out_dir / f"sleep_edf_{tag}_n1slice_manifest.json"
    if npz_path.exists() and man_path.exists():
        man = json.loads(man_path.read_text())
        if man.get("head_c_labels") and "stage_raw_counts" in man:
            print(f"[{tag}] skip existing windows n={man.get('n_windows_total')}", flush=True)
            return {"skipped": True, "manifest": man, "npz_path": npz_path, "man_path": man_path}

    # Sleep-EDF uses "stage 1"; HMC path uses separate builder
    around = "stage 1"
    rec = load_sleep_edf_recording(
        psg,
        hyp,
        target_sr=TARGET_SR,
        around_stage=around,
        pre_sec=PRE_SEC,
        post_sec=POST_SEC,
    )
    X, labels, keep = windows_with_head_a_labels(
        rec["data"],
        rec["stages"],
        sfreq=TARGET_SR,
        window_sec=WINDOW_SEC,
        hop_sec=HOP_SEC,
        majority_frac=MAJORITY,
    )
    mask = [lab in HEAD_A_BINARY_LABELS for lab in labels]
    X = X[np.asarray(mask)] if len(labels) else X
    labels = [lab for lab, m in zip(labels, mask) if m]
    y = labels_to_ids(labels, HEAD_A_BINARY_LABELS) if labels else np.zeros((0,), dtype=np.int64)
    hop = int(round(HOP_SEC * TARGET_SR))
    starts = np.asarray([keep[i] * hop for i, m in enumerate(mask) if m], dtype=np.int64)
    win = int(round(WINDOW_SEC * TARGET_SR))
    stages = rec["stages"]
    stage_raw = []
    stage_coarse = []
    for st in starts:
        raw = majority_stage_in_window(stages, int(st), win)
        stage_raw.append(raw)
        stage_coarse.append(stage_to_coarse(raw))
    counts = dict(Counter(labels))
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y,
        starts=starts,
        label_names=np.asarray(HEAD_A_BINARY_LABELS),
        stage_raw=np.asarray(stage_raw, dtype=object),
        stage_coarse=np.asarray(stage_coarse, dtype=object),
    )
    manifest = {
        "psg_file": psg.name,
        "hypno_file": hyp.name,
        "psg_path": str(psg),
        "hypno_path": str(hyp),
        "tag": tag,
        "recording_id": meta["recording_id"],
        "subject_id": meta["subject_id"],
        "study": meta.get("study"),
        "source": meta.get("source", "sleep_edf"),
        "slice_start_sec": rec["slice_start_sec"],
        "pre_sec": PRE_SEC,
        "post_sec": POST_SEC,
        "around_stage": around,
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "majority_frac": MAJORITY,
        "target_sr": TARGET_SR,
        "channel_proxy_note": PROXY_NOTE,
        "montage": {
            "kind": "muse4_proxy",
            "ch_count": 4,
            "device_class": "professional",
            "source_channels": ["EEG Fpz-Cz", "EEG Pz-Oz"],
            "mapped_to": ["AF7", "AF8", "TP9", "TP10"],
        },
        "stage_to_label_map": {k: v for k, v in STAGE_TO_HEAD_A.items()},
        "stage_to_coarse_map": dict(STAGE_TO_COARSE),
        "n_windows_per_label": counts,
        "n_windows_total": int(X.shape[0]),
        "window_shape": list(X.shape),
        "stage_raw_counts": dict(Counter(stage_raw)),
        "stage_coarse_counts": dict(Counter(stage_coarse)),
        "head_c_labels": True,
        "npz_path": str(npz_path),
        "npz_sha256": sha256(npz_path),
        "psg_sha256": sha256(psg),
        "hypno_sha256": sha256(hyp),
        "label_list": list(HEAD_A_BINARY_LABELS),
        "created_utc": utc_now(),
        "license_attribution": {
            "Sleep-EDF": "PhysioNet ODC-By — attribute PhysioNet / Sleep-EDF Expanded",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
        "source_url_base": BASE_URL,
        "step": "expand_head_c_sleep_corpus",
    }
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[{tag}] slice={rec['slice_start_sec']} windows={X.shape} "
        f"counts={counts} coarse={dict(Counter(stage_coarse))}",
        flush=True,
    )
    return {"skipped": False, "manifest": manifest, "npz_path": npz_path, "man_path": man_path, "counts": counts}


def link_into_corpus(recording_id: str, tag: str, psg: Path, hyp: Path, npz: Path, man: Path) -> None:
    win_dir = CORPUS / "windows"
    raw_dir = CORPUS / "raw"
    win_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _relink(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    _relink(npz, win_dir / f"{recording_id}_windows.npz")
    _relink(man, win_dir / f"{recording_id}_manifest.json")
    _relink(psg, raw_dir / psg.name)
    _relink(hyp, raw_dir / hyp.name)
    # optional package mirror
    WINDOWS_PKG.mkdir(parents=True, exist_ok=True)
    shutil.copy2(npz, WINDOWS_PKG / npz.name)
    shutil.copy2(man, WINDOWS_PKG / man.name)


def write_splits(items_done: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Fixed subject-wise splits. Preserve prior SC400=test, SC403=val anchors."""
    recordings: Dict[str, Any] = {}
    subjects: Dict[str, Any] = {}
    for it in items_done:
        rid = it["recording_id"]
        sid = it["subject_id"]
        recordings[rid] = {
            "subject_id": sid,
            "night": int(it.get("night") or 1),
            "tag": it["tag"],
            "source": it.get("source", "sleep_edf"),
            "study": it.get("study"),
        }
        subjects.setdefault(sid, {"recordings": [], "notes": "", "source": it.get("source")})
        if rid not in subjects[sid]["recordings"]:
            subjects[sid]["recordings"].append(rid)

    # Preserve anchors from v0
    test = ["SC400"]
    val = ["SC403"]
    if "SC400" in subjects:
        subjects["SC400"]["notes"] = (
            "prior primary holdout; both nights held out when present for subject purity"
        )
    if "SC403" in subjects:
        subjects["SC403"]["notes"] = "prior secondary holdout / val anchor"

    remaining = sorted(s for s in subjects if s not in test and s not in val)
    # ~70/15/15 of remaining, deterministic
    n = len(remaining)
    n_test_extra = max(1, round(n * 0.15)) if n >= 10 else max(1, n // 6 or 1)
    n_val_extra = max(1, round(n * 0.15)) if n >= 10 else max(1, n // 6 or 1)
    # keep anchors; add extras from end (stable)
    test_extra = remaining[-n_test_extra:] if n_test_extra else []
    mid = remaining[-(n_test_extra + n_val_extra) : -n_test_extra] if n_val_extra else []
    train = [s for s in remaining if s not in test_extra and s not in mid]
    val = val + [s for s in mid if s not in val]
    test = test + [s for s in test_extra if s not in test]

    # Ensure every subject assigned
    assigned = set(train) | set(val) | set(test)
    for s in subjects:
        if s not in assigned:
            train.append(s)

    policy = {
        "corpus": "vigilance_sleep_edf",
        "policy": "fixed_subject_json",
        "policy_note": (
            "Option A: fixed subject-level train/val/test JSON. "
            "No subject appears in more than one split. "
            "SC400 remains test anchor; SC403 remains val anchor; "
            "remaining subjects split ~70/15/15 deterministically by sorted id."
        ),
        "created_utc": utc_now(),
        "subject_key_rule": (
            "Sleep-EDF: SC4ss / ST7ss from recording id; HMC: SN### (1:1 subject)."
        ),
        "recordings": recordings,
        "subjects": subjects,
        "splits": {"train": train, "val": val, "test": test},
        "counts": {
            "n_subjects": len(subjects),
            "n_recordings": len(recordings),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
        "target_subjects": 120,
        "montage_report": {
            "sleep_edf": {"ch_count": 4, "device_class": "professional", "proxy": "muse4 Fpz-Cz/Pz-Oz"},
            "hmc": {"ch_count": 4, "device_class": "professional", "proxy": "muse4 F4/C3/O2"},
        },
    }

    splits_dir = CORPUS / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    subjects_doc = {
        "corpus": policy["corpus"],
        "policy": policy["policy"],
        "subjects": subjects,
        "recordings": recordings,
        "created_utc": policy["created_utc"],
    }
    (splits_dir / "subjects.json").write_text(json.dumps(subjects_doc, indent=2) + "\n")
    for split_name in ("train", "val", "test"):
        subj = policy["splits"][split_name]
        recs: List[str] = []
        for sid in subj:
            recs.extend(subjects[sid]["recordings"])
        payload = {
            "corpus": policy["corpus"],
            "split": split_name,
            "policy": policy["policy"],
            "subjects": subj,
            "recordings": recs,
            "created_utc": policy["created_utc"],
        }
        (splits_dir / f"{split_name}_subjects.json").write_text(json.dumps(payload, indent=2) + "\n")
    (splits_dir / "split_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    return policy


def process_sleep_edf(limit: Optional[int], start_after: Optional[str], night1_only: bool) -> List[Dict[str, Any]]:
    sha_path = SLEEP_ROOT / "SHA256SUMS.txt"
    rec_path = SLEEP_ROOT / "RECORDS"
    if not sha_path.exists() or not rec_path.exists():
        download(BASE_URL + "SHA256SUMS.txt", sha_path, pause=0.5)
        download(BASE_URL + "RECORDS", rec_path, pause=0.5)
    sha_map = load_sha_map(sha_path)
    items = inventory_sleep_edf(rec_path, sha_map, night1_only=night1_only)
    # Always include existing SC4002 if present locally even when night1_only
    # (already on disk from pilot) — inventory night1_only skips night2 except missing-n1.
    # Keep SC4002 by injecting if night1_only and file exists.
    if night1_only:
        extra = {
            "subject_id": "SC400",
            "recording_id": "SC4002",
            "tag": "sc4002",
            "study": "cassette",
            "night": "2",
            "psg_rel": "sleep-cassette/SC4002E0-PSG.edf",
            "hyp_rel": hypnogram_index(sha_map)["SC4002"],
            "source": "sleep_edf",
        }
        if not any(i["recording_id"] == "SC4002" for i in items):
            # insert after SC4001
            items.append(extra)
            items = sorted(items, key=lambda x: (x["subject_id"], x["recording_id"]))

    if start_after:
        items = [i for i in items if i["recording_id"] > start_after]
    if limit is not None:
        items = items[:limit]

    done: List[Dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        print(f"\n=== [{i}/{len(items)}] {item['recording_id']} ({item['subject_id']}) ===", flush=True)
        try:
            psg, hyp = ensure_sleep_pair(item)
            out_dir = EXPORTS / f"windows_{item['tag']}"
            built = build_night_windows(psg, hyp, out_dir, item["tag"], item)
            link_into_corpus(item["recording_id"], item["tag"], psg, hyp, built["npz_path"], built["man_path"])
            entry = {**item, "n_windows": built["manifest"].get("n_windows_total"), "ok": True}
            done.append(entry)
        except Exception as e:
            print(f"[ERROR] {item['recording_id']}: {type(e).__name__}: {e}", flush=True)
            done.append({**item, "ok": False, "error": f"{type(e).__name__}: {e}"})
        # progress ping every 10
        if i % 10 == 0 or i == len(items):
            ok_subj = sorted({d["subject_id"] for d in done if d.get("ok")})
            ping = {
                "updated_utc": utc_now(),
                "processed_recordings": i,
                "ok_recordings": sum(1 for d in done if d.get("ok")),
                "unique_subjects_ok": len(ok_subj),
                "target_subjects": 120,
                "last_recording": item["recording_id"],
            }
            PROGRESS.mkdir(parents=True, exist_ok=True)
            (PROGRESS / "progress.json").write_text(json.dumps(ping, indent=2) + "\n")
            print(f"[progress] subjects_ok={ping['unique_subjects_ok']}/120", flush=True)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Max recordings this run")
    ap.add_argument("--start-after", type=str, default=None, help="Resume after recording id")
    ap.add_argument("--all-nights", action="store_true", help="Both nights when available")
    ap.add_argument("--skip-window", action="store_true", help="Download only")
    ap.add_argument("--splits-only", action="store_true", help="Rewrite splits from existing windows")
    args = ap.parse_args()

    PROGRESS.mkdir(parents=True, exist_ok=True)
    CASS_DIR.mkdir(parents=True, exist_ok=True)
    TELEM_DIR.mkdir(parents=True, exist_ok=True)

    if args.splits_only:
        # Discover from exports
        items = []
        for man in sorted(EXPORTS.glob("windows_*/sleep_edf_*_n1slice_manifest.json")):
            m = json.loads(man.read_text())
            rid = m.get("recording_id") or m["tag"].upper()
            sid = m.get("subject_id")
            if not sid:
                if rid.startswith("SC"):
                    sid = rid[:4] + rid[4:6] if len(rid) >= 6 else rid[:5]
                    # SC4001 -> SC400
                    sid = rid[:5] if rid.startswith("SC") else rid[:5]
                    sid = "SC4" + rid[3:5]
                elif rid.startswith("ST"):
                    sid = "ST7" + rid[3:5]
                else:
                    sid = rid
            items.append(
                {
                    "subject_id": m.get("subject_id") or sid,
                    "recording_id": rid,
                    "tag": m["tag"],
                    "night": str(m.get("night") or (rid[-1] if rid[-1].isdigit() else "1")),
                    "study": m.get("study"),
                    "source": m.get("source", "sleep_edf"),
                    "ok": True,
                }
            )
        policy = write_splits(items)
        print(json.dumps(policy["counts"], indent=2))
        return

    done = process_sleep_edf(
        limit=args.limit,
        start_after=args.start_after,
        night1_only=not args.all_nights,
    )
    ok = [d for d in done if d.get("ok")]
    # Also fold in any previously built nights not in this run
    existing_ok = {d["recording_id"] for d in ok}
    for man in EXPORTS.glob("windows_*/sleep_edf_*_n1slice_manifest.json"):
        m = json.loads(man.read_text())
        rid = m.get("recording_id") or m["tag"].upper()
        if rid in existing_ok:
            continue
        if rid.startswith("SC"):
            sid = "SC4" + rid[3:5]
            study = "cassette"
        elif rid.startswith("ST"):
            sid = "ST7" + rid[3:5]
            study = "telemetry"
        else:
            continue
        ok.append(
            {
                "subject_id": m.get("subject_id") or sid,
                "recording_id": rid,
                "tag": m["tag"],
                "night": rid[-1],
                "study": m.get("study") or study,
                "source": "sleep_edf",
                "ok": True,
                "n_windows": m.get("n_windows_total"),
            }
        )

    policy = write_splits(ok)
    summary = {
        "step": "expand_head_c_sleep_corpus",
        "completed_utc": utc_now(),
        "this_run": {
            "attempted": len(done),
            "ok": sum(1 for d in done if d.get("ok")),
            "failed": [d for d in done if not d.get("ok")],
        },
        "corpus_totals": policy["counts"],
        "vs_target_120": {
            "unique_subjects": policy["counts"]["n_subjects"],
            "remaining": max(0, 120 - policy["counts"]["n_subjects"]),
        },
        "montage": "(ch_count=4, professional) muse4 proxy Sleep-EDF Fpz-Cz/Pz-Oz",
        "paths": {
            "raw_cassette": str(CASS_DIR),
            "raw_telemetry": str(TELEM_DIR),
            "windows_exports": str(EXPORTS),
            "corpus": str(CORPUS),
            "progress": str(PROGRESS / "progress.json"),
        },
        "recordings_ok": [
            {"recording_id": d["recording_id"], "subject_id": d["subject_id"], "n_windows": d.get("n_windows")}
            for d in ok
        ],
    }
    out = EXPORTS / "expand_head_c_sleep_corpus_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("corpus_totals", "vs_target_120", "montage")}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

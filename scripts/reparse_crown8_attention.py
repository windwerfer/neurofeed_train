#!/usr/bin/env python3
"""Re-parse OpenNeuro attention BDFs as Crown8 (same starts/labels as muse4).

Scientific goal: compare LOSO vs muse4 ~0.36 macro-F1. Same subjects/labels;
only montage changes (C=8 Crown stream order). Never mix muse4/crown8 in one
example. Sleep-EDF is NOT used.

Writes:
  datasets/attention_ds003969_crown8/{windows,splits,annotations,README.md}
  datasets/attention_ds001787_crown8/{windows,splits,annotations,README.md}
  exports/crown8_reparse_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.channel_map import CROWN8_CHANNELS  # noqa: E402
from src.head_a import ATTENTION_LABELS  # noqa: E402

TARGET_SR = 256.0
WINDOW_SEC = 2.0
MONTAGE_ID = "crown8"
STEP = "reparse_crown8_attention"

RAW_DS003969 = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data/ds003969/raw"
RAW_DS001787 = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data/ds001787/raw"
MUSE4_DS003969 = ROOT / "datasets/attention_ds003969"
MUSE4_DS001787 = ROOT / "datasets/attention_ds001787"
OUT_DS003969 = ROOT / "datasets/attention_ds003969_crown8"
OUT_DS001787 = ROOT / "datasets/attention_ds001787_crown8"
EXPORT_SUMMARY = ROOT / "exports/crown8_reparse_summary.json"

DS003969_TASKS = ("med1breath", "think1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def biosemi_ab_to_1020(raw):
    import mne

    montage = mne.channels.make_standard_montage("biosemi64")
    ab = [f"A{i}" for i in range(1, 33)] + [f"B{i}" for i in range(1, 33)]
    if all(c in raw.ch_names for c in ab[:8]):
        mapping = {old: new for old, new in zip(ab, montage.ch_names) if old in raw.ch_names}
        raw.rename_channels(mapping)
        raw.set_montage(montage, on_missing="ignore")
    return raw


def load_crown8(
    bdf: Path,
    *,
    need_biosemi_rename: bool,
) -> Tuple[np.ndarray, float, str]:
    """Return (8, T) @ TARGET_SR in CROWN8 stream order, sfreq, strategy."""
    import mne

    raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
    strategy = "native_10-10_names"
    if need_biosemi_rename:
        raw = biosemi_ab_to_1020(raw)
        strategy = "biosemi64_A/B->10-10 then pick Crown8"
    names_upper = {c.upper(): c for c in raw.ch_names}
    missing = [c for c in CROWN8_CHANNELS if c.upper() not in names_upper]
    if missing:
        raise KeyError(f"missing Crown channels {missing} in {bdf.name}; have {raw.ch_names[:24]}…")
    picks = [names_upper[c.upper()] for c in CROWN8_CHANNELS]
    raw.pick(picks)
    raw.reorder_channels(picks)
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - TARGET_SR) > 1e-3:
        raw.resample(TARGET_SR, npad="auto")
        sfreq = TARGET_SR
    # Re-resolve names after possible rename/pick
    names_upper = {c.upper(): c for c in raw.ch_names}
    ordered = [names_upper[c.upper()] for c in CROWN8_CHANNELS]
    data = raw.get_data(picks=ordered).astype(np.float32)
    return data, sfreq, strategy


def slice_windows(
    data: np.ndarray,
    starts: np.ndarray,
    win: int,
) -> np.ndarray:
    n_times = data.shape[1]
    Xs = []
    for s in starts:
        s = int(s)
        if s < 0 or s + win > n_times:
            raise IndexError(f"start {s} win {win} oob for T={n_times}")
        Xs.append(data[:, s : s + win])
    return np.stack(Xs, axis=0).astype(np.float32)


def copy_splits(src_corpus: Path, dst_corpus: Path, new_corpus_name: str) -> None:
    src_splits = src_corpus / "splits"
    dst_splits = dst_corpus / "splits"
    dst_splits.mkdir(parents=True, exist_ok=True)
    for name in ("subjects.json", "train_subjects.json", "val_subjects.json", "test_subjects.json", "split_policy.json"):
        src = src_splits / name
        if not src.exists():
            continue
        doc = json.loads(src.read_text())
        if "corpus" in doc:
            doc["corpus"] = new_corpus_name
        doc["montage_id"] = MONTAGE_ID
        doc["parallel_to_muse4"] = str(src_corpus.relative_to(ROOT))
        doc["updated_utc"] = datetime.now(timezone.utc).isoformat()
        doc["updated_by"] = STEP
        (dst_splits / name).write_text(json.dumps(doc, indent=2) + "\n")


def write_corpus_readme(path: Path, corpus: str, source_ds: str, n_rec: int) -> None:
    path.write_text(
        f"""# {corpus}

Crown8 re-parse of OpenNeuro `{source_ds}` attention windows.

- **montage_id:** `{MONTAGE_ID}`
- **channels (stream order):** {", ".join(CROWN8_CHANNELS)}
- **sample_rate_hz:** {TARGET_SR}
- **labels:** same as muse4 pack (`concentration` / `mind_wandering`)
- **window alignment:** identical `starts` / `y` as `datasets/attention_{source_ds}/windows/`
- **license:** CC0-1.0 (OpenNeuro)
- **recordings:** {n_rec}

Never mix with muse4 examples. Sleep-EDF is not used for Crown.
"""
    )


def reparse_ds003969(limit: int = 0, force: bool = False) -> List[Dict[str, Any]]:
    win_src = MUSE4_DS003969 / "windows"
    out_win = OUT_DS003969 / "windows"
    out_win.mkdir(parents=True, exist_ok=True)
    (OUT_DS003969 / "annotations").mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    npzs = sorted(win_src.glob("*_windows.npz"))
    if limit > 0:
        npzs = npzs[:limit]
    for npz_path in npzs:
        rid = npz_path.name.replace("_windows.npz", "")  # sub001
        num = rid.replace("sub", "")
        sid = f"sub-{num.zfill(3)}"
        out_npz = out_win / f"{rid}_windows.npz"
        out_man = out_win / f"{rid}_manifest.json"
        if out_npz.exists() and out_man.exists() and not force:
            print(f"[skip] {rid} crown8 exists", flush=True)
            man = json.loads(out_man.read_text())
            results.append({"recording_id": rid, "subject": sid, "skipped": True, **{k: man.get(k) for k in ("n_windows_total", "npz_sha256", "channel_strategy")}})
            continue
        muse = np.load(npz_path, allow_pickle=True)
        starts = muse["starts"].astype(np.int64)
        y = muse["y"].astype(np.int64)
        task_ids = muse["task_ids"].astype(np.int64)
        label_names = [str(x) for x in muse["label_names"].tolist()]
        task_names = [str(x) for x in muse["task_names"].tolist()] if "task_names" in muse.files else list(DS003969_TASKS)
        win = int(round(WINDOW_SEC * TARGET_SR))
        eeg_dir = RAW_DS003969 / sid / "eeg"
        per_task_data: Dict[int, np.ndarray] = {}
        strategies = []
        bdf_meta = []
        for ti, task in enumerate(task_names):
            bdf = eeg_dir / f"{sid}_task-{task}_eeg.bdf"
            if not bdf.exists():
                # try glob
                cands = list(eeg_dir.glob(f"*_task-{task}_eeg.bdf"))
                if not cands:
                    raise FileNotFoundError(bdf)
                bdf = cands[0]
            data, sfreq, strategy = load_crown8(bdf, need_biosemi_rename=False)
            # also try biosemi rename if native pick somehow failed — already succeeded
            per_task_data[ti] = data
            strategies.append(strategy)
            bdf_meta.append({"task": task, "bdf": str(bdf), "bdf_sha256": sha256(bdf), "sfreq": sfreq, "T": int(data.shape[1])})
            print(f"[{rid}/{task}] crown8 loaded T={data.shape[1]} sfreq={sfreq}", flush=True)
        # slice per window using task_ids
        Xs = []
        for i in range(len(starts)):
            ti = int(task_ids[i])
            s = int(starts[i])
            d = per_task_data[ti]
            if s + win > d.shape[1]:
                raise IndexError(f"{rid} i={i} task={ti} start={s} win={win} T={d.shape[1]}")
            Xs.append(d[:, s : s + win])
        X = np.stack(Xs, axis=0).astype(np.float32)
        assert X.shape == (len(starts), 8, win), X.shape
        assert X.shape[0] == y.shape[0]
        np.savez_compressed(
            out_npz,
            X=X,
            y=y,
            starts=starts,
            task_ids=task_ids,
            label_names=np.asarray(label_names),
            channels=np.asarray(CROWN8_CHANNELS),
            task_names=np.asarray(task_names),
            montage_id=np.asarray(MONTAGE_ID),
        )
        src_man = win_src / f"{rid}_manifest.json"
        base_man = json.loads(src_man.read_text()) if src_man.exists() else {}
        counts = {ATTENTION_LABELS[i]: int((y == i).sum()) for i in range(len(ATTENTION_LABELS))}
        manifest = {
            **{k: base_man[k] for k in ("dataset", "doi", "license_spdx", "subject", "group", "first_session", "tag", "tasks", "label_rule") if k in base_man},
            "recording_id": rid,
            "montage_id": MONTAGE_ID,
            "channels": list(CROWN8_CHANNELS),
            "channel_strategy": strategies[0] if strategies else "crown8",
            "channel_note": "Native Crown-named 10-10 electrodes from ds003969 channels.tsv / BDF",
            "n_windows_per_label": counts,
            "n_windows_total": int(X.shape[0]),
            "window_shape": list(X.shape),
            "window_sec": WINDOW_SEC,
            "target_sr": TARGET_SR,
            "units": "V",
            "aligned_to_muse4_starts": True,
            "muse4_windows_ref": str(npz_path.relative_to(ROOT)),
            "per_task_bdf": bdf_meta,
            "npz_path": str(out_npz.relative_to(ROOT)),
            "npz_sha256": sha256(out_npz),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "step": STEP,
            "never_mix_with": "muse4",
        }
        out_man.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[{rid}] wrote {X.shape} sha={manifest['npz_sha256'][:12]}…", flush=True)
        results.append({
            "recording_id": rid,
            "subject": sid,
            "n_windows_total": int(X.shape[0]),
            "counts": counts,
            "npz_sha256": manifest["npz_sha256"],
            "channel_strategy": manifest["channel_strategy"],
            "skipped": False,
        })
        del X, per_task_data, muse
    copy_splits(MUSE4_DS003969, OUT_DS003969, "attention_ds003969_crown8")
    write_corpus_readme(OUT_DS003969 / "README.md", "attention_ds003969_crown8", "ds003969", len(results))
    # attribution
    src_attr = MUSE4_DS003969 / "ATTRIBUTION.md"
    if src_attr.exists():
        shutil.copy2(src_attr, OUT_DS003969 / "ATTRIBUTION.md")
    return results


def reparse_ds001787(limit: int = 0, force: bool = False) -> List[Dict[str, Any]]:
    win_src = MUSE4_DS001787 / "windows"
    out_win = OUT_DS001787 / "windows"
    out_win.mkdir(parents=True, exist_ok=True)
    (OUT_DS001787 / "annotations").mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    npzs = sorted(win_src.glob("*_windows.npz"))
    if limit > 0:
        npzs = npzs[:limit]
    for npz_path in npzs:
        rid = npz_path.name.replace("_windows.npz", "")  # sub001_ses01
        parts = rid.split("_")
        num = parts[0].replace("sub", "")
        ses = parts[1].replace("ses", "") if len(parts) > 1 else "01"
        sid = f"sub-{num.zfill(3)}"
        out_npz = out_win / f"{rid}_windows.npz"
        out_man = out_win / f"{rid}_manifest.json"
        if out_npz.exists() and out_man.exists() and not force:
            print(f"[skip] {rid} crown8 exists", flush=True)
            man = json.loads(out_man.read_text())
            results.append({"recording_id": rid, "subject": sid, "skipped": True, **{k: man.get(k) for k in ("n_windows_total", "npz_sha256", "channel_strategy")}})
            continue
        muse = np.load(npz_path, allow_pickle=True)
        starts = muse["starts"].astype(np.int64)
        y = muse["y"].astype(np.int64)
        label_names = [str(x) for x in muse["label_names"].tolist()]
        probe_ids = muse["probe_ids"].astype(np.int64) if "probe_ids" in muse.files else None
        win = int(round(WINDOW_SEC * TARGET_SR))
        eeg_dir = RAW_DS001787 / sid / f"ses-{ses}" / "eeg"
        bdf = eeg_dir / f"{sid}_ses-{ses}_task-meditation_eeg.bdf"
        if not bdf.exists():
            cands = list(eeg_dir.glob("*_task-meditation_eeg.bdf"))
            if not cands:
                raise FileNotFoundError(bdf)
            bdf = cands[0]
        data, sfreq, strategy = load_crown8(bdf, need_biosemi_rename=True)
        print(f"[{rid}] crown8 loaded T={data.shape[1]} sfreq={sfreq} strategy={strategy}", flush=True)
        X = slice_windows(data, starts, win)
        assert X.shape == (len(starts), 8, win), X.shape
        payload = {
            "X": X,
            "y": y,
            "starts": starts,
            "label_names": np.asarray(label_names),
            "channels": np.asarray(CROWN8_CHANNELS),
            "montage_id": np.asarray(MONTAGE_ID),
        }
        if probe_ids is not None:
            payload["probe_ids"] = probe_ids
        np.savez_compressed(out_npz, **payload)
        src_man = win_src / f"{rid}_manifest.json"
        base_man = json.loads(src_man.read_text()) if src_man.exists() else {}
        counts = {ATTENTION_LABELS[i]: int((y == i).sum()) for i in range(len(ATTENTION_LABELS))}
        manifest = {
            **{k: base_man[k] for k in ("dataset", "doi", "license_spdx", "subject", "session", "group", "tag", "label_rule", "merge_mode") if k in base_man},
            "recording_id": rid,
            "montage_id": MONTAGE_ID,
            "channels": list(CROWN8_CHANNELS),
            "channel_strategy": strategy,
            "channel_note": "BioSemi64 A1..B32 → MNE biosemi64 10-10; pick Crown stream order",
            "n_windows_per_label": counts,
            "n_windows_total": int(X.shape[0]),
            "window_shape": list(X.shape),
            "window_sec": WINDOW_SEC,
            "target_sr": TARGET_SR,
            "units": "V",
            "aligned_to_muse4_starts": True,
            "muse4_windows_ref": str(npz_path.relative_to(ROOT)),
            "bdf": str(bdf),
            "bdf_sha256": sha256(bdf),
            "npz_path": str(out_npz.relative_to(ROOT)),
            "npz_sha256": sha256(out_npz),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "step": STEP,
            "never_mix_with": "muse4",
        }
        out_man.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[{rid}] wrote {X.shape} sha={manifest['npz_sha256'][:12]}…", flush=True)
        results.append({
            "recording_id": rid,
            "subject": sid,
            "n_windows_total": int(X.shape[0]),
            "counts": counts,
            "npz_sha256": manifest["npz_sha256"],
            "channel_strategy": strategy,
            "skipped": False,
        })
        del X, data, muse
    copy_splits(MUSE4_DS001787, OUT_DS001787, "attention_ds001787_crown8")
    write_corpus_readme(OUT_DS001787 / "README.md", "attention_ds001787_crown8", "ds001787", len(results))
    src_attr = MUSE4_DS001787 / "ATTRIBUTION.md"
    if src_attr.exists():
        shutil.copy2(src_attr, OUT_DS001787 / "ATTRIBUTION.md")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="max recordings per corpus (0=all)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", choices=("ds003969", "ds001787", "both"), default="both")
    args = ap.parse_args()

    summary: Dict[str, Any] = {
        "step": STEP,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "montage_id": MONTAGE_ID,
        "channels": list(CROWN8_CHANNELS),
        "target_sr": TARGET_SR,
        "policy": "same starts/labels as muse4; never mix montages; no Sleep-EDF",
        "corpora": {},
    }
    if args.only in ("ds003969", "both"):
        r969 = reparse_ds003969(limit=args.limit, force=args.force)
        summary["corpora"]["attention_ds003969_crown8"] = {
            "n_recordings": len(r969),
            "n_windows_total": int(sum(x.get("n_windows_total") or 0 for x in r969)),
            "recordings": r969,
        }
    if args.only in ("ds001787", "both"):
        r1787 = reparse_ds001787(limit=args.limit, force=args.force)
        summary["corpora"]["attention_ds001787_crown8"] = {
            "n_recordings": len(r1787),
            "n_windows_total": int(sum(x.get("n_windows_total") or 0 for x in r1787)),
            "recordings": r1787,
        }
    EXPORT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    rollup = {
        ck: {kk: vv for kk, vv in cv.items() if kk != "recordings"}
        for ck, cv in summary["corpora"].items()
    }
    print(json.dumps({"step": STEP, "corpora_rollup": rollup, "summary": str(EXPORT_SUMMARY)}, indent=2), flush=True)
    print(f"wrote {EXPORT_SUMMARY}", flush=True)


if __name__ == "__main__":
    main()

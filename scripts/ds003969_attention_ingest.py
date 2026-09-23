#!/usr/bin/env python3
"""Invented continuum step ds003969_attention_ingest: Muse-proxy AF7/AF8+TP7/TP8 block windows."""
from __future__ import annotations

import hashlib
import json
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

from src.head_a import ATTENTION_LABELS, labels_to_ids  # noqa: E402

SEED = 42
WINDOW_SEC = 2.0
HOP_SEC = 1.0  # longer hop for ~10 min blocks (less overlap than probe epochs)
TARGET_SR = 256.0
EDGE_TRIM_SEC = 30.0  # drop block edges
MAX_WINDOWS_PER_BLOCK = 400  # cap correlated windows per task block
MUSE_SRC = ["AF7", "AF8", "TP7", "TP8"]
MUSE_OUT = ["AF7", "AF8", "TP9", "TP10"]  # TP7/TP8 → TP9/TP10 proxies
TASKS = ("med1breath", "think1")  # one med + one think block per subject
TASK_LABEL = {
    "med1breath": "concentration",
    "med2": "concentration",
    "think1": "mind_wandering",
    "think2": "mind_wandering",
}

# 2 Himalayan-tradition + 1 control; first_session diversifies order
SUBJECTS: List[Dict[str, str]] = [
    {"sub": "001", "group": "htr", "first_session": "meditation", "tree": "ea3163ed1e9c60c02e92694f2c55d6114b032818"},
    {"sub": "002", "group": "htr", "first_session": "thinking", "tree": "43d5213699e18147cdc023fbb9cc159a4dffc1a7"},
    {"sub": "025", "group": "ctr", "first_session": "thinking", "tree": "2595bce81d49a60f12a91c2ab25d108f1303a68f"},
]

CACHE = ROOT / "kaggle_datasets/muse-eeg-heads-cache/data/ds003969"
RAW = CACHE / "raw"
WINDOWS_PKG = ROOT / "kaggle_datasets/muse-eeg-heads-windows"
EXPORT = ROOT / "exports/windows_ds003969"
PAUSE_SEC = 3.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pause(msg: str) -> None:
    print(f"[pace] {msg} (sleep {PAUSE_SEC}s)", flush=True)
    time.sleep(PAUSE_SEC)


def gql(query: str) -> Dict[str, Any]:
    req = urllib.request.Request(
        "https://openneuro.org/crn/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def list_tree(tree_id: str) -> List[Dict[str, Any]]:
    q = (
        'query { dataset(id: "ds003969") { latestSnapshot { '
        f'files(tree: "{tree_id}") {{ id filename size directory urls }}'
        " } } }"
    )
    return gql(q)["data"]["dataset"]["latestSnapshot"]["files"]


def download(url: str, dest: Path, expected_size: Optional[int] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"[skip] {dest.name} already present ({dest.stat().st_size} B)", flush=True)
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get] {dest.name} ← {url[:100]}…", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    if expected_size is not None and tmp.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch {dest.name}: got {tmp.stat().st_size} want {expected_size}")
    tmp.replace(dest)


def resolve_task_files(spec: Dict[str, str]) -> List[Dict[str, Any]]:
    pause(f"graphql tree {spec['sub']}")
    top = list_tree(spec["tree"])
    eeg_dir = next(f for f in top if f["directory"] and f["filename"] == "eeg")
    pause(f"graphql eeg {spec['sub']}")
    files = list_tree(eeg_dir["id"])
    by_task: Dict[str, Dict[str, Any]] = {}
    for f in files:
        if f.get("directory"):
            continue
        name = f["filename"]
        task = None
        for t in ("med1breath", "med2", "think1", "think2"):
            if f"_task-{t}_" in name:
                task = t
                break
        if task not in TASKS:
            continue
        slot = by_task.setdefault(task, {"task": task, "label": TASK_LABEL[task]})
        if name.endswith("_eeg.bdf"):
            slot["bdf"] = f
        elif name.endswith("_eeg.json"):
            slot["json"] = f
        elif name.endswith("_channels.tsv"):
            slot["channels"] = f
    missing = [t for t in TASKS if t not in by_task or "bdf" not in by_task[t]]
    if missing:
        raise FileNotFoundError(f"sub-{spec['sub']} missing tasks {missing}")
    return [by_task[t] for t in TASKS]


def fetch_subject(spec: Dict[str, str]) -> List[Dict[str, Any]]:
    sub = spec["sub"]
    eeg_dir = RAW / f"sub-{sub}" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    tasks = resolve_task_files(spec)
    out = []
    for t in tasks:
        bdf_meta = t["bdf"]
        bdf_path = eeg_dir / bdf_meta["filename"]
        urls = bdf_meta.get("urls") or []
        if not urls:
            raise RuntimeError(f"no url for {bdf_meta['filename']}")
        pause(f"before bdf {bdf_meta['filename']}")
        download(urls[0], bdf_path, expected_size=int(bdf_meta["size"]))
        ch_path = None
        if t.get("channels"):
            ch_meta = t["channels"]
            ch_path = eeg_dir / ch_meta["filename"]
            pause(f"before channels {ch_meta['filename']}")
            download((ch_meta.get("urls") or [None])[0], ch_path, expected_size=int(ch_meta["size"]))
        js_path = None
        if t.get("json"):
            js_meta = t["json"]
            js_path = eeg_dir / js_meta["filename"]
            pause(f"before json {js_meta['filename']}")
            download((js_meta.get("urls") or [None])[0], js_path, expected_size=int(js_meta["size"]))
        out.append(
            {
                "task": t["task"],
                "label": t["label"],
                "bdf": bdf_path,
                "channels": ch_path,
                "json": js_path,
                "bdf_url": urls[0],
                "bdf_bytes": int(bdf_meta["size"]),
            }
        )
    return out


def load_muse_proxy(bdf: Path) -> Tuple[np.ndarray, float, List[str], str]:
    import mne

    raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
    # Prefer 10-20 names already present; else try biosemi A/B rename
    names_upper = {c.upper(): c for c in raw.ch_names}
    if not all(c in names_upper for c in MUSE_SRC):
        montage = mne.channels.make_standard_montage("biosemi64")
        ab = [f"A{i}" for i in range(1, 33)] + [f"B{i}" for i in range(1, 33)]
        if all(c in raw.ch_names for c in ab[:8]):
            mapping = {old: new for old, new in zip(ab, montage.ch_names) if old in raw.ch_names}
            raw.rename_channels(mapping)
            names_upper = {c.upper(): c for c in raw.ch_names}
    missing = [c for c in MUSE_SRC if c not in names_upper]
    if missing:
        raise KeyError(f"missing Muse-proxy channels {missing} in {bdf.name}; have {raw.ch_names[:20]}…")
    picks = [names_upper[c] for c in MUSE_SRC]
    raw.pick(picks)
    sfreq = float(raw.info["sfreq"])
    if abs(sfreq - TARGET_SR) > 1e-3:
        raw.resample(TARGET_SR, npad="auto")
        sfreq = TARGET_SR
    data = raw.get_data().astype(np.float32)
    strategy = "af7_af8_tp7_tp8->AF7/AF8/TP9/TP10"
    return data, sfreq, list(MUSE_OUT), strategy


def extract_block_windows(
    data: np.ndarray,
    sfreq: float,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    rng = np.random.default_rng(SEED)
    win = int(round(WINDOW_SEC * sfreq))
    hop = int(round(HOP_SEC * sfreq))
    trim = int(round(EDGE_TRIM_SEC * sfreq))
    n_times = data.shape[1]
    s0 = trim
    s1 = max(trim, n_times - trim)
    drops = Counter()
    if s1 - s0 < win:
        drops["too_short"] += 1
        empty = np.zeros((0, 4, win), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), dict(drops)
    segment = data[:, s0:s1]
    starts_local: List[int] = []
    pos = 0
    while pos + win <= segment.shape[1]:
        starts_local.append(pos)
        pos += hop
    if not starts_local:
        drops["no_window"] += 1
        empty = np.zeros((0, 4, win), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), dict(drops)
    if len(starts_local) > MAX_WINDOWS_PER_BLOCK:
        drops["capped"] = len(starts_local) - MAX_WINDOWS_PER_BLOCK
        choose = rng.choice(len(starts_local), size=MAX_WINDOWS_PER_BLOCK, replace=False)
        choose.sort()
        starts_local = [starts_local[i] for i in choose]
    Xs = [segment[:, p : p + win] for p in starts_local]
    X = np.stack(Xs, axis=0).astype(np.float32)
    y = labels_to_ids([label] * len(Xs), ATTENTION_LABELS)
    starts = np.asarray([s0 + p for p in starts_local], dtype=np.int64)
    return X, y, starts, dict(drops)


def process_subject(spec: Dict[str, str], recordings: List[Dict[str, Any]]) -> Dict[str, Any]:
    sub = spec["sub"]
    tag = f"sub{sub}"
    Xs, ys, starts_all, task_ids = [], [], [], []
    per_task = []
    for ti, rec in enumerate(recordings):
        data, sfreq, ch_names, strategy = load_muse_proxy(rec["bdf"])
        X, y, starts, drops = extract_block_windows(data, sfreq, rec["label"])
        per_task.append(
            {
                "task": rec["task"],
                "label": rec["label"],
                "n_windows": int(X.shape[0]),
                "drops": drops,
                "bdf_sha256": sha256(rec["bdf"]),
                "bdf_bytes": rec["bdf_bytes"],
                "sfreq_export": sfreq,
                "channel_strategy": strategy,
            }
        )
        if X.shape[0] == 0:
            continue
        Xs.append(X)
        ys.append(y)
        starts_all.append(starts)
        task_ids.append(np.full((X.shape[0],), ti, dtype=np.int64))
        print(f"[{tag}/{rec['task']}] windows={X.shape} label={rec['label']} drops={drops}", flush=True)
    if not Xs:
        raise RuntimeError(f"no windows for {tag}")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    starts = np.concatenate(starts_all, axis=0)
    task_id_arr = np.concatenate(task_ids, axis=0)
    counts = {ATTENTION_LABELS[i]: int((y == i).sum()) for i in range(len(ATTENTION_LABELS))}
    out_dir = EXPORT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"ds003969_{tag}_attention_windows.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        y=y,
        starts=starts,
        task_ids=task_id_arr,
        label_names=np.asarray(ATTENTION_LABELS),
        channels=np.asarray(ch_names),
        task_names=np.asarray([r["task"] for r in recordings]),
    )
    manifest = {
        "dataset": "ds003969",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "subject": f"sub-{sub}",
        "group": spec["group"],
        "first_session": spec["first_session"],
        "tag": tag,
        "tasks": [r["task"] for r in recordings],
        "label_rule": "block-level: med*→concentration; think*→mind_wandering (protocol proxy, weaker than probe ratings)",
        "n_windows_per_label": counts,
        "n_windows_total": int(X.shape[0]),
        "window_shape": list(X.shape),
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "edge_trim_sec": EDGE_TRIM_SEC,
        "max_windows_per_block": MAX_WINDOWS_PER_BLOCK,
        "target_sr": TARGET_SR,
        "channels": ch_names,
        "channel_strategy": strategy,
        "channel_note": "Native AF7/AF8; TP9/TP10←TP7/TP8 temporal proxies (no TP9/TP10 in montage)",
        "source_sfreq_hz": 1024,
        "per_task": per_task,
        "npz_path": str(npz_path),
        "npz_sha256": sha256(npz_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "step": "ds003969_attention_ingest",
    }
    man_path = out_dir / f"ds003969_{tag}_attention_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for src in (npz_path, man_path):
        shutil.copy2(src, WINDOWS_PKG / src.name)
    return {
        "tag": tag,
        "subject": f"sub-{sub}",
        "group": spec["group"],
        "first_session": spec["first_session"],
        "counts": counts,
        "n_windows_total": int(X.shape[0]),
        "channel_strategy": strategy,
        "channels": ch_names,
        "npz_sha256": manifest["npz_sha256"],
        "per_task": per_task,
        "npz": str(npz_path),
        "manifest": str(man_path),
    }


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    EXPORT.mkdir(parents=True, exist_ok=True)
    WINDOWS_PKG.mkdir(parents=True, exist_ok=True)
    # participants.tsv for provenance
    pause("participants.tsv")
    download(
        "https://s3.amazonaws.com/openneuro.org/ds003969/participants.tsv",
        CACHE / "participants.tsv",
    )
    pause("dataset_description.json")
    download(
        "https://s3.amazonaws.com/openneuro.org/ds003969/dataset_description.json",
        CACHE / "dataset_description.json",
    )

    results: Dict[str, Any] = {
        "step": "ds003969_attention_ingest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": [],
        "why": "Muse-native AF7/AF8 (+TP7/TP8→TP9/TP10) attention blocks after ds001787 subject-holdout collapse",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "tasks": list(TASKS),
        "label_rule": "med*→concentration; think*→mind_wandering",
    }
    totals = Counter()
    provenance = {
        "step": "ds003969_attention_ingest",
        "source": "https://openneuro.org/datasets/ds003969",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "added_utc": datetime.now(timezone.utc).isoformat(),
        "recordings": [],
    }

    for spec in SUBJECTS:
        recs = fetch_subject(spec)
        info = process_subject(spec, recs)
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
                    "sha256": sha256(rec["bdf"]),
                    "bytes": rec["bdf_bytes"],
                    "url": rec["bdf_url"],
                }
            )

    results["totals"] = dict(totals)
    results["n_windows_total"] = int(sum(totals.values()))
    prov_path = CACHE / "provenance_attention_ingest.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    results["provenance_path"] = str(prov_path)

    rollup_path = WINDOWS_PKG / "manifest.json"
    rollup: Dict[str, Any] = {}
    if rollup_path.exists():
        try:
            rollup = json.loads(rollup_path.read_text())
        except json.JSONDecodeError:
            rollup = {}
    rollup["ds003969_attention"] = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "subjects": [s["tag"] for s in results["subjects"]],
        "totals": results["totals"],
        "n_windows_total": results["n_windows_total"],
        "recipe": {
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "edge_trim_sec": EDGE_TRIM_SEC,
            "max_windows_per_block": MAX_WINDOWS_PER_BLOCK,
            "target_sr": TARGET_SR,
            "tasks": list(TASKS),
            "label_rule": "med* concentration; think* mind_wandering",
            "channel_proxy": "AF7/AF8/TP7/TP8→AF7/AF8/TP9/TP10",
        },
        "files": [],
    }
    for s in results["subjects"]:
        tag = s["tag"]
        rollup["ds003969_attention"]["files"].extend(
            [
                f"ds003969_{tag}_attention_windows.npz",
                f"ds003969_{tag}_attention_manifest.json",
            ]
        )
        rollup["ds003969_attention"][tag] = {
            "n_windows_per_label": s["counts"],
            "n_windows_total": s["n_windows_total"],
            "npz_sha256": s["npz_sha256"],
            "group": s["group"],
            "channel_strategy": s["channel_strategy"],
        }
    rollup["updated_utc"] = datetime.now(timezone.utc).isoformat()
    rollup_path.write_text(json.dumps(rollup, indent=2) + "\n")

    summary_path = ROOT / "exports/ds003969_attention_ingest_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: results[k] for k in ("totals", "n_windows_total", "why")}, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

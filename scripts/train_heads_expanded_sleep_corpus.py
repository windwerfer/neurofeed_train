#!/usr/bin/env python3
"""Train Head C + Head A-vig on expanded sleep corpus (frozen CBraMod).

Corpus: datasets/vigilance_sleep_edf (Sleep-EDF + HMC N1-slice, ~124 subjects).
Strategy: embed-once per recording → cache → train tiny linear heads.
NO backbone fine-tune.

N1-slice note: stage_coarse is effectively wake/light only (deep/REM absent).
Head C therefore trains as 2-way wake/light; product 4-way staging still needs
sleep-period windows (not this pass).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder
from src.head_c import HEAD_C_COARSE_LABELS, HeadCLinear, class_weights_from_y, undersample_balanced
from src.heads.head_a_vig import HEAD_A_VIG_LABELS_2, HeadAVigLinear
from src.metrics import confusion_matrix, per_class_report

SEED = 42
TARGET_SR = 256.0
EPOCHS = 20
BATCH_ENC = 32
BATCH_HEAD = 256
LR = 1e-3
PATIENCE = 5
FLAT_STD = 0.1
PEAK_ABS = 350.0
# Soft cap per class per recording before encode (0 = no cap / full corpus)
DEFAULT_PER_CLASS_CAP = 0
# After pooling train: undersample to balance (True) or class-weighted CE on all
TRAIN_BALANCE = True
# Max train windows after balance (0 = unlimited minority*n_classes)
MAX_TRAIN_BALANCED = 80_000
# Eval: optional per-class cap on val/test for speed docs (0 = full)
EVAL_PER_CLASS_CAP = 0

CORPUS = "vigilance_sleep_edf"
WIN_DIR = ROOT / "datasets" / CORPUS / "windows"
SPLITS_DIR = ROOT / "datasets" / CORPUS / "splits"

OUT_C = ROOT / "exports" / "head_c_full_corpus"
OUT_A = ROOT / "exports" / "head_a_vig_full_corpus"
EMB_CACHE = OUT_C / "emb_cache"
DOCS_C = ROOT / "docs" / "head_c_full_corpus_train.md"
DOCS_A = ROOT / "docs" / "head_a_vig_full_corpus_train.md"

# Head C on this corpus: only wake/light present
HEAD_C_LABELS_ACTIVE: List[str] = ["wake", "light"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_weights() -> Path:
    cands = [
        ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth",
        Path("/tmp/kaggle_out3/models/CBraMod/pretrained_weights.pth"),
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("CBraMod weights not found")


def load_policy() -> Dict[str, Any]:
    return json.loads((SPLITS_DIR / "split_policy.json").read_text())


def load_split_map() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for sp in ("train", "val", "test"):
        for sid in json.loads((SPLITS_DIR / f"{sp}_subjects.json").read_text())["subjects"]:
            out[sid] = sp
    return out


def qc_pass_mask(X: np.ndarray) -> np.ndarray:
    std = X.std(axis=-1)
    peak = np.max(np.abs(X), axis=-1)
    flat = (std < FLAT_STD).any(axis=-1)
    hot = (peak > PEAK_ABS).any(axis=-1)
    return ~(flat | hot)


def cap_idxs_per_class(
    y: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if per_class <= 0:
        return np.arange(len(y))
    picks: List[np.ndarray] = []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        n = min(int(per_class), len(cand))
        if n == 0:
            continue
        picks.append(rng.choice(cand, size=n, replace=False))
    if not picks:
        return np.zeros(0, dtype=np.int64)
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


def recording_subject_id(rid: str, policy: Dict[str, Any]) -> str:
    recs = policy.get("recordings", {})
    if rid in recs:
        return str(recs[rid]["subject_id"])
    # fallbacks
    if rid.startswith("SN"):
        return rid[:5] if len(rid) >= 5 else rid
    if rid.startswith(("SC", "ST")):
        return rid[:5]
    return rid


def list_recordings(policy: Dict[str, Any]) -> List[str]:
    rids = sorted(policy.get("recordings", {}).keys())
    if rids:
        return rids
    return sorted(p.name.replace("_windows.npz", "") for p in WIN_DIR.glob("*_windows.npz"))


def encode_batch(
    encoder: FrozenCBraModEncoder,
    X: np.ndarray,
    device: torch.device,
    batch: int = BATCH_ENC,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            chunks.append(encoder(xb).cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, 200), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def map_stage_coarse(coarse: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_c ids in HEAD_C_LABELS_ACTIVE, keep mask). Drop unknown/deep/rem."""
    table = {n: i for i, n in enumerate(HEAD_C_LABELS_ACTIVE)}
    keep = []
    ys = []
    for c in coarse:
        s = str(c)
        if s in table:
            keep.append(True)
            ys.append(table[s])
        else:
            keep.append(False)
            ys.append(-1)
    return np.asarray(ys, dtype=np.int64), np.asarray(keep, dtype=bool)


def ensure_emb_cache(
    encoder: FrozenCBraModEncoder,
    device: torch.device,
    policy: Dict[str, Any],
    split_map: Dict[str, str],
    rng: np.random.Generator,
    per_class_cap: int,
) -> Dict[str, Any]:
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "recordings": [],
        "per_class_cap": per_class_cap,
        "sampling_note": (
            "no per-recording cap; full N1-slice windows after light QC"
            if per_class_cap <= 0
            else f"capped to ≤{per_class_cap} windows/class/recording after QC (documented subsample)"
        ),
        "qc": {"flat_std": FLAT_STD, "peak_abs_uv": PEAK_ABS},
    }
    rids = list_recordings(policy)
    for i, rid in enumerate(rids):
        cache_path = EMB_CACHE / f"{rid}_emb.npz"
        npz_path = WIN_DIR / f"{rid}_windows.npz"
        if not npz_path.exists():
            print(f"  SKIP missing windows {rid}", flush=True)
            continue
        sid = recording_subject_id(rid, policy)
        split = split_map.get(sid)
        if split is None:
            print(f"  SKIP no split for subject {sid} ({rid})", flush=True)
            continue
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            meta["recordings"].append(
                {
                    "recording_id": rid,
                    "subject_id": sid,
                    "split": split,
                    "n": int(z["emb"].shape[0]),
                    "cached": True,
                    "source": str(policy["recordings"].get(rid, {}).get("source", "")),
                }
            )
            if (i + 1) % 20 == 0:
                print(f"  cache hit {i+1}/{len(rids)} last={rid} n={z['emb'].shape[0]}", flush=True)
            continue

        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        stage_coarse = np.asarray(data["stage_coarse"], dtype=object)
        label_names = [str(x) for x in data["label_names"].tolist()]
        y_a_raw = data["y"].astype(np.int64)
        # Remap Head A labels to HEAD_A_VIG_LABELS_2 order
        name_to_vig = {n: HEAD_A_VIG_LABELS_2.index(n) for n in label_names if n in HEAD_A_VIG_LABELS_2}
        y_a = np.full(len(y_a_raw), -1, dtype=np.int64)
        for old_i, name in enumerate(label_names):
            if name in name_to_vig:
                y_a[y_a_raw == old_i] = name_to_vig[name]

        y_c_all, keep_c = map_stage_coarse(stage_coarse)
        qc = qc_pass_mask(X)
        # Keep windows that have valid A-vig OR valid Head-C label and pass QC
        keep = qc & (y_a >= 0) & keep_c
        idxs = np.where(keep)[0]
        if len(idxs) == 0:
            print(f"  SKIP empty after QC {rid}", flush=True)
            continue
        # Cap using Head-C / A labels (same 2-way here)
        y_for_cap = y_a[idxs]
        if per_class_cap > 0:
            local = cap_idxs_per_class(y_for_cap, per_class_cap, rng)
            idxs = idxs[local]

        X_use = X[idxs]
        emb = encode_batch(encoder, X_use, device)
        y_a_u = y_a[idxs]
        y_c_u = y_c_all[idxs]
        assert (y_c_u >= 0).all()
        np.savez_compressed(
            cache_path,
            emb=emb.astype(np.float32),
            y_a=y_a_u.astype(np.int64),
            y_c=y_c_u.astype(np.int64),
            subject_id=np.asarray(sid),
            split=np.asarray(split),
            recording_id=np.asarray(rid),
            n_before_qc=np.asarray(len(X)),
            n_qc_pass=np.asarray(int(qc.sum())),
            n_after_cap=np.asarray(len(idxs)),
            per_class_cap=np.asarray(per_class_cap),
        )
        meta["recordings"].append(
            {
                "recording_id": rid,
                "subject_id": sid,
                "split": split,
                "n": int(len(idxs)),
                "n_before_qc": int(len(X)),
                "n_qc_pass": int(qc.sum()),
                "cached": False,
                "source": str(policy["recordings"].get(rid, {}).get("source", "")),
                "y_a_counts": {HEAD_A_VIG_LABELS_2[j]: int((y_a_u == j).sum()) for j in range(2)},
                "y_c_counts": {HEAD_C_LABELS_ACTIVE[j]: int((y_c_u == j).sum()) for j in range(2)},
            }
        )
        print(
            f"  encoded {i+1}/{len(rids)} {rid} split={split} n={len(idxs)} "
            f"a={Counter(y_a_u.tolist())}",
            flush=True,
        )
        del data, X, X_use, emb
        gc.collect()

    (EMB_CACHE / "cache_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_pooled(
    split: str,
    task: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """task: 'a' or 'c'."""
    embs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    detail: Dict[str, Any] = {"recordings": [], "counts": Counter(), "n_subjects": 0}
    subjects = set()
    key = "y_a" if task == "a" else "y_c"
    labels = HEAD_A_VIG_LABELS_2 if task == "a" else HEAD_C_LABELS_ACTIVE
    for path in sorted(EMB_CACHE.glob("*_emb.npz")):
        z = np.load(path, allow_pickle=True)
        if str(z["split"]) != split:
            continue
        emb = z["emb"].astype(np.float32)
        y = z[key].astype(np.int64)
        mask = y >= 0
        emb, y = emb[mask], y[mask]
        if len(y) == 0:
            continue
        embs.append(emb)
        ys.append(y)
        sid = str(z["subject_id"])
        subjects.add(sid)
        c = Counter({labels[j]: int((y == j).sum()) for j in range(len(labels))})
        detail["recordings"].append(
            {
                "recording_id": str(z["recording_id"]),
                "subject_id": sid,
                "n": int(len(y)),
                "counts": dict(c),
            }
        )
        detail["counts"].update(c)
    if not embs:
        raise RuntimeError(f"no embeddings for split={split} task={task}")
    detail["counts"] = dict(detail["counts"])
    detail["n_subjects"] = len(subjects)
    detail["subjects"] = sorted(subjects)
    return np.concatenate(embs), np.concatenate(ys), detail


def maybe_cap_eval(
    emb: np.ndarray,
    y: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if per_class <= 0:
        return emb, y
    idxs = cap_idxs_per_class(y, per_class, rng)
    return emb[idxs], y[idxs]


def eval_head(
    head: nn.Module,
    emb: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    device: torch.device,
) -> Dict[str, Any]:
    head.eval()
    preds: List[int] = []
    with torch.no_grad():
        for i in range(0, len(emb), BATCH_HEAD):
            logits = head(torch.from_numpy(emb[i : i + BATCH_HEAD]).to(device))
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    pred = np.asarray(preds, dtype=np.int64)
    report = per_class_report(y.tolist(), pred.tolist(), list(labels))
    cm = confusion_matrix(y.tolist(), pred.tolist(), list(labels))
    acc = float((pred == y).mean()) if len(y) else 0.0
    return {
        "n": int(len(y)),
        "accuracy": acc,
        "macro_f1": float(report["macro_f1"]["f1"]),
        "per_class": {k: v for k, v in report.items() if k != "macro_f1"},
        "confusion_matrix": cm.tolist(),
        "label_list": list(labels),
        "pred_counts": {labels[i]: int((pred == i).sum()) for i in range(len(labels))},
        "true_counts": {labels[i]: int((y == i).sum()) for i in range(len(labels))},
    }


def train_head(
    emb_fit: np.ndarray,
    y_fit: np.ndarray,
    emb_val: np.ndarray,
    y_val: np.ndarray,
    labels: Sequence[str],
    device: torch.device,
    head_ctor,
) -> Tuple[nn.Module, List[Dict[str, Any]], float]:
    head = head_ctor(in_dim=emb_fit.shape[-1], n_classes=len(labels)).to(device)
    w = class_weights_from_y(y_fit, n_classes=len(labels))
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(emb_fit), torch.from_numpy(y_fit)),
        batch_size=BATCH_HEAD,
        shuffle=True,
    )
    history: List[Dict[str, Any]] = []
    best_val = -1.0
    best_state = None
    stale = 0
    for ep in range(EPOCHS):
        head.train()
        total, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        train_m = eval_head(head, emb_fit, y_fit, labels, device)
        val_m = eval_head(head, emb_val, y_val, labels, device)
        row = {
            "epoch": ep + 1,
            "loss": total / max(n, 1),
            "train_macro_f1": train_m["macro_f1"],
            "val_macro_f1": val_m["macro_f1"],
            "val_acc": val_m["accuracy"],
        }
        history.append(row)
        print(row, flush=True)
        if val_m["macro_f1"] > best_val + 1e-4:
            best_val = val_m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop at epoch {ep+1}", flush=True)
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, history, best_val


def ship_a_vig(test_m: Dict[str, Any], val_m: Dict[str, Any]) -> Tuple[bool, str]:
    """Honest ship bar for A-vig on subject holdout."""
    reasons = []
    ok = True
    if test_m["macro_f1"] < 0.65:
        ok = False
        reasons.append(f"test macro-F1 {test_m['macro_f1']:.3f} < 0.65")
    if val_m["macro_f1"] < 0.60:
        ok = False
        reasons.append(f"val macro-F1 {val_m['macro_f1']:.3f} < 0.60")
    for name in HEAD_A_VIG_LABELS_2:
        f1 = test_m["per_class"][name]["f1"]
        if f1 < 0.50:
            ok = False
            reasons.append(f"test {name} F1 {f1:.3f} < 0.50")
        if test_m["pred_counts"].get(name, 0) == 0:
            ok = False
            reasons.append(f"never predicts {name}")
    # Domain-gap honesty: muse proxy — allow ship of *proxy* decoder, not claim Muse-native
    note = (
        "ship_candidate for muse4-proxy Sleep-EDF/HMC vig decoder only; "
        "true Muse domain adaptation still recommended before consumer ship."
    )
    if ok:
        return True, note + " Metrics clear subject-holdout bar."
    return False, "; ".join(reasons) if reasons else "failed bar"


def ship_head_c(test_m: Dict[str, Any]) -> Tuple[bool, str]:
    return (
        False,
        "Head C on N1-slice corpus is 2-way wake/light only (deep/REM absent). "
        "Not a product 4-way stage decoder; treat as transfer probe / label sanity. "
        f"test macro-F1={test_m['macro_f1']:.3f}.",
    )


def write_docs_c(metrics: Dict[str, Any], sampling: str) -> None:
    val, test = metrics["val"], metrics["test"]
    lines = [
        "# Head C train — expanded sleep corpus (frozen CBraMod)",
        "",
        f"**Step:** `head_c_full_corpus_train`  ",
        f"**Date (UTC):** {metrics['created_utc']}  ",
        f"**ship_candidate:** **{metrics['ship_candidate']}** — {metrics['ship_reason']}",
        "",
        "## Corpus",
        "",
        "- Path: `datasets/vigilance_sleep_edf/`",
        "- Sources: Sleep-EDF Expanded (ODC-By) + HMC (CC-BY-4.0)",
        "- Montage tags: `(ch_count=4, professional)` muse4 proxy (Sleep-EDF Fpz-Cz/Pz-Oz; HMC F4/C4/C3/O2)",
        "- Recipe: N1-slice (pre 20 min / post 40 min); windows 2 s @ 256 Hz",
        f"- Subjects: {metrics['n_subjects']} unique / recordings used: {metrics['n_recordings']}",
        "- Splits: policy A (SC400=test, SC403=val anchors); subject-wise no leakage",
        "",
        "## Task honesty",
        "",
        "N1-slice labeled windows only contain **wake** and **light** (W / N1).",
        "**deep / rem are absent** → Head C trained as **2-way** `wake`/`light` (subset of `stage_coarse`).",
        "Full 4-way Head C needs sleep-period windows (out of scope; no backbone fine-tune this pass).",
        "",
        "## Setup",
        "",
        "- Encoder: **frozen** CBraMod (embed-once → head train)",
        "- Head: `HeadCLinear`",
        f"- Sampling: {sampling}",
        "- Backbone fine-tune: **no**",
        "",
        "## Metrics",
        "",
        "| Split | n | accuracy | macro-F1 |",
        "|-------|--:|---------:|---------:|",
        f"| train (fit) | {metrics['train_fit']['n']} | {metrics['train_fit']['accuracy']:.3f} | {metrics['train_fit']['macro_f1']:.3f} |",
        f"| val | {val['n']} | {val['accuracy']:.3f} | {val['macro_f1']:.3f} |",
        f"| test | {test['n']} | {test['accuracy']:.3f} | {test['macro_f1']:.3f} |",
        "",
        "### Val per-class",
        "",
        "| class | precision | recall | f1 | support |",
        "|-------|----------:|-------:|---:|--------:|",
    ]
    for name in HEAD_C_LABELS_ACTIVE:
        r = val["per_class"][name]
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {int(r['support'])} |"
        )
    lines += [
        "",
        "### Test per-class",
        "",
        "| class | precision | recall | f1 | support |",
        "|-------|----------:|-------:|---:|--------:|",
    ]
    for name in HEAD_C_LABELS_ACTIVE:
        r = test["per_class"][name]
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {int(r['support'])} |"
        )
    lines += [
        "",
        "## Recommendation",
        "",
        metrics["recommendation"],
        "",
        "## Artifacts",
        "",
        f"- Emb cache: `{EMB_CACHE.relative_to(ROOT)}/`",
        f"- Weights: `{metrics['head_path']}`",
        f"- Metrics: `exports/head_c_full_corpus/metrics_summary.json`",
        f"- Script: `scripts/train_heads_expanded_sleep_corpus.py`",
        "",
    ]
    DOCS_C.write_text("\n".join(lines))


def write_docs_a(metrics: Dict[str, Any], sampling: str) -> None:
    val, test = metrics["val"], metrics["test"]
    lines = [
        "# Head A-vig train — expanded sleep corpus (frozen CBraMod)",
        "",
        f"**Step:** `head_a_vig_full_corpus_train`  ",
        f"**Date (UTC):** {metrics['created_utc']}  ",
        f"**ship_candidate:** **{metrics['ship_candidate']}** — {metrics['ship_reason']}",
        "",
        "## Corpus",
        "",
        "- Same pool as Head C: `datasets/vigilance_sleep_edf/` (not SC400-only)",
        "- Sleep-EDF (ODC-By) + HMC (CC-BY-4.0 ship-OK)",
        "- Tags: `(ch_count=4, professional)` muse4 proxy",
        f"- Subjects: {metrics['n_subjects']} / recordings: {metrics['n_recordings']}",
        "- Labels: W→`drowsy`, N1→`hypnagogic`",
        "- Splits: policy A subject-wise (SC400 test anchor, SC403 val anchor)",
        "",
        "## Setup",
        "",
        "- Encoder: **frozen** CBraMod (shared emb cache with Head C)",
        "- Head: `HeadAVigLinear` (2-way)",
        f"- Sampling: {sampling}",
        "- Backbone fine-tune: **no** (user decides after metrics)",
        "- Skipped: A-med / A-eng ship training",
        "",
        "## Metrics",
        "",
        "| Split | n | accuracy | macro-F1 |",
        "|-------|--:|---------:|---------:|",
        f"| train (fit) | {metrics['train_fit']['n']} | {metrics['train_fit']['accuracy']:.3f} | {metrics['train_fit']['macro_f1']:.3f} |",
        f"| val | {val['n']} | {val['accuracy']:.3f} | {val['macro_f1']:.3f} |",
        f"| test | {test['n']} | {test['accuracy']:.3f} | {test['macro_f1']:.3f} |",
        "",
        "### Val per-class",
        "",
        "| class | precision | recall | f1 | support |",
        "|-------|----------:|-------:|---:|--------:|",
    ]
    for name in HEAD_A_VIG_LABELS_2:
        r = val["per_class"][name]
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {int(r['support'])} |"
        )
    lines += [
        "",
        "### Test per-class",
        "",
        "| class | precision | recall | f1 | support |",
        "|-------|----------:|-------:|---:|--------:|",
    ]
    for name in HEAD_A_VIG_LABELS_2:
        r = test["per_class"][name]
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {int(r['support'])} |"
        )
    lines += [
        "",
        "## vs prior SC400-only / night-holdout",
        "",
        "Prior Head A binary on SC4001→SC4002 night holdout had bal macro-F1 ~0.74–0.82 (same-subject nights).",
        "This run is **subject-wise** holdout on 124 subjects — harder / more honest generalization.",
        "",
        "## Recommendation",
        "",
        metrics["recommendation"],
        "",
        "## Artifacts",
        "",
        f"- Weights: `{metrics['head_path']}`",
        f"- Metrics: `exports/head_a_vig_full_corpus/metrics_summary.json`",
        f"- Shared emb: `{EMB_CACHE.relative_to(ROOT)}/`",
        f"- Script: `scripts/train_heads_expanded_sleep_corpus.py`",
        "",
    ]
    DOCS_A.write_text("\n".join(lines))


def recommend(a_ship: bool, a_test_f1: float, c_test_f1: float) -> str:
    parts = [
        "**Freeze vs fine-tune (user decision):**",
        "",
    ]
    if a_ship and a_test_f1 >= 0.70:
        parts.append(
            f"- **A-vig:** subject-holdout macro-F1={a_test_f1:.3f} clears proxy ship bar → "
            "**prefer keep backbone frozen** for vig; fine-tune only if true-Muse cal data arrives "
            "or val/test gap suggests underfit to domain."
        )
    elif a_test_f1 >= 0.55:
        parts.append(
            f"- **A-vig:** macro-F1={a_test_f1:.3f} above chance but below strong ship → "
            "**try light fine-tune or more aggressive balance/thresholding** before declaring frozen ship; "
            "or keep frozen and rely on personal Muse calibration."
        )
    else:
        parts.append(
            f"- **A-vig:** macro-F1={a_test_f1:.3f} weak on subject holdout → "
            "frozen linear may be insufficient; **consider fine-tune** or label/montage revisit."
        )
    parts.append(
        f"- **Head C:** 2-way wake/light probe F1={c_test_f1:.3f} on N1-slice; "
        "**do not fine-tune backbone for 4-way staging** until sleep-period deep/REM windows exist. "
        "Frozen is fine for this probe."
    )
    parts.append(
        "- **A-med/A-eng:** skipped (known weak/confounded); no fine-tune recommendation from this run."
    )
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class-cap", type=int, default=DEFAULT_PER_CLASS_CAP)
    ap.add_argument("--max-train-balanced", type=int, default=MAX_TRAIN_BALANCED)
    ap.add_argument("--skip-encode", action="store_true", help="reuse existing emb_cache")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    OUT_C.mkdir(parents=True, exist_ok=True)
    OUT_A.mkdir(parents=True, exist_ok=True)
    EMB_CACHE.mkdir(parents=True, exist_ok=True)

    policy = load_policy()
    split_map = load_split_map()
    weights = find_weights()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA requested but unavailable → CPU", flush=True)

    print(f"device={device} weights={weights}", flush=True)
    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
    encoder.to(device)
    notes = encoder.adapter_notes()

    if args.skip_encode and any(EMB_CACHE.glob("*_emb.npz")):
        print("skip-encode: reusing emb_cache", flush=True)
        meta = json.loads((EMB_CACHE / "cache_meta.json").read_text()) if (EMB_CACHE / "cache_meta.json").exists() else {}
    else:
        print("embedding corpus (per-recording)…", flush=True)
        meta = ensure_emb_cache(
            encoder, device, policy, split_map, rng, per_class_cap=args.per_class_cap
        )

    # Free encoder before pooling large emb arrays
    del encoder
    gc.collect()

    sampling = meta.get(
        "sampling_note",
        f"per_class_cap={args.per_class_cap}; train_balance={TRAIN_BALANCE}; max_train_balanced={args.max_train_balanced}",
    )
    if args.per_class_cap > 0:
        sampling = (
            f"subsampled ≤{args.per_class_cap} windows/class/recording after QC; "
            f"train undersample balanced (cap {args.max_train_balanced})"
        )
    else:
        sampling = (
            f"full corpus after light QC (no per-recording cap); "
            f"train undersample balanced (cap {args.max_train_balanced})"
        )

    created = datetime.now(timezone.utc).isoformat()
    n_rec = len(list(EMB_CACHE.glob("*_emb.npz")))
    n_subj = len(set(split_map.keys()))

    # ---- Head A-vig ----
    print("pooling A-vig…", flush=True)
    emb_tr, y_tr, det_tr = load_pooled("train", "a")
    emb_va, y_va, det_va = load_pooled("val", "a")
    emb_te, y_te, det_te = load_pooled("test", "a")
    print("A train", det_tr["counts"], "n", len(y_tr), flush=True)
    print("A val", det_va["counts"], "n", len(y_va), flush=True)
    print("A test", det_te["counts"], "n", len(y_te), flush=True)

    if TRAIN_BALANCE:
        emb_fit, y_fit = undersample_balanced(emb_tr, y_tr, rng)
        if args.max_train_balanced > 0 and len(y_fit) > args.max_train_balanced:
            # further downsample keeping balance
            per = args.max_train_balanced // 2
            idxs = cap_idxs_per_class(y_fit, per, rng)
            emb_fit, y_fit = emb_fit[idxs], y_fit[idxs]
    else:
        emb_fit, y_fit = emb_tr, y_tr
    print("A fit", Counter(y_fit.tolist()), flush=True)

    emb_va_e, y_va_e = maybe_cap_eval(emb_va, y_va, EVAL_PER_CLASS_CAP, rng)
    emb_te_e, y_te_e = maybe_cap_eval(emb_te, y_te, EVAL_PER_CLASS_CAP, rng)

    print("training Head A-vig…", flush=True)
    head_a, hist_a, best_va_a = train_head(
        emb_fit,
        y_fit,
        emb_va_e,
        y_va_e,
        HEAD_A_VIG_LABELS_2,
        device,
        lambda in_dim, n_classes: HeadAVigLinear(in_dim=in_dim, n_classes=n_classes),
    )
    train_fit_a = eval_head(head_a, emb_fit, y_fit, HEAD_A_VIG_LABELS_2, device)
    val_a = eval_head(head_a, emb_va_e, y_va_e, HEAD_A_VIG_LABELS_2, device)
    test_a = eval_head(head_a, emb_te_e, y_te_e, HEAD_A_VIG_LABELS_2, device)
    a_ship, a_reason = ship_a_vig(test_a, val_a)
    head_a_path = OUT_A / "head_a_vig_linear.pt"
    torch.save(
        {
            "state_dict": head_a.state_dict(),
            "label_list": HEAD_A_VIG_LABELS_2,
            "in_dim": int(emb_fit.shape[-1]),
            "encoder": "CBraMod_frozen",
        },
        head_a_path,
    )

    # ---- Head C (same emb, y_c) ----
    print("pooling Head C…", flush=True)
    emb_tr_c, y_tr_c, det_tr_c = load_pooled("train", "c")
    emb_va_c, y_va_c, det_va_c = load_pooled("val", "c")
    emb_te_c, y_te_c, det_te_c = load_pooled("test", "c")
    print("C train", det_tr_c["counts"], flush=True)

    if TRAIN_BALANCE:
        emb_fit_c, y_fit_c = undersample_balanced(emb_tr_c, y_tr_c, rng)
        if args.max_train_balanced > 0 and len(y_fit_c) > args.max_train_balanced:
            per = args.max_train_balanced // 2
            idxs = cap_idxs_per_class(y_fit_c, per, rng)
            emb_fit_c, y_fit_c = emb_fit_c[idxs], y_fit_c[idxs]
    else:
        emb_fit_c, y_fit_c = emb_tr_c, y_tr_c

    emb_va_c_e, y_va_c_e = maybe_cap_eval(emb_va_c, y_va_c, EVAL_PER_CLASS_CAP, rng)
    emb_te_c_e, y_te_c_e = maybe_cap_eval(emb_te_c, y_te_c, EVAL_PER_CLASS_CAP, rng)

    print("training Head C…", flush=True)
    head_c, hist_c, best_va_c = train_head(
        emb_fit_c,
        y_fit_c,
        emb_va_c_e,
        y_va_c_e,
        HEAD_C_LABELS_ACTIVE,
        device,
        lambda in_dim, n_classes: HeadCLinear(in_dim=in_dim, n_classes=n_classes),
    )
    train_fit_c = eval_head(head_c, emb_fit_c, y_fit_c, HEAD_C_LABELS_ACTIVE, device)
    val_c = eval_head(head_c, emb_va_c_e, y_va_c_e, HEAD_C_LABELS_ACTIVE, device)
    test_c = eval_head(head_c, emb_te_c_e, y_te_c_e, HEAD_C_LABELS_ACTIVE, device)
    c_ship, c_reason = ship_head_c(test_c)
    head_c_path = OUT_C / "head_c_wake_light_linear.pt"
    torch.save(
        {
            "state_dict": head_c.state_dict(),
            "label_list": HEAD_C_LABELS_ACTIVE,
            "full_stage_coarse_labels": HEAD_C_COARSE_LABELS,
            "in_dim": int(emb_fit_c.shape[-1]),
            "encoder": "CBraMod_frozen",
            "note": "2-way wake/light on N1-slice; deep/rem absent",
        },
        head_c_path,
    )

    rec_text = recommend(a_ship, test_a["macro_f1"], test_c["macro_f1"])

    metrics_a = {
        "step": "head_a_vig_full_corpus_train",
        "created_utc": created,
        "ship_candidate": a_ship,
        "ship_reason": a_reason,
        "task": "drowsy_vs_hypnagogic",
        "label_list": HEAD_A_VIG_LABELS_2,
        "n_subjects": n_subj,
        "n_recordings": n_rec,
        "train_fit": train_fit_a,
        "val": val_a,
        "test": test_a,
        "history": hist_a,
        "best_val_macro_f1": best_va_a,
        "splits": {"train": det_tr, "val": det_va, "test": det_te},
        "sampling": sampling,
        "encoder_adapter": notes,
        "head_path": str(head_a_path.relative_to(ROOT)),
        "backbone_finetuned": False,
        "montage_tags": {
            "sleep_edf": "(ch_count=4, professional) muse4 proxy Fpz-Cz/Pz-Oz",
            "hmc": "(ch_count=4, professional) muse4 proxy F4-M1/C4-M1/C3-M2/O2-M1",
        },
        "recommendation": rec_text,
    }
    (OUT_A / "metrics_summary.json").write_text(json.dumps(metrics_a, indent=2) + "\n")
    (OUT_A / "run_manifest.json").write_text(
        json.dumps(
            {
                "step": "head_a_vig_full_corpus_train",
                "created_utc": created,
                "seed": SEED,
                "epochs_max": EPOCHS,
                "patience": PATIENCE,
                "lr": LR,
                "batch_enc": BATCH_ENC,
                "batch_head": BATCH_HEAD,
                "per_class_cap": args.per_class_cap,
                "max_train_balanced": args.max_train_balanced,
                "split_policy": "datasets/vigilance_sleep_edf/splits/",
                "emb_cache": str(EMB_CACHE.relative_to(ROOT)),
                "docs_path": str(DOCS_A.relative_to(ROOT)),
                "license_attribution": {
                    "Sleep-EDF": "PhysioNet ODC-By",
                    "HMC": "PhysioNet CC-BY-4.0 hmc-sleep-staging 1.1",
                    "CBraMod": "Apache-2.0",
                },
            },
            indent=2,
        )
        + "\n"
    )
    (OUT_A / "step_summary.json").write_text(
        json.dumps(
            {
                "step": "head_a_vig_full_corpus_train",
                "completed_utc": created,
                "ship_candidate": a_ship,
                "val_macro_f1": val_a["macro_f1"],
                "test_macro_f1": test_a["macro_f1"],
                "val_acc": val_a["accuracy"],
                "test_acc": test_a["accuracy"],
                "test_per_class_f1": {k: test_a["per_class"][k]["f1"] for k in HEAD_A_VIG_LABELS_2},
            },
            indent=2,
        )
        + "\n"
    )
    write_docs_a(metrics_a, sampling)

    metrics_c = {
        "step": "head_c_full_corpus_train",
        "created_utc": created,
        "ship_candidate": c_ship,
        "ship_reason": c_reason,
        "task": "stage_coarse_wake_light_2way_n1slice",
        "label_list": HEAD_C_LABELS_ACTIVE,
        "n_subjects": n_subj,
        "n_recordings": n_rec,
        "train_fit": train_fit_c,
        "val": val_c,
        "test": test_c,
        "history": hist_c,
        "best_val_macro_f1": best_va_c,
        "splits": {"train": det_tr_c, "val": det_va_c, "test": det_te_c},
        "sampling": sampling,
        "encoder_adapter": notes,
        "head_path": str(head_c_path.relative_to(ROOT)),
        "backbone_finetuned": False,
        "montage_tags": metrics_a["montage_tags"],
        "recommendation": rec_text,
        "note_identical_to_a_vig_on_n1slice": (
            "On this N1-slice corpus wake↔drowsy and light↔hypnagogic are 1:1; "
            "metrics should nearly match A-vig (same emb, same binary task)."
        ),
    }
    (OUT_C / "metrics_summary.json").write_text(json.dumps(metrics_c, indent=2) + "\n")
    (OUT_C / "run_manifest.json").write_text(
        json.dumps(
            {
                "step": "head_c_full_corpus_train",
                "created_utc": created,
                "seed": SEED,
                "label_list": HEAD_C_LABELS_ACTIVE,
                "docs_path": str(DOCS_C.relative_to(ROOT)),
                "emb_cache": str(EMB_CACHE.relative_to(ROOT)),
                "per_class_cap": args.per_class_cap,
            },
            indent=2,
        )
        + "\n"
    )
    (OUT_C / "step_summary.json").write_text(
        json.dumps(
            {
                "step": "head_c_full_corpus_train",
                "completed_utc": created,
                "ship_candidate": c_ship,
                "val_macro_f1": val_c["macro_f1"],
                "test_macro_f1": test_c["macro_f1"],
                "test_per_class_f1": {k: test_c["per_class"][k]["f1"] for k in HEAD_C_LABELS_ACTIVE},
            },
            indent=2,
        )
        + "\n"
    )
    write_docs_c(metrics_c, sampling)

    # Combined parent-facing summary
    summary = {
        "step": "train_heads_expanded_sleep_corpus",
        "completed_utc": created,
        "device": str(device),
        "n_subjects": n_subj,
        "n_recordings": n_rec,
        "sampling": sampling,
        "backbone_finetuned": False,
        "head_a_vig": {
            "ship_candidate": a_ship,
            "val_macro_f1": val_a["macro_f1"],
            "test_macro_f1": test_a["macro_f1"],
            "val_acc": val_a["accuracy"],
            "test_acc": test_a["accuracy"],
            "test_per_class_f1": {k: test_a["per_class"][k]["f1"] for k in HEAD_A_VIG_LABELS_2},
        },
        "head_c": {
            "ship_candidate": c_ship,
            "task": "wake_light_2way_n1slice",
            "val_macro_f1": val_c["macro_f1"],
            "test_macro_f1": test_c["macro_f1"],
            "test_per_class_f1": {k: test_c["per_class"][k]["f1"] for k in HEAD_C_LABELS_ACTIVE},
        },
        "recommendation_freeze_vs_finetune": rec_text,
        "paths": {
            "a_vig": str(OUT_A.relative_to(ROOT)),
            "head_c": str(OUT_C.relative_to(ROOT)),
            "docs_a": str(DOCS_A.relative_to(ROOT)),
            "docs_c": str(DOCS_C.relative_to(ROOT)),
        },
    }
    (ROOT / "exports" / "train_heads_expanded_sleep_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("DONE", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

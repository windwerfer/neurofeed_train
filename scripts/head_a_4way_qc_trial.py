#!/usr/bin/env python3
"""4-way Head A trial on QC-passed windows + fixed subject splits.

Frozen CBraMod + HeadALinear(4) with alternating vigilance/attention batches
and missing-class masked CE. Train/val/test from datasets/*/splits/.
"""
from __future__ import annotations

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
from src.head_a import (
    HEAD_A_4WAY_LABELS,
    HeadALinear,
    binary_ids_to_4way,
    class_weights_from_y,
    group_ids,
    masked_cross_entropy,
    present_class_ids,
    predict_present,
    undersample_balanced,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report

SEED = 42
EPOCHS = 6
BATCH = 32
LR = 1e-3
PATIENCE = 2  # early stop on mean(val_vig_f1, val_att_f1)
# Cap per recording/class for train balance (trial speed + avoid night domination)
VIG_PER_CLASS_PER_REC = 400
ATT_MAX_PER_SUBJECT = 400  # after per-subject undersample

OUT_DIR = ROOT / "exports" / "head_a_4way_qc_trial"
DOCS_PATH = ROOT / "docs" / "head_a_4way_qc_trial.md"
STEP = "head_a_4way_qc_trial"

VIG_CORPUS = "vigilance_sleep_edf"
ATT_CORPORA = ["attention_ds001787", "attention_ds003969"]


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


def load_split_subjects(corpus: str) -> Dict[str, List[str]]:
    d = ROOT / "datasets" / corpus / "splits"
    return {
        sp: list(json.loads((d / f"{sp}_subjects.json").read_text())["subjects"])
        for sp in ("train", "val", "test")
    }


def subject_for_recording(corpus: str, recording_id: str) -> str:
    if corpus == VIG_CORPUS:
        return recording_id[:5]
    if corpus == "attention_ds001787":
        core = recording_id.split("_")[0]
        num = core.replace("sub", "")
        return f"sub-{num.zfill(3) if num.isdigit() else num}"
    if corpus == "attention_ds003969":
        num = recording_id.replace("sub", "")
        return f"sub-{num.zfill(3) if num.isdigit() else num}"
    return recording_id


def map_y_to_4way(y_local: np.ndarray, label_names: Sequence[str]) -> np.ndarray:
    names = [str(n) for n in label_names]
    # Build local→global map
    table = {i: HEAD_A_4WAY_LABELS.index(n) for i, n in enumerate(names) if n in HEAD_A_4WAY_LABELS}
    out = np.empty_like(y_local)
    for i, v in enumerate(y_local):
        if int(v) not in table:
            raise ValueError(f"label id {v} not in {names}")
        out[i] = table[int(v)]
    return out.astype(np.int64)


def load_corpus_windows(
    corpus: str,
    qc_only: bool = True,
) -> List[Dict[str, Any]]:
    win_dir = ROOT / "datasets" / corpus / "windows"
    packs: List[Dict[str, Any]] = []
    for npz_path in sorted(win_dir.glob("*_windows.npz")):
        if npz_path.name.endswith("_qc.npz"):
            continue
        rid = npz_path.name.replace("_windows.npz", "")
        qc_path = win_dir / f"{rid}_qc.npz"
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y_local = data["y"].astype(np.int64)
        label_names = [str(v) for v in data["label_names"].tolist()]
        starts = (
            data["starts"].astype(np.int64)
            if "starts" in data.files
            else np.arange(len(y_local), dtype=np.int64)
        )
        if qc_only and qc_path.exists():
            qc = np.load(qc_path, allow_pickle=True)
            mask = qc["qc_pass"].astype(bool)
        else:
            mask = np.ones(len(y_local), dtype=bool)
        if not mask.any():
            continue
        X = X[mask]
        y_local = y_local[mask]
        starts = starts[mask]
        y = map_y_to_4way(y_local, label_names)
        sid = subject_for_recording(corpus, rid)
        packs.append(
            {
                "corpus": corpus,
                "recording_id": rid,
                "subject_id": sid,
                "X": X,
                "y": y,
                "starts": starts,
                "n_raw": int(len(mask)),
                "n_qc": int(mask.sum()),
                "npz_path": str(npz_path.relative_to(ROOT)),
                "qc_path": str(qc_path.relative_to(ROOT)) if qc_path.exists() else None,
                "npz_sha256": sha256(npz_path),
                "counts": {HEAD_A_4WAY_LABELS[i]: int((y == i).sum()) for i in range(4)},
            }
        )
    return packs


def assign_splits(
    packs: List[Dict[str, Any]], split_map: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for p in packs:
        sp = split_map.get(p["subject_id"])
        if sp is None:
            continue
        out[sp].append(p)
    return out


def balance_vig_train(
    packs: List[Dict[str, Any]], per_class: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    Xs, ys, starts, detail = [], [], [], {}
    for p in packs:
        y, X, st = p["y"], p["X"], p["starts"]
        idxs = []
        used = {}
        for c in (2, 3):  # drowsy, hypnagogic
            cand = np.where(y == c)[0]
            n = min(per_class, len(cand))
            if n == 0:
                used[HEAD_A_4WAY_LABELS[c]] = 0
                continue
            pick = rng.choice(cand, size=n, replace=False)
            idxs.append(pick)
            used[HEAD_A_4WAY_LABELS[c]] = int(n)
        if not idxs:
            detail[p["recording_id"]] = {"skipped": True, "reason": "no_vig_labels"}
            continue
        idxs = np.concatenate(idxs)
        rng.shuffle(idxs)
        Xs.append(X[idxs])
        ys.append(y[idxs])
        starts.append(st[idxs])
        detail[p["recording_id"]] = {
            "subject_id": p["subject_id"],
            "per_class_used": used,
            "n_total": int(len(idxs)),
        }
    if not Xs:
        raise RuntimeError("no vigilance train windows after balance")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(starts), detail


def balance_att_train(
    packs: List[Dict[str, Any]],
    max_per_subject: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    # Group by subject so multi-recording subjects don't double-count imbalance
    by_subj: Dict[str, List[Dict[str, Any]]] = {}
    for p in packs:
        by_subj.setdefault(p["subject_id"], []).append(p)

    Xs, ys, starts, detail = [], [], [], {}
    for sid, plist in by_subj.items():
        X = np.concatenate([p["X"] for p in plist])
        y = np.concatenate([p["y"] for p in plist])
        st = np.concatenate([p["starts"] for p in plist])
        present = present_class_ids(y, 4)
        # Prefer subjects with both attention classes
        att_present = [c for c in present if c in (0, 1)]
        if len(att_present) < 2:
            detail[sid] = {
                "skipped": True,
                "reason": "single_class",
                "counts": {HEAD_A_4WAY_LABELS[i]: int((y == i).sum()) for i in range(4)},
            }
            continue
        # Keep only attention labels
        mask = (y == 0) | (y == 1)
        Xb, yb = undersample_balanced(X[mask], y[mask], rng)
        st_b = st[mask]
        # undersample_balanced reshuffles; re-align starts by resampling same way —
        # simpler: drop starts alignment for train (not needed for CE)
        if len(yb) > max_per_subject:
            # keep class balance while capping
            half = max_per_subject // 2
            keep = []
            for c in (0, 1):
                cand = np.where(yb == c)[0]
                n = min(half, len(cand))
                keep.append(rng.choice(cand, size=n, replace=False))
            keep = np.concatenate(keep)
            rng.shuffle(keep)
            Xb, yb = Xb[keep], yb[keep]
        Xs.append(Xb)
        ys.append(yb)
        starts.append(np.arange(len(yb), dtype=np.int64))  # placeholder
        detail[sid] = {
            "skipped": False,
            "n_total": int(len(yb)),
            "per_class": {HEAD_A_4WAY_LABELS[i]: int((yb == i).sum()) for i in (0, 1)},
            "recordings": [p["recording_id"] for p in plist],
        }
    if not Xs:
        raise RuntimeError("no attention train subjects with both classes")
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(starts), detail


def concat_packs(
    packs: List[Dict[str, Any]], domain: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not packs:
        return (
            np.zeros((0, 4, 512), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    Xs, ys, sts = [], [], []
    for p in packs:
        y = p["y"]
        if domain == "vigilance":
            mask = (y == 2) | (y == 3)
        else:
            mask = (y == 0) | (y == 1)
        if not mask.any():
            continue
        Xs.append(p["X"][mask])
        ys.append(y[mask])
        sts.append(p["starts"][mask])
    if not Xs:
        return (
            np.zeros((0, 4, 512), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(sts)


def stride4(X: np.ndarray, y: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) == 0:
        return X, y
    order = np.argsort(starts)
    idx = order[::4]
    return X[idx], y[idx]


def encode_all(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(X) == 0:
        return torch.zeros((0, 200), dtype=torch.float32)
    Xt = torch.from_numpy(X.astype(np.float32))
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            emb_list.append(encoder(Xt[i : i + BATCH].to(device)).cpu())
    return torch.cat(emb_list, dim=0)


def present_macro_f1(y: np.ndarray, pred: np.ndarray, present: List[int]) -> float:
    if len(y) == 0:
        return float("nan")
    names = [HEAD_A_4WAY_LABELS[i] for i in present]
    remap = {c: i for i, c in enumerate(present)}
    yt = [remap[int(v)] for v in y]
    yp = [remap[int(v)] for v in pred]
    return macro_f1(yt, yp, names)


def eval_split(
    head: nn.Module,
    emb: torch.Tensor,
    y: np.ndarray,
    present: List[int],
    name: str,
) -> Dict[str, Any]:
    head.eval()
    if len(y) == 0:
        return {
            "name": name,
            "n": 0,
            "acc": None,
            "macro_f1_present": None,
            "present_classes": [HEAD_A_4WAY_LABELS[i] for i in present],
            "per_class": {},
            "confusion_4way": [[0] * 4 for _ in range(4)],
            "counts_true": {HEAD_A_4WAY_LABELS[i]: 0 for i in range(4)},
        }
    with torch.no_grad():
        logits = head(emb)
        pred = predict_present(logits, present).numpy()
    report_full = per_class_report(y.tolist(), pred.tolist(), HEAD_A_4WAY_LABELS)
    present_names = [HEAD_A_4WAY_LABELS[i] for i in present]
    return {
        "name": name,
        "n": int(len(y)),
        "acc": float((pred == y).mean()),
        "macro_f1_present": present_macro_f1(y, pred, present),
        "present_classes": present_names,
        "per_class": {k: report_full[k] for k in present_names if k in report_full},
        "confusion_4way": confusion_matrix(y.tolist(), pred.tolist(), HEAD_A_4WAY_LABELS).tolist(),
        "counts_true": {HEAD_A_4WAY_LABELS[i]: int((y == i).sum()) for i in range(4)},
    }


def train_alternating(
    emb_vig: torch.Tensor,
    y_vig: np.ndarray,
    emb_att: torch.Tensor,
    y_att: np.ndarray,
    emb_v_va: torch.Tensor,
    y_v_va: np.ndarray,
    emb_a_va: torch.Tensor,
    y_a_va: np.ndarray,
    device: torch.device,
) -> Tuple[HeadALinear, List[Dict[str, Any]], Dict[str, Any]]:
    att_ids, vig_ids = group_ids()
    head = HeadALinear(in_dim=emb_vig.shape[-1], n_classes=4).to(device)
    y_all = np.concatenate([y_vig, y_att])
    w = class_weights_from_y(y_all, n_classes=4).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=LR)

    vig_loader = DataLoader(
        TensorDataset(emb_vig, torch.from_numpy(y_vig)),
        batch_size=BATCH,
        shuffle=True,
    )
    att_loader = DataLoader(
        TensorDataset(emb_att, torch.from_numpy(y_att)),
        batch_size=BATCH,
        shuffle=True,
    )

    history: List[Dict[str, Any]] = []
    best_state = None
    best_score = -1.0
    best_epoch = 0
    stale = 0

    for ep in range(EPOCHS):
        head.train()
        total, n_seen = 0.0, 0
        vig_it = iter(vig_loader)
        att_it = iter(att_loader)
        n_steps = max(len(vig_loader), len(att_loader))
        for _ in range(n_steps):
            try:
                xb_v, yb_v = next(vig_it)
            except StopIteration:
                vig_it = iter(vig_loader)
                xb_v, yb_v = next(vig_it)
            try:
                xb_a, yb_a = next(att_it)
            except StopIteration:
                att_it = iter(att_loader)
                xb_a, yb_a = next(att_it)
            for xb, yb, present in (
                (xb_v, yb_v, vig_ids),
                (xb_a, yb_a, att_ids),
            ):
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = head(xb)
                loss = masked_cross_entropy(logits, yb, present, weight=w)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(yb)
                n_seen += len(yb)

        head.eval()
        with torch.no_grad():
            tr_v = predict_present(head(emb_vig.to(device)), vig_ids).cpu().numpy()
            va_v = predict_present(head(emb_v_va.to(device)), vig_ids).cpu().numpy()
            tr_a = predict_present(head(emb_att.to(device)), att_ids).cpu().numpy()
            va_a = predict_present(head(emb_a_va.to(device)), att_ids).cpu().numpy()
        row = {
            "epoch": ep + 1,
            "loss": total / max(n_seen, 1),
            "train_vig_macro_f1": present_macro_f1(y_vig, tr_v, vig_ids),
            "val_vig_macro_f1": present_macro_f1(y_v_va, va_v, vig_ids),
            "train_att_macro_f1": present_macro_f1(y_att, tr_a, att_ids),
            "val_att_macro_f1": present_macro_f1(y_a_va, va_a, att_ids),
        }
        row["val_mean_macro_f1"] = float(
            np.nanmean([row["val_vig_macro_f1"], row["val_att_macro_f1"]])
        )
        history.append(row)
        print(row, flush=True)

        score = row["val_mean_macro_f1"]
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop at epoch {ep+1}; best={best_epoch}", flush=True)
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    splits_meta = {
        "vig_train_n": int(len(y_vig)),
        "att_train_n": int(len(y_att)),
        "vig_val_n": int(len(y_v_va)),
        "att_val_n": int(len(y_a_va)),
        "best_epoch": best_epoch,
        "best_val_mean_macro_f1": best_score,
        "val_vig_macro_f1": history[best_epoch - 1]["val_vig_macro_f1"] if history else None,
        "val_att_macro_f1": history[best_epoch - 1]["val_att_macro_f1"] if history else None,
    }
    return head, history, splits_meta


def write_docs(metrics: Dict[str, Any], summary: Dict[str, Any]) -> None:
    h = metrics["headline"]
    lines = [
        "# Head A 4-way QC trial",
        "",
        f"**Step:** `{STEP}`",
        f"**Completed (UTC):** {summary['completed_utc']}",
        "",
        "## Design",
        "",
        "| Piece | Choice |",
        "|-------|--------|",
        "| QC | light artifact QC (`qc_pass` only) |",
        "| Splits | fixed subject JSON under `datasets/*/splits/` |",
        "| Head | `HeadALinear` → 4 logits |",
        "| Schedule | alternating masked vigilance / attention batches |",
        "| Encoder | frozen CBraMod (CPU) |",
        f"| Epochs | {EPOCHS} (early stop patience={PATIENCE}) |",
        "",
        "## QC drop rates (pre-train)",
        "",
        "See `docs/artifact_qc_light.md` / `exports/artifact_qc_light/summary.json`.",
        "",
        f"- vigilance: drop_rate={summary['qc']['vigilance_sleep_edf']['drop_rate']:.4f}",
        f"- attention_ds001787: drop_rate={summary['qc']['attention_ds001787']['drop_rate']:.4f}",
        f"- attention_ds003969: drop_rate={summary['qc']['attention_ds003969']['drop_rate']:.4f}",
        "",
        "## Headline metrics",
        "",
        "| Split | macro-F1 (present) |",
        "|-------|-------------------:|",
        f"| Val vigilance | {h['val_vig_macro_f1']:.4f} |",
        f"| Val attention | {h['val_att_macro_f1']:.4f} |",
        f"| Test vigilance (subject holdout) | {h['test_vig_macro_f1']:.4f} |",
        f"| Test vigilance stride×4 | {h['test_vig_stride4_macro_f1']:.4f} |",
        f"| Test attention (subject holdout, full) | {h['test_att_full_macro_f1']:.4f} |",
        f"| Test attention balanced | {h['test_att_balanced_macro_f1']:.4f} |",
        f"| ship_candidate | {h['ship_candidate']} |",
        "",
        "## Attention holdout collapsed?",
        "",
        f"**{summary['attention_holdout_collapsed_yes_no']}** — {summary['attention_note']}",
        "",
        "## Takeaway",
        "",
        summary["takeaway"],
        "",
        "## Artifacts",
        "",
        f"- Script: `scripts/head_a_4way_qc_trial.py`",
        f"- Head: `exports/head_a_4way_qc_trial/head_a_4way_qc_trial.pt`",
        f"- Metrics: `exports/head_a_4way_qc_trial/metrics_summary.json`",
        f"- Manifest: `exports/head_a_4way_qc_trial/run_manifest.json`",
        "",
    ]
    DOCS_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    att_ids, vig_ids = group_ids()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    qc_summary = json.loads(
        (ROOT / "exports" / "artifact_qc_light" / "summary.json").read_text()
    )
    qc_rates = {
        c: {
            "drop_rate": qc_summary["corpora"][c]["drop_rate"],
            "n_total": qc_summary["corpora"][c]["n_total"],
            "n_drop": qc_summary["corpora"][c]["n_drop"],
            "n_pass": qc_summary["corpora"][c]["n_pass"],
        }
        for c in [VIG_CORPUS] + ATT_CORPORA
    }

    # --- load + split ---
    vig_packs = load_corpus_windows(VIG_CORPUS, qc_only=True)
    vig_split_subjects = load_split_subjects(VIG_CORPUS)
    vig_split_map = {
        sid: sp for sp, sids in vig_split_subjects.items() for sid in sids
    }
    vig_by = assign_splits(vig_packs, vig_split_map)

    att_packs: List[Dict[str, Any]] = []
    att_split_map: Dict[str, str] = {}
    for corp in ATT_CORPORA:
        packs = load_corpus_windows(corp, qc_only=True)
        att_packs.extend(packs)
        subjects = load_split_subjects(corp)
        # Namespace subject keys by corpus to avoid sub-001 collision across corpora
        for sp, sids in subjects.items():
            for sid in sids:
                att_split_map[f"{corp}::{sid}"] = sp
        for p in packs:
            p["subject_key"] = f"{corp}::{p['subject_id']}"

    # Remap assign_splits for attention using subject_key
    att_by: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for p in att_packs:
        sp = att_split_map.get(p["subject_key"])
        if sp:
            att_by[sp].append(p)

    print("vig subjects", {k: sorted({p['subject_id'] for p in v}) for k, v in vig_by.items()})
    print(
        "att subjects",
        {k: sorted({p['subject_key'] for p in v}) for k, v in att_by.items()},
    )

    X_v_tr, y_v_tr, st_v_tr, vig_bal = balance_vig_train(
        vig_by["train"], VIG_PER_CLASS_PER_REC, rng
    )
    X_a_tr, y_a_tr, st_a_tr, att_bal = balance_att_train(
        att_by["train"], ATT_MAX_PER_SUBJECT, rng
    )
    X_v_va, y_v_va, st_v_va = concat_packs(vig_by["val"], "vigilance")
    X_a_va, y_a_va, st_a_va = concat_packs(att_by["val"], "attention")
    # Val attention: undersample for clearer F1 signal
    if len(y_a_va) and len(present_class_ids(y_a_va, 4)) >= 2:
        X_a_va_b, y_a_va_b = undersample_balanced(X_a_va, y_a_va, rng)
    else:
        X_a_va_b, y_a_va_b = X_a_va, y_a_va

    X_v_te, y_v_te, st_v_te = concat_packs(vig_by["test"], "vigilance")
    X_v_te_s4, y_v_te_s4 = stride4(X_v_te, y_v_te, st_v_te)
    X_a_te, y_a_te, st_a_te = concat_packs(att_by["test"], "attention")
    if len(y_a_te) and len(present_class_ids(y_a_te, 4)) >= 2:
        X_a_te_b, y_a_te_b = undersample_balanced(X_a_te, y_a_te, rng)
    else:
        X_a_te_b, y_a_te_b = X_a_te, y_a_te

    print("shapes",
          "v_tr", X_v_tr.shape, Counter(y_v_tr.tolist()),
          "a_tr", X_a_tr.shape, Counter(y_a_tr.tolist()),
          "v_va", X_v_va.shape, "a_va", X_a_va_b.shape,
          "v_te", X_v_te.shape, "a_te", X_a_te.shape,
          flush=True)

    weights = find_weights()
    encoder = FrozenCBraModEncoder(weights, source_sr=256.0, pool="mean")
    notes = encoder.adapter_notes()
    encoder.to(device)
    print("encoder", notes["native_input"], "<-", notes["fed_input"], flush=True)

    print("encoding…", flush=True)
    emb_v_tr = encode_all(encoder, X_v_tr, device)
    emb_a_tr = encode_all(encoder, X_a_tr, device)
    emb_v_va = encode_all(encoder, X_v_va, device)
    emb_a_va = encode_all(encoder, X_a_va_b, device)
    emb_v_te = encode_all(encoder, X_v_te, device)
    emb_v_te_s4 = encode_all(encoder, X_v_te_s4, device)
    emb_a_te = encode_all(encoder, X_a_te, device)
    emb_a_te_b = encode_all(encoder, X_a_te_b, device)

    head, history, splits_meta = train_alternating(
        emb_v_tr, y_v_tr, emb_a_tr, y_a_tr,
        emb_v_va, y_v_va, emb_a_va, y_a_va_b,
        device,
    )

    evals = {
        "val_vig": eval_split(head, emb_v_va, y_v_va, vig_ids, "val_vig"),
        "val_att_balanced": eval_split(head, emb_a_va, y_a_va_b, att_ids, "val_att_balanced"),
        "test_vig_full": eval_split(head, emb_v_te, y_v_te, vig_ids, "test_vig_full"),
        "test_vig_stride4": eval_split(head, emb_v_te_s4, y_v_te_s4, vig_ids, "test_vig_stride4"),
        "test_att_full": eval_split(head, emb_a_te, y_a_te, att_ids, "test_att_full"),
        "test_att_balanced": eval_split(head, emb_a_te_b, y_a_te_b, att_ids, "test_att_balanced"),
        "train_vig": eval_split(head, emb_v_tr, y_v_tr, vig_ids, "train_vig"),
        "train_att": eval_split(head, emb_a_tr, y_a_tr, att_ids, "train_att"),
    }
    for k, v in evals.items():
        print(k, {kk: v[kk] for kk in ("n", "acc", "macro_f1_present")}, flush=True)

    # ship_candidate: both domain holdouts clearly above chance (~0.5 for balanced binary)
    att_te_f1 = evals["test_att_balanced"]["macro_f1_present"] or 0.0
    vig_te_f1 = evals["test_vig_stride4"]["macro_f1_present"] or 0.0
    ship = bool(att_te_f1 >= 0.55 and vig_te_f1 >= 0.55)
    att_collapsed = bool(att_te_f1 < 0.45)

    headline = {
        "val_vig_macro_f1": evals["val_vig"]["macro_f1_present"],
        "val_att_macro_f1": evals["val_att_balanced"]["macro_f1_present"],
        "test_vig_macro_f1": evals["test_vig_full"]["macro_f1_present"],
        "test_vig_stride4_macro_f1": vig_te_f1,
        "test_att_full_macro_f1": evals["test_att_full"]["macro_f1_present"],
        "test_att_balanced_macro_f1": att_te_f1,
        "ship_candidate": ship,
        "attention_holdout_collapsed": att_collapsed,
    }

    head_path = OUT_DIR / "head_a_4way_qc_trial.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": HEAD_A_4WAY_LABELS,
            "train_schedule": "alternating_masked_vigilance_attention_batches",
            "in_dim": int(emb_v_tr.shape[-1]),
            "n_classes": 4,
            "head": "HeadALinear",
            "qc_only": True,
            "fixed_subject_splits": True,
            "best_epoch": splits_meta["best_epoch"],
        },
        head_path,
    )

    metrics = {
        "qc_rates": qc_rates,
        "vig_balance": vig_bal,
        "att_balance": att_bal,
        "splits_meta": splits_meta,
        "history": history,
        "evals": evals,
        "headline": headline,
        "subjects": {
            "vig_train": vig_split_subjects["train"],
            "vig_val": vig_split_subjects["val"],
            "vig_test": vig_split_subjects["test"],
            "att_train": sorted({p["subject_key"] for p in att_by["train"]}),
            "att_val": sorted({p["subject_key"] for p in att_by["val"]}),
            "att_test": sorted({p["subject_key"] for p in att_by["test"]}),
        },
    }
    metrics_path = OUT_DIR / "metrics_summary.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    manifest = {
        "task": STEP,
        "label_list": HEAD_A_4WAY_LABELS,
        "qc_only": True,
        "qc_summary_ref": "exports/artifact_qc_light/summary.json",
        "train_schedule": {
            "type": "alternating_masked_batches",
            "vigilance_present": [HEAD_A_4WAY_LABELS[i] for i in vig_ids],
            "attention_present": [HEAD_A_4WAY_LABELS[i] for i in att_ids],
        },
        "encoder_name": "CBraMod",
        "encoder_weights_path": str(weights),
        "encoder_weights_sha256": notes["weights_sha256"],
        "encoder_adapter": notes,
        "head": "HeadALinear",
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch": BATCH,
        "lr": LR,
        "seed": SEED,
        "vig_per_class_per_rec": VIG_PER_CLASS_PER_REC,
        "att_max_per_subject": ATT_MAX_PER_SUBJECT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_ref": str(metrics_path.relative_to(ROOT)),
        "head_path": str(head_path.relative_to(ROOT)),
        "caveats": [
            "montage/domain gap: Sleep-EDF uV vs attention V-scale vs TUEG pretrain",
            "QC is light (obvious junk only); most windows pass",
            "attention subject transfer historically near chance",
        ],
    }
    man_path = OUT_DIR / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))

    yes_no = "YES" if att_collapsed else "NO"
    att_note = (
        f"test attention balanced macro-F1={att_te_f1:.3f} "
        + ("still near/below chance — collapsed." if att_collapsed else "above collapse threshold.")
    )
    takeaway = (
        f"4-way Head A QC trial (fixed subject splits, QC-passed only). "
        f"Val vig/att F1={headline['val_vig_macro_f1']:.3f}/{headline['val_att_macro_f1']:.3f}; "
        f"test vig s4 F1={vig_te_f1:.3f}; test att bal F1={att_te_f1:.3f}; "
        f"ship_candidate={ship}."
    )
    summary = {
        "step": STEP,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "qc": qc_rates,
        "headline": headline,
        "attention_holdout_collapsed_yes_no": yes_no,
        "attention_note": att_note,
        "takeaway": takeaway,
        "artifacts": {
            "script": "scripts/head_a_4way_qc_trial.py",
            "qc_script": "scripts/dataset/artifact_qc_windows.py",
            "qc_docs": "docs/artifact_qc_light.md",
            "qc_summary": "exports/artifact_qc_light/summary.json",
            "docs": "docs/head_a_4way_qc_trial.md",
            "metrics": str(metrics_path.relative_to(ROOT)),
            "manifest": str(man_path.relative_to(ROOT)),
            "head": str(head_path.relative_to(ROOT)),
        },
    }
    (OUT_DIR / "step_summary.json").write_text(json.dumps(summary, indent=2))
    write_docs(metrics, summary)
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

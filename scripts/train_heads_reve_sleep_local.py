#!/usr/bin/env python3
"""Local REVE frozen A-vig + Head C on expanded sleep corpus (CPU subsample).

Full-corpus REVE embed is ~13 win/s on CPU (~11h for 519k) → first pass uses
per-recording class cap; document sampling. Prefer Kaggle T4 for full encode.
Offline weights: kaggle_datasets/muse-eeg-heads-cache/models/reve-*.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.reve_encoder import FrozenREVEEncoder
from src.head_c import HeadCLinear, class_weights_from_y, undersample_balanced
from src.heads.head_a_vig import HEAD_A_VIG_LABELS_2, HeadAVigLinear
from src.metrics import confusion_matrix, per_class_report

SEED = 42
TARGET_SR = 256.0
BATCH_ENC = 16
BATCH_HEAD = 256
EPOCHS = 20
LR = 1e-3
PATIENCE = 5
FLAT_STD = 0.1
PEAK_ABS = 350.0
MAX_TRAIN_BALANCED = 40_000
HEAD_C_LABELS = ["wake", "light"]

WIN_DIR = ROOT / "datasets/vigilance_sleep_edf/windows"
SPLITS_DIR = ROOT / "datasets/vigilance_sleep_edf/splits"
OUT = ROOT / "exports/reve_sleep_heads_local"
EMB_CACHE = OUT / "emb_cache"
DOCS = ROOT / "docs/reve_sleep_heads_local.md"


def load_split_map():
    out = {}
    for sp in ("train", "val", "test"):
        for sid in json.loads((SPLITS_DIR / f"{sp}_subjects.json").read_text())["subjects"]:
            out[sid] = sp
    return out


def load_policy():
    return json.loads((SPLITS_DIR / "split_policy.json").read_text())


def subject_of(rid, policy):
    recs = policy.get("recordings", {})
    if rid in recs:
        return recs[rid]["subject_id"]
    return rid if rid.startswith("SN") else rid[:5]


def qc_mask(X):
    std = X.std(axis=-1)
    peak = np.max(np.abs(X), axis=-1)
    return ~((std < FLAT_STD).any(axis=-1) | (peak > PEAK_ABS).any(axis=-1))


def cap_idxs(y, per_class, rng):
    if per_class <= 0:
        return np.arange(len(y))
    picks = []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        n = min(per_class, len(cand))
        if n:
            picks.append(rng.choice(cand, size=n, replace=False))
    if not picks:
        return np.zeros(0, dtype=np.int64)
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


@torch.no_grad()
def encode(enc, X, device):
    chunks = []
    for i in range(0, len(X), BATCH_ENC):
        xb = torch.from_numpy(X[i : i + BATCH_ENC]).to(device)
        chunks.append(enc(xb).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, 0) if chunks else np.zeros((0, enc.emb_dim), np.float32)


def eval_head(head, emb, y, labels, device):
    head.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(emb), BATCH_HEAD):
            logits = head(torch.from_numpy(emb[i : i + BATCH_HEAD]).to(device))
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
    pred = np.asarray(preds, np.int64)
    report = per_class_report(y.tolist(), pred.tolist(), list(labels))
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()) if len(y) else 0.0,
        "macro_f1": float(report["macro_f1"]["f1"]),
        "per_class": {k: v for k, v in report.items() if k != "macro_f1"},
        "pred_counts": {labels[i]: int((pred == i).sum()) for i in range(len(labels))},
        "true_counts": {labels[i]: int((y == i).sum()) for i in range(len(labels))},
    }


def train_head(emb_fit, y_fit, emb_val, y_val, labels, ctor, device):
    head = ctor(in_dim=emb_fit.shape[-1], n_classes=len(labels)).to(device)
    w = class_weights_from_y(y_fit, n_classes=len(labels))
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(torch.from_numpy(emb_fit), torch.from_numpy(y_fit)), batch_size=BATCH_HEAD, shuffle=True)
    best = -1.0
    best_state = None
    stale = 0
    hist = []
    for ep in range(EPOCHS):
        head.train()
        total = n = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        tr = eval_head(head, emb_fit, y_fit, labels, device)
        va = eval_head(head, emb_val, y_val, labels, device)
        row = {"epoch": ep + 1, "loss": total / max(n, 1), "train_macro_f1": tr["macro_f1"], "val_macro_f1": va["macro_f1"], "val_acc": va["accuracy"]}
        hist.append(row)
        print(row, flush=True)
        if va["macro_f1"] > best + 1e-4:
            best = va["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop {ep+1}", flush=True)
                break
    if best_state:
        head.load_state_dict(best_state)
    return head, hist, best


def load_pooled(split, task):
    embs, ys = [], []
    key = "y_a" if task == "a" else "y_c"
    labels = HEAD_A_VIG_LABELS_2 if task == "a" else HEAD_C_LABELS
    detail = {"counts": Counter(), "n_subjects": 0}
    subs = set()
    for p in sorted(EMB_CACHE.glob("*_emb.npz")):
        z = np.load(p, allow_pickle=True)
        if str(z["split"]) != split:
            continue
        emb = z["emb"].astype(np.float32)
        y = z[key].astype(np.int64)
        m = y >= 0
        emb, y = emb[m], y[m]
        if len(y) == 0:
            continue
        embs.append(emb)
        ys.append(y)
        subs.add(str(z["subject_id"]))
        detail["counts"].update({labels[j]: int((y == j).sum()) for j in range(len(labels))})
    detail["counts"] = dict(detail["counts"])
    detail["n_subjects"] = len(subs)
    return np.concatenate(embs), np.concatenate(ys), detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class-cap", type=int, default=80, help="windows/class/recording; 0=full (slow on CPU)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    policy = load_policy()
    split_map = load_split_map()
    sampling = (
        f"REVE local CPU subsample ≤{args.per_class_cap} windows/class/recording after QC; "
        f"train undersample balanced (cap {MAX_TRAIN_BALANCED}). Full encode deferred to Kaggle T4."
        if args.per_class_cap > 0
        else "full corpus after QC (CPU — expect hours)"
    )
    print("loading REVE…", flush=True)
    enc = FrozenREVEEncoder(source_sr=TARGET_SR, prefer_local=True, device=device)
    enc.to(device)
    enc.eval()
    notes = enc.adapter_notes()
    print("local", enc.model_from_local, notes.get("weights_source", notes), flush=True)

    rids = sorted(policy.get("recordings", {}).keys()) or sorted(p.name.replace("_windows.npz", "") for p in WIN_DIR.glob("*_windows.npz"))
    for i, rid in enumerate(rids):
        cache = EMB_CACHE / f"{rid}_emb.npz"
        sid = subject_of(rid, policy)
        split = split_map.get(sid)
        if split is None:
            continue
        if cache.exists():
            if (i + 1) % 20 == 0:
                print(f"cache {i+1}/{len(rids)}", flush=True)
            continue
        npz = WIN_DIR / f"{rid}_windows.npz"
        if not npz.exists():
            continue
        data = np.load(npz, allow_pickle=True)
        X = data["X"].astype(np.float32)
        names = [str(x) for x in data["label_names"].tolist()]
        y_raw = data["y"].astype(np.int64)
        name_to = {n: HEAD_A_VIG_LABELS_2.index(n) for n in names if n in HEAD_A_VIG_LABELS_2}
        y_a = np.full(len(y_raw), -1, np.int64)
        for oi, n in enumerate(names):
            if n in name_to:
                y_a[y_raw == oi] = name_to[n]
        coarse = np.asarray(data["stage_coarse"], dtype=object)
        y_c = np.array([HEAD_C_LABELS.index(str(c)) if str(c) in HEAD_C_LABELS else -1 for c in coarse], dtype=np.int64)
        keep = qc_mask(X) & (y_a >= 0) & (y_c >= 0)
        idxs = np.where(keep)[0]
        if len(idxs) == 0:
            continue
        if args.per_class_cap > 0:
            local = cap_idxs(y_a[idxs], args.per_class_cap, rng)
            idxs = idxs[local]
        X_use = X[idxs]
        emb = encode(enc, X_use, device)
        np.savez_compressed(
            cache,
            emb=emb,
            y_a=y_a[idxs].astype(np.int64),
            y_c=y_c[idxs].astype(np.int64),
            subject_id=np.asarray(sid),
            split=np.asarray(split),
            recording_id=np.asarray(rid),
            per_class_cap=np.asarray(args.per_class_cap),
        )
        print(f"encoded {i+1}/{len(rids)} {rid} split={split} n={len(idxs)}", flush=True)
        del data, X, X_use, emb
        gc.collect()

    del enc
    gc.collect()
    created = datetime.now(timezone.utc).isoformat()

    def run_task(task, labels, ctor, out_name):
        emb_tr, y_tr, dtr = load_pooled("train", task)
        emb_va, y_va, dva = load_pooled("val", task)
        emb_te, y_te, dte = load_pooled("test", task)
        print(task, "train", dtr["counts"], "val", dva["counts"], "test", dte["counts"], flush=True)
        emb_fit, y_fit = undersample_balanced(emb_tr, y_tr, rng)
        if len(y_fit) > MAX_TRAIN_BALANCED:
            per = MAX_TRAIN_BALANCED // 2
            idx = cap_idxs(y_fit, per, rng)
            emb_fit, y_fit = emb_fit[idx], y_fit[idx]
        head, hist, best = train_head(emb_fit, y_fit, emb_va, y_va, labels, ctor, device)
        train_m = eval_head(head, emb_fit, y_fit, labels, device)
        val_m = eval_head(head, emb_va, y_va, labels, device)
        test_m = eval_head(head, emb_te, y_te, labels, device)
        path = OUT / out_name
        torch.save({"state_dict": head.state_dict(), "label_list": labels, "encoder": "REVE_frozen"}, path)
        return {"train_fit": train_m, "val": val_m, "test": test_m, "history": hist, "best_val_macro_f1": best, "splits": {"train": dtr, "val": dva, "test": dte}, "head_path": str(path.relative_to(ROOT))}

    print("training A-vig…", flush=True)
    a = run_task("a", HEAD_A_VIG_LABELS_2, lambda in_dim, n_classes: HeadAVigLinear(in_dim=in_dim, n_classes=n_classes), "head_a_vig_reve_linear.pt")
    print("training Head C…", flush=True)
    c = run_task("c", HEAD_C_LABELS, lambda in_dim, n_classes: HeadCLinear(in_dim=in_dim, n_classes=n_classes), "head_c_wake_light_reve_linear.pt")

    a_ship = (
        a["test"]["macro_f1"] >= 0.65
        and a["val"]["macro_f1"] >= 0.60
        and all(a["test"]["per_class"][n]["f1"] >= 0.50 for n in HEAD_A_VIG_LABELS_2)
        and all(a["test"]["pred_counts"].get(n, 0) > 0 for n in HEAD_A_VIG_LABELS_2)
    )
    summary = {
        "step": "reve_sleep_heads_local",
        "completed_utc": created,
        "encoder": "REVE_frozen",
        "device": "cpu",
        "sampling": sampling,
        "per_class_cap": args.per_class_cap,
        "backbone_finetuned": False,
        "n_recordings_cached": len(list(EMB_CACHE.glob("*_emb.npz"))),
        "head_a_vig": {
            "ship_candidate": bool(a_ship),
            "val_macro_f1": a["val"]["macro_f1"],
            "test_macro_f1": a["test"]["macro_f1"],
            "val_acc": a["val"]["accuracy"],
            "test_acc": a["test"]["accuracy"],
            "test_per_class_f1": {k: a["test"]["per_class"][k]["f1"] for k in HEAD_A_VIG_LABELS_2},
            "val": a["val"],
            "test": a["test"],
        },
        "head_c": {
            "ship_candidate": False,
            "task": "wake_light_2way_n1slice",
            "val_macro_f1": c["val"]["macro_f1"],
            "test_macro_f1": c["test"]["macro_f1"],
            "test_per_class_f1": {k: c["test"]["per_class"][k]["f1"] for k in HEAD_C_LABELS},
            "val": c["val"],
            "test": c["test"],
        },
        "compare_cbramod_path": "exports/train_heads_expanded_sleep_summary.json",
        "kaggle_full_encode": "kaggle_kernel_07_reve_sleep_heads (T4 preferred)",
        "encoder_notes": notes,
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "step_summary.json").write_text(json.dumps({"step": summary["step"], "completed_utc": created, "sampling": sampling, "head_a_vig": summary["head_a_vig"], "head_c": summary["head_c"]}, indent=2) + "\n")
    # docs
    lines = [
        "# REVE sleep heads — local CPU subsample",
        "",
        f"**Date (UTC):** {created}  ",
        f"**Encoder:** frozen REVE-base (offline cache)  ",
        f"**Sampling:** {sampling}",
        "",
        "## A-vig",
        "",
        f"| split | n | acc | macro-F1 |",
        f"|-------|--:|----:|---------:|",
        f"| val | {a['val']['n']} | {a['val']['accuracy']:.3f} | {a['val']['macro_f1']:.3f} |",
        f"| test | {a['test']['n']} | {a['test']['accuracy']:.3f} | {a['test']['macro_f1']:.3f} |",
        f"",
        f"**ship_candidate:** {bool(a_ship)}",
        "",
        "## Head C (wake/light)",
        "",
        f"| split | n | acc | macro-F1 |",
        f"|-------|--:|----:|---------:|",
        f"| val | {c['val']['n']} | {c['val']['accuracy']:.3f} | {c['val']['macro_f1']:.3f} |",
        f"| test | {c['test']['n']} | {c['test']['accuracy']:.3f} | {c['test']['macro_f1']:.3f} |",
        "",
        "**ship_candidate:** False (N1-slice 2-way only)",
        "",
        "## vs CBraMod (full corpus)",
        "",
        "See `exports/train_heads_expanded_sleep_summary.json`. Local REVE is **subsampled** — treat as directional; full REVE on Kaggle T4 is authoritative.",
        "",
        "## Freeze vs fine-tune",
        "",
        "No backbone fine-tune yet. Compare full REVE (Kaggle) to CBraMod before deciding.",
        "",
    ]
    DOCS.write_text("\n".join(lines))
    print("DONE", json.dumps({k: summary[k] for k in ("head_a_vig", "head_c", "sampling")}, indent=2), flush=True)


if __name__ == "__main__":
    main()

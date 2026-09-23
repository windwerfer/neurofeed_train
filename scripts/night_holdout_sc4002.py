#!/usr/bin/env python3
"""Overnight step 1: train Head A on SC4001, evaluate night holdout SC4002."""
from __future__ import annotations

import json
import hashlib
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sleep_edf import (
    PROXY_NOTE,
    STAGE_TO_HEAD_A,
    find_pilot_pairs,
    load_sleep_edf_recording,
    windows_with_head_a_labels,
)
from src.cbramod_encoder import FrozenCBraModEncoder
from src.head_a import (
    HEAD_A_BINARY_LABELS,
    HeadALinear,
    class_weights_from_y,
    labels_to_ids,
    undersample_balanced,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report

SEED = 42
WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0
PRE_SEC = 20 * 60
POST_SEC = 40 * 60
MAJORITY = 0.7
EPOCHS = 5
BATCH = 32
LR = 1e-3
VAL_FRAC = 0.2  # same-night SC4001 val split of balanced train


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_data_root() -> Path:
    cands = [
        Path("/tmp/kaggle_out3/data/sleep-edfx-pilot"),
        ROOT / "kaggle_datasets/muse-eeg-heads-cache/data/sleep-edfx-pilot",
    ]
    for r in cands:
        if r.exists() and find_pilot_pairs(r):
            return r
    raise FileNotFoundError("No Sleep-EDF pilot pairs found")


def find_weights() -> Path:
    cands = [
        Path("/tmp/kaggle_out3/models/CBraMod/pretrained_weights.pth"),
        ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth",
    ]
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("CBraMod weights not found")


def pair_for_subject(pairs: List[Tuple[Path, Path]], subject: str) -> Tuple[Path, Path]:
    for psg, hyp in pairs:
        if subject in psg.name:
            return psg, hyp
    raise FileNotFoundError(f"No pair for {subject}")


def build_night_windows(
    psg: Path,
    hyp: Path,
    out_dir: Path,
    tag: str,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = load_sleep_edf_recording(
        psg,
        hyp,
        target_sr=TARGET_SR,
        around_stage="stage 1",
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
    counts = dict(Counter(labels))
    npz_path = out_dir / f"sleep_edf_{tag}_n1slice_windows.npz"
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y,
        starts=starts,
        label_names=np.asarray(HEAD_A_BINARY_LABELS),
    )
    manifest = {
        "psg_file": psg.name,
        "hypno_file": hyp.name,
        "psg_path": str(psg),
        "hypno_path": str(hyp),
        "tag": tag,
        "slice_start_sec": rec["slice_start_sec"],
        "pre_sec": PRE_SEC,
        "post_sec": POST_SEC,
        "around_stage": "stage 1",
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "majority_frac": MAJORITY,
        "target_sr": TARGET_SR,
        "channel_proxy_note": PROXY_NOTE,
        "stage_to_label_map": {k: v for k, v in STAGE_TO_HEAD_A.items()},
        "n_windows_per_label": counts,
        "n_windows_total": int(X.shape[0]),
        "window_shape": list(X.shape),
        "npz_path": str(npz_path),
        "npz_sha256": sha256(npz_path),
        "label_list": HEAD_A_BINARY_LABELS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_attribution": {
            "Sleep-EDF": "PhysioNet ODC-By — attribute PhysioNet / Sleep-EDF Expanded",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
    }
    man_path = out_dir / f"sleep_edf_{tag}_n1slice_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{tag}] slice_start_sec={rec['slice_start_sec']} windows={X.shape} counts={counts}")
    print(f"  wrote {npz_path}")
    print(f"  wrote {man_path}")
    return {
        "X": X,
        "y": y,
        "labels": labels,
        "starts": starts,
        "manifest": manifest,
        "npz_path": npz_path,
        "man_path": man_path,
        "counts": counts,
    }


def encode_all(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(X) == 0:
        return torch.zeros((0, 200), dtype=torch.float32)
    Xt = torch.from_numpy(X.astype(np.float32))
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            emb_list.append(encoder(Xt[i : i + BATCH].to(device)).cpu())
    return torch.cat(emb_list, dim=0)


def eval_split(
    head: nn.Module,
    emb: torch.Tensor,
    y: np.ndarray,
    name: str,
) -> Dict[str, Any]:
    head.eval()
    if len(y) == 0:
        out = {
            "name": name,
            "n": 0,
            "acc": None,
            "macro_f1": None,
            "per_class": {},
            "confusion": [],
            "counts_true": {},
            "note": "empty split — no labeled windows",
        }
        print(json.dumps(out, indent=2))
        return out
    with torch.no_grad():
        pred = head(emb).argmax(dim=-1).numpy()
    acc = float((pred == y).mean())
    f1 = macro_f1(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    report = per_class_report(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    cm = confusion_matrix(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
    # strip nested macro_f1 key for cleaner JSON
    per_class = {k: v for k, v in report.items() if k != "macro_f1"}
    true_counts = {HEAD_A_BINARY_LABELS[i]: int((y == i).sum()) for i in range(2)}
    out = {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1": f1,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": list(HEAD_A_BINARY_LABELS),
        "counts_true": true_counts,
    }
    print(f"\n=== {name} ===")
    print(f"n={out['n']} acc={acc:.4f} macro_f1={f1:.4f}")
    print("per_class:", json.dumps(per_class, indent=2))
    print("confusion (rows=true, cols=pred):", cm.tolist())
    return out


def train_head(emb: torch.Tensor, y: np.ndarray, device: torch.device) -> Tuple[HeadALinear, List[dict]]:
    yt = torch.from_numpy(y)
    head = HeadALinear(in_dim=emb.shape[-1], n_classes=2).to(device)
    w = class_weights_from_y(y, n_classes=2)
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(emb, yt), batch_size=BATCH, shuffle=True)
    history = []
    for ep in range(EPOCHS):
        head.train()
        total, n = 0.0, 0
        for xb, ybatch in loader:
            xb = xb.to(device)
            ybatch = ybatch.to(device)
            opt.zero_grad()
            logits = head(xb)
            loss = crit(logits, ybatch)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(ybatch)
            n += len(ybatch)
        head.eval()
        with torch.no_grad():
            pred = head(emb.to(device)).argmax(dim=-1).cpu().numpy()
        acc = float((pred == y).mean())
        f1 = macro_f1(y.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
        row = {"epoch": ep + 1, "loss": total / max(n, 1), "train_acc": acc, "train_macro_f1": f1}
        history.append(row)
        print(row)
    return head, history


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    data_root = find_data_root()
    pairs = find_pilot_pairs(data_root)
    print("data_root", data_root)
    print("pairs", [(p.name, h.name) for p, h in pairs])

    psg1, hyp1 = pair_for_subject(pairs, "SC4001")
    psg2, hyp2 = pair_for_subject(pairs, "SC4002")
    weights = find_weights()
    print("weights", weights)

    night1 = build_night_windows(psg1, hyp1, ROOT / "exports" / "windows_sc4001", "sc4001")
    night2 = build_night_windows(psg2, hyp2, ROOT / "exports" / "windows_sc4002", "sc4002")

    X1, y1 = night1["X"], night1["y"]
    X2, y2 = night2["X"], night2["y"]

    # Balance train night
    Xb, yb = undersample_balanced(X1, y1, rng)
    print("balanced SC4001", Xb.shape, Counter(yb.tolist()))

    # Same-night val: hold out VAL_FRAC of balanced set
    n_bal = len(yb)
    perm = rng.permutation(n_bal)
    n_val = max(1, int(round(n_bal * VAL_FRAC)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, y_tr = Xb[tr_idx], yb[tr_idx]
    X_val, y_val = Xb[val_idx], yb[val_idx]
    print(f"train {X_tr.shape} val {X_val.shape}")

    device = torch.device("cpu")
    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
    notes = encoder.adapter_notes()
    encoder.to(device)

    print("encoding train...")
    emb_tr = encode_all(encoder, X_tr, device)
    print("encoding val...")
    emb_val = encode_all(encoder, X_val, device)
    print("encoding SC4002 all...")
    emb2 = encode_all(encoder, X2, device)

    # Balanced SC4002 subsample (if both classes present)
    limitation = None
    counts2 = night2["counts"]
    n_min2 = min(counts2.get("drowsy", 0), counts2.get("hypnagogic", 0)) if counts2 else 0
    if n_min2 == 0:
        limitation = (
            f"SC4002 hypnagogic or drowsy missing after labeling: {counts2}. "
            "Balanced holdout subsample skipped; all-labeled eval still reported."
        )
        print("LIMITATION:", limitation)
        X2b, y2b = np.zeros((0, *X2.shape[1:]), dtype=np.float32) if len(X2) else X2, np.zeros((0,), dtype=np.int64)
        emb2b = torch.zeros((0, 200), dtype=torch.float32)
    else:
        X2b, y2b = undersample_balanced(X2, y2, rng)
        print("balanced SC4002", X2b.shape, Counter(y2b.tolist()))
        emb2b = encode_all(encoder, X2b, device)

    print("training head...")
    head, history = train_head(emb_tr, y_tr, device)

    metrics = {
        "sc4001_train_balanced": eval_split(head, emb_tr, y_tr, "sc4001_train_balanced"),
        "sc4001_val_same_night": eval_split(head, emb_val, y_val, "sc4001_val_same_night"),
        "sc4002_all_labeled": eval_split(head, emb2, y2, "sc4002_all_labeled"),
        "sc4002_balanced": eval_split(head, emb2b, y2b, "sc4002_balanced"),
    }

    run_dir = ROOT / "exports" / "head_a_holdout"
    run_dir.mkdir(parents=True, exist_ok=True)
    head_path = run_dir / "head_a_binary_state_dict.pt"
    torch.save(head.state_dict(), head_path)

    run_manifest = {
        "task": "binary_drowsy_vs_hypnagogic_night_holdout",
        "train_night": "SC4001",
        "holdout_night": "SC4002",
        "slice_recipe": {
            "around_stage": "stage 1",
            "pre_sec": PRE_SEC,
            "post_sec": POST_SEC,
            "window_sec": WINDOW_SEC,
            "hop_sec": HOP_SEC,
            "majority_frac": MAJORITY,
        },
        "sc4001": night1["manifest"],
        "sc4002": night2["manifest"],
        "balanced_train_n_per_class": int(Counter(y_tr.tolist()).get(0, 0)),  # after val split — use min
        "balanced_train_counts": dict(Counter({HEAD_A_BINARY_LABELS[i]: int((y_tr == i).sum()) for i in range(2)})),
        "val_frac": VAL_FRAC,
        "random_seed": SEED,
        "epochs": EPOCHS,
        "batch": BATCH,
        "lr": LR,
        "encoder_name": "CBraMod",
        "encoder_weights_path": str(weights),
        "encoder_weights_sha256": notes["weights_sha256"],
        "encoder_adapter": notes,
        "head": "HeadALinear",
        "label_list": HEAD_A_BINARY_LABELS,
        "history": history,
        "metrics": metrics,
        "limitation": limitation,
        "head_path": str(head_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_attribution": {
            "Sleep-EDF": "PhysioNet ODC-By — attribute PhysioNet / Sleep-EDF Expanded",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
    }
    # fix balanced counts key to be readable
    run_manifest["balanced_train_counts"] = {
        HEAD_A_BINARY_LABELS[i]: int((y_tr == i).sum()) for i in range(2)
    }
    run_manifest["balanced_before_split_n_per_class"] = int(len(yb) // 2)

    man_path = run_dir / "run_manifest.json"
    man_path.write_text(json.dumps(run_manifest, indent=2))
    print("wrote", man_path)
    print("wrote", head_path)

    # Package windows for Kaggle dataset (both nights)
    win_pkg = ROOT / "kaggle_datasets" / "muse-eeg-heads-windows"
    win_pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy(night1["npz_path"], win_pkg / night1["npz_path"].name)
    shutil.copy(night2["npz_path"], win_pkg / night2["npz_path"].name)
    shutil.copy(night1["man_path"], win_pkg / night1["man_path"].name)
    shutil.copy(night2["man_path"], win_pkg / night2["man_path"].name)
    # Combined package manifest
    pkg_manifest = {
        "dataset": "muse-eeg-heads-windows",
        "nights": ["SC4001", "SC4002"],
        "files": [
            night1["npz_path"].name,
            night2["npz_path"].name,
            night1["man_path"].name,
            night2["man_path"].name,
        ],
        "sc4001": night1["manifest"],
        "sc4002": night2["manifest"],
        "holdout_run_summary": {
            "metrics": {
                k: {
                    "n": v.get("n"),
                    "acc": v.get("acc"),
                    "macro_f1": v.get("macro_f1"),
                    "counts_true": v.get("counts_true"),
                }
                for k, v in metrics.items()
            },
            "limitation": limitation,
            "created_utc": run_manifest["created_utc"],
        },
        "channel_proxy_note": PROXY_NOTE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (win_pkg / "manifest.json").write_text(json.dumps(pkg_manifest, indent=2))
    (win_pkg / "ATTRIBUTION.txt").write_text(
        "Sleep-EDF Expanded (PhysioNet) — ODC-By. Attribute PhysioNet and Sleep-EDF authors.\n"
        "Derived windows only; not a redistribution of full PSG nights beyond pilot terms.\n"
        "CBraMod weights used for encoder smoke are Apache-2.0 (HF weighting666/CBraMod);\n"
        "this dataset does NOT bundle CBraMod weights — windows + labels + manifest only.\n"
        "Nights: SC4001 (train) + SC4002 (holdout).\n"
        "NO LUNA / NO L-FAME / NO SEED-VIG.\n"
    )
    readme = win_pkg / "README.md"
    readme.write_text(
        "# muse-eeg-heads-windows\n\n"
        "Derived N1-slice Muse-proxy windows from Sleep-EDF Expanded pilot nights.\n\n"
        "- `sleep_edf_sc4001_n1slice_windows.npz` — train night\n"
        "- `sleep_edf_sc4002_n1slice_windows.npz` — holdout night\n"
        "- Per-night `*_manifest.json` + package `manifest.json`\n\n"
        "Recipe: around first stage 1, pre=20min, post=40min; window 2s hop 0.5s; majority 0.7.\n"
        "Labels: W→drowsy, N1→hypnagogic. Private dataset; PhysioNet ODC-By attribution required.\n"
    )
    print("updated package", win_pkg)

    # Also copy run manifest into exports for docs
    metrics_path = run_dir / "metrics_summary.json"
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "history": history,
                "limitation": limitation,
                "sc4001_counts": night1["counts"],
                "sc4002_counts": night2["counts"],
                "slice_start_sec": {
                    "SC4001": night1["manifest"]["slice_start_sec"],
                    "SC4002": night2["manifest"]["slice_start_sec"],
                },
            },
            indent=2,
        )
    )
    print("wrote", metrics_path)


if __name__ == "__main__":
    main()

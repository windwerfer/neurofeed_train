#!/usr/bin/env python3
"""Fair frozen REVE-base linear probe on HMC 5-stage sleep (paper-matched).

Matches REVE NeurIPS 2025 sleep protocol as closely as our local corpus allows:
  - 30 s non-overlapping epochs @ 256 Hz (resampled to 200 Hz inside FrozenREVEEncoder)
  - 4 EEG channels F4, C4, C3, O2 (PhysioNet HMC bipolar labels -> electrode names for pos bank)
  - 5 AASM classes: W / N1 / N2 / N3 / REM
  - Subject-wise train/val/test (deterministic on available SN###)
  - Frozen REVE-base + mean-pool tokens (paper Table 4 "Pool") + linear head
  - Metrics: balanced accuracy, Cohen kappa, macro-F1, weighted F1

Local corpus: 24 HMC nights under kaggle_datasets/.../hmc-sleep-staging/recordings
(paper uses full 151 nights / 137,243 epochs — document subsample gap).

No backbone fine-tune (ship path).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.expand_head_c_hmc import HMC_DIR, HMC_EEG, parse_hmc_scoring  # noqa: E402
from src.edf_io import read_edf_header, read_edf_signals  # noqa: E402
from src.reve_encoder import FrozenREVEEncoder  # noqa: E402
from src.sleep_edf import bandpass_fft, resample_poly  # noqa: E402

STAGE_MAP = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3,
    "Sleep stage R": 4,
    "Sleep stage REM": 4,
}
LABELS = ["W", "N1", "N2", "N3", "REM"]
ELECTRODE_NAMES = ["F4", "C4", "C3", "O2"]
EPOCH_SEC = 30.0
SOURCE_SR = 256.0
OUT_DIR = ROOT / "exports" / "reve_hmc_paper_compare"
CACHE_DIR = OUT_DIR / "emb_cache"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_subjects() -> List[str]:
    sids = []
    for p in sorted(HMC_DIR.glob("SN*.edf")):
        if "sleepscoring" in p.name:
            continue
        sid = p.stem
        if (HMC_DIR / f"{sid}_sleepscoring.txt").exists():
            sids.append(sid)
    return sids


def subject_split(sids: Sequence[str]) -> Dict[str, List[str]]:
    """Deterministic ~70/15/15 subject-wise split on sorted IDs.

    Tiny-n fallback (smoke): if n < 3, put 1 in each of train/val/test with wrap.
    If n == 3..5, ensure at least 1 per split.
    """
    s = sorted(sids)
    n = len(s)
    if n < 1:
        raise ValueError("no subjects")
    if n == 1:
        return {"train": [s[0]], "val": [s[0]], "test": [s[0]]}
    if n == 2:
        return {"train": [s[0]], "val": [s[0]], "test": [s[1]]}
    n_test = max(1, int(round(n * 0.15)))
    n_val = max(1, int(round(n * 0.15)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        n_val = 1
        n_test = n - 2
    return {
        "train": list(s[:n_train]),
        "val": list(s[n_train : n_train + n_val]),
        "test": list(s[n_train + n_val :]),
    }


def load_full_night_epochs(
    sid: str,
    *,
    bandpass: bool = True,
    max_epochs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return X (N,4,T), y (N,) int labels for one subject full night."""
    psg = HMC_DIR / f"{sid}.edf"
    scoring = HMC_DIR / f"{sid}_sleepscoring.txt"
    events = parse_hmc_scoring(scoring)
    eeg, chs, sfreq = read_edf_signals(psg, labels=HMC_EEG, max_records=None)
    if abs(sfreq - SOURCE_SR) > 1e-3:
        eeg = resample_poly(eeg.astype(np.float64), sfreq, SOURCE_SR).astype(np.float32)
        sfreq = SOURCE_SR
    else:
        eeg = eeg.astype(np.float32)
    if bandpass:
        eeg = bandpass_fft(eeg, sfreq, l_freq=0.5, h_freq=30.0).astype(np.float32)

    win = int(round(EPOCH_SEC * sfreq))
    xs: List[np.ndarray] = []
    ys: List[int] = []
    for onset, dur, ann in events:
        lab = STAGE_MAP.get(ann.strip())
        if lab is None:
            continue
        if dur < EPOCH_SEC * 0.9:
            continue
        start = int(round(onset * sfreq))
        end = start + win
        if start < 0 or end > eeg.shape[1]:
            continue
        xs.append(eeg[:, start:end])
        ys.append(lab)
        if max_epochs is not None and len(xs) >= max_epochs:
            break
    if not xs:
        raise ValueError(f"no epochs for {sid}")
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return X, y


def encode_subject(
    enc: FrozenREVEEncoder,
    sid: str,
    device: torch.device,
    *,
    batch_size: int = 8,
    max_epochs: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cap = f"_cap{max_epochs}" if max_epochs else ""
    cache_path = CACHE_DIR / f"{sid}_30s_{enc.pool}{cap}.npz"
    if cache_path.exists() and not force:
        z = np.load(cache_path, allow_pickle=False)
        return {"emb": z["emb"], "y": z["y"], "sid": sid, "cached": True, "path": str(cache_path)}

    X, y = load_full_night_epochs(sid, max_epochs=max_epochs)
    embs = []
    enc.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            emb = enc(xb).float().cpu().numpy()
            embs.append(emb)
            if (i // batch_size) % 20 == 0:
                print(f"  {sid}: {min(i+batch_size,len(X))}/{len(X)} epochs", flush=True)
    emb = np.concatenate(embs, axis=0)
    np.savez_compressed(cache_path, emb=emb.astype(np.float32), y=y.astype(np.int64))
    dt = time.time() - t0
    print(f"  {sid}: encoded {len(y)} epochs in {dt:.1f}s -> {cache_path.name}", flush=True)
    return {"emb": emb, "y": y, "sid": sid, "cached": False, "path": str(cache_path), "encode_sec": dt}


def gather_split(
    cache: Dict[str, Dict[str, Any]], sids: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    embs, ys, owners = [], [], []
    for sid in sids:
        d = cache[sid]
        embs.append(d["emb"])
        ys.append(d["y"])
        owners.extend([sid] * len(d["y"]))
    return np.concatenate(embs), np.concatenate(ys), owners


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    return {
        "n": int(len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "true_counts": {LABELS[i]: int((y_true == i).sum()) for i in range(5)},
        "pred_counts": {LABELS[i]: int((y_pred == i).sum()) for i in range(5)},
        "report": classification_report(
            y_true, y_pred, labels=list(range(5)), target_names=LABELS, output_dict=True, zero_division=0
        ),
    }


def train_linear_probe(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    best = None
    best_c = None
    best_clf = None
    for C in (0.1, 1.0, 10.0):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                C=C,
                random_state=seed,
            ),
        )
        clf.fit(Xtr, ytr)
        pred_va = clf.predict(Xva)
        bal = balanced_accuracy_score(yva, pred_va)
        if best is None or bal > best:
            best = bal
            best_c = C
            best_clf = clf
    assert best_clf is not None and best_c is not None
    pred_va = best_clf.predict(Xva)
    pred_te = best_clf.predict(Xte)
    lr = best_clf.named_steps["logisticregression"]
    return {
        "C": best_c,
        "val": metrics_dict(yva, pred_va),
        "test": metrics_dict(yte, pred_te),
        "coef_shape": list(lr.coef_.shape),
        "sklearn": "StandardScaler + LogisticRegression lbfgs max_iter=500",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-epochs-per-subject", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pool", default="mean", choices=["mean", "attention"])
    ap.add_argument("--force-reencode", action="store_true")
    ap.add_argument("--subjects", default=None, help="comma-separated SN ids (default: all local)")
    ap.add_argument("--smoke", action="store_true", help="2 subjects, 40 epochs each")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_sids = list_subjects()
    if args.subjects:
        all_sids = [s.strip() for s in args.subjects.split(",") if s.strip()]
    max_ep = args.max_epochs_per_subject
    if args.smoke:
        all_sids = all_sids[:2]
        max_ep = 40 if max_ep is None else max_ep

    splits = subject_split(all_sids)
    print("subjects", all_sids, flush=True)
    print("splits", {k: v for k, v in splits.items()}, flush=True)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA unavailable — falling back to CPU", flush=True)

    local_model = ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/reve-base"
    local_pos = ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/reve-positions"
    enc = FrozenREVEEncoder(
        source_sr=SOURCE_SR,
        channel_names=ELECTRODE_NAMES,
        pool=args.pool,
        local_model=local_model,
        local_positions=local_pos,
        device=device,
        prefer_local=True,
    )
    notes = enc.adapter_notes()
    notes["window_policy"] = (
        "Full-night 30 s non-overlapping epochs @ 256 Hz -> resample 200 Hz "
        f"(6000 samples; patch=200 overlap=20 -> ~33 patches/ch). pool={args.pool}."
    )
    notes["channel_order"] = list(ELECTRODE_NAMES)
    notes["hmc_edf_labels"] = list(HMC_EEG)
    notes["paper_compare"] = True

    cache: Dict[str, Dict[str, Any]] = {}
    encode_meta = []
    for sid in all_sids:
        print(f"encode {sid} ...", flush=True)
        d = encode_subject(
            enc, sid, device, batch_size=args.batch_size, max_epochs=max_ep, force=args.force_reencode
        )
        cache[sid] = d
        encode_meta.append(
            {
                "sid": sid,
                "n": int(len(d["y"])),
                "cached": bool(d.get("cached")),
                "label_counts": {LABELS[i]: int((d["y"] == i).sum()) for i in range(5)},
                "encode_sec": d.get("encode_sec"),
            }
        )

    Xtr, ytr, _ = gather_split(cache, splits["train"])
    Xva, yva, _ = gather_split(cache, splits["val"])
    Xte, yte, _ = gather_split(cache, splits["test"])
    print(
        f"shapes train={Xtr.shape} val={Xva.shape} test={Xte.shape} "
        f"y counts train={Counter(ytr.tolist())}",
        flush=True,
    )

    probe = train_linear_probe(Xtr, ytr, Xva, yva, Xte, yte)

    paper_ref = {
        "REVE-Base_HMC_linear_probe_pool_bal_acc": 0.647,
        "REVE-Base_HMC_linear_probe_nopool_bal_acc": 0.604,
        "REVE-Base_HMC_finetune_bal_acc": 0.7401,
        "REVE-Base_HMC_finetune_kappa": 0.6982,
        "REVE-Base_HMC_finetune_weighted_f1": 0.7638,
        "REVE-Base_ISRUC_linear_probe_pool_bal_acc": 0.697,
        "REVE-Base_ISRUC_finetune_bal_acc": 0.7819,
        "source": "docs/reve_team_sleep_benchmark.md / arXiv:2510.21585 Tables 4,13,14",
    }

    summary = {
        "step": "reve_hmc_paper_compare",
        "completed_utc": utc_now(),
        "encoder": "REVE_frozen",
        "pool": args.pool,
        "task": "HMC_5stage_30s",
        "backbone_finetuned": False,
        "n_subjects_local": len(all_sids),
        "n_subjects_paper": 151,
        "n_samples_paper": 137243,
        "subjects": all_sids,
        "splits": splits,
        "epochs_per_subject_cap": max_ep,
        "encode_meta": encode_meta,
        "n_epochs": {"train": int(len(ytr)), "val": int(len(yva)), "test": int(len(yte))},
        "linear_probe": probe,
        "paper_reference": paper_ref,
        "gap_test_bal_acc_vs_paper_pool": float(
            probe["test"]["balanced_accuracy"] - paper_ref["REVE-Base_HMC_linear_probe_pool_bal_acc"]
        ),
        "encoder_notes": notes,
        "protocol_notes": [
            "30s @ 256 Hz full-night epochs; 5 AASM classes",
            "Channels: EEG F4-M1/C4-M1/C3-M2/O2-M1 -> electrode names F4,C4,C3,O2 for reve-positions",
            "Subject-wise ~70/15/15 on available local nights (paper uses full 151 + prior baseline splits)",
            "Frozen REVE-base + mean-pool + sklearn multinomial LR (C chosen on val)",
            "Band-pass 0.5-30 Hz before epoching (TorchEEG HMC default)",
            "Not sequence-to-sequence (ISRUC FT uses 20-epoch sequences; LP Table 4 is embedding+linear)",
        ],
    }

    out_json = OUT_DIR / "metrics_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    compare = {
        "ours_test": {
            "balanced_accuracy": probe["test"]["balanced_accuracy"],
            "cohen_kappa": probe["test"]["cohen_kappa"],
            "macro_f1": probe["test"]["macro_f1"],
            "weighted_f1": probe["test"]["weighted_f1"],
            "n": probe["test"]["n"],
        },
        "paper_REVE_Base_HMC_LP_pool": {"balanced_accuracy": 0.647, "n_note": "full 151 subjects"},
        "paper_REVE_Base_HMC_LP_nopool": {"balanced_accuracy": 0.604},
        "paper_REVE_Base_HMC_FT": {
            "balanced_accuracy": 0.7401,
            "cohen_kappa": 0.6982,
            "weighted_f1": 0.7638,
        },
    }
    (OUT_DIR / "compare_table.json").write_text(json.dumps(compare, indent=2))

    md = OUT_DIR / "README.md"
    md.write_text(
        f"""# HMC 5-stage frozen REVE linear probe (paper compare)

**Completed (UTC):** {summary['completed_utc']}

## Ours (local {len(all_sids)} HMC nights, frozen + {args.pool} pool + LR)

| split | n | bal_acc | kappa | macro-F1 | weighted-F1 |
|-------|--:|-------:|------:|---------:|------------:|
| val | {probe['val']['n']} | {probe['val']['balanced_accuracy']:.3f} | {probe['val']['cohen_kappa']:.3f} | {probe['val']['macro_f1']:.3f} | {probe['val']['weighted_f1']:.3f} |
| **test** | {probe['test']['n']} | **{probe['test']['balanced_accuracy']:.3f}** | **{probe['test']['cohen_kappa']:.3f}** | **{probe['test']['macro_f1']:.3f}** | **{probe['test']['weighted_f1']:.3f}** |

## REVE team (paper)

| setting | bal_acc | kappa | weighted-F1 |
|---------|--------:|------:|------------:|
| HMC linear probe Pool (Table 4) | 0.647 ± 0.008 | — | — |
| HMC linear probe no-pool (Table 4) | 0.604 ± 0.008 | — | — |
| HMC fine-tune (Table 14) | 0.7401 ± 0.0075 | 0.6982 | 0.7638 |
| ISRUC linear probe Pool (Table 4) | 0.697 ± 0.011 | — | — |
| ISRUC fine-tune (Table 13) | 0.7819 ± 0.0078 | 0.7500 | 0.8005 |

See `docs/reve_team_sleep_benchmark.md` and `metrics_summary.json`.
"""
    )
    print(json.dumps(compare, indent=2), flush=True)
    print("wrote", out_json, md, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

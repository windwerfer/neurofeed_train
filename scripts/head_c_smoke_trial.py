#!/usr/bin/env python3
"""Head C smoke: Sleep-EDF stage_coarse (wake/light/deep/rem) on frozen CBraMod.

NOT a ship candidate — pipeline verification only. Muse-proxy deep/REM expected weak.
Uses fixed subject splits under datasets/vigilance_sleep_edf/splits/.
Builds sleep-period windows (not N1-slice) so deep/REM are present; caps per class.
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
from src.edf_io import read_edf_header
from src.head_c import (
    HEAD_C_COARSE_LABELS,
    HeadCLinear,
    cap_per_class,
    class_weights_from_y,
    labels_to_ids,
    undersample_balanced,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report
from src.sleep_edf import (
    PROXY_NOTE,
    load_sleep_edf_recording,
    sleep_period_bounds,
    windows_with_stage_coarse,
)

SEED = 42
WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0
WAKE_MARGIN_SEC = 20 * 60
EPOCHS = 8
BATCH = 32
LR = 1e-3
PATIENCE = 3
# Cap for smoke speed / class balance (per recording before split pool)
PER_CLASS_PER_REC = 200
# Train undersample after pooling
TRAIN_BALANCE = True
# Light QC (uV, matches artifact_qc_light for vigilance)
FLAT_STD = 0.1
PEAK_ABS = 350.0

OUT_DIR = ROOT / "exports" / "head_c_smoke"
DOCS_PATH = ROOT / "docs" / "head_c_smoke.md"
WIN_DIR = OUT_DIR / "windows"
STEP = "head_c_smoke_trial"
CORPUS = "vigilance_sleep_edf"


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


def load_split_map() -> Dict[str, str]:
    d = ROOT / "datasets" / CORPUS / "splits"
    out: Dict[str, str] = {}
    for sp in ("train", "val", "test"):
        for sid in json.loads((d / f"{sp}_subjects.json").read_text())["subjects"]:
            out[sid] = sp
    return out


def recording_pairs() -> List[Tuple[str, Path, Path]]:
    raw = ROOT / "datasets" / CORPUS / "raw"
    pairs = []
    for psg in sorted(raw.glob("*-PSG.edf")):
        prefix = psg.name[:6]  # SC4001
        hyps = list(psg.parent.glob(f"{prefix}*-Hypnogram.edf"))
        if not hyps:
            continue
        pairs.append((prefix, psg.resolve(), hyps[0].resolve()))
    return pairs


def qc_pass_mask(X: np.ndarray) -> np.ndarray:
    """Obvious junk only (flat / extreme peak)."""
    # X: (N, C, T)
    std = X.std(axis=-1)  # (N, C)
    peak = np.max(np.abs(X), axis=-1)
    flat = (std < FLAT_STD).any(axis=-1)
    hot = (peak > PEAK_ABS).any(axis=-1)
    return ~(flat | hot)


def build_or_load_recording(
    rid: str,
    psg: Path,
    hyp: Path,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    npz_path = WIN_DIR / f"{rid}_head_c_windows.npz"
    qc_path = WIN_DIR / f"{rid}_head_c_qc.npz"
    man_path = WIN_DIR / f"{rid}_head_c_manifest.json"

    if npz_path.exists() and qc_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        qc = np.load(qc_path, allow_pickle=True)
        return {
            "recording_id": rid,
            "subject_id": rid[:5],
            "X": data["X"].astype(np.float32),
            "y": data["y"].astype(np.int64),
            "stage_raw": data["stage_raw"],
            "stage_coarse": data["stage_coarse"],
            "starts": data["starts"].astype(np.int64),
            "qc_pass": qc["qc_pass"].astype(bool),
            "npz_path": str(npz_path.relative_to(ROOT)),
            "qc_path": str(qc_path.relative_to(ROOT)),
            "manifest": json.loads(man_path.read_text()) if man_path.exists() else {},
            "cached": True,
        }

    info = read_edf_header(psg)
    rec_end = float(info.n_records) * float(info.record_duration)
    from src.edf_io import read_edf_annotations

    events = read_edf_annotations(hyp)
    t0, t1 = sleep_period_bounds(events, wake_margin_sec=WAKE_MARGIN_SEC, recording_end_sec=rec_end)
    duration = max(60.0, t1 - t0)
    print(f"  {rid}: sleep-period [{t0:.0f}, {t1:.0f}] sec ({duration/3600:.2f} h)", flush=True)

    rec = load_sleep_edf_recording(
        psg,
        hyp,
        target_sr=TARGET_SR,
        start_sec=t0,
        duration_sec=duration,
    )
    X, stage_raws, stage_coarses, keep = windows_with_stage_coarse(
        rec["data"],
        rec["stages"],
        sfreq=TARGET_SR,
        window_sec=WINDOW_SEC,
        hop_sec=HOP_SEC,
        drop_unknown=True,
    )
    hop = int(round(HOP_SEC * TARGET_SR))
    starts = np.asarray([k * hop for k in keep], dtype=np.int64)
    # Keep only preferred coarse classes
    mask = np.asarray([c in HEAD_C_COARSE_LABELS for c in stage_coarses], dtype=bool)
    X = X[mask]
    stage_raws = [s for s, m in zip(stage_raws, mask) if m]
    stage_coarses = [c for c, m in zip(stage_coarses, mask) if m]
    starts = starts[mask]
    y = labels_to_ids(stage_coarses, HEAD_C_COARSE_LABELS)

    qc = qc_pass_mask(X)
    X_qc, y_qc, st_qc = X[qc], y[qc], starts[qc]
    raw_qc = np.asarray(stage_raws, dtype=object)[qc]
    coarse_qc = np.asarray(stage_coarses, dtype=object)[qc]

    # Cap per class for smoke (track indices so stage_raw stays aligned)
    idxs: List[np.ndarray] = []
    for c in range(len(HEAD_C_COARSE_LABELS)):
        cand = np.where(y_qc == c)[0]
        n = min(PER_CLASS_PER_REC, len(cand))
        if n == 0:
            continue
        idxs.append(rng.choice(cand, size=n, replace=False))
    if idxs:
        pick = np.concatenate(idxs)
        rng.shuffle(pick)
        X_cap = X_qc[pick]
        y_cap = y_qc[pick]
        st_cap = st_qc[pick]
        raw_cap = raw_qc[pick]
        coarse_cap = coarse_qc[pick]
    else:
        X_cap = X_qc[:0]
        y_cap = y_qc[:0]
        st_cap = st_qc[:0]
        raw_cap = raw_qc[:0]
        coarse_cap = coarse_qc[:0]

    WIN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X=X_cap.astype(np.float32),
        y=y_cap.astype(np.int64),
        starts=st_cap.astype(np.int64),
        label_names=np.asarray(HEAD_C_COARSE_LABELS),
        stage_raw=raw_cap,
        stage_coarse=coarse_cap,
    )
    np.savez_compressed(
        qc_path,
        qc_pass=np.ones(len(y_cap), dtype=bool),  # already filtered
        qc_reason=np.asarray([""] * len(y_cap), dtype="<U32"),
        n_total_before_qc=int(len(y)),
        n_pass_qc=int(qc.sum()),
        n_after_cap=int(len(y_cap)),
        source_npz=str(npz_path.relative_to(ROOT)),
        starts=st_cap.astype(np.int64),
    )
    counts = {HEAD_C_COARSE_LABELS[i]: int((y_cap == i).sum()) for i in range(4)}
    man = {
        "recording_id": rid,
        "subject_id": rid[:5],
        "psg": psg.name,
        "hypno": hyp.name,
        "slice_start_sec": float(t0),
        "slice_end_sec": float(t1),
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "target_sr": TARGET_SR,
        "label_list": HEAD_C_COARSE_LABELS,
        "n_windows_before_qc": int(len(y)),
        "n_qc_pass": int(qc.sum()),
        "n_after_cap": int(len(y_cap)),
        "per_class_cap": PER_CLASS_PER_REC,
        "counts_after_cap": counts,
        "channel_proxy_note": PROXY_NOTE,
        "npz_sha256": sha256(npz_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    man_path.write_text(json.dumps(man, indent=2) + "\n")
    print(f"  {rid}: counts={counts} qc_pass={int(qc.sum())}/{len(y)}", flush=True)
    return {
        "recording_id": rid,
        "subject_id": rid[:5],
        "X": X_cap.astype(np.float32),
        "y": y_cap.astype(np.int64),
        "stage_raw": raw_cap,
        "stage_coarse": coarse_cap,
        "starts": st_cap.astype(np.int64),
        "qc_pass": np.ones(len(y_cap), dtype=bool),
        "npz_path": str(npz_path.relative_to(ROOT)),
        "qc_path": str(qc_path.relative_to(ROOT)),
        "manifest": man,
        "cached": False,
    }


def pool_split(
    packs: List[Dict[str, Any]], split: str, split_map: Dict[str, str]
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    Xs, ys = [], []
    detail = {"recordings": [], "counts": Counter()}
    for p in packs:
        if split_map.get(p["subject_id"]) != split:
            continue
        mask = p["qc_pass"]
        X, y = p["X"][mask], p["y"][mask]
        if len(y) == 0:
            continue
        Xs.append(X)
        ys.append(y)
        c = Counter({HEAD_C_COARSE_LABELS[i]: int((y == i).sum()) for i in range(4)})
        detail["recordings"].append(
            {
                "recording_id": p["recording_id"],
                "subject_id": p["subject_id"],
                "counts": dict(c),
                "n": int(len(y)),
            }
        )
        detail["counts"].update(c)
    if not Xs:
        raise RuntimeError(f"no windows for split={split}")
    detail["counts"] = dict(detail["counts"])
    return np.concatenate(Xs), np.concatenate(ys), detail


def encode(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    emb_list = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.from_numpy(X[i : i + BATCH]).to(device)
            emb_list.append(encoder(xb).cpu())
    return torch.cat(emb_list, dim=0)


def eval_split(
    head: nn.Module,
    emb: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    head.eval()
    with torch.no_grad():
        logits = head(emb.to(device)).cpu()
        pred = logits.argmax(dim=-1).numpy()
    report = per_class_report(y.tolist(), pred.tolist(), HEAD_C_COARSE_LABELS)
    cm = confusion_matrix(y.tolist(), pred.tolist(), HEAD_C_COARSE_LABELS)
    acc = float((pred == y).mean()) if len(y) else 0.0
    return {
        "n": int(len(y)),
        "accuracy": acc,
        "macro_f1": float(report["macro_f1"]["f1"]),
        "per_class": {
            k: v for k, v in report.items() if k != "macro_f1"
        },
        "confusion_matrix": cm.tolist(),
        "label_list": HEAD_C_COARSE_LABELS,
        "pred_counts": dict(Counter(pred.tolist())),
        "true_counts": dict(Counter(y.tolist())),
    }


def write_docs(metrics: Dict[str, Any]) -> None:
    val = metrics["val"]
    test = metrics["test"]
    lines = [
        "# Head C smoke trial",
        "",
        f"**Step:** `{STEP}`  ",
        f"**Date (UTC):** {metrics['created_utc']}  ",
        f"**ship_candidate:** **{metrics['ship_candidate']}** — pipeline check only; not a product head.",
        "",
        "## Setup",
        "",
        "- Encoder: **frozen CBraMod** (same path as Head A)",
        "- Head: `HeadCLinear` (dropout + linear)",
        f"- Classes: `stage_coarse` **{', '.join(HEAD_C_COARSE_LABELS)}** (unknown dropped)",
        "- Windows: Sleep-EDF **sleep-period** slices (wake margin), not N1-slice — so deep/REM exist",
        f"- Caps: ≤{PER_CLASS_PER_REC} windows/class/recording after light QC",
        "- Splits: fixed subject JSON in `datasets/vigilance_sleep_edf/splits/`",
        f"- Channel note: {PROXY_NOTE}",
        "",
        "## Metrics",
        "",
        f"| Split | n | accuracy | macro-F1 |",
        f"|-------|--:|---------:|---------:|",
        f"| train (balanced fit set) | {metrics['train_fit']['n']} | {metrics['train_fit']['accuracy']:.3f} | {metrics['train_fit']['macro_f1']:.3f} |",
        f"| val | {val['n']} | {val['accuracy']:.3f} | {val['macro_f1']:.3f} |",
        f"| test | {test['n']} | {test['accuracy']:.3f} | {test['macro_f1']:.3f} |",
        "",
        "### Val per-class",
        "",
        "| class | precision | recall | f1 | support |",
        "|-------|----------:|-------:|---:|--------:|",
    ]
    for name in HEAD_C_COARSE_LABELS:
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
    for name in HEAD_C_COARSE_LABELS:
        r = test["per_class"][name]
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {int(r['support'])} |"
        )
    lines += [
        "",
        "## Muse-proxy weakness (deep / REM)",
        "",
        "Sleep-EDF Fpz-Cz / Pz-Oz duplicated into Muse AF7/AF8/TP9/TP10 is a **coarse frontal/posterior proxy**.",
        "Deep sleep (slow waves) and REM (occipital/EOG-ish patterns) transfer poorly to true Muse montage;",
        "treat deep/REM F1 as diagnostic of pipeline wiring, not product quality.",
        "",
        "## Artifacts",
        "",
        f"- Windows: `{WIN_DIR.relative_to(ROOT)}/`",
        f"- Weights: `exports/head_c_smoke/head_c_coarse_linear.pt`",
        f"- Metrics: `exports/head_c_smoke/metrics_summary.json`",
        f"- Manifest: `exports/head_c_smoke/run_manifest.json`",
        f"- Script: `scripts/head_c_smoke_trial.py`",
        f"- Module: `src/head_c.py`",
        "",
    ]
    DOCS_PATH.write_text("\n".join(lines))


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WIN_DIR.mkdir(parents=True, exist_ok=True)

    split_map = load_split_map()
    pairs = recording_pairs()
    if not pairs:
        raise FileNotFoundError("No Sleep-EDF PSG/Hypnogram pairs under datasets/vigilance_sleep_edf/raw")

    print(f"recordings={len(pairs)} splits={split_map}", flush=True)
    packs = []
    for rid, psg, hyp in pairs:
        packs.append(build_or_load_recording(rid, psg, hyp, rng))

    X_train, y_train, train_detail = pool_split(packs, "train", split_map)
    X_val, y_val, val_detail = pool_split(packs, "val", split_map)
    X_test, y_test, test_detail = pool_split(packs, "test", split_map)
    print("train counts", train_detail["counts"], flush=True)
    print("val counts", val_detail["counts"], flush=True)
    print("test counts", test_detail["counts"], flush=True)

    if TRAIN_BALANCE:
        X_fit, y_fit = undersample_balanced(X_train, y_train, rng)
    else:
        X_fit, y_fit = X_train, y_train
    print("fit balanced", Counter(y_fit.tolist()), flush=True)

    weights = find_weights()
    device = torch.device("cpu")
    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
    encoder.to(device)
    notes = encoder.adapter_notes()
    print("encoding…", flush=True)
    emb_fit = encode(encoder, X_fit, device)
    emb_val = encode(encoder, X_val, device)
    emb_test = encode(encoder, X_test, device)
    print("emb", tuple(emb_fit.shape), flush=True)

    head = HeadCLinear(in_dim=emb_fit.shape[-1], n_classes=len(HEAD_C_COARSE_LABELS)).to(device)
    w = class_weights_from_y(y_fit, n_classes=len(HEAD_C_COARSE_LABELS))
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(emb_fit, torch.from_numpy(y_fit)),
        batch_size=BATCH,
        shuffle=True,
    )

    history = []
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
        train_m = eval_split(head, emb_fit, y_fit, device)
        val_m = eval_split(head, emb_val, y_val, device)
        row = {
            "epoch": ep + 1,
            "loss": total / max(n, 1),
            "train_macro_f1": train_m["macro_f1"],
            "val_macro_f1": val_m["macro_f1"],
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

    train_fit_m = eval_split(head, emb_fit, y_fit, device)
    val_m = eval_split(head, emb_val, y_val, device)
    test_m = eval_split(head, emb_test, y_test, device)

    head_path = OUT_DIR / "head_c_coarse_linear.pt"
    torch.save(head.state_dict(), head_path)

    created = datetime.now(timezone.utc).isoformat()
    metrics = {
        "step": STEP,
        "created_utc": created,
        "ship_candidate": False,
        "task": "stage_coarse_wake_light_deep_rem",
        "label_list": HEAD_C_COARSE_LABELS,
        "train_fit": train_fit_m,
        "val": val_m,
        "test": test_m,
        "history": history,
        "best_val_macro_f1": best_val,
        "muse_proxy_note": PROXY_NOTE,
        "muse_proxy_weakness": (
            "Deep/REM on Sleep-EDF→Muse duplicate proxy is diagnostic only; "
            "do not treat as product quality."
        ),
        "splits": {
            "train": train_detail,
            "val": val_detail,
            "test": test_detail,
        },
        "encoder_weights_sha256": notes.get("weights_sha256"),
        "encoder_adapter": notes,
        "head_path": str(head_path.relative_to(ROOT)),
    }
    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")

    manifest = {
        "step": STEP,
        "created_utc": created,
        "ship_candidate": False,
        "seed": SEED,
        "epochs_max": EPOCHS,
        "patience": PATIENCE,
        "lr": LR,
        "batch": BATCH,
        "per_class_per_rec": PER_CLASS_PER_REC,
        "wake_margin_sec": WAKE_MARGIN_SEC,
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "target_sr": TARGET_SR,
        "split_policy": "datasets/vigilance_sleep_edf/splits/",
        "recordings": [p["recording_id"] for p in packs],
        "windows_dir": str(WIN_DIR.relative_to(ROOT)),
        "metrics_path": "exports/head_c_smoke/metrics_summary.json",
        "docs_path": "docs/head_c_smoke.md",
        "license_attribution": {
            "Sleep-EDF": "PhysioNet ODC-By — attribute PhysioNet / Sleep-EDF Expanded",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
    }
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT_DIR / "step_summary.json").write_text(
        json.dumps(
            {
                "step": STEP,
                "completed_utc": created,
                "ship_candidate": False,
                "val_macro_f1": val_m["macro_f1"],
                "test_macro_f1": test_m["macro_f1"],
                "val_per_class_f1": {
                    k: val_m["per_class"][k]["f1"] for k in HEAD_C_COARSE_LABELS
                },
                "test_per_class_f1": {
                    k: test_m["per_class"][k]["f1"] for k in HEAD_C_COARSE_LABELS
                },
            },
            indent=2,
        )
        + "\n"
    )
    write_docs(metrics)
    print(
        f"DONE val_macro_f1={val_m['macro_f1']:.3f} test_macro_f1={test_m['macro_f1']:.3f} "
        f"ship_candidate=false",
        flush=True,
    )


if __name__ == "__main__":
    main()

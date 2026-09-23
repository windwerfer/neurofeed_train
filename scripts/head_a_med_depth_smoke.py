#!/usr/bin/env python3
"""Head A-med depth CPU smoke: frozen CBraMod + HeadADepth on ds001787 muse4.

Labels (Brandmeyer & Delorme probe Q1, 0–3 meditation depth):
  Q1 ≤ 1 → meditation_depth_low
  Q1 ≥ 2 → meditation_depth_high
  (integer scale → no mid band; incomplete Q1 dropped)

Data: existing local muse4 windows under datasets/attention_ds001787
(probe_ids remapped via events+behavioral logs). Prefer **one window per probe**.
Montage: muse4 only — never mix crown8.
No Kaggle. No REVE.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ds001787_attention_ingest import (  # noqa: E402
    merge_log_ratings,
    parse_behavioral_log,
    parse_events_probes,
)
from src.cbramod_encoder import FrozenCBraModEncoder  # noqa: E402
from src.heads.base import class_weights_from_y, undersample_balanced  # noqa: E402
from src.heads.head_a_depth import (  # noqa: E402
    HEAD_A_DEPTH_LABELS,
    Q1_HIGH_MIN,
    Q1_LOW_MAX,
    HeadADepthLinear,
    q1_to_depth_label,
)
from src.metrics import confusion_matrix, macro_f1, per_class_report  # noqa: E402

SEED = 42
EPOCHS = 8
BATCH = 32
LR = 1e-3
VAL_FRAC = 0.15
CHANCE_MACRO_F1 = 0.5
SHIP_F1_MIN = 0.55  # clear above chance on holdout balanced
VERSION_NOTE = "med_depth_q1_le1_ge2_onewin_2026-09-08"

WIN_DIR = ROOT / "datasets/attention_ds001787/windows"
OUT_DIR = ROOT / "exports" / "head_a_med_depth_smoke"
DOC_PATH = ROOT / "docs" / "head_a_med_depth_smoke.md"
ALT_DOC = ROOT / "docs" / "head_a_med_alt_label_pairs.md"
MULTI_DOC = ROOT / "docs" / "head_a_multihead.md"

# Best local balance for subject-holdout eval (both low+high probes).
HOLDOUT_TAG = "sub020_ses01"
HOLDOUT_SUBJECT = "sub-020"

DATASET_STRING = (
    "ds001787 (64-ch BioSemi, professional) — muse4 proxy channels "
    "AF7/AF8/TP9/TP10 extracted (TP9/TP10←P9/P10)"
)
DATASET_DOI = "doi:10.18112/openneuro.ds001787.v1.1.1"
LICENSE_SPDX = "CC0-1.0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_weights() -> Path:
    p = ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def bangkok_now() -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=7)).strftime(
        "%Y-%m-%d %H:%M:%S ICT"
    )


def load_probes_for_manifest(man: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    events = Path(man["events"])
    if not events.exists():
        raise FileNotFoundError(events)
    log_path = Path(man["log_file"]) if man.get("log_file") else None
    ep = parse_events_probes(events)
    lp = parse_behavioral_log(log_path) if log_path and log_path.exists() else []
    return merge_log_ratings(ep, lp)


def one_window_per_probe(
    X: np.ndarray,
    starts: np.ndarray,
    probe_ids: np.ndarray,
    q1_by_pid: Dict[int, Optional[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Keep one window per probe (median start within probe), map Q1→depth y."""
    Xs: List[np.ndarray] = []
    ys: List[int] = []
    st: List[int] = []
    pids: List[int] = []
    q1s: List[int] = []
    drops = Counter()
    for pid in sorted(set(int(p) for p in probe_ids)):
        q1 = q1_by_pid.get(pid)
        lab = q1_to_depth_label(q1)
        if lab is None:
            drops["missing_or_out_of_rule"] += 1
            continue
        mask = probe_ids == pid
        idxs = np.where(mask)[0]
        # pick window closest to median start (center of pre-Q1 cluster)
        med = float(np.median(starts[idxs]))
        best = idxs[int(np.argmin(np.abs(starts[idxs].astype(float) - med)))]
        Xs.append(X[best])
        ys.append(HEAD_A_DEPTH_LABELS.index(lab))
        st.append(int(starts[best]))
        pids.append(pid)
        q1s.append(int(q1))
        drops[lab] += 1
    if not Xs:
        return (
            np.zeros((0, X.shape[1], X.shape[2]), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            dict(drops),
        )
    return (
        np.stack(Xs, axis=0).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(st, dtype=np.int64),
        np.asarray(q1s, dtype=np.int64),
        dict(drops),
    )


def load_depth_pack(tag: str) -> Dict[str, Any]:
    man_path = WIN_DIR / f"{tag}_manifest.json"
    npz_path = WIN_DIR / f"{tag}_windows.npz"
    man = json.loads(man_path.read_text())
    data = np.load(npz_path)
    X = data["X"].astype(np.float32)
    starts = data["starts"].astype(np.int64)
    probe_ids = data["probe_ids"].astype(np.int64)
    assert list(data["channels"]) == ["AF7", "AF8", "TP9", "TP10"], data["channels"]
    assert man.get("montage_id", "muse4") == "muse4"

    probes, merge_mode = load_probes_for_manifest(man)
    q1_by_pid = {i: probes[i].get("q1") for i in range(len(probes))}
    X1, y1, st1, q1s, drop_stats = one_window_per_probe(X, starts, probe_ids, q1_by_pid)
    counts = {HEAD_A_DEPTH_LABELS[i]: int((y1 == i).sum()) for i in range(2)}
    q1_hist = dict(Counter(int(q) for q in q1s))
    return {
        "tag": tag,
        "subject": man.get("subject"),
        "group": man.get("group"),
        "X": X1,
        "y": y1,
        "starts": st1,
        "q1": q1s,
        "counts": counts,
        "q1_hist": q1_hist,
        "n_probes_kept": int(len(y1)),
        "n_windows_raw": int(len(probe_ids)),
        "merge_mode": merge_mode,
        "drop_stats": drop_stats,
        "npz_path": str(npz_path.relative_to(ROOT)),
        "npz_sha256": sha256(npz_path),
        "manifest_path": str(man_path.relative_to(ROOT)),
        "montage": "muse4",
        "channels": ["AF7", "AF8", "TP9", "TP10"],
    }


def balance_train(
    packs: List[Dict[str, Any]],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    Xs, ys = [], []
    detail: Dict[str, Any] = {}
    for p in packs:
        present = sorted(set(int(c) for c in np.unique(p["y"]))) if len(p["y"]) else []
        if len(present) < 2:
            detail[p["tag"]] = {
                "skipped": True,
                "reason": "single_class_or_empty",
                "counts": p["counts"],
            }
            continue
        Xb, yb = undersample_balanced(p["X"], p["y"], rng)
        Xs.append(Xb)
        ys.append(yb)
        detail[p["tag"]] = {
            "skipped": False,
            "group": p.get("group"),
            "n_total": int(len(yb)),
            "per_class": {HEAD_A_DEPTH_LABELS[i]: int((yb == i).sum()) for i in present},
            "counts_full": p["counts"],
            "q1_hist": p["q1_hist"],
        }
    if not Xs:
        raise RuntimeError("no train subjects with both depth classes")
    return np.concatenate(Xs), np.concatenate(ys), detail


def encode_all(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> torch.Tensor:
    if len(X) == 0:
        return torch.zeros((0, 200), dtype=torch.float32)
    Xt = torch.from_numpy(X.astype(np.float32))
    emb_list = []
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            emb_list.append(encoder(Xt[i : i + BATCH].to(device)).cpu())
    return torch.cat(emb_list, dim=0)


def eval_split(head: nn.Module, emb: torch.Tensor, y: np.ndarray, name: str) -> Dict[str, Any]:
    head.eval()
    if len(y) == 0:
        return {"name": name, "n": 0, "acc": None, "macro_f1": None, "note": "empty"}
    with torch.no_grad():
        pred = head(emb).argmax(dim=-1).numpy()
    acc = float((pred == y).mean())
    f1 = macro_f1(y.tolist(), pred.tolist(), HEAD_A_DEPTH_LABELS)
    report = per_class_report(y.tolist(), pred.tolist(), HEAD_A_DEPTH_LABELS)
    per_class = {k: v for k, v in report.items() if k != "macro_f1"}
    cm = confusion_matrix(y.tolist(), pred.tolist(), HEAD_A_DEPTH_LABELS)
    return {
        "name": name,
        "n": int(len(y)),
        "acc": acc,
        "macro_f1": f1,
        "delta_vs_chance": float(f1 - CHANCE_MACRO_F1) if f1 is not None else None,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "confusion_labels": list(HEAD_A_DEPTH_LABELS),
        "counts_true": {HEAD_A_DEPTH_LABELS[i]: int((y == i).sum()) for i in range(2)},
        "pred_counts": {HEAD_A_DEPTH_LABELS[i]: int((pred == i).sum()) for i in range(2)},
    }


def train_head(
    emb: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
) -> Tuple[HeadADepthLinear, List[Dict[str, Any]], Dict[str, Any]]:
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(round(VAL_FRAC * n)))
    va_idx, tr_idx = idx[:n_val], idx[n_val:]
    emb_tr, y_tr = emb[tr_idx], y[tr_idx]
    emb_va, y_va = emb[va_idx], y[va_idx]

    head = HeadADepthLinear(in_dim=emb.shape[-1]).to(device)
    w = class_weights_from_y(y_tr, n_classes=2).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(emb_tr, torch.from_numpy(y_tr)),
        batch_size=BATCH,
        shuffle=True,
    )

    history: List[Dict[str, Any]] = []
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    for ep in range(EPOCHS):
        head.train()
        total, n_seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n_seen += len(yb)
        head.eval()
        with torch.no_grad():
            pred_tr = head(emb_tr.to(device)).argmax(dim=-1).cpu().numpy()
            pred_va = head(emb_va.to(device)).argmax(dim=-1).cpu().numpy()
        row = {
            "epoch": ep + 1,
            "loss": total / max(n_seen, 1),
            "train_acc": float((pred_tr == y_tr).mean()),
            "train_macro_f1": macro_f1(
                y_tr.tolist(), pred_tr.tolist(), HEAD_A_DEPTH_LABELS
            ),
            "val_acc": float((pred_va == y_va).mean()),
            "val_macro_f1": macro_f1(
                y_va.tolist(), pred_va.tolist(), HEAD_A_DEPTH_LABELS
            ),
        }
        history.append(row)
        print(row, flush=True)
        if row["val_macro_f1"] >= best_val_f1:
            best_val_f1 = float(row["val_macro_f1"])
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    pick = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
    }
    return head, history, pick


def loso_lite(
    packs: List[Dict[str, Any]],
    encoder: FrozenCBraModEncoder,
    device: torch.device,
    rng: np.random.Generator,
    max_folds: int = 5,
) -> Dict[str, Any]:
    """Quick LOSO-lite: up to max_folds subjects that have both classes as holdout."""
    eligible = [
        p
        for p in packs
        if p["counts"].get("meditation_depth_low", 0) > 0
        and p["counts"].get("meditation_depth_high", 0) > 0
    ]
    # Prefer larger holdouts first for stable F1
    eligible = sorted(
        eligible,
        key=lambda p: min(p["counts"].values()),
        reverse=True,
    )[:max_folds]
    folds = []
    for hold in eligible:
        train_packs = [p for p in packs if p["tag"] != hold["tag"]]
        try:
            Xtr, ytr, _ = balance_train(train_packs, rng)
        except RuntimeError:
            continue
        emb_tr = encode_all(encoder, Xtr, device)
        head, _, pick = train_head(emb_tr, ytr, device, np.random.default_rng(SEED + 7))
        # holdout full + balanced
        emb_h = encode_all(encoder, hold["X"], device)
        full = eval_split(head.cpu(), emb_h, hold["y"], "holdout_full")
        Xb, yb = undersample_balanced(hold["X"], hold["y"], np.random.default_rng(SEED))
        emb_b = encode_all(encoder, Xb, device)
        bal = eval_split(head, emb_b, yb, "holdout_balanced")
        folds.append(
            {
                "holdout": hold["tag"],
                "subject": hold["subject"],
                "n_train_windows": int(len(ytr)),
                "best_val_macro_f1": pick["best_val_macro_f1"],
                "holdout_full_macro_f1": full["macro_f1"],
                "holdout_balanced_macro_f1": bal["macro_f1"],
                "holdout_counts": hold["counts"],
            }
        )
        print(f"[loso-lite] {hold['tag']}: full_f1={full['macro_f1']:.3f} bal_f1={bal['macro_f1']:.3f}", flush=True)
    if not folds:
        return {"n_folds": 0, "folds": []}
    mean_bal = float(np.mean([f["holdout_balanced_macro_f1"] for f in folds]))
    mean_full = float(np.mean([f["holdout_full_macro_f1"] for f in folds]))
    return {
        "n_folds": len(folds),
        "mean_holdout_full_macro_f1": mean_full,
        "mean_holdout_balanced_macro_f1": mean_bal,
        "delta_vs_chance_balanced": mean_bal - CHANCE_MACRO_F1,
        "folds": folds,
    }


def write_doc(summary: Dict[str, Any]) -> None:
    h = summary["holdout_metrics"]
    loso = summary.get("loso_lite", {})
    ship = summary["ship_candidate"]
    lines = [
        "# Head A-med depth smoke (CBraMod, CPU)",
        "",
        f"**Completed (Asia/Bangkok):** {summary['completed_bangkok']}",
        f"**UTC:** {summary['completed_utc']}",
        f"**Version:** `{VERSION_NOTE}`",
        "",
        "## Goal",
        "",
        "Frozen **CBraMod** + **HeadADepth** (`meditation_depth_low` vs `meditation_depth_high`) "
        "local CPU smoke on OpenNeuro **ds001787 (64-ch BioSemi, professional)** muse4 proxy "
        "(AF7/AF8/TP9/TP10 extracted; TP9/TP10←P9/P10). Subject holdout + LOSO-lite.",
        "",
        "## Dataset",
        "",
        f"- **Corpus:** {DATASET_STRING}",
        f"- **DOI / license:** `{DATASET_DOI}` / `{LICENSE_SPDX}`",
        "- **Source windows:** `datasets/attention_ds001787/windows/` (existing muse4 attention pack)",
        "- **Not used:** crown8 pack; Kaggle; REVE",
        "",
        "## Mapping rule (exact Q1 thresholds)",
        "",
        "Brandmeyer & Delorme probe **Q1** = depth of meditation / concentration on **0–3** scale.",
        "",
        "| Q1 | A-depth label |",
        "|----|---------------|",
        f"| **≤ {Q1_LOW_MAX}** (0 or 1) | `meditation_depth_low` |",
        f"| **≥ {Q1_HIGH_MIN}** (2 or 3) | `meditation_depth_high` |",
        "| missing Q1 | **drop** |",
        "",
        "Integer 0–3 scale → **no mid band** between 1 and 2. Extremes-only (0 vs 3) was too sparse "
        "locally (only 3 subjects with both) — not used for this smoke.",
        "",
        "**Window policy:** **one window per probe** (median-start window inside the existing "
        "~10 s pre-Q1 epoch). Overlapping hop-0.5 s windows from the attention ingest are collapsed.",
        "",
        "## Split",
        "",
        f"- Train subjects (both classes after remap): {', '.join(summary['train_subjects'])} "
        f"(n={summary['n_train_subjects']})",
        f"- Holdout: `{summary['holdout_subject']}` / `{summary['holdout_tag']}`",
        "- Per-subject `undersample_balanced` then concat; internal 15% val for early pick.",
        "",
        "## Metrics (headline)",
        "",
        "| Split | n | accuracy | macro-F1 | Δ vs chance (0.5) |",
        "|-------|--:|---------:|---------:|------------------:|",
    ]
    for key in ("train_balanced", "val", "holdout_full", "holdout_balanced"):
        m = h[key]
        f1 = m.get("macro_f1")
        d = "" if f1 is None else f"{f1 - CHANCE_MACRO_F1:+.3f}"
        f1s = "" if f1 is None else f"{f1:.3f}"
        acc = "" if m.get("acc") is None else f"{m['acc']:.3f}"
        lines.append(f"| {key} | {m['n']} | {acc} | {f1s} | {d} |")
    lines += [
        "",
        f"**ship_candidate:** `{ship}` "
        f"(criterion: holdout balanced macro-F1 ≥ {SHIP_F1_MIN} and both classes predicted)",
        "",
        "### Holdout confusion (balanced)",
        "",
        "```",
        f"labels: {h['holdout_balanced']['confusion_labels']}",
        f"matrix: {h['holdout_balanced']['confusion']}",
        f"pred_counts: {h['holdout_balanced'].get('pred_counts')}",
        "```",
        "",
        "## LOSO-lite",
        "",
        f"- Folds: **{loso.get('n_folds', 0)}** (subjects with both classes; largest minority first)",
        f"- Mean holdout full macro-F1: **{loso.get('mean_holdout_full_macro_f1', float('nan')):.3f}**",
        f"- Mean holdout balanced macro-F1: **{loso.get('mean_holdout_balanced_macro_f1', float('nan')):.3f}** "
        f"(Δ vs chance {loso.get('delta_vs_chance_balanced', float('nan')):+.3f})",
        "",
        "## Counts",
        "",
        f"- Subjects loaded: **{summary['n_subjects_loaded']}**",
        f"- Train subjects used: **{summary['n_train_subjects']}**",
        f"- Holdout probes/windows (1/probe): **{summary['holdout_n_windows']}** "
        f"(low={summary['holdout_counts']['meditation_depth_low']}, "
        f"high={summary['holdout_counts']['meditation_depth_high']})",
        f"- Train balanced windows: **{summary['n_train_balanced']}**",
        "",
        "## Provenance",
        "",
        f"- Encoder: CBraMod Apache-2.0 `pretrained_weights.pth` (local cache)",
        f"- Head: `HeadADepthLinear` (`src/heads/head_a_depth.py`)",
        f"- Script: `scripts/head_a_med_depth_smoke.py`",
        f"- Exports: `exports/head_a_med_depth_smoke/`",
        f"- Montage: **muse4 only**",
        "",
        "## Caveats",
        "",
        "- Q1 is subjective probe noise (same family as MW LOSO ~0.36).",
        "- Domain gap: CBraMod TUEG → 4-ch Muse proxy from BioSemi.",
        "- One window/probe reduces n dramatically vs overlapping attention pack.",
        "- Rest↔med on ds003816 remains a separate (weaker UX) operationalization.",
        "",
        "## Takeaway",
        "",
        summary["takeaway"],
        "",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n")
    print(f"[doc] wrote {DOC_PATH}", flush=True)


def patch_alt_and_multi(summary: Dict[str, Any]) -> None:
    """Briefly note depth smoke result + preferred A-med operationalization."""
    h = summary["holdout_metrics"]["holdout_balanced"]
    f1 = h.get("macro_f1")
    ship = summary["ship_candidate"]
    note = (
        f"\n\n---\n\n## Depth smoke result (2026-09-08)\n\n"
        f"Ran `scripts/head_a_med_depth_smoke.py` on **{DATASET_STRING}** muse4, "
        f"Q1≤{Q1_LOW_MAX}→low / Q1≥{Q1_HIGH_MIN}→high, **1 window/probe**, "
        f"holdout `{summary['holdout_subject']}`.\n\n"
        f"- Holdout balanced macro-F1: **{f1:.3f}** (chance 0.50, Δ {f1 - CHANCE_MACRO_F1:+.3f})\n"
        f"- LOSO-lite mean balanced macro-F1: "
        f"**{summary['loso_lite'].get('mean_holdout_balanced_macro_f1', float('nan')):.3f}** "
        f"({summary['loso_lite'].get('n_folds', 0)} folds)\n"
        f"- `ship_candidate`: `{ship}`\n"
        f"- Doc: [`head_a_med_depth_smoke.md`](head_a_med_depth_smoke.md)\n\n"
        f"**Preferred A-med live operationalization:** meditation depth high vs low "
        f"(this smoke) over protocol rest↔med — better in-session UX even if public "
        f"subject-general F1 stays near chance (personal Muse cal still likely for ship).\n"
    )
    if ALT_DOC.exists():
        text = ALT_DOC.read_text()
        marker = "## Depth smoke result (2026-09-08)"
        if marker in text:
            text = text.split(marker)[0].rstrip() + note
        else:
            text = text.rstrip() + note
        ALT_DOC.write_text(text + "\n")
        print(f"[doc] patched {ALT_DOC}", flush=True)

    if MULTI_DOC.exists():
        text = MULTI_DOC.read_text()
        depth_blurb = (
            "\n\n### A-med depth operationalization (2026-09-08)\n\n"
            f"Preferred **in-session** A-med label pair: `meditation_depth_low` vs "
            f"`meditation_depth_high` from **{DATASET_STRING}** probe Q1 "
            f"(≤{Q1_LOW_MAX} vs ≥{Q1_HIGH_MIN}), muse4, `HeadADepthLinear`. "
            f"Smoke: holdout bal macro-F1 **{f1:.3f}**, ship_candidate `{ship}` — "
            f"see [`head_a_med_depth_smoke.md`](head_a_med_depth_smoke.md). "
            f"Protocol `rest`↔`meditation` (ds003816) remains a secondary / beginners "
            f"block contrast, not the live depth meter.\n"
        )
        marker = "### A-med depth operationalization (2026-09-08)"
        if marker in text:
            # replace from marker to next ## or end of A-med section — append after A-med table block
            pre, _, rest = text.partition(marker)
            # drop old blurb until next ### or ## at line start after first line
            lines = rest.splitlines()
            # first line is header already consumed; skip until blank+### or ##
            cut = 0
            for i, ln in enumerate(lines):
                if i == 0:
                    continue
                if ln.startswith("### ") or ln.startswith("## "):
                    cut = i
                    break
            else:
                cut = len(lines)
            text = pre.rstrip() + depth_blurb + "\n" + "\n".join(lines[cut:])
        else:
            # insert after A-med section labels block — after "**UX:** beginners"
            anchor = "**UX:** beginners “am I in rest or meditation?” — **not** mind-wandering vs concentration."
            if anchor in text:
                text = text.replace(anchor, anchor + depth_blurb, 1)
            else:
                text = text.rstrip() + depth_blurb
        MULTI_DOC.write_text(text if text.endswith("\n") else text + "\n")
        print(f"[doc] patched {MULTI_DOC}", flush=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tags = sorted(p.name.replace("_manifest.json", "") for p in WIN_DIR.glob("*_manifest.json"))
    print(f"[load] {len(tags)} local muse4 recordings from {WIN_DIR}", flush=True)

    packs: List[Dict[str, Any]] = []
    for tag in tags:
        pack = load_depth_pack(tag)
        packs.append(pack)
        print(
            f"  {tag} {pack['subject']} group={pack['group']} "
            f"probes={pack['n_probes_kept']} counts={pack['counts']} q1={pack['q1_hist']}",
            flush=True,
        )

    hold = next(p for p in packs if p["tag"] == HOLDOUT_TAG)
    train_packs = [p for p in packs if p["tag"] != HOLDOUT_TAG]
    Xtr, ytr, bal_detail = balance_train(train_packs, rng)
    train_subjects = [
        p["subject"] for p in train_packs if not bal_detail.get(p["tag"], {}).get("skipped", True)
    ]

    device = torch.device("cpu")
    weights = find_weights()
    print(f"[encoder] {weights} device={device}", flush=True)
    encoder = FrozenCBraModEncoder(weights, source_sr=256.0, pool="mean")

    print(f"[encode] train balanced n={len(ytr)}", flush=True)
    emb_tr = encode_all(encoder, Xtr, device)
    head, history, pick = train_head(emb_tr, ytr, device, rng)

    # re-eval train/val with best head (val metrics already in pick via last history best)
    # Reconstruct val metrics from best epoch row
    best_row = next(r for r in history if r["epoch"] == pick["best_epoch"])
    train_metrics = {
        "name": "train_balanced",
        "n": int(len(ytr)),
        "acc": best_row["train_acc"],
        "macro_f1": best_row["train_macro_f1"],
    }
    val_metrics = {
        "name": "val",
        "n": pick["n_val"],
        "acc": best_row["val_acc"],
        "macro_f1": best_row["val_macro_f1"],
    }

    emb_h = encode_all(encoder, hold["X"], device)
    hold_full = eval_split(head.cpu(), emb_h, hold["y"], "holdout_full")
    Xb, yb = undersample_balanced(hold["X"], hold["y"], np.random.default_rng(SEED))
    emb_b = encode_all(encoder, Xb, device)
    hold_bal = eval_split(head, emb_b, yb, "holdout_balanced")

    both_pred = (
        hold_bal.get("pred_counts", {}).get("meditation_depth_low", 0) > 0
        and hold_bal.get("pred_counts", {}).get("meditation_depth_high", 0) > 0
    )
    ship = bool(
        hold_bal.get("macro_f1") is not None
        and hold_bal["macro_f1"] >= SHIP_F1_MIN
        and both_pred
    )

    print("[loso-lite] starting…", flush=True)
    loso = loso_lite(packs, encoder, device, np.random.default_rng(SEED + 1), max_folds=5)

    completed_utc = datetime.now(timezone.utc).isoformat()
    takeaway = (
        f"A-med depth smoke on {DATASET_STRING} muse4: "
        f"Q1≤{Q1_LOW_MAX} vs ≥{Q1_HIGH_MIN}, 1 win/probe; "
        f"n_subjects_loaded={len(packs)}, train_used={len(train_subjects)}, "
        f"holdout={hold['subject']} n={hold_bal['n']}; "
        f"holdout bal macro-F1={hold_bal['macro_f1']:.3f} "
        f"(chance {CHANCE_MACRO_F1:.2f}, Δ {hold_bal['macro_f1'] - CHANCE_MACRO_F1:+.3f}); "
        f"LOSO-lite mean bal F1={loso.get('mean_holdout_balanced_macro_f1', float('nan')):.3f} "
        f"({loso.get('n_folds', 0)} folds); ship_candidate={ship}."
    )

    summary: Dict[str, Any] = {
        "version": VERSION_NOTE,
        "completed_utc": completed_utc,
        "completed_bangkok": bangkok_now(),
        "dataset_string": DATASET_STRING,
        "doi": DATASET_DOI,
        "license_spdx": LICENSE_SPDX,
        "montage": "muse4",
        "mapping": {
            "scale": "Q1 meditation depth 0–3 (Brandmeyer & Delorme)",
            "low": f"Q1 ≤ {Q1_LOW_MAX}",
            "high": f"Q1 ≥ {Q1_HIGH_MIN}",
            "mid": "none (integer split)",
            "window_policy": "one_window_per_probe_median_start",
        },
        "labels": list(HEAD_A_DEPTH_LABELS),
        "holdout_tag": HOLDOUT_TAG,
        "holdout_subject": HOLDOUT_SUBJECT,
        "holdout_n_windows": int(len(hold["y"])),
        "holdout_counts": hold["counts"],
        "n_subjects_loaded": len(packs),
        "n_train_subjects": len(train_subjects),
        "train_subjects": train_subjects,
        "n_train_balanced": int(len(ytr)),
        "balance_detail": bal_detail,
        "train_pick": pick,
        "history": history,
        "holdout_metrics": {
            "train_balanced": train_metrics,
            "val": val_metrics,
            "holdout_full": hold_full,
            "holdout_balanced": hold_bal,
        },
        "loso_lite": loso,
        "ship_candidate": ship,
        "chance_macro_f1": CHANCE_MACRO_F1,
        "takeaway": takeaway,
        "packs_meta": [
            {
                k: v
                for k, v in p.items()
                if k not in ("X", "y", "starts", "q1")
            }
            for p in packs
        ],
    }

    # save artifacts
    ckpt = OUT_DIR / "head_a_depth_linear_ds001787.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "labels": HEAD_A_DEPTH_LABELS,
            "in_dim": 200,
            "mapping": summary["mapping"],
            "holdout": HOLDOUT_SUBJECT,
            "version": VERSION_NOTE,
        },
        ckpt,
    )
    (OUT_DIR / "metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(
            {
                "script": "scripts/head_a_med_depth_smoke.py",
                "weights_sha256": sha256(weights),
                "ckpt": str(ckpt.relative_to(ROOT)),
                "seed": SEED,
                "epochs": EPOCHS,
                "dataset_string": DATASET_STRING,
            },
            indent=2,
        )
    )
    (OUT_DIR / "step_summary.json").write_text(
        json.dumps(
            {
                "holdout_balanced_macro_f1": hold_bal["macro_f1"],
                "holdout_full_macro_f1": hold_full["macro_f1"],
                "loso_lite_mean_balanced_macro_f1": loso.get(
                    "mean_holdout_balanced_macro_f1"
                ),
                "n_subjects": len(packs),
                "n_train_subjects": len(train_subjects),
                "n_holdout_windows": int(len(hold["y"])),
                "q1_low_max": Q1_LOW_MAX,
                "q1_high_min": Q1_HIGH_MIN,
                "ship_candidate": ship,
                "dataset_string": DATASET_STRING,
                "takeaway": takeaway,
            },
            indent=2,
        )
    )

    # export remapped one-window packs for inspection
    wdir = OUT_DIR / "windows"
    wdir.mkdir(parents=True, exist_ok=True)
    for p in packs:
        np.savez_compressed(
            wdir / f"{p['tag']}_depth_windows.npz",
            X=p["X"],
            y=p["y"],
            starts=p["starts"],
            q1=p["q1"],
            label_names=np.asarray(HEAD_A_DEPTH_LABELS),
            channels=np.asarray(p["channels"]),
        )
        (wdir / f"{p['tag']}_depth_manifest.json").write_text(
            json.dumps(
                {k: v for k, v in p.items() if k not in ("X", "y", "starts", "q1")},
                indent=2,
            )
        )

    write_doc(summary)
    patch_alt_and_multi(summary)
    print("[done]", takeaway, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train frozen A-eng head on expanded engagement corpus (REVE, local CPU/GPU).

Corpus: datasets/engagement_a_eng (133 unique persons, muse4, 19706 windows).
Strategy: embed-once per pack → cache → train HeadAEngLinear.
NO backbone fine-tune. Subject/person-wise splits from splits.json.
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

from src.reve_encoder import FrozenREVEEncoder
from src.heads.base import class_weights_from_y, undersample_balanced
from src.heads.head_a_eng import HEAD_A_ENG_LABELS, HeadAEngLinear
from src.metrics import confusion_matrix, per_class_report

SEED = 42
TARGET_SR = 256.0
EPOCHS = 20
BATCH_ENC = 8
BATCH_HEAD = 256
LR = 1e-3
PATIENCE = 5
MAX_TRAIN_BALANCED = 40_000
CHANCE_F1 = 0.5
SHIP_F1_MIN = 0.55

WIN_DIR = ROOT / "datasets/engagement_a_eng/windows"
SPLITS_PATH = ROOT / "datasets/engagement_a_eng/splits/splits.json"
OUT = ROOT / "exports/head_a_eng_train_reve"
EMB_CACHE = OUT / "emb_cache"
DOCS = ROOT / "docs/head_a_eng_train_reve.md"
DOCS_DUAL = ROOT / "docs/head_a_eng_dual_encoder.md"

# Sources with known residual order confounds (documented in corpus expansion)
ORDER_CONFOUND_SOURCES = {"ds007169", "eegmat", "stew"}
CLEANER_SOURCES = {"ds007262"}  # difficulty randomized
HOBBYIST_SOURCES = {"stew"}
PRO_SOURCES = {"ds007169", "ds007262", "ds007554", "eegmat"}


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


def load_splits() -> Dict[str, Any]:
    return json.loads(SPLITS_PATH.read_text())


def person_split_map(splits: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for sp, persons in splits["splits"].items():
        for pid in persons:
            out[pid] = sp
    return out


def pack_meta(pack_key: str) -> Dict[str, Any]:
    man = WIN_DIR / f"{pack_key}_aeng_manifest.json"
    if man.exists():
        return json.loads(man.read_text())
    # fallback from name
    source = pack_key.split("_")[0]
    return {"source": source, "unique_person_id": pack_key, "subject_key": pack_key}


@torch.no_grad()
def encode_batch(encoder: FrozenREVEEncoder, X: np.ndarray, device: torch.device) -> np.ndarray:
    chunks: List[np.ndarray] = []
    encoder.eval()
    for i in range(0, len(X), BATCH_ENC):
        xb = torch.from_numpy(X[i : i + BATCH_ENC]).to(device)
        chunks.append(encoder(xb).cpu().numpy().astype(np.float32))
    if not chunks:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def remap_y(y_raw: np.ndarray, label_names: Sequence[str]) -> np.ndarray:
    name_to_id = {n: i for i, n in enumerate(HEAD_A_ENG_LABELS)}
    y = np.full(len(y_raw), -1, dtype=np.int64)
    for old_i, name in enumerate(label_names):
        if str(name) in name_to_id:
            y[y_raw == old_i] = name_to_id[str(name)]
    return y


def ensure_emb_cache(
    encoder: FrozenREVEEncoder,
    device: torch.device,
    splits: Dict[str, Any],
    split_map: Dict[str, str],
) -> Dict[str, Any]:
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {"packs": [], "encoder": "REVE", "frozen": True}
    persons = splits["persons"]
    packs: List[Tuple[str, str, str]] = []  # pack_key, person_id, split
    for pid, info in persons.items():
        sp = split_map.get(pid)
        if sp is None:
            continue
        for pk in info["packs"]:
            packs.append((pk, pid, sp))
    packs.sort(key=lambda t: t[0])

    for i, (pk, pid, sp) in enumerate(packs):
        cache_path = EMB_CACHE / f"{pk}_emb.npz"
        npz_path = WIN_DIR / f"{pk}_aeng_windows.npz"
        man = pack_meta(pk)
        source = str(man.get("source", pk.split("_")[0]))
        if not npz_path.exists():
            print(f"  SKIP missing {pk}", flush=True)
            continue
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            meta["packs"].append(
                {
                    "pack_key": pk,
                    "unique_person_id": pid,
                    "split": sp,
                    "source": source,
                    "n": int(z["emb"].shape[0]),
                    "cached": True,
                    "device_class": man.get("device_class"),
                }
            )
            if (i + 1) % 20 == 0:
                print(f"  cache hit {i+1}/{len(packs)} last={pk} n={z['emb'].shape[0]}", flush=True)
            continue

        data = np.load(npz_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = remap_y(data["y"].astype(np.int64), [str(x) for x in data["label_names"].tolist()])
        keep = y >= 0
        X, y = X[keep], y[keep]
        if len(y) == 0:
            print(f"  SKIP empty labels {pk}", flush=True)
            continue
        emb = encode_batch(encoder, X, device)
        np.savez_compressed(
            cache_path,
            emb=emb.astype(np.float32),
            y=y.astype(np.int64),
            unique_person_id=np.asarray(pid),
            split=np.asarray(sp),
            pack_key=np.asarray(pk),
            source=np.asarray(source),
            device_class=np.asarray(str(man.get("device_class", "unknown"))),
        )
        meta["packs"].append(
            {
                "pack_key": pk,
                "unique_person_id": pid,
                "split": sp,
                "source": source,
                "n": int(len(y)),
                "cached": False,
                "y_counts": {HEAD_A_ENG_LABELS[j]: int((y == j).sum()) for j in range(2)},
                "device_class": man.get("device_class"),
            }
        )
        print(
            f"  encoded {i+1}/{len(packs)} {pk} split={sp} src={source} n={len(y)} "
            f"y={Counter(y.tolist())}",
            flush=True,
        )
        del data, X, emb
        gc.collect()

    (EMB_CACHE / "cache_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_pooled(
    split: str,
    source_filter: Optional[set] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    embs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    detail: Dict[str, Any] = {
        "packs": [],
        "counts": Counter(),
        "n_persons": 0,
        "sources": Counter(),
        "device_classes": Counter(),
    }
    persons = set()
    for path in sorted(EMB_CACHE.glob("*_emb.npz")):
        z = np.load(path, allow_pickle=True)
        if str(z["split"]) != split:
            continue
        src = str(z["source"]) if "source" in z.files else "unknown"
        if source_filter is not None and src not in source_filter:
            continue
        emb = z["emb"].astype(np.float32)
        y = z["y"].astype(np.int64)
        mask = y >= 0
        emb, y = emb[mask], y[mask]
        if len(y) == 0:
            continue
        embs.append(emb)
        ys.append(y)
        pid = str(z["unique_person_id"])
        persons.add(pid)
        dc = str(z["device_class"]) if "device_class" in z.files else "unknown"
        c = Counter({HEAD_A_ENG_LABELS[j]: int((y == j).sum()) for j in range(2)})
        detail["packs"].append(
            {
                "pack_key": str(z["pack_key"]),
                "unique_person_id": pid,
                "source": src,
                "device_class": dc,
                "n": int(len(y)),
                "counts": dict(c),
            }
        )
        detail["counts"].update(c)
        detail["sources"][src] += int(len(y))
        detail["device_classes"][dc] += int(len(y))
    if not embs:
        raise RuntimeError(f"no embeddings for split={split} filter={source_filter}")
    detail["counts"] = dict(detail["counts"])
    detail["sources"] = dict(detail["sources"])
    detail["device_classes"] = dict(detail["device_classes"])
    detail["n_persons"] = len(persons)
    detail["persons"] = sorted(persons)
    return np.concatenate(embs), np.concatenate(ys), detail


def eval_head(head, emb, y, device) -> Dict[str, Any]:
    head.eval()
    preds: List[int] = []
    with torch.no_grad():
        for i in range(0, len(emb), BATCH_HEAD):
            logits = head(torch.from_numpy(emb[i : i + BATCH_HEAD]).to(device))
            preds.extend(logits.argmax(-1).cpu().numpy().tolist())
    pred = np.asarray(preds, dtype=np.int64)
    report = per_class_report(y.tolist(), pred.tolist(), list(HEAD_A_ENG_LABELS))
    cm = confusion_matrix(y.tolist(), pred.tolist(), list(HEAD_A_ENG_LABELS))
    acc = float((pred == y).mean()) if len(y) else 0.0
    return {
        "n": int(len(y)),
        "accuracy": acc,
        "macro_f1": float(report["macro_f1"]["f1"]),
        "per_class": {k: v for k, v in report.items() if k != "macro_f1"},
        "confusion_matrix": cm.tolist(),
        "label_list": list(HEAD_A_ENG_LABELS),
        "pred_counts": {HEAD_A_ENG_LABELS[i]: int((pred == i).sum()) for i in range(2)},
        "true_counts": {HEAD_A_ENG_LABELS[i]: int((y == i).sum()) for i in range(2)},
        "delta_vs_chance": float(report["macro_f1"]["f1"]) - CHANCE_F1,
    }


def train_head(emb_fit, y_fit, emb_val, y_val, device):
    head = HeadAEngLinear(in_dim=emb_fit.shape[-1]).to(device)
    w = class_weights_from_y(y_fit, n_classes=2)
    crit = nn.CrossEntropyLoss(weight=w.to(device))
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(emb_fit), torch.from_numpy(y_fit)),
        batch_size=BATCH_HEAD,
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
        tr = eval_head(head, emb_fit, y_fit, device)
        va = eval_head(head, emb_val, y_val, device)
        row = {
            "epoch": ep + 1,
            "loss": total / max(n, 1),
            "train_macro_f1": tr["macro_f1"],
            "val_macro_f1": va["macro_f1"],
            "val_acc": va["accuracy"],
        }
        history.append(row)
        print(row, flush=True)
        if va["macro_f1"] > best_val + 1e-4:
            best_val = va["macro_f1"]
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


def ship_decision(test_m: Dict[str, Any], by_source: Dict[str, Any]) -> Tuple[bool, str]:
    """Honest fit/misfit for public A-eng."""
    reasons = []
    ok_metrics = True
    if test_m["macro_f1"] < SHIP_F1_MIN:
        ok_metrics = False
        reasons.append(f"test macro-F1 {test_m['macro_f1']:.3f} < {SHIP_F1_MIN}")
    for name in HEAD_A_ENG_LABELS:
        f1 = test_m["per_class"][name]["f1"]
        if f1 < 0.45:
            ok_metrics = False
            reasons.append(f"test {name} F1 {f1:.3f} < 0.45")
        if test_m["pred_counts"].get(name, 0) == 0:
            ok_metrics = False
            reasons.append(f"never predicts {name}")

    cleaner_ok = False
    for src in CLEANER_SOURCES:
        if src in by_source and by_source[src]["n"] >= 50:
            if by_source[src]["macro_f1"] >= SHIP_F1_MIN:
                cleaner_ok = True
            else:
                reasons.append(
                    f"cleaner source {src} test macro-F1 {by_source[src]['macro_f1']:.3f} < {SHIP_F1_MIN}"
                )

    confound_note = (
        "Residual order confounds remain on ds007169 (L1→L4), eegmat (rest→arith), "
        "and STEW (rest→SIMKAP). Domain mix: professional 10–20/32-ch + STEW Emotiv hobbyist. "
        "muse4 proxy montage — not true Muse-native."
    )

    if not ok_metrics:
        return False, "; ".join(reasons) + ". " + confound_note

    if not cleaner_ok:
        return (
            False,
            "Overall metrics clear soft bar but cleaner randomized source (ds007262) "
            "does not independently clear 0.55 — risk that F1 is partly order/time shortcut. "
            + confound_note,
        )

    return (
        True,
        "Metrics clear subject-holdout bar including cleaner ds007262; still proxy/domain-gap "
        "and residual order confounds on other sources — ship only as research/proxy engagement "
        "decoder, not consumer Muse product claim. " + confound_note,
    )


def write_docs(metrics: Dict[str, Any]) -> None:
    val, test = metrics["val"], metrics["test"]
    ship = metrics["ship_candidate"]
    lines = [
        "# Head A-eng train — expanded corpus (frozen REVE)",
        "",
        f"**Step:** `head_a_eng_train_reve`  ",
        f"**Date (UTC):** {metrics['created_utc']}  ",
        f"**ship_candidate:** **{ship}** — {metrics['ship_reason']}",
        "",
        "## Corpus",
        "",
        "- Path: `datasets/engagement_a_eng/`",
        "- Sources: ds007169, ds007262, ds007554, eegmat (professional) + stew (hobbyist Emotiv)",
        "- Unique persons: **133**; packs: 150; windows: ~19706 muse4",
        "- Splits: subject/person-wise 93 / 20 / 20 on `unique_person_id` (Barras tasks co-split)",
        "- Labels: `low_engagement` / `high_engagement`",
        "",
        "## Domain mix & confounds (honest)",
        "",
        "| Issue | Detail |",
        "|-------|--------|",
        "| Device mix | professional 10–20/32-ch vs STEW Emotiv 14-ch hobbyist |",
        "| Order | ds007169 L1→L4; eegmat rest→arith; STEW rest→SIMKAP |",
        "| Cleaner | ds007262 difficulty randomized (preferred signal check) |",
        "| Montage | muse4 proxy only — not true Muse |",
        "",
        "## Setup",
        "",
        "- Encoder: **frozen** REVE-base (embed-once → head train)",
        "- Head: `HeadAEngLinear` (in_dim=512)",
        "- Train: undersample_balanced (cap 40000); class-weighted CE; best-by-val; patience 5",
        "- Backbone fine-tune: **no**",
        "- Skipped: Head B, A-med",
        "",
        "## Metrics vs chance 0.5",
        "",
        "| Split | n | accuracy | macro-F1 | Δ vs chance |",
        "|-------|--:|---------:|---------:|------------:|",
        f"| train (fit) | {metrics['train_fit']['n']} | {metrics['train_fit']['accuracy']:.3f} | {metrics['train_fit']['macro_f1']:.3f} | {metrics['train_fit']['macro_f1']-CHANCE_F1:+.3f} |",
        f"| val | {val['n']} | {val['accuracy']:.3f} | {val['macro_f1']:.3f} | {val['macro_f1']-CHANCE_F1:+.3f} |",
        f"| test | {test['n']} | {test['accuracy']:.3f} | {test['macro_f1']:.3f} | {test['macro_f1']-CHANCE_F1:+.3f} |",
        "",
        "### Test per-class",
        "",
        "| class | precision | recall | f1 | support |",
        "|-------|----------:|-------:|---:|--------:|",
    ]
    for name in HEAD_A_ENG_LABELS:
        r = test["per_class"][name]
        lines.append(
            f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {int(r['support'])} |"
        )
    lines += [
        "",
        "### Test by source (subject-holdout packs)",
        "",
        "| source | n | macro-F1 | note |",
        "|--------|--:|---------:|------|",
    ]
    for src, m in sorted(metrics["test_by_source"].items()):
        note = []
        if src in ORDER_CONFOUND_SOURCES:
            note.append("order confound")
        if src in HOBBYIST_SOURCES:
            note.append("hobbyist")
        if src in CLEANER_SOURCES:
            note.append("randomized difficulty")
        lines.append(
            f"| {src} | {m['n']} | {m['macro_f1']:.3f} | {', '.join(note) or '—'} |"
        )
    lines += [
        "",
        "## Fit vs misfit recommendation",
        "",
        metrics["recommendation"],
        "",
        "## Artifacts",
        "",
        f"- Emb cache: `{EMB_CACHE.relative_to(ROOT)}/`",
        f"- Weights: `{metrics['head_path']}`",
        f"- Metrics: `exports/head_a_eng_train_reve/metrics_summary.json`",
        f"- Script: `scripts/train_head_a_eng_cbramod.py`",
        "",
        "See also dual-encoder doc: `docs/head_a_eng_dual_encoder.md`.",
        "",
    ]
    DOCS.write_text("\n".join(lines))


def write_dual_stub(metrics: Dict[str, Any]) -> None:
    """Write dual-encoder doc merging CBraMod + REVE metrics when present."""
    cbr_path = ROOT / "exports/head_a_eng_train_cbramod/metrics_summary.json"
    rev_path = ROOT / "exports/head_a_eng_train_reve/metrics_summary.json"
    c = json.loads(cbr_path.read_text()) if cbr_path.exists() else None
    r = json.loads(rev_path.read_text()) if rev_path.exists() else None
    if metrics.get("encoder") == "CBraMod":
        c = metrics
    if metrics.get("encoder") == "REVE":
        r = metrics
    lines = [
        "# Head A-eng dual-encoder fit vs misfit (CBraMod + REVE)",
        "",
        f"**Updated (UTC):** {metrics['created_utc']}",
        "",
        "## Task",
        "",
        "Frozen heads only (no backbone fine-tune) on `engagement_a_eng` "
        "(133 unique persons, muse4, person-wise 93/20/20).",
        "",
        "## Headline metrics (test macro-F1 vs chance 0.5)",
        "",
        "| Encoder | test n | test macro-F1 | delta vs chance | ship_candidate |",
        "|---------|-------:|--------------:|------------:|:--------------:|",
    ]
    if c:
        lines.append(
            f"| **CBraMod** | {c['test']['n']} | {c['test']['macro_f1']:.3f} | "
            f"{c['test']['macro_f1']-CHANCE_F1:+.3f} | **{c['ship_candidate']}** |"
        )
    else:
        lines.append("| **CBraMod** | — | — | — | — |")
    if r:
        lines.append(
            f"| **REVE** | {r['test']['n']} | {r['test']['macro_f1']:.3f} | "
            f"{r['test']['macro_f1']-CHANCE_F1:+.3f} | **{r['ship_candidate']}** |"
        )
    else:
        lines.append("| **REVE** | — | pending | — | pending |")
    lines += [
        "",
        "## CBraMod detail",
        "",
        "- Docs: `docs/head_a_eng_train_cbramod.md`",
        "- Exports: `exports/head_a_eng_train_cbramod/`",
        f"- ship_reason: {(c or {}).get('ship_reason', 'n/a')}",
        "",
        "## REVE detail",
        "",
        "- Docs: `docs/head_a_eng_train_reve.md`",
        "- Exports: `exports/head_a_eng_train_reve/`",
        f"- ship_reason: {(r or {}).get('ship_reason', 'pending')}",
        "- Prefer Kaggle T4; local CPU used if Kaggle P100 incompatible with current PyTorch.",
        "",
        "## Domain mix & confounds",
        "",
        "- Professional (ds007169/262/554, eegmat) + STEW hobbyist Emotiv.",
        "- Residual order confounds: ds007169, eegmat, STEW; cleaner: ds007262 randomized.",
        "",
        "## Combined recommendation",
        "",
        str(metrics.get("recommendation", "")),
        "",
        "Skipped: Head B, A-med. No backbone fine-tune.",
        "",
    ]
    DOCS_DUAL.write_text('\n'.join(lines))




def recommend(ship: bool, test_f1: float, by_source: Dict[str, Any]) -> str:
    parts = [
        "**Fit vs misfit (A-eng public head):**",
        "",
    ]
    if ship:
        parts.append(
            f"- **FIT (conditional):** test macro-F1={test_f1:.3f} (Δ vs chance {test_f1-CHANCE_F1:+.3f}) "
            "clears soft bar including cleaner ds007262. Keep backbone **frozen**. "
            "Ship only as muse4-proxy / research engagement decoder — not consumer Muse claim. "
            "Prefer personal Muse cal for product UX."
        )
    elif test_f1 >= SHIP_F1_MIN:
        parts.append(
            f"- **MISFIT for public ship (despite F1={test_f1:.3f}):** metrics above chance but "
            "order/domain confounds or cleaner-source check fail honesty bar. "
            "Keep exploring; do **not** fine-tune backbone to chase confounded signal. "
            "Optional: STEW-only / ds007262-only ablations; true-Muse personal cal."
        )
    elif test_f1 >= 0.52:
        parts.append(
            f"- **WEAK / MISFIT:** test macro-F1={test_f1:.3f} barely above chance. "
            "Frozen linear insufficient for public A-eng. Do not fine-tune yet — "
            "fixulate labels/montage/domain mix first."
        )
    else:
        parts.append(
            f"- **MISFIT:** test macro-F1={test_f1:.3f} ≈ chance. Public A-eng not viable on this corpus/encoder. "
            "No backbone fine-tune. Revisit labels or drop A-eng from ship set."
        )
    parts.append("- **Skipped:** Head B, A-med.")
    parts.append("- **REVE:** compare on same splits via Kaggle T4 before any fine-tune decision.")
    # source blurb
    if by_source:
        bits = ", ".join(f"{s}={m['macro_f1']:.3f}" for s, m in sorted(by_source.items()))
        parts.append(f"- Test-by-source macro-F1: {bits}")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-encode", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-train-balanced", type=int, default=MAX_TRAIN_BALANCED)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    EMB_CACHE.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    split_map = person_split_map(splits)
    print(
        f"persons={splits['n_unique_persons']} packs={splits['n_window_packs']} "
        f"splits train/val/test="
        f"{len(splits['splits']['train'])}/{len(splits['splits']['val'])}/{len(splits['splits']['test'])} "
        f"device={device}",
        flush=True,
    )

    if not args.skip_encode:
        cache_root = ROOT / "kaggle_datasets/muse-eeg-heads-cache/models"
        print(f"loading REVE (prefer local cache {cache_root})", flush=True)
        encoder = FrozenREVEEncoder(
            source_sr=TARGET_SR,
            cache_roots=[cache_root, cache_root.parent],
            prefer_local=True,
        ).to(device)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        print(
            f"emb_dim={encoder.emb_dim} local={getattr(encoder, 'model_from_local', None)}",
            flush=True,
        )
        ensure_emb_cache(encoder, device, splits, split_map)
        del encoder
        gc.collect()
    else:
        print("reusing emb_cache", flush=True)

    emb_tr, y_tr, det_tr = load_pooled("train")
    emb_va, y_va, det_va = load_pooled("val")
    emb_te, y_te, det_te = load_pooled("test")
    print(
        f"pooled train n={len(y_tr)} persons={det_tr['n_persons']} "
        f"val n={len(y_va)} test n={len(y_te)}",
        flush=True,
    )
    print("train sources", det_tr["sources"], "devices", det_tr["device_classes"], flush=True)

    emb_fit, y_fit = undersample_balanced(emb_tr, y_tr, rng=rng)
    if args.max_train_balanced > 0 and len(y_fit) > args.max_train_balanced:
        # further downsample keeping balance
        per = args.max_train_balanced // 2
        picks = []
        for c in (0, 1):
            cand = np.where(y_fit == c)[0]
            picks.append(rng.choice(cand, size=min(per, len(cand)), replace=False))
        sel = np.concatenate(picks)
        rng.shuffle(sel)
        emb_fit, y_fit = emb_fit[sel], y_fit[sel]
    print(f"train fit balanced n={len(y_fit)} counts={Counter(y_fit.tolist())}", flush=True)

    head, history, best_val = train_head(emb_fit, y_fit, emb_va, y_va, device)
    train_m = eval_head(head, emb_fit, y_fit, device)
    val_m = eval_head(head, emb_va, y_va, device)
    test_m = eval_head(head, emb_te, y_te, device)
    print("TEST", {k: test_m[k] for k in ("n", "accuracy", "macro_f1", "delta_vs_chance")}, flush=True)

    # per-source test
    by_source: Dict[str, Any] = {}
    for src in sorted({p["source"] for p in det_te["packs"]}):
        try:
            e_s, y_s, _ = load_pooled("test", source_filter={src})
            by_source[src] = eval_head(head, e_s, y_s, device)
        except RuntimeError:
            continue

    ship, ship_reason = ship_decision(test_m, by_source)
    rec = recommend(ship, test_m["macro_f1"], by_source)

    head_path = OUT / "head_a_eng_linear.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": list(HEAD_A_ENG_LABELS),
            "in_dim": int(emb_fit.shape[-1]),
            "encoder": "REVE",
            "frozen_backbone": True,
            "train_status": "trained_fit_misfit_eval",
        },
        head_path,
    )

    created = datetime.now(timezone.utc).isoformat()
    metrics = {
        "created_utc": created,
        "encoder": "REVE",
        "frozen_backbone": True,
        "chance_macro_f1": CHANCE_F1,
        "n_unique_persons": splits["n_unique_persons"],
        "n_packs_cached": len(list(EMB_CACHE.glob("*_emb.npz"))),
        "split_counts": {
            "train_persons": len(splits["splits"]["train"]),
            "val_persons": len(splits["splits"]["val"]),
            "test_persons": len(splits["splits"]["test"]),
        },
        "train_pool": {"n": int(len(y_tr)), **{k: det_tr[k] for k in ("n_persons", "sources", "device_classes", "counts")}},
        "val_pool": {"n": int(len(y_va)), **{k: det_va[k] for k in ("n_persons", "sources", "device_classes", "counts")}},
        "test_pool": {"n": int(len(y_te)), **{k: det_te[k] for k in ("n_persons", "sources", "device_classes", "counts")}},
        "train_fit": train_m,
        "val": val_m,
        "test": test_m,
        "test_by_source": by_source,
        "history": history,
        "best_val_macro_f1": best_val,
        "ship_candidate": ship,
        "ship_reason": ship_reason,
        "recommendation": rec,
        "head_path": str(head_path.relative_to(ROOT)),
        "weights_sha256": None,
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (OUT / "run_manifest.json").write_text(
        json.dumps(
            {
                "step": "head_a_eng_train_reve",
                "created_utc": created,
                "device": str(device),
                "seed": SEED,
                "corpus": "engagement_a_eng",
                "ship_candidate": ship,
            },
            indent=2,
        )
        + "\n"
    )
    (OUT / "step_summary.json").write_text(
        json.dumps(
            {
                "test_macro_f1": test_m["macro_f1"],
                "delta_vs_chance": test_m["macro_f1"] - CHANCE_F1,
                "ship_candidate": ship,
                "ship_reason": ship_reason,
            },
            indent=2,
        )
        + "\n"
    )
    write_docs(metrics)
    write_dual_stub(metrics)

    # update ENG_TRAIN_STATUS in source for honesty
    eng_py = ROOT / "src/heads/head_a_eng.py"
    txt = eng_py.read_text()
    old = 'ENG_TRAIN_STATUS = "corpus_ready_not_ship"'
    new = f'ENG_TRAIN_STATUS = "trained_reve_ship_{str(ship).lower()}"'
    if old in txt:
        eng_py.write_text(txt.replace(old, new, 1))

    print(
        f"DONE ship={ship} test_macro_f1={test_m['macro_f1']:.3f} "
        f"delta={test_m['macro_f1']-CHANCE_F1:+.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

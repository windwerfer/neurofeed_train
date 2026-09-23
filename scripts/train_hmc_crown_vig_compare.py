#!/usr/bin/env python3
"""Train Head A-vig (drowsy vs hypnagogic) with frozen CBraMod on HMC crown2 vs crown4.

True C=2 / C=4 tensors. FrozenCBraModEncoder is channel-flexible (mean-pool → 200-d).
No zero-pad to Crown8.

Packs: datasets/vigilance_hmc_crown2 , datasets/vigilance_hmc_crown4
Shared subject splits. Writes exports/hmc_crown_vig_compare/ and docs/crown_hmc_vig_compare.md.
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
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cbramod_encoder import FrozenCBraModEncoder
from src.head_c import class_weights_from_y, undersample_balanced
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
TRAIN_BALANCE = True
MAX_TRAIN_BALANCED = 80_000
SHIP_TEST_F1 = 0.60

PACKS = {
    "crown2_strong": "vigilance_hmc_crown2",
    "crown4_hmc": "vigilance_hmc_crown4",
}

ISRUC_PLAN = """OUT OF SCOPE for public ship. Research-only note (do not publish ISRUC windows/heads here).

- Dataset: ISRUC-Sleep (https://sleeptight.isr.uc.pt/) — non-PhysioNet; check institutional license / request form before download.
- Caveat: not CC-BY; redistribution often restricted. Keep private; do not push to public HF/GitHub.
- Suggested montage map (research proxy, document honesty):
  - crown2: C3-A2, C4-A1 (or C3/C4 as available)
  - crown4: add F3/F4 or O1/O2 closest available
- Same N1-slice + Head A-vig labels; same subject-wise split recipe; frozen CBraMod.
- Goal: smoke whether crown2/4 transfer holds outside HMC (~30-subject subset).
- Scripts later: scripts/expand_isruc_crown_vig.py mirroring HMC expand; reuse this train compare.
"""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_weights() -> Path:
    cands = [
        Path(os.environ.get("CBRAMOD_WEIGHTS", str(ROOT / "data/models/CBraMod/pretrained_weights.pth"))),
        Path("/kaggle/input/muse-eeg-heads-crown-vig/models/CBraMod/pretrained_weights.pth")  # optional private scratch,
        Path("/kaggle/input/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth"),
    ]
    kin = Path("/kaggle/input")
    if kin.exists():
        cands.extend(sorted(kin.rglob("pretrained_weights.pth")))
    for p in cands:
        if p.exists():
            return p
    raise FileNotFoundError("CBraMod weights not found")


def qc_mask(X: np.ndarray) -> np.ndarray:
    std = X.std(axis=-1)
    peak = np.max(np.abs(X), axis=-1)
    return ~((std < FLAT_STD).any(axis=-1) | (peak > PEAK_ABS).any(axis=-1))


def cap_idxs_per_class(y: np.ndarray, per_class: int, rng: np.random.Generator) -> np.ndarray:
    if per_class <= 0:
        return np.arange(len(y))
    picks = []
    for c in np.unique(y):
        cand = np.where(y == c)[0]
        n = min(int(per_class), len(cand))
        if n:
            picks.append(rng.choice(cand, size=n, replace=False))
    if not picks:
        return np.zeros(0, dtype=np.int64)
    out = np.concatenate(picks)
    rng.shuffle(out)
    return out


def encode_batch(encoder: FrozenCBraModEncoder, X: np.ndarray, device: torch.device) -> np.ndarray:
    chunks: List[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(X), BATCH_ENC):
            xb = torch.from_numpy(X[i : i + BATCH_ENC]).to(device)
            chunks.append(encoder(xb).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 200), dtype=np.float32)


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
) -> Tuple[nn.Module, List[Dict[str, Any]], float]:
    head = HeadAVigLinear(in_dim=emb_fit.shape[-1], n_classes=len(labels)).to(device)
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
                print(f"early stop at epoch {ep + 1}", flush=True)
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head, history, float(best_val)


def load_split_map(pack: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for sp in ("train", "val", "test"):
        data = json.loads((pack / "splits" / f"{sp}_subjects.json").read_text())
        for sid in data["subjects"]:
            out[sid] = sp
    return out


def ensure_emb_cache(
    pack: Path,
    out_dir: Path,
    encoder: FrozenCBraModEncoder,
    device: torch.device,
    split_map: Dict[str, str],
    expected_c: int,
) -> Dict[str, Any]:
    emb_cache = out_dir / "emb_cache"
    emb_cache.mkdir(parents=True, exist_ok=True)
    win_dir = pack / "windows"
    meta: Dict[str, Any] = {"recordings": [], "expected_c": expected_c}
    rids = sorted(p.name.replace("_windows.npz", "") for p in win_dir.glob("*_windows.npz"))
    for i, rid in enumerate(rids):
        split = split_map.get(rid)
        if split is None:
            print(f"  SKIP no split {rid}", flush=True)
            continue
        cache_path = emb_cache / f"{rid}_emb.npz"
        if cache_path.exists():
            z = np.load(cache_path, allow_pickle=True)
            meta["recordings"].append(
                {"recording_id": rid, "split": split, "n": int(z["emb"].shape[0]), "cached": True}
            )
            continue
        data = np.load(win_dir / f"{rid}_windows.npz", allow_pickle=True)
        X = data["X"].astype(np.float32)
        if X.ndim != 3 or X.shape[1] != expected_c:
            raise RuntimeError(f"{rid}: expected C={expected_c}, got {X.shape}")
        y_raw = data["y"].astype(np.int64)
        label_names = [str(x) for x in data["label_names"].tolist()]
        name_to_vig = {n: HEAD_A_VIG_LABELS_2.index(n) for n in label_names if n in HEAD_A_VIG_LABELS_2}
        y = np.full(len(y_raw), -1, dtype=np.int64)
        for old_i, name in enumerate(label_names):
            if name in name_to_vig:
                y[y_raw == old_i] = name_to_vig[name]
        keep = qc_mask(X) & (y >= 0)
        idxs = np.where(keep)[0]
        if len(idxs) == 0:
            print(f"  SKIP empty after QC {rid}", flush=True)
            continue
        emb = encode_batch(encoder, X[idxs], device)
        y_u = y[idxs]
        np.savez_compressed(
            cache_path,
            emb=emb.astype(np.float32),
            y_a=y_u.astype(np.int64),
            subject_id=np.asarray(rid),
            split=np.asarray(split),
            recording_id=np.asarray(rid),
            n_channels=np.asarray(expected_c),
        )
        meta["recordings"].append(
            {
                "recording_id": rid,
                "split": split,
                "n": int(len(idxs)),
                "n_before_qc": int(len(X)),
                "cached": False,
                "y_a_counts": {HEAD_A_VIG_LABELS_2[j]: int((y_u == j).sum()) for j in range(2)},
            }
        )
        print(
            f"  encoded {i + 1}/{len(rids)} {rid} C={expected_c} split={split} n={len(idxs)} {Counter(y_u.tolist())}",
            flush=True,
        )
        del data, X, emb
        gc.collect()
    (emb_cache / "cache_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_pooled(emb_cache: Path, split: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    embs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    detail: Dict[str, Any] = {"recordings": [], "counts": Counter(), "subjects": set()}
    for path in sorted(emb_cache.glob("*_emb.npz")):
        z = np.load(path, allow_pickle=True)
        if str(z["split"]) != split:
            continue
        emb = z["emb"].astype(np.float32)
        y = z["y_a"].astype(np.int64)
        mask = y >= 0
        emb, y = emb[mask], y[mask]
        if len(y) == 0:
            continue
        embs.append(emb)
        ys.append(y)
        sid = str(z["subject_id"])
        detail["subjects"].add(sid)
        c = Counter({HEAD_A_VIG_LABELS_2[j]: int((y == j).sum()) for j in range(2)})
        detail["recordings"].append(
            {"recording_id": str(z["recording_id"]), "subject_id": sid, "n": int(len(y)), "counts": dict(c)}
        )
        detail["counts"].update(c)
    if not embs:
        raise RuntimeError(f"no embeddings for split={split}")
    detail["counts"] = dict(detail["counts"])
    detail["n_subjects"] = len(detail["subjects"])
    detail["subjects"] = sorted(detail["subjects"])
    return np.concatenate(embs), np.concatenate(ys), detail


def run_one(
    montage_key: str,
    pack_name: str,
    weights: Path,
    device: torch.device,
    rng: np.random.Generator,
    out_root: Path,
) -> Dict[str, Any]:
    pack = ROOT / "datasets" / pack_name
    policy_path = pack / "splits" / "split_policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"missing {policy_path}")
    policy = json.loads(policy_path.read_text())
    montage = policy.get("montage") or {}
    expected_c = int(montage.get("ch_count") or 0)
    if expected_c <= 0:
        sample = next((pack / "windows").glob("*_windows.npz"))
        expected_c = int(np.load(sample)["X"].shape[1])
    split_map = load_split_map(pack)
    out_dir = out_root / montage_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== {montage_key} C={expected_c} pack={pack_name} =====", flush=True)
    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean", map_location=str(device))
    encoder.to(device)
    print("encoder notes:", encoder.adapter_notes(), flush=True)
    smoke = torch.zeros(2, expected_c, 512, device=device)
    emb_smoke = encoder(smoke)
    assert emb_smoke.shape == (2, 200), emb_smoke.shape
    print(f"smoke OK: (2,{expected_c},512) → {tuple(emb_smoke.shape)}", flush=True)

    meta = ensure_emb_cache(pack, out_dir, encoder, device, split_map, expected_c)
    del encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    emb_tr, y_tr, det_tr = load_pooled(out_dir / "emb_cache", "train")
    emb_va, y_va, det_va = load_pooled(out_dir / "emb_cache", "val")
    emb_te, y_te, det_te = load_pooled(out_dir / "emb_cache", "test")
    print("train", det_tr["counts"], "n", len(y_tr), flush=True)
    print("val", det_va["counts"], "n", len(y_va), flush=True)
    print("test", det_te["counts"], "n", len(y_te), flush=True)

    if TRAIN_BALANCE:
        emb_fit, y_fit = undersample_balanced(emb_tr, y_tr, rng)
        if MAX_TRAIN_BALANCED > 0 and len(y_fit) > MAX_TRAIN_BALANCED:
            idxs = cap_idxs_per_class(y_fit, MAX_TRAIN_BALANCED // 2, rng)
            emb_fit, y_fit = emb_fit[idxs], y_fit[idxs]
    else:
        emb_fit, y_fit = emb_tr, y_tr
    print("fit", Counter(y_fit.tolist()), flush=True)

    head, hist, best_va = train_head(emb_fit, y_fit, emb_va, y_va, HEAD_A_VIG_LABELS_2, device)
    train_fit = eval_head(head, emb_fit, y_fit, HEAD_A_VIG_LABELS_2, device)
    val_m = eval_head(head, emb_va, y_va, HEAD_A_VIG_LABELS_2, device)
    test_m = eval_head(head, emb_te, y_te, HEAD_A_VIG_LABELS_2, device)
    ship = bool(test_m["macro_f1"] >= SHIP_TEST_F1)
    ship_reason = (
        f"test macro-F1 {test_m['macro_f1']:.3f} >= {SHIP_TEST_F1}"
        if ship
        else f"test macro-F1 {test_m['macro_f1']:.3f} < {SHIP_TEST_F1} (user bar)"
    )

    head_path = out_dir / "head_a_vig_linear.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "label_list": HEAD_A_VIG_LABELS_2,
            "in_dim": int(emb_fit.shape[-1]),
            "encoder": "CBraMod_frozen",
            "montage": montage_key,
            "n_channels": expected_c,
            "ch_names": montage.get("ch_names"),
        },
        head_path,
    )
    counts = policy.get("counts") or {}
    result = {
        "montage": montage_key,
        "pack": pack_name,
        "n_channels": expected_c,
        "ch_names": montage.get("ch_names"),
        "n_subjects": counts.get("n_subjects_ok") or counts.get("n_subjects") or len(split_map),
        "splits": counts,
        "encode_meta_n_recordings": len(meta["recordings"]),
        "train_detail": {"n_subjects": det_tr["n_subjects"], "counts": det_tr["counts"]},
        "val_detail": {"n_subjects": det_va["n_subjects"], "counts": det_va["counts"]},
        "test_detail": {"n_subjects": det_te["n_subjects"], "counts": det_te["counts"]},
        "best_val_macro_f1": best_va,
        "metrics": {"train_fit": train_fit, "val": val_m, "test": test_m},
        "history": hist,
        "ship_candidate": ship,
        "ship_reason": ship_reason,
        "ship_bar": SHIP_TEST_F1,
        "head_path": str(head_path),
        "weights_sha256": sha256_file(weights),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[{montage_key}] val_f1={val_m['macro_f1']:.3f} test_f1={test_m['macro_f1']:.3f} ship={ship}",
        flush=True,
    )
    return result


def write_docs(compare: Dict[str, Any], path: Path) -> None:
    rows = compare["results"]
    lines = [
        "# Crown vigilance HMC compare (Head A-vig, frozen CBraMod)",
        "",
        f"Created (UTC): {compare.get('created_utc')}",
        f"Ship bar (test macro-F1): **{SHIP_TEST_F1}**",
        "",
        "## Setup",
        "- Corpus: PhysioNet hmc-sleep-staging 1.1 (CC-BY-4.0), N1-slice (2s@256Hz, hop 0.5s, 20/40 min, majority 0.7)",
        "- Labels: W→drowsy, N1→hypnagogic",
        "- Encoder: frozen CBraMod (channel-flexible; mean pool → 200-d). True C=2 / C=4 — **no zero-pad to 8**.",
        "- Splits: subject-wise ~70/15/15, identical subjects across montages.",
        "",
        "## Metrics",
        "",
        "| montage | C | channels | n_subj | val macro-F1 | test macro-F1 | ship (≥0.60) |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['montage']} | {r['n_channels']} | {', '.join(r.get('ch_names') or [])} | "
            f"{r.get('n_subjects')} | {m['val']['macro_f1']:.3f} | {m['test']['macro_f1']:.3f} | "
            f"{'yes' if r['ship_candidate'] else 'no'} |"
        )
    lines += ["", "## Per-montage notes", ""]
    for r in rows:
        lines.append(f"### {r['montage']}")
        lines.append(f"- {r['ship_reason']}")
        lines.append(f"- test true counts: {r['metrics']['test'].get('true_counts')}")
        lines.append(f"- test pred counts: {r['metrics']['test'].get('pred_counts')}")
        lines.append("")
    if compare.get("isruc_next_steps"):
        lines += ["## ISRUC next steps (if both < 0.60)", "", compare["isruc_next_steps"], ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--montages", nargs="*", default=list(PACKS.keys()))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "exports" / "hmc_crown_vig_compare"))
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}", flush=True)
    weights = find_weights()
    print(f"weights={weights}", flush=True)
    rng = np.random.default_rng(SEED)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    results = [run_one(k, PACKS[k], weights, device, rng, out_root) for k in args.montages]
    any_ship = any(r["ship_candidate"] for r in results)
    both_below = all(r["metrics"]["test"]["macro_f1"] < SHIP_TEST_F1 for r in results)
    compare = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "ship_bar": SHIP_TEST_F1,
        "any_ship_candidate": any_ship,
        "both_below_bar": both_below,
        "results": results,
        "isruc_next_steps": ISRUC_PLAN if both_below else None,
        "isruc_smoke_warranted": both_below,
    }
    (out_root / "compare.json").write_text(json.dumps(compare, indent=2) + "\n")
    write_docs(compare, ROOT / "docs" / "crown_hmc_vig_compare.md")
    print(
        json.dumps(
            {
                "any_ship": any_ship,
                "both_below": both_below,
                "f1": {r["montage"]: r["metrics"]["test"]["macro_f1"] for r in results},
            },
            indent=2,
        )
    )
    print(f"docs → {ROOT / 'docs' / 'crown_hmc_vig_compare.md'}")


if __name__ == "__main__":
    main()

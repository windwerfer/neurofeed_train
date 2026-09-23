#!/usr/bin/env python3
"""Local smoke: SC4001 N1-slice windows → frozen CBraMod → Head A binary fit."""
from __future__ import annotations

import json
import hashlib
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

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
from src.metrics import macro_f1

SEED = 42
WINDOW_SEC = 2.0
HOP_SEC = 0.5
TARGET_SR = 256.0
PRE_SEC = 20 * 60
POST_SEC = 40 * 60
EPOCHS = 3
BATCH = 32
LR = 1e-3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    cache_roots = [
        ROOT / "kaggle_datasets/muse-eeg-heads-cache/data/sleep-edfx-pilot",
        Path("/tmp/kaggle_out3/data/sleep-edfx-pilot"),
    ]
    pairs = []
    for r in cache_roots:
        if r.exists():
            pairs = find_pilot_pairs(r)
            if pairs:
                break
    if not pairs:
        raise FileNotFoundError("No Sleep-EDF pilot pairs found")

    psg, hyp = pairs[0]
    print("PSG", psg.name, "HYP", hyp.name)

    weights_cands = [
        ROOT / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth",
        Path("/tmp/kaggle_out3/models/CBraMod/pretrained_weights.pth"),
    ]
    weights = next(p for p in weights_cands if p.exists())

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
    )
    mask = [lab in HEAD_A_BINARY_LABELS for lab in labels]
    X = X[np.asarray(mask)]
    labels = [lab for lab, m in zip(labels, mask) if m]
    y = labels_to_ids(labels, HEAD_A_BINARY_LABELS)
    counts = Counter(labels)
    print("windows", X.shape, "counts", dict(counts))

    hop = int(round(HOP_SEC * TARGET_SR))
    starts = np.asarray([keep[i] * hop for i, m in enumerate(mask) if m], dtype=np.int64)

    out_dir = ROOT / "exports" / "windows_sc4001"
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "sleep_edf_sc4001_n1slice_windows.npz"
    np.savez_compressed(
        npz_path,
        X=X.astype(np.float32),
        y=y,
        starts=starts,
        label_names=np.asarray(HEAD_A_BINARY_LABELS),
    )

    encoder = FrozenCBraModEncoder(weights, source_sr=TARGET_SR, pool="mean")
    notes = encoder.adapter_notes()
    print("encoder", notes["native_input"], "<-", notes["fed_input"])

    Xb, yb = undersample_balanced(X, y, rng)
    print("balanced", Xb.shape, Counter(yb.tolist()))

    device = torch.device("cpu")
    Xt = torch.from_numpy(Xb)
    yt = torch.from_numpy(yb)
    emb_list = []
    encoder.to(device)
    with torch.no_grad():
        for i in range(0, len(Xt), BATCH):
            emb_list.append(encoder(Xt[i : i + BATCH].to(device)).cpu())
    emb = torch.cat(emb_list, dim=0)
    print("emb", tuple(emb.shape))

    head = HeadALinear(in_dim=emb.shape[-1], n_classes=2).to(device)
    w = class_weights_from_y(yb, n_classes=2)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(emb, yt), batch_size=BATCH, shuffle=True)

    history = []
    for ep in range(EPOCHS):
        head.train()
        total, n = 0.0, 0
        for xb, ybatch in loader:
            opt.zero_grad()
            logits = head(xb)
            loss = crit(logits, ybatch)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(ybatch)
            n += len(ybatch)
        head.eval()
        with torch.no_grad():
            pred = head(emb).argmax(dim=-1).numpy()
        acc = float((pred == yb).mean())
        f1 = macro_f1(yb.tolist(), pred.tolist(), HEAD_A_BINARY_LABELS)
        row = {"epoch": ep + 1, "loss": total / max(n, 1), "acc": acc, "macro_f1": f1}
        history.append(row)
        print(row)

    run_dir = ROOT / "exports" / "head_a_smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    head_path = run_dir / "head_a_binary_state_dict.pt"
    torch.save(head.state_dict(), head_path)

    manifest = {
        "psg_file": psg.name,
        "hypno_file": hyp.name,
        "slice_start_sec": rec["slice_start_sec"],
        "pre_sec": PRE_SEC,
        "post_sec": POST_SEC,
        "window_sec": WINDOW_SEC,
        "hop_sec": HOP_SEC,
        "target_sr": TARGET_SR,
        "channel_proxy_note": PROXY_NOTE,
        "stage_to_label_map": {k: v for k, v in STAGE_TO_HEAD_A.items()},
        "n_windows_per_label": dict(counts),
        "n_windows_total": int(X.shape[0]),
        "balanced_n_per_class": int(len(yb) // 2),
        "random_seed": SEED,
        "encoder_name": "CBraMod",
        "encoder_weights_path": str(weights),
        "encoder_weights_sha256": notes["weights_sha256"],
        "encoder_adapter": notes,
        "window_shape": list(X.shape),
        "npz_sha256": sha256(npz_path),
        "head": "HeadALinear",
        "task": "binary_drowsy_vs_hypnagogic",
        "label_list": HEAD_A_BINARY_LABELS,
        "history": history,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "license_attribution": {
            "Sleep-EDF": "PhysioNet ODC-By — attribute PhysioNet / Sleep-EDF Expanded",
            "CBraMod": "Apache-2.0 — weighting666/CBraMod + wjq-learning/CBraMod",
        },
    }
    man_path = run_dir / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))

    win_pkg = ROOT / "kaggle_datasets" / "muse-eeg-heads-windows"
    win_pkg.mkdir(parents=True, exist_ok=True)
    shutil.copy(npz_path, win_pkg / npz_path.name)
    (win_pkg / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (win_pkg / "ATTRIBUTION.txt").write_text(
        "Sleep-EDF Expanded (PhysioNet) — ODC-By. Attribute PhysioNet and Sleep-EDF authors.\n"
        "Derived windows only; not a redistribution of full PSG nights beyond pilot terms.\n"
        "CBraMod weights used for encoder smoke are Apache-2.0 (HF weighting666/CBraMod);\n"
        "this dataset does NOT bundle CBraMod weights — windows + labels + manifest only.\n"
        "NO LUNA / NO L-FAME / NO SEED-VIG.\n"
    )
    print("wrote", npz_path)
    print("wrote", man_path)
    print("wrote package", win_pkg)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build notebooks/06_reve_attention_loso.ipynb + kaggle_kernel_06 folder."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "notebooks" / "06_reve_attention_loso.ipynb"
KDIR = ROOT / "kaggle_kernel_06_reve_attention_loso"
CODE = "muse-eeg-heads-reve-attention-loso.ipynb"


def lines(s: str) -> list[str]:
    s = dedent(s).strip("\n") + "\n"
    return [ln + "\n" for ln in s.split("\n")[:-1]] + ([s.split("\n")[-1] + "\n"] if True else [])


def md(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(s)}


def code(s: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines(s),
    }


CELLS = [
    md(
        """
        # REVE attention LOSO (muse4, ~80 subjects)

        **Pipeline:** embed-once with frozen `brain-bzh/reve-base` → head-only LOSO (linear + MLP) on Z cache.

        **Compare to:** CBraMod muse4 LOSO macro-F1 ≈ 0.36 (`docs/loso_head_a.md`).

        **Window policy:** native 2.0 s @ 256 Hz → resample **200 Hz → T=400** (2 patches × 200, overlap 20).
        Not inventing 4 s/5 s from 2 s data. Channels AF7,AF8,TP9,TP10 + official positions.

        **Cache:** attach private `windwerfer/muse-eeg-heads-cache` (offline `models/reve-base` + `reve-positions`).

        **Secrets (fallback only):** `HF_TOKEN` if cache missing — HF account that accepted gated `brain-bzh/reve-base`.

        **GPU:** enable GPU in kernel metadata. Internet optional when cache attached. Private kernel.
        """
    ),
    md("## 1. Setup"),
    code(
        r"""
        import os, sys, json, time, traceback, shutil, subprocess
        from pathlib import Path
        from collections import Counter
        from datetime import datetime, timezone

        WORKING = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".").resolve()
        INPUT_ROOT = Path("/kaggle/input")
        OUT = WORKING / "reve_attention_loso"
        OUT.mkdir(parents=True, exist_ok=True)
        LOG = []

        def log(msg):
            line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
            print(line, flush=True)
            LOG.append(line)

        def _list_tree(root: Path, max_depth=3):
            out = []
            if not root.exists():
                return out
            for p in sorted(root.rglob("*")):
                try:
                    rel = p.relative_to(root)
                except Exception:
                    continue
                if len(rel.parts) <= max_depth:
                    out.append(str(rel) + ("/" if p.is_dir() else ""))
            return out[:100]

        log(f"input tree sample: {_list_tree(INPUT_ROOT)}")

        def ensure_pkgs():
            pkgs = ["transformers>=4.40", "einops", "accelerate", "safetensors", "huggingface_hub"]
            uv_bin = shutil.which("uv")
            if uv_bin:
                log(f"using uv at {uv_bin}")
                subprocess.check_call([uv_bin, "pip", "install", "--system", "-q", *pkgs])
            else:
                log("uv not found; using pip")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

        ensure_pkgs()
        import numpy as np
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            log(f"gpu={torch.cuda.get_device_name(0)}")
        """
    ),
    code(
        r"""
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        try:
            from kaggle_secrets import UserSecretsClient
            secret = UserSecretsClient().get_secret("HF_TOKEN")
            if secret:
                hf_token = secret
                os.environ["HF_TOKEN"] = secret
                os.environ["HUGGINGFACE_HUB_TOKEN"] = secret
                log("HF_TOKEN loaded from Kaggle UserSecrets")
        except Exception as e:
            log(f"kaggle_secrets note: {e}")

        if not hf_token:
            log("NOTE: no HF_TOKEN — OK if muse-eeg-heads-cache attached; else gated hub download will fail")
        else:
            log(f"HF_TOKEN present (len={len(hf_token)})")

        def find_dir(names, marker_globs):
            if INPUT_ROOT.exists():
                for name in names:
                    d = INPUT_ROOT / name
                    if d.exists():
                        for g in marker_globs:
                            if list(d.glob(g)):
                                return d
                for g in marker_globs:
                    hits = list(INPUT_ROOT.rglob(g))
                    if hits:
                        return hits[0].parent
            return None

        src_dir = find_dir(("muse-eeg-heads-src",), ("reve_encoder.py", "head_a.py", "sleep_edf.py"))
        win_dir = find_dir(("muse-eeg-heads-windows",), ("*_attention_windows.npz",))
        cache_dir = find_dir(
            ("muse-eeg-heads-cache",),
            ("models/reve-base/model.safetensors", "models/CBraMod/pretrained_weights.pth", "models/MANIFEST.md"),
        )
        log(f"src_dir={src_dir}")
        log(f"win_dir={win_dir}")
        log(f"cache_dir={cache_dir}")
        if src_dir is None:
            raise FileNotFoundError("Attach windwerfer/muse-eeg-heads-src")
        if win_dir is None:
            raise FileNotFoundError("Attach windwerfer/muse-eeg-heads-windows")
        if cache_dir is None:
            log("WARN: muse-eeg-heads-cache not attached — REVE will fall back to HF hub + HF_TOKEN")

        SRC = WORKING / "src"
        SRC.mkdir(parents=True, exist_ok=True)
        for p in src_dir.glob("*.py"):
            shutil.copy2(p, SRC / p.name)
        sys.path.insert(0, str(WORKING))
        from src.head_a import ATTENTION_LABELS, HeadALinear, HeadAMLP, class_weights_from_y
        from src.metrics import per_class_report
        from src.reve_encoder import FrozenREVEEncoder, REVE_EMB
        from src.channel_map import MUSE4_MODEL_ORDER
        log(f"helpers ok; REVE_EMB={REVE_EMB} channels={MUSE4_MODEL_ORDER}")
        """
    ),
    md("## 2. Load QC-pass attention windows"),
    code(
        r"""
        SEED = 42
        EPOCHS = 5
        BATCH_EMB = 64
        BATCH_HEAD = 256
        LR = 1e-3
        MIN_TEST = 20

        def subject_for(stem: str) -> str:
            name = stem
            if name.endswith("_attention_windows"):
                name = name[: -len("_attention_windows")]
            if name.startswith("ds001787_"):
                rid = name[len("ds001787_") :]
                core = rid.split("_")[0]
                num = core.replace("sub", "")
                return f"ds001787/sub-{num.zfill(3) if num.isdigit() else num}"
            if name.startswith("ds003969_"):
                rid = name[len("ds003969_") :]
                num = rid.replace("sub", "")
                return f"ds003969/sub-{num.zfill(3) if num.isdigit() else num}"
            raise ValueError(stem)

        packs = []
        npz_paths = sorted(win_dir.glob("*_attention_windows.npz"))
        log(f"found {len(npz_paths)} attention window files")
        for npz_path in npz_paths:
            stem = npz_path.name.replace(".npz", "")
            data = np.load(npz_path, allow_pickle=True)
            X = data["X"].astype(np.float32)
            y_local = data["y"].astype(np.int64)
            names = [str(n) for n in data["label_names"].tolist()]
            table = {i: ATTENTION_LABELS.index(n) for i, n in enumerate(names) if n in ATTENTION_LABELS}
            mask = np.asarray([int(v) in table for v in y_local], dtype=bool)
            qc_path = win_dir / (stem.replace("_attention_windows", "_attention_qc") + ".npz")
            if qc_path.exists():
                mask &= np.load(qc_path)["qc_pass"].astype(bool)
            else:
                log(f"WARN missing qc for {stem}")
            if not mask.any():
                continue
            y = np.asarray([table[int(v)] for v in y_local[mask]], dtype=np.int64)
            sid = subject_for(stem)
            packs.append({"subject_id": sid, "stem": stem, "X": X[mask], "y": y, "n": int(mask.sum())})

        by_subj = {}
        for p in packs:
            by_subj.setdefault(p["subject_id"], []).append(p)
        subjects = sorted(by_subj)
        log(f"loaded {len(packs)} recordings / {len(subjects)} subjects; windows={sum(p['n'] for p in packs)}")

        raw_ids = {}
        for sid in subjects:
            corpus, raw = sid.split("/", 1)
            raw_ids.setdefault(raw, []).append(corpus)
        collisions = {k: v for k, v in raw_ids.items() if len(v) > 1}
        log(f"raw_id_collisions={len(collisions)}")

        fold_subjects, skipped = [], []
        for sid in subjects:
            y = np.concatenate([p["y"] for p in by_subj[sid]])
            counts = Counter(y.tolist())
            if len(y) < MIN_TEST:
                skipped.append({"subject": sid, "reason": "too_few_windows", "n": int(len(y))})
                continue
            if len(counts) < 2:
                skipped.append({"subject": sid, "reason": "single_class", "counts": {str(k): int(v) for k, v in counts.items()}})
                continue
            fold_subjects.append(sid)
        log(f"fold_subjects={len(fold_subjects)} skipped={len(skipped)}")
        (OUT / "fold_plan.json").write_text(json.dumps({
            "step": "reve_attention_loso",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_subjects_available": len(subjects),
            "n_folds": len(fold_subjects),
            "fold_subjects": fold_subjects,
            "skipped": skipped,
            "raw_id_collisions": collisions,
            "id_policy": "corpus-qualified (ds001787/sub-XXX | ds003969/sub-XXX)",
            "labels": list(ATTENTION_LABELS),
            "qc": "qc_pass windows only",
            "window_policy": "2.0s@256->400@200Hz (2 patches)",
            "encoder": "FrozenREVE + HeadALinear/MLP",
        }, indent=2) + "\n")
        """
    ),
    md("## 3. Load frozen REVE (gated) + embed once"),
    code(
        r"""
        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        montages = None
        for cand in [win_dir / "common" / "montages.json"]:
            if cand.exists():
                montages = cand
                break

        # Load order: private Kaggle cache models/reve-* → HF hub (HF_TOKEN fallback)
        local_model = None
        local_pos = None
        if cache_dir is not None:
            for cand in [
                cache_dir / "models" / "reve-base",
                cache_dir / "models" / "REVE" / "reve-base",
            ]:
                if (cand / "model.safetensors").exists() and (cand / "config.json").exists():
                    local_model = cand
                    break
            for cand in [
                cache_dir / "models" / "reve-positions",
                cache_dir / "models" / "REVE" / "reve-positions",
            ]:
                if (cand / "model.safetensors").exists() and (cand / "config.json").exists():
                    local_pos = cand
                    break
            log(f"reve local_model={local_model} local_pos={local_pos}")

        encoder = None
        try:
            encoder = FrozenREVEEncoder(
                source_sr=256.0,
                channel_names=MUSE4_MODEL_ORDER,
                pool="attention",
                hf_token=hf_token,
                montages_path=montages,
                device=device,
                local_model=local_model,
                local_positions=local_pos,
                prefer_local=True,
            )
            log(f"REVE loaded on {device}: {encoder.adapter_notes()}")
        except Exception as e:
            gated_error = traceback.format_exc()
            log("REVE LOAD FAILED — attach muse-eeg-heads-cache or enable HF_TOKEN")
            log(gated_error)
            summary = {
                "step": "reve_attention_loso",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "ship_candidate": False,
                "error": "reve_load_failed",
                "detail": str(e),
                "traceback": gated_error,
                "blocker": (
                    "Attach private windwerfer/muse-eeg-heads-cache (models/reve-base + reve-positions), "
                    "or accept https://huggingface.co/brain-bzh/reve-base with HF_TOKEN / Kaggle secret as fallback."
                ),
                "n_subjects_available": len(subjects),
                "n_folds": len(fold_subjects),
                "macro_f1_mean": None,
                "compare_cbramod_macro_f1": 0.36,
            }
            (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (WORKING / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            (OUT / "run.log").write_text("\n".join(LOG) + "\n")
            raise

        notes = encoder.adapter_notes()
        (OUT / "encoder_notes.json").write_text(json.dumps(notes, indent=2) + "\n")

        emb_by_subj = {}
        y_by_subj = {}
        torch.manual_seed(SEED)

        def encode_X(X, enc, device, batch=BATCH_EMB):
            outs = []
            enc.eval()
            with torch.no_grad():
                for i in range(0, len(X), batch):
                    xb = torch.from_numpy(X[i : i + batch]).to(device)
                    outs.append(enc(xb).float().cpu())
            return torch.cat(outs, dim=0)

        for i, sid in enumerate(subjects, 1):
            X = np.concatenate([p["X"] for p in by_subj[sid]])
            y = np.concatenate([p["y"] for p in by_subj[sid]])
            log(f"encode {i}/{len(subjects)} {sid} n={len(y)} shape={X.shape}")
            emb_by_subj[sid] = encode_X(X, encoder, device)
            y_by_subj[sid] = y
            for p in by_subj[sid]:
                p.pop("X", None)

        cache_path = OUT / "z_cache.pt"
        torch.save({
            "subjects": subjects,
            "emb": {s: emb_by_subj[s] for s in subjects},
            "y": {s: y_by_subj[s] for s in subjects},
            "encoder_notes": notes,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }, cache_path)
        log(f"wrote Z cache {cache_path} bytes={cache_path.stat().st_size}")
        embed_sec = time.time() - t0
        log(f"embed_elapsed_sec={embed_sec:.1f}")
        del encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        """
    ),
    md("## 4. Head-only LOSO (linear + MLP)"),
    code(
        r"""
        def train_eval_fold(emb_tr, y_tr, emb_te, y_te, device, rng, head_kind="linear"):
            idx0 = np.where(y_tr == 0)[0]
            idx1 = np.where(y_tr == 1)[0]
            n = min(len(idx0), len(idx1))
            if n == 0:
                return {
                    "n_train_balanced": 0,
                    "n_test": int(len(y_te)),
                    "test_counts": {ATTENTION_LABELS[i]: int((y_te == i).sum()) for i in range(2)},
                    "macro_f1": 0.0,
                    "accuracy": 0.0,
                    "per_class": {},
                    "skip_reason": "no both-class train",
                }
            pick0 = rng.choice(idx0, size=n, replace=False)
            pick1 = rng.choice(idx1, size=n, replace=False)
            bal = np.concatenate([pick0, pick1])
            rng.shuffle(bal)
            emb_b = emb_tr[bal]
            yb = y_tr[bal]
            in_dim = int(emb_b.shape[-1])
            if head_kind == "mlp":
                head = HeadAMLP(in_dim=in_dim, n_classes=2, hidden=128).to(device)
            else:
                head = HeadALinear(in_dim=in_dim, n_classes=2).to(device)
            w = class_weights_from_y(yb, n_classes=2).to(device)
            crit = nn.CrossEntropyLoss(weight=w)
            opt = torch.optim.Adam(head.parameters(), lr=LR)
            loader = DataLoader(TensorDataset(emb_b, torch.from_numpy(yb)), batch_size=BATCH_HEAD, shuffle=True)
            for _ in range(EPOCHS):
                head.train()
                for xb, ybatch in loader:
                    opt.zero_grad()
                    loss = crit(head(xb.to(device)), ybatch.to(device))
                    loss.backward()
                    opt.step()
            head.eval()
            with torch.no_grad():
                pred = head(emb_te.to(device)).argmax(dim=-1).cpu().numpy()
            report = per_class_report(y_te.tolist(), pred.tolist(), ATTENTION_LABELS)
            return {
                "n_train_balanced": int(len(yb)),
                "n_test": int(len(y_te)),
                "test_counts": {ATTENTION_LABELS[i]: int((y_te == i).sum()) for i in range(2)},
                "macro_f1": float(report["macro_f1"]["f1"]),
                "accuracy": float((pred == y_te).mean()) if len(y_te) else 0.0,
                "per_class": {k: v for k, v in report.items() if k != "macro_f1"},
                "head": head_kind,
                "in_dim": in_dim,
            }

        t_loso0 = time.time()
        results = {}
        for head_kind in ("linear", "mlp"):
            rng = np.random.default_rng(SEED)
            folds = []
            for i, sid in enumerate(fold_subjects, 1):
                emb_te = emb_by_subj[sid]
                y_te = y_by_subj[sid]
                tr_sids = [s for s in subjects if s != sid]
                emb_tr = torch.cat([emb_by_subj[s] for s in tr_sids], dim=0)
                y_tr = np.concatenate([y_by_subj[s] for s in tr_sids])
                m = train_eval_fold(emb_tr, y_tr, emb_te, y_te, device, rng, head_kind=head_kind)
                m["holdout_subject"] = sid
                folds.append(m)
                if i % 10 == 0 or i == 1 or i == len(fold_subjects):
                    log(f"{head_kind} fold {i}/{len(fold_subjects)} holdout={sid} f1={m['macro_f1']:.3f}")
            f1s = [f["macro_f1"] for f in folds]
            mean_f1 = float(np.mean(f1s)) if f1s else None
            std_f1 = float(np.std(f1s)) if f1s else None
            results[head_kind] = {
                "n_folds": len(folds),
                "macro_f1_mean": mean_f1,
                "macro_f1_std": std_f1,
                "accuracy_mean": float(np.mean([f["accuracy"] for f in folds])) if folds else None,
                "folds_gt_0_5": int(sum(1 for f in f1s if f > 0.5)),
                "folds": folds,
            }
            log(f"{head_kind}: macro_f1={mean_f1:.4f}+/-{std_f1:.4f}")

        loso_sec = time.time() - t_loso0
        best = max(results, key=lambda k: (results[k]["macro_f1_mean"] or -1))
        best_f1 = results[best]["macro_f1_mean"]
        best_std = results[best]["macro_f1_std"] or 0.0
        ship = bool(best_f1 is not None and best_f1 > 0.60 and (best_f1 - best_std) > 0.55)

        heads_compact = {}
        for k, v in results.items():
            heads_compact[k] = {
                "n_folds": v["n_folds"],
                "macro_f1_mean": v["macro_f1_mean"],
                "macro_f1_std": v["macro_f1_std"],
                "accuracy_mean": v["accuracy_mean"],
                "folds_gt_0_5": v["folds_gt_0_5"],
            }

        summary = {
            "step": "reve_attention_loso",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "ship_candidate": ship,
            "n_subjects_available": len(subjects),
            "n_folds": len(fold_subjects),
            "skipped": skipped,
            "raw_id_collisions": collisions,
            "best_head": best,
            "macro_f1_mean": best_f1,
            "macro_f1_std": best_std,
            "accuracy_mean": results[best]["accuracy_mean"],
            "heads": heads_compact,
            "folds_linear": results["linear"]["folds"],
            "folds_mlp": results["mlp"]["folds"],
            "compare_cbramod_macro_f1": 0.36098,
            "delta_vs_cbramod": (best_f1 - 0.36098) if best_f1 is not None else None,
            "encoder": notes,
            "timing_sec": {"embed": embed_sec, "loso_heads": loso_sec, "total": time.time() - t0},
            "qc": "qc_pass only",
            "labels": list(ATTENTION_LABELS),
            "chance_note": "Binary chance approx 0.5 macro-F1 if balanced; treat <=chance as not shippable.",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
        (OUT / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (WORKING / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        lines_out = []
        for f in results["linear"]["folds"]:
            lines_out.append(json.dumps({"head": "linear", "holdout": f["holdout_subject"], "macro_f1": f["macro_f1"], "accuracy": f["accuracy"], "n_test": f["n_test"]}))
        for f in results["mlp"]["folds"]:
            lines_out.append(json.dumps({"head": "mlp", "holdout": f["holdout_subject"], "macro_f1": f["macro_f1"], "accuracy": f["accuracy"], "n_test": f["n_test"]}))
        (OUT / "folds.jsonl").write_text("\n".join(lines_out) + "\n")
        (OUT / "run.log").write_text("\n".join(LOG) + "\n")
        log(f"DONE best={best} macro_f1={best_f1:.4f}+/-{best_std:.4f} ship={ship} delta_vs_cbramod={summary['delta_vs_cbramod']}")
        print(json.dumps({k: summary[k] for k in ("ship_candidate", "macro_f1_mean", "macro_f1_std", "best_head", "n_folds", "delta_vs_cbramod", "timing_sec")}, indent=2))
        """
    ),
]


def main() -> None:
    KDIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "accelerator": "GPU",
        },
        "cells": CELLS,
    }
    NB.parent.mkdir(parents=True, exist_ok=True)
    NB.write_text(json.dumps(nb, indent=1) + "\n")
    shutil.copy2(NB, KDIR / CODE)
    meta = {
        "id": "windwerfer/muse-eeg-heads-reve-attention-loso",
        "title": "Muse EEG Heads REVE Attention LOSO",
        "code_file": CODE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["eeg", "reve", "loso"],
        "dataset_sources": [
            "windwerfer/muse-eeg-heads-src",
            "windwerfer/muse-eeg-heads-windows",
            "windwerfer/muse-eeg-heads-cache",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (KDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {NB}")
    print(f"wrote {KDIR / CODE}")
    print(f"cells={len(CELLS)}")


if __name__ == "__main__":
    main()

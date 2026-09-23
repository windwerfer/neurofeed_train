#!/usr/bin/env python3
"""Build a ship-ready CBraMod + Head A export pack for flutter_muse-rs_ml.

Pins expected frozen encoder SHA256; packages head-only checkpoints + labels +
attribution. Does not re-upload encoder weights (already in private cache);
documents Apache-2.0 fetch URL for redistributable pin.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENCODER_PATH = (
    ROOT
    / "kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth"
)
ENCODER_HF = "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth"
ENCODER_CODE = "https://github.com/wjq-learning/CBraMod"
OUT = ROOT / "exports/cbramod_pack"
DOCS = ROOT / "docs/export_cbramod_packaging.md"

MUSE_CHANNELS = ["AF7", "AF8", "TP9", "TP10"]
HEAD_A_BINARY = ["drowsy", "hypnagogic"]
HEAD_A_4WAY = ["concentration", "mind_wandering", "drowsy", "hypnagogic"]

HEAD_ARTIFACTS = [
    {
        "role": "binary_single_night_baseline",
        "src": ROOT / "exports/head_a_holdout/head_a_binary_state_dict.pt",
        "filename": "head_a_binary_sc4001.pt",
        "n_classes": 2,
        "label_list": HEAD_A_BINARY,
        "recommended_default": False,
        "notes": "SC4001 train → SC4002 holdout binary HeadALinear; raw logits, thresh 0.5",
    },
    {
        "role": "binary_precision_tuned",
        "src": ROOT / "exports/head_a_holdout/head_a_binary_precision_tuned.pt",
        "filename": "head_a_binary_precision_tuned.pt",
        "n_classes": 2,
        "label_list": HEAD_A_BINARY,
        "recommended_default": True,
        "decision_threshold_hypnagogic": 0.53,
        "notes": "Same weights family as baseline; preferred binary ship head (hypnagogic P↑)",
    },
    {
        "role": "binary_multi_night",
        "src": ROOT / "exports/head_a_multi_night/head_a_binary_multi_night.pt",
        "filename": "head_a_binary_multi_night.pt",
        "n_classes": 2,
        "label_list": HEAD_A_BINARY,
        "recommended_default": False,
        "notes": "Train SC4001+SC4011; did not beat single-night all-labeled macro-F1",
    },
    {
        "role": "four_way_masked_smoke",
        "src": ROOT / "exports/head_a_4way_scaffold/head_a_4way_masked_smoke.pt",
        "filename": "head_a_4way_masked_smoke.pt",
        "n_classes": 4,
        "label_list": HEAD_A_4WAY,
        "recommended_default": False,
        "notes": "Scaffold only: attention classes masked/untrained; not for production yet",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> int:
    now = datetime.now(timezone.utc).isoformat()
    if not ENCODER_PATH.is_file():
        raise SystemExit(f"missing encoder weights: {ENCODER_PATH}")
    enc_sha = sha256(ENCODER_PATH)
    expected = "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178"
    if enc_sha != expected:
        raise SystemExit(f"encoder sha mismatch: got {enc_sha}, expected {expected}")

    if OUT.exists():
        shutil.rmtree(OUT)
    heads_dir = OUT / "heads"
    heads_dir.mkdir(parents=True)
    (OUT / "encoder").mkdir()

    packed_heads: List[Dict[str, Any]] = []
    for spec in HEAD_ARTIFACTS:
        src: Path = spec["src"]
        if not src.is_file():
            raise SystemExit(f"missing head artifact: {src}")
        dest = heads_dir / spec["filename"]
        shutil.copy2(src, dest)
        entry = {
            "role": spec["role"],
            "filename": spec["filename"],
            "path": f"heads/{spec['filename']}",
            "sha256": sha256(dest),
            "bytes": dest.stat().st_size,
            "n_classes": spec["n_classes"],
            "label_list": list(spec["label_list"]),
            "in_dim": 200,
            "head_arch": "HeadALinear",
            "recommended_default": bool(spec["recommended_default"]),
            "notes": spec["notes"],
            "source_rel": str(src.relative_to(ROOT)),
        }
        if "decision_threshold_hypnagogic" in spec:
            entry["decision_threshold_hypnagogic"] = spec[
                "decision_threshold_hypnagogic"
            ]
        packed_heads.append(entry)

    labels = {
        "muse_channels_order": MUSE_CHANNELS,
        "head_a_binary": HEAD_A_BINARY,
        "head_a_4way": HEAD_A_4WAY,
        "head_b_deferred": [
            "blink",
            "double_blink",
            "jaw",
            "double_jaw",
            "clean",
        ],
        "sleep_edf_head_a_map": {
            "Sleep stage W": "drowsy",
            "Sleep stage 1": "hypnagogic",
        },
        "units": "microvolts",
        "window_sec": 2.0,
        "input_sr_hz": 256.0,
        "encoder_native_sr_hz": 200.0,
        "encoder_patch_samples": 200,
        "tensor_layout_after_adapter": "(B, 4, n_patches, 200)",
        "embedding_dim": 200,
        "pool": "mean over C x patches",
    }
    write_json(OUT / "labels.json", labels)
    write_json(
        OUT / "channels.json",
        {
            "order": MUSE_CHANNELS,
            "sleep_edf_proxy": {
                "AF7": "Fpz-Cz",
                "AF8": "Fpz-Cz",
                "TP9": "Pz-Oz",
                "TP10": "Pz-Oz",
                "note": "Proxy only; replace with true Muse for domain adaptation",
            },
        },
    )

    encoder_pin = {
        "encoder_name": "CBraMod",
        "weights_filename": "pretrained_weights.pth",
        "weights_sha256": enc_sha,
        "weights_bytes": ENCODER_PATH.stat().st_size,
        "license": "Apache-2.0",
        "hf_repo": "weighting666/CBraMod",
        "hf_url": ENCODER_HF,
        "code_url": ENCODER_CODE,
        "paper": "Wang et al., CBraMod, ICLR 2025 / arXiv:2412.07236",
        "local_cache_rel": str(ENCODER_PATH.relative_to(ROOT)),
        "ship_policy": (
            "Pin checksum in app; fetch Apache-2.0 weights from HF or private "
            "muse-eeg-heads-cache. Default publish ships heads + this pin; "
            "encoder blob may be redistributed under Apache-2.0 if desired."
        ),
        "do_not_ship": ["REVE-base gated weights", "LUNA", "BY-NC training mix"],
        "frozen": True,
        "adapter": {
            "fed_input": "(B, 4, T) @ 256 Hz",
            "native_input": "(B, C, n_patches, 200) @ 200 Hz",
            "channel_policy": "keep Muse 4-ch; no fabricated 10-20 montage",
            "pool": "mean",
        },
    }
    write_json(OUT / "encoder/EXPECTED.json", encoder_pin)
    (OUT / "encoder/README.md").write_text(
        "\n".join(
            [
                "# CBraMod encoder pin",
                "",
                f"- SHA256: `{enc_sha}`",
                f"- Fetch: {ENCODER_HF}",
                f"- License: Apache-2.0",
                f"- Local cache (dev): `{ENCODER_PATH.relative_to(ROOT)}`",
                "",
                "App load path: verify SHA256 before `load_state_dict`.",
                "Do not ship gated REVE. CBraMod may be redistributed under Apache-2.0.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    attribution = "\n".join(
        [
            "# Attribution — CBraMod Head A pack",
            "",
            "## Encoder",
            "- CBraMod pretrained weights — Apache-2.0 — weighting666/CBraMod (HF)",
            "- Code — Apache-2.0 — wjq-learning/CBraMod",
            "- Paper — Wang et al., CBraMod, ICLR 2025 / arXiv:2412.07236",
            "",
            "## Training data (derived heads only)",
            "- Sleep-EDF Expanded (PhysioNet) — attribute; W→drowsy, N1→hypnagogic",
            "- Nights used in current heads: SC4001, SC4002, SC4011 (± SC4031 holdout)",
            "- Open-license policy: no LUNA / L-FAME / SEED-VIG / BY-NC in shipping mix",
            "",
            "## Channels",
            "- Muse order: AF7, AF8, TP9, TP10",
            "- Sleep-EDF proxy montage documented in channels.json",
            "",
            "## Heads",
            "- Head-only Linear classifiers (in_dim=200); MIT/BSD-preferred release posture",
            "- Recommended binary default: head_a_binary_precision_tuned.pt (thresh 0.53)",
            "- 4-way scaffold is not production-ready (attention classes untrained)",
            "",
            f"Pack built: {now}",
            "",
        ]
    )
    (OUT / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")

    # Flutter / app integration stub
    app_card = {
        "package": "cbramod_head_a_v0",
        "encoder_family": "CBraMod",
        "encoder_sha256": enc_sha,
        "recommended_head": "heads/head_a_binary_precision_tuned.pt",
        "decision_threshold_hypnagogic": 0.53,
        "muse_channels": MUSE_CHANNELS,
        "embedding_dim": 200,
        "window_sec": 2.0,
        "input_sr_hz": 256.0,
        "status": "vigilance_binary_ship_candidate",
        "not_ready": [
            "head_a_4way (attention untrained)",
            "head_b artifacts (deferred)",
            "REVE family (gated; separate pin)",
        ],
        "verify_steps": [
            "sha256sum encoder pretrained_weights.pth == EXPECTED.json",
            "load HeadALinear(in_dim=200, n_classes=2) from recommended_head",
            "softmax; if P(hypnagogic) >= 0.53 predict hypnagogic else drowsy",
        ],
    }
    write_json(OUT / "app_integration.json", app_card)

    pack_manifest = {
        "step": "export_cbramod_packaging",
        "created_utc": now,
        "pack_root": str(OUT.relative_to(ROOT)),
        "encoder": encoder_pin,
        "heads": packed_heads,
        "recommended_default_head": "heads/head_a_binary_precision_tuned.pt",
        "labels_path": "labels.json",
        "channels_path": "channels.json",
        "attribution_path": "ATTRIBUTION.md",
        "app_integration_path": "app_integration.json",
        "kaggle_private_encoder_cache": "windwerfer/muse-eeg-heads-cache",
        "ship_notes": [
            "Heads only in this pack directory (~15 KB total).",
            "Encoder pin via SHA256; fetch from HF or private cache (19 MB).",
            "CBraMod Apache-2.0 allows optional encoder redistribution; REVE does not.",
            "Binary vigilance is ship candidate; 4-way and Head B are not.",
        ],
        "headline_metrics_ref": {
            "precision_tuned_sc4002_macro_f1": 0.7552,
            "single_night_sc4002_macro_f1": 0.7388,
            "docs": "docs/wave2_writeup.md",
        },
    }
    write_json(OUT / "pack_manifest.json", pack_manifest)

    # Docs
    docs = f"""# CBraMod export packaging

Built `{now}` for Muse AF7/AF8/TP9/TP10 + frozen CBraMod Spur A.

## What this pack is

Ship-ready **head-only** bundle under `exports/cbramod_pack/`:

| Piece | Role |
|-------|------|
| `encoder/EXPECTED.json` | Pinned CBraMod `pretrained_weights.pth` SHA256 + HF URL |
| `heads/*.pt` | Head A Linear checkpoints (binary + 4-way scaffold) |
| `labels.json` / `channels.json` | Label maps + Muse channel order |
| `ATTRIBUTION.md` | Datasets + encoder licenses |
| `app_integration.json` | flutter_muse-rs_ml load recipe |
| `pack_manifest.json` | Full checksum inventory |

**Recommended default head:** `heads/head_a_binary_precision_tuned.pt`  
Hypnagogic decision threshold: **0.53** (not 0.5).

Encoder SHA256 (must match):

```
{enc_sha}
```

Fetch: `{ENCODER_HF}` (Apache-2.0). Local/dev cache: private Kaggle `windwerfer/muse-eeg-heads-cache`.

## App load sketch

1. Ensure Muse window is `(4, T)` @ 256 Hz, order AF7/AF8/TP9/TP10, ~2 s → T=512.
2. Resample → 200 Hz patches `(1, 4, 2, 200)`; freeze CBraMod; mean-pool → `(1, 200)`.
3. `HeadALinear(200, 2)` ← recommended binary head; softmax.
4. If `P(hypnagogic) >= 0.53` → hypnagogic else drowsy.
5. Refuse load if encoder SHA256 ≠ pin.

## Honest limits

- Sleep-EDF Fpz-Cz/Pz-Oz **proxy** montage ≠ true Muse; treat as vigilance transfer probe.
- 4-way head is **scaffold only** (concentration / mind_wandering untrained).
- Head B artifacts deferred (personal Muse cal gold).
- Do **not** ship gated REVE; separate encoder family + checksum if ever used.
- No LUNA / L-FAME / SEED-VIG / BY-NC in shipping mix.

## Rebuild

```bash
cd /workspace/muse-eeg-heads
.venv/bin/python scripts/export_cbramod_packaging.py
```

## Artifacts

- Pack: `exports/cbramod_pack/`
- Script: `scripts/export_cbramod_packaging.py`
- Policy: `docs/LICENSE_NOTES.md`, `docs/cbramod_notes.md`
"""
    DOCS.write_text(docs, encoding="utf-8")

    summary = {
        "step": "export_cbramod_packaging",
        "completed_utc": now,
        "encoder_sha256": enc_sha,
        "n_heads": len(packed_heads),
        "recommended_default_head": "heads/head_a_binary_precision_tuned.pt",
        "decision_threshold_hypnagogic": 0.53,
        "pack_bytes_heads_only": sum(h["bytes"] for h in packed_heads),
        "encoder_bytes_not_copied": ENCODER_PATH.stat().st_size,
        "artifacts": {
            "pack": str(OUT.relative_to(ROOT)),
            "docs": str(DOCS.relative_to(ROOT)),
            "script": "scripts/export_cbramod_packaging.py",
            "manifest": "exports/cbramod_pack/pack_manifest.json",
        },
    }
    write_json(ROOT / "exports/export_cbramod_packaging_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

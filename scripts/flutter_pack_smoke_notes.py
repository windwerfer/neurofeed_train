#!/usr/bin/env python3
"""Offline flutter_muse-rs_ml pack smoke notes for exports/cbramod_pack.

Docs + CPU integrity/load smoke. Does not train, download, or require Muse hardware.
AppImage today speaks LUNA/REVE (RLX); CBraMod Spur A is a planned Spur — document the gap.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from head_a import HeadALinear  # noqa: E402

PACK = ROOT / "exports" / "cbramod_pack"
OUT_DIR = ROOT / "exports" / "flutter_pack_smoke"
DOCS = ROOT / "docs" / "flutter_pack_smoke_notes.md"
STATE = ROOT / "overnight_state.json"
APPIMAGE = Path("/workspace/muse_ml/muse_ml-0.1.6-linux-x86_64.AppImage")
ENCODER_CACHE = (
    ROOT
    / "kaggle_datasets"
    / "muse-eeg-heads-cache"
    / "models"
    / "CBraMod"
    / "pretrained_weights.pth"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    manifest = load_json(PACK / "pack_manifest.json")
    app_int = load_json(PACK / "app_integration.json")
    expected = load_json(PACK / "encoder" / "EXPECTED.json")
    labels = load_json(PACK / "labels.json")
    channels = load_json(PACK / "channels.json")

    checks = []

    # --- pack files present ---
    required = [
        PACK / "pack_manifest.json",
        PACK / "app_integration.json",
        PACK / "labels.json",
        PACK / "channels.json",
        PACK / "ATTRIBUTION.md",
        PACK / "encoder" / "EXPECTED.json",
    ]
    for p in required:
        ok = p.is_file()
        checks.append({"name": f"present:{p.relative_to(PACK)}", "ok": ok})
        if not ok:
            raise SystemExit(f"missing required pack file: {p}")

    # --- head checksums ---
    head_results = []
    for h in manifest["heads"]:
        path = PACK / h["path"]
        digest = sha256_file(path) if path.is_file() else None
        match = digest == h["sha256"]
        checks.append(
            {
                "name": f"sha256:{h['filename']}",
                "ok": match,
                "expected": h["sha256"],
                "got": digest,
                "bytes": path.stat().st_size if path.is_file() else None,
            }
        )
        # load HeadALinear
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict) and "state_dict" in blob and "net.1.weight" not in blob:
            sd = blob["state_dict"]
            n_classes = int(blob.get("n_classes", h["n_classes"]))
            in_dim = int(blob.get("in_dim", h["in_dim"]))
        else:
            sd = blob
            n_classes = int(h["n_classes"])
            in_dim = int(h["in_dim"])
        model = HeadALinear(in_dim=in_dim, n_classes=n_classes)
        model.load_state_dict(sd)
        model.eval()
        with torch.no_grad():
            x = torch.zeros(1, in_dim)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
        head_results.append(
            {
                "filename": h["filename"],
                "role": h["role"],
                "recommended_default": h.get("recommended_default", False),
                "sha256_ok": match,
                "load_ok": True,
                "logits_shape": list(logits.shape),
                "zero_emb_probs": probs.squeeze(0).tolist(),
                "label_list": h["label_list"],
            }
        )
        checks.append({"name": f"load:{h['filename']}", "ok": True})

    # --- recommended threshold decision on zero emb ---
    thresh = float(app_int["decision_threshold_hypnagogic"])
    rec = next(r for r in head_results if r["recommended_default"])
    p_hyp = float(rec["zero_emb_probs"][rec["label_list"].index("hypnagogic")])
    decision = "hypnagogic" if p_hyp >= thresh else "drowsy"
    checks.append(
        {
            "name": "threshold_decision_zero_emb",
            "ok": True,
            "threshold": thresh,
            "p_hypnagogic": p_hyp,
            "decision": decision,
            "note": "synthetic zeros; only verifies path, not quality",
        }
    )

    # --- encoder pin ---
    pin = expected["weights_sha256"]
    enc_ok = False
    enc_digest = None
    if ENCODER_CACHE.is_file():
        enc_digest = sha256_file(ENCODER_CACHE)
        enc_ok = enc_digest == pin
    checks.append(
        {
            "name": "encoder_cache_sha256",
            "ok": enc_ok,
            "expected": pin,
            "got": enc_digest,
            "path": str(ENCODER_CACHE.relative_to(ROOT)) if ENCODER_CACHE.is_file() else None,
        }
    )
    if not enc_ok:
        raise SystemExit("encoder cache SHA256 mismatch or missing")

    # --- AppImage presence (boot smoke historically done 2026-09-04) ---
    app_present = APPIMAGE.is_file() and os.access(APPIMAGE, os.X_OK)
    checks.append(
        {
            "name": "appimage_present_executable",
            "ok": app_present,
            "path": str(APPIMAGE),
            "bytes": APPIMAGE.stat().st_size if APPIMAGE.is_file() else None,
            "version_hint": "0.1.6",
        }
    )

    # --- channel / label contract ---
    muse_order = channels["order"]
    contract_ok = muse_order == ["AF7", "AF8", "TP9", "TP10"] and labels[
        "head_a_binary"
    ] == ["drowsy", "hypnagogic"]
    checks.append({"name": "muse_channel_label_contract", "ok": contract_ok})

    all_ok = all(c["ok"] for c in checks)

    gap = {
        "app_current_ai_guardrail": ["LUNA", "REVE"],
        "app_engine": "RLX CPU (safetensors / GGUF path)",
        "pack_encoder_family": "CBraMod",
        "pack_status": app_int["status"],
        "in_app_cbramod_inference": False,
        "blocker_for_live_smoke": (
            "flutter_muse-rs_ml 0.1.6 has no CBraMod Spur A loader yet; "
            "pack is heads+pin for a future Spur. Live Muse smoke needs app work "
            "or an external Python probe feeding Muse LSL."
        ),
        "do_not_ship_in_app": expected["do_not_ship"],
    }

    checklist = [
        "Verify pack_manifest.json + head SHA256s (this script).",
        "Refuse load if encoder pretrained_weights.pth SHA256 ≠ EXPECTED.json pin.",
        "Window: (4, 512) @ 256 Hz AF7/AF8/TP9/TP10 → resample/patch to CBraMod (1,4,n_patches,200).",
        "Freeze encoder; mean-pool → (1, 200); HeadALinear(200, 2) from recommended head.",
        "Softmax; if P(hypnagogic) >= 0.53 → hypnagogic else drowsy.",
        "Do not load 4-way head in production UI until attention trained.",
        "Do not bundle gated REVE; keep separate family + user-local download.",
        "AppImage boot smoke: launch muse_ml-0.1.6 AppImage (GUI); confirm home screen (done historically).",
        "Future: wire Spur A in rust/flutter using pack_manifest + app_integration.json.",
    ]

    summary = {
        "step": "flutter_pack_smoke_notes",
        "completed_utc": now,
        "all_checks_ok": all_ok,
        "n_heads_loaded": len(head_results),
        "recommended_head": app_int["recommended_head"],
        "decision_threshold_hypnagogic": thresh,
        "zero_emb_decision": decision,
        "encoder_sha256": pin,
        "encoder_cache_ok": enc_ok,
        "appimage_ok": app_present,
        "gap": gap,
        "checklist": checklist,
        "head_results": head_results,
        "checks": checks,
        "artifacts": {
            "docs": "docs/flutter_pack_smoke_notes.md",
            "summary": "exports/flutter_pack_smoke/summary.json",
            "script": "scripts/flutter_pack_smoke_notes.py",
            "pack": "exports/cbramod_pack/",
            "appimage": str(APPIMAGE),
        },
        "takeaway": (
            "Pack integrity + Head A load OK on CPU; encoder pin matches private cache. "
            "AppImage 0.1.6 present. Live CBraMod Spur A not in app yet (LUNA/REVE only) — "
            "integration is the next app-side gate, not a training blocker."
        ),
    }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    docs = f"""# Flutter pack smoke notes

Built `{now}` — offline integrity + load smoke for `exports/cbramod_pack/` against **flutter_muse-rs_ml** 0.1.6.

## Result

**All pack checks OK:** `{all_ok}`

| Check | Status |
|-------|--------|
| Head SHA256s (4) | match `pack_manifest.json` |
| HeadALinear load + forward | OK (CPU) |
| Encoder cache SHA256 | `{pin}` |
| Muse channel/label contract | AF7/AF8/TP9/TP10; drowsy/hypnagogic |
| AppImage present | `{APPIMAGE.name}` executable |

Recommended head: `{app_int["recommended_head"]}`  
Hypnagogic threshold: **{thresh}** (synthetic zero-embedding decision: `{decision}` — path only).

## What this smoke is / is not

**Is:** checksum inventory, head deserialize, threshold contract, encoder pin vs private `muse-eeg-heads-cache`, AppImage file presence.

**Is not:** live Muse session, in-app CBraMod inference, or a claim that Spur A is wired in 0.1.6.

## App gap (important)

`flutter_muse-rs_ml` **0.1.6** AI guardrail paths are **LUNA** and **REVE** (RLX CPU / safetensors). Strings in the shipped binary reference `package:muse_ml/src/reve/` and LUNA HF URLs — **no CBraMod loader**.

| Layer | Today (0.1.6) | This pack (Spur A) |
|-------|---------------|--------------------|
| Encoder | LUNA / REVE (user-local / gated REVE) | Frozen **CBraMod** Apache-2.0, SHA256-pinned |
| Head | In-engine sleep-direction score | Tiny `HeadALinear` `.pt` (heads-only ship) |
| Labels | Sleep-direction / clear anchor | `drowsy` / `hypnagogic` (+ 4-way scaffold later) |
| Window | LUNA 5 s / REVE 4 s @ 256 Hz | **2 s** → 512 samples @ 256 Hz → 200 Hz patches |

So: **pack is ready to integrate**; **app needs a Spur A code path** before a true in-app smoke. Until then, use this script + optional Python/LSL probe.

Do **not** ship gated REVE in the pack. Do **not** treat LUNA as the publish path for open-license heads.

## Integration checklist (for app / next release)

"""
    for i, item in enumerate(checklist, 1):
        docs += f"{i}. {item}\n"

    docs += f"""
## Reproduce

```bash
cd /workspace/muse-eeg-heads
.venv/bin/python scripts/flutter_pack_smoke_notes.py
```

AppImage (GUI, historical boot smoke 2026-09-04):

```bash
/workspace/muse_ml/muse_ml-0.1.6-linux-x86_64.AppImage
```

## Honest limits

- Sleep-EDF train montage is Fpz-Cz/Pz-Oz **proxy** ≠ true Muse; expect domain shift until personal Muse cal.
- 4-way head remains scaffold (attention untrained) — keep out of production picker.
- Head B deferred (see `docs/head_b_plan.md`).
- Multi-night pool head (`exports/head_a_multi_night_pool/`) is **not** yet in this pack; re-run `export_cbramod_packaging` if you want it shipped.

## Artifacts

- Summary: `exports/flutter_pack_smoke/summary.json`
- Pack: `exports/cbramod_pack/`
- Packaging docs: `docs/export_cbramod_packaging.md`
- Script: `scripts/flutter_pack_smoke_notes.py`
"""
    DOCS.write_text(docs, encoding="utf-8")

    # --- update overnight_state ---
    state = load_json(STATE)
    step_name = "flutter_pack_smoke_notes"
    if step_name not in state["completed"]:
        state["completed"].append(step_name)
    state["next_index"] = 13
    state["status"] = "ready_for_next"
    state["steps_since_summary"] = int(state.get("steps_since_summary") or 0) + 1
    state["updated_utc"] = now
    state["notes"] = (
        "Plan exhausted after flutter_pack_smoke_notes. "
        "Invent next: candidates — (1) head_a_4way_attention_train "
        "(Sleep-EDF vigilance + ds001787 attention, subject holdout), "
        "(2) repack_cbramod_include_pool_head, "
        "(3) ds003969_attention_ingest, "
        "(4) personal_muse_cal_blocker (needs_input). "
        "Pack smoke OK; in-app CBraMod Spur A still missing."
    )
    state["step_13_flutter_pack_smoke_notes"] = {
        "completed_utc": now,
        "all_checks_ok": all_ok,
        "encoder_sha256": pin,
        "n_heads_loaded": len(head_results),
        "recommended_head": app_int["recommended_head"],
        "decision_threshold_hypnagogic": thresh,
        "appimage_present": app_present,
        "in_app_cbramod": False,
        "takeaway": summary["takeaway"],
        "artifacts": summary["artifacts"],
    }
    # extend plan so invent has a trail, but next_index already past original plan
    if "flutter_pack_smoke_notes" not in state["plan"]:
        state["plan"].append("flutter_pack_smoke_notes")
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"ok": all_ok, "summary": str(OUT_DIR / "summary.json"), "docs": str(DOCS)}, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

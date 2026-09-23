# Flutter pack smoke notes

Built `2026-09-07T02:27:24.000438+00:00` — offline integrity + load smoke for `exports/cbramod_pack/` against **flutter_muse-rs_ml** 0.1.6.

## Result

**All pack checks OK:** `True`

| Check | Status |
|-------|--------|
| Head SHA256s (4) | match `pack_manifest.json` |
| HeadALinear load + forward | OK (CPU) |
| Encoder cache SHA256 | `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178` |
| Muse channel/label contract | AF7/AF8/TP9/TP10; drowsy/hypnagogic |
| AppImage present | `muse_ml-0.1.6-linux-x86_64.AppImage` executable |

Recommended head: `heads/head_a_binary_precision_tuned.pt`  
Hypnagogic threshold: **0.53** (synthetic zero-embedding decision: `drowsy` — path only).

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

1. Verify pack_manifest.json + head SHA256s (this script).
2. Refuse load if encoder pretrained_weights.pth SHA256 ≠ EXPECTED.json pin.
3. Window: (4, 512) @ 256 Hz AF7/AF8/TP9/TP10 → resample/patch to CBraMod (1,4,n_patches,200).
4. Freeze encoder; mean-pool → (1, 200); HeadALinear(200, 2) from recommended head.
5. Softmax; if P(hypnagogic) >= 0.53 → hypnagogic else drowsy.
6. Do not load 4-way head in production UI until attention trained.
7. Do not bundle gated REVE; keep separate family + user-local download.
8. AppImage boot smoke: launch muse_ml-0.1.6 AppImage (GUI); confirm home screen (done historically).
9. Future: wire Spur A in rust/flutter using pack_manifest + app_integration.json.

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

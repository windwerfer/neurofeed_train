# CBraMod export packaging

Built `2026-09-07T01:39:28.857546+00:00` for Muse AF7/AF8/TP9/TP10 + frozen CBraMod Spur A.

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
0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178
```

Fetch: `https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth` (Apache-2.0). Local/dev cache: private Kaggle `windwerfer/muse-eeg-heads-cache`.

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

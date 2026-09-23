# Montages: Muse 4-ch + Crown 8-ch (dual Head A/B/C packs)

Source: user (flutter_muse-rs_ml experimental crown; 2026-09-07). Applies to **REVE** training details and, in spirit, **CBraMod / LaBraM** (consistent rate/window; never mix devices in one example).

## Policy

| Pack | Montage ID | Channels | Heads |
|------|------------|----------|-------|
| Default ship path today | `muse4` | 4 | Head A/B/C_muse4 |
| Second pack | `crown8` | 8 | Head A/B/C_crown8 |

- **Never** mix Muse and Crown channels in the same training example.
- Train **one head family per montage** (or a clearly documented multi-montage head — not the default).
- Tag every window / manifest with `montage_id`, `channel_names`, `sample_rate_hz`, `units`.
- Sleep-EDF proxy stays **`muse4` only** (Fpz-Cz/Pz-Oz → 4 slots). Do **not** invent Crown channels from Sleep-EDF.

## Muse (what the app streams)

Stream order @ **256 Hz**, µV:

| idx | name | where |
|-----|------|--------|
| 0 | TP9 | left mastoid / rear |
| 1 | AF7 | left forehead |
| 2 | AF8 | right forehead |
| 3 | TP10 | right mastoid / rear |

- Ref/DRL: forehead. Classic Muse does **not** emit FPz as data (reference). Athena can emit FPz + AUX — app still keeps four above.
- Signal gate / reward pads: **AF7 + AF8**.
- Live REVE path remaps to **AF7, AF8, TP9, TP10** row order (not stream order), 4 s = 1024 @ 256 Hz — see App vs train table.

### Official REVE position bank (meters; stream order TP9, AF7, AF8, TP10)

```
TP9  [-0.08562, -0.04651, -0.04571]
AF7  [-0.05484,  0.06857, -0.01059]
AF8  [ 0.05574,  0.06966, -0.01075]
TP10 [ 0.08616, -0.04704, -0.04587]
```

## Crown (Neurosity)

Stream order @ **256 Hz**, µV (dry Ag/AgCl):

| idx | name | region |
|-----|------|--------|
| 0 | CP3 | left centro-parietal |
| 1 | C3 | left motor |
| 2 | F5 | left frontal |
| 3 | PO3 | left parieto-occipital |
| 4 | PO4 | right parieto-occipital |
| 5 | F6 | right frontal |
| 6 | C4 | right motor |
| 7 | CP4 | right centro-parietal |

- Ref/bias: **T7, T8** (not in the 8-channel stream). Do not put T7/T8 in the EEG tensor.
- OSC `/raw` is already filtered on-device (2nd-order Butterworth 2–45 Hz + 50/60 notch). Epochs 16×8. App uses `/raw`. `/rawUnfiltered` exists for matched offline filtering.
- Reward / gate pads: **PO3, PO4**. Crown guardrail parked; live REVE scoring is Muse-only today.

### Official REVE position bank (meters; stream order)

```
CP3  [-0.06356, -0.04701,  0.06562]
C3   [-0.06536, -0.01163,  0.06436]
F5   [-0.06447,  0.04804,  0.01692]
PO3  [-0.03651, -0.10085,  0.03717]
PO4  [ 0.03678, -0.10085,  0.03640]
F6   [ 0.06791,  0.04983,  0.01637]
C4   [ 0.06712, -0.01090,  0.06358]
CP4  [ 0.06661, -0.04664,  0.06558]
```

## REVE head training (strict)

1. Resample to **200 Hz**. Patch = 200 samples = 1 s; do not train at native 256 Hz if matching reve-base.
2. Window: 4 s → 800 samples **or** 5 s → 1000 (`n_times=1000`); stay consistent.
3. Layout `[C, T]` float32 µV, channel-major. C=4 or C=8 OK (montage-agnostic Fourier); **positions must match channel order**.
4. Band-pass: pretrain 0.5–99.5 Hz; downstream often 0.5–45 + notch. Muse ≈ raw; Crown `/raw` already 2–45 — prefer `/rawUnfiltered` + same filter when possible.
5. 3D positions from **official bank** (https://huggingface.co/brain-bzh/reve-positions), meters on ~10 cm head (+x right, +y anterior, +z superior).
6. Do **not** z-score before encoder if using reve-rs / braindecode REVE (they per-channel z-score + clip ±15σ).
7. Frozen reve-base → **512-d** pooled vector; reve-large → 1250-d. Linear/MLP head on that.

Weights: `brain-bzh/reve-base` (gated). User fetches; we ship heads + checksums only.

## App vs train (don’t silently copy live path)

| | App today | Train a REVE head |
|--|-----------|-------------------|
| Rate | 256 Hz, 1024 (4 s) | 200 Hz, 800 or 1000 |
| Positions | hardcoded mm guesses | official bank meters |
| Order into model | AF7, AF8, TP9, TP10 | any matching positions |
| Devices | Muse 4-ch only | Muse and/or Crown |
| Pre-filter | none Muse; Crown already 2–45 | match device / unfilter Crown |

Better: train resampled + official positions, then change the app forwarder to match.

## CBraMod / LaBraM (similar spirit)

- Keep **montage-specific** heads (`muse4` vs `crown8`).
- Match the encoder’s expected rate/patch conventions (CBraMod path here already uses 200 Hz / 200-sample patches).
- Don’t mix montages in one batch item.
- Positions: use if the encoder needs them; otherwise still record official coords in manifests for REVE parity later.

## Dataset rework checklist

1. Add `datasets/common/montages.json` with both layouts + position banks + stream vs model order.
2. Extend `channel_map.py`: `MUSE4_STREAM`, `MUSE4_MODEL`, `CROWN8_STREAM`.
3. Tag existing windows `montage_id=muse4`.
4. Crown corpora: only real 8-ch sources (or user Crown recordings); new tree `datasets/*_crown8/` or `montage=` field.
5. Export packs: `exports/cbramod_pack_muse4/`, later `exports/cbramod_pack_crown8/` and REVE equivalents.
6. Continue ~80-subject **muse4** attention expansion; Crown is a parallel track once data exists.

## Head B / C

- Same label taxonomies per montage; Head B calibration must be recorded on the **same** device/montage the head expects.
- Head C sleep stages: Crown may transfer better for posterior (PO3/PO4) than Muse alone — still open-license + provenance.

## Attention Crown8 LOSO control

See `docs/crown8_attention_loso.md` (same OpenNeuro labels/starts as muse4; Crown8 did not beat chance).

# Head A-eng train — expanded corpus (frozen CBraMod)

**Step:** `head_a_eng_train_cbramod`  
**Date (UTC):** 2026-09-08T13:53:58.401111+00:00  
**ship_candidate:** **False** — test macro-F1 0.548 < 0.55; cleaner source ds007262 test macro-F1 0.436 < 0.55. Residual order confounds remain on ds007169 (L1→L4), eegmat (rest→arith), and STEW (rest→SIMKAP). Domain mix: professional 10–20/32-ch + STEW Emotiv hobbyist. muse4 proxy montage — not true Muse-native.

## Corpus

- Path: `datasets/engagement_a_eng/`
- Sources: ds007169, ds007262, ds007554, eegmat (professional) + stew (hobbyist Emotiv)
- Unique persons: **133**; packs: 150; windows: ~19706 muse4
- Splits: subject/person-wise 93 / 20 / 20 on `unique_person_id` (Barras tasks co-split)
- Labels: `low_engagement` / `high_engagement`

## Domain mix & confounds (honest)

| Issue | Detail |
|-------|--------|
| Device mix | professional 10–20/32-ch vs STEW Emotiv 14-ch hobbyist |
| Order | ds007169 L1→L4; eegmat rest→arith; STEW rest→SIMKAP |
| Cleaner | ds007262 difficulty randomized (preferred signal check) |
| Montage | muse4 proxy only — not true Muse |

## Setup

- Encoder: **frozen** CBraMod (embed-once → head train)
- Head: `HeadAEngLinear` (in_dim=200)
- Train: undersample_balanced (cap 40000); class-weighted CE; best-by-val; patience 5
- Backbone fine-tune: **no**
- Skipped: Head B, A-med

## Metrics vs chance 0.5

| Split | n | accuracy | macro-F1 | Δ vs chance |
|-------|--:|---------:|---------:|------------:|
| train (fit) | 13312 | 0.608 | 0.605 | +0.105 |
| val | 2914 | 0.592 | 0.588 | +0.088 |
| test | 3084 | 0.551 | 0.548 | +0.048 |

### Test per-class

| class | precision | recall | f1 | support |
|-------|----------:|-------:|---:|--------:|
| low_engagement | 0.528 | 0.657 | 0.585 | 1488 |
| high_engagement | 0.586 | 0.453 | 0.511 | 1596 |

### Test by source (subject-holdout packs)

| source | n | macro-F1 | note |
|--------|--:|---------:|------|
| ds007169 | 480 | 0.727 | order confound |
| ds007262 | 324 | 0.436 | randomized difficulty |
| ds007554 | 600 | 0.375 | — |
| eegmat | 400 | 0.425 | order confound |
| stew | 1280 | 0.561 | order confound, hobbyist |

## Fit vs misfit recommendation

**Fit vs misfit (A-eng public head):**

- **WEAK / MISFIT:** test macro-F1=0.548 barely above chance. Frozen linear insufficient for public A-eng. Do not fine-tune yet — fixulate labels/montage/domain mix first.
- **Skipped:** Head B, A-med.
- **REVE:** compare on same splits via Kaggle T4 before any fine-tune decision.
- Test-by-source macro-F1: ds007169=0.727, ds007262=0.436, ds007554=0.375, eegmat=0.425, stew=0.561

## Artifacts

- Emb cache: `exports/head_a_eng_train_cbramod/emb_cache/`
- Weights: `exports/head_a_eng_train_cbramod/head_a_eng_linear.pt`
- Metrics: `exports/head_a_eng_train_cbramod/metrics_summary.json`
- Script: `scripts/train_head_a_eng_cbramod.py`

See also dual-encoder doc: `docs/head_a_eng_dual_encoder.md`.

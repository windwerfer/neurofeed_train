# Head A-eng train — expanded corpus (frozen REVE)

**Step:** `head_a_eng_train_reve`  
**Date (UTC):** 2026-09-08T14:20:42.521972+00:00  
**ship_candidate:** **False** — Overall metrics clear soft bar but cleaner randomized source (ds007262) does not independently clear 0.55 — risk that F1 is partly order/time shortcut. Residual order confounds remain on ds007169 (L1→L4), eegmat (rest→arith), and STEW (rest→SIMKAP). Domain mix: professional 10–20/32-ch + STEW Emotiv hobbyist. muse4 proxy montage — not true Muse-native.

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

- Encoder: **frozen** REVE-base (embed-once → head train)
- Head: `HeadAEngLinear` (in_dim=512)
- Train: undersample_balanced (cap 40000); class-weighted CE; best-by-val; patience 5
- Backbone fine-tune: **no**
- Skipped: Head B, A-med

## Metrics vs chance 0.5

| Split | n | accuracy | macro-F1 | Δ vs chance |
|-------|--:|---------:|---------:|------------:|
| train (fit) | 13312 | 0.627 | 0.626 | +0.126 |
| val | 2914 | 0.602 | 0.602 | +0.102 |
| test | 3084 | 0.590 | 0.590 | +0.090 |

### Test per-class

| class | precision | recall | f1 | support |
|-------|----------:|-------:|---:|--------:|
| low_engagement | 0.574 | 0.584 | 0.579 | 1488 |
| high_engagement | 0.605 | 0.595 | 0.600 | 1596 |

### Test by source (subject-holdout packs)

| source | n | macro-F1 | note |
|--------|--:|---------:|------|
| ds007169 | 480 | 0.552 | order confound |
| ds007262 | 324 | 0.515 | randomized difficulty |
| ds007554 | 600 | 0.552 | — |
| eegmat | 400 | 0.591 | order confound |
| stew | 1280 | 0.616 | order confound, hobbyist |

## Fit vs misfit recommendation

**Fit vs misfit (A-eng public head):**

- **MISFIT for public ship (despite F1=0.590):** metrics above chance but order/domain confounds or cleaner-source check fail honesty bar. Keep exploring; do **not** fine-tune backbone to chase confounded signal. Optional: STEW-only / ds007262-only ablations; true-Muse personal cal.
- **Skipped:** Head B, A-med.
- **REVE:** compare on same splits via Kaggle T4 before any fine-tune decision.
- Test-by-source macro-F1: ds007169=0.552, ds007262=0.515, ds007554=0.552, eegmat=0.591, stew=0.616

## Artifacts

- Emb cache: `exports/head_a_eng_train_reve/emb_cache/`
- Weights: `exports/head_a_eng_train_reve/head_a_eng_linear.pt`
- Metrics: `exports/head_a_eng_train_reve/metrics_summary.json`
- Script: `scripts/train_head_a_eng_cbramod.py`

See also dual-encoder doc: `docs/head_a_eng_dual_encoder.md`.

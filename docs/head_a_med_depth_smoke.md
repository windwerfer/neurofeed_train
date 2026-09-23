# Head A-med depth smoke (CBraMod, CPU)

**Completed (Asia/Bangkok):** 2026-09-08 02:00:57 ICT
**UTC:** 2026-09-07T19:00:57.140292+00:00
**Version:** `med_depth_q1_le1_ge2_onewin_2026-09-08`

## Goal

Frozen **CBraMod** + **HeadADepth** (`meditation_depth_low` vs `meditation_depth_high`) local CPU smoke on OpenNeuro **ds001787 (64-ch BioSemi, professional)** muse4 proxy (AF7/AF8/TP9/TP10 extracted; TP9/TP10←P9/P10). Subject holdout + LOSO-lite.

## Dataset

- **Corpus:** ds001787 (64-ch BioSemi, professional) — muse4 proxy channels AF7/AF8/TP9/TP10 extracted (TP9/TP10←P9/P10)
- **DOI / license:** `doi:10.18112/openneuro.ds001787.v1.1.1` / `CC0-1.0`
- **Source windows:** `datasets/attention_ds001787/windows/` (existing muse4 attention pack)
- **Not used:** crown8 pack; Kaggle; REVE

## Mapping rule (exact Q1 thresholds)

Brandmeyer & Delorme probe **Q1** = depth of meditation / concentration on **0–3** scale.

| Q1 | A-depth label |
|----|---------------|
| **≤ 1** (0 or 1) | `meditation_depth_low` |
| **≥ 2** (2 or 3) | `meditation_depth_high` |
| missing Q1 | **drop** |

Integer 0–3 scale → **no mid band** between 1 and 2. Extremes-only (0 vs 3) was too sparse locally (only 3 subjects with both) — not used for this smoke.

**Window policy:** **one window per probe** (median-start window inside the existing ~10 s pre-Q1 epoch). Overlapping hop-0.5 s windows from the attention ingest are collapsed.

## Split

- Train subjects (both classes after remap): sub-001, sub-002, sub-003, sub-004, sub-005, sub-006, sub-007, sub-013, sub-015, sub-016, sub-017, sub-019, sub-021 (n=13)
- Holdout: `sub-020` / `sub020_ses01`
- Per-subject `undersample_balanced` then concat; internal 15% val for early pick.

## Metrics (headline)

| Split | n | accuracy | macro-F1 | Δ vs chance (0.5) |
|-------|--:|---------:|---------:|------------------:|
| train_balanced | 132 | 0.491 | 0.329 | -0.171 |
| val | 20 | 0.550 | 0.355 | -0.145 |
| holdout_full | 37 | 0.541 | 0.351 | -0.149 |
| holdout_balanced | 34 | 0.500 | 0.333 | -0.167 |

**ship_candidate:** `False` (criterion: holdout balanced macro-F1 ≥ 0.55 and both classes predicted)

### Holdout confusion (balanced)

```
labels: ['meditation_depth_low', 'meditation_depth_high']
matrix: [[17, 0], [17, 0]]
pred_counts: {'meditation_depth_low': 34, 'meditation_depth_high': 0}
```

## LOSO-lite

- Folds: **5** (subjects with both classes; largest minority first)
- Mean holdout full macro-F1: **0.299**
- Mean holdout balanced macro-F1: **0.333** (Δ vs chance -0.167)

## Counts

- Subjects loaded: **16**
- Train subjects used: **13**
- Holdout probes/windows (1/probe): **37** (low=20, high=17)
- Train balanced windows: **132**

## Provenance

- Encoder: CBraMod Apache-2.0 `pretrained_weights.pth` (local cache)
- Head: `HeadADepthLinear` (`src/heads/head_a_depth.py`)
- Script: `scripts/head_a_med_depth_smoke.py`
- Exports: `exports/head_a_med_depth_smoke/`
- Montage: **muse4 only**

## Caveats

- Q1 is subjective probe noise (same family as MW LOSO ~0.36).
- Holdout head **collapsed to always `meditation_depth_low`** (never fires high) — same failure mode as n=12 rest↔med always-`rest`.
- Domain gap: CBraMod TUEG → 4-ch Muse proxy from BioSemi.
- One window/probe reduces n dramatically vs overlapping attention pack.
- Rest↔med on ds003816 remains a separate (weaker UX) operationalization.

## Takeaway

A-med depth smoke on ds001787 (64-ch BioSemi, professional) — muse4 proxy channels AF7/AF8/TP9/TP10 extracted (TP9/TP10←P9/P10): Q1≤1 vs ≥2, 1 win/probe; n_subjects_loaded=16, train_used=13, holdout=sub-020 n=34; holdout bal macro-F1=0.333 (chance 0.50, Δ -0.167); LOSO-lite mean bal F1=0.333 (5 folds); ship_candidate=False.


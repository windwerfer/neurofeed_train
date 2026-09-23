# Stress/calm smoke (CBraMod, CPU)

**Completed (Asia/Bangkok):** 2026-09-08 15:35:05 ICT
**UTC:** 2026-09-08T08:35:05.735850+00:00
**Version:** `stress_calm_eegmat_n8_2026-09-08`

## Gate

Ran only after **order-aware A-eng PASS** (primary mid 2vs3 holdout bal F1=0.8330926015085861 > 0.55). See `docs/head_a_eng_smoke.md` § Order-aware re-smoke.

## Goal

Frozen **CBraMod** chance-check for **stress vs calm** (`calm` vs `stress`) on a **size-bounded** Muse-mappable open corpus.

## Data pick

| Candidate | License | Why not / why used |
|-----------|---------|-------------------|
| PhysioNet MATB `neuro-stress-resilience-hci` | ODbL | ~606 MB/subject; AF7/AF8 fNIRS — aborted earlier |
| SAM-40 Emotiv Flex | CC-BY-4.0 | Single 760 MB RAR — over smoke bound |
| alkabbany Muse-S | CC-BY-4.0 | Google Drive; n=5 |
| STEW | CC-BY-4.0 | IEEE login for raw |
| **PhysioNet EEG During Mental Arithmetic Tasks / eegmat (23-ch 10–20 professional)** | **ODC-By-1.0** | **USED** — ~5 MB/subject; rest vs arithmetic |

**Dataset bracket tag:** `PhysioNet EEG During Mental Arithmetic Tasks / eegmat (23-ch 10–20 professional)`

## Label mapping (honest)

| Source | Label | Notes |
|--------|-------|-------|
| `SubjectXX_1.edf` background rest | `calm` | Last 60 s of rest file |
| `SubjectXX_2.edf` mental arithmetic | `stress` | ~60 s; standardized stress-induction protocol (serial subtraction) |

Product UX caveat: rest≠meditation calm; arithmetic stress≠psychosocial stress.

## Montage

- **muse4 only** (never mixed with crown8)
- Proxy: `F7→AF7`, `F8→AF8`, `T3→TP9`, `T4→TP10`
- Resample 500 Hz → 256 Hz; 2 s / 1 s hop; edge trim 2 s; cap 50 windows/condition
- Per-channel z-score + clip±15 before encode

## Split

- Train: Subject00, Subject01, Subject02, Subject03, Subject04, Subject05, Subject06 (n=7)
- Holdout: Subject07
- Per-subject undersample_balanced; 15% val; best-by-val

## Metrics (headline)

| Split | n | accuracy | macro-F1 |
|-------|--:|---------:|---------:|
| Train (balanced) | 595 | 0.620 | 0.606 |
| Val | 105 | 0.610 | 0.609 |
| Holdout full | 100 | 0.540 | 0.495 |
| Holdout balanced | 100 | 0.540 | 0.495 |
| Holdout stride×4 | 25 | 0.440 | 0.417 |

**Chance macro-F1 (balanced binary):** `0.5`
**Δ holdout-balanced vs chance:** `-0.005`
**ship_candidate:** `False`

## Confounds

1. **Order:** rest always before arithmetic — time/drift may contribute.
2. Cognitive load / stress-induction ≠ meditation calm UX.
3. Single holdout subject; overlapping windows; muse4 proxy.

## Recommendation

- **Code:** `personal_cal_only`
- **Note:** Near chance on this smoke — drop as public stress/calm head; personal Muse cal only if product wants stress UX.

## Provenance

- Dataset: `PhysioNet EEG During Mental Arithmetic Tasks / eegmat (23-ch 10–20 professional)` — SPDX `ODC-By-1.0` — `doi:10.13026/C2JQ1P`
- Encoder: CBraMod Apache-2.0 `pretrained_weights.pth`
- Montage: **muse4** only
- Script: `scripts/head_stress_calm_smoke.py`
- Exports: `exports/head_stress_calm_smoke/`
- No Kaggle push

## Takeaway

Stress/calm smoke: PhysioNet EEG During Mental Arithmetic Tasks / eegmat (23-ch 10–20 professional); muse4 proxy F7/F8/T3/T4; n=8 holdout Subject07; holdout bal macro-F1=0.495 vs chance 0.5 (Δ-0.005); ship_candidate=False; rec=personal_cal_only.

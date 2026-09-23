# ds001787 attention ingest (Head A)

**Step:** `ds001787_attention_ingest`  
**Completed (UTC):** 2026-09-07  
**Dataset:** OpenNeuro `ds001787` v1.1.1 — SPDX **CC0-1.0** — DOI `doi:10.18112/openneuro.ds001787.v1.1.1`

## What landed

Cached **4 recordings** (ses-01): `sub-001` / `sub-002` (expert) + `sub-013` / `sub-017` (novice).

| Tag | Group | concentration | mind_wandering | total | merge |
|-----|-------|--------------:|---------------:|------:|-------|
| sub001_ses01 | expert | 119 | 153 | 272 | events+log nearest |
| sub002_ses01 | expert | 187 | 119 | 306 | events onset + log ratings |
| sub013_ses01 | novice | 136 | 34 | 170 | events+log nearest |
| sub017_ses01 | novice | 476 | 0 | 476 | events onset + log ratings |
| **total** | | **918** | **306** | **1224** | |

Windows: **2 s / 0.5 s hop** inside the paper pre-Q1 epoch **[−10.05, −0.05] s**, shape `(N, 4, 512)` @ 256 Hz.

## Label rule (unchanged)

- After stimulus `128`: ordered Q1 (meditation) then Q2 (mind wandering).
- **Q1 > Q2 → `concentration`**; **Q1 < Q2 → `mind_wandering`**; ties / incomplete → drop.
- Prefer behavioral-log ratings when aligned; EEG epoch onsets always from `events.tsv` `value=128`.
- **Hard reject:** do not map raw event values `2`/`4` to classes.

## Channel proxy

BioSemi ActiveTwo A1–B32 renamed via MNE `biosemi64` montage. Export order **AF7 / AF8 / TP9 / TP10** where **TP9/TP10 ← P9/P10** (mastoid-adjacent; TP7/TP8 fallback). Not native Muse geometry — document in manifests.

## Artifacts

- Script: `scripts/ds001787_attention_ingest.py`
- Exports: `exports/windows_ds001787/`
- Summary: `exports/ds001787_attention_ingest_summary.json`
- Provenance: `kaggle_datasets/muse-eeg-heads-cache/data/ds001787/provenance_attention_ingest.json`
- Private windows copies under `kaggle_datasets/muse-eeg-heads-windows/ds001787_*`

## Out of scope this step

- No Head A retraining / 4-way attention logit update (next themes).
- No bulk download of remaining subjects/sessions.
- Head B untouched.

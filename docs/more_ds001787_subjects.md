# more_ds001787_subjects (attention pool expand)

**Step:** `more_ds001787_subjects`  
**Completed (UTC):** 2026-09-07  
**Dataset:** OpenNeuro `ds001787` — SPDX **CC0-1.0** — DOI `doi:10.18112/openneuro.ds001787.v1.1.1`

## Why

Prior attention-only / joint holdouts collapsed (~chance). Add **2 expert + 2 novice** ses-01 recordings to the Muse-proxy attention pool before any retrain.

## New subjects

| Tag | Group | concentration | mind_wandering | total | merge |
|-----|-------|--------------:|---------------:|------:|-------|
| sub003_ses01 | expert | 51 | 85 | 136 | events_onset_log_ratings |
| sub004_ses01 | expert | 170 | 17 | 187 | events_onset_log_ratings |
| sub014_ses01 | novice | 85 | 0 | 85 | events_plus_log_nearest |
| sub019_ses01 | novice | 170 | 425 | 595 | events_onset_log_ratings |
| **new total** | | **476** | **527** | **1003** | |

**Pool now:** sub001_ses01, sub002_ses01, sub013_ses01, sub017_ses01, sub003_ses01, sub004_ses01, sub014_ses01, sub019_ses01  
**Pool windows:** concentration=1394, mind_wandering=833, total=2227

## Label / channel rules (unchanged)

- Q1>Q2 → concentration; Q1<Q2 → mind_wandering; ties dropped
- **Hard reject:** do not map events value 2/4 to classes
- Channels: AF7/AF8 + P9/P10→TP9/TP10 (BioSemi64)

## Artifacts

- Script: `scripts/more_ds001787_subjects.py`
- Exports: `exports/windows_ds001787/` (new tags)
- Summary: `exports/more_ds001787_subjects_summary.json`
- Provenance: `kaggle_datasets/muse-eeg-heads-cache/data/ds001787/provenance_more_ds001787_subjects.json`

## Out of scope

- No Head A retrain this step (next invent candidate: attention retrain with subject holdout on expanded pool)
- Head B / Head C training untouched

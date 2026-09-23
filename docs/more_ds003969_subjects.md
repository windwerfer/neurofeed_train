# more_ds003969_subjects (attention pool expand)

**Step:** `more_ds003969_subjects`  
**Completed (UTC):** 2026-09-07  
**Dataset:** OpenNeuro `ds003969` v1.0.0 — SPDX **CC0-1.0** — DOI `doi:10.18112/openneuro.ds003969.v1.0.0`

## Why

Prior attention-only / ds001787 expanded subject-holdout near chance. Add **2 htr + 2 ctr** Muse-proximal subjects (`med1breath` + `think1`) to the ds003969 attention pool before any retrain.

## New subjects

| Tag | Group | concentration | mind_wandering | total |
|-----|-------|--------------:|---------------:|------:|
| sub003 | htr | 400 | 400 | 800 |
| sub004 | htr | 400 | 400 | 800 |
| sub026 | ctr | 400 | 400 | 800 |
| sub027 | ctr | 400 | 400 | 800 |
| **new total** | | **1600** | **1600** | **3200** |

**Pool now:** sub001, sub002, sub025, sub003, sub004, sub026, sub027 (7 tags)  
**Pool windows:** concentration=2800, mind_wandering=2800, total=5600

Windows: **2 s / 1.0 s hop**, edge trim **30 s**, cap **400**/block, shape `(N, 4, 512)` @ 256 Hz (resampled from 1024 Hz).

## Label rule

- Block-level protocol proxy: **`med*` → `concentration`**, **`think*` → `mind_wandering`**.
- Weaker than ds001787 probe ratings — fine for Muse-channel bootstrap with missing-class mask.
- No event-code remapping (tasks are separate BIDS runs).

## Channel proxy

**AF7 / AF8** native; **TP9/TP10 ← TP7/TP8** (dataset has no TP9/TP10). Export order Muse names.

## Artifacts

- Script: `scripts/more_ds003969_subjects.py`
- Exports: `exports/windows_ds003969/` (new tags sub003/sub004/sub026/sub027)
- Summary: `exports/more_ds003969_subjects_summary.json`
- Run log: `exports/more_ds003969_subjects_run.log`
- Provenance: `kaggle_datasets/muse-eeg-heads-cache/data/ds003969/provenance_more_ds003969_subjects.json`
- Private windows copies under `kaggle_datasets/muse-eeg-heads-windows/ds003969_*`

## Out of scope this step

- No Head A retrain (next invent candidates: mix ds001787+ds003969 smoke, or attention-only retrain on expanded ds003969).
- No remaining med2/think2 blocks or other subjects.
- Head B / Head C training untouched; do not add attention head to CBraMod pack (vigilance only).

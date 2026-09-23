# ds003969 attention ingest (Head A)

**Step:** `ds003969_attention_ingest`  
**Completed (UTC):** 2026-09-07  
**Dataset:** OpenNeuro `ds003969` v1.0.0 — SPDX **CC0-1.0** — DOI `doi:10.18112/openneuro.ds003969.v1.0.0`

## What landed

Cached **3 subjects × 2 blocks** (`med1breath` + `think1`): `sub-001` / `sub-002` (htr) + `sub-025` (ctr).

| Tag | Group | concentration | mind_wandering | total |
|-----|-------|--------------:|---------------:|------:|
| sub001 | htr | 400 | 400 | 800 |
| sub002 | htr | 400 | 400 | 800 |
| sub025 | ctr | 400 | 400 | 800 |
| **total** | | **1200** | **1200** | **2400** |

Windows: **2 s / 1.0 s hop**, edge trim **30 s**, cap **400**/block, shape `(N, 4, 512)` @ 256 Hz (resampled from 1024 Hz).

## Label rule

- Block-level protocol proxy: **`med*` → `concentration`**, **`think*` → `mind_wandering`**.
- Weaker than ds001787 probe ratings — fine for Muse-channel bootstrap with missing-class mask.
- No event-code remapping (tasks are separate BIDS runs).

## Channel proxy

**AF7 / AF8** native; **TP9/TP10 ← TP7/TP8** (dataset has no TP9/TP10). Export order Muse names.

## Artifacts

- Script: `scripts/ds003969_attention_ingest.py`
- Exports: `exports/windows_ds003969/`
- Summary: `exports/ds003969_attention_ingest_summary.json`
- Provenance: `kaggle_datasets/muse-eeg-heads-cache/data/ds003969/provenance_attention_ingest.json`
- Private windows copies under `kaggle_datasets/muse-eeg-heads-windows/ds003969_*`

## Out of scope this step

- No Head A retraining (next: attention-only smoke or joint 4-way with Muse-channel subjects).
- No remaining med2/think2 blocks or other subjects.
- Head B / Head C training untouched.

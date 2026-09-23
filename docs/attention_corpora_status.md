# Attention corpora status (concentration / mind_wandering)

**Updated:** 2026-09-07 (Asia/Bangkok)  
**Head A labels confirmed:** `concentration`, `mind_wandering`, `drowsy`, `hypnagogic`  
**Sleep-EDF vigilance v0:** **FROZEN** (not expanded further).  
**ship_candidate:** **false** — subject holdouts historically near chance; do not claim attention head shippable until honest holdout beats chance.

## Goal this pass

Complete **local** OpenNeuro attention corpora for `concentration` / `mind_wandering` (no Kaggle upload). Fixed subject splits (policy A); document class balance.

## Corpora overview

| Corpus | Source | License | Subjects | Windows | Muse-proxy |
|--------|--------|---------|----------:|--------:|------------|
| `attention_ds001787` | OpenNeuro ds001787 v1.1.1 | CC0-1.0 | 12 | 3774 | AF7/AF8 + P9/P10→TP9/TP10 (BioSemi64) |
| `attention_ds003969` | OpenNeuro ds003969 v1.0.0 | CC0-1.0 | 11 | 8800 | AF7/AF8 native; TP7/TP8→TP9/TP10 |
| `vigilance_sleep_edf` | Sleep-EDF (frozen) | ODbL | 5 | 24618 | existing n1-slice recipe |

## Label rules

- **ds001787:** Q1>Q2→`concentration`, Q1<Q2→`mind_wandering`, ties/incomplete dropped. **Do not** map events value 2/4 to classes. See `docs/ds001787_codebook_confirm.md`.
- **ds003969:** `med*` blocks→`concentration`, `think*`→`mind_wandering` (protocol proxy; weaker than probe ratings).

## Counts by split (attention only)

### attention_ds001787 (probe ratings)

| Split | Subjects | concentration | mind_wandering | total |
|-------|----------|--------------:|---------------:|------:|
| train | sub-001,002,003,004,005,014,017,020 | 1615 | 765 | 2380 |
| val | sub-006,015 | 357 | 272 | 629 |
| test | sub-019,013 (frozen honest holdouts) | 306 | 459 | 765 |
| **all** | 12 | **2278** | **1496** | **3774** |

**Val fix:** prior val (sub-014, sub-017) was concentration-only after ingest. Moved those to train; val now sub-006 + sub-015 (both classes). Test holdouts unchanged.

### attention_ds003969 (block labels)

| Split | Subjects | concentration | mind_wandering | total |
|-------|----------|--------------:|---------------:|------:|
| train | sub-001…006,029 | 2800 | 2800 | 5600 |
| val | sub-026,028 | 800 | 800 | 1600 |
| test | sub-025,027 (frozen honest holdouts) | 800 | 800 | 1600 |
| **all** | 11 | **4400** | **4400** | **8800** |

## What’s done

- [x] Local ingest expanded: ds001787 +4 (005,006,015,020); ds003969 +4 (005,006,028,029)
- [x] Symlinked into `datasets/attention_*` via `scripts/dataset/migrate_existing_exports.py`
- [x] Fixed subject JSON splits (policy A); `validate_splits.py` → no leakage
- [x] Annotations include `subject_id`, `recording_id`, `y_head_a`, `provenance`, `license_spdx`, split
- [x] Checksums verified against window manifests
- [x] Muse-proxy channel mapping documented in manifests + corpus READMEs

## What’s still thin / gaps

- **Holdout quality:** prior attention-only / expanded holdouts were ~chance → `ship_candidate: false`
- **ds001787 subject coverage:** 12/24 subjects; many remaining experts are concentration-skewed (e.g. sub-009)
- **ds001787 single-class subjects:** sub-014, sub-017 remain concentration-only in train (log/event merge artifact)
- **ds003969:** 11/98 subjects; block labels are weaker than probe ratings
- **HF EEGMeditation (CC BY):** ALLOW but gated — not ingested this pass
- **No raw BDF in `datasets/*/raw`** for attention (windows symlinked from `exports/`; cache under `kaggle_datasets/.../raw`)

## Scripts

| Role | Path |
|------|------|
| Expand ds001787 | `scripts/expand_attention_ds001787.py` |
| Expand ds003969 | `scripts/expand_attention_ds003969.py` |
| Base ingest | `scripts/ds001787_attention_ingest.py`, `scripts/ds003969_attention_ingest.py` |
| Migrate / splits / index / validate / summarize | `scripts/dataset/*.py` |

## Completeness verdict (v0 attention data)

**Local corpora are substantially complete for a v0 attention bootstrap** (both labels, both sources, fixed splits, balanced val on both corpora).  

**Still need more subjects and a non-chance subject holdout** before treating attention Head A as shippable. Prefer additional ds001787 novices with both classes and/or HF EEGMeditation (after gate) over Sleep-EDF expansion.

## Out of scope this pass

- No Sleep-EDF expansion  
- No Kaggle private version bump  
- No AppImage / in-app test  
- No attention head retrain claimed shippable  

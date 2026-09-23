# Head C sleep-stage corpus expansion (~120 subjects)

**Status:** **DONE enough to train** — 124 unique subjects / 125 recordings  
**Date:** 2026-09-08 (ICT)  
**Windows:** Sleep-EDF 445,522 + HMC 73,487 N1-slice labeled windows  
**Goal:** grow labeled sleep-stage windows toward **~120 unique subjects** for later Head C training (no full head training in this step).

## ALLOW sources

| Source | License | Subjects (target) | Montage report |
|--------|---------|-------------------|----------------|
| PhysioNet **Sleep-EDF Expanded** cassette | ODC-By | ~78 | `(ch_count=4, professional)` muse4 proxy Fpz-Cz/Pz-Oz → AF7/AF8/TP9/TP10 |
| PhysioNet **Sleep-EDF Expanded** telemetry | ODC-By | ~22 | same muse4 proxy |
| PhysioNet **HMC** sleep staging v1.1 | **CC-BY-4.0** (ship OK) | ~24 top-up | `(ch_count=4, professional)` muse4 proxy F4-M1/C4-M1/C3-M2/O2-M1 |

**DENY (not used):** LUNA, L-FAME, SEED-VIG. Private Kaggle version update optional later — local-first.

## Recipe (unchanged N1-slice)

- Around first N1 / stage 1: pre **20 min**, post **40 min**
- Window **2 s**, hop **0.5 s**, majority frac **0.7**, target **256 Hz**
- Head A vig: W → `drowsy`, N1 → `hypnagogic` (other stages excluded from Head A)
- Head C (required on every window): `stage_raw`, `stage_coarse` (`wake`/`light`/`deep`/`rem`/`unknown`)

## Scripts

| Script | Role |
|--------|------|
| `scripts/expand_head_c_sleep_corpus.py` | Inventory + download (resume) + window Sleep-EDF cassette/telemetry; rewrite splits |
| `scripts/expand_head_c_hmc.py` | Window local HMC SN* with scoring txt |
| `scripts/finalize_head_c_corpus.py` | Discover all manifests → fixed subject splits + validate |
| legacy | `add_more_sleep_edf_nights*.py`, `persist_head_c_stage_labels.py` |

Raw cache: `kaggle_datasets/muse-eeg-heads-cache/data/sleep-edfx-pilot/{sleep-cassette,sleep-telemetry}/`, `.../hmc-sleep-staging/recordings/`.  
Windows: `exports/windows_<tag>/` symlinked into `datasets/vigilance_sleep_edf/{raw,windows}/`.

Downloads: PhysioNet **open S3** (`aws s3 sync --no-sign-request s3://physionet-open/...`) — much faster than HTTPS.

## Splits (policy A)

Fixed subject-wise JSON under `datasets/vigilance_sleep_edf/splits/`.

- **Anchors preserved:** `SC400` → test, `SC403` → val  
- Remaining subjects ~70 / 15 / 15 by sorted id (no subject leakage)  
- Validate: `scripts/dataset/validate_splits.py`

## Progress artifacts

- `exports/head_c_expand/progress.json`
- `exports/expand_head_c_sleep_corpus_summary.json`
- `exports/expand_head_c_hmc_summary.json`
- `exports/finalize_head_c_corpus_summary.json`

## Next (out of scope here)

Train Head C (and revisit Head A vig) on the expanded corpus; then decide fine-tune.

## Final counts (2026-09-08)

| Metric | Value |
|--------|-------|
| Unique subjects | **124** (target ~120) |
| Recordings / nights | 125 |
| Sleep-EDF subjects | 100 (101 nights, 445,522 windows) |
| HMC subjects | 24 (73,487 windows) |
| Splits | train 86 / val 19 / test 19 |
| Validate | OK (`validate_splits.py` rc=0) |

**Blockers:** none for corpus; Head C training not run yet (per plan).

**Ready:** train heads next, then decide fine-tune.

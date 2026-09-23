# Persist Head C stage labels (backfill)

**Step:** `persist_head_c_stage_labels` (queued; dataset-only)  
**Date:** 2026-09-07

## What changed
Existing Head A N1-slice window npzs for SC4001/02/11/21/31/41 now include:
- `stage_raw` — majority hypnogram text per window (e.g. `Sleep stage W`)
- `stage_coarse` — `wake` | `light` | `deep` | `rem` | `unknown`

Head A labels (`drowsy` / `hypnagogic`) are unchanged. No Head C training.

## Counts
All 24 618 windows aligned with Head A (0 mismatches): W→wake / N1→light only in this N1-slice export (N2+/REM were already excluded from Head A windows).

## Artifacts
- Script: `scripts/persist_head_c_stage_labels.py`
- Helpers: `src/sleep_edf.py` (`STAGE_TO_COARSE`, `stage_to_coarse`, `majority_stage_in_window`)
- Summary: `exports/persist_head_c_stage_labels_summary.json`
- Policy: `docs/head_c_labels_policy.md`
- Private Kaggle: `windwerfer/muse-eeg-heads-windows` (versioned)

Future Sleep-EDF window builds should write these fields at creation time.

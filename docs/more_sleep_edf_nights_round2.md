# More Sleep-EDF nights round 2 (SC4021 + SC4041)

Status: complete (wave 2 step `more_sleep_edf_nights_round2`). Private Kaggle only.

## Why these nights

Added **two more cassette subjects** (02 and 04) so the multi-subject pool spans six nights across five subjects:

| Night | Subject | Role intent |
|-------|---------|-------------|
| SC4001 | 00 | existing train |
| SC4002 | 00 | existing holdout (same subject) |
| SC4011 | 01 | prior round |
| SC4021 | 02 | **new** — multi-subject pool |
| SC4031 | 03 | prior round / secondary holdout |
| SC4041 | 04 | **new** — multi-subject pool |

## Raw cache

PhysioNet Sleep-EDF Expanded (`sleep-cassette/`), ODC-By. Files under `kaggle_datasets/muse-eeg-heads-cache/data/sleep-edfx-pilot/sleep-cassette/`:

- `SC4021E0-PSG.edf` / `SC4021EH-Hypnogram.edf`
- `SC4041E0-PSG.edf` / `SC4041EC-Hypnogram.edf`

Provenance hashes: `data/sleep-edfx-pilot/provenance_more_nights_round2.json`.

## Windows recipe (unchanged)

- Around first Sleep stage 1: pre 20 min, post 40 min
- Window 2 s, hop 0.5 s, majority frac 0.7
- Labels: W → drowsy, N1 → hypnagogic
- Muse proxy: AF7=AF8=Fpz-Cz, TP9=TP10=Pz-Oz

## Counts

| Night | slice_start_sec | drowsy | hypnagogic | total | npz sha256 (prefix) |
|-------|-----------------|--------|------------|-------|---------------------|
| SC4021 | 20670.0 | 2516 | 2458 | 4974 | `56fb1a6f…` |
| SC4041 | 23160.0 | 2876 | 2397 | 5273 | `e0fef45a…` |

Both nights have near-balanced drowsy/hypnagogic counts in the N1 slice — useful for equal-night training pools.

## Packages

- Local: `exports/windows_sc4021/`, `exports/windows_sc4041/`, mirrored into `kaggle_datasets/muse-eeg-heads-windows/`
- Private Kaggle: `windwerfer/muse-eeg-heads-cache`, `windwerfer/muse-eeg-heads-windows` (version update only; never `-u`)

Script: `scripts/add_more_sleep_edf_nights_round2.py`

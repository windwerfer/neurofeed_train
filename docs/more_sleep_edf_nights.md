# More Sleep-EDF nights (SC4011 + SC4031)

Status: complete (wave 2 step `more_sleep_edf_nights`). Private Kaggle only.

## Why these nights

Added **two new cassette subjects** (not second nights of SC4000) so the next multi-night holdout can train on ≥2 subjects and hold out another:

| Night | Subject | Role intent |
|-------|---------|-------------|
| SC4001 | 00 | existing train |
| SC4002 | 00 | existing holdout (same subject) |
| SC4011 | 01 | new — multi-subject pool |
| SC4031 | 03 | new — multi-subject pool |

## Raw cache

PhysioNet Sleep-EDF Expanded (`sleep-cassette/`), ODC-By. Files landed under `kaggle_datasets/muse-eeg-heads-cache/data/sleep-edfx-pilot/sleep-cassette/`:

- `SC4011E0-PSG.edf` / `SC4011EH-Hypnogram.edf`
- `SC4031E0-PSG.edf` / `SC4031EC-Hypnogram.edf`

Provenance hashes: `data/sleep-edfx-pilot/provenance_more_nights.json`.

## Windows recipe (unchanged)

- Around first Sleep stage 1: pre 20 min, post 40 min
- Window 2 s, hop 0.5 s, majority frac 0.7
- Labels: W → drowsy, N1 → hypnagogic
- Muse proxy: AF7=AF8=Fpz-Cz, TP9=TP10=Pz-Oz

## Counts

| Night | slice_start_sec | drowsy | hypnagogic | total | npz sha256 (prefix) |
|-------|-----------------|--------|------------|-------|---------------------|
| SC4011 | 20340.0 | 2576 | 2934 | 5510 | `e2588e90…` |
| SC4031 | 24930.0 | 2753 | 776 | 3529 | `67174d30…` |

Note: SC4011 has unusually many hypnagogic windows in the N1 slice (long/early N1 stretch) — useful for recall but balance carefully when training.

## Packages

- Local: `exports/windows_sc4011/`, `exports/windows_sc4031/`, mirrored into `kaggle_datasets/muse-eeg-heads-windows/`
- Private Kaggle: `windwerfer/muse-eeg-heads-cache`, `windwerfer/muse-eeg-heads-windows` (version update only; never `-u`)

Script: `scripts/add_more_sleep_edf_nights.py`

# Both-nights windows dataset

Status: complete (overnight step 2). Private Kaggle only.

## Package

`windwerfer/muse-eeg-heads-windows` (private) holds Muse-proxy N1-slice windows for **SC4001** (train) and **SC4002** (holdout):

| File | Role |
|------|------|
| `sleep_edf_sc4001_n1slice_windows.npz` | Train-night windows `[N,4,512]` @ 256 Hz |
| `sleep_edf_sc4002_n1slice_windows.npz` | Holdout-night windows |
| `sleep_edf_sc4001_n1slice_manifest.json` | Per-night provenance (hashes, slice, map) |
| `sleep_edf_sc4002_n1slice_manifest.json` | Same for SC4002 |
| `manifest.json` | Package rollup + holdout summary |
| `ATTRIBUTION.txt` | PhysioNet ODC-By / CBraMod Apache-2.0 |

Local mirror: `kaggle_datasets/muse-eeg-heads-windows/` (byte-identical to `exports/windows_sc400*`).

## Recipe (both nights)

- Around first Sleep stage 1: pre 20 min, post 40 min
- Window 2 s, hop 0.5 s, majority frac 0.7
- Labels: W → drowsy, N1 → hypnagogic (drop 2/3/4/R/?/movement)
- Channel proxy: AF7=AF8=Fpz-Cz, TP9=TP10=Pz-Oz

## Counts

| Night | slice_start_sec | drowsy | hypnagogic | total | npz sha256 (prefix) |
|-------|-----------------|--------|------------|-------|---------------------|
| SC4001 | 29430.0 | 2457 | 298 | 2755 | `9f82c62e…` |
| SC4002 | 24870.0 | 2398 | 179 | 2577 | `2cf870bb…` |

Verified private on Kaggle (`isPrivate: true`) with all seven package files present. Do not make public (`-u` forbidden for this set).

URL: https://www.kaggle.com/datasets/windwerfer/muse-eeg-heads-windows


See also: [more_sleep_edf_nights.md](more_sleep_edf_nights.md) for SC4011/SC4031.

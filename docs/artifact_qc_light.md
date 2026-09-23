# Light artifact QC (windows)

**Script:** `scripts/dataset/artifact_qc_windows.py`

Conservative obvious-junk filters only — not a full EEG artifact pipeline.
Does **not** rescale units; thresholds are corpus-specific for units as stored.

## Units as stored

| Corpus | Units | Notes |
|--------|-------|-------|
| `vigilance_sleep_edf` | approx uV | Sleep-EDF Muse-proxy windows |
| `attention_ds001787` | approx V | OpenNeuro BioSemi to Muse-proxy, unscaled |
| `attention_ds003969` | approx V | OpenNeuro Muse-native scale as exported |
| `attention_ds001787_crown8` | approx V | Same BDFs, Crown8 montage |
| `attention_ds003969_crown8` | approx V | Same BDFs, Crown8 montage |

## Thresholds

| Check | Vigilance (uV) | Attention (V) | Action |
|-------|----------------|---------------|--------|
| NaN / Inf | any | any | reject `nan_inf` |
| Flat / near-zero channel std | < 0.1 | < 1e-09 | reject `flat` |
| Extreme peak abs(x) | > 350.0 | > 0.15 | reject `peak_abs` |
| Within-window z (per ch) | abs(z) > 25.0 | same | reject `peak_z` |
| Line-noise proxy (48-52 U 58-62 Hz power frac) | >= 0.85 | same | reject `line_noise` |

First-fail reason is stored; windows may fail multiple checks but only one reason is recorded.

## Outputs

- `datasets/<corpus>/windows/<recording>_qc.npz` — `qc_pass`, `qc_reason`, aligned `y`/`starts`
- `datasets/<corpus>/annotations/windows_qc.csv` — per-window mask
- `exports/artifact_qc_light/summary.json` — drop rates

## Drop-rate summary

### attention_ds003969_crown8

- total=51200 pass=39767 drop=11433 drop_rate=0.2233
- reasons: `{'pass': 39767, 'line_noise': 5833, 'peak_abs': 5600}`
- by split:
  - **test**: n=1600 drop=0 rate=0.0000
  - **train**: n=48000 drop=11433 rate=0.2382
  - **val**: n=1600 drop=0 rate=0.0000
- by label:
  - **concentration**: n=25600 drop=6146 rate=0.2401
  - **mind_wandering**: n=25600 drop=5287 rate=0.2065

### attention_ds001787_crown8

- total=5185 pass=5185 drop=0 drop_rate=0.0000
- reasons: `{'pass': 5185}`
- by split:
  - **test**: n=765 drop=0 rate=0.0000
  - **train**: n=3791 drop=0 rate=0.0000
  - **val**: n=629 drop=0 rate=0.0000
- by label:
  - **concentration**: n=3145 drop=0 rate=0.0000
  - **mind_wandering**: n=2040 drop=0 rate=0.0000

### vigilance_sleep_edf

- total=24618 pass=24618 drop=0 drop_rate=0.0000
- reasons: `{'pass': 24618}`
- by split:
  - **test**: n=5332 drop=0 rate=0.0000
  - **train**: n=15757 drop=0 rate=0.0000
  - **val**: n=3529 drop=0 rate=0.0000
- by label:
  - **drowsy**: n=15576 drop=0 rate=0.0000
  - **hypnagogic**: n=9042 drop=0 rate=0.0000

### attention_ds001787

- total=5185 pass=5185 drop=0 drop_rate=0.0000
- reasons: `{'pass': 5185}`
- by split:
  - **test**: n=765 drop=0 rate=0.0000
  - **train**: n=3791 drop=0 rate=0.0000
  - **val**: n=629 drop=0 rate=0.0000
- by label:
  - **concentration**: n=3145 drop=0 rate=0.0000
  - **mind_wandering**: n=2040 drop=0 rate=0.0000

### attention_ds003969

- total=51200 pass=49827 drop=1373 drop_rate=0.0268
- reasons: `{'pass': 49827, 'line_noise': 973, 'peak_abs': 400}`
- by split:
  - **test**: n=1600 drop=0 rate=0.0000
  - **train**: n=48000 drop=1164 rate=0.0243
  - **val**: n=1600 drop=209 rate=0.1306
- by label:
  - **concentration**: n=25600 drop=829 rate=0.0324
  - **mind_wandering**: n=25600 drop=544 rate=0.0213

_Generated 2026-09-07T08:45:34.194442+00:00_

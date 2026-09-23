# Head C smoke trial

**Step:** `head_c_smoke_trial`  
**Date (UTC):** 2026-09-07T07:01:48.323886+00:00  
**ship_candidate:** **False** — pipeline check only; not a product head.

## Setup

- Encoder: **frozen CBraMod** (same path as Head A)
- Head: `HeadCLinear` (dropout + linear)
- Classes: `stage_coarse` **wake, light, deep, rem** (unknown dropped)
- Windows: Sleep-EDF **sleep-period** slices (wake margin), not N1-slice — so deep/REM exist
- Caps: ≤200 windows/class/recording after light QC
- Splits: fixed subject JSON in `datasets/vigilance_sleep_edf/splits/`
- Channel note: Sleep-EDF has Fpz-Cz / Pz-Oz only. Mapped to Muse order as AF7=AF8=Fpz-Cz (frontal proxy), TP9=TP10=Pz-Oz (posterior proxy). Replace with true Muse / Schreer Muse-S for domain adaptation.

## Metrics

| Split | n | accuracy | macro-F1 |
|-------|--:|---------:|---------:|
| train (balanced fit set) | 2400 | 0.658 | 0.652 |
| val | 800 | 0.549 | 0.508 |
| test | 1600 | 0.554 | 0.549 |

### Val per-class

| class | precision | recall | f1 | support |
|-------|----------:|-------:|---:|--------:|
| wake | 0.885 | 0.805 | 0.843 | 200 |
| light | 0.230 | 0.295 | 0.259 | 200 |
| deep | 0.593 | 0.985 | 0.741 | 200 |
| rem | 0.733 | 0.110 | 0.191 | 200 |

### Test per-class

| class | precision | recall | f1 | support |
|-------|----------:|-------:|---:|--------:|
| wake | 0.877 | 0.482 | 0.623 | 400 |
| light | 0.314 | 0.420 | 0.359 | 400 |
| deep | 0.622 | 0.948 | 0.751 | 400 |
| rem | 0.623 | 0.367 | 0.462 | 400 |

## Muse-proxy weakness (deep / REM)

Sleep-EDF Fpz-Cz / Pz-Oz duplicated into Muse AF7/AF8/TP9/TP10 is a **coarse frontal/posterior proxy**.
Deep sleep (slow waves) and REM (occipital/EOG-ish patterns) transfer poorly to true Muse montage;
treat deep/REM F1 as diagnostic of pipeline wiring, not product quality.

## Artifacts

- Windows: `exports/head_c_smoke/windows/`
- Weights: `exports/head_c_smoke/head_c_coarse_linear.pt`
- Metrics: `exports/head_c_smoke/metrics_summary.json`
- Manifest: `exports/head_c_smoke/run_manifest.json`
- Script: `scripts/head_c_smoke_trial.py`
- Module: `src/head_c.py`

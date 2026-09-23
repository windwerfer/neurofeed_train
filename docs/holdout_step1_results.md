# Overnight Step 1 — SC4002 night holdout results

Generated: 2026-09-06T17:28:24.867777+00:00 (UTC). User TZ Asia/Bangkok (UTC+7).

## What we did
- Built N1-slice Muse-proxy windows for **SC4001** (train) and **SC4002** (holdout night).
- Recipe: around first `stage 1`, pre=20 min, post=40 min; window 2 s / hop 0.5 s; majority 0.7.
- Labels: Sleep stage W → drowsy, Sleep stage 1 → hypnagogic.
- Undersampled SC4001 to balance classes; froze CBraMod; trained HeadALinear 5 epochs on CPU.
- Evaluated same-night SC4001 val (20% of balanced) + SC4002 all-labeled + SC4002 balanced.

## Window counts / provenance

| Night | PSG | slice_start_sec | drowsy | hypnagogic | total |
|-------|-----|-----------------|--------|------------|-------|
| SC4001 | SC4001E0-PSG.edf | 29430.0 | 2457 | 298 | 2755 |
| SC4002 | SC4002E0-PSG.edf | 24870.0 | 2398 | 179 | 2577 |

SC4002 has **179** hypnagogic windows (enough for a balanced subsample of that size). No labels invented.

## Metrics table (train night vs holdout night)

| Split | n | accuracy | macro-F1 | notes |
|-------|---|----------|----------|-------|
| sc4001_train_balanced | 477 | 0.8428 | 0.8428 | fit on balanced train (optimistic) |
| sc4001_val_same_night | 119 | 0.7815 | 0.7815 | same night, 20% held out of balanced |
| sc4002_all_labeled | 2577 | 0.9131 | 0.7388 | true night holdout; class-imbalanced (mostly drowsy) |
| sc4002_balanced | 358 | 0.8268 | 0.8236 | holdout undersampled — fairest vs train |

### Per-class (holdout night)

**sc4002_all_labeled** confusion (rows=true drowsy/hypnagogic, cols=pred): `[[2229, 169], [55, 124]]`

| class | precision | recall | f1 | support |
|-------|-----------|--------|----|---------|
| drowsy | 0.976 | 0.930 | 0.952 | 2398 |
| hypnagogic | 0.423 | 0.693 | 0.525 | 179 |

**sc4002_balanced** confusion (rows=true drowsy/hypnagogic, cols=pred): `[[172, 7], [55, 124]]`

| class | precision | recall | f1 | support |
|-------|-----------|--------|----|---------|
| drowsy | 0.758 | 0.961 | 0.847 | 179 |
| hypnagogic | 0.947 | 0.693 | 0.800 | 179 |

**sc4001_val_same_night** confusion (rows=true drowsy/hypnagogic, cols=pred): `[[47, 13], [13, 46]]`

| class | precision | recall | f1 | support |
|-------|-----------|--------|----|---------|
| drowsy | 0.783 | 0.783 | 0.783 | 60 |
| hypnagogic | 0.780 | 0.780 | 0.780 | 59 |

## How to read this when half-awake
- Same-night val ~**0.78** acc / macro-F1.
- Balanced other-night holdout ~**0.83** acc / ~**0.82** macro-F1 (surprisingly close to train).
- All-labeled SC4002 accuracy looks high (~**0.91**) because the night is mostly drowsy; hypnagogic precision is weaker (~0.42) while recall is ~0.69. Trust **macro-F1 (~0.74)** + per-class numbers more than raw acc here.

## Artifacts
- Windows: `exports/windows_sc4001/`, `exports/windows_sc4002/` (+ manifests)
- Run: `exports/head_a_holdout/run_manifest.json`, `head_a_binary_state_dict.pt`, `metrics_summary.json`
- Notebook: `notebooks/04_night_holdout.ipynb`
- Kernel package: `kaggle_kernel_04_night_holdout/` → `windwerfer/muse-eeg-heads-night-holdout`
- Windows dataset package: `kaggle_datasets/muse-eeg-heads-windows/` (SC4001 + SC4002)

## Follow-up: stride-aware scoring
See [`docs/holdout_eval.md`](holdout_eval.md) for non-overlapping / stride×4 holdout metrics on the same Head A (overlap removed). Prefer those tables when quoting night-holdout numbers.

## Limitation
None for counts; SC4002 hypnagogic n=179 is modest but usable. Muse-proxy + frozen TUEG encoder domain gap remains.

## Kaggle URLs (private)
- Windows dataset: https://www.kaggle.com/datasets/windwerfer/muse-eeg-heads-windows
- Kernel: https://www.kaggle.com/code/windwerfer/muse-eeg-heads-night-holdout-sc4002


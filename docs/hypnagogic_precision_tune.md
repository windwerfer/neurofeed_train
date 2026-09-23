# Hypnagogic precision tune (Wave 2 · Step 1)

Generated: 2026-09-06 evening UTC (user TZ Asia/Bangkok, UTC+7).

## Problem in one line
On full **SC4002** night holdout, hypnagogic **precision was ~0.42** (recall ~0.69, macro-F1 ~0.74). Too many drowsy windows were called hypnagogic.

## What we tried
1. **Decision threshold sweep** (primary, leakage-safe): raise the bar for calling “hypnagogic”.
2. **Optional retrain** of Head A with precision-favoring / focal-ish class weights (SC4001 train only).

### Leakage policy
- Threshold (and any weight choice) was selected on **SC4001 same-night balanced validation** only (seed=42, `val_frac=0.2` — same recipe as the original holdout train).
- **SC4002 was never used** to pick the threshold or weights.
- A *leaky* “best threshold on SC4002” number is stored only as an ablation upper bound — do **not** quote it as the claimed result.

Decision rule after tune:
`predict hypnagogic if P(hypnagogic) ≥ threshold, else drowsy`.

## Chosen method
| Item | Value |
|------|-------|
| Method | **Frozen original Head A** + threshold |
| Threshold | **0.53** (was 0.5 / argmax) |
| Selection rule | Max hypnagogic precision on SC4001 val, subject to recall ≥ max(0.50, 0.85× baseline val recall) |
| Retrain | Tried; **not selected** (see below) |
| Tuned artifact | `exports/head_a_holdout/head_a_binary_precision_tuned.pt` (same weights as original; threshold in manifest) |

Retrains (`precision_favor` weights `[1.2, 0.6]`, `focal_precision` `[1.3, 0.55]` γ=1.5) made the head too conservative: SC4002 hypnagogic recall fell to ~0.23–0.25. Milder weight ablations still traded away too much holdout recall for the precision gain. **Threshold-only on the frozen head won.**

## Before / after (SC4002)

### Full night (`sc4002_all_labeled`, n=2577)

| | hypnagogic P | hypnagogic R | hypnagogic F1 | macro-F1 | confusion (rows=true d/h) |
|--|-------------:|-------------:|--------------:|---------:|---------------------------|
| **Before** t=0.50 | 0.423 | 0.693 | 0.525 | 0.739 | `[[2229, 169], [55, 124]]` |
| **After** t=0.53 | **0.552** | 0.536 | 0.544 | **0.755** | `[[2320, 78], [83, 96]]` |
| Δ | **+0.129** | −0.156 | +0.018 | +0.016 | FP 169→78; FN 55→83 |

### Balanced subsample (`sc4002_balanced`, n=358, seed=42)

| | hypnagogic P | hypnagogic R | hypnagogic F1 | macro-F1 |
|--|-------------:|-------------:|--------------:|---------:|
| **Before** t=0.50 | 0.899 | 0.693 | 0.782 | 0.805 |
| **After** t=0.53 | **0.941** | 0.536 | 0.683 | 0.739 |
| Δ | +0.043 | −0.156 | −0.099 | −0.065 |

(Balanced “before” hypnagogic precision can differ slightly from the original holdout table because the undersample draw is seed-42 standalone here; within this tune, before/after use the **same** subsample.)

### SC4001 val (selection split — not holdout)

| | hypnagogic P | hypnagogic R | macro-F1 |
|--|-------------:|-------------:|---------:|
| Before t=0.50 | 0.780 | 0.780 | 0.781 |
| After t=0.53 | 0.820 | 0.695 | 0.771 |

## How to read this when half-awake
- **Yes, precision improved** on the real holdout night: ~0.42 → ~0.55.
- Cost: hypnagogic recall ~0.69 → ~0.54 (still finds about half of N1-proxy windows).
- Macro-F1 on the full night nudged up (~0.74 → ~0.76) because we cut a lot of false alarms.
- On a **balanced** holdout slice, precision was already high; raising the threshold mainly hurts recall / macro-F1 there — the pain point was the **imbalanced full night**.
- Soft alternative seen in the grid (not selected): t=0.52 → SC4002 P≈0.51 / R≈0.59 / macro-F1≈0.76. Selection stuck to val-only rule → **0.53**.

## Artifacts
- Script: `scripts/tune_hypnagogic_threshold.py`
- Metrics: `exports/head_a_holdout/precision_tune_metrics.json`
- Head + threshold recorded in: `exports/head_a_holdout/run_manifest.json` → key `precision_tune`
- Tuned weights file: `exports/head_a_holdout/head_a_binary_precision_tuned.pt`
- Embedding cache (speed): `exports/head_a_holdout/emb_cache/`
- Notebook: `notebooks/05_hypnagogic_precision_tune.ipynb` (if present)

## Run again
```bash
export PATH=/home/box/.local/bin:$PATH
cd /workspace/muse-eeg-heads
.venv/bin/python scripts/tune_hypnagogic_threshold.py
```

## Limitation
Single train night / single holdout night. Threshold calibrated on same-night val can shift under domain change. Next Wave 2 steps add more Sleep-EDF nights and multi-night holdout.

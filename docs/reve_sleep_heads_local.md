# REVE sleep heads — local CPU subsample

**Date (UTC):** 2026-09-08T13:44:39.956293+00:00  
**Encoder:** frozen REVE-base (offline cache)  
**Sampling:** REVE local CPU subsample ≤80 windows/class/recording after QC; train undersample balanced (cap 40000). Full encode deferred to Kaggle T4.

## A-vig

| split | n | acc | macro-F1 |
|-------|--:|----:|---------:|
| val | 3029 | 0.709 | 0.709 |
| test | 3179 | 0.785 | 0.783 |

**ship_candidate:** True

## Head C (wake/light)

| split | n | acc | macro-F1 |
|-------|--:|----:|---------:|
| val | 3029 | 0.707 | 0.707 |
| test | 3179 | 0.792 | 0.791 |

**ship_candidate:** False (N1-slice 2-way only)

## vs CBraMod (full corpus)

See `exports/train_heads_expanded_sleep_summary.json`. Local REVE is **subsampled** — treat as directional; full REVE on Kaggle T4 is authoritative.

## Freeze vs fine-tune

No backbone fine-tune yet. Compare full REVE (Kaggle) to CBraMod before deciding.

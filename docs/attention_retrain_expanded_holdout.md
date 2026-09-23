# attention_retrain_expanded_holdout

**Step:** `attention_retrain_expanded_holdout`  
**Completed (UTC):** 2026-09-07  
**Dataset:** OpenNeuro `ds001787` — SPDX **CC0-1.0**

## Split

| Role | Tags |
|------|------|
| Train (experts, both classes, per-subject undersample) | sub001_ses01, sub002_ses01, sub003_ses01, sub004_ses01 |
| Primary holdout (novice) | sub019_ses01 |
| Secondary holdout (novice) | sub013_ses01 |
| Single-class probes | sub014_ses01, sub017_ses01 |

## Headline metrics

| Metric | Value |
|--------|------:|
| val macro-F1 | 0.4250 |
| sub019 balanced macro-F1 | 0.3241 |
| sub019 full macro-F1 | 0.4103 |
| sub019 stride×4 macro-F1 | 0.4087 |
| sub013 balanced macro-F1 | 0.3333 |
| sub013 full macro-F1 | 0.4444 |
| sub013 stride×4 macro-F1 | 0.4342 |
| probe sub014 acc | 1.0000 |
| probe sub017 acc | 1.0000 |
| ship_candidate | False |

## Takeaway

Expanded ds001787 attention retrain (4 expert train → novice holdouts): val F1=0.425; sub019 bal/full/s4 F1=0.324/0.410/0.409; sub013 bal/full/s4 F1=0.333/0.444/0.434. Holdouts still near chance — subject transfer remains weak; do not ship attention head; next: more_ds003969_subjects or personal Muse cal.

## Artifacts

- Script: `scripts/attention_retrain_expanded_holdout.py`
- Head: `exports/attention_retrain_expanded_holdout/head_a_attention_binary_ds001787_expanded.pt`
- Metrics: `exports/attention_retrain_expanded_holdout/metrics_summary.json`
- Manifest: `exports/attention_retrain_expanded_holdout/run_manifest.json`

## Out of scope

- No Kaggle reversion this step (windows already versioned)
- No joint 4-way retrain; no Head B/C training
- Packaging only if ship_candidate (see next invent: repack if true)

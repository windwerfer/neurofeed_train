# Head A 4-way QC trial

> **2026-09-07 architecture lock:** product Head A is **multi-head** ([`head_a_multihead.md`](head_a_multihead.md)). Public MW/concentration frozen decoder remains deferred / personal-cal only.


**Step:** `head_a_4way_qc_trial`
**Completed (UTC):** 2026-09-07T05:49:49.286260+00:00

## Design

| Piece | Choice |
|-------|--------|
| QC | light artifact QC (`qc_pass` only) |
| Splits | fixed subject JSON under `datasets/*/splits/` |
| Head | `HeadALinear` → 4 logits |
| Schedule | alternating masked vigilance / attention batches |
| Encoder | frozen CBraMod (CPU) |
| Epochs | 6 (early stop patience=2) |

## QC drop rates (pre-train)

See `docs/artifact_qc_light.md` / `exports/artifact_qc_light/summary.json`.

- vigilance: drop_rate=0.0000
- attention_ds001787: drop_rate=0.0000
- attention_ds003969: drop_rate=0.0238

## Headline metrics

| Split | macro-F1 (present) |
|-------|-------------------:|
| Val vigilance | 0.6972 |
| Val attention | 0.4847 |
| Test vigilance (subject holdout) | 0.7139 |
| Test vigilance stride×4 | 0.7258 |
| Test attention (subject holdout, full) | 0.5359 |
| Test attention balanced | 0.5443 |
| ship_candidate | False |

## Attention holdout collapsed?

**NO** — not collapsed like the prior joint-train ~0.33. Test attention balanced macro-F1=0.544 is only barely above chance (~0.5); still too weak to ship (`ship_candidate=false`).

## Takeaway

4-way Head A QC trial (fixed subject splits, QC-passed only). Val vig/att F1=0.697/0.485; test vig s4 F1=0.726; test att bal F1=0.544; ship_candidate=False.

## Artifacts

- Script: `scripts/head_a_4way_qc_trial.py`
- Head: `exports/head_a_4way_qc_trial/head_a_4way_qc_trial.pt`
- Metrics: `exports/head_a_4way_qc_trial/metrics_summary.json`
- Manifest: `exports/head_a_4way_qc_trial/run_manifest.json`


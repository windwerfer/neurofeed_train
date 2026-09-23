# LOSO eval for Head A attention (muse4)

**Script:** `scripts/loso_eval_head_a.py`  
**Exports:** `exports/loso_head_a_attention/`  
**Completed (UTC):** 2026-09-07T08:04:30.536281+00:00  
**Montage:** `montage_id=muse4` only (never mix crown8)

## Setup

- Labels: concentration / mind_wandering
- Encoder: frozen CBraMod + tiny `HeadALinear` (5 epochs, balanced undersample)
- QC-pass windows only
- Subject IDs **corpus-qualified** (`ds001787/sub-XXX` | `ds003969/sub-XXX`) — 16 raw ID collisions across corpora
- Pre-encode once, then 77 LOSO folds

## Results (N=80 subjects)

| Metric | Value |
|--------|------:|
| Subjects available | **80** |
| Folds (both-class) | **77** |
| Skipped (single-class) | 3 (ds001787/014, ds001787/017, ds003969/036 after QC) |
| **macro-F1 mean±std** | **0.3610 ± 0.2090** |
| accuracy mean | 0.4909 |
| folds with macro-F1 > 0.5 | 8 / 77 |
| **ship_candidate** | **false** |

Below chance (~0.5 balanced binary macro-F1). Not shippable.

Per-corpus holdout means (informal):
- ds001787: ~0.29 (14 folds)
- ds003969: ~0.38 (63 folds)

## Files

- `fold_plan.json` — fold list + collisions
- `metrics_summary.json` — full per-fold reports
- `folds.jsonl` — compact per-fold lines
- `loso_run.log` — run log

## Usage

```bash
uv run python scripts/loso_eval_head_a.py --dry-run
uv run python scripts/loso_eval_head_a.py --max-folds 3
uv run python scripts/loso_eval_head_a.py
```

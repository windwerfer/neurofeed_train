# Head A attention expansion status (muse4 → ~80 + LOSO)

**Completed (UTC):** 2026-09-07  
**Montage:** `montage_id=muse4` (AF7/AF8/TP9/TP10 model order). Crown8 is separate — never mixed.  
**Plan:** `docs/head_a_40_subjects_plan.md` (target raised to ~80 for reliable LOSO)

## Before → after (this journey)

| Corpus | Subjects before journey | Subjects now | Notes |
|--------|------------------------:|-------------:|-------|
| ds001787 | 12→16 | **16** | probe labels; CC0 |
| ds003969 | 11→19→24 | **64** | Muse-proximal CC0; +40 in toward80 batch |
| **Attention total (corpus-qualified)** | 23→35→40 | **80** | target met |
| vigilance Sleep-EDF | 5 | 5 | frozen; not used as crown |

## Batches

### Close gap 35→40 (`expand_attention_batch40_close_ds003969`)
- +5: sub-011–013 (htr), sub-034–035 (ctr)
- Windows new: 4000; pool ds003969 → 24 subjects / 19200 windows

### Toward 80 (`expand_attention_toward80_ds003969`)
- +40: sub-014–024 (htr), 036–055 (ctr), 056–059 (tm), 060–064 (vip)
- Windows new: 32000; pool ds003969 → **64** subjects / **51200** windows
- Errors: **0**
- Summary: `exports/expand_attention_toward80_ds003969_summary.json`

## ID collisions (document for LOSO)

Raw `sub-00x` IDs collide across corpora (16 shared numbers). **LOSO uses corpus-qualified IDs**  
(`ds001787/sub-XXX` vs `ds003969/sub-XXX`) — different people.

## Montage policy

- Windows stamped `montage_id=muse4` via `scripts/stamp_montage_muse4.py`
- Channel model order: AF7, AF8, TP9, TP10 (ds003969: native AF7/AF8; TP7/TP8→TP9/TP10)
- **Do not** invent crown8 from Sleep-EDF or mix Muse+Crown in one example
- See `docs/montages_muse_crown.md`, `datasets/common/montages.json`

## QC / splits

- `validate_splits.py`: **no leakage** (frozen ds003969 test sub-025/027; val sub-026/028)
- ds003969 QC drop_rate ≈ 2.7% (mostly sub-028/036/037/044 line noise / peak)
- ds001787 QC drop_rate = 0%
- Annotations refreshed

## LOSO (completed)

- Script: `scripts/loso_eval_head_a.py` (pre-encode once; frozen CBraMod + HeadALinear)
- Labels: concentration / mind_wandering; QC-pass only; **montage_id=muse4**
- Exports: `exports/loso_head_a_attention/`
- Subjects: **80**; folds: **77** (skipped 3 single-class)
- **macro-F1 mean±std: 0.3610 ± 0.2090** (accuracy mean 0.4909)
- **ship_candidate: false** (≤ chance; not shippable)
- Details: `docs/loso_head_a.md`, `exports/loso_head_a_attention/metrics_summary.json`

## Open-license ceiling (no HF)

| Source | License | Usable for muse4 attention | Notes |
|--------|---------|---------------------------:|-------|
| ds003969 | CC0 | 98 total; **64 used** | +34 remain if needed |
| ds001787 | CC0 | ~24 pool; **16 used** | +8 remain (probe logs) |
| HF EEGMeditation | CC-BY | gated | not used (user has not accepted) |
| **Ceiling without HF** | | **~114** corpus-qualified | 80 already reached |

## Disk

~84G free after toward80 (~12G used this expansion). Binaries gitignored.

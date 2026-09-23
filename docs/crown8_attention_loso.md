# Crown8 attention LOSO (vs muse4)

**Completed:** 2026-09-07T08:51:47Z (UTC)

## Scientific question

Re-parse the **same** OpenNeuro attention raw (same subjects, labels, and window `starts`) as **Crown8** and ask:

> If Crown8 LOSO ≫ chance while muse4 ≈ chance → montage mattered (Muse-proxy 4-ch was a weak substrate; labels not purely broken).  
> If Crown8 also ≈ chance → **label / task mismatch** is the prime suspect.

## Headline metrics

| Pack | Channels | LOSO folds | macro-F1 mean ± std | accuracy mean | ship_candidate |
|------|----------|------------|---------------------|---------------|----------------|
| **muse4** (ref) | 4 (AF7/AF8/TP9/TP10 proxies) | 77 | **0.361 ± 0.209** | 0.491 | false |
| **crown8** | 8 (CP3,C3,F5,PO3,PO4,F6,C4,CP4) | 67 | **0.351 ± 0.218** | 0.477 | false |

- **Δ(crown8 − muse4) ≈ −0.010** (essentially flat).
- Crown8 median fold macro-F1 ≈ 0.33; only ~10% of folds > 0.5.
- **Verdict:** `crown8_near_or_below_chance_label_task_mismatch_prime_suspect`

### Interpretation

Crown8 does **not** rescue Head A. Matching window indices and labels while swapping to a denser, motor/parietal/occipital Crown montage still yields ~chance LOSO. That points away from “Muse 4-ch montage alone was the failure mode” and toward:

1. **ds003969 block labels** (`med*` → concentration, `think*` → mind_wandering) are a coarse protocol proxy, not verified attentional state.
2. **ds001787 probe Q1/Q2** labels may be noisy / weakly coupled to the pre-Q1 EEG window under this frozen encoder + linear head.
3. Domain gap / frozen CBraMod transfer may still hurt, but the **matched-label Crown control** shows montage upgrade alone is insufficient.

**Not shippable** as an attention head for either montage under current labels.

## What was built

| Step | Artifact |
|------|----------|
| Re-parse | `scripts/reparse_crown8_attention.py` |
| Corpora | `datasets/attention_ds003969_crown8/`, `datasets/attention_ds001787_crown8/` |
| Splits | Copied from muse4 (same subject assignment) |
| Light QC | Same V thresholds; 8-ch windows |
| LOSO | `scripts/loso_eval_head_a_crown8.py` → `exports/loso_head_a_attention_crown8/` |
| Rollup | `exports/crown8_reparse_summary.json` |

### Window counts (aligned to muse4)

| Corpus | Recordings | Windows |
|--------|------------|---------|
| `attention_ds003969_crown8` | 64 | 51200 |
| `attention_ds001787_crown8` | 16 | 5185 |

### Channel mapping

- **ds003969:** BDF / `channels.tsv` already list Crown 10-10 names → pick stream order.
- **ds001787:** BioSemi `A1..B32` → MNE `biosemi64` → pick Crown names.
- Stream order @ 256 Hz: `CP3, C3, F5, PO3, PO4, F6, C4, CP4` (`montage_id=crown8`).
- **Never** mix muse4 and crown8 in one example; Sleep-EDF unused for Crown.

### QC note (Crown vs Muse)

Crown ds003969 drop_rate ≈ **0.22** (line_noise + peak_abs) vs muse4 ≈ 0.027 on the same recordings. Posterior/central channels are noisier under the same absolute thresholds. LOSO uses `qc_pass` only; several subjects became single-class after QC (5 skipped; 67 folds vs muse4’s 77). The near-chance result holds on the surviving set.

## Reproduce

```bash
uv run python scripts/reparse_crown8_attention.py
uv run python scripts/dataset/artifact_qc_windows.py \
  --corpora attention_ds003969_crown8 attention_ds001787_crown8
uv run python scripts/loso_eval_head_a_crown8.py
```

## Next (label-side)

- Prefer probe-level or continuous ratings over block proxies where available.
- Audit ds003969 think/med blocks for behavioral confirmation.
- Consider subject-calibrated Head B only after cleaner labels — not more montage churn.

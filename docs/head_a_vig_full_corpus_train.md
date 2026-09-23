# Head A-vig train — expanded sleep corpus (frozen CBraMod)

**Step:** `head_a_vig_full_corpus_train`  
**Date (UTC):** 2026-09-08T13:31:01.772919+00:00  
**ship_candidate:** **True** — ship_candidate for muse4-proxy Sleep-EDF/HMC vig decoder only; true Muse domain adaptation still recommended before consumer ship. Metrics clear subject-holdout bar.

## Corpus

- Same pool as Head C: `datasets/vigilance_sleep_edf/` (not SC400-only)
- Sleep-EDF (ODC-By) + HMC (CC-BY-4.0 ship-OK)
- Tags: `(ch_count=4, professional)` muse4 proxy
- Subjects: 124 / recordings: 125
- Labels: W→`drowsy`, N1→`hypnagogic`
- Splits: policy A subject-wise (SC400 test anchor, SC403 val anchor)

## Setup

- Encoder: **frozen** CBraMod (shared emb cache with Head C)
- Head: `HeadAVigLinear` (2-way)
- Sampling: full corpus after light QC (no per-recording cap); train undersample balanced (cap 80000)
- Backbone fine-tune: **no** (user decides after metrics)
- Skipped: A-med / A-eng ship training

## Metrics

| Split | n | accuracy | macro-F1 |
|-------|--:|---------:|---------:|
| train (fit) | 80000 | 0.747 | 0.746 |
| val | 57215 | 0.664 | 0.625 |
| test | 61736 | 0.768 | 0.747 |

### Val per-class

| class | precision | recall | f1 | support |
|-------|----------:|-------:|---:|--------:|
| drowsy | 0.866 | 0.656 | 0.747 | 43114 |
| hypnagogic | 0.396 | 0.690 | 0.503 | 14101 |

### Test per-class

| class | precision | recall | f1 | support |
|-------|----------:|-------:|---:|--------:|
| drowsy | 0.894 | 0.759 | 0.821 | 43109 |
| hypnagogic | 0.586 | 0.791 | 0.673 | 18627 |

## vs prior SC400-only / night-holdout

Prior Head A binary on SC4001→SC4002 night holdout had bal macro-F1 ~0.74–0.82 (same-subject nights).
This run is **subject-wise** holdout on 124 subjects — harder / more honest generalization.

## Recommendation

**Freeze vs fine-tune (user decision):**

- **A-vig (CBraMod):** subject-holdout test macro-F1=0.747 clears muse4-proxy ship bar → **prefer keep backbone frozen** for vig; fine-tune only if true-Muse cal arrives or domain gap hurts.
- **Head C (CBraMod):** 2-way wake/light on N1-slice test F1=0.760; **ship_candidate=False** for product 4-way staging (deep/REM absent). Do **not** fine-tune backbone for staging until sleep-period windows exist.
- **A-eng:** KEEP — corpus growth in progress toward ~120 unique (ds007169 + ds007262 + EEGMAT + STEW-HF + ds007554). Train frozen CBraMod+REVE after corpus lands; decide fit vs misfit then. No fine-tune yet.
- **REVE:** next — Kaggle T4 with private muse-eeg-heads-cache (offline); compare vs CBraMod before any fine-tune.
- **Skipped:** Head B, A-med/MW/depth/stress public heads.


## Artifacts

- Weights: `exports/head_a_vig_full_corpus/head_a_vig_linear.pt`
- Metrics: `exports/head_a_vig_full_corpus/metrics_summary.json`
- Shared emb: `exports/head_c_full_corpus/emb_cache/`
- Script: `scripts/train_heads_expanded_sleep_corpus.py`

# Freeze heads progress — CBraMod + REVE (A-vig / Head C / A-eng)

**Date (UTC):** 2026-09-08T13:45:30.205283+00:00  
**Fine-tune:** **none yet**

## CBraMod (full 124-subject sleep corpus, frozen)

| Head | Val macro-F1 | Test macro-F1 | Test acc | ship_candidate |
|------|-------------:|--------------:|---------:|:--------------:|
| A-vig | 0.625 | 0.747 | 0.768 | **True** |
| Head C wake/light | 0.648 | 0.760 | — | **False** (N1-slice 2-way) |

Sampling: full after light QC; train undersample balanced (cap 80k).  
Docs: `docs/head_a_vig_full_corpus_train.md`, `docs/head_c_full_corpus_train.md`.

## REVE (local CPU subsample — directional)

| Head | Val macro-F1 | Test macro-F1 | ship_candidate |
|------|-------------:|--------------:|:--------------:|
| A-vig | 0.709 | 0.783 | **True** |
| Head C wake/light | 0.707 | 0.791 | False |

**Caveat:** ≤80 windows/class/recording → val/test ~balanced and much smaller than CBraMod full holdout. **Not a fair encoder bake-off.** Full REVE encode on Kaggle T4 is the authoritative compare (`kaggle_kernel_07_reve_sleep_heads`).

## A-eng corpus

**121 unique persons** / 138 packs / ~18k windows — ready to train.  
Sources: ds007169+ds007262 (Barras once) + EEGMAT + STEW-HF + ds007554.  
Splits: train 85 / val 18 / test 18 under `datasets/engagement_a_eng/`.  
Next: frozen CBraMod + REVE A-eng → fit vs misfit.

## Recommendation

KEEP BACKBONES FROZEN for now. CBraMod A-vig full subject-holdout test macro-F1=0.747 (ship_candidate True for muse4-proxy vig). REVE local subsample looks strong (test F1~0.78) but is capped — confirm on Kaggle T4 full encode before choosing REVE vs CBraMod. Head C remains 2-way wake/light probe (ship False for 4-way). A-eng corpus ready at 121 unique — next train frozen CBraMod+REVE A-eng for fit vs misfit. Do NOT fine-tune until A-eng frozen metrics land and full REVE sleep compare is in.

## Skipped

Head B; A-med / MW / depth / stress public heads.

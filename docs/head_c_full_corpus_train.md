# Head C train — expanded sleep corpus (frozen CBraMod)

**Step:** `head_c_full_corpus_train`  
**Date (UTC):** 2026-09-08T13:31:01.772919+00:00  
**ship_candidate:** **False** — Head C on N1-slice corpus is 2-way wake/light only (deep/REM absent). Not a product 4-way stage decoder; treat as transfer probe / label sanity. test macro-F1=0.760.

## Corpus

- Path: `datasets/vigilance_sleep_edf/`
- Sources: Sleep-EDF Expanded (ODC-By) + HMC (CC-BY-4.0)
- Montage tags: `(ch_count=4, professional)` muse4 proxy (Sleep-EDF Fpz-Cz/Pz-Oz; HMC F4/C4/C3/O2)
- Recipe: N1-slice (pre 20 min / post 40 min); windows 2 s @ 256 Hz
- Subjects: 124 unique / recordings used: 125
- Splits: policy A (SC400=test, SC403=val anchors); subject-wise no leakage

## Task honesty

N1-slice labeled windows only contain **wake** and **light** (W / N1).
**deep / rem are absent** → Head C trained as **2-way** `wake`/`light` (subset of `stage_coarse`).
Full 4-way Head C needs sleep-period windows (out of scope; no backbone fine-tune this pass).

## Setup

- Encoder: **frozen** CBraMod (embed-once → head train)
- Head: `HeadCLinear`
- Sampling: full corpus after light QC (no per-recording cap); train undersample balanced (cap 80000)
- Backbone fine-tune: **no**

## Metrics

| Split | n | accuracy | macro-F1 |
|-------|--:|---------:|---------:|
| train (fit) | 80000 | 0.754 | 0.753 |
| val | 57215 | 0.691 | 0.648 |
| test | 61736 | 0.782 | 0.760 |

### Val per-class

| class | precision | recall | f1 | support |
|-------|----------:|-------:|---:|--------:|
| wake | 0.871 | 0.693 | 0.772 | 43114 |
| light | 0.423 | 0.687 | 0.523 | 14101 |

### Test per-class

| class | precision | recall | f1 | support |
|-------|----------:|-------:|---:|--------:|
| wake | 0.897 | 0.776 | 0.832 | 43109 |
| light | 0.605 | 0.794 | 0.687 | 18627 |

## Recommendation

**Freeze vs fine-tune (user decision):**

- **A-vig (CBraMod):** subject-holdout test macro-F1=0.747 clears muse4-proxy ship bar → **prefer keep backbone frozen** for vig; fine-tune only if true-Muse cal arrives or domain gap hurts.
- **Head C (CBraMod):** 2-way wake/light on N1-slice test F1=0.760; **ship_candidate=False** for product 4-way staging (deep/REM absent). Do **not** fine-tune backbone for staging until sleep-period windows exist.
- **A-eng:** KEEP — corpus growth in progress toward ~120 unique (ds007169 + ds007262 + EEGMAT + STEW-HF + ds007554). Train frozen CBraMod+REVE after corpus lands; decide fit vs misfit then. No fine-tune yet.
- **REVE:** next — Kaggle T4 with private muse-eeg-heads-cache (offline); compare vs CBraMod before any fine-tune.
- **Skipped:** Head B, A-med/MW/depth/stress public heads.


## Artifacts

- Emb cache: `exports/head_c_full_corpus/emb_cache/`
- Weights: `exports/head_c_full_corpus/head_c_wake_light_linear.pt`
- Metrics: `exports/head_c_full_corpus/metrics_summary.json`
- Script: `scripts/train_heads_expanded_sleep_corpus.py`

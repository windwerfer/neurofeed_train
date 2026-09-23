# REVE team published sleep-staging benchmarks

**Sources (verified 2026-09-09):**
- Paper HTML: https://arxiv.org/html/2510.21585
- Paper PDF: https://arxiv.org/pdf/2510.21585 (also NeurIPS proceedings)
- Project: https://brain-bzh.github.io/reve/
- HF card `brain-bzh/reve-base`: ISRUC accuracy **0.782** (matches Table 2 / Table 13 bal. acc. rounded)

**Paper:** El Ouahidi et al., *REVE: A Foundation Model for EEG*, NeurIPS 2025.

## Table 1 — sleep datasets (paper)

| Dataset | # Channels | Duration | # Samples | Rate | # Classes |
|---------|-----------:|---------:|----------:|-----:|----------:|
| **HMC** | 4 | 30s | 137,243 | 256 Hz | 5 |
| **ISRUC** | 6 | 30s | 89,240 | 200 Hz | 5 |

Classes (AASM): Wake, N1, N2, N3, REM.

### Channel sets (from prior work / TorchEEG defaults; paper cites CBraMod/LaBraM/BIOT protocols)

- **HMC:** 4 EEG bipolar derivations typically `EEG F4-M1`, `EEG C4-M1`, `EEG O2-M1`, `EEG C3-M2` (PhysioNet HMC). REVE position bank uses electrode labels **F4, C4, O2, C3**.
- **ISRUC (Subgroup 1):** 6 EEG channels @ 200 Hz. Paper notes a baseline bug that used a chin electrode; **REVE results exclude the chin electrode**.

## Eval protocol (paper)

- **Splits:** Follow CBraMod / LaBraM / BIOT protocols for fair comparison (same train/val/test subject splits as prior studies; same preprocessing pipeline as baselines).
- **ISRUC (Appendix C.6):** Subjects **1–80 train / 81–90 val / 91–100 test**. Framed as **sequence-to-sequence** with **20 consecutive 30 s epochs** (for fine-tune / main Table 2). Freely accessible.
- **HMC (Appendix C.6):** 151 full-night PSG recordings @ 256 Hz; 5 stages scored by technicians. Paper does **not** list explicit SN### subject IDs in the appendix text; states consistency with prior baseline splits. CC-BY-4.0.
- **Primary metric:** balanced accuracy (± std). Appendix also reports **Cohen’s κ** and **weighted F1**.
- **Fine-tune recipe:** two-step continuous run — (1) linear probe with frozen encoder, then (2) unfreeze + LoRA on QKVO, Mixup, dropout, warmup + Reduce-on-Plateau; optional model souping.
- **Linear probe (Table 4):** frozen backbone; report **with mean pooling over tokens (Pool)** vs **without pooling** (classification head on full token sequence, matching CBraMod’s published LP pipeline).

## Main fine-tune numbers — REVE-Base (Tables 2 / 13 / 14)

These are the headline **fine-tuned** (not frozen) results.

### ISRUC (5-class) — Table 13 / Table 2

| Method | Balanced Acc | Cohen’s κ | Weighted F1 |
|--------|-------------:|----------:|------------:|
| CBraMod | 0.7865 ± 0.0110 | 0.7442 ± 0.0152 | 0.8011 ± 0.0099 |
| **REVE-Base*** | **0.7819 ± 0.0078** | **0.7500 ± 0.0156** | **0.8005 ± 0.0135** |

\*Without chin electrode (see paper footnote). HF card lists ISRUC accuracy **0.782**.

### HMC (5-class) — Table 14

| Method | Balanced Acc | Cohen’s κ | Weighted F1 |
|--------|-------------:|----------:|------------:|
| BIOT | 0.6862 ± 0.0041 | 0.6295 ± 0.0113 | 0.7091 ± 0.0147 |
| LaBraM-Base | 0.7286 ± 0.0101 | 0.6812 ± 0.0073 | 0.7554 ± 0.0024 |
| **REVE-Base** | **0.7401 ± 0.0075** | **0.6982 ± 0.0078** | **0.7638 ± 0.0074** |

Note: Table 14 does **not** list CBraMod for HMC. Paper emphasizes that REVE generalizes from 10 s pretraining patches to **30 s** sleep inputs via 4D positional encoding.

## Linear probe / frozen — REVE-Base (Table 4) — balanced accuracy

| Dataset | REVE-B (Pool) | REVE-B (no pool) | REVE-L (Pool) | REVE-L (no pool) | CBraMod (Pool) | CBraMod (no pool) |
|---------|--------------:|-----------------:|--------------:|-----------------:|---------------:|------------------:|
| **ISRUC** | **0.697 ± 0.011** | **0.662 ± 0.030** | 0.743 ± 0.004 | 0.758 ± 0.001 | 0.407 ± 0.049 | 0.430 ± 0.043 |
| **HMC** | **0.647 ± 0.008** | **0.604 ± 0.008** | 0.703 ± 0.003 | 0.710 ± 0.007 | 0.368 ± 0.001 | 0.538 ± 0.009 |

Table 4 does **not** publish κ / weighted F1 for linear probe (only balanced accuracy). For frozen vs FT ablations on REVE-**Small**, see Table 17 (ISRUC LP ~0.699 / FT ~0.777; HMC LP ~0.598 / FT ~0.713) — different model size, not Base.

## What “strong on sleep linear probe” means in the paper

- On **HMC**, frozen REVE-Base with pooling reaches **~0.65 bal. acc.** vs CBraMod LP ~0.37–0.54 — large frozen-embedding gap favoring REVE.
- On **ISRUC**, frozen REVE-Base (Pool) **~0.70** vs CBraMod LP ~0.41–0.43.
- Fine-tune closes the gap / reaches SOTA-ish (ISRUC REVE slightly below CBraMod FT bal. acc.; HMC REVE above LaBraM/BIOT).

## Implications for our ship path

Our product path is **frozen REVE-base + tiny linear heads** (no backbone FT). The paper’s **Table 4 HMC/ISRUC LP** numbers are the fair published reference — **not** the fine-tune Table 13/14 headlines.

See `exports/reve_hmc_paper_compare/` for our matched HMC 5-stage frozen linear-probe run.

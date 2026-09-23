# Datasets catalog

Muse-oriented and transfer datasets for Head A (Attention + Vigilance) and Head B (artifacts).  
Channels of interest when present: **AF7, AF8, TP9, TP10**.

**Head A architecture (2026-09-07):** multi-head A-vig / A-med / A-eng — see [`head_a_multihead.md`](head_a_multihead.md). Catalog rows below still list legacy attention corpora used for research MW and A-med candidates.


## ALLOW (shipping-eligible — open licenses)

| Dataset | URL | License / access | Muse notes | Primary use |
|---------|-----|------------------|------------|-------------|
| **OpenNeuro ds003969** | https://openneuro.org/datasets/ds003969 | **CC0** (verified) | AF7/AF8 native; TP7/TP8→TP9/TP10 proxy | Attention / cognitive states → Head A |
| **OpenNeuro ds001787** | https://openneuro.org/datasets/ds001787 | **CC0** (verified) | BioSemi 64 @ 256 Hz; map to Muse; **Q1>Q2→conc / Q1<Q2→MW** (see codebook confirm) | Attention / task EEG → Head A |
| **HF EEGMeditation** | https://huggingface.co/datasets/alexeykashevnik/EEGMeditation | **CC BY 4.0** (gated) | 64-ch 10–10 @ 2048 Hz; explicit conc/MW tasks | Head A (concentration / mind_wandering) |
| **Sleep-EDF Expanded** | https://physionet.org/content/sleep-edfx/ | PhysioNet credentialed / open research | Not Muse; Fpz-Cz / Pz-Oz — use for vigilance / drowsy / hypnagogic **transfer** only | Head A vigilance (drowsy, hypnagogic) |
| **HMC (optional)** | Search “HMC sleep EEG” / hospital sleep corpora with open terms | Varies — confirm before use | Optional sleep staging transfer | Head A vigilance |
| **OpenNeuro ds007169** | https://openneuro.org/datasets/ds007169 | **CC0** | 19-ch mobile; F7/F8/T3/T4→muse4 proxy | A-eng load/engagement research |
| **OpenNeuro ds007262** | https://openneuro.org/datasets/ds007262 | **CC0** | Same Barras cohort as ds007169; arithmetic difficulty randomized | A-eng (order-friendlier Barras task) |
| **PhysioNet EEGMAT** | https://physionet.org/content/eegmat/ | **ODC-By-1.0** | 23-ch 10–20; F7/F8/T3/T4→muse4 | A-eng rest vs arithmetic |
| **STEW (HF processed)** | https://huggingface.co/datasets/monster-monash/STEW | **CC-BY-4.0** (raw IEEE login) | Emotiv 14-ch hobbyist; AF3/AF4/T7/T8→muse4 | A-eng rating-bin low/high |
| **OpenNeuro ds007554** | https://openneuro.org/datasets/ds007554 | **CC0** | 32-ch; F7/F8/T7/T8→muse4 | A-eng PassiveMotor/MI vs NbackArith/n-back |
| **Schreer Muse-S** | https://doi.org/10.7910/DVN/V2CWJW (Harvard Dataverse) | **CC0-1.0** (verified 2026-09-07) | **Native Muse-S** Mind Monitor CSV — channel map & heuristic artifacts (no cued blink/jaw) | Head B weak-label bootstrap; personal cal remains gold |

### Quick links

- OpenNeuro ds003969: https://openneuro.org/datasets/ds003969  
- OpenNeuro ds001787: https://openneuro.org/datasets/ds001787  
- HF alexeykashevnik/EEGMeditation: https://huggingface.co/datasets/alexeykashevnik/EEGMeditation  
- Sleep-EDF Expanded: https://physionet.org/content/sleep-edfx/  
- Schreer Muse-S: https://doi.org/10.7910/DVN/V2CWJW  

## DENY (not in shipping training mix)

| Dataset | Decision | Why |
|---------|----------|-----|
| **LUNA** | **DENY** | Explicitly excluded from this project — no LUNA in shipping mix or published artifacts |
| **L-FAME** | **DENY** | BY-NC — not suitable for shipping redistribution |
| **SEED-VIG** | **DENY** | Academic-only / restricted — do not bundle in shipping builds |
| Any **BY-NC** or unclear academic-only set | **DENY** | Prefer CC0 / CC BY / ODC-By / MIT / BSD only |

Do **not** train published heads on DENY sets. Offline research on DENY data is out of scope for the shipping pipeline. See [`LICENSE_NOTES.md`](LICENSE_NOTES.md).

## Practical notes

1. Prefer **subject-wise** splits on all corpora.
2. Resample to a common rate (e.g. 256 Hz) before windowing; document the rate in export metadata.
3. For non-Muse sleep sets, train vigilance features with domain adaptation or as auxiliary pretrain — do not claim native Muse geometry without remapping.
4. Artifact labels for Head B: prefer Muse-S / personal calibration over proxy EOG when possible.

## Derived private windows (project)

| Artifact | URL | Notes |
|----------|-----|-------|
| **muse-eeg-heads-windows** | https://www.kaggle.com/datasets/windwerfer/muse-eeg-heads-windows | Private. Sleep-EDF N1-slice pool (SC4001–SC4041) + **ds001787** attention pre-Q1 Muse-proxy npzs. See [`windows_both_nights.md`](windows_both_nights.md), [`ds001787_attention_ingest.md`](ds001787_attention_ingest.md). |
| **muse-eeg-heads-cache** | https://www.kaggle.com/datasets/windwerfer/muse-eeg-heads-cache | Private. CBraMod + **REVE-base/positions** offline snapshots (gated — no redistribute). See [`reve_model_cache.md`](reve_model_cache.md). |

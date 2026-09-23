# Crown vigilance HMC compare (Head A-vig, frozen CBraMod)

Created (UTC): 2026-09-23T09:32:01.758644+00:00
Ship bar (test macro-F1): **0.6**

## Setup
- Corpus: PhysioNet hmc-sleep-staging 1.1 (CC-BY-4.0), N1-slice (2s@256Hz, hop 0.5s, pre 20 / post 40 min, majority 0.7)
- Labels: W→drowsy, N1→hypnagogic
- Encoder: frozen CBraMod (channel-flexible; mean pool → 200-d). True C=2 / C=4 — **no zero-pad to 8**.
- Splits: subject-wise ~70/15/15 (seed=42), identical subjects across montages (151 subjects: 106/23/22).

## Metrics

| montage | C | channels | n_subj | val macro-F1 | test macro-F1 | ship (≥0.60) |
|---|---:|---|---:|---:|---:|---|
| crown2_strong | 2 | C3, C4 | 151 | 0.641 | 0.670 | yes |
| crown4_hmc | 4 | C3, C4, F6, PO4 | 151 | 0.668 | 0.680 | yes |

## Per-montage notes

### crown2_strong
- test macro-F1 0.670 >= 0.6
- test true counts: {'drowsy': 37363, 'hypnagogic': 17159}
- test pred counts: {'drowsy': 33129, 'hypnagogic': 21393}

### crown4_hmc
- test macro-F1 0.680 >= 0.6
- test true counts: {'drowsy': 37045, 'hypnagogic': 16843}
- test pred counts: {'drowsy': 35301, 'hypnagogic': 18587}

## ISRUC (out of scope)

ISRUC is **out of scope** for this public release. HMC alone met the ≥0.60 Crown vig proxy bar. Do not publish ISRUC windows or heads in the public neurofeed repos.

## Artifacts
- Packs: `datasets/vigilance_hmc_crown2`, `datasets/vigilance_hmc_crown4`
- Scripts: `scripts/expand_hmc_crown_vig.py`, `scripts/train_hmc_crown_vig_compare.py`
- Metrics/heads: `exports/hmc_crown_vig_compare/`
- Training: local CPU (Kaggle private dataset uploaded but still size=0 / processing; kernel ERROR on empty attach).

## Maintainer scratch (optional)

Private Kaggle datasets/kernels were used as GPU scratch during development. They are **not required** for public use and must not redistribute gated REVE weights. Prefer HF `windwerfer/neurofeed-eeg-windows` configs `crown2_vigilance_hmc` / `crown4_vigilance_hmc`.


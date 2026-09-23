# Crown vigilance HMC — Head A-vig, frozen REVE-base

Created (UTC): 2026-09-23T12:28:47.440345+00:00
Ship bar (test macro-F1): **0.6**
Device: `cuda`
Sampling: full after QC (per_class_cap=0); train undersample balanced (cap 80000)

## Setup
- Corpus: PhysioNet hmc-sleep-staging 1.1 (CC-BY-4.0), N1-slice (2s@256Hz, hop 0.5s, pre 20 / post 40 min, majority 0.7)
- Labels: W→drowsy, N1→hypnagogic
- Encoder: **frozen REVE-base** (private local/Kaggle cache; do not redistribute weights). Resample 256→200 Hz + 3D positions → attention-pool **512-d**.
- Head: linear `HeadAVigLinear` on frozen embeddings; early stop on val macro-F1; train undersample balance (cap 80k).
- Splits: subject-wise ~70/15/15, identical subjects across montages (same as CBraMod compare).
- ISRUC out of scope for public ship.

## Metrics (REVE)

| montage | C | channels | n_subj | val macro-F1 | test macro-F1 | ship (≥0.60) |
|---|---:|---|---:|---:|---:|---|
| crown2_strong | 2 | C3, C4 | 151 | 0.647 | 0.649 | yes |
| crown4_hmc | 4 | C3, C4, F6, PO4 | 151 | 0.679 | 0.681 | yes |

## Side-by-side vs CBraMod (same packs / splits)

CBraMod refs from `exports/hmc_crown_vig_compare/` (crown2 test 0.670 / crown4 0.680).

| montage | CBraMod val | CBraMod test | REVE val | REVE test | Δ test (REVE−CBraMod) |
|---|---:|---:|---:|---:|---:|
| crown2_strong | 0.641 | 0.670 | 0.647 | 0.649 | -0.021 |
| crown4_hmc | 0.668 | 0.680 | 0.679 | 0.681 | +0.001 |

## Per-montage notes

### crown2_strong
- test macro-F1 0.649 >= 0.6
- test true counts: {'drowsy': 37363, 'hypnagogic': 17159}
- test pred counts: {'drowsy': 34628, 'hypnagogic': 19894}

### crown4_hmc
- test macro-F1 0.681 >= 0.6
- test true counts: {'drowsy': 37045, 'hypnagogic': 16843}
- test pred counts: {'drowsy': 32971, 'hypnagogic': 20917}

## Artifacts

- Script: `scripts/train_hmc_crown_vig_reve.py`
- Metrics/heads/emb cache: `exports/hmc_crown_vig_reve/`
- CBraMod compare: `exports/hmc_crown_vig_compare/`, `docs/crown_hmc_vig_compare.md`
- REVE-base: user-local gated download only (never redistributed in public packs).

## Maintainer scratch (optional)

Private Kaggle datasets/kernels were used as GPU scratch during development. They are **not required** for public use and must not redistribute gated REVE weights. Prefer HF `windwerfer/neurofeed-eeg-windows` configs `crown2_vigilance_hmc` / `crown4_vigilance_hmc`.

## License / redistribute

HMC windows CC-BY-4.0. REVE-base gated — keep weights under private Kaggle / local cache; shipping heads-only is fine later.


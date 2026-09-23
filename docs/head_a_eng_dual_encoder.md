# Head A-eng dual-encoder fit vs misfit (CBraMod + REVE)

**Updated (UTC):** 2026-09-08T14:20:42.521972+00:00

## Task

Frozen heads only (no backbone fine-tune) on `engagement_a_eng` (133 unique persons, muse4, person-wise 93/20/20).

## Headline metrics (test macro-F1 vs chance 0.5)

| Encoder | test n | test macro-F1 | delta vs chance | ship_candidate |
|---------|-------:|--------------:|------------:|:--------------:|
| **CBraMod** | 3084 | 0.548 | +0.048 | **False** |
| **REVE** | 3084 | 0.590 | +0.090 | **False** |

## CBraMod detail

- Docs: `docs/head_a_eng_train_cbramod.md`
- Exports: `exports/head_a_eng_train_cbramod/`
- ship_reason: test macro-F1 0.548 < 0.55; cleaner source ds007262 test macro-F1 0.436 < 0.55. Residual order confounds remain on ds007169 (L1→L4), eegmat (rest→arith), and STEW (rest→SIMKAP). Domain mix: professional 10–20/32-ch + STEW Emotiv hobbyist. muse4 proxy montage — not true Muse-native.

## REVE detail

- Docs: `docs/head_a_eng_train_reve.md`
- Exports: `exports/head_a_eng_train_reve/`
- ship_reason: Overall metrics clear soft bar but cleaner randomized source (ds007262) does not independently clear 0.55 — risk that F1 is partly order/time shortcut. Residual order confounds remain on ds007169 (L1→L4), eegmat (rest→arith), and STEW (rest→SIMKAP). Domain mix: professional 10–20/32-ch + STEW Emotiv hobbyist. muse4 proxy montage — not true Muse-native.
- Prefer Kaggle T4; local CPU used if Kaggle P100 incompatible with current PyTorch.
- Kaggle run (`muse-eeg-heads-reve-a-eng`) also COMPLETE: accelerator=Tesla P100-PCIE-16GB, device_used=cpu, test macro-F1=0.590, ship=False. Artifacts under `exports/head_a_eng_train_reve/kaggle_run/`.


## Domain mix & confounds

- Professional (ds007169/262/554, eegmat) + STEW hobbyist Emotiv.
- Residual order confounds: ds007169, eegmat, STEW; cleaner: ds007262 randomized.

## Combined recommendation

**Fit vs misfit (A-eng public head) — both encoders:**

- **MISFIT / do not ship** for public frozen A-eng on either encoder.
- CBraMod test macro-F1=0.548 (delta +0.048 vs chance 0.5); REVE=0.590 (delta +0.090).
- Cleaner source **ds007262** (randomized difficulty) stays weak on both (CBraMod 0.436, REVE 0.515) while order-confounded **ds007169** is strong — classic time/order shortcut risk.
- Domain mix: professional corpora + STEW hobbyist Emotiv; muse4 proxy only.
- **Do not fine-tune backbone** to chase this signal. Optional next: ds007262-only / order-aware ablations; true-Muse personal cal for product UX.
- Skipped this pass: Head B, A-med.

## Test by source (macro-F1)

| source | CBraMod | REVE | note |
|--------|--------:|-----:|------|
| ds007169 | 0.727 | 0.552 | order confound L1→L4 |
| ds007262 | 0.436 | 0.515 | randomized difficulty (cleaner) |
| ds007554 | 0.375 | 0.552 | task-order residual |
| eegmat | 0.425 | 0.591 | rest→arith order |
| stew | 0.561 | 0.616 | hobbyist; rest→SIMKAP order |

## Parallel work note

- Head C + A-vig **CBraMod** full-corpus trains were already finished (`exports/head_c_full_corpus/`, `exports/head_a_vig_full_corpus/`); not duplicated.
- REVE sleep kernels (`muse-eeg-heads-reve-sleep-*`) are **ERROR** (src path / P100 issues) — out of scope for this A-eng pass.
- REVE A-eng metrics above are from **local CPU** embed-once using private `muse-eeg-heads-cache` offline weights (Kaggle T4 preferred; CLI often got P100 incompatible with current Kaggle PyTorch).

## Artifacts

- `exports/head_a_eng_train_cbramod/`
- `exports/head_a_eng_train_reve/`
- Scripts: `scripts/train_head_a_eng_cbramod.py`, `scripts/train_head_a_eng_reve.py`
- Kaggle: `kaggle_kernel_08_reve_a_eng/` + dataset `windwerfer/muse-eeg-heads-aeng`

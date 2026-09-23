# Gap analysis: ours vs REVE-team sleep staging

**Date:** 2026-09-09 (Asia/Bangkok)  
**Primary fair task:** frozen REVE-base + linear probe on **HMC 5-stage, 30 s @ 256 Hz**  
**Artifacts:** `docs/reve_team_sleep_benchmark.md`, `exports/reve_hmc_paper_compare/`

## Parent summary table (filled 2026-09-09 06:09 ICT)

| Setting | bal_acc | kappa | macro-F1 | weighted-F1 | n / subjects |
|---------|--------:|------:|---------:|------------:|--------------|
| **Ours** HMC 5-stage LP (frozen, mean pool, test) | **0.649** | **0.604** | **0.654** | **0.696** | 3640 epochs / 4 test nights (24 local total) |
| Ours HMC 5-stage LP (val) | 0.641 | 0.563 | 0.627 | 0.681 | 3606 / 4 val nights |
| **Paper** HMC LP Pool (Table 4) | 0.647 ± 0.008 | — | — | — | 151 / 137,243 epochs |
| **Paper** HMC LP no-pool (Table 4) | 0.604 ± 0.008 | — | — | — | same |
| **Paper** HMC fine-tune (Table 14) | 0.7401 ± 0.0075 | 0.6982 | — | 0.7638 | same |
| **Paper** ISRUC LP Pool (Table 4) | 0.697 ± 0.011 | — | — | — | 100 subj / 89,240 |
| **Paper** ISRUC FT (Table 13) | 0.7819 ± 0.0078 | 0.7500 | — | 0.8005 | HF card 0.782 |
| **Ours secondary** Head C wake/light N1-slice (frozen REVE) | test acc 0.792 / macro-F1 0.791 | — | 0.791 | — | Sleep-EDF+HMC 2-way; **not** comparable to 5-stage |

**Gap vs paper HMC LP Pool:** +0.0017 bal_acc (essentially matched).

## Do we reach the same conclusion?

**Paper claim:** REVE is **strong under linear probing** on sleep (HMC/ISRUC), with a large gap vs CBraMod frozen embeddings (Table 4: HMC REVE-B Pool 0.647 vs CBraMod Pool 0.368 / no-pool 0.538).

**Our ship path:** frozen REVE + tiny heads — so Table 4 (not FT Table 14) is the right bar.

Expected honest outcomes:
1. If our HMC-24 LP bal_acc is in the **~0.55–0.70** band → same qualitative conclusion (REVE frozen embeddings carry sleep stage info under a linear head), with quantitative gap explained by subsample/protocol.
2. If we land **≪0.50** → mismatch (montage naming, preprocessing, split leakage, class imbalance, encode bug) — investigate before claiming replication.
3. We **do not** expect to match FT 0.74 without LoRA/unfreeze/Mixup/souping.

## Why match / mismatch (checklist)

| Factor | Paper | Ours | Impact |
|--------|-------|------|--------|
| **Corpus size** | HMC 151 nights, 137,243 epochs | **24** local nights (SN001–025 minus SN014) | High — variance + distribution shift |
| **Subject split** | CBraMod/LaBraM/BIOT fixed splits (IDs not listed in REVE appendix for HMC) | Deterministic ~70/15/15 on sorted local SN### | Medium — different test subjects |
| **Windowing** | 30 s epochs; ISRUC FT uses **20-epoch sequences**; LP Table 4 is embedding+linear | Per-epoch 30 s, **no** sequence model | Medium for FT; low for LP |
| **Channels** | 4 ch HMC (F4-M1, C4-M1, O2-M1, C3-M2) | Same EDF labels; REVE pos bank names **F4,C4,C3,O2** | Low if names resolve (verified in bank) |
| **Sampling** | 256 Hz listed; baselines resample (often → 200 Hz for foundation models) | 256 → REVE native **200 Hz** inside encoder | Low (matches REVE native) |
| **Pooling** | Table 4 Pool vs no-pool | **mean** pool over (C, patches) → 512-d (attention optional) | Medium — compare to Pool column |
| **Probe** | Paper linear layer (torch CE; hyperparams in appendix) | sklearn multinomial LR, C∈{0.1,1,10} on val | Low–medium |
| **Preprocess** | “same pipeline as baselines” | 0.5–30 Hz band-pass (TorchEEG HMC default) | Medium if baselines used different filters |
| **Class set** | 5-stage full night | 5-stage full night (primary) | — |
| **Secondary Head C** | n/a | N1-slice **2-way** wake/light @ 2 s, Muse4 proxy montage, Sleep-EDF+HMC mix | **Not comparable** to paper 5-stage |

## Secondary: our Head C / A-vig (document only)

From `exports/reve_sleep_heads_local/metrics_summary.json` (CPU subsample ≤80 windows/class/recording):

- **A-vig** (drowsy/hypnagogic): test acc 0.785, macro-F1 0.783 — ship_candidate True
- **Head C** (wake/light on N1-slice): test acc 0.792, macro-F1 0.791 — ship_candidate False

These use **2 s** Muse-proxy windows around first N1, not full-night 30 s 5-stage. High 2-way scores do **not** imply paper-level 5-stage LP.

## Kaggle T4 note

Prior kernel `muse-eeg-heads-reve-sleep-heads` **ERROR**ed: (1) assigned **P100** (sm_60 incompatible with current Kaggle torch CUDA), (2) missing `/kaggle/input/muse-eeg-heads-src`. Prefer **UI Run All on T4** with private `muse-eeg-heads-cache` + synced src/windows packs. Local CPU full-night HMC encode is feasible (~tens of min for 24 nights) and is the authoritative compare in this run if T4 is unavailable.

## Conclusion

- Same conclusion? **YES** — frozen REVE-base linear probe is strong on HMC 5-stage sleep.
- Closest paper number: Table 4 HMC Pool **0.647**; ours **0.649** (Delta +0.0017).
- We do **not** claim FT Table 14 (0.74); ship path stays frozen.
- Dominant residual caveats: 24 vs 151 nights, our own subject split (paper HMC SN IDs not listed), sklearn LR vs paper torch head, 0.5-30 Hz filter. Despite that, bal_acc landed on the paper Pool number.
- Head C 2-way N1-slice remains a **different task** (directional only).

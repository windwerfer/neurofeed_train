# Docs index — Muse EEG Heads

**Entry point for any bot** (R&D or tutor). Project root: `/workspace/muse-eeg-heads`.

## What this repo is

Flutter **Muse / Crown** EEG **heads** on **frozen** foundation encoders (**CBraMod** default publish path; **REVE-base** optional/experimental). Train and ship **head-only** weights — not full encoders.

| Head | Role | Public ship status |
|------|------|--------------------|
| **A-vig** | `drowsy` / `hypnagogic` (± optional `awake`) | **Ship path** — frozen CBraMod on Sleep-EDF+HMC muse4 proxy |
| **A-med** | `rest` / `meditation` (or depth low/high research) | **Not public ship** — personal Muse cal |
| **A-eng** | `low_engagement` / `high_engagement` | **Misfit** — do not ship |
| **MW / concentration** | research probes | Deferred; personal cal only |
| **Head C** | sleep stage probe (`wake`/`light` on N1-slice) | Probe only — not 4-way product staging |
| **Head B** | blink / jaw / clean artifacts | **No Head B yet** — plan only |

## Read order (newcomers / tutors)

1. **This page** (`docs/INDEX.md`) — map + hard rules  
2. [`dataset_confidence_table.md`](dataset_confidence_table.md) — corpora confidence at a glance  
3. [`RD_JOURNEY.md`](RD_JOURNEY.md) — chronological what we did and why  
4. [`head_a_multihead.md`](head_a_multihead.md) — locked Head A architecture  
5. [`LICENSE_NOTES.md`](LICENSE_NOTES.md) — ALLOW/DENY + publish posture  
6. [`montages_muse_crown.md`](montages_muse_crown.md) — muse4 vs crown8  
7. Root [`README.md`](../README.md) + [`datasets/CATALOG.md`](../datasets/CATALOG.md) — layout and on-disk corpora  

Then dive by topic: A-vig / Head C full-corpus trains, A-eng dual-encoder misfit, label smokes (med/MW/depth/L-FAME/stress), REVE HMC paper compare.

## Key docs (pointers)

| Doc | Why |
|-----|-----|
| [`dataset_confidence_table.md`](dataset_confidence_table.md) | Summary table: license, role, ship?, confidence 0–5, evidence |
| [`RD_JOURNEY.md`](RD_JOURNEY.md) | Narrative for tutors |
| [`head_a_multihead.md`](head_a_multihead.md) | Multi-head lock (A-vig / A-med / A-eng) |
| [`head_a_vig_full_corpus_train.md`](head_a_vig_full_corpus_train.md) | A-vig ship metrics (124-subj, test macro-F1 ≈ 0.75) |
| [`head_c_full_corpus_train.md`](head_c_full_corpus_train.md) | Head C 2-way probe |
| [`head_a_eng_dual_encoder.md`](head_a_eng_dual_encoder.md) | A-eng CBraMod+REVE misfit |
| [`reve_hmc_paper_compare_gap.md`](reve_hmc_paper_compare_gap.md) | REVE HMC bal_acc ≈ paper LP |
| [`LICENSE_NOTES.md`](LICENSE_NOTES.md) | Open-license ship rules |
| [`montages_muse_crown.md`](montages_muse_crown.md) | Never mix muse4 / crown8 |
| [`labeling.md`](labeling.md) | Label definitions |
| [`datasets.md`](datasets.md) | Full ALLOW/DENY catalog |
| [`head_b_plan.md`](head_b_plan.md) | Head B deferred plan |
| [`grok_thread_catalog.md`](grok_thread_catalog.md) | Thread decision catalog |
| [`reve_model_cache.md`](reve_model_cache.md) / [`cbramod_notes.md`](cbramod_notes.md) | Encoder caches (not datasets) |

## Hard rules (do not violate)

1. **Open licenses only for ship** — prefer CC0 / CC BY / ODC-By / MIT / BSD. Publish **head-only** weights with attribution. See [`LICENSE_NOTES.md`](LICENSE_NOTES.md).
2. **No Head B yet** — labels locked; training deferred until personal Muse cal or an explicit weak-label pilot queue.
3. **No L-FAME / BY-NC in ship mix** — also no LUNA, no SEED-VIG (academic-only). Research smokes on DENY sets stay out of published heads.
4. **Montages `muse4` vs `crown8` never mixed** in one example or one head pack.
5. **Report datasets as** `Name (N-ch, professional|hobbyist)` — e.g. `Sleep-EDF Expanded (2→4 muse4 proxy, professional)`, `STEW (14-ch Emotiv, hobbyist)`.

## Current ship set (one-liner)

**A-vig frozen CBraMod** (muse4 proxy, Sleep-EDF+HMC); **Head C** = probe only; **A-eng** = misfit; **med / MW / depth / stress** → personal cal; **skip Head B** for now.

## Band-math track (peer)

See [`PEER_BAND_MATH_HANDOFF.md`](PEER_BAND_MATH_HANDOFF.md). Peer owns `band_math/`.

## Crown HMC vigilance

- [`crown_hmc_vig_compare.md`](crown_hmc_vig_compare.md) — CBraMod Crown2/4 HMC vig (ship proxy)
- [`crown_hmc_vig_reve.md`](crown_hmc_vig_reve.md) — REVE Crown2/4 HMC vig (experimental heads-only)

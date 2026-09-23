# License notes — shipping & publish path

## Project code license

Original project code in this repository (scripts, docs, recipes, subject-split JSON authored here) is licensed under **Apache-2.0** — see [`LICENSE`](../LICENSE) at the repo root.

This grant covers **windwerfer-authored code only**. It does **not** re-license:

- third-party snippets under `vendor/` (see `vendor/cbramod/NOTICE.md` and upstream licenses);
- training corpora or derived window binaries on Hugging Face (source licenses; table below).

This project trains and **publishes head-only weights** on top of frozen encoders.  
Stay within **open licenses** for anything that touches the shipping training mix or redistributed artifacts.

## Hard rules

| Rule | Detail |
|------|--------|
| **NO LUNA** | Do not use LUNA (or LUNA-derived) data/weights in shipping training or published heads. |
| **NO BY-NC** | No Creative Commons NonCommercial (BY-NC) corpora in the shipping mix. |
| **NO academic-only** | No datasets that restrict use to academic research only (e.g. SEED-VIG) in shipping. |
| **Prefer open** | Prefer **CC0**, **CC BY**, **ODC-By**, **MIT**, **BSD** (and clear OpenNeuro/PhysioNet open terms). |
| **Heads only** | Publish **head-only** checkpoints + label maps + attribution; do not re-publish gated full foundation weights. |
| **Default publish path** | **CBraMod** (± **LaBraM** if license-compatible). **REVE** heads are **optional / experimental** — users must fetch any gated REVE-base themselves; do not ship REVE base weights. |

## ALLOW / DENY datasets (shipping training mix)

| Dataset | Decision | License posture (verify on source) | Notes |
|---------|----------|--------------------------------------|-------|
| OpenNeuro **ds003969** | **ALLOW** | **CC0** (verified 2026-09-06 via `dataset_description.json` / GraphQL / NEMAR) | Head A attention/cognitive |
| OpenNeuro **ds001787** | **ALLOW** | **CC0** (verified 2026-09-06 via `dataset_description.json` / GraphQL / NEMAR) | Head A attention probes |
| HF **alexeykashevnik/EEGMeditation** | **ALLOW** | **CC BY 4.0** (verified 2026-09-06 HF card; gated access) | Head A concentration / mind_wandering; attribute |
| **Sleep-EDF Expanded** | **ALLOW** for research→publish if PhysioNet terms permit redistribution of **derived heads** | PhysioNet open research — attribute; no raw redistrib beyond terms | Vigilance transfer (drowsy / hypnagogic) |
| **HMC** sleep staging v1.1 | **ALLOW** | **CC-BY-4.0** (verified 2026-09-08 PhysioNet) | Head C / vigilance top-up; attribute |
| OpenNeuro **ds007169** / **ds007262** | **ALLOW** | **CC0** | A-eng Barras workload |
| PhysioNet **EEGMAT** | **ALLOW** | **ODC-By-1.0** | A-eng rest vs arithmetic |
| **STEW** (HF processed MONSTER) | **ALLOW** w/ attribution | **CC-BY-4.0** (raw IEEE DataPort login) | A-eng hobbyist Emotiv; prefer HF mirror for ship path |
| OpenNeuro **ds007554** | **ALLOW** | **CC0** | A-eng hierarchical cognitive-motor |
| Schreer **Muse-S** (doi:10.7910/DVN/V2CWJW) | **ALLOW** | **CC0-1.0** (verified 2026-09-07 Dataverse API) | Native Muse-S Mind Monitor CSV; Head B channel map / weak labels; no cued blink/jaw |
| **L-FAME** | **DENY** | BY-NC | Not in shipping mix |
| **SEED-VIG** | **DENY** | Academic-only / restricted | Not in shipping mix |
| **LUNA** | **DENY** | — | Explicitly excluded; no LUNA in mix or published artifacts |

When in doubt: **DENY** until a clear open SPDX / deed is recorded in the experiment config.

## Publishing trained heads

1. Train with frozen **CBraMod** (default) and/or **LaBraM** under their upstream licenses; ship **heads only**.
2. Include `ATTRIBUTION.md` (or model card) listing: datasets used, licenses, encoder name/version, Muse channel order, label lists.
3. Do **not** bundle L-FAME, SEED-VIG, LUNA, or BY-NC-derived samples in the training mix that produced published heads.
4. **REVE-base**: optional experimental path only. Document “user brings gated REVE weights”; published REVE-*head* artifacts must not include the gated base.
5. Prefer releasing heads under **MIT** or **BSD-2/3** + dataset attribution, or **CC BY** if that better matches upstream requirements.

## Encoder publish posture

| Encoder | Shipping default? | Notes |
|---------|-------------------|-------|
| **CBraMod** | **Yes (default Spur A)** | Freeze backbone; publish heads |
| **LaBraM** | **Yes (±)** | Same recipe if license-compatible |
| **REVE-base** | **No (optional/experimental)** | User fetches gated base; heads optional |

## Checklist before a public head release

- [ ] Training mix ⊆ ALLOW table; zero LUNA / L-FAME / SEED-VIG / BY-NC
- [ ] Dataset licenses recorded (SPDX or URL + deed date)
- [ ] Head-only checkpoint (no full gated encoder weights)
- [ ] Attribution for datasets + encoder
- [ ] Label maps + Muse channel order in card
- [ ] If REVE: marked experimental; base not redistributed

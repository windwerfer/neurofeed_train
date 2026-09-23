# Peer handoff — band math vs AI heads

**Owner bot:** `peer` (band-math eval).  
**R&D bot:** diggus maximus (AI heads).  
**User:** windwerfer — will review results with peer.

## Goal

On the **already gathered** muse-eeg-heads corpora, evaluate **classical band math** (PSD / relative band powers / ratios) and compare to **frozen AI head** scores. Answer: does band math outperform the AI models on the same labels/splits?

Deliverable: **`band_math/COMPARE_AI_VS_BAND.md`** (and supporting tables) with one row per label task.

## Scope (user locked)

Score **all** of these (not only ship labels):

| Task | Labels | Primary corpora | AI baseline (reference) |
|------|--------|-----------------|-------------------------|
| A-vig | drowsy / hypnagogic | `datasets/vigilance_sleep_edf/` (~124 subj) | CBraMod test macro-F1 ≈ **0.747** (`exports/head_a_vig_full_corpus/`) |
| Head C | wake / light (N1-slice 2-way) and/or 5-stage if you rebuild windows | Sleep-EDF+HMC | CBraMod 2-way ≈ **0.760**; REVE HMC 5-stage bal_acc ≈ **0.649** |
| A-eng | low / high engagement | `datasets/engagement_a_eng/` (~133 persons) | CBraMod ≈ **0.548**, REVE ≈ **0.590** (misfit) |
| MW / concentration | concentration / mind_wandering | `attention_ds001787`, `attention_ds003969` | muse4 LOSO ≈ **0.36** |
| A-med rest↔med | rest / meditation | ds003816 windows if present / re-derive | holdout ≈ **0.33** |
| Med depth | depth low / high | ds001787 Q1 | holdout ≈ **0.33** |

Use **same subject splits** as AI evals when files exist under `datasets/*/splits/`. Report `Name (N-ch, professional|hobbyist)`.

## Must evaluate: Muse app band features

App repo (fresh clone): `/workspace/flutter_muse-rs_ml`  
Canonical catalog: `assets/features.json` + `rust/src/api/features.rs`

| Feature id | Meaning | Default electrodes (Muse) |
|------------|---------|---------------------------|
| `band.atr` | alpha ÷ theta | AF7, AF8 |
| `band.tar` | theta ÷ alpha | AF7, AF8 |
| `band.btr` | beta ÷ theta | AF7, AF8 |
| `band.alpha` | relative alpha | AF7, AF8 |
| `band.delta` | absolute/relative frontal delta (guard) | AF7, AF8 |

Also try other classical band math freely (θ/β, α/(α+β), frontal–posterior asymmetry, peak alpha Hz, etc.) if it can beat AI.

`device.focus` / `device.calm` are Crown-native — optional, not required for muse4 corpora.

## Workspace rules for peer

1. Create your own tree: **`/workspace/muse-eeg-heads/band_math/`** (scripts, features, results, summaries). Do not overwrite AI `exports/head_*` blindly; you may read them.
2. Prefer **uv** for Python; local CPU OK; Kaggle optional.
3. **muse4 only** for fair compare unless you explicitly run a crown8 side study.
4. Open licenses only for ship claims; L-FAME remains research-only if used.
5. Metrics: macro-F1 (balanced when possible), accuracy; for HMC 5-stage also balanced accuracy to match REVE paper.
6. Final artifact: `band_math/COMPARE_AI_VS_BAND.md` + machine-readable `band_math/compare_summary.json`.

## Read first

- `docs/INDEX.md`, `docs/dataset_confidence_table.md`, `docs/RD_JOURNEY.md`
- `datasets/CATALOG.md`
- App: `assets/features.json`, `rust/src/api/features.rs`

## Done when

User can open a clear comparison table with peer and see, per label, AI score vs best band-math method, with win/lose/tie and method name.

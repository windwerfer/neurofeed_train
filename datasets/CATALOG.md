# Dataset catalog

| Corpus | Role | Source | Windows on disk | Subjects (stable) | Notes |
|--------|------|--------|-----------------|-------------------|-------|
| `vigilance_sleep_edf` | Head A vigilance + Head C stage labels | Sleep-EDF Expanded cassette+telemetry (ODC-By) + HMC (CC-BY-4.0) | n1-slice npz per recording | **124 subjects** / 125 nights (Sleep-EDF 100+HMC 24) | muse4 proxy `(ch_count=4, professional)`; **stage_raw + stage_coarse required**; expanded 2026-09-08 toward ~120 — ready to train heads |
| `attention_ds001787` | Head A attention | OpenNeuro ds001787 | ses-01 attention npz | 12 (`sub-001`…`020`) | BioSemi→AF7/AF8/TP9/TP10; Q1/Q2 probe labels; 3774 windows |
| `attention_ds003969` | Head A attention | OpenNeuro ds003969 | med1breath/think1 blocks | 11 (`sub-001`…`029`) | Muse-proximal; med→concentration, think→mind_wandering; 8800 windows |
| `engagement_a_eng` | Head A-eng load/engagement | ds007169+ds007262+eegmat+STEW(HF)+ds007554 | muse4 aeng npz | **133 unique persons** / 150 packs / 19706 windows | muse4 only; tags (ch, professional\|hobbyist); see `docs/head_a_eng_corpus_expansion.md` |

Status writeup: `docs/attention_corpora_status.md` (`ship_candidate: false`).

**Bot docs:** [`docs/INDEX.md`](../docs/INDEX.md) · [`docs/dataset_confidence_table.md`](../docs/dataset_confidence_table.md) · [`docs/RD_JOURNEY.md`](../docs/RD_JOURNEY.md).

## Label heads

- **Head A (multi-head, locked 2026-09-07):** see `docs/head_a_multihead.md`
  - **A-vig:** drowsy / hypnagogic (+ optional awake) — **ship path** frozen CBraMod on `vigilance_sleep_edf` (test macro-F1 ≈ 0.75; see `docs/head_a_vig_full_corpus_train.md`)
  - **A-med:** rest / meditation — **not public ship** (open-set smokes near chance; personal Muse cal)
  - **A-eng:** low/high engagement — **misfit** (CBraMod ≈ 0.548 / REVE ≈ 0.590 macro-F1; do not ship; see `docs/head_a_eng_dual_encoder.md`)
  - Public MW / concentration frozen decoder deferred (personal cal / research only)
- **Legacy windows:** may still store concentration / mind_wandering / drowsy / hypnagogic in `y_head_a`.
- **Head C (Sleep-EDF + HMC):** `stage_raw`, `stage_coarse` — ~124 subjects; **2-way wake/light probe** trained (ship_candidate False for 4-way; see `docs/head_c_full_corpus_train.md`).
- **Head B:** artifact classes deferred — **no Head B yet** (see `docs/head_b_plan.md`).

## Provenance

**Public windows** live on Hugging Face: [`windwerfer/neurofeed-eeg-windows`](https://huggingface.co/datasets/windwerfer/neurofeed-eeg-windows). This git tree ships **splits JSON + cards only** — no `windows/`, `raw/`, or `annotations/` blobs.

Historical maintainer layout used local `exports/windows_*` and private Kaggle caches; that is optional scratch only. Attention annotations carry `provenance` (DOI) + `license_spdx` (`CC0-1.0`). See `docs/head_c_corpus_expansion.md`.

A-eng: Barras ds007169+ds007262 share `unique_person_id`. STEW uses HF processed CC-BY-4.0 (raw IEEE still gated).

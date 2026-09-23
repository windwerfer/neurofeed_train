# Head A → ~40 subjects + LOSO (shippable data journey)

**Updated (UTC):** 2026-09-07  
**Goal:** Grow open-license Head A labels (esp. `concentration` / `mind_wandering`; vigilance may grow) toward **~80 subjects** (raised from 40 for reliable LOSO), then run **LOSO** alongside fixed splits. Montage: **muse4** only for this pack.  
**ship_candidate:** still **false** until honest subject holdout / LOSO beats chance.

## Current inventory (as of this plan)

| Corpus | License | Labels | Subjects now | Windows (approx) | Muse-proximal |
|--------|---------|--------|-------------:|-----------------:|---------------|
| `attention_ds001787` | CC0-1.0 | concentration / mind_wandering (probe Q1/Q2) | **12** / 24 | ~3774 | BioSemi64 → AF7/AF8 + P9/P10→TP9/TP10 |
| `attention_ds003969` | CC0-1.0 | concentration / mind_wandering (block med*/think*) | **11** / 98 | ~8800 | **AF7/AF8 native**; TP7/TP8→TP9/TP10 |
| `vigilance_sleep_edf` | ODC-By | drowsy / hypnagogic (W/N1 N1-slice) | **5** | ~24618 | Fpz-Cz/Pz-Oz duplicate proxy |

**Attention unique subjects:** 12 + 11 = **23**  
**Attention unique (corpus-qualified):** **80** (target ~80 met; open ceiling ~114 without HF)  
**All Head A subjects (attn+vig, disjoint people):** 23 + 5 = **28** (gap to 40 = 12 if counting vigilance too)

Frozen test holdouts (do not move):
- ds001787: sub-019, sub-013
- ds003969: sub-025, sub-027
- Sleep-EDF: SC400 (both nights)

## Gap fill strategy (ALLOW only)

**Hard rejects:** LUNA, L-FAME, SEED-VIG, BY-NC / academic-only.

| Priority | Source | Why | Next subjects (this journey) |
|---------:|--------|-----|------------------------------|
| 1 | **ds003969** more sessions | Muse-proximal; CC0; 98 pool | Batch A: 007–010 (htr) + 030–033 (ctr) → +8 |
| 2 | **ds001787** remaining | Best probe labels; CC0; only 12 left in corpus | Batch B: 007,008,016,021,022,023 (prefer both-class logs) → +6 |
| 3 | More Sleep-EDF cassette nights | Vigilance volume; ODC-By | Optional; not blocking attention 40 |
| 4 | HF EEGMeditation (CC BY) | Strong MW/concentration blocks | **Gated** — needs user HF accept; metadata only on disk now |
| 5 | Other OpenNeuro Muse-proximal | Only if CC0/CC-BY/ODC-By + AF7/AF8 or clear 10–10 map | Probe before ingest |

After Batch A+B: attention subjects ≈ 23+8+6 = **37** (near 40). One more small ds003969 batch (+3–4) closes 40.

## Fixed splits vs LOSO

- **Keep** `datasets/*/splits/{train,val,test}_subjects.json` (policy A) for regression / packaging.
- **Add** LOSO (leave-one-subject-out) once N ≳ 30 attention subjects:
  - Per fold: train on all-but-one subject (optionally exclude tiny single-class subjects), evaluate held-out subject macro-F1 / accuracy.
  - Report mean±std across folds; never claim ship on fixed-split alone.
  - Skeleton: `scripts/loso_eval_head_a.py` (+ this section).

## Disk / ops constraints

- ~100G free on box; ds003969 ≈ **~285 MB raw/subject** (2 BDFs); ds001787 ≈ **~80–150 MB/subject**.
- Local-first; gitignore binaries (`*.bdf`, `*.npz`, `*.pt`); no Kaggle upload unless needed.
- Pace OpenNeuro downloads (existing pause helpers); prefer `uv` / `.venv` scripts over notebooks.
- After ingest: symlink/migrate into `datasets/attention_*`, refresh annotations + splits, run light QC.

## First expansion batch (this run)

1. Implement `scripts/expand_attention_batch40_ds003969.py` (+ optional ds001787 companion).
2. Download/process **8× ds003969** (007–010, 030–033).
3. Migrate windows → `datasets/attention_ds003969/`, update splits (new → train; frozen test untouched), annotations index, QC.
4. If disk/time remain: start **4× ds001787** (007,008,016,021).
5. Ship LOSO skeleton; full LOSO after N larger.

## Success criteria (later)

- ≥40 subjects with Head A attention labels under ALLOW licenses.
- Fixed-split + LOSO metrics documented; `ship_candidate` only if holdout/LOSO clearly > chance with per-class support.


## Progress log

| UTC date | Change | Attention N | Gap to 40 |
|----------|--------|------------:|----------:|
| 2026-09-07 (plan) | baseline | 23 | 17 |
| 2026-09-07 | Batch A ds003969 +8; Batch B ds001787 +4 | **35** | **5** |
| 2026-09-07 | Close +5 ds003969 (011–013,034–035) | **40** | 0 (40 target) |
| 2026-09-07 | Toward80 +40 ds003969 (014–024,036–064); montage_id=muse4 | **80** | 0 (80 target) |

See `docs/head_a_batch40_status.md`.

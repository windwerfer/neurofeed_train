# Local-first Muse EEG datasets

Curated corpora for experienced EEG trainers (CBraMod + Heads A/B/C). Raw EDFs live on disk (symlinked); git tracks code, docs, annotations, splits, and manifests — not huge binaries (no git-lfs installed → `*.npz` / `*.edf` / `*.bdf` gitignored).

## Split policy A — fixed subject JSON

Each corpus has:

- `splits/subjects.json` — catalog of subjects + recordings
- `splits/train_subjects.json` / `val_subjects.json` / `test_subjects.json`
- `splits/split_policy.json` — full policy + prior holdout notes

**No subject appears in more than one split.** Validate:

```bash
python scripts/dataset/validate_splits.py
```

### Sleep-EDF (`vigilance_sleep_edf`) — v0 FROZEN

Stable subject key = `SC4sss` from cassette id `SC4sssn` (sss=subject, n=night).

| Split | Subjects | Recordings |
|-------|----------|------------|
| test | `SC400` | `SC4001`, `SC4002` (prior primary holdout night was SC4002; both nights held out — no subject leakage) |
| val | `SC403` | `SC4031` (prior secondary holdout) |
| train | `SC401`, `SC402`, `SC404` | `SC4011`, `SC4021`, `SC4041` |

**Head A (unchanged):** W→`drowsy`, N1→`hypnagogic` only.  
**Head C (labels required):** every window keeps `stage_raw` (hypnogram text) and `stage_coarse` (`wake`/`light`/`deep`/`rem`/`unknown`). Do not drop these in migrate/schema/docs.

### Attention ds001787 (12 subjects / 3774 windows)

| Split | Subjects |
|-------|----------|
| train | `sub-001`…`005`, `sub-014`, `sub-017`, `sub-020` |
| val | `sub-006`, `sub-015` (both classes; prior val was concentration-only) |
| test | `sub-019`, `sub-013` (prior failed holdouts — honest, frozen) |

### Attention ds003969 (11 subjects / 8800 windows)

| Split | Subjects |
|-------|----------|
| train | `sub-001`…`006`, `sub-029` |
| val | `sub-026`, `sub-028` (balanced) |
| test | `sub-025`, `sub-027` (incl. prior failed holdout sub025; frozen) |

See `docs/attention_corpora_status.md` — **ship_candidate: false** until subject holdout beats chance.

## Layout

See [CATALOG.md](CATALOG.md), [common/schema_windows.md](common/schema_windows.md), and `docs/dataset_layout.md`.

## Scripts

```bash
python scripts/dataset/init_layout.py
python scripts/dataset/write_fixed_splits.py
python scripts/dataset/migrate_existing_exports.py   # symlink windows + raw
python scripts/dataset/make_annotations_index.py     # CSV + recording_manifest (incl. provenance/license)
python scripts/dataset/validate_splits.py
python scripts/dataset/checksum_corpus.py --verify-manifest
python scripts/dataset/summarize_corpus.py
# expand local attention subjects (no Kaggle):
python scripts/expand_attention_ds001787.py
python scripts/expand_attention_ds003969.py
```

- `engagement_a_eng/` — Head A-eng low/high engagement (~133 unique persons). See `docs/head_a_eng_corpus_expansion.md`.

## Public distribution

Window binaries are published on Hugging Face [`windwerfer/neurofeed-eeg-windows`](https://huggingface.co/datasets/windwerfer/neurofeed-eeg-windows). This folder keeps schemas, READMEs/ATTRIBUTION, and `splits/*.json` only.

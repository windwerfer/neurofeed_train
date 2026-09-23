# Dataset layout (local-first)

Short guide for the curated tree under `datasets/`. Continuum paused; AppImage tests skipped.

## Tree

```
datasets/
  README.md                 # usage + split policy A + Head A/B/C notes
  CATALOG.md
  common/
    label_maps.json         # Head A, Head C coarse, stage_raw
    schema_windows.md       # required npz keys incl. stage_raw/stage_coarse
  vigilance_sleep_edf/
    README.md, ATTRIBUTION.md
    raw/                    # PSG+hypnogram symlinks
    windows/                # *.npz (+ manifests) symlinks
    annotations/            # windows_index.csv, recording_manifest.json, checksums.json
    splits/                 # subjects + train/val/test JSON
  attention_ds001787/       # same pattern (no raw EDFs required locally)
  attention_ds003969/
```

## Head C on Sleep-EDF

Every Sleep-EDF window must carry:

- `stage_raw` — hypnogram text  
- `stage_coarse` — wake/light/deep/rem/unknown  

Head A mapping stays W→drowsy, N1→hypnagogic only. Migration and `make_annotations_index.py` preserve these fields from existing npz exports.

## Git policy

- Track: scripts, docs, annotations CSV/JSON, splits, manifests, checksums.
- Ignore: `__pycache__`, `.venv`, `*.edf`, `*.bdf`, `*.npz` (git-lfs not installed).
- Binaries live on disk; SHA256 in `annotations/checksums.json` / recording manifests.

## Commands

```bash
python scripts/dataset/init_layout.py
python scripts/dataset/write_fixed_splits.py
python scripts/dataset/migrate_existing_exports.py
python scripts/dataset/make_annotations_index.py
python scripts/dataset/validate_splits.py
python scripts/dataset/checksum_corpus.py --verify-manifest
python scripts/dataset/summarize_corpus.py
```

# REVE model cache (private)

Offline snapshots of gated `brain-bzh/reve-base` and `brain-bzh/reve-positions` live in the **private** Kaggle dataset:

**https://www.kaggle.com/datasets/windwerfer/muse-eeg-heads-cache**

Paths inside the dataset:

- `models/reve-base/` — full HF snapshot (`config.json`, `modeling_reve.py`, `model.safetensors`, …)
- `models/reve-positions/` — position bank snapshot
- `models/CBraMod/` — already present (~19 MiB Apache-2.0)
- `models/MANIFEST.md` — SHA256 + **private-only / no-redistribute** policy

## Load order (kernels + `FrozenREVEEncoder`)

1. **Local cache** under `/kaggle/input/muse-eeg-heads-cache/models/reve-*` (or glob `**/models/reve-*`)
2. **HF hub fallback** with `HF_TOKEN` / Kaggle UserSecrets (only if cache missing)

`from_pretrained(local_path, local_files_only=True)` when local. Do not flip the cache dataset public; version-update only.

## Local box mirror

`/workspace/muse-eeg-heads/kaggle_datasets/muse-eeg-heads-cache/`

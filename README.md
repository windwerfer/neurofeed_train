# neurofeed_train

**Train / eval lab** for Muse4 + Crown8 EEG heads used by [neurofeed](https://github.com/windwerfer/neurofeed).

This repo is **code, docs, recipes, and subject-split JSON** — not the app, not shipped runtime packs, and not window binaries.

| Repo / artifact | Role |
|-----------------|------|
| **This repo** (`neurofeed_train`) | Training scripts, notebooks, band-math baselines, split recipes |
| [`neurofeed`](https://github.com/windwerfer/neurofeed) | App (+ `feedback_gym` path) |
| [`neurofeed_heads`](https://github.com/windwerfer/neurofeed_heads) | Frozen head packs for runtime |
| [`neurofeed_eeg_datasets`](https://github.com/windwerfer/neurofeed_eeg_datasets) | Dataset cards / layout docs |
| [HF `neurofeed-eeg-windows`](https://huggingface.co/datasets/windwerfer/neurofeed-eeg-windows) | **Window binaries** (`.npz` / manifests) |

## Honest status

- **Vigilance (Muse4 / A-vig)** — primary ship path on top of frozen CBraMod (Sleep-EDF + HMC muse4 proxy).
- **Attention (OpenNeuro ds001787 / ds003969)** — research; Crown8 attention is **not shippable** as a public frozen pack.
- **Engagement (A-eng)** — research / misfit; do not treat as a ship head.
- **A-med / mind-wandering** — personal calibration / research; not a public ship decoder.
- **L-FAME, LUNA, SEED-VIG** — **DENY** in the shipping mix (see [`docs/LICENSE_NOTES.md`](docs/LICENSE_NOTES.md)).

## Get windows (required for training)

Window blobs are **not** in this git tree. Mirror the Hugging Face dataset layout locally, then point scripts at that root.

```bash
# Example: full snapshot under ./data/hf_windows
pip install -U huggingface_hub
huggingface-cli download windwerfer/neurofeed-eeg-windows \
  --repo-type dataset \
  --local-dir ./data/hf_windows

# Or selective files
python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
snapshot_download(
    "windwerfer/neurofeed-eeg-windows",
    repo_type="dataset",
    local_dir="./data/hf_windows",
)
PY
```

Expected layout idea (match HF / `datasets/` corpus names):

```
data/hf_windows/
  vigilance_sleep_edf/windows/...
  attention_ds001787/windows/...
  ...
```

Point train scripts / env at that mirror (e.g. `DATASETS_ROOT` or CLI `--windows-root`). Split JSON under `datasets/*/splits/` in this repo is the subject split source of truth.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# After HF windows are local:
python scripts/dataset/validate_splits.py   # if applicable
# then run a train/smoke script under scripts/ (see docs/INDEX.md)
```

Doc entry points:

- [`docs/INDEX.md`](docs/INDEX.md) — bot / tutor index  
- [`docs/LICENSE_NOTES.md`](docs/LICENSE_NOTES.md) — **ALLOW / DENY** and publish posture  
- [`docs/dataset_confidence_table.md`](docs/dataset_confidence_table.md)  
- [`datasets/CATALOG.md`](datasets/CATALOG.md)

## Layout

```
neurofeed_train/
├── src/                 # encoders wrappers, heads, windowing, metrics
├── scripts/             # train / ingest / eval / corpus tools
├── band_math/           # classical band baselines (no large feature caches)
├── notebooks/           # exploratory notebooks (de-Kaggle notes OK)
├── docs/                # experiment notes + LICENSE_NOTES
├── datasets/
│   ├── CATALOG.md
│   ├── common/          # schemas, montages, label maps
│   └── <corpus>/
│       ├── README.md / ATTRIBUTION.md
│       └── splits/*.json   # subject splits only — no windows/raw
├── kaggle_kernel*/      # optional GPU scratch kernels (maintainers)
├── vendor/cbramod/      # small CBraMod source snippets
└── requirements.txt
```

## Kaggle (optional, maintainers only)

Private Kaggle datasets / kernels were used as **GPU scratch** during development. They are **not required** for public use. Prefer the HF windows dataset above. Kernel folders here are thin notebook wrappers; rewrite local dataset mounts to a HF mirror if you reuse them.

## What is intentionally excluded

- `kaggle_datasets/` caches and window packs  
- `exports/` large windows / embeddings / `.pt` caches  
- `datasets/*/windows|raw|annotations` blobs  
- `*.npz`, `*.edf`, `*.bdf`, encoder/emb `.pt` / `.pth`  
- Secrets (`.env`, tokens, `kaggle.json`)  
- L-FAME / LUNA / SEED-VIG training data  

## License posture

- **Original code in this repo** (scripts, docs, recipes, subject-split JSON authored here): **[Apache-2.0](LICENSE)** — see [`LICENSE`](LICENSE).
- **Third-party snippets under `vendor/`**: upstream licenses apply (see [`vendor/cbramod/NOTICE.md`](vendor/cbramod/NOTICE.md)). This Apache grant does not re-license those materials.
- **Training corpora / Hugging Face window binaries**: remain under their **source licenses**; see [`docs/LICENSE_NOTES.md`](docs/LICENSE_NOTES.md). This Apache grant does **not** re-license those materials.

Prefer open corpora (CC0 / CC BY / ODC-By / MIT / BSD). Publish **head-only** weights with dataset + encoder attribution. Do not redistribute gated foundation bases (e.g. REVE-base).

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

- **Vigilance (Muse4 / A-vig)** — primary Muse ship path on top of frozen CBraMod (Sleep-EDF + HMC muse4 proxy).
- **Vigilance (Crown HMC proxy)** — shippable Crown vig path: HF `crown2_vigilance_hmc` / `crown4_vigilance_hmc`; CBraMod test macro-F1 **0.670 / 0.680** (see `docs/crown_hmc_vig_compare.md`). Honest proxy F6≈F4, PO4≈O2. Do not mix with muse4 vig.
- **Attention (OpenNeuro ds001787 / ds003969)** — research; Crown8 attention is **not shippable** as a public frozen pack.
- **Engagement (A-eng)** — research / misfit; do not treat as a ship head.
- **A-med / mind-wandering** — personal calibration / research; not a public ship decoder.
- **L-FAME, LUNA, SEED-VIG** — **DENY** in the shipping mix (see [`docs/LICENSE_NOTES.md`](docs/LICENSE_NOTES.md)).

## Get windows (required for training)

Window blobs are **not** in this git tree. Mirror the Hugging Face dataset layout locally, then point scripts at that root.

```bash
# Example: full snapshot under ./data/hf_windows (about 1.56M windows; select configs to save space)
uvx --from huggingface_hub huggingface-cli download windwerfer/neurofeed-eeg-windows \
  --repo-type dataset \
  --local-dir ./data/hf_windows

# Or selective configs
uv run --with huggingface_hub python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "windwerfer/neurofeed-eeg-windows",
    repo_type="dataset",
    local_dir="./data/hf_windows",
    allow_patterns=["muse4_attention_ds001787/*", "muse4_attention_ds003969/*"],
)
PY
```

Expected layout idea (match HF / `datasets/` corpus names):

```
data/hf_windows/
  muse4_vigilance_sleep_edf/windows/...
  crown2_vigilance_hmc/windows/...
  crown4_vigilance_hmc/windows/...
  ...
```

Point scripts at that mirror with `HF_WINDOWS_ROOT=./data/hf_windows`, or pass `--hf` to scripts that support it (they download the needed config; see [`src/public_io.py`](src/public_io.py)). CBraMod weights: `CBRAMOD_WEIGHTS=<path>`, else the public `weighting666/CBraMod` download. Split JSON under `datasets/*/splits/` in this repo is the subject split source of truth.

## Quick start

```bash
uv venv && uv pip install -r requirements.txt

uv run python scripts/dataset/validate_splits.py   # if applicable
uv run python scripts/loso_eval_head_a.py --hf --dry-run   # attention LOSO fold plan from HF windows
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
├── notebooks/           # public notebooks (02 Sleep-EDF windows, 06 REVE attention LOSO)
├── docs/                # experiment notes + LICENSE_NOTES
├── datasets/
│   ├── CATALOG.md
│   ├── common/          # schemas, montages, label maps
│   └── <corpus>/
│       ├── README.md / ATTRIBUTION.md
│       └── splits/*.json   # subject splits only — no windows/raw
├── kaggle_kernel_{02,06,09,10,11}*/  # optional GPU kernel wrappers
├── archive/             # maintainer-only notebooks/kernels (01, 03, 04, 05, 07, 08); not for public use
├── vendor/cbramod/      # small CBraMod source snippets
└── requirements.txt
```

## Kaggle (optional, maintainers only)

Private Kaggle datasets / kernels were used as **GPU scratch** during development. They are **not required** for public use. Prefer the HF windows dataset above. Kernel folders here are thin notebook wrappers. Notebook/kernel 06 reads the HF windows directly; older maintainer notebooks are in [`archive/`](archive/) and are not maintained for public use.

Crown HMC vigilance wrappers: [`kaggle_kernel_10_hmc_crown_vig/`](kaggle_kernel_10_hmc_crown_vig/) (CBraMod) and [`kaggle_kernel_11_hmc_crown_vig_reve/`](kaggle_kernel_11_hmc_crown_vig_reve/) (REVE, bring-your-own gated base). See `docs/crown_hmc_vig_compare.md` / `docs/crown_hmc_vig_reve.md`.

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

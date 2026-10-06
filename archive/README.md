# archive/ (maintainer-only)

These notebooks and Kaggle kernel wrappers are kept for the maintainer's history. They are **not maintained for public
use** and are **not linked** from the Hugging Face dataset cards.

They read private scratch inputs (private Kaggle datasets and local working folders) that are not published and will not
be. Expect hard-coded paths and missing inputs if you try to run them.

| Item | What it was |
|---|---|
| `notebooks/01_kaggle_outline.ipynb`, `kaggle_kernel/` | First CBraMod outline on Kaggle |
| `notebooks/03_cbramod_head_a_smoke.ipynb`, `kaggle_kernel_03_cbramod_head_a/` | Head A smoke test |
| `notebooks/04_night_holdout.ipynb`, `kaggle_kernel_04_night_holdout/` | Night-holdout experiment |
| `notebooks/05_hypnagogic_precision_tune.ipynb` | Hypnagogic precision tuning |
| `kaggle_kernel_07_reve_sleep_heads/` | REVE sleep heads (notebook 07) |
| `kaggle_kernel_08_reve_a_eng/` | REVE engagement (notebook 08) |

Public, maintained entry points: `notebooks/06_reve_attention_loso.ipynb`, `kaggle_kernel_10_hmc_crown_vig/`,
`kaggle_kernel_11_hmc_crown_vig_reve/`, and `scripts/` (for example `scripts/loso_eval_head_a.py --hf`). All of them read the
public Hugging Face dataset [`windwerfer/neurofeed-eeg-windows`](https://huggingface.co/datasets/windwerfer/neurofeed-eeg-windows).

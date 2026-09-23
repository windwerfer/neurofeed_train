# kaggle_kernel_10_hmc_crown_vig (optional GPU scratch)

Maintainer notebook wrapper for Crown HMC vigilance (frozen CBraMod Head A-vig).

**Public users:** do **not** depend on private Kaggle datasets. Download windows from
[Hugging Face `windwerfer/neurofeed-eeg-windows`](https://huggingface.co/datasets/windwerfer/neurofeed-eeg-windows)
configs `crown2_vigilance_hmc` / `crown4_vigilance_hmc` and point `HF_WINDOWS_ROOT` at a local mirror
(see repo root README). Subject splits live under `datasets/vigilance_hmc_crown2|4/splits/`.

Use repo-root `scripts/train_hmc_crown_vig_compare.py`, `src/`, and `requirements.txt`.
Live metrics: `docs/crown_hmc_vig_compare.md`.

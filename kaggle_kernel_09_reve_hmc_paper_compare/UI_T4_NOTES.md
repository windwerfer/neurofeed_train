# Run on Kaggle T4 (UI)

CLI often lands on **P100** which breaks current Kaggle PyTorch (needs sm_70+). Prefer:

1. Open https://www.kaggle.com/code/windwerfer/muse-eeg-heads-reve-hmc-paper-compare (after push)
2. Settings → Accelerator → **GPU T4 x2** (or single T4)
3. Ensure datasets attached: `muse-eeg-heads-cache` (REVE + HMC raw), `muse-eeg-heads-src`
4. Run All
5. Download `metrics_summary.json` / `compare_table.json` into `exports/reve_hmc_paper_compare/`

If src pack missing on Kaggle, re-version `windwerfer/muse-eeg-heads-src` from `kaggle_datasets/muse-eeg-heads-src`.

Local CPU path: `scripts/reve_hmc_paper_compare.py` (used for this compare when GPU unavailable).

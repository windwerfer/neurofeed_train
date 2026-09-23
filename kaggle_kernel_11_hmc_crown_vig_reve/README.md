# kaggle_kernel_11_hmc_crown_vig_reve (optional GPU scratch)

Maintainer notebook wrapper for Crown HMC vigilance (frozen REVE-base Head A-vig, experimental).

**Public users:** do **not** depend on private Kaggle datasets. Use the same HF window configs as kernel 10
plus `scripts/train_hmc_crown_vig_reve.py`.

**Bring your own gated REVE-base** (and positions bank). Do not redistribute foundation weights.
Set `REVE_BASE_DIR` / `REVE_POSITIONS_DIR` or place under `data/models/`. Shipped packs are heads-only
in `neurofeed_heads` (`reve-a-vig-crown2/4-hmc`).

Live metrics: `docs/crown_hmc_vig_reve.md`.

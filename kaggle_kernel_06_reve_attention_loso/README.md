# kaggle_kernel_06_reve_attention_loso (optional GPU wrapper)

Same notebook as [`notebooks/06_reve_attention_loso.ipynb`](../notebooks/06_reve_attention_loso.ipynb), packaged for
`kaggle kernels push`. It reads the public Hugging Face windows (`muse4_attention_ds001787`, `muse4_attention_ds003969`)
and needs no private Kaggle datasets.

- Internet on (to download the HF windows) and a GPU.
- REVE is **bring-your-own**: accept the gated `brain-bzh/reve-base` / `reve-positions` terms and set `HF_TOKEN`
  (Kaggle secret), or set `REVE_LOCAL_MODEL` / `REVE_LOCAL_POSITIONS` to your own snapshots.
- Set `NEUROFEED_TRAIN_ROOT` to a clone of this repo so the notebook can import `src/`.

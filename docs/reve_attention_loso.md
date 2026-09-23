# REVE attention LOSO (muse4, ~80 subjects)

**Goal:** Compare frozen **REVE-base** + tiny heads vs prior CBraMod muse4 LOSO (macro-F1 ≈ **0.361**).

**Kernel:** https://www.kaggle.com/code/windwerfer/muse-eeg-heads-reve-attention-loso (private, GPU on, internet on)  
**Notebook:** `notebooks/06_reve_attention_loso.ipynb` (synced to `kaggle_kernel_06_reve_attention_loso/`)  
**Helpers:** `src/reve_encoder.py`  
**Data:** `windwerfer/muse-eeg-heads-windows` (versioned with QC) + `windwerfer/muse-eeg-heads-src`  
**Exports:** `exports/reve_attention_loso/`


## Offline REVE cache

Prefer private dataset `windwerfer/muse-eeg-heads-cache` (`models/reve-base`, `models/reve-positions`). HF_TOKEN is fallback only. See [`reve_model_cache.md`](reve_model_cache.md).

## Design: embed-once → head-only LOSO

1. Load QC-pass muse4 attention windows (ds001787 + ds003969), corpus-qualified subject IDs.
2. Load gated `brain-bzh/reve-base` + `brain-bzh/reve-positions` once (`HF_TOKEN` Kaggle secret).
3. Embed all windows → **512-d** Z cache `/kaggle/working/reve_attention_loso/z_cache.pt`.
4. LOSO folds (~77 both-class / 80 available): train **HeadALinear** + **HeadAMLP** on Z only (5 epochs, balanced undersample).
5. Write `metrics_summary.json` comparable to CBraMod `exports/loso_head_a_attention/metrics_summary.json`.

## Window / montage policy

| Item | Choice |
|------|--------|
| Source windows | 2.0 s @ 256 Hz, `(N,4,512)`, **AF7,AF8,TP9,TP10** |
| REVE rate | resample → **200 Hz** → **T=400** |
| Patches | patch=200, overlap=20 → **2 patches** |
| Why not 4 s→800 / 5 s→1000? | Corpus is 2 s; inventing longer windows would pad silence. |
| Positions | `reve-positions` (fallback `common/montages.json`) |
| Norm | per-channel z-score + clip ±15σ |
| Pool | attention pooling → 512-d |

## HF_TOKEN loading (hardened)

Kernel prefers sources in order:

1. `kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")`
2. `os.environ`: `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` → `HUGGINGFACE_HUB_TOKEN`

No hardcoded tokens. On failure it soft-exits with `error=reve_load_failed`.

**Known CLI limitation:** `kaggle kernels push` starts a worker that **cannot** reach UserSecrets (`ConnectionError: Connection error trying to communicate with service.`). Env vars are also empty on CLI-triggered runs. There is **no** `kaggle kernels run` subcommand in CLI 2.2.4.

## P100 / torch compatibility patch (2026-09-07)

**Problem (UI GPU run):** Tesla P100 (sm_60) + Kaggle default `torch 2.10+cu128` (sm_70+ only) → `AcceleratorError: CUDA error: no kernel image is available for execution on the device` in `F.interpolate` / `reve_encoder.resample_bct`. HF_TOKEN *did* load from UserSecrets in that UI session.

**Fixes shipped (kernel v9 + muse-eeg-heads-src reversioned):**

1. **Notebook setup cell** (before `import torch`): probe `nvidia-smi --query-gpu=compute_cap`; if major **&lt; 7**, `pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118` from `https://download.pytorch.org/whl/cu118`. If major **≥ 7** (T4 etc.), keep default Kaggle torch. Log `torch` version + `cuda.get_device_capability` after import.
2. **`src/reve_encoder.resample_bct`**: always run `F.interpolate` on **CPU**, then move tensor back to the original device (belt-and-suspenders).

**Trigger note:** CLI push still cannot inject `HF_TOKEN`. User UI **Save Version → Run All** on **Tesla T4** (2026-09-07) succeeded with default torch 2.10+cu128 (sm_70+ skip-pin path).

## Latest Kaggle run (COMPLETE SUCCESS, UI T4, 2026-09-07 ~10:31–10:41 UTC)

Outputs: `exports/reve_attention_loso/kaggle_out_t4/` (+ top-level `metrics_summary.json`, `run.log`).

| Item | Value |
|------|------:|
| GPU | **Tesla T4** (capability 7.5; default torch 2.10.0+cu128, pin skipped) |
| HF_TOKEN | loaded from Kaggle UserSecrets |
| Subjects / QC windows | **80** / **55012** |
| Both-class folds | **77** (3 single-class skipped) |
| Best head | **mlp** |
| REVE macro-F1 (mlp) | **0.500 ± 0.144** |
| REVE macro-F1 (linear) | **0.480 ± 0.153** |
| REVE accuracy (mlp) | **0.536** |
| folds mlp F1>0.5 | **34 / 77** |
| CBraMod muse4 LOSO macro-F1 | **0.361** |
| **Δ vs CBraMod** | **+0.139** (mlp) |
| ship_candidate | **false** (needs mean>0.60 and mean−std>0.55) |
| Timing | embed **52.4 s**, LOSO heads **463 s**, total **~515 s** (~8.6 min) |

### Interpretation

Frozen REVE-base + tiny heads **beats** prior CBraMod muse4 LOSO by ~14 macro-F1 points on the same QC-pass attention windows / LOSO protocol, but remains near chance-ish for a hard subject-out binary split (std high; not shippable under current threshold).

### Prior failures (resolved)

- CLI push: no UserSecrets → gated `reve-base` 401.
- UI P100: torch 2.10+cu128 lacked sm_60 kernels → `AcceleratorError` in `F.interpolate` (fixed by torch pin for major&lt;7 + CPU interpolate).

## Sync / push

```bash
uv run python scripts/kaggle/sync_attention_windows_pack.py
sleep 3; kaggle datasets version -p kaggle_datasets/muse-eeg-heads-windows -m "attention QC for REVE LOSO" --dir-mode zip
sleep 3; kaggle datasets version -p kaggle_datasets/muse-eeg-heads-src -m "add reve_encoder.py" --dir-mode zip
sleep 3; kaggle kernels push -p kaggle_kernel_06_reve_attention_loso
# NOTE: push alone will soft-fail on HF_TOKEN; use UI Run All for secrets.
```

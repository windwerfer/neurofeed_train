# Attention-only Head A smoke (ds003969)

**Step:** `attention_only_head_smoke`  
**Goal:** binary concentration vs mind_wandering Head A on Muse-proximal OpenNeuro ds003969 with **subject holdout**, after joint 4-way collapsed on ds001787.

## Design

| Piece | Choice |
|-------|--------|
| Head | `HeadALinear` → 2 logits (`ATTENTION_LABELS`) |
| Encoder | Frozen CBraMod (CPU), mean pool |
| Labels | concentration=0, mind_wandering=1 (block-level med* / think*) |
| Train | sub001 + sub002 (htr); per-subject `undersample_balanced` then concat |
| Val | 15% window-level from train pool; **best-by-val** checkpoint restored |
| Holdout | sub025 (ctr) — full, balanced, stride×4 |
| Hyper | SEED=42, EPOCHS=8, BATCH=32, LR=1e-3 |

Windows: `exports/windows_ds003969/{tag}/ds003969_{tag}_attention_windows.npz` (N,4,512 @256 Hz; AF7/AF8 + TP7/TP8→TP9/TP10).

## Headline metrics (CPU)

| Split | ACC | macro-F1 |
|-------|-----|----------|
| Val (best epoch) | ~0.75 | **~0.74** |
| Holdout sub025 full | 0.50 | **~0.33** |
| Holdout sub025 balanced | 0.50 | **~0.33** |
| Holdout sub025 stride×4 | ~0.49 | **~0.33** |

Training oscillated after epoch 1 (val F1 drifted toward collapse by epoch 8); smoke keeps the best val checkpoint. Holdout ACC≈0.5 with macro-F1≈0.33 is consistent with **always predicting one class** on the held-out subject.

## Takeaway

Same-subject window val looks strong (~0.74 F1) but **subject transfer to sub025 is near chance**. Muse-proximal montage alone does not fix attention; need more subjects and/or stronger labels (probe ratings vs protocol med/think blocks) before shipping an attention head. Do not pack this binary yet.

## Artifacts

- `scripts/attention_only_head_smoke.py`
- `exports/attention_only_head_smoke/head_a_attention_binary_ds003969.pt`
- `exports/attention_only_head_smoke/metrics_summary.json`
- `exports/attention_only_head_smoke/run_manifest.json`
- `exports/attention_only_head_smoke/step_summary.json`

## Next invent candidates

1. `more_ds001787_subjects` — more BioSemi attention subjects (still open-license).
2. `repack_cbramod_include_pool_head` — ship multi-night pool vigilance binary as pack default.
3. `personal_muse_cal_blocker` — needs user input / personal Muse calibration.
4. More ds003969 subjects if expanding attention smoke (holdout was weak with only 2 train / 1 holdout).

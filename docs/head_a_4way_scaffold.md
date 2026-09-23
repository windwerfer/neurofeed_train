# Head A 4-way scaffold (missing-class mask)

> **Superseded for product architecture (2026-09-07):** Head A is now **multi-head** (separate A-vig / A-med / A-eng), not one exclusive 4-way softmax. See [`head_a_multihead.md`](head_a_multihead.md). This doc remains as the historical masked-CE smoke for legacy 4-way experiments.


**Step:** `head_a_4way_scaffold`  
**Goal:** one shared 4-way Head A (`concentration`, `mind_wandering`, `drowsy`, `hypnagogic`) that can train on partial label corpora without poisoning missing-class logits.

## Design

| Piece | Choice |
|-------|--------|
| Head | `HeadALinear` → 4 logits |
| Labels | `HEAD_A_4WAY_LABELS` in `src/head_a.py` (same order as `src/metrics.py`) |
| Groups | Attention `{concentration, mind_wandering}`; Vigilance `{drowsy, hypnagogic}` |
| Train loss | `masked_cross_entropy`: CE on **present-class logit subset only** |
| Predict (partial) | `predict_present`: argmax restricted to present classes |
| Binary map | Sleep-EDF binary ids `0/1` → 4-way ids `2/3` via `binary_ids_to_4way` |

**Why mask:** Sleep-EDF only supplies vigilance; OpenNeuro/EEGMeditation will supply attention. Alternating or mixing batches must not push softmax mass onto never-seen classes or backprop into those columns. Unit check: missing-class weight grads ≈ 0; present grads move.

## Smoke (this step)

- **Train:** SC4001 balanced drowsy/hypnagogic windows (existing `exports/windows_sc4001`).
- **Holdout:** SC4002 stride×4 windows.
- **Encoder:** frozen CBraMod (CPU OK).
- **Present mask:** `{drowsy, hypnagogic}` only; attention classes remain in the head but unused.
- **Metrics:** report `macro_f1_present` (vigilance only). Naive 4-way macro-F1 is lower because missing classes contribute zero support — do not use it for gating.

Results (CPU smoke): train macro-F1_present ≈ 0.83; val ≈ 0.85; SC4002 stride4 ≈ 0.79. Plumbing / montage caveats from earlier binary smokes still apply.

## Artifacts

- `scripts/head_a_4way_scaffold.py`
- `exports/head_a_4way_scaffold/head_a_4way_masked_smoke.pt`
- `exports/head_a_4way_scaffold/metrics_summary.json`
- `exports/head_a_4way_scaffold/run_manifest.json`
- `src/head_a.py` — 4-way constants + mask helpers

## Next (not this step)

1. Confirm ds001787 response codebook → small Muse-proxy attention windows.
2. Joint schedule: vigilance batches (mask attention) ↔ attention batches (mask vigilance).
3. Optional paired heads (2+2) later if 4-way softmax proves unstable under severe missingness.

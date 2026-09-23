# Classical band-math vs frozen AI heads (CBraMod + REVE)

**Updated (UTC):** 2026-09-17T09:49:02.328015+00:00

Peer-owned classical eval under `band_math/`. AI heads **not** retrained.

## Protocol

| Item | Value |
|------|-------|
| FFT | 256 samples @ 256 Hz (Flutter `muse.rs`) |
| Bands (Hz) | δ 1–4, θ 4–8, α 8–13, β 13–30, γ 30–50 |
| Pads | AF7, AF8 (muse4 order AF7/AF8/TP9/TP10) |
| Window | 2 s `(N,4,512)` npz; average two 1 s FFT halves |
| App features | `band.atr` α/θ, `band.tar` θ/α, `band.btr` β/θ, `band.alpha` rel α, `band.delta` abs δ |
| Metric | macro-F1 on **same test subjects** as AI splits (where available) |
| CBraMod A-vig | `head_a_vig_linear.pt` argmax; test macro-F1 ≈ **0.747** (full corpus) |
| REVE A-vig / Head C | `exports/reve_sleep_heads_local/` — **subsample**, not fair vs full CBraMod |

## Comparison table

| Task | Dataset | Band F1 | CBraMod F1 | REVE F1 | Band vs CBraMod | Band vs REVE | Best classical |
|------|---------|--------:|-----------:|--------:|:---------------:|:------------:|----------------|
| a_vig | Sleep-EDF Expanded + HMC (2→4 muse4 proxy, profe | 0.630 | 0.747 | 0.783* | **lose** | **lose** | `logistic_all_features` |
| head_c_wake_light | Sleep-EDF Expanded + HMC (2→4 muse4 proxy, profe | 0.630 | 0.760 | 0.791* | **lose** | **lose** | `logistic_all_features` |
| a_eng | engagement_a_eng union (mix 19/23/32-ch professi | 0.574 | 0.548 | 0.590 | **win** | **lose** | `threshold:band.atr` |
| mw_attention_ds001787 | ds001787 (64-ch BioSemi→muse4 proxy, professiona | 0.625 | 0.360 | 0.500* | **win** | **win** | `threshold:band.atr` |
| mw_attention_ds003969 | ds003969 (64-ch Muse-proximal muse4, professiona | 0.592 | 0.360 | 0.500* | **win** | **win** | `threshold:extra.delta_theta` |
| a_med_rest_meditation | ds003816 (Muse-native AF7/AF8/TP9/TP10, professi | 0.584 | 0.333 | — | **win** | **—** | `loso_threshold:band.delta` |
| med_depth_q1 | ds001787 (64-ch BioSemi, professional) — muse4 p | 0.704 | 0.333 | — | **win** | **—** | `threshold:extra.alpha_beta` |

\*REVE sleep numbers are **local subsample** (≤80 windows/class/recording) — Diggus: not apples-to-apples with full CBraMod subject-holdout. Prefer a full Kaggle T4 REVE encode before treating REVE as the ship winner.

### Decision rule
`win` / `lose` / `tie` if |band − AI| > 0.01, else tie.

## REVE vs CBraMod (ship heads)

| Task | CBraMod (full) | REVE (local subsample*) | Δ REVE−CBraMod | Fair? |
|------|---------------:|------------------------:|---------------:|:-----:|
| A-vig | 0.747 | 0.783* | +0.036 | **no** |
| Head C wake/light | 0.760 | 0.791* | +0.031 | **no** |
| A-eng | 0.548 | 0.590 | +0.042 | yes (same splits; both misfit) |
| MW attention LOSO | ~0.361 | ~0.500 | +0.139 | LOSO pooled; not ship |

Optional: REVE HMC 5-stage linear-probe bal_acc ≈ **0.649** (`exports/reve_hmc_paper_compare/`) — 30 s epochs; no band-math row.

## Per-task notes

### A-vig / Head C
- **CBraMod beats band** on the fair full-corpus bake-off (0.747 / 0.760 vs band 0.630).
- REVE *looks* higher (~0.78 / ~0.79) but Diggus froze that as **directional subsample only**; do not crown REVE ship winner until full encode.

### A-eng
- Band 0.574 beats CBraMod 0.548, **loses** to REVE 0.590. All still misfit / not ship.

### MW / med
- Band wins vs soft CBraMod bars; REVE attention LOSO ~0.50 is stronger than CBraMod ~0.36 but still not ship. No REVE med/depth metrics on disk.

## Gaps
- Full-corpus REVE A-vig / Head C encode (Kaggle T4) still outstanding for a fair encoder bake-off.
- HMC 5-stage not scored with band math (window mismatch).

## Layout
```
band_math/
  COMPARE_AI_VS_BAND.md
  compare_summary.json
  features/flutter_bands.py
  scripts/run_compare.py
  results/
```


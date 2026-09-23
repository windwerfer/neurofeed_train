# Holdout eval (tightened) — SC4002 night, stride-aware

Generated: 2026-09-06T19:08:35.031939+00:00 (UTC). See `exports/head_a_holdout/tighten_eval_metrics.json`.
User TZ Asia/Bangkok (UTC+7).

## Why tighten

Step 1 scored **every** 2 s window at hop **0.5 s** (4× overlap). Neighbor windows share ~75% of samples, so n=2577 overstates independent evidence and can inflate confidence.

This step keeps the **same frozen Head A** from `exports/head_a_holdout/head_a_binary_state_dict.pt` and re-scores SC4002 with **non-overlapping / stride-aware** windows (≥2 s apart).

Script: `scripts/tighten_holdout_eval.py`. Artifacts: `exports/head_a_holdout/tighten_eval_metrics.json`.

## Recipe reminder

- Window 2 s, train hop 0.5 s, majority 0.7; W→drowsy, N1→hypnagogic.
- Non-overlap grid = every **4th** window (effective hop = window = 2 s).
- Greedy non-overlap = left-to-right by `starts`, keep if start ≥ previous end.

## Headline metrics (SC4002 holdout)

| Scoring | n | accuracy | macro-F1 | notes |
|---------|---|----------|----------|-------|
| Overlap hop 0.5 s (step 1) | 2577 | 0.9131 | 0.7388 | correlated neighbors |
| Non-overlap stride×4 (offset 0) | 645 | 0.9163 | 0.7442 | primary tightened all-labeled |
| Stride×4 mean over offsets 0–3 | ~644 | 0.9131 | 0.7389 | stable across phase |
| Non-overlap greedy | 645 | 0.9178 | 0.7509 | ≈ stride grid |
| Non-overlap greedy **balanced** | 90 | 0.8444 | 0.8416 | fairest vs earlier balanced |
| Non-overlap stride×4 **balanced** | 90 | 0.7778 | 0.7760 | offset-0 subsample |

Takeaway: dropping overlap does **not** collapse the story. All-labeled macro-F1 stays ~**0.74**; balanced non-overlap sits ~**0.78–0.84** (small n=90, one hypnagogic window per 2 s of N1).

## Per-class precision / recall (clear tables)

### A) All-labeled, non-overlap stride×4 (offset 0) — preferred all-night view

Confusion (rows=true drowsy/hypnagogic, cols=pred): `[[560, 40], [14, 31]]`

| class | precision | recall | f1 | support |
|-------|-----------|--------|----|---------|
| drowsy | 0.976 | 0.933 | 0.954 | 600 |
| hypnagogic | 0.437 | 0.689 | 0.534 | 45 |

### B) Non-overlap greedy balanced — preferred balanced view

Confusion: `[[44, 1], [13, 32]]`

| class | precision | recall | f1 | support |
|-------|-----------|--------|----|---------|
| drowsy | 0.772 | 0.978 | 0.863 | 45 |
| hypnagogic | 0.970 | 0.711 | 0.821 | 45 |

### C) Overlap (step 1 reference)

| class | precision | recall | f1 | support |
|-------|-----------|--------|----|---------|
| drowsy | 0.976 | 0.930 | 0.952 | 2398 |
| hypnagogic | 0.423 | 0.693 | 0.525 | 179 |

Hypnagogic **recall ~0.69** is stable overlap→non-overlap. Hypnagogic **precision ~0.42–0.44** on the imbalanced night stays the weak spot (many drowsy false alarms relative to rare N1). On the balanced non-overlap cut, hypnagogic precision rises (~0.97) because the drowsy prior is removed — trust that only as a balanced diagnostic, not as deployment prior.

## How to read vs earlier same-night ~0.83 smoke

- Same-night SC4001 smoke (~0.83) and overlapping balanced holdout (~0.83) were **optimistic / correlated**.
- Tightened **balanced non-overlap** (~0.78–0.84 macro-F1 on n=90) is in the same ballpark but with honest independence and tiny support.
- Tightened **all-labeled** still shows high accuracy from drowsy majority; prefer **macro-F1 ~0.74** + per-class P/R above.

## Artifacts

- Metrics JSON: `exports/head_a_holdout/tighten_eval_metrics.json`
- Head weights (unchanged): `exports/head_a_holdout/head_a_binary_state_dict.pt`
- Windows: `exports/windows_sc4002/sleep_edf_sc4002_n1slice_windows.npz` (`starts` used for greedy)
- Step 1 narrative: `docs/holdout_step1_results.md`

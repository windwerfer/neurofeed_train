# Holdout metrics writeup (plain language)

Overnight Muse EEG Head A — finished 2026-09-07 (Asia/Bangkok).

## What we measured

Train Head A on **SC4001** (one Sleep-EDF night), freeze CBraMod, then score an unseen night **SC4002**. Labels: wake → drowsy, N1 → hypnagogic (sleep-onset). Earlier same-night smoke on SC4001 was ~**0.83** accuracy / macro-F1 — optimistic because train and test share the same night and overlapping windows.

## Headline numbers (trust these)

| View | n | accuracy | macro-F1 | what it means |
|------|---|----------|----------|---------------|
| Same-night smoke (old) | ~300 bal. | ~0.83 | ~0.83 | plumbing only; correlated |
| SC4002 balanced, overlapping | 358 | 0.83 | 0.82 | other-night, still overlap |
| SC4002 all-labeled, **non-overlap** | 645 | 0.92 | **0.74** | honest night view |
| SC4002 balanced, **non-overlap** | 90 | 0.78–0.84 | **0.78–0.84** | fairest vs smoke; tiny n |

Dropping window overlap does **not** collapse the story. Prefer **macro-F1 ~0.74** (all night) or **~0.78–0.84** (balanced non-overlap) over raw accuracy.

## Per class on the holdout night (non-overlap)

- **Drowsy**: precision ~0.98, recall ~0.93 — easy majority class.
- **Hypnagogic**: recall ~**0.69** (catches most true N1), precision ~**0.44** on the full night (many false alarms because drowsy dominates). On a balanced cut, hypnagogic precision jumps (~0.97) — useful as a diagnostic, not a deployment prior.

## vs the ~0.83 same-night smoke

The old 0.83 looked strong but mixed same-night leakage with 4× overlapping windows. True other-night holdout lands in a similar ballpark on a **balanced** cut (~0.78–0.84), while the full night’s honest summary is **macro-F1 ~0.74** with hypnagogic precision as the weak spot. Domain gap (Sleep-EDF montage vs Muse / TUEG pretrain) remains.

## What overnight finished

1. SC4002 night holdout train/eval + private Kaggle kernel/dataset.
2. Both-nights windows verified in `windwerfer/muse-eeg-heads-windows` (private).
3. Stride-aware / non-overlap rescoring → `docs/holdout_eval.md`.
4. This writeup.

Artifacts: `docs/holdout_step1_results.md`, `docs/holdout_eval.md`, `exports/head_a_holdout/`.

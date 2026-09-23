# REVE frozen heads on expanded sleep corpus (plan)

**Status:** WAITING on CBraMod Head C + A-vig local run
**Date:** 2026-09-08

## Goal

Train frozen REVE-base tiny heads for:

1. Head A-vig (drowsy/hypnagogic) on 124-subject sleep corpus
2. Head C (wake/light on N1-slice; deep/REM absent)
3. Head A-eng (later, once engagement_load ~120)

Same subject splits as CBraMod. No backbone fine-tune.

## Runtime

Prefer private Kaggle GPU T4 (not P100) with muse-eeg-heads-cache offline REVE (no HF_TOKEN), windows pack, and src pack.

## After CBraMod metrics

1. Sync sleep windows + splits into Kaggle windows pack
2. Sync src (heads/, train helpers)
3. Push private kernel muse-eeg-heads-reve-sleep-heads (enable_gpu=true, T4)
4. Compare CBraMod vs REVE metrics; freeze vs fine-tune recommendation

## Skip

Head B, A-med/MW/depth/stress public heads.

## Status update 2026-09-08

- Local REVE subsample DONE (`exports/reve_sleep_heads_local/`)
- Windows pack versioned; kernel pushed: https://www.kaggle.com/code/windwerfer/muse-eeg-heads-reve-sleep-heads
- Prefer T4 in UI; pull metrics when complete

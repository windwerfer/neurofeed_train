# Head A-eng corpus expansion (~120 unique subjects)

**Status:** DONE enough for fit/misfit — **133** unique persons / target ~120
**Date:** 2026-09-08T13:41:14.161914+00:00
**Windows:** 19706 across 150 packs
**Goal:** grow labeled low vs high engagement windows toward **~120 unique subjects** for A-eng fit vs misfit (no full REVE/CBraMod train required in this step).

## ALLOW sources ingested

| Source | License | Persons | Packs | Windows | Device | Order / confound |
|--------|---------|--------:|------:|--------:|--------|------------------|
| `ds007169` | CC0-1.0 | 18 | 18 | 2880 | professional (ch tag in manifest) | strict L1→L2→L3→L4; 1-back always before 4-back |
| `ds007262` | CC0-1.0 | 18 | 18 | 1946 | professional (ch tag in manifest) | difficulty randomized across trials (better than ds007169) |
| `ds007554` | CC0-1.0 | 30 | 30 | 3600 | professional (ch tag in manifest) | task order within session not fully counterbalanced; multi-session available |
| `eegmat` | ODC-By-1.0 | 36 | 36 | 3600 | professional (ch tag in manifest) | rest always precedes arithmetic |
| `stew` | CC-BY-4.0 | 48 | 48 | 7680 | hobbyist (ch tag in manifest) | rest then SIMKAP within subject; windows pre-segmented 2s@128Hz |

### Dataset tags (ch, professional|hobbyist)

- **ds007169:** Multimodal Cognitive Workload n-back / ds007169 (19-ch 10–20 mobile EEG, professional)
- **ds007262:** Cognitive Workload 8-level arithmetic / ds007262 (19-ch 10–20 mobile EEG, professional)
- **ds007554:** CMx7-MM hierarchical cognitive-motor / ds007554 (32-ch EEG, professional)
- **eegmat:** PhysioNet EEG During Mental Arithmetic Tasks / eegmat (23-ch 10–20 professional)
- **stew:** STEW Simultaneous Task EEG Workload (Emotiv EPOC 14-ch, hobbyist) — HF processed MONSTER mirror

## Label mapping (honest)

| Source | low_engagement | high_engagement |
|--------|----------------|-----------------|
| ds007169 | main 1-back | main 4-back |
| ds007262 | difficulty 0.6–1.5 | difficulty 5.1–6.9 |
| eegmat | rest (`_1`) | mental arithmetic (`_2`) |
| stew | rating ≤4 (HF y=0) | rating >4 (HF y=1) |
| ds007554 | PassiveMotor or MotorImagery | NbackArithmetic / mental arithmetic / n-back |

## Unique-person policy

- Barras **ds007169** and **ds007262** share participant IDs/demographics → `unique_person_id = barras_{sub}`; both task packs stay in the **same** split. Union ≈ **19** unique Barras people (sub-002 n-back-only; sub-013 arithmetic-only).
- Other corpora use `{source}_{id}`.
- Splits: subject-wise ~70/15/15 on unique_person_id (`datasets/engagement_a_eng/splits/splits.json`).

## Montage

- **muse4 only** — never mixed with crown8.
- Proxies: F7/F8/T3|T7/T4|T8 or Emotiv AF3/AF4/T7/T8 → AF7/AF8/TP9/TP10.
- 2 s windows / 1 s hop @ 256 Hz; per-channel (or per-window) z-score + clip±15.

## Blockers / notes

- **STEW raw** remains IEEE DataPort login-walled; this corpus uses the **HF monster-monash/STEW processed** CC-BY-4.0 mirror (2 s @ 128 Hz windows).
- **UNIVERSE** Muse-S (CC-BY-4.0, n=24) is ~20 GB — deferred (size); would top up if needed.
- ds007169 order confound (L1→L4) remains; ds007262 randomized difficulty is the cleaner Barras signal.
- EEGMAT rest→arith is ordered (time confound).

## Scripts / paths

| Path | Role |
|------|------|
| `scripts/expand_head_a_eng_corpus.py` | download + window + splits |
| `exports/windows_aeng/{source}/` | per-source npz + manifests |
| `datasets/engagement_a_eng/` | corpus symlinks + splits |
| `exports/head_a_eng_expand/` | progress / summary JSON |

## Next (out of scope here)

Optional cheap CBraMod chance-check on expanded set; full A-eng train for fit vs misfit.

## Final counts

| Metric | Value |
|--------|-------|
| Unique persons | **133** (target ~120) |
| Window packs | 150 |
| Windows | 19706 |
| Splits | train 93 / val 20 / test 20 |


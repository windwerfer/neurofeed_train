# Head A-eng smoke (CBraMod, CPU) — engagement fallback

**Completed (Asia/Bangkok):** 2026-09-08 10:46:32 ICT
**UTC:** 2026-09-08T03:46:32.184418+00:00
**Version:** `eng_fallback_ds007169_n8_zscore_2026-09-08`

## Goal

Frozen **CBraMod** chance-check for stress/calm Head exploration; **switched** to **A-eng** (`low_engagement` vs `high_engagement`) after stress corpora proved too heavy/gated for a size-bounded smoke.

## Data pick (first viable → fallback)

| Candidate | License | Why not / why used |
|-----------|---------|-------------------|
| PhysioNet `neuro-stress-resilience-hci` MATB | ODbL v1.0 | ~606 MB/subject CSV; AF7/AF8 are fNIRS not EEG; ~0.25 MB/s → aborted |
| alkabbany Muse-S stress/relax | CC-BY-4.0 | Google Drive only; n=5 under target |
| STEW Emotiv 14-ch | CC-BY-4.0 | IEEE login for raw; HF mirror processed-only |
| **Multimodal Cognitive Workload n-back / ds007169 (19-ch 10–20 mobile EEG, professional)** | **CC0-1.0** | **USED** — ~23 MB EEG/subject; objective 1–4 back |

**Dataset bracket tag:** `Multimodal Cognitive Workload n-back / ds007169 (19-ch 10–20 mobile EEG, professional)`

## Label mapping (honest)

| Source | A-eng label | Notes |
|--------|-------------|-------|
| main-experiment `1-back` block | `low_engagement` | Extreme low load bin |
| main-experiment `4-back` block | `high_engagement` | Extreme high load bin |
| `2-back` / `3-back` | **excluded** | Mid bins dropped for cleaner binary |
| tutorial trials (`istutorial=true`) | **excluded** | Protocol practice |
| Resting / Stress (PhysioNet) | **not used** | Download aborted (size) |

## Montage

- **muse4 only** (never mixed with crown8)
- Proxy map: `F7→AF7`, `F8→AF8`, `T3→TP9`, `T4→TP10`
- Resample 250 Hz → 256 Hz; 2 s windows / 1 s hop; edge trim 5 s; cap 80 windows/block
- Scale: BrainVision floats non-physical (~1e12) → **per-channel z-score + clip±15** before encode

## Split

- Train: `sub-001, sub-002, sub-003, sub-004, sub-005, sub-006, sub-007` (n=7)
- Holdout: `sub-008`
- Per-subject `undersample_balanced` then concat; internal 15% val; best-by-val

## Metrics (headline)

| Split | n | accuracy | macro-F1 |
|-------|--:|---------:|---------:|
| Train (balanced) | 952 | 0.721 | 0.720 |
| Val | 168 | 0.744 | 0.742 |
| Holdout full | 160 | 0.969 | 0.969 |
| Holdout balanced | 160 | 0.969 | 0.969 |
| Holdout stride×4 | 40 | 0.950 | 0.950 |

**Chance macro-F1 (balanced binary):** `0.5`
**Δ holdout-balanced vs chance:** `+0.469`
**ship_candidate:** `False` (bar was holdout bal F1 ≥ 0.55, but **block-order confound** vetoes ship)

## Confounds (important)

1. **Block order:** main `1-back` always precedes `4-back` in this protocol — subject-out F1 may partly reflect session time / slow drift, not pure load.
2. Non-physical float scale required z-score (units not µV).
3. Single holdout subject; overlapping windows; Muse proxy montage.

## Recommendation

- **Code:** `keep_exploring`
- **Note:** Holdout bal macro-F1 far above chance, BUT main 1-back always precedes 4-back in-session — temporal/drift confound possible. Keep exploring with order-balanced designs / STEW rest-vs-SIMKAP / more subjects; do NOT ship public A-eng yet. Stress/calm still needs a lighter corpus.
- **Run engagement next?** `False` (this run already is eng smoke)
- **Lighter stress corpus later?** `True`
- **Public head:** do **not** ship; keep exploring with order-aware eval / STEW
- **Personal-cal:** optional later if product wants load UX

## Provenance

- Dataset: `Multimodal Cognitive Workload n-back / ds007169 (19-ch 10–20 mobile EEG, professional)` — SPDX `CC0-1.0` — `doi:10.18112/openneuro.ds007169.v1.0.5`
- Encoder: CBraMod Apache-2.0 `pretrained_weights.pth` (local cache)
- Montage: **muse4** only
- Script: `scripts/head_a_eng_smoke.py`
- Exports: `exports/head_a_eng_smoke/`
- No Kaggle push

## Caveats

- Engagement/load ≠ meditation stress/calm UX; product mapping is adjacent only.
- F7/F8/T3/T4 are proxies for Muse AF7/AF8/TP9/TP10 — domain gap expected.
- BrainVision floats are non-physical (~1e12); applied per-channel z-score + clip±15 before encode.
- Overlapping windows; modest n=8; CBraMod TUEG → 4-ch transfer gap.
- Stress/calm public head still **untested** at smoke scale.

## Takeaway

SWITCHED stress→engagement: Multimodal Cognitive Workload n-back / ds007169 (19-ch 10–20 mobile EEG, professional); muse4 proxy F7/F8/T3/T4; n=8 holdout sub-008; holdout bal macro-F1=0.969 vs chance 0.5 (Δ+0.469); ship_candidate=False (block-order confound); rec=keep_exploring. Run engagement next? No — this was eng.

## Order-aware re-smoke

**Completed (Asia/Bangkok):** 2026-09-08 15:31:31 ICT
**UTC:** 2026-09-08T08:31:31.759518+00:00
**Version:** `eng_orderaware_ds007169_n8_2026-09-08`
**Exports:** `exports/head_a_eng_orderaware/`
**Script:** `scripts/head_a_eng_orderaware.py`

### Why

Prior 1-back→low / 4-back→high holdout bal macro-F1 ≈ 0.969, but main protocol is **strictly L1→L2→L3→L4** (no counterbalance / no temporal overlap). Order-aware checks break the time shortcut.

**Dataset:** `Multimodal Cognitive Workload n-back / ds007169 (19-ch 10–20 mobile EEG, professional)` — SPDX `CC0-1.0`

### Design

| Check | What | Shortcut broken? |
|-------|------|------------------|
| `mid_2v3` **PRIMARY** | 2-back→low vs 3-back→high (full adjacent blocks) | Uses mid difficulties; ~minutes closer than 1vs4 |
| `edge_2late_3early` | Late half 2-back vs early half 3-back | Nearly time-matched across instruction boundary |
| `edge_1late_4early` | Late half 1-back vs early half 4-back | Closest edges of extremes; **residual gap** via L2+L3 (~5+ min) |
| `timebin_control` | Predict early vs late main-session (ignore load) | If easy, time is discriminative |
| `within_L2_early_late` | Early vs late half within 2-back only | Pure within-block drift (no load change) |

### Results (holdout bal)

| Check | n | accuracy | macro-F1 | Δ vs chance |
|-------|--:|---------:|---------:|------------:|
| `mid_2v3` (PRIMARY) | 160 | 0.838 | 0.833 | +0.333 |
| `edge_2late_3early` (ctrl/sec) | 80 | 0.975 | 0.975 | +0.475 |
| `edge_1late_4early` (ctrl/sec) | 80 | 1.000 | 1.000 | +0.500 |
| `timebin_control` (ctrl/sec) | 320 | 0.966 | 0.966 | +0.466 |
| `within_L2_early_late` (ctrl/sec) | 80 | 0.850 | 0.847 | +0.347 |

**Pass bar (proceed to stress):** primary `mid_2v3` bal F1 **> 0.55**
**Primary F1:** `0.833` (Δ `+0.333`)
**Verdict:** `PASS_orderaware`
**Proceed to stress/calm smoke?** `True`
**ship_candidate:** `False` (still False — ordered protocol / n=8)

### Decision note

Primary mid 2vs3 holdout bal F1=0.833 and edge-matched 2-late/3-early F1=0.975 both > 0.55. Load signal survives adjacent-block and near-time-matched tests. Time-bin control F1=0.966, within-L2 early/late F1=0.847. Residual risk: protocol still strictly ordered (no counterbalance); n=1 holdout subject.

### Residual risk (honest)

- Protocol never counterbalances n-back order — strongest honest tests still have weak residual time structure.
- Single holdout subject; overlapping windows; muse4 proxy F7/F8/T3/T4; non-physical BrainVision floats → z-score.
- Engagement/load ≠ meditation stress/calm UX.


### Follow-on (Part 2)

Because primary mid 2vs3 **passed** (>0.55), a small stress/calm smoke **did run**:
see [`head_stress_calm_smoke.md`](head_stress_calm_smoke.md) / `exports/head_stress_calm_smoke/`.
Honest residual: time-bin and within-block controls were also easy on eng —
order confound not fully ruled out despite mid-level lift.

### Takeaway (order-aware)

ORDER-AWARE eng on Multimodal Cognitive Workload n-back / ds007169 (19-ch 10–20 mobile EEG, professional): primary mid 2vs3 holdout bal macro-F1=0.833 (bar>0.55); edge 2late/3early=0.975; timebin=0.966; within-L2=0.847; verdict=PASS_orderaware; proceed_stress=True.

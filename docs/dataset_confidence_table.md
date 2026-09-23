# Dataset confidence table

Summary of major corpora touched in muse-eeg-heads R&D. Use with [`INDEX.md`](INDEX.md) and [`LICENSE_NOTES.md`](LICENSE_NOTES.md).

**Report format:** `Name (N-ch, professional|hobbyist)`.

## Confidence scale (0–5)

| Score | Meaning |
|------:|---------|
| **5** | Ship-ready subject-holdout (clears bar; open license; honest split) |
| **4** | Strong research match / near ship (e.g. paper-matched LP, or high probe F1 but not product-complete) |
| **3** | Promising but incomplete (volume/protocol OK; missing pieces or soft bar) |
| **2** | Weak / confounded (order shortcuts, noisy labels, domain mix) |
| **1** | Chance / collapse (holdout ≈ chance or majority-class collapse) |
| **0** | Unusable **or DENY license for ship** (BY-NC / academic-only / excluded) |

## Summary table

| Dataset | Electrodes (class) | License | Role | Useful for ship? | Confidence (0–5) | Evidence (metric) | Notes |
|---------|-------------------|---------|------|:----------------:|:----------------:|-------------------|-------|
| **Sleep-EDF Expanded** | 2→4 muse4 proxy, professional | PhysioNet / ODC-By (attribute; derived heads) | **A-vig** primary; **Head C** volume | **Yes** (A-vig) | **5** | CBraMod A-vig test **macro-F1 ≈ 0.747** on **124-subj** pool (subject-holdout); Head C 2-way wake/light test macro-F1 ≈ 0.760 (probe, not 4-way ship) | W→drowsy, N1→hypnagogic; part of `vigilance_sleep_edf` with HMC |
| **HMC** sleep staging | 4-ch clinical, professional | CC-BY-4.0 | Sleep LP / vig top-up; part of 124 | **Yes** (with Sleep-EDF) | **4–5** | Frozen REVE HMC 5-stage LP **bal_acc ≈ 0.649** ≈ paper Pool **0.647**; also in 124-subj CBraMod vig/C trains | Fair paper match on 24 local nights; Full-night 5-stage ≠ Head C N1-slice 2-way |
| **ds001787** | 64-ch BioSemi, professional | CC0 | Attention / MW probes; A-med **depth** Q1 | No | **1** | Depth smoke holdout bal macro-F1 **≈ 0.333**; MW/attention LOSO historically near chance (~0.36 CBraMod) | muse4 proxy AF7/AF8 + P9/P10→TP9/TP10; sparse subjective probes |
| **ds003969** | 64-ch (Muse-proximal), professional | CC0 | MW **block** proxy (med* vs think*) | No | **1** | Attention pack `ship_candidate: false`; think-blocks **poison** for true MW | Volume only; instructed thinking ≠ probe MW |
| **ds003816** | Muse-native AF7/AF8/TP9/TP10, professional | CC0 | A-med rest↔med (PreResting / LKMSelf) | No | **1** | Holdout bal macro-F1 **≈ 0.333** (n=12; no lift vs n=6) | Collapse to always-`rest` on holdout |
| **L-FAME** | 64-ch 10–20, professional | **CC-BY-NC** | Research-only A-med rest↔med | **No — DENY** | **0** | Holdout bal macro-F1 **≈ 0.332**; `research_invest=False` | BY-NC — never in ship mix / public Kaggle ship packs |
| **engagement_a_eng** union (ds007169, ds007262, eegmat, STEW, ds007554) | mix 19/23/32-ch professional + STEW 14-ch Emotiv **hobbyist** | CC0 / ODC-By / CC-BY-4.0 (STEW HF) | **A-eng** load/engagement | **No** | **2** | Frozen CBraMod test macro-F1 **≈ 0.548**; REVE **≈ 0.590** — **misfit**; cleaner ds007262 weak | Order confounds (L1→L4, rest→arith, rest→SIMKAP); ~133 unique persons |
| **eegmat** (stress/calm smoke) | 23-ch 10–20, professional | ODC-By-1.0 | Stress vs calm chance-check | No | **1** | Holdout bal macro-F1 **≈ 0.495** (≈ chance) | Also used inside A-eng union; rest→arith order confound |
| **REVE / CBraMod model caches** | n/a (encoders) | CBraMod Apache-2.0; REVE gated | Frozen backbones | Cache only | — | Not datasets — private `muse-eeg-heads-cache` | Do not redistribute gated REVE base; publish heads only |

### Union members (A-eng detail)

| Source in union | Electrodes (class) | License | Note in dual-encoder test |
|-----------------|-------------------|---------|---------------------------|
| ds007169 | 19-ch mobile, professional | CC0 | Strong F1 but **order** L1→L4 confound |
| ds007262 | 19-ch mobile, professional | CC0 | Randomized difficulty (**cleaner**) — stays weak |
| eegmat | 23-ch, professional | ODC-By | rest→arith order |
| STEW (HF) | 14-ch Emotiv, **hobbyist** | CC-BY-4.0 | rest→SIMKAP order; device mix |
| ds007554 | 32-ch, professional | CC0 | Task-order residual |

## How to use this table

- **Ship mix:** only rows with open ALLOW licenses **and** confidence that supports the claimed head (today: Sleep-EDF + HMC for **A-vig**).
- **Tutor bots:** cite Evidence column; do not invent metrics — prefer the linked train/smoke docs under `docs/`.
- **R&D bots:** confidence ≤2 → personal cal or research-only; do not queue public ship heads.

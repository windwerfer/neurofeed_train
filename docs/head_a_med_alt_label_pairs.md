# A-med alternate label pairs (post n=12 rest↔med smoke)

**Date:** 2026-09-07 (Asia/Bangkok)  
**Trigger:** `exports/head_a_med_smoke_n12/` — holdout `sub-23st` balanced macro-F1 **0.333** (Δ vs n=6 = **0.000**); `ship_candidate: false`. Holdout confusion = **all predicted `rest`** (never fires `meditation`). Val F1 rose to 0.64 (in-distribution fit) without subject-holdout lift.

**Question:** Which **other binary label pair** is more useful for a Muse meditation app than protocol `PreResting`↔`LKMSelf`?

---

## Recommendation (short)

**Next A-med / meditation-state smoke target: `meditation_depth_high` vs `meditation_depth_low` on OpenNeuro ds001787 (CC0), muse4, using probe Q1 (concentration/depth 0–3) extremes.**

**Why:** Rest↔med is a weak product signal (the app already knows the user started a session). Live UX wants **in-session depth / quality**. ds001787 already has muse4 plumbing, honest probe labels, and CC0 — no new gated/NC corpus. FA vs OM and novice vs experienced lack strong open Muse-mapped sets or are trait-not-state.

Optional diagnostic (not a new product label): one **PSD/CSP muse4** rest↔LKM check on ds003816 — literature CSP+LDA on the same corpus is strong on dense EEG; if classical features also fail on muse4, drop rest↔med for subject-general shipping and keep it for personal cal only.

---

## Ranked alternatives

| Rank | Pair | Product UX | Corpus | License | n | Muse fit | Verdict |
|-----:|------|------------|--------|---------|--:|----------|---------|
| **1** | **Med depth high vs low** (Q1≥2 vs Q1≤1, or extremes 3 vs 0) | Live “going deeper” meter during a session | **ds001787** Brandmeyer & Delorme | **CC0** | 24 (sparse probes ~2 min) | **Excellent** — already muse4/crown8 in-repo | **Do next** |
| **2** | **Richer rest↔med** — Pre+PostResting vs LKMSelf+LKMOther | Same binary as today; more blocks/subject | **ds003816** (same) | **CC0** | 48 (st+lt); native AF7/AF8/TP9/TP10 | **Native muse4** | Salvage attempt / CSP diagnostic only; n=12 already failed same mapping |
| **3** | **Eyes-closed rest vs breath-focus meditation** | Closest to Muse breath sessions | **L-FAME** (HF / arXiv 2605.22893) | **CC BY-NC 4.0** | 74 pre / 44 post | 64-ch 10–20 → AF7/AF8/TP proxy | Research-only unless NC waived; **not ship base** |
| **4** | **LKM vs visualization** (LKMSelf vs VisualizeSelf) | Mode switch (compassion vs imagery) | **ds003816** | **CC0** | same | Native muse4 | Niche UX; try after depth |
| **5** | **Breath-counting vs tradition-specific med** (ds003969 med1 vs med2) | Technique ID, not “meditating?” | **ds003969** | **CC0** | 98 | Good (local muse4/crown8) | Weak primary UX; both classes are meditation |
| **6** | **FA vs OM / state mindfulness** | Style-aware coaching | OSF [buxah](https://osf.io/buxah/); Rodriguez-Larios OSF 3uszv | Open / CC0 (check deed) | 16; 58 | Poor–partial (19-ch or small n) | **Not ready** — no large Muse-mapped open FA/OM set |
| **7** | **Novice vs experienced** (st vs lt on ds003816) | Onboarding / trait badge | **ds003816** | **CC0** | 33 st / 15 lt | Native muse4 | **Trait, not live state** — skip for A-med head |

---

## Detail on top picks

### 1. Meditation depth (recommended)

- **Labels:** From ds001787 thought probes — Q1 “depth of meditation / concentration” ordinal 0–3. Binary: high (2–3) vs low (0–1), or extremes only to cut noise.
- **Why better than rest↔med:** Operates **during** meditation (what Muse users care about). App can show depth without needing a separate rest block.
- **Fit:** Same montage/windowing stack as Head A attention; no OpenNeuro pacing beyond what’s already cached.
- **Risks:** Same subjective-probe noise that caps MW LOSO (~0.36 CBraMod); expect within-subject >> LOSO. Prefer personal cal for ship, public LOSO as plumbing check.
- **Not the same as:** Q1 vs Q2 concentration↔MW (already tried) — this is **within-meditation depth**, one scale.

### 2. Richer ds003816 rest↔med (last salvage)

- Add `PostResting`→rest and `LKMOther`→meditation; keep holdout subject-disjoint.
- CSP paper on this corpus (Brain Informatics 2023, DOI 10.1186/s40708-023-00204-9) reports usable rest vs LKM with CSP+LDA on dense EEG — suggests **label is real**, but **frozen CBraMod + muse4 + subject holdout** is the failure mode (n=12 holdout = always `rest`).
- If PSD/CSP muse4 also ≈ chance LOSO → abandon subject-general rest↔med.

### 3. L-FAME rest vs breath-focus

- Segments: `restCE*` vs `Medita` / `slMedita`; groups include Breath Focus (BF) — best open match to Muse breath UX.
- **License blocker:** CC BY-NC 4.0 → fine for internal research, **not** a commercial pack without permission.
- Prefer only after #1 if depth also fails and NC is acceptable for a research track.
- **Done 2026-09-08 (research-only):** `docs/head_a_med_lfame_smoke.md` — BF n=16, `restCE01`↔`slMedita`, muse4 native AF7/AF8/TP9/TP10, holdout `sub-066` bal macro-F1 **0.332** (Δ −0.168 vs chance); collapse-to-`rest`; `verdict_invest: false`; `ship_mix: excluded`. Same subject-general failure as ds003816 / depth → **do not invest**; personal cal + A-vig.

### 4–7 (brief)

- **LKM vs Visualize:** Honest block contrast, native muse4; product story is secondary.
- **ds003969 technique contrast:** Local volume; both sides are meditation.
- **FA vs OM:** User-suggested; literature-valid distinction, but open Muse-mapped corpora are tiny or montage-poor.
- **Novice vs experienced:** Useful for analytics, not live session feedback.

---

## Explicitly deprioritized

| Idea | Why not now |
|------|-------------|
| HF `alexeykashevnik/EEGMeditation` | Gated — policy: no gated shipping bases |
| Sleep-EDF W as “rest” | A-vig drowsy proxy — dishonest for A-med |
| ds003969 `think*` as rest | Instructed thinking ≠ rest |
| REVE encoder for this pair | Out of scope unless tiny free path; prior attention REVE still ~0.50 F1 |
| More ds003816 subjects beyond n=12 with same PreResting↔LKMSelf | Val overfit without holdout lift — **label/encoder/montage**, not n |

---

## Suggested next experiment

1. ~~Smoke **HeadAMedDepth** on **ds001787** muse4~~ — **done 2026-09-08** (`docs/head_a_med_depth_smoke.md`): holdout bal macro-F1 **0.333**, always predicts `low`; `ship_candidate: false`.
2. ~~L-FAME BF restCE01↔slMedita research-only~~ — **done 2026-09-08** (`docs/head_a_med_lfame_smoke.md`): holdout bal macro-F1 **0.332**; BY-NC DENY; `verdict_invest: false`.
3. Optional **tiny** PSD/CSP diagnostic on ds003816 muse4 PreResting↔LKMSelf (no Kaggle push).
4. Depth + L-FAME both ~chance → treat meditation-state heads as **personal Muse calibration only**; keep vigilance as the subject-general ship path.

**Artifacts:** n=6 `exports/head_a_med_smoke/`; n=12 `exports/head_a_med_smoke_n12/`; smoke doc `docs/head_a_med_smoke.md`.

---

## Depth smoke result (2026-09-08)

Ran `scripts/head_a_med_depth_smoke.py` on **ds001787 (64-ch BioSemi, professional) — muse4 proxy channels AF7/AF8/TP9/TP10 extracted (TP9/TP10←P9/P10)** muse4, Q1≤1→low / Q1≥2→high, **1 window/probe**, holdout `sub-020`.

- Holdout balanced macro-F1: **0.333** (chance 0.50, Δ -0.167)
- LOSO-lite mean balanced macro-F1: **0.333** (5 folds)
- `ship_candidate`: `False`
- Doc: [`head_a_med_depth_smoke.md`](head_a_med_depth_smoke.md)

**Preferred A-med live operationalization:** meditation depth high vs low (this smoke) over protocol rest↔med — better in-session UX even if public subject-general F1 stays near chance (personal Muse cal still likely for ship).


# Mind-wandering / concentration label options (Head A attention)

> **2026-09-07 architecture lock:** product Head A is **multi-head** ([`head_a_multihead.md`](head_a_multihead.md)). Public MW/concentration frozen decoder remains deferred / personal-cal only.


**Date:** 2026-09-07 (UTC)  
**Scope:** Research for windwerfer / muse-eeg-heads Flutter Muse meditation app — Head A locked labels `concentration`, `mind_wandering`, `drowsy`, `hypnagogic`.  
**Question:** Is there a realistic path to **better MW / concentration labels**, or should we **change the pipeline test target** for Head A attention (while vigilance stays the ship path)?

**Current baseline (known):** Attention pack = OpenNeuro **ds001787** + **ds003969** (CC0), muse4 + crown8. LOSO: CBraMod muse4 macro-F1 ≈ **0.36**; Crown8 ≈ **0.35**; REVE muse4 ≈ **0.50**. `ship_candidate: false`. Diagnosis leans **noisy/weak labels**, not montage alone.

---

## Executive answer

**There is no realistic path to a shippable, subject-general Muse4 `concentration`↔`mind_wandering` head from public corpora alone.** Literature LOSO ceilings for probe-labeled MW are typically **~0.60–0.72 AUC / ~0.55–0.68 accuracy**; REVE at **0.50 macro-F1** is already in the “noisy self-report” regime. Better open corpora exist and are worth ingesting for a research track, but they will not turn Head A attention into a ship feature.

**Recommendation (ranked):** prioritize **(C) change the pipeline test target** + **(D) personal Muse calibration for product attention**, with **(B) filter/re-weight ds001787** as a cheap experiment and **(A) ATTLAPSE / ROAMM** as optional stronger MW research packs. Keep **vigilance (`drowsy`/`hypnagogic`)** as the Head A ship path.

---

## 1. Candidate corpora

### 1.1 Thought-probe / experience-sampling MW

| Dataset | License | n | Channels / rate | Label quality | Muse4 map | Download | Head A fit | Verdict |
|---------|---------|--:|-----------------|---------------|-----------|----------|------------|---------|
| **ATTLAPSE** ([Zenodo 17314289](https://doi.org/10.5281/zenodo.17314289)) | **CC-BY-4.0** | 56 (28 ADHD + 28 HC) | BioSemi **64** @ 2048→**512 Hz** | **Best open probe set:** SART blocks + end-of-block probes: on-task / **spontaneous MW** / **deliberate MW**; 20-trial pre-probe windows; also boredom/fatigue/motivation VAS | AF7/AF8 + TP7/TP8 or P9/P10→TP9/TP10 (standard 10–20) | Open ZIP ~41 GB | Strong for `concentration` vs `mind_wandering` (merge MW-S+MW-D or keep 3-way) | **A1 — ingest next if continuing MW research** |
| **ROAMM** OpenNeuro [ds007629](https://openneuro.org/datasets/ds007629) / NEMAR [on007629](https://nemar.org/dataexplorer/detail?dataset_id=ds007629) | **CC0** | **44** (~50 h) | BioSemi **64** @ **256 Hz**; confirmed **AF7, AF8, P9, P10, TP7, TP8** | Span-level MW via **retrospective** ReMind self-report (not online thought probes); paper LOSO MW detection up to **~0.609 AUROC** (EEG±eye) | **Excellent** muse4 proxy (same P9/P10 recipe as ds001787) | OpenNeuro/NEMAR; synced ML pickles under `derivatives/synced/`; raw BIDS still expanding | Good MW detector bench; **task = reading**, not meditation | **A2 — strong MW bench; weaker product taxonomy** |
| **Rodriguez-Larios** ([OSF 3uszv](https://osf.io/3uszv/), [NeuroImage 2021](https://doi.org/10.1016/j.neuroimage.2021.118669)) | **CC0** | 58 (29 meditators + 29 novices) | **19-ch** @ typical research rate | Meditation + experience sampling: breath vs thoughts vs other; **5 s pre-bell**; confidence + drowsiness follow-ups | Poor (19-ch; no native AF7/AF8/TP9/TP10) | OSF open | Best **meditation-native** probe taxonomy among open sets | **A3 — taxonomy gold, montage mismatch** |
| **MM-SART** (Chen et al. [JBHI 2022](https://doi.org/10.1109/jbhi.2022.3187346) / [arXiv](https://arxiv.org/abs/2005.12076)) | Claimed open; site dead | 82 | 32 EEG + PPG/GSR/eye | Pseudo-random probes during SART; LOSO AUC ~0.71–0.725 in paper | Partial (need channel list) | **mmsart.ee.ntu.edu.tw offline** — not downloadable | Would be top-tier if recoverable | **Blocked — contact authors; do not plan around** |
| **BCIT Mind Wandering** [ds004121](https://openneuro.org/datasets/ds004121) | **CC0** | **20** | 64 (10–10) | Driving + audio conditions + vigilance performance; **not classic thought probes** — fatigue / time-on-task proxies | Muse-proxy possible | OpenNeuro | Closer to vigilance/fatigue than meditation MW | **Low priority for Head A attention** |

### 1.2 Sustained attention / SART / CPT (behavioral labels)

| Dataset | License | n | Notes | Fit |
|---------|---------|--:|-------|-----|
| **ds004350** Executive Functioning ([OpenNeuro](https://openneuro.org/datasets/ds004350)) | **CC0** | 24 | SART + n-back + local-global; **no thought probes** — use commission errors / RTV as **objective lapse** proxies | Good **alternate pipeline target** (on-task vs lapse), not true MW |
| ATTLAPSE (above) | CC-BY-4.0 | 56 | SART **with** probes — preferred over ds004350 for MW | Prefer ATTLAPSE |

### 1.3 Meditation depth / FA vs OM

| Dataset | License | n | Label style | Fit |
|---------|---------|--:|-------------|-----|
| **ds001787** (current) | CC0 | 24 | ~2 min probes: Q1 med depth vs Q2 MW depth (0–3); paper epoch −10 s | Already best open **meditation probe** set; sparse + subjective |
| **ds003969** (current) | CC0 | 98 | **Block** med* vs think* — instructed “think actively”, not probe-caught MW | Volume only; **weak MW ground truth** |
| HF **alexeykashevnik/EEGMeditation** | CC-BY-4.0 **gated** | 33 | Explicit MW + internal/external concentration **blocks** | Stronger blocks than ds003969 but **gated** — policy: no gated bases shipping |
| **L-FAME** ([arXiv](https://arxiv.org/html/2605.22893v1) / HF) | TBD on release | 74 | Rest vs FA meditation techniques; longitudinal | Rest↔med, **not** MW probes |
| OSF longitudinal FA/OM ([osf.io/buxah](https://osf.io/buxah/)) | Open (check deed) | 16 × many sessions | State mindfulness ratings during FA/OM | Small n; continuous ratings not Head A taxonomy |

### 1.4 Engagement / cognitive load (alternate product-relevant targets)

| Dataset | License | n | Channels | Label | Muse fit | Notes |
|---------|---------|--:|----------|-------|----------|-------|
| **STEW** ([IEEE DataPort](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset), HF mirror CC-BY-4.0) | **CC-BY-4.0** | 48 | Emotiv 14 @ 128 Hz (AF3/AF4, T7/T8, P7/P8…) | Rest vs SIMKAP multitask; 1–9 perceived workload | Frontal AF≈Muse AF; temporal/parietal proxies | Classic **engagement/load** bench; consumer dry-electrode closer to Muse than BioSemi |
| **ds007169** n-back workload ([OpenNeuro](https://openneuro.org/datasets/ds007169)) | **CC0** | 18 | 19-ch 10–20 @ 250 Hz | 1–4 back difficulty (objective) | Fp1/Fp2/F7/F8/T3/T4… — approximate | Clean **objective** difficulty labels |
| **ds007262** arithmetic workload | **CC0** | 18 | Same mobile 19-ch | 8 difficulty bins | Same | Objective load continuum |
| PhysioNet stress/MATB multimodal | PhysioNet ODC-By-ish | ~37 | 32 EEG incl. AF3/AF4/AF7/AF8 | Task stress / MATB | Partial AF7/AF8 | Stress ≠ meditation engagement |

These are the best **pipeline-hardening** substitutes if the goal is “prove Head A training/eval works,” not “ship MW.”

---

## 2. Critique: current ds001787 / ds003969 vs MW decoding best practices

### What we do today

| Corpus | Rule | Gap vs literature |
|--------|------|-------------------|
| **ds001787** | Q1>Q2→`concentration`, Q1<Q2→`mind_wandering`; ties drop; ~10 s pre-Q1 | Matches Brandmeyer & Delorme paper contrast — **correct for that study**. Still: probes every ~2 min → few independent labels/subject; ordinal scales forced to binary; no confidence filter; absolute rating bias across people (paper itself notes this) |
| **ds003969** | `med*`→concentration, `think*`→mind_wandering | **Protocol proxy, not MW.** “Think actively” is intentional mentation, often **task-related**, not probe-caught task-unrelated thought. Inflates N while **poisoning** the MW class definition |

### Literature best practices (probe-era MW EEG)

Sources: [Jin et al. PLOS ONE 2021](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251490); [Jin et al. bioRxiv 2020](https://doi.org/10.1101/2020.12.08.416040); [Groot et al. CABN 2019](https://doi.org/10.3758/s13415-019-00707-1); ATTLAPSE / MM-SART papers.

1. **Online experience sampling** with clear OT vs TUT definitions (and ideally MW subtype: spontaneous vs deliberate).
2. Label **short pre-probe windows** (≈10–20 s / ~10 trials), not entire inter-probe blocks.
3. Prefer **extreme / high-confidence** reports; drop mid-scale and low-confidence answers (reduces label noise).
4. Expect **within-subject** >> **LOSO** performance; LOSO ~0.55–0.65 accuracy / ~0.60–0.72 AUC is “success,” not failure.
5. Do **not** treat instructed rest / instructed thinking / meditation blocks as MW ground truth without probes.
6. Behavioral co-labels (SART commission errors, RT variability) help validate but are not identical to self-reported MW.

### Diagnosis of our ~0.36–0.50 LOSO

Consistent with **label noise + domain mismatch**, not a broken encoder:

- Mixing **probe** (ds001787) with **think-blocks** (ds003969) trains conflicting notions of `mind_wandering`.
- Sparse probes → heavy window duplication / temporal autocorrelation → optimistic windows, weak subject-out.
- Muse4 spatial under-sampling of DMN/posterior markers that papers use for MW.
- REVE **0.50** beating CBraMod **0.36** shows encoder headroom, but both sit near the noisy-label floor for ship thresholds (project wants mean F1>0.60 and mean−std>0.55).

---

## 3. Ranked options

### (A) Better MW corpora — *research track, not ship*

| Rank | Action | Why | Cost / risk |
|-----:|--------|-----|-------------|
| 1 | Ingest **ATTLAPSE** muse4/crown8 windows; LOSO OT vs (MW-S∪MW-D) | True probes + SART behavior; CC-BY-4.0; n=56; 64-ch mappable | ~41 GB; ADHD half of cohort — stratify or restrict to HC for first LOSO |
| 2 | Ingest **ROAMM** muse4 (AF7/AF8/P9/P10@256 Hz); use span MW labels / synced derivatives | CC0; n=44; published MW LOSO ~0.61 AUROC | Reading domain ≠ meditation UX; retrospective labels |
| 3 | Optional: Rodriguez-Larios OSF for meditation-native probes | Best taxonomy match | 19-ch — only for encoder transfer experiments, not Muse ship packs |
| — | Skip MM-SART until authors rehost | Best published AUC, unavailable | — |
| — | HF EEGMeditation only if policy allows gated **research** (not ship pack) | Strong blocks | Gated; large |

**Expected outcome:** LOSO may climb toward **~0.55–0.65** macro-F1/AUC on ATTLAPSE/ROAMM — enough to validate the pipeline, **unlikely** to clear ship_candidate under current thresholds.

### (B) Re-label / filter current data — *cheap experiment*

Do this **before** any large new download:

1. **Train/eval on ds001787-only** (drop ds003969 from attention LOSO) — removes think-block poison.
2. **Confidence / extremity filter:** keep only |Q1−Q2| ≥ 2; drop ties and ±1 soft calls.
3. **Subject-normalize** Q1−Q2 (z-score within subject) before thresholding — addresses rating bias.
4. Use **one window per probe** (or ≤2 non-overlap) instead of dense sliding windows across the 10 s epoch.
5. Optionally soft-label with Q3 tiredness → down-weight drowsy-confounded “MW.”
6. Keep ds003969 only as an **auxiliary** med-vs-think task head, **not** as MW.

**Expected outcome:** modest LOSO gain (few points); clarifies whether ds003969 was hurting. Will not create ship-quality MW alone (n≈24, sparse probes).

### (C) Alternate pipeline test target — *recommended for Head A attention slot*

Keep taxonomy labels in the app schema, but **stop using MW LOSO as the gate** for attention.

| Alternate target | Corpus | Why product-relevant | Why better for pipeline tests |
|------------------|--------|----------------------|-------------------------------|
| **Cognitive load / engagement** low vs high | STEW (CC-BY) or ds007169 (CC0) | “Session intensity / mental effort” UX adjacent to meditation apps | Objective or block difficulty; literature reports much higher subject-out accuracy than MW |
| **SART on-task vs behavioral lapse** | ATTLAPSE or ds004350 | Attention stability / “in the zone” | Commission errors / RTV less subjective than MW probes |
| **Rest vs focused breath** (not MW) | L-FAME / ds001787 deep-Q1-only | Meditation “focused” indicator | Cleaner binary than Q1 vs Q2 |

**Vigilance stays ship path:** Sleep-EDF / multi-night `drowsy`↔`hypnagogic` already the honest product win for Head A.

### (D) Personal Muse calibration for attention — *product path (like Head B)*

Mirror Head B few-shot:

1. In-app **5–10 min** calibration: breath-count focus block + free mind-wander / “let thoughts come” block (+ optional SART-like tap task).
2. User-confirmed state tags (and optional mid-block probe).
3. Fine-tune or calibrate only the attention logits / temperature / bias on-device or per-account; **never** mix into public test folds.
4. Ship public Head A as **vigilance + optional engagement**, with MW as **personalized** soft score.

This matches the physics of the problem: MW self-report is idiosyncratic; consumer 4-ch needs subject adaptation.

---

## 4. Clear recommendation

| Priority | Decision |
|----------|----------|
| **1** | **Change the attention pipeline test target (C):** evaluate Head A attention engineering on **load/engagement or SART-lapse** (STEW / ds007169 / ATTLAPSE behavior), not on ds001787∪ds003969 MW LOSO. |
| **2** | **Ship path unchanged:** vigilance (`drowsy`/`hypnagogic`) remains the Head A feature that can clear `ship_candidate`. |
| **3** | **Product attention UX → (D)** personal Muse calibration (Head-B-style), not a frozen public MW decoder. |
| **4** | **Cheap check (B):** ds001787-only + |Q1−Q2|≥2 + one-window-per-probe LOSO to confirm label-noise diagnosis. |
| **5** | **Optional research (A):** ATTLAPSE then ROAMM muse4 packs if you want a credible MW science track; do not block Flutter ship on their F1. |

**Bottom line:** A realistic path exists to **better** MW labels (ATTLAPSE/ROAMM/filters), but **not** to a shippable public MW head on Muse4. Treat MW as personalized; retarget pipeline tests; keep vigilance as the shippable Head A attention-adjacent win.

---

## 5. Key links

- Current packs: [ds001787](https://openneuro.org/datasets/ds001787), [ds003969](https://openneuro.org/datasets/ds003969)
- ATTLAPSE: https://doi.org/10.5281/zenodo.17314289 (CC-BY-4.0)
- ROAMM: https://openneuro.org/datasets/ds007629 · https://nemar.org/dataexplorer/detail?dataset_id=ds007629 (CC0)
- Rodriguez-Larios OSF: https://osf.io/3uszv/ (CC0)
- STEW: https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset
- Workload CC0: [ds007169](https://openneuro.org/datasets/ds007169), [ds007262](https://openneuro.org/datasets/ds007262)
- BCIT driving MW: [ds004121](https://openneuro.org/datasets/ds004121)
- Project context: `docs/attention_corpora_status.md`, `docs/ds001787_codebook_confirm.md`, `docs/reve_attention_loso.md`, `docs/labeling.md`

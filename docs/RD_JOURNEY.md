# R&D journey — Muse EEG Heads

Chronological but scannable narrative for **tutor bots** and onboarding. Facts from `docs/` smokes and trains — not invented. Times are experiment dates (mostly 2026-09-06 → 2026-09-09).

See also: [`INDEX.md`](INDEX.md), [`dataset_confidence_table.md`](dataset_confidence_table.md).

---

## 1. Lock: Head A multi-head (2026-09-07)

**Decision:** drop exclusive 4-way/5-way softmax as the product architecture.

```
window → frozen encoder (CBraMod / REVE) → emb
           ├─ A-vig  drowsy / hypnagogic (± awake)     ← ship first
           ├─ A-med  rest / meditation                 ← beginners UX
           └─ A-eng  low / high engagement             ← needs own corpora
```

- Public **MW / concentration** frozen decoder **deferred** → personal Muse calibration.
- Montages **muse4** vs **crown8** never mixed.
- Doc: [`head_a_multihead.md`](head_a_multihead.md).

---

## 2. Label smokes (honest chance-checks)

All: frozen CBraMod, muse4, subject holdout unless noted. **ship_candidate: False** throughout this section.

| Smoke | Corpus | Metric | Verdict |
|-------|--------|--------|---------|
| **MW / attention** | ds001787 + ds003969 | LOSO ~0.36 CBraMod / ~0.50 REVE muse4 | Noisy probes + think-block **poison** (ds003969) — no public MW head ([`mind_wandering_label_options.md`](mind_wandering_label_options.md), [`attention_corpora_status.md`](attention_corpora_status.md)) |
| **A-med rest↔med** | ds003816 PreResting / LKMSelf | Holdout bal macro-F1 **≈ 0.333** | Collapse; n=12 no lift vs n=6 ([`head_a_med_smoke.md`](head_a_med_smoke.md)) |
| **A-med depth** | ds001787 Q1 ≤1 vs ≥2 | Holdout bal macro-F1 **≈ 0.333** | Always-low collapse ([`head_a_med_depth_smoke.md`](head_a_med_depth_smoke.md)) |
| **L-FAME** rest↔med | L-FAME BF n=16 | Holdout bal macro-F1 **≈ 0.332** | Research-only; **BY-NC DENY ship** ([`head_a_med_lfame_smoke.md`](head_a_med_lfame_smoke.md)) |
| **A-eng early** | ds007169 slice | High F1 but **block-order confound** | `ship_candidate=False` ([`head_a_eng_smoke.md`](head_a_eng_smoke.md)) |
| **Stress/calm** | eegmat | Holdout bal macro-F1 **≈ 0.495** | Near chance → personal_cal_only ([`head_stress_calm_smoke.md`](head_stress_calm_smoke.md)) |

**Takeaway:** subject-general meditation / MW / stress heads from public open sets did not clear ship bars. Prefer **personal Muse cal** for those UX surfaces.

---

## 3. Sleep expansion → ~120 subjects

- Grew `vigilance_sleep_edf` from small SC400 pilot toward **~124 unique subjects / 125 nights**: Sleep-EDF Expanded + **HMC** (CC-BY-4.0).
- Tags: `(ch_count=4, professional)` muse4 proxy.
- Stage labels persisted for **Head C** (`stage_raw` / `stage_coarse`); N1-slice recipe still 2-way wake/light only.
- Docs: [`head_c_corpus_expansion.md`](head_c_corpus_expansion.md), [`datasets/CATALOG.md`](../datasets/CATALOG.md).

---

## 4. Dual-encoder full-corpus trains (frozen)

### A-vig + Head C (CBraMod, 124-subj subject-holdout)

| Head | Test macro-F1 | ship_candidate |
|------|--------------:|:--------------:|
| **A-vig** | **0.747** | **True** (muse4-proxy vig) |
| **Head C** wake/light | 0.760 | **False** (not 4-way staging) |

Docs: [`head_a_vig_full_corpus_train.md`](head_a_vig_full_corpus_train.md), [`head_c_full_corpus_train.md`](head_c_full_corpus_train.md), [`freeze_heads_cbramod_reve_progress.md`](freeze_heads_cbramod_reve_progress.md).

### A-eng (133 unique persons, muse4)

| Encoder | Test macro-F1 | ship_candidate |
|---------|--------------:|:--------------:|
| CBraMod | ≈ **0.548** | False |
| REVE | ≈ **0.590** | False |

Cleaner source **ds007262** stays weak while order-confounded sources look strong → **MISFIT / do not ship**. Doc: [`head_a_eng_dual_encoder.md`](head_a_eng_dual_encoder.md).

---

## 5. REVE vs paper (HMC)

Frozen REVE-base + linear probe on **HMC 5-stage, 30 s**:

- Ours test **bal_acc ≈ 0.649** vs paper Table 4 Pool **0.647 ± 0.008** → **essentially matched**.
- Same qualitative conclusion: REVE frozen embeddings carry sleep under a linear head.
- Do **not** claim fine-tune Table 14 (~0.74); ship path stays frozen.

Doc: [`reve_hmc_paper_compare_gap.md`](reve_hmc_paper_compare_gap.md).

---

## 6. Freeze vs fine-tune decision

**KEEP BACKBONES FROZEN.**

- A-vig already clears subject-holdout ship bar on frozen CBraMod.
- A-eng misfit is label/confound/domain — **do not** fine-tune backbone to chase it.
- Head C needs sleep-period windows before any staging FT discussion.
- Fine-tune only if true-Muse calibration shows a clear domain gap worth LoRA/unfreeze later.

---

## 7. Current ship set (end state)

| Item | Status |
|------|--------|
| **A-vig frozen CBraMod** | **Ship path** — Sleep-EDF+HMC muse4 proxy; test macro-F1 ≈ 0.75 |
| **Head C** | **Probe** — 2-way wake/light on N1-slice; not product 4-way |
| **A-eng** | **Misfit** — ~0.55–0.59 macro-F1; confounds; do not ship |
| **A-med / MW / depth / stress** | **Personal Muse calibration** (not public frozen heads) |
| **Head B** | **Skip for now** — plan only ([`head_b_plan.md`](head_b_plan.md)) |
| **L-FAME / BY-NC** | **DENY** ship mix |
| **REVE** | Optional/experimental; HMC LP matched paper; user brings gated base |

**One-liner for bots:** Ship **A-vig** on frozen CBraMod; treat Head C as a probe; call A-eng a misfit; route med/MW to personal cal; do not start Head B until queued.

# Head A multi-head (locked)

**Decision date:** 2026-09-07 (UTC)  
**Decided by:** windwerfer  
**Status:** locked taxonomy + light code scaffold; vig ship path first.

Supersedes the single exclusive **4-way softmax** Head A as the product architecture. Legacy `HEAD_A_4WAY_LABELS` / masked-CE helpers in `src/head_a.py` remain for historical smokes and research comparisons — new work targets **separate tiny heads** on a **shared frozen encoder**.

See also: [`labeling.md`](labeling.md), [`head_a_4way_scaffold.md`](head_a_4way_scaffold.md) (legacy), [`mind_wandering_label_options.md`](mind_wandering_label_options.md), [`montages_muse_crown.md`](montages_muse_crown.md).

---

## Architecture (locked)

```
window → frozen encoder (REVE and/or CBraMod) → emb [B, D]
              ├─ Head A-vig  → logits_vig   [B, 2] or [B, 3]
              ├─ Head A-med  → logits_med   [B, 2]
              └─ Head A-eng  → logits_eng   [B, 2]   # corpus ~133 subjects; fit/misfit train next
```

| Rule | Detail |
|------|--------|
| Encoder | **Frozen** REVE-base and/or CBraMod (publish path prefers CBraMod; REVE optional/gated) |
| Heads | **Separate** Linear (or tiny MLP) modules — **not** one exclusive 5-way / 4-way softmax |
| Shared encode API | One `encode(windows) → emb`; each head `forward(emb) → logits` |
| Montages | Separate packs **`muse4`** vs **`crown8`**; **never mix** channels in one example ([`montages_muse_crown.md`](montages_muse_crown.md)) |
| MW / concentration | Public **frozen** decoder **deferred**; product path = **personal Muse calibration** later |
| Losses | Train each head on its own corpora with its own CE (no cross-head exclusive softmax) |

---

## Taxonomy & output shapes

### Head A-vig (ship path)

| Field | Value |
|-------|--------|
| **ID** | `head_a_vig` |
| **Labels (2-way default)** | `drowsy`, `hypnagogic` |
| **Optional 3-way** | `awake`, `drowsy`, `hypnagogic` — ship can stay 2-way until awake is needed |
| **Logits** | `[B, 2]` default (`n_classes=2`); optional `[B, 3]` |
| **Order** | index 0 = first label in list below |

```
HEAD_A_VIG_LABELS_2 = ["drowsy", "hypnagogic"]
HEAD_A_VIG_LABELS_3 = ["awake", "drowsy", "hypnagogic"]  # optional
```

**UX:** sleep-onset / vigilance continuum for the meditation app (already works on Sleep-EDF).

### Head A-med (beginners UX)

| Field | Value |
|-------|--------|
| **ID** | `head_a_med` |
| **Labels** | `rest`, `meditation` |
| **Logits** | `[B, 2]` |

```
HEAD_A_MED_LABELS = ["rest", "meditation"]
```

**UX:** beginners “am I in rest or meditation?” — **not** mind-wandering vs concentration.

### A-med depth operationalization (2026-09-08)

Preferred **in-session** A-med label pair: `meditation_depth_low` vs `meditation_depth_high` from **ds001787 (64-ch BioSemi, professional) — muse4 proxy channels AF7/AF8/TP9/TP10 extracted (TP9/TP10←P9/P10)** probe Q1 (≤1 vs ≥2), muse4, `HeadADepthLinear`. Smoke: holdout bal macro-F1 **0.333**, ship_candidate `False` — see [`head_a_med_depth_smoke.md`](head_a_med_depth_smoke.md). Protocol `rest`↔`meditation` (ds003816) remains a secondary / beginners block contrast, not the live depth meter.


### Head A-eng (fit/misfit evaluated)

| Field | Value |
|-------|--------|
| **ID** | `head_a_eng` |
| **Labels** | `low_engagement`, `high_engagement` (aliases: load low/high) |
| **Logits** | `[B, 2]` |
| **Train now?** | **Trained** frozen CBraMod (test macro-F1≈0.548, ship=False); REVE Kaggle pending — see [`head_a_eng_dual_encoder.md`](head_a_eng_dual_encoder.md) |

```
HEAD_A_ENG_LABELS = ["low_engagement", "high_engagement"]
```

### Explicitly out of public ship heads

| Label / head | Status |
|--------------|--------|
| `concentration` / `mind_wandering` as a public frozen decoder | **Deferred** — research track + **personal Muse calibration** only ([`mind_wandering_label_options.md`](mind_wandering_label_options.md)) |
| Single exclusive 4-way / 5-way softmax | **Not** the locked product architecture |

---

## Train / eval corpora candidates

### A-vig (ship first)

| Corpus | Local? | Role | Notes |
|--------|--------|------|-------|
| **Sleep-EDF Expanded** (`vigilance_sleep_edf`) | **Yes** | Primary train/eval | W→`drowsy`, N1→`hypnagogic`; muse4 proxy only; v0 frozen nights OK |
| Optional awake extension | No dedicated pack | Optional 3-way | Eyes-open rest / clear Wake blocks if/when needed — do not invent from N2+ |

### A-med (second)

| Corpus | Local? | Role | Notes |
|--------|--------|------|-------|
| **ds003969** med\* blocks | **Yes** (as `concentration` today) | Candidate **meditation** | Remap `med*` → `meditation`; **do not** treat `think*` as `rest` (instructed thinking ≠ rest) |
| **ds001787** high-Q1 probes | **Yes** (as attention today) | Weak **meditation** | Deep meditation ratings only; still probe-sparse |
| Rest epochs | **Partial / thin** | Candidate **rest** | Need explicit rest / eyes-open baseline blocks from open sets or personal cal — **not** Sleep-EDF W (drowsy proxy) |
| HF EEGMeditation | Not ingested (gated) | Stronger blocks if policy allows research | Not ship-pack default |
| Personal Muse cal | Later | Product gold | Rest block + breath-focus meditation block |

### A-eng (corpus ready + train)

| Corpus | Local? | Role | Notes |
|--------|--------|------|-------|
| **STEW** (IEEE / HF CC-BY-4.0) | **No** | Primary candidate | Rest vs SIMKAP multitask + perceived workload |
| **OpenNeuro ds007169** n-back | **No** | Objective load | 1–4 back difficulty → low/high bins |
| **ds007262** arithmetic | **No** | Alternate objective load | Same mobile 19-ch family |
| Local attention / vigilance packs | Yes | **Unusable for eng** | No load/engagement labels on disk |

---

## Engagement cheapness assessment (honest)

**Update 2026-09-08:** small A-eng chance-check smoke ran on ds007169 (see `docs/head_a_eng_smoke.md`) — holdout bal macro-F1 high but **block-order confound** → `ship_candidate=False`; still not shipping. STEW still deferred (IEEE login). Prior verdict below kept for history.

**Verdict (2026-09-07): eng is NOT cheap now → scaffold + deferred ingest. Do not download STEW / ds007169 in this step.**

| Check | Result |
|-------|--------|
| Local `y_head_a` vocab | Only `concentration`, `mind_wandering`, `drowsy`, `hypnagogic` (windows_index across `attention_*` + `vigilance_sleep_edf`) |
| Load / engagement / workload columns | **None** in local manifests or annotations |
| Free byproduct of vig or med? | **No** — different tasks, different corpora; vig/med labels do not encode cognitive load |
| Existing data supports eng train? | **No** |
| Cost to make cheap | Need STEW and/or ds007169 (or similar) ingest — not tiny; defer until queued |

Therefore Head A-eng = **interface scaffold** + doc note **“deferred ingest (STEW / ds007169)”** — no full train, no huge downloads.

---

## Ship order

| Order | Head | Action now | Gate |
|------:|------|------------|------|
| **1** | **A-vig** | Ship path; Sleep-EDF already works | Continue LOSO / holdout on vigilance; muse4 pack first |
| **2** | **A-med** | Train after vig stable; remap/ingest rest↔meditation carefully | Beginners UX; subject-wise splits; never mix montages |
| **3** | **A-eng** | Scaffold only until STEW/ds007169 (or equivalent) ingested | Optional product; not blocking vig/med |

Montage packs: train/export **`muse4`** and **`crown8`** heads separately (Sleep-EDF stays muse4-only).

---

## Code scaffold (light)

| Path | Role |
|------|------|
| `src/heads/__init__.py` | Package exports + shared encode notes |
| `src/heads/base.py` | Tiny Linear/MLP building blocks + `encode` contract |
| `src/heads/head_a_vig.py` | Vigilance head + label constants |
| `src/heads/head_a_med.py` | Rest vs meditation head |
| `src/heads/head_a_eng.py` | Engagement scaffold (deferred train) |
| `src/head_a.py` | Legacy 4-way + mask helpers (compat); points here |

No full training runs required for this lock step.

---

## Memory note

> **2026-09-07** — windwerfer locked Head A as **multi-head**: shared frozen encoder + separate tiny heads **A-vig** (drowsy/hypnagogic, optional awake; ship first), **A-med** (rest/meditation; second), **A-eng** (engagement/load; **deferred** — not a free byproduct, needs STEW/ds007169-style data). Public MW/concentration frozen decoder deferred; personal Muse calibration later. Montages muse4 vs crown8 never mixed. Doc: `docs/head_a_multihead.md`.


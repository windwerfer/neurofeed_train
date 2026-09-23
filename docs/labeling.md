# Label mapping & calibration

**Updated:** 2026-09-07 — Head A is **multi-head** (locked). See [`head_a_multihead.md`](head_a_multihead.md).

## Head A — multi-head (locked product)

Shared **frozen** encoder (REVE and/or CBraMod); **separate tiny heads** (not one exclusive 4-/5-way softmax). Montages: **`muse4`** vs **`crown8`** packs — never mix.

| Head | Labels | Logits | Ship |
|------|--------|--------|------|
| **A-vig** | `drowsy`, `hypnagogic` (+ optional `awake`) | `[B, 2]` or `[B, 3]` | **1st** — Sleep-EDF path works |
| **A-med** | `rest`, `meditation` | `[B, 2]` | **2nd** — beginners UX |
| **A-eng** | `low_engagement`, `high_engagement` | `[B, 2]` | **Deferred** — scaffold only until STEW / ds007169 ingest |

### Public MW / concentration

| Labels | Status |
|--------|--------|
| `concentration`, `mind_wandering` | **Research / personal Muse calibration only** — no public frozen MW decoder in the ship plan ([`mind_wandering_label_options.md`](mind_wandering_label_options.md)) |

Legacy 4-way constants (`concentration`, `mind_wandering`, `drowsy`, `hypnagogic`) remain in `src/head_a.py` / older windows for historical smokes — do not treat that exclusive softmax as the product architecture.

### Mapping rules (public → multi-head)

Document every remapping in experiment configs:

- Sleep stages W / light drowsiness → **A-vig** `drowsy`; clear N1 / sleep-onset → **A-vig** `hypnagogic`. Optional eyes-open clear wake → `awake` (3-way).
- Meditation blocks / deep-Q1 probes → **A-med** `meditation`; true rest / baseline (not instructed “think”) → **A-med** `rest`. Do **not** map Sleep-EDF W to med-rest.
- Workload / n-back difficulty / SIMKAP → **A-eng** low/high — **only after** STEW / ds007169 (or equivalent) ingest; **not** derived from vig/med labels.
- Ambiguous windows → **exclude** rather than force a label.

## Head B — Artifacts / events

| Label | Intent | How to elicit (calibration) |
|-------|--------|-----------------------------|
| `blink` | Single eye blink | Soft blink on cue |
| `double_blink` | Two blinks in short succession | Double blink on cue |
| `jaw` | Jaw clench / EMG burst | Gentle jaw clench |
| `double_jaw` | Two clench events | Double clench on cue |
| `clean` | No artifact / usable EEG | Eyes open/closed rest without movement |

## Calibration note (human-critical)

Collect a **personal Muse calibration** (~5–10 min):

1. Eyes open rest (`clean` / A-med `rest`), eyes closed rest (optional A-vig / rest).
2. Cued single & double blinks; single & double jaw (Head B).
3. Breath-focus meditation block (A-med `meditation`); optional free mind-wander block (**personal MW only** — not public frozen MW).

Align timestamps to Muse stream; window with the same `src/windowing.py` settings as training. Use this set for:

- Channel order / polarity checks (`AF7, AF8, TP9, TP10`).
- Head B few-shot fine-tune or threshold calibration.
- Subject-specific A-med / MW bias check before trusting public-only scores.

Do not mix calibration windows into the public test fold.

## Sleep-EDF pilot specifics (A-vig)

- Channels: Fpz-Cz / Pz-Oz proxied to AF7/AF8/TP9/TP10 (duplicate frontal/posterior). Not true Muse.
- Stages: W→`drowsy`, N1→`hypnagogic`; N2+ / REM / ? / Movement excluded from A-vig.
- Prefer slicing around first N1 so both labels appear without loading the full night.
- Montage pack: **`muse4` only** — do not invent Crown channels from Sleep-EDF.

## ds001787 / ds003969 (legacy attention + A-med candidates)

- After stimulus `128`: ordered keypad ratings Q1 (meditation), Q2 (mind wandering), Q3 (tiredness); values `1/2/4/8` = ratings `0/1/2/3`.
- Historical Head A attention map: **Q1 > Q2 → `concentration`**; **Q1 < Q2 → `mind_wandering`**; ties / incomplete → exclude — **research/personal only**, not public ship MW.
- For **A-med**: prefer high-Q1 / `med*` → `meditation`; do not treat `think*` or MW as `rest`.
- Details: [`ds001787_codebook_confirm.md`](ds001787_codebook_confirm.md), [`attention_corpora_status.md`](attention_corpora_status.md).

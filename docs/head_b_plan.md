# Head B plan — artifact markers

Docs-only planning step. **No training / no bulk download** in this step.

Locked labels (from thread + `src/metrics.py`):

| Label | Intent |
|-------|--------|
| `blink` | Single eye blink |
| `double_blink` | Two blinks in short succession |
| `jaw` | Jaw clench / EMG burst |
| `double_jaw` | Two clench events |
| `clean` | No artifact / usable EEG |

Channels: Muse **AF7, AF8, TP9, TP10**. Encoder: frozen **CBraMod** (default) ± LaBraM; heads only.

## Priority vs Head A

| Track | Status | Recommendation |
|-------|--------|----------------|
| Head A vigilance (Sleep-EDF) | Smoke + multi-night holdout ~0.73 macro-F1 | Keep iterating; packaging next |
| Head A attention (ds001787 / ds003969 / EEGMeditation) | Codebook confirmed; no ingest yet | Next high-value after packaging |
| **Head B** | Labels locked; plan only | **Defer training** until personal Muse calibration exists **or** a weak-label Schreer pilot is explicitly queued |
| Head C | Dataset-only stage labels | Persist only; do not train |

Head B is **shipping-useful** for flutter_muse-rs_ml (reject bad windows / cue artifacts) but **not** on the critical path for Head A publish. Prefer finishing `export_cbramod_packaging` and attention ingest before a full Head B train cycle.

## Data sources (open licenses only)

| Source | License | Muse native? | Labeled blink/jaw? | Role |
|--------|---------|--------------|--------------------|------|
| **Personal Muse calibration** (~5–10 min) | User-owned | Yes | **Yes (cued)** | **Gold** for Head B + channel/polarity sanity |
| **Schreer Muse-S** (doi:10.7910/DVN/V2CWJW) | **CC0-1.0** (verified 2026-09-07 Dataverse API) | Yes (Mind Monitor CSV) | **No** cued markers | Channel map, raw Muse realism, **heuristic weak labels** only |
| Sleep-EDF / OpenNeuro attention sets | See ALLOW table | No / partial | No | **Out of scope for Head B** (wrong events / montage) |
| LUNA / L-FAME / SEED-VIG | DENY | — | — | Never |

Schreer files are large Mind Monitor CSVs (~200–330 MB each, 18 files). Do **not** bulk-download until a dedicated ingest step is queued. `fileAccessRequest: true` but files are listed unrestricted + CC0.

### Mind Monitor columns (expected)

RAW EEG: `RAW_AF7`, `RAW_AF8`, `RAW_TP9`, `RAW_TP10` (plus aux). Optional accelerometer / gyroscope for jaw/motion proxies. Elements/markers are usually sparse meditation tags — **not** blink/jaw ground truth.

### Weak-label heuristics (Schreer / unlabeled Muse) — bootstrap only

Document in any experiment config if used:

1. **Blink:** short high-amplitude frontal (AF7/AF8) deflections; refractory; optional double-peak → `double_blink`.
2. **Jaw:** TP9/TP10 or accel burst patterns; double peak → `double_jaw`.
3. **Clean:** windows failing blink/jaw detectors + amplitude QC.
4. Ambiguous → **exclude**, never force.

Treat heuristic labels as **weak**; final ship path prefers personal cued calibration ± few-shot.

## Architecture

Mirror Head A:

- `src/head_b.py`: `HeadBLinear` / `HeadBMLP` on frozen encoder embeddings; `HEAD_B_LABELS` already in `src/metrics.py`.
- Same window length / rate as Head A exports (document rate, e.g. 256 Hz).
- Event-centric windows: center on cue or detected peak; short overlap OK for doubles.
- Metrics: per-class precision/recall/F1 + confusion; watch `clean` majority bias.
- Separate head per encoder family (CBraMod vs LaBraM vs REVE); do not share Head B across encoders.

## Phased execution (when queued)

1. **`head_b_scaffold`** — `src/head_b.py` + unit smoke with synthetic / tiny embeddings (no dataset).
2. **`schreer_license_cache_note`** — already done here (CC0); optional private Kaggle cache of **one** CSV slice for channel-map smoke (pace CLI; private only).
3. **`head_b_weak_label_pilot`** — heuristic labels on a small Schreer slice → frozen CBraMod Head B smoke; mark weak in manifests.
4. **`personal_muse_calib`** — **user action**: 5–10 min cued session (see `docs/labeling.md`); ingest → few-shot / threshold cal.
5. **`head_b_export`** — head-only `.pt` + checksum + label map for flutter_muse-rs_ml alongside Head A pack.

Do **not** start steps 3–5 until user queues them or personal cal lands. Continuum default after this doc: **`export_cbramod_packaging`**.

## Shipping / provenance

- Training mix ⊆ ALLOW; zero LUNA / L-FAME / SEED-VIG / BY-NC.
- Publish **head-only** + attribution (datasets, encoder pin, Muse channel order, label list).
- Provenance manifest per export: source URIs, SPDX, window params, weak vs cued flag, subject split.
- Personal cal windows: never mix into public test fold; optional private Kaggle only.

## User decisions (not blockers for continuum)

1. Collect personal Muse calibration soon, or defer Head B train until after attention bootstrap?
2. Allow a Schreer weak-label pilot (downloads ~GBs) before personal cal?
3. In-app: Head B as hard gate (drop windows) vs soft score for UI?

Continuum proceeds to packaging without answers.

## Artifacts

- This doc: `docs/head_b_plan.md`
- Summary: `exports/head_b_plan/manifest.json`
- Related: `docs/labeling.md`, `docs/datasets.md`, `docs/LICENSE_NOTES.md`

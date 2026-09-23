# vigilance_sleep_edf

Sleep-stage / vigilance corpus for **Head A vig** (drowsy / hypnagogic) and **Head C** (`stage_raw` / `stage_coarse`).

Expanded **2026-09-08** to **124 unique subjects** / 125 recordings (target ~120).

## Sources

| Source | License | Subjects | Montage |
|--------|---------|----------|---------|
| Sleep-EDF Expanded cassette + telemetry | ODC-By | 100 | `(ch_count=4, professional)` muse4 proxy Fpz-Cz/Pz-Oz |
| HMC sleep staging v1.1 | CC-BY-4.0 | 24 | `(ch_count=4, professional)` muse4 proxy F4/C4/C3/O2 |

## Head A (unchanged mapping)

| Hypnogram | Head A |
|-----------|--------|
| Sleep stage W | `drowsy` |
| Sleep stage 1 / N1 | `hypnagogic` |
| N2 / N3 / N4 / R / ? / Movement | excluded from Head A |

## Head C (required — do not drop)

| Field | Values |
|-------|--------|
| `stage_raw` | Hypnogram text |
| `stage_coarse` | `wake` / `light` / `deep` / `rem` / `unknown` |

## Splits (policy A)

See `splits/`. Anchor: `SC400` → **test**, `SC403` → **val**; remaining ~70/15/15 subject-wise. Validated no leakage.

Counts: train=86 val=19 test=19 subjects.

## Paths

- `raw/` — PSG + hypnogram / scoring (symlinks)
- `windows/` — `*_windows.npz` + manifests
- `annotations/` — index/qc (may need refresh after expansion)
- Docs: `docs/head_c_corpus_expansion.md`, `docs/head_c_labels_policy.md`

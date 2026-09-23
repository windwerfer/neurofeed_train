# Windows NPZ + annotation schema

## Required NPZ keys (all corpora)

| Key | Type / shape | Notes |
|-----|----------------|-------|
| `X` | `(N, 4, 512)` float32 | Muse order AF7, AF8, TP9, TP10 @ 256 Hz, 2 s |
| `y` | `(N,)` int64 | Index into `label_names` (Head A labels present in file) |
| `starts` | `(N,)` int64 | Window start sample within the exported slice/recording |
| `label_names` | `(C,)` str | Head A class names for this file |

Preferred alias for trainers: treat `y` as **`y_head_a`** (integer index). Annotation CSV uses the string form in column `y_head_a`.

## Required for Sleep-EDF / vigilance (Head C — do not drop)

| Key | Type / shape | Notes |
|-----|----------------|-------|
| `stage_raw` | `(N,)` object/str | Hypnogram text, e.g. `Sleep stage W`, `Sleep stage 1`, … |
| `stage_coarse` | `(N,)` object/str | `wake` \| `light` \| `deep` \| `rem` \| `unknown` |

**Head A mapping (unchanged):** only `Sleep stage W` → `drowsy`, `Sleep stage 1` → `hypnagogic`. Other stages are excluded from Head A training windows in current n1-slice exports, but when present in fuller exports they still carry Head C fields.

See `label_maps.json` for the full `stage_raw` → `stage_coarse` map and Head A map.

## Strongly recommended metadata (manifest / index)

| Field | Notes |
|-------|--------|
| `subject_id` | Stable subject key (Sleep-EDF: `SC4sss`; OpenNeuro: `sub-XXX`) |
| `recording_id` | Night/session id (`SC4001`, `sub001_ses01`, `sub001`, …) |
| `slice_start_sec` | Absolute start of exported slice in recording time |
| `window_start_sample` / `window_start_sec` | Per-window onset |
| `split` | `train` \| `val` \| `test` from fixed subject JSON (policy A) |
| `npz_sha256` | Content hash of the npz on disk |
| `provenance` | DOI or source id (e.g. openneuro-ds001787) |
| `license_spdx` | SPDX license id (e.g. CC0-1.0) |

## Attention corpora extras

| Key | Notes |
|-----|--------|
| `channels` | `(4,)` str — usually AF7/AF8/TP9/TP10 |
| `probe_ids` | ds001787 probe grouping when present |

Attention windows do **not** require `stage_raw` / `stage_coarse` (leave empty in CSV).

## Annotation artifacts per corpus

- `annotations/windows_index.csv` — one row per window; Sleep-EDF rows **must** include `stage_raw` and `stage_coarse`
- `annotations/recording_manifest.json` — per-recording counts + paths + checksums
- `annotations/checksums.json` — SHA256 of raw + windows binaries

## Split policy

Fixed subject JSON under `splits/` (option A). Validate with `scripts/dataset/validate_splits.py` (no subject leakage).

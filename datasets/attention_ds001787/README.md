# attention_ds001787

OpenNeuro **ds001787** attention probe windows (`concentration` / `mind_wandering`) for Head A.

## Labels

- Q1 > Q2 → `concentration`; Q1 < Q2 → `mind_wandering`; ties/incomplete dropped.
- Do **not** map events value 2/4 directly to classes.
- Channels: AF7/AF8 + P9/P10→TP9/TP10 (BioSemi64 Muse-proxy). Not native Muse geometry.

## Splits (policy A)

- **train:** sub-001,002,003,004,005,014,017,020
- **val:** sub-006,015 (both classes — prior val was concentration-only)
- **test:** sub-019,013 (prior near-chance holdouts; frozen for honesty)

See `docs/attention_corpora_status.md`. `ship_candidate: false` until holdout improves.

## Provenance

- DOI `doi:10.18112/openneuro.ds001787.v1.1.1` — SPDX **CC0-1.0**
- Annotations: `annotations/windows_index.csv` (+ provenance / license_spdx columns)

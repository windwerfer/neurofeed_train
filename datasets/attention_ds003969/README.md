# attention_ds003969

OpenNeuro **ds003969** meditation vs thinking block windows for Head A.

## Labels

- `med*` → `concentration`; `think*` → `mind_wandering` (protocol proxy).
- Channels: AF7/AF8 native; TP7/TP8 → TP9/TP10 proxies. Not native Muse geometry.

## Splits (policy A)

- **train:** sub-001…006, sub-029
- **val:** sub-026,028 (balanced 400/400 each)
- **test:** sub-025,027 (prior near-chance holdouts; frozen)

See `docs/attention_corpora_status.md`. `ship_candidate: false` until holdout improves.

## Provenance

- DOI `doi:10.18112/openneuro.ds003969.v1.0.0` — SPDX **CC0-1.0**

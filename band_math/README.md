# band_math — classical EEG features vs frozen AI heads

Peer-owned. Flutter-matched band power / ratios on the same muse4 windows and splits as Diggus’s frozen heads.

## Quick start

```bash
cd /workspace/muse-eeg-heads/band_math
python3 -m venv .venv && .venv/bin/pip install numpy scikit-learn
.venv/bin/python -u scripts/run_compare.py
```

Outputs: `COMPARE_AI_VS_BAND.md`, `compare_summary.json`, `results/`.

## Feature contract

See `features/flutter_bands.py` and Flutter:

- `rust/src/api/muse.rs` — `compute_fft_bands` (δ1–4 θ4–8 α8–13 β13–30 γ30–50)
- `rust/src/api/features.rs` — `aggregate_band_feature` (`band.atr|tar|btr|alpha|delta` on AF7/AF8)

Do not modify AI head code or overwrite `exports/` AI artifacts.

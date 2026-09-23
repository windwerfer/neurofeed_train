# vigilance_hmc_crown2

Head A-vig (drowsy vs hypnagogic) N1-slice windows from PhysioNet HMC v1.1.

## Montage: `crown2_strong`
- Channels (2): C3, C4
- HMC source labels: EEG C3-M2, EEG C4-M1
- Note: True central pair from HMC PSG; no fabricated channels.
- **True C=2 tensors** — no zero-pad to Crown8.

## Window recipe
- 2.0 s @ 256 Hz, hop 0.5 s
- Slice: first N1 ± (20 min pre / 40 min post)
- Majority stage frac ≥ 0.7; keep only drowsy (W) / hypnagogic (N1)

## Splits
Subject-wise ~70/15/15 (seed=42), shared with sibling crown pack.
- train: 106 subjects
- val: 23 subjects
- test: 22 subjects
- windows built for: 151 subjects/nights

## License
See ATTRIBUTION.md (CC-BY-4.0). Public windows: HF `windwerfer/neurofeed-eeg-windows` config `crown2_vigilance_hmc`.

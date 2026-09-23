# vigilance_hmc_crown4

Head A-vig (drowsy vs hypnagogic) N1-slice windows from PhysioNet HMC v1.1.

## Montage: `crown4_hmc`
- Channels (4): C3, C4, F6, PO4
- HMC source labels: EEG C3-M2, EEG C4-M1, EEG F4-M1, EEG O2-M1
- Note: C3/C4 exact; F6≈EEG F4-M1 (nearest frontal-right in HMC 4-EEG set); PO4≈EEG O2-M1 (nearest posterior-right). Honest approximation, not Muse Crown8.
- **True C=4 tensors** — no zero-pad to Crown8.

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
See ATTRIBUTION.md (CC-BY-4.0). Public windows: HF `windwerfer/neurofeed-eeg-windows` config `crown4_vigilance_hmc`.

# Head C labels policy (dataset only for now)

**Approved 2026-09-07:** persist sleep-stage labels while building Sleep-EDF windows. **Do not prioritize training Head C.**

## Head A (unchanged)
- W → `drowsy`
- N1 → `hypnagogic`
- N2 / N3 / N4 / R / ? / Movement → excluded from Head A

## Head C (labels only)
Per window, store:
- `stage_raw` — hypnogram text (e.g. `Sleep stage 2`)
- `stage_coarse` — `wake` | `light` | `deep` | `rem` | `unknown`
  - W → wake; N1/N2 → light; N3/N4 → deep; R → rem

When regenerating or adding Sleep-EDF nights, include these fields in npz + manifest provenance. Optional later: train Head C on frozen CBraMod (Muse-proxy transfer expected weak).

Corpora: Sleep-EDF Expanded + HMC (CC-BY-4.0) after 2026-09-08 expansion; see `docs/head_c_corpus_expansion.md`.

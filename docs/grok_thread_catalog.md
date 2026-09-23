# Grok EEG thread — dataset decision catalog

Clean summary of dataset include/exclude decisions from the Muse EEG heads planning thread.

## Decision principles

1. **Shipping license first** — prefer **CC0 / CC BY / ODC-By / MIT / BSD**. If redistribution is unclear, BY-NC, or academic-only → **DENY** for shipping.
2. **NO LUNA** — LUNA is explicitly excluded from the shipping training mix and published artifacts.
3. **Muse geometry preferred** — native AF7/AF8/TP9/TP10 (or Muse-S equivalents) beat proxy montages for Head B and channel_map validation.
4. **Vigilance transfer allowed** — sleep corpora (Sleep-EDF, optional HMC) may support drowsy/hypnagogic Head A labels even without Muse electrodes, with explicit remap caveats and open-terms checks.
5. **Encoders frozen; heads published** — **CBraMod** default publish path (± **LaBraM**); **REVE-base** optional/experimental (user fetches gated base). Only heads train/ship.

## INCLUDE (USE)

| Dataset | Role in thread | Rationale |
|---------|----------------|-----------|
| OpenNeuro **ds003969** | Primary open BIDS cognitive/attention EEG | Open terms; task structure maps toward concentration / mind_wandering |
| OpenNeuro **ds001787** | Secondary open BIDS | Complements ds003969; same open pipeline |
| HF **alexeykashevnik/EEGMeditation** | Meditation / attention prior | Convenient HF access; aligns with concentration vs wandering narratives |
| **Sleep-EDF Expanded** | Vigilance / drowsiness / hypnagogia | Standard PhysioNet sleep staging → drowsy & hypnagogic proxies |
| **HMC** (optional) | Extra sleep staging | Optional only; confirm license before any shipping use |
| Harvard Dataverse **Schreer Muse-S** (doi:10.7910/DVN/V2CWJW) | Native Muse-S reference | Best-in-catalog for Muse channels + artifact realism (Head B) |

## DENY from shipping

| Dataset | Thread decision | Reason |
|---------|-----------------|--------|
| **LUNA** | **DENY** | Explicit project exclusion — not in mix or published heads |
| **L-FAME** | **DENY** | BY-NC — incompatible with shipping redistribution goals |
| **SEED-VIG** | **DENY** | Academic / restricted access — do not ship derived artifacts |

Research-only local experiments on DENY sets are out of scope for this scaffold’s documented pipeline. See `docs/LICENSE_NOTES.md`.

## Label lock (thread outcome)

- **Head A:** concentration, mind_wandering, drowsy, hypnagogic (Attention + Vigilance multitask pairs).
- **Head B:** blink, double_blink, jaw, double_jaw, clean.
- **Encoders:** CBraMod (± LaBraM) frozen — **default publish path**; REVE-base frozen optional/experimental; **heads only** published.
- **Channels:** Muse AF7, AF8, TP9, TP10.
- **Licenses:** open only; NO LUNA / L-FAME / SEED-VIG in shipping mix.

## Follow-ups called out in thread

- Human HF secret on Kaggle before EEGMeditation pulls.
- Personal Muse calibration for Head B and channel sanity.
- Subject-wise splits; DENY LUNA / L-FAME / SEED-VIG in shipping exports.
- Publish head-only weights with attribution (CBraMod default).

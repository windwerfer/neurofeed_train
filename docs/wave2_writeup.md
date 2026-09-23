# Wave 2 writeup — Muse EEG Head A continuum

Finished 2026-09-07 (Asia/Bangkok). Frozen CBraMod + Head A on open Sleep-EDF only; attention datasets licensed in docs (no bulk dumps yet).

## What wave 2 did

1. **Hypnagogic precision tune** — raise decision threshold on frozen binary head (SC4001 val only). SC4002 all-labeled hypnagogic precision **0.42 → 0.55**; recall **0.69 → 0.54**; macro-F1 **0.74 → 0.76**.
2. **More Sleep-EDF nights** — added cassette subjects **SC4011** + **SC4031** (not same-subject night-2) into private `muse-eeg-heads-cache` / `muse-eeg-heads-windows` with provenance.
3. **Multi-night holdout** — train SC4001+SC4011 (equal-night balance 298/class), hold out SC4002 + SC4031.
4. **Attention allow-probe** — ds001787 / ds003969 = CC0 allow; EEGMeditation = CC-BY-4.0 gated allow. Bootstrap order: ds001787 → ds003969 → EEGMeditation.
5. **4-way Head A scaffold** — `HeadALinear` + masked CE; concentration / mind_wandering masked; smoke on drowsy/hypnagogic only.

## Headline metrics (trust these)

| Setup | View | macro-F1 | notes |
|-------|------|----------|-------|
| Wave 1 single-night | SC4001→SC4002 all / non-overlap | ~**0.74** | honest night baseline |
| Precision-tuned binary | SC4002 all (thresh 0.53) | ~**0.76** | better hypnagogic P, lower R |
| Multi-night train | SC4002 stride4 all | ~**0.73** | did **not** beat single-night |
| Multi-night train | SC4002 balanced greedy | ~**0.84** | tiny n=90 |
| Multi-night train | SC4031 stride4 all | ~**0.73** | secondary subject holdout |
| Multi-night train | SC4031 balanced greedy | ~**0.79** | n=390 |
| 4-way masked smoke | SC4002 stride4 present-classes | ~**0.79** | attention logits untrained (grad≈0) |

**Takeaway:** subject transfer works in the ballpark of single-night (~0.73–0.74 all-labeled), but multi-subject train did not clearly improve it. Hypnagogic precision on imbalanced full nights remains the weak spot. Same-night / balanced cuts still look ~0.83–0.85 — treat as optimistic.

## Artifacts

- Precision: `docs/hypnagogic_precision_tune.md`, `exports/head_a_holdout/precision_tune_metrics.json`
- Nights: `docs/more_sleep_edf_nights.md`, `exports/windows_sc4011/`, `exports/windows_sc4031/`
- Multi-night: `docs/multi_night_holdout.md`, `exports/head_a_multi_night/`
- Attention licenses: `docs/attention_allow_probe.md`, `docs/LICENSE_NOTES.md`
- 4-way: `docs/head_a_4way_scaffold.md`, `exports/head_a_4way_scaffold/`, `src/head_a.py`

Private Kaggle only: `windwerfer/muse-eeg-heads-cache`, `windwerfer/muse-eeg-heads-windows`, `windwerfer/muse-eeg-heads-src`.

## Open questions (need user later, not blockers yet)

1. **Accept HF gate for EEGMeditation** before any download?
2. **ds001787 codebook** — confirm probe events 128/2/4 → concentration / mind_wandering mapping before bootstrap (next plan step is docs-only).
3. **Ship path priority** — keep iterating Sleep-EDF vigilance heads, or jump to attention bootstrap once codebook is confirmed?
4. **Head B artifact markers** — when to start (blink/jaw/clean) vs more nights first?
5. **CBraMod export packaging** — pin encoder blob checksum + Head A for flutter_muse-rs_ml when ready?

## Suggested next plan (already queued)

1. `ds001787_codebook_confirm` — docs only, no bulk dump  
2. `more_sleep_edf_nights_round2`  
3. `head_b_plan_doc`  
4. `export_cbramod_packaging`

No user decision required to continue those.

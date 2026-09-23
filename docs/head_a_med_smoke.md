# Head A-med smoke (CBraMod, CPU)

**Completed (Asia/Bangkok):** 2026-09-07 20:40:55 ICT
**UTC:** 2026-09-07T13:40:55.358460+00:00
**Version:** `n12_double_subjects_2026-09-07` — n=12 double-subjects (n=6 kept in exports/head_a_med_smoke/)

## Goal

Frozen **CBraMod** + **HeadAMed** (`rest` vs `meditation`) local CPU smoke, muse4 only, subject holdout.

## Data strategy

1. Searched local `ds001787` / `ds003969` / cache for honest rest↔meditation — **no usable rest**: ds003969 only has `med*` + `think*` (think ≠ rest); ds001787 is probe-rated meditation task without baseline rest blocks; Sleep-EDF W is A-vig drowsy proxy — **not** used.
2. Downloaded a **small** open slice: OpenNeuro **ds003816** (CC0) — Loving-Kindness Meditation study with explicit **PreResting** and **LKMSelf** blocks; native AF7/AF8/TP9/TP10.
3. **n=6 (prior):** six lightest `st` subjects with both tasks ≥5 MB (~162 MB).
4. **n=12 (this run):** same six + next six lightest unused `st` (`sub-16st, sub-24st, sub-36st, sub-02st, sub-05st, sub-31st`; ~406 MB new).

## Label mapping (honest)

| Source task | A-med label | Notes |
|-------------|-------------|-------|
| `PreResting` | `rest` | Eyes-closed pre-session resting state (protocol) |
| `LKMSelf` | `meditation` | Radiating LKM to Self (eyes closed) |
| ds003969 `think*` | **excluded** | Instructed thinking ≠ rest |
| Sleep-EDF W | **excluded** | A-vig drowsy proxy, not med-rest |
| HF EEGMeditation | **not used** | Gated; prefer ungated CC0 |

## Split

- Train: `sub-09st, sub-22st, sub-48st, sub-26st, sub-19st, sub-16st, sub-24st, sub-36st, sub-02st, sub-05st, sub-31st` (n=11)
- Holdout: `sub-23st` (same as n=6 for fair compare)
- Per-subject undersample_balanced then concat; internal 15% val for early pick.

## Metrics (headline) — n=12

| Split | n | accuracy | macro-F1 |
|-------|--:|---------:|---------:|
| Train (balanced) | 446 | 0.637 | 0.607 |
| Val | 67 | 0.687 | 0.640 |
| Holdout full | 34 | 0.559 | 0.358 |
| Holdout balanced | 30 | 0.500 | 0.333 |
| Holdout stride×4 | 9 | 0.556 | 0.357 |

**ship_candidate:** `False`

## n=6 vs n=12 comparison

| Metric | n=6 | n=12 | Δ |
|--------|----:|-----:|--:|
| Val macro-F1 | 0.364 | 0.640 | +0.276 |
| Holdout full macro-F1 | 0.358 | 0.358 | +0.000 |
| Holdout balanced macro-F1 | 0.333 | 0.333 | +0.000 |
| Holdout stride×4 macro-F1 | 0.357 | 0.357 | +0.000 |
| Train subjects | 5 | 11 | +6 |
| Holdout subject | `sub-23st` | `sub-23st` | same |

**Meaningful improvement vs n=6?** `False` (criterion: holdout balanced macro-F1 lift ≥ +0.05 and ≥ 0.45 absolute).

## Provenance

- Dataset: OpenNeuro `ds003816` — SPDX `CC0-1.0` — `doi:10.18112/openneuro.ds003816.v1.0.1`
- Encoder: CBraMod Apache-2.0 pretrained_weights (local cache)
- Montage: **muse4** only (never mixed with crown8)
- Script: `scripts/head_a_med_smoke.py`
- Exports n=6: `exports/head_a_med_smoke/`
- Exports n=12: `exports/head_a_med_smoke_n12/`
- Alt pairs doc: `docs/head_a_med_alt_label_pairs.md`

## Caveats

- PreResting is protocol resting state (not personal Muse cal); LKM ≠ breath-focus meditation used in app UX — proxy only.
- Still modest n subjects; overlapping windows; domain gap CBraMod TUEG → 4-ch Muse proxy.
- No Kaggle push; REVE not trained for this smoke.

## Takeaway

A-med n=12 smoke on ds003816 muse4: val F1=0.640; holdout sub-23st full/bal/stride4 F1=0.358/0.333/0.357 (n6 bal was 0.333; improved=False). No meaningful lift vs n=6 — rest↔med still near chance; see docs/head_a_med_alt_label_pairs.md for next label pair.

## Alternate label pairs (Task 2)

No meaningful holdout lift at n=12 (holdout predicts all `rest`). Ranked alternatives and recommendation: [`docs/head_a_med_alt_label_pairs.md`](head_a_med_alt_label_pairs.md).

**Top recommendation:** meditation depth high vs low on **ds001787** (CC0, muse4, probe Q1) — better live Muse UX than rest↔med.

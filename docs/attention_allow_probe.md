# Attention ALLOW probe (Head A concentration / mind_wandering)

**Step:** `attention_allow_probe`  
**Probed (UTC):** 2026-09-06 — metadata only; **no bulk EEG dumps**.  
**Goal:** confirm open licenses + Muse-proxy fit for shipping Head A attention classes.

## Decision summary

| Dataset | Shipping decision | License (verified) | Attention label quality | Muse AF7/AF8/TP9/TP10 fit |
|---------|-------------------|--------------------|-------------------------|---------------------------|
| OpenNeuro **ds003969** | **ALLOW** | **CC0** (`dataset_description.json` + GraphQL + NEMAR) | Block-level: meditation → `concentration`; thinking → `mind_wandering` (protocol proxy) | **AF7/AF8 yes**; no TP9/TP10 — use **TP7/TP8** (or T7/T8) as temporal proxies |
| OpenNeuro **ds001787** | **ALLOW** | **CC0** (same sources) | **Best open probe labels**: ~2 min interruptions for concentration vs mind-wandering ratings | BioSemi 64 (A1–B32 in BDF); map via standard BioSemi→10–10 then Muse subset — **no `channels.tsv`** |
| HF **alexeykashevnik/EEGMeditation** | **ALLOW** | **CC BY 4.0** (dataset card / README `license: cc-by-4.0`) | **Strongest task alignment**: explicit mind-wandering + internal/external concentration blocks | 64-ch 10–10 (ANT Neuro); expect AF7/AF8; confirm TP9/TP10 after gate access |

All three clear the project shipping bar (CC0 / CC BY; no BY-NC / academic-only / LUNA). Prefer **derived windows + provenance** in private Kaggle cache — never republish raw corpora wholesale.

## ds003969 — Meditation vs thinking task

- **DOI:** `doi:10.18112/openneuro.ds003969.v1.0.0` (snapshot `1.0.0`)
- **Authors:** Arnaud Delorme, Claire Braboszcz
- **Paper:** https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0170647
- **Scale:** 98 subjects; ~54.5 GB raw — **subset subjects/tasks only** if ingested
- **Tasks (BIDS):** `med1breath`, `med2`, `think1`, `think2` (NEMAR)
- **Channels (sample `sub-001`):** includes `AF7`, `AF8`, `TP7`, `TP8`, `T7`, `T8`, `P7`, `P8`, … (79 rows incl. EXG/misc). **Missing:** `TP9`, `TP10`.
- **Proposed Muse proxy:** `AF7, AF8, TP7→TP9-proxy, TP8→TP10-proxy` (document as proxy montage; not native Muse).
- **Label map (draft):** meditation blocks → `concentration`; thinking (“think actively”) blocks → `mind_wandering`. Weaker than probe ratings — fine for bootstrap with missing-class mask.
- **Access (small metadata):** OpenNeuro S3 `dataset_description.json`, GraphQL `latestSnapshot`, NEMAR `on003969`.

## ds001787 — EEG meditation study (probe ratings)

- **DOI:** `doi:10.18112/openneuro.ds001787.v1.1.1` (snapshot `1.1.1`)
- **Authors:** Arnaud Delorme, Tracy Brandmeyer
- **Ref:** https://www.ncbi.nlm.nih.gov/pubmed/27815577
- **Scale:** 24 subjects × up to 3 sessions; ~5.7 GB — **preferred first attention ingest**
- **Recording:** BioSemi ActiveTwo, **256 Hz**, 64 EEG + 15 misc; continuous BDF (~45 min/session sample)
- **Events (sample `sub-001/ses-01`):** `stimulus` value `128` (n=28); `response` values `2` (n=41) and `4` (n=18). README: interrupted ~every 2 minutes for concentration / mind-wandering level.
- **Label map (confirmed — see `ds001787_codebook_confirm.md`):** do **not** map raw values `2`/`4` to classes. Values are ratings (`1/2/4/8` ↔ keys `0/1/2/3`); after stimulus `128`, ordered responses are Q1 med / Q2 MW / Q3 tired. Label with **Q1>Q2→concentration**, **Q1<Q2→mind_wandering**, ties drop. Window **~10 s pre-Q1** (paper), not the inter-probe gap.
- **Muse path:** load BDF with montage map → pick AF7/AF8 + closest temporal-parietal (TP9/TP10 or TP7/TP8).
- **Access:** events/json are small; **do not** curl full `.bdf` in probe scripts.

## HF alexeykashevnik/EEGMeditation

- **License:** **CC BY 4.0** (ALLOW)
- **Access:** **gated** (`gated: auto` on HF API) — user must accept contact-share conditions once before CLI/download
- **Scale:** 33 participants; card reports very large total size (~hundreds of GB with video) — ingest **EEG-only, few subjects**, no video
- **Protocol fit:** Task 2 mind-wandering → `mind_wandering`; Task 3 internal concentration (+ optional Task 4 external) → `concentration`; rest blocks exclude or separate
- **Sampling:** 2048 Hz BrainVision → resample to project rate (e.g. 256 Hz) before windowing
- **Attribution:** required under CC BY (authors + dataset card citation in `ATTRIBUTION.md` when used)

## Recommended bootstrap order (no dumps this step)

1. **ds001787** — smallest open set with subjective attention probes; confirm response codebook; cache 2–4 subjects’ Muse-proxy windows privately.
2. **ds003969** — add a few subjects’ med/think blocks for volume; same Muse-proxy recipe.
3. **EEGMeditation** — after HF gate accept; strong block labels; keep subset tiny.

Pair with existing Sleep-EDF vigilance (`drowsy` / `hypnagogic`) under a **4-way Head A + missing-class mask** so attention-only batches do not train the vigilance logits (and vice versa).

## Provenance checklist (when ingesting later)

- [ ] Record SPDX / deed URL + probe date in experiment config
- [ ] Subject-wise splits; never leak holdout subjects across attention corpora
- [ ] Channel remap table + sampling-rate transform in window manifest
- [ ] Cite DOIs / HF card in published head attribution
- [ ] Private Kaggle version bumps only (`muse-eeg-heads-cache` / `windows`) — never `datasets create -u`

## Artifacts this step

- This doc: `docs/attention_allow_probe.md`
- Confirmed metadata copies used for the probe live under `/tmp/attn_probe/` on the box (ephemeral); source of truth remains OpenNeuro/NEMAR/HF cards

## Out of scope here

- No OpenNeuro/HF bulk download
- No Kaggle dataset version bump
- No Head A training — deferred to `head_a_4way_scaffold`

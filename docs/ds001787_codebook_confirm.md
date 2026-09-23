# ds001787 codebook confirm (Head A attention labels)

**Step:** `ds001787_codebook_confirm`  
**Confirmed (UTC):** 2026-09-07 — metadata + paper + behavioral logs only; **no BDF / bulk EEG download**.  
**Dataset:** OpenNeuro `ds001787` v1.1.1 (CC0) — Brandmeyer & Delorme meditation probe study.

## Verdict

**Confirmed for bootstrap.** Use **ordered Q1 vs Q2 ratings**, not raw `events.tsv` value→class shortcuts.

### Hard reject (wrong draft map)

| Naive guess | Reality |
|-------------|---------|
| `value=2` → concentration, `value=4` → mind_wandering | **False.** `2` / `4` / `8` (and `1`) are **keypad rating codes**, not class labels. Which question they answer is determined by **order after stimulus `128`**. |

## Event value codebook (BIDS + presentation script)

From `task-meditation_events.json` + `code/run_mw_experiment6.m` (parallel-port `Line(keynum+1)`):

| `value` | Meaning |
|---------|---------|
| `128` | **Q1 onset** — first probe question starts (most important marker) |
| `1` | Rating **0** (key `0`) — used in behavioral logs; underdocumented in BIDS Levels |
| `2` | Rating **1** (key `1`) — BIDS “Response 1” |
| `4` | Rating **2** (key `2`) — BIDS “Response 2” |
| `8` | Rating **3** (key `3`) — BIDS “Response 3” |
| `16` | Involuntary / cancel-style button press |

Formula: rating \(k \in \{0,1,2,3\}\) → EEG trigger \(2^{k}\).

`trial_type`: `stimulus` = onset of Q1; `response` = answer to Q1, Q2, or Q3 (order after the preceding `128`).

### Probe question order (paper + script)

After each `128` / “MW question asked”:

1. **Q1** — depth of **meditation** (0 = not meditating … 3 = deep) → Head A proxy target **concentration** when dominant  
2. **Q2** — depth of **mind wandering** (0 … 3 immersed) → **mind_wandering** when dominant  
3. **Q3** — **tiredness / drowsiness** (0 … 3) — optional aux; not a Head A attention class by itself  

Script status machine: probe → status 1 (await Q1) → 2 (await Q2) → 3 (await Q3) → 0 (resume).

## Head A label rule (paper analysis)

Brandmeyer & Delorme (*Exp Brain Res*): split trials by comparing the two scales on the **same pre-probe interval**:

- **Q1 > Q2** → meditation trial → label **`concentration`**
- **Q1 < Q2** → mind-wandering trial → label **`mind_wandering`**
- **Q1 == Q2** → **drop** (ties ignored in their EEG contrast)
- Incomplete probes (missing Q1 or Q2) → drop

**Epoch:** about **10 s immediately before Q1 onset** (paper: −10.05 s to −0.05 s relative to first question). Prefer that over post-probe EEG (ratings/speech contaminate).

Optional later: subject-normalized Q1−Q2 difference (paper notes absolute rating bias across people). Keep binary Q1 vs Q2 for first ingest.

## Prefer behavioral logs over BDF events when they disagree

`code/MW_Current_TextFileBIDS.zip` README: *some BDF files miss events (connection glitch); text logs are more reliable for answers.*

Ingest path:

1. Parse `*_info.txt` lines: `MW question asked` + following `Key … status 1/2/3`  
2. Align probe times to EEG via nearest `value=128` (or log time if synchronized)  
3. Fall back to ordered responses in `*_events.tsv` only when the log is missing for that session  

## Behavioral-log split sanity (all 41 info files in the zip)

Using the Q1 vs Q2 rule above (no EEG):

| Label | Count |
|-------|------:|
| `concentration` | 472 |
| `mind_wandering` | 291 |
| tie drop | 212 |
| incomplete | 21 |

Enough imbalance-aware sampling for a small subject-wise attention bootstrap; still pair with 4-way missing-class mask vs Sleep-EDF vigilance.

## Muse / channel note (still open for ingest)

- Recording: BioSemi ActiveTwo, **64 EEG @ 256 Hz** (downsampled from 2048 Hz).  
- No `channels.tsv` in the public tree checked this step — use standard BioSemi→10–10 names when loading BDF, then take **AF7/AF8** plus closest temporal-parietal (**TP9/TP10** or **TP7/TP8** proxies). Document remap in the window manifest.  
- Do **not** claim native Muse geometry.

## Recommended next ingest (not this step)

1. Cache **2–4 subjects** (mix expert/novice from `participants.tsv`) Muse-proxy windows privately.  
2. Subject-wise split; provenance SPDX CC0 + DOI `doi:10.18112/openneuro.ds001787.v1.1.1`.  
3. Train attention logits only under Head A 4-way + mask (vigilance classes masked).

## Artifacts

- This doc: `docs/ds001787_codebook_confirm.md`  
- `exports/ds001787_codebook_confirm/` — `task-meditation_events.json`, `manifest.json`, `label_split_summary.json`, presentation script copy, code README  

## Out of scope here

- No OpenNeuro BDF download / no Kaggle version bump  
- No Head A retraining — deferred until windows exist  

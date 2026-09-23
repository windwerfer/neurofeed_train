# Head A 4-way joint train (vigilance + attention)

> **2026-09-07 architecture lock:** product Head A is **multi-head** ([`head_a_multihead.md`](head_a_multihead.md)). Public MW/concentration frozen decoder remains deferred / personal-cal only.


**Step:** `head_a_4way_attention_train`  
**Goal:** first joint train of the 4-way Head A on Sleep-EDF vigilance **and** ds001787 attention, with night/subject holdouts.

## Design

| Piece | Choice |
|-------|--------|
| Head | `HeadALinear` → 4 logits |
| Schedule | Alternating mini-batches: vigilance batches use present `{drowsy, hypnagogic}`; attention batches use `{concentration, mind_wandering}` via `masked_cross_entropy` |
| Encoder | Frozen CBraMod (CPU) |
| Vigilance train | SC4001 / SC4011 / SC4021 / SC4041, equal-night 298/class |
| Vigilance holdout | SC4002 + SC4031 stride×4 |
| Attention train | sub001 + sub002 (expert, both classes), per-subject undersample |
| Attention holdout | sub013 (novice, both classes) |
| Attention probe | sub017 (all concentration) — excluded from train |

## Headline metrics (CPU)

| Split | macro-F1 (present) |
|-------|-------------------|
| Val vigilance | ~0.80 |
| Val attention | ~0.42 |
| SC4002 stride4 | ~0.70 |
| SC4031 stride4 | ~0.72 |
| sub013 balanced | ~0.33 |
| sub017 probe ACC | 0.0 (collapsed away from concentration) |

**Takeaway:** Vigilance stays in the ballpark of the multi-night pool (~0.70–0.72 vs prior ~0.75) but dips a bit under joint training. Attention does **not** transfer — train attention Acc≈0.5 / holdout near chance / sub017 probe Acc=0. Suggests domain+montage gap (BioSemi→Muse-proxy vs Sleep-EDF vs TUEG pretrain) and/or too few attention subjects. Do **not** ship the joint head for attention yet; keep precision-tuned binary as default pack head.

## Artifacts

- `scripts/head_a_4way_attention_train.py`
- `exports/head_a_4way_attention_train/head_a_4way_joint.pt`
- `exports/head_a_4way_attention_train/metrics_summary.json`
- `exports/head_a_4way_attention_train/run_manifest.json`

## Next candidates

1. Ingest more attention (ds003969 Muse-native AF7/AF8, or more ds001787 subjects).
2. Attention-only head smoke before re-joining.
3. Repack CBraMod pack with multi-night pool binary (still best vigilance artifact).

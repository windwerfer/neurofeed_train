# Multi-night pool (SC4021+SC4041) — Head A binary (drowsy vs hypnagogic)

Generated: 2026-09-07T02:17:01.399077+00:00 (UTC). User TZ Asia/Bangkok (UTC+7).

## Protocol

- **Train:** SC4001 + SC4011 + SC4021 + SC4041 (four cassette subjects). Per-night undersample, then equalize to the smallest night's per-class count so abundant-N1 nights do not dominate.
- **Primary holdout:** SC4002 (same holdout night as wave-1 single-night train→holdout, for comparison).
- **Secondary holdout:** SC4031 (unseen subject).
- Encoder: frozen CBraMod. Head: `HeadALinear`. Recipe: first N1 ±20/40 min; window 2 s hop 0.5 s; majority 0.7; W→drowsy, N1→hypnagogic.
- Prefer **stride×4 / greedy non-overlap** scores over hop-0.5 overlap.

Script: `scripts/multi_night_pool_sc4021_sc4041.py`. Artifacts under `exports/head_a_multi_night_pool/`.

## Equal-night train pool

```json
{
  "sc4001": {
    "per_class_used": 298,
    "n_total": 596
  },
  "sc4011": {
    "per_class_used": 298,
    "n_total": 596
  },
  "sc4021": {
    "per_class_used": 298,
    "n_total": 596
  },
  "sc4041": {
    "per_class_used": 298,
    "n_total": 596
  }
}
```

Train pool after equalize: n=2384 (val_frac=0.2). Final train n=1907, val n=477.

## Headline holdout metrics

| Split | n | accuracy | macro-F1 |
|-------|---|----------|----------|
| Train balanced (multi-night) | 1907 | 0.8128 | 0.8098 |
| Val same pool | 477 | 0.7945 | 0.7939 |
| SC4002 all stride×4 | 645 | 0.9178 | 0.7470 |
| SC4002 balanced greedy | 90 | 0.8111111111111111 | 0.8070861177657295 |
| SC4031 all stride×4 | 883 | 0.8063 | 0.7408 |
| SC4031 balanced greedy | 390 | 0.7897435897435897 | 0.7892947501581278 |

## vs prior single-night and 2-night multi

Prior single-night overlap all-labeled macro-F1 ≈ 0.7387904632961433; balanced ≈ 0.8236453201970444.
Prior 2-night (SC4001+SC4011) SC4002 stride×4 ≈ 0.7311207311207311; SC4031 stride×4 ≈ 0.7308610617221234.
4-night pool SC4002 all-overlap macro-F1 = 0.7317; stride×4 = 0.7470.

## Per-class (SC4002 stride×4)

```json
{
  "drowsy": {
    "precision": 0.9756521739130435,
    "recall": 0.935,
    "f1": 0.9548936170212766,
    "support": 600.0
  },
  "hypnagogic": {
    "precision": 0.44285714285714284,
    "recall": 0.6888888888888889,
    "f1": 0.5391304347826087,
    "support": 45.0
  }
}
```

Confusion: `[[561, 39], [14, 31]]`

## Per-class (SC4031 stride×4)

```json
{
  "drowsy": {
    "precision": 0.9045383411580594,
    "recall": 0.8401162790697675,
    "f1": 0.8711379050489826,
    "support": 688.0
  },
  "hypnagogic": {
    "precision": 0.5491803278688525,
    "recall": 0.6871794871794872,
    "f1": 0.6104783599088839,
    "support": 195.0
  }
}
```

Confusion: `[[578, 110], [61, 134]]`

## Caveats

- Montage mismatch vs CBraMod TUEG pretrain; Muse-proxy channels only.
- Overlap hop scores inflate effective n; trust stride/greedy.
- Still binary drowsy/hypnagogic proxies — concentration / mind_wandering need attention datasets next.

## Artifacts

- Head: `exports/head_a_multi_night_pool/head_a_binary_multi_night_pool.pt`
- Metrics: `exports/head_a_multi_night_pool/metrics_summary.json`
- Full run: `exports/head_a_multi_night_pool/run_manifest.json`

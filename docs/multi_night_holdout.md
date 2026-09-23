# Multi-night holdout — Head A binary (drowsy vs hypnagogic)

Generated: 2026-09-06T23:49:33.148216+00:00 (UTC). User TZ Asia/Bangkok (UTC+7).

## Protocol

- **Train:** SC4001 + SC4011 (two cassette subjects). Per-night undersample, then equalize to the smaller night's per-class count so SC4011's abundant N1 does not dominate.
- **Primary holdout:** SC4002 (same holdout night as wave-1 single-night train→holdout, for comparison).
- **Secondary holdout:** SC4031 (unseen subject).
- Encoder: frozen CBraMod. Head: `HeadALinear`. Recipe: first N1 ±20/40 min; window 2 s hop 0.5 s; majority 0.7; W→drowsy, N1→hypnagogic.
- Prefer **stride×4 / greedy non-overlap** scores over hop-0.5 overlap.

Script: `scripts/multi_night_holdout.py`. Artifacts under `exports/head_a_multi_night/`.

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
  }
}
```

Train pool after equalize: n=1192 (val_frac=0.2). Final train n=954, val n=238.

## Headline holdout metrics

| Split | n | accuracy | macro-F1 |
|-------|---|----------|----------|
| Train balanced (multi-night) | 954 | 0.8354 | 0.8349 |
| Val same pool | 238 | 0.8361 | 0.8340 |
| SC4002 all stride×4 | 645 | 0.8992 | 0.7311 |
| SC4002 balanced greedy | 90 | 0.8444444444444444 | 0.8441365660564077 |
| SC4031 all stride×4 | 883 | 0.7826 | 0.7309 |
| SC4031 balanced greedy | 390 | 0.7948717948717948 | 0.7948664001683148 |

## vs single-night (SC4001→SC4002)

Prior single-night overlap all-labeled macro-F1 ≈ 0.7387904632961433; balanced ≈ 0.8236453201970444.
Multi-night SC4002 all-overlap macro-F1 = 0.7258; stride×4 = 0.7311.

## Per-class (SC4002 stride×4)

```json
{
  "drowsy": {
    "precision": 0.9819819819819819,
    "recall": 0.9083333333333333,
    "f1": 0.9437229437229437,
    "support": 600.0
  },
  "hypnagogic": {
    "precision": 0.3888888888888889,
    "recall": 0.7777777777777778,
    "f1": 0.5185185185185185,
    "support": 45.0
  }
}
```

Confusion: `[[545, 55], [10, 35]]`

## Per-class (SC4031 stride×4)

```json
{
  "drowsy": {
    "precision": 0.9261168384879725,
    "recall": 0.7834302325581395,
    "f1": 0.8488188976377952,
    "support": 688.0
  },
  "hypnagogic": {
    "precision": 0.5049833887043189,
    "recall": 0.7794871794871795,
    "f1": 0.6129032258064515,
    "support": 195.0
  }
}
```

Confusion: `[[539, 149], [43, 152]]`

## Caveats

- Montage mismatch vs CBraMod TUEG pretrain; Muse-proxy channels only.
- Overlap hop scores inflate effective n; trust stride/greedy.
- Still binary drowsy/hypnagogic proxies — concentration / mind_wandering need attention datasets next.

## Artifacts

- Head: `exports/head_a_multi_night/head_a_binary_multi_night.pt`
- Metrics: `exports/head_a_multi_night/metrics_summary.json`
- Full run: `exports/head_a_multi_night/run_manifest.json`

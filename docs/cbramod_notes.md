# CBraMod notes (Muse EEG heads)

## Sources

| Item | Location |
|------|----------|
| Code | [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod) (Apache-2.0) |
| Weights | [weighting666/CBraMod](https://huggingface.co/weighting666/CBraMod) (`pretrained_weights.pth`) |
| Paper | Wang et al., *CBraMod*, ICLR 2025 / arXiv:2412.07236 |

Vendored (import-path adapted) in `src/cbramod_backbone.py` + `src/criss_cross_transformer.py`.

## Native input

- Tensor layout: **`(B, C, n_patches, 200)`**
- **200 samples / patch = 1 second @ 200 Hz** (hard assumption: spectral branch uses `rfft` → 101 bins)
- **C** is flexible (architecture has no fixed channel projection); pretrain used ~19 standard 10–20 channels on TUEG @ 200 Hz
- Output of backbone: **`(B, C, n_patches, 200)`** (same layout; `proj_out` is Linear 200→200)
- Load: `CBraMod(...).load_state_dict(torch.load(path, map_location=..., weights_only=True))`
- Downstream recipe often sets `model.proj_out = nn.Identity()` then pools/flattens; we keep pretrained `proj_out` and **mean-pool** over C×patches → `(B, 200)`

## What we feed (Muse / Sleep-EDF proxy)

| Ours | Adapter |
|------|---------|
| `(B, 4, T)` @ **256 Hz**, typically T=512 (2 s) | Resample → 200 Hz (`T→400`), view as `(B, 4, 2, 200)` |
| Channels AF7, AF8, TP9, TP10 (Sleep-EDF: AF7=AF8=Fpz-Cz, TP9=TP10=Pz-Oz) | **Keep C=4** — do **not** invent missing 10–20 electrodes |

## Honest limitations

1. **Domain / montage gap**: TUEG 19-ch ≠ Muse 4-ch proxy. Spatial attention still runs over 4 channels; embeddings are transfer probes, not calibrated clinical features.
2. **SR / patch**: Windows must yield an integer number of 1 s patches after 200 Hz resample; trailing partial patches are **cropped** (short windows may zero-pad to one patch — rare for our 2 s smoke).
3. **Frozen only**: all encoder `requires_grad=False`; train **Head A** only.
4. Positional depthwise Conv2d uses kernel `(19, 7)` with padding that preserves arbitrary C / n_patches — verified against upstream.

## Smoke Head A

Binary **drowsy vs hypnagogic** (Sleep-EDF W / N1). Weighted CE or undersample majority. See `src/cbramod_encoder.py`, `src/head_a.py`, notebook `03_cbramod_head_a_smoke`.

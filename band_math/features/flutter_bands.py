"""Flutter Muse classical band features — match rust/src/api/muse.rs + features.rs.

FFT: 256 samples @ 256 Hz (1 s), Cooley–Tukey power = |X[k]|^2 / n^2.
Band edges (inclusive bin via round(hz)): δ 1–4, θ 4–8, α 8–13, β 13–30, γ 30–50.
Muse feature pads: AF7, AF8 (indices 0,1 in AF7/AF8/TP9/TP10 order).

2 s windows (512 samples): average bands from both 1 s halves (app scores ~1 Hz).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

SR_HZ = 256.0
FFT_N = 256
CHANNEL_ORDER = ("AF7", "AF8", "TP9", "TP10")
FEATURE_ELECTRODES = ("AF7", "AF8")  # indices 0, 1
FEATURE_CH_IDX = (0, 1)

# (lo_hz, hi_hz) — hi uses round; gamma hi capped at min(n/2, 50)
BAND_EDGES_HZ = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

APP_FEATURE_IDS = (
    "band.atr",
    "band.tar",
    "band.btr",
    "band.alpha",
    "band.delta",
)


def _bin(hz: float, hz_per_bin: float) -> int:
    return int(round(hz / hz_per_bin))


def compute_channel_bands(samples: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Absolute band powers [δ,θ,α,β,γ] for one channel, length FFT_N.

    Mirrors muse.rs compute_fft_bands power terms (not peak/line-noise).
    """
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if x.size < FFT_N:
        pad = np.zeros(FFT_N, dtype=np.float64)
        pad[: x.size] = x
        x = pad
    elif x.size > FFT_N:
        x = x[:FFT_N]
    n = FFT_N
    hz_per_bin = SR_HZ / n
    # rfft: bins 0..n/2; |rfft[k]|^2 == |full FFT[k]|^2 for k=0..n/2
    spec = np.fft.rfft(x)
    p = (spec.real * spec.real + spec.imag * spec.imag)  # length n/2+1
    half_n = n // 2
    nn = float(n * n)

    def band_power(lo_hz: float, hi_hz: float) -> float:
        lo = _bin(lo_hz, hz_per_bin)
        hi = _bin(hi_hz, hz_per_bin)
        if lo_hz == 1.0:
            lo = max(lo, 1)  # rust: (1..=bin(4))
        lo = max(lo, 0)
        hi = min(hi, half_n)
        if hi < lo:
            return 0.0
        # rfft index == full FFT index for these bins
        return float(p[lo : hi + 1].sum() / nn)

    gamma_hi = min(half_n, 50)
    return (
        band_power(1.0, 4.0),
        band_power(4.0, 8.0),
        band_power(8.0, 13.0),
        band_power(13.0, 30.0),
        band_power(30.0, float(gamma_hi)),
    )


def _bands_from_2s(ch: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Average absolute bands over two 1 s halves of a 2 s (512) window."""
    ch = np.asarray(ch, dtype=np.float64).reshape(-1)
    if ch.size < FFT_N:
        return compute_channel_bands(ch)
    halves = []
    n_halves = max(1, ch.size // FFT_N)
    for i in range(n_halves):
        halves.append(compute_channel_bands(ch[i * FFT_N : (i + 1) * FFT_N]))
    arr = np.asarray(halves, dtype=np.float64)
    return tuple(arr.mean(axis=0).tolist())  # type: ignore[return-value]


def aggregate_band_feature(fid: str, pads: Sequence[Tuple[float, float, float, float, float]]) -> float:
    """Match features.rs aggregate_band_feature. NaN if undefined."""
    if not pads:
        return float("nan")
    if fid == "band.delta":
        return float(sum(p[0] for p in pads) / len(pads))
    rels = []
    for d, t, a, b, g in pads:
        tot = d + t + a + b + g
        if tot <= 0:
            continue
        rels.append((d / tot, t / tot, a / tot, b / tot, g / tot))
    if not rels:
        return float("nan")
    inv = 1.0 / len(rels)
    t = sum(r[1] for r in rels) * inv
    a = sum(r[2] for r in rels) * inv
    b = sum(r[3] for r in rels) * inv
    if fid == "band.atr":
        return float("nan") if t <= 0 else a / t
    if fid == "band.tar":
        return float("nan") if a <= 0 else t / a
    if fid == "band.btr":
        return float("nan") if t <= 0 else b / t
    if fid == "band.alpha":
        return a
    return float("nan")


def compute_window_features(X: np.ndarray) -> Dict[str, np.ndarray]:
    """X: (N, 4, T) muse4. Returns dict of feature vectors length N + relative band matrix."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    # per-window absolute bands on AF7/AF8
    pads_d = np.zeros((n, 2, 5), dtype=np.float64)
    for i in range(n):
        for j, ch_i in enumerate(FEATURE_CH_IDX):
            pads_d[i, j] = _bands_from_2s(X[i, ch_i])

    out: Dict[str, np.ndarray] = {}
    for fid in APP_FEATURE_IDS:
        vals = np.empty(n, dtype=np.float64)
        for i in range(n):
            pads = [tuple(pads_d[i, j]) for j in range(2)]
            vals[i] = aggregate_band_feature(fid, pads)
        out[fid] = vals

    # relative bands averaged AF7/AF8 (extras for logistic)
    rel = np.zeros((n, 5), dtype=np.float64)
    abs_mean = pads_d.mean(axis=1)  # (N, 5)
    tot = abs_mean.sum(axis=1, keepdims=True)
    good = tot[:, 0] > 0
    rel[good] = abs_mean[good] / tot[good]
    out["rel_delta"] = rel[:, 0]
    out["rel_theta"] = rel[:, 1]
    out["rel_alpha"] = rel[:, 2]
    out["rel_beta"] = rel[:, 3]
    out["rel_gamma"] = rel[:, 4]
    out["abs_delta"] = abs_mean[:, 0]
    out["abs_theta"] = abs_mean[:, 1]
    out["abs_alpha"] = abs_mean[:, 2]
    out["abs_beta"] = abs_mean[:, 3]
    out["abs_gamma"] = abs_mean[:, 4]
    # extras
    with np.errstate(divide="ignore", invalid="ignore"):
        out["extra.theta_beta"] = np.where(rel[:, 3] > 0, rel[:, 1] / rel[:, 3], np.nan)
        out["extra.delta_theta"] = np.where(rel[:, 1] > 0, rel[:, 0] / rel[:, 1], np.nan)
        out["extra.alpha_beta"] = np.where(rel[:, 3] > 0, rel[:, 2] / rel[:, 3], np.nan)
    return out


def feature_names() -> List[str]:
    return list(APP_FEATURE_IDS) + [
        "rel_delta",
        "rel_theta",
        "rel_alpha",
        "rel_beta",
        "rel_gamma",
        "abs_delta",
        "abs_theta",
        "abs_alpha",
        "abs_beta",
        "abs_gamma",
        "extra.theta_beta",
        "extra.delta_theta",
        "extra.alpha_beta",
    ]


def compute_window_features_fast(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Vectorized PSD for (N,4,T) — same math as compute_window_features."""
    X = np.asarray(X, dtype=np.float64)
    n, c, t = X.shape
    assert c >= 2
    n_halves = max(1, t // FFT_N)
    # gather halves: (n, 2 pads, n_halves, FFT_N)
    halves = np.zeros((n, 2, n_halves, FFT_N), dtype=np.float64)
    for h in range(n_halves):
        halves[:, :, h, :] = X[:, :2, h * FFT_N : (h + 1) * FFT_N]

    flat = halves.reshape(-1, FFT_N)
    spec = np.fft.rfft(flat, axis=-1)
    p = spec.real**2 + spec.imag**2  # (M, n/2+1)
    nn = float(FFT_N * FFT_N)
    hz_per_bin = SR_HZ / FFT_N
    half_n = FFT_N // 2

    def slice_power(lo_hz: float, hi_hz: float, start_at_1: bool = False) -> np.ndarray:
        lo = _bin(lo_hz, hz_per_bin)
        hi = _bin(hi_hz, hz_per_bin)
        if start_at_1:
            lo = max(lo, 1)
        hi = min(hi, half_n)
        return p[:, lo : hi + 1].sum(axis=-1) / nn

    gamma_hi = min(half_n, 50)
    bands = np.stack(
        [
            slice_power(1.0, 4.0, start_at_1=True),
            slice_power(4.0, 8.0),
            slice_power(8.0, 13.0),
            slice_power(13.0, 30.0),
            slice_power(30.0, float(gamma_hi)),
        ],
        axis=-1,
    )  # (M, 5)
    bands = bands.reshape(n, 2, n_halves, 5).mean(axis=2)  # (n, 2, 5)

    out: Dict[str, np.ndarray] = {}
    # aggregate app features
    abs_delta = bands[:, :, 0].mean(axis=1)
    out["band.delta"] = abs_delta

    tot = bands.sum(axis=-1)  # (n, 2)
    rel = np.zeros_like(bands)
    mask = tot > 0
    rel[mask] = bands[mask] / tot[mask, None]
    # average relative across usable pads (both assumed usable offline)
    rel_avg = rel.mean(axis=1)  # (n, 5)
    t_ = rel_avg[:, 1]
    a_ = rel_avg[:, 2]
    b_ = rel_avg[:, 3]
    eps = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        atr = np.where(t_ > eps, a_ / t_, np.nan)
        tar = np.where(a_ > eps, t_ / a_, np.nan)
        btr = np.where(t_ > eps, b_ / t_, np.nan)
    out["band.atr"] = atr
    out["band.tar"] = tar
    out["band.btr"] = btr
    out["band.alpha"] = a_

    abs_mean = bands.mean(axis=1)
    tot_m = abs_mean.sum(axis=1, keepdims=True)
    rel_m = np.zeros_like(abs_mean)
    good = tot_m[:, 0] > 0
    rel_m[good] = abs_mean[good] / tot_m[good]
    out["rel_delta"] = rel_m[:, 0]
    out["rel_theta"] = rel_m[:, 1]
    out["rel_alpha"] = rel_m[:, 2]
    out["rel_beta"] = rel_m[:, 3]
    out["rel_gamma"] = rel_m[:, 4]
    out["abs_delta"] = abs_mean[:, 0]
    out["abs_theta"] = abs_mean[:, 1]
    out["abs_alpha"] = abs_mean[:, 2]
    out["abs_beta"] = abs_mean[:, 3]
    out["abs_gamma"] = abs_mean[:, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["extra.theta_beta"] = np.where(rel_m[:, 3] > 0, rel_m[:, 1] / rel_m[:, 3], np.nan)
        out["extra.delta_theta"] = np.where(rel_m[:, 1] > 0, rel_m[:, 0] / rel_m[:, 1], np.nan)
        out["extra.alpha_beta"] = np.where(rel_m[:, 3] > 0, rel_m[:, 2] / rel_m[:, 3], np.nan)
    return out

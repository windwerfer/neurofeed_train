"""Sleep-EDF Expanded pilot → Muse-proxy tensors + Head A vigilance labels."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .edf_io import read_edf_annotations, read_edf_signals
from .metrics import HEAD_A_LABELS
from .windowing import stack_windows

PathLike = Union[str, Path]

# Sleep-EDF cassette EEG (not Muse). Proxied into Muse order for pipeline bring-up.
SLEEP_EDF_EEG = ["EEG Fpz-Cz", "EEG Pz-Oz"]

PROXY_NOTE = (
    "Sleep-EDF has Fpz-Cz / Pz-Oz only. Mapped to Muse order as "
    "AF7=AF8=Fpz-Cz (frontal proxy), TP9=TP10=Pz-Oz (posterior proxy). "
    "Replace with true Muse / Schreer Muse-S for domain adaptation."
)

# Hypnogram text → Head A label or None (exclude)
# Sleep-EDF uses: "Sleep stage W/1/2/3/4/R/?" and sometimes "Movement time"
STAGE_TO_HEAD_A: Dict[str, Optional[str]] = {
    "Sleep stage W": "drowsy",       # coarse wake/vigilance proxy; refine later
    "Sleep stage 1": "hypnagogic",   # N1 / sleep-onset (Sleep-EDF R&K)
    "Sleep stage N1": "hypnagogic",  # AASM (HMC)
    "Sleep stage 2": None,
    "Sleep stage N2": None,
    "Sleep stage 3": None,
    "Sleep stage N3": None,
    "Sleep stage 4": None,
    "Sleep stage R": None,
    "Sleep stage REM": None,
    "Sleep stage ?": None,
    "Movement time": None,
}


def resample_poly(x: np.ndarray, orig_sr: float, target_sr: float) -> np.ndarray:
    """Resample along last axis. Uses scipy if present, else linear interpolate."""
    if abs(orig_sr - target_sr) < 1e-6:
        return x.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly as _rp
        from math import gcd
        up = int(round(target_sr))
        down = int(round(orig_sr))
        g = gcd(up, down)
        up //= g
        down //= g
        # scipy along axis=-1 by default for 1d; for 2d apply per channel
        if x.ndim == 1:
            return _rp(x, up, down).astype(np.float32)
        return np.stack([_rp(x[i], up, down) for i in range(x.shape[0])], axis=0).astype(np.float32)
    except Exception:
        n_old = x.shape[-1]
        duration = n_old / orig_sr
        n_new = int(round(duration * target_sr))
        t_old = np.linspace(0.0, duration, num=n_old, endpoint=False)
        t_new = np.linspace(0.0, duration, num=n_new, endpoint=False)
        if x.ndim == 1:
            return np.interp(t_new, t_old, x).astype(np.float32)
        return np.stack([np.interp(t_new, t_old, x[i]) for i in range(x.shape[0])], axis=0).astype(np.float32)


def bandpass_fft(x: np.ndarray, sfreq: float, l_freq: float = 1.0, h_freq: float = 45.0) -> np.ndarray:
    """Simple FFT bandpass along last axis (good enough for pilot)."""
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    X = np.fft.rfft(x, axis=-1)
    mask = (freqs >= l_freq) & (freqs <= h_freq)
    X = X * mask
    y = np.fft.irfft(X, n=n, axis=-1)
    return y.astype(np.float32)


def sleep_edf_to_muse_proxy(eeg_2ch: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """(2, T) Fpz-Cz/Pz-Oz → (4, T) AF7,AF8,TP9,TP10 proxy."""
    if eeg_2ch.shape[0] != 2:
        raise ValueError(f"expected 2 EEG channels, got {eeg_2ch.shape}")
    fpz, pz = eeg_2ch[0], eeg_2ch[1]
    out = np.stack([fpz, fpz, pz, pz], axis=0)
    return out.astype(np.float32), ["AF7", "AF8", "TP9", "TP10"]


def hypnogram_to_stage_series(
    events: Sequence[Tuple[float, float, str]],
    n_times: int,
    sfreq: float,
) -> np.ndarray:
    """Build per-sample stage label strings (object array), default 'Sleep stage ?'."""
    stages = np.empty(n_times, dtype=object)
    stages[:] = "Sleep stage ?"
    for onset, dur, text in events:
        text = text.strip()
        if not text.startswith("Sleep stage") and text != "Movement time":
            continue
        a = int(max(0, round(onset * sfreq)))
        if dur and dur > 0:
            b = int(min(n_times, round((onset + dur) * sfreq)))
        else:
            # Sleep-EDF often uses 30s epochs with duration filled; if 0, assume 30s
            b = int(min(n_times, round((onset + 30.0) * sfreq)))
        stages[a:b] = text
    return stages


def stage_to_head_a(stage: str) -> Optional[str]:
    return STAGE_TO_HEAD_A.get(stage.strip(), None)


def windows_with_head_a_labels(
    x: np.ndarray,
    stages: np.ndarray,
    sfreq: float,
    window_sec: float = 2.0,
    hop_sec: float = 0.5,
    majority_frac: float = 0.7,
    max_windows: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Window x (C,T) and assign Head A labels by majority stage inside each window.

    Returns windows (N,C,W), labels (N,), keep_indices into full window grid.
    Only windows whose majority stage maps to a Head A label are kept.
    """
    wins = stack_windows(x, sfreq, window_sec=window_sec, hop_sec=hop_sec, max_windows=max_windows)
    win = int(round(window_sec * sfreq))
    hop = int(round(hop_sec * sfreq))
    labels: List[str] = []
    keep: List[int] = []
    start = 0
    wi = 0
    n_times = x.shape[-1]
    while start + win <= n_times:
        if max_windows is not None and wi >= max_windows:
            break
        seg = stages[start : start + win]
        # majority vote
        vals, counts = np.unique(seg, return_counts=True)
        maj = vals[int(np.argmax(counts))]
        frac = float(counts.max()) / float(len(seg))
        lab = stage_to_head_a(str(maj)) if frac >= majority_frac else None
        if lab is not None:
            keep.append(wi)
            labels.append(lab)
        start += hop
        wi += 1
    if not keep:
        return np.zeros((0, x.shape[0], win), dtype=np.float32), [], []
    return wins[np.asarray(keep)], labels, keep


def first_stage_onset(events: Sequence[Tuple[float, float, str]], stage_substr: str = "stage 1") -> Optional[float]:
    """First hypnogram onset matching stage_substr (case-sensitive substring).

    For AASM corpora pass stage_substr="stage N1" (or rely on aliases below).
    """
    aliases = {
        "stage 1": ("stage 1", "stage N1"),
        "stage N1": ("stage N1", "stage 1"),
    }
    needles = aliases.get(stage_substr, (stage_substr,))
    for onset, _dur, text in events:
        for n in needles:
            if n in text:
                return float(onset)
    return None


def load_sleep_edf_recording(
    psg_path: PathLike,
    hypno_path: PathLike,
    target_sr: float = 256.0,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    max_records: Optional[int] = None,
    start_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    around_stage: Optional[str] = None,
    pre_sec: float = 30 * 60,
    post_sec: float = 60 * 60,
) -> Dict:
    """
    Load one Sleep-EDF cassette night → Muse-proxy EEG + per-sample stages.

    Time selection (record size = 30s in Sleep-EDF):
    - max_records: cap from start (legacy smoke)
    - start_sec / duration_sec: absolute slice
    - around_stage: e.g. "stage 1" → [onset-pre_sec, onset+post_sec]
    """
    from .edf_io import read_edf_header

    events = read_edf_annotations(hypno_path)
    info = read_edf_header(psg_path)
    rec_dur = float(info.record_duration)

    if around_stage:
        onset = first_stage_onset(events, around_stage)
        if onset is None:
            raise ValueError(f"No hypnogram event matching {around_stage!r}")
        start_sec = max(0.0, onset - pre_sec)
        duration_sec = pre_sec + post_sec

    start_rec = int(start_sec // rec_dur)
    if duration_sec is not None:
        n_rec = int(max(1, round(duration_sec / rec_dur)))
    elif max_records is not None:
        n_rec = max_records
        start_rec = 0
    else:
        n_rec = info.n_records - start_rec

    n_rec = min(n_rec, info.n_records - start_rec)
    # read_edf_signals only supports prefix max_records — use skip via new helper
    eeg, chs, sfreq = read_edf_signals(
        psg_path,
        labels=SLEEP_EDF_EEG,
        max_records=start_rec + n_rec,
    )
    # drop leading records if start_rec > 0
    if start_rec > 0:
        spr = int(round(sfreq * rec_dur))
        eeg = eeg[:, start_rec * spr : (start_rec + n_rec) * spr]

    eeg = bandpass_fft(eeg, sfreq, l_freq=l_freq, h_freq=h_freq)
    eeg = resample_poly(eeg, sfreq, target_sr)
    x, muse_order = sleep_edf_to_muse_proxy(eeg)

    # Shift annotations so stage series aligns to sliced timeline
    t0 = start_rec * rec_dur
    shifted = [(onset - t0, dur, text) for onset, dur, text in events]
    stages = hypnogram_to_stage_series(shifted, n_times=x.shape[-1], sfreq=target_sr)
    return {
        "data": x,
        "ch_names": muse_order,
        "sfreq": float(target_sr),
        "stages": stages,
        "events": events,
        "slice_start_sec": float(t0),
        "source_channels": list(chs),
        "source_sfreq": float(sfreq),
        "proxy_note": PROXY_NOTE,
        "psg_path": str(psg_path),
        "hypno_path": str(hypno_path),
        "head_a_labels": list(HEAD_A_LABELS),
    }


def find_pilot_pairs(root: PathLike) -> List[Tuple[Path, Path]]:
    """Match SC*PSG.edf with SC*Hypnogram.edf under a sleep-edfx-pilot tree."""
    root = Path(root)
    psgs = sorted(root.rglob("*-PSG.edf"))
    pairs = []
    for psg in psgs:
        # SC4001E0-PSG.edf → SC4001EC-Hypnogram.edf (prefix SC4001)
        stem = psg.name.replace("-PSG.edf", "")
        # common pattern: SC4001E0 → SC4001EC
        prefix = stem[:6]  # SC4001
        cands = list(psg.parent.glob(f"{prefix}*-Hypnogram.edf"))
        if not cands:
            cands = list(root.rglob(f"{prefix}*-Hypnogram.edf"))
        if cands:
            pairs.append((psg, cands[0]))
    return pairs


# Head C (dataset-only): hypnogram → coarse stage; Head A mapping unchanged above.
STAGE_TO_COARSE: Dict[str, str] = {
    "Sleep stage W": "wake",
    "Sleep stage 1": "light",
    "Sleep stage N1": "light",
    "Sleep stage 2": "light",
    "Sleep stage N2": "light",
    "Sleep stage 3": "deep",
    "Sleep stage N3": "deep",
    "Sleep stage 4": "deep",
    "Sleep stage R": "rem",
    "Sleep stage REM": "rem",
    "Sleep stage ?": "unknown",
    "Movement time": "unknown",
}


def stage_to_coarse(stage: str) -> str:
    return STAGE_TO_COARSE.get(stage.strip(), "unknown")


def majority_stage_in_window(
    stages: np.ndarray,
    start: int,
    win: int,
) -> str:
    """Majority hypnogram label inside stages[start:start+win]."""
    seg = stages[start : start + win]
    if len(seg) == 0:
        return "Sleep stage ?"
    vals, counts = np.unique(seg, return_counts=True)
    return str(vals[int(np.argmax(counts))])


def sleep_period_bounds(
    events: Sequence[Tuple[float, float, str]],
    wake_margin_sec: float = 30 * 60,
    recording_end_sec: Optional[float] = None,
) -> Tuple[float, float]:
    """Return [start, end] sec covering sleep + wake margins (drop long pre/post wake).

    Sleep onset = first non-wake/non-unknown stage; offset = last such stage end.
    """
    sleep_onsets: List[float] = []
    sleep_ends: List[float] = []
    for onset, dur, text in events:
        t = text.strip()
        if t in ("Sleep stage W", "Sleep stage ?", "Movement time"):
            continue
        if not t.startswith("Sleep stage"):
            continue
        d = float(dur) if dur and dur > 0 else 30.0
        sleep_onsets.append(float(onset))
        sleep_ends.append(float(onset) + d)
    if not sleep_onsets:
        raise ValueError("no non-wake sleep stages in hypnogram")
    start = max(0.0, min(sleep_onsets) - wake_margin_sec)
    end = max(sleep_ends) + wake_margin_sec
    if recording_end_sec is not None:
        end = min(end, float(recording_end_sec))
    return start, end


def windows_with_stage_coarse(
    x: np.ndarray,
    stages: np.ndarray,
    sfreq: float,
    window_sec: float = 2.0,
    hop_sec: float = 0.5,
    majority_frac: float = 0.7,
    drop_unknown: bool = True,
    max_windows: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[str], List[int]]:
    """Window x and assign stage_raw + stage_coarse by majority hypnogram vote.

    Returns windows (N,C,W), stage_raw labels, stage_coarse labels, keep indices.
    """
    wins = stack_windows(x, sfreq, window_sec=window_sec, hop_sec=hop_sec, max_windows=max_windows)
    win = int(round(window_sec * sfreq))
    hop = int(round(hop_sec * sfreq))
    stage_raws: List[str] = []
    stage_coarses: List[str] = []
    keep: List[int] = []
    start = 0
    wi = 0
    n_times = x.shape[-1]
    while start + win <= n_times:
        if max_windows is not None and wi >= max_windows:
            break
        maj = majority_stage_in_window(stages, start, win)
        seg = stages[start : start + win]
        vals, counts = np.unique(seg, return_counts=True)
        frac = float(counts.max()) / float(len(seg))
        coarse = stage_to_coarse(str(maj))
        if frac >= majority_frac and (not drop_unknown or coarse != "unknown"):
            keep.append(wi)
            stage_raws.append(str(maj))
            stage_coarses.append(coarse)
        start += hop
        wi += 1
    if not keep:
        return (
            np.zeros((0, x.shape[0], win), dtype=np.float32),
            [],
            [],
            [],
        )
    return wins[np.asarray(keep)], stage_raws, stage_coarses, keep

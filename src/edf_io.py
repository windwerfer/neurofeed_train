"""Minimal pure-Python EDF / EDF+ reader (no native deps).

Enough for Sleep-EDF Expanded PSG + Hypnogram annotation files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union
import numpy as np


PathLike = Union[str, Path]


@dataclass
class EdfSignal:
    label: str
    samples_per_record: int
    digital_min: int
    digital_max: int
    physical_min: float
    physical_max: float
    transducer: str = ""
    dimension: str = ""


@dataclass
class EdfFile:
    path: Path
    n_records: int
    record_duration: float
    signals: List[EdfSignal]
    data_offset: int


def _decode_ascii(b: bytes) -> str:
    return b.decode("latin1").strip()


def read_edf_header(path: PathLike) -> EdfFile:
    path = Path(path)
    with path.open("rb") as f:
        header = f.read(256)
        if len(header) < 256:
            raise ValueError(f"EDF header too short: {path}")
        n_records = int(_decode_ascii(header[236:244]))
        record_duration = float(_decode_ascii(header[244:252]))
        nsig = int(_decode_ascii(header[252:256]))
        sig_hdr = f.read(256 * nsig)
        if len(sig_hdr) < 256 * nsig:
            raise ValueError(f"EDF signal headers truncated: {path}")

    def field(start: int, width: int) -> List[str]:
        out = []
        for i in range(nsig):
            o = start + i * width
            out.append(_decode_ascii(sig_hdr[o : o + width]))
        return out

    # Sequential fields in EDF signal header block
    o = 0
    labels = field(o, 16); o += 16 * nsig
    transducers = field(o, 80); o += 80 * nsig
    dims = field(o, 8); o += 8 * nsig
    phys_min = field(o, 8); o += 8 * nsig
    phys_max = field(o, 8); o += 8 * nsig
    dig_min = field(o, 8); o += 8 * nsig
    dig_max = field(o, 8); o += 8 * nsig
    o += 80 * nsig  # prefilter
    spr = field(o, 8)

    signals = []
    for i in range(nsig):
        signals.append(
            EdfSignal(
                label=labels[i],
                samples_per_record=int(spr[i]),
                digital_min=int(dig_min[i]),
                digital_max=int(dig_max[i]),
                physical_min=float(phys_min[i]),
                physical_max=float(phys_max[i]),
                transducer=transducers[i],
                dimension=dims[i],
            )
        )
    return EdfFile(
        path=path,
        n_records=n_records,
        record_duration=record_duration,
        signals=signals,
        data_offset=256 * (1 + nsig),
    )


def _scale_digital(sig: EdfSignal, dig: np.ndarray) -> np.ndarray:
    dmin, dmax = sig.digital_min, sig.digital_max
    pmin, pmax = sig.physical_min, sig.physical_max
    if dmax == dmin:
        return dig.astype(np.float64)
    return (dig.astype(np.float64) - dmin) / (dmax - dmin) * (pmax - pmin) + pmin


def read_edf_signals(
    path: PathLike,
    labels: Sequence[str] | None = None,
    max_records: int | None = None,
) -> Tuple[np.ndarray, List[str], float]:
    """
    Read selected signals as (n_channels, n_times) float64 physical units.

    All selected signals must share the same samples_per_record (same rate).
    Returns data, labels_used, sample_rate_hz.
    """
    info = read_edf_header(path)
    if labels is None:
        idxs = list(range(len(info.signals)))
    else:
        by = {s.label: i for i, s in enumerate(info.signals)}
        missing = [l for l in labels if l not in by]
        if missing:
            raise KeyError(f"EDF missing labels {missing}; have {[s.label for s in info.signals]}")
        idxs = [by[l] for l in labels]

    sels = [info.signals[i] for i in idxs]
    spr0 = sels[0].samples_per_record
    if any(s.samples_per_record != spr0 for s in sels):
        raise ValueError("Selected signals have mismatched sample rates")

    n_rec = info.n_records if max_records is None else min(info.n_records, max_records)
    samples_per_rec_all = [s.samples_per_record for s in info.signals]
    rec_bytes = 2 * sum(samples_per_rec_all)

    # Preallocate
    out = np.zeros((len(idxs), n_rec * spr0), dtype=np.float64)
    # Offsets within a record for each signal
    offsets = []
    acc = 0
    for spr in samples_per_rec_all:
        offsets.append(acc)
        acc += spr

    with Path(path).open("rb") as f:
        f.seek(info.data_offset)
        for r in range(n_rec):
            blob = f.read(rec_bytes)
            if len(blob) < rec_bytes:
                raise EOFError(f"Truncated EDF data record {r} in {path}")
            dig_rec = np.frombuffer(blob, dtype="<i2")
            for ch, si in enumerate(idxs):
                o = offsets[si]
                dig = dig_rec[o : o + spr0]
                out[ch, r * spr0 : (r + 1) * spr0] = _scale_digital(info.signals[si], dig)

    sfreq = spr0 / info.record_duration
    used = [info.signals[i].label for i in idxs]
    return out, used, float(sfreq)


def read_edf_annotations(path: PathLike) -> List[Tuple[float, float, str]]:
    """
    Read EDF+ annotations as list of (onset_sec, duration_sec, text).

    Sleep-EDF Hypnogram files store stages in the annotation signal.
    """
    info = read_edf_header(path)
    # Find annotation signal(s): label starts with 'EDF Annotations'
    ann_idxs = [i for i, s in enumerate(info.signals) if s.label.startswith("EDF Annotations")]
    if not ann_idxs:
        # Some hypnograms still use EDF Annotations
        raise ValueError(f"No EDF Annotations signal in {path}; labels={[s.label for s in info.signals]}")

    samples_per_rec_all = [s.samples_per_record for s in info.signals]
    rec_bytes = 2 * sum(samples_per_rec_all)
    offsets = []
    acc = 0
    for spr in samples_per_rec_all:
        offsets.append(acc)
        acc += spr

    events: List[Tuple[float, float, str]] = []
    with Path(path).open("rb") as f:
        f.seek(info.data_offset)
        for r in range(info.n_records):
            blob = f.read(rec_bytes)
            if len(blob) < rec_bytes:
                break
            dig_rec = np.frombuffer(blob, dtype="<i2")
            for si in ann_idxs:
                o = offsets[si]
                spr = samples_per_rec_all[si]
                # Annotation signal is stored as little-endian int16 but is really bytes
                raw = dig_rec[o : o + spr].astype("<i2").tobytes()
                # TAL: +onset\x15duration\x14text\x14\x00 ...
                # Multiple TALs separated by \x00
                parts = raw.split(b"\x00")
                for part in parts:
                    if not part or part == b"\x14":
                        continue
                    # strip trailing \x14
                    if b"\x14" not in part:
                        continue
                    try:
                        # format: +onset[\x15duration]\x14annotation[\x14annotation...]\x14
                        head, *rest = part.split(b"\x14")
                        head = head.decode("latin1", errors="replace")
                        if not head.startswith("+") and not head.startswith("-"):
                            # record start marker sometimes without annotations
                            continue
                        if "\x15" in head:
                            onset_s, dur_s = head.split("\x15", 1)
                            onset = float(onset_s)
                            dur = float(dur_s) if dur_s else 0.0
                        else:
                            onset = float(head)
                            dur = 0.0
                        for ann in rest:
                            text = ann.decode("latin1", errors="replace").strip()
                            if text:
                                events.append((onset, dur, text))
                    except Exception:
                        continue
    return events

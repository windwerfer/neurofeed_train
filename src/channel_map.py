"""Muse channel layout and encoder index helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence

# Locked Muse montage order for this project
MUSE_CHANNELS: List[str] = ["AF7", "AF8", "TP9", "TP10"]

# Common aliases seen in BIDS / vendor exports
_ALIASES: Dict[str, str] = {
    "af7": "AF7",
    "af8": "AF8",
    "tp9": "TP9",
    "tp10": "TP10",
    "EEG-AF7": "AF7",
    "EEG-AF8": "AF8",
    "EEG-TP9": "TP9",
    "EEG-TP10": "TP10",
}


def normalize_channel_name(name: str) -> str:
    key = name.strip()
    if key in MUSE_CHANNELS:
        return key
    return _ALIASES.get(key, _ALIASES.get(key.lower(), key))


def select_muse_channels(
    ch_names: Sequence[str],
    data_channels_first,
):
    """
    Reorder array (n_channels, n_times) to MUSE_CHANNELS order.

    Raises KeyError if a required Muse channel is missing after alias normalize.
    """
    import numpy as np

    normalized = [normalize_channel_name(c) for c in ch_names]
    index = {n: i for i, n in enumerate(normalized)}
    missing = [c for c in MUSE_CHANNELS if c not in index]
    if missing:
        raise KeyError(f"Missing Muse channels: {missing}; have {normalized}")
    idxs = [index[c] for c in MUSE_CHANNELS]
    arr = np.asarray(data_channels_first)
    return arr[idxs], list(MUSE_CHANNELS)


def muse_to_encoder_indices(
    encoder_channel_names: Sequence[str],
) -> List[int]:
    """
    Map Muse AF7/AF8/TP9/TP10 onto an encoder's expected channel list.

    Returns indices into encoder_channel_names for each Muse channel in order.
    Unmatched channels raise KeyError (caller may pad or subsample instead).
    """
    enc_norm = [normalize_channel_name(c) for c in encoder_channel_names]
    index = {n: i for i, n in enumerate(enc_norm)}
    out: List[int] = []
    for ch in MUSE_CHANNELS:
        if ch not in index:
            raise KeyError(
                f"Encoder layout missing {ch}; encoder has {list(encoder_channel_names)}"
            )
        out.append(index[ch])
    return out


# CBraMod accepts variable C; pretrained on ~19 10-20 ch @ 200 Hz.
# We keep Muse 4-ch and do NOT fabricate a denser montage (see docs/cbramod_notes.md).
CBRAMOD_MUSE_SUBSET_HINT = {
    "strategy": "keep_muse_4ch",
    "muse_order": MUSE_CHANNELS,
    "native_layout": "(B, C, n_patches, 200) @ 200 Hz",
    "note": "Adapter resamples Muse windows to 200 Hz patches; C stays 4.",
}


# Dual-montage support (see docs/montages_muse_crown.md)
MUSE4_STREAM_ORDER = ["TP9", "AF7", "AF8", "TP10"]
MUSE4_MODEL_ORDER = ["AF7", "AF8", "TP9", "TP10"]  # app REVE remap
CROWN8_CHANNELS = ["CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"]


def muse_stream_to_model_order(data_channels_first):
    """Reorder (4, T) from stream TP9,AF7,AF8,TP10 → AF7,AF8,TP9,TP10."""
    import numpy as np
    x = np.asarray(data_channels_first)
    if x.shape[0] != 4:
        raise ValueError(f"expected 4 Muse channels, got {x.shape}")
    # stream: 0 TP9, 1 AF7, 2 AF8, 3 TP10
    return x[[1, 2, 0, 3]], list(MUSE4_MODEL_ORDER)

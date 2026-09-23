"""Frozen REVE (brain-bzh/reve-base) encoder + Muse4 input adapter.

REVE native input (HF AutoModel):
  eeg: (B, C, T) float32 @ **exactly 200 Hz**
  pos: (B, C, 3) meters (+x right, +y anterior, +z superior)
  Output tokens: (B, C, n_patches, 512); we attention-pool → (B, 512).

Our Muse attention windows:
  (B, 4, T) @ source_sr (typically 256 Hz), window_sec typically 2.0 → T=512.
  Channel order in NPZ: AF7, AF8, TP9, TP10 (= MUSE4_MODEL_ORDER / app REVE remap).

Honest adapter:
  1. Resample last axis source_sr → 200 Hz (linear interpolate).
  2. Crop/pad so T_200 >= patch_size (200) and is compatible with unfold
     (patch_size=200, overlap=20 → step=180). Default: keep native duration
     (2.0 s → 400 samples @ 200 Hz = 2 patches). Documented choice — we do
     **not** invent a 4 s / 5 s window from 2 s data.
  3. Per-channel z-score + clip ±15σ (HF Reve forward does not normalize;
     matches pretraining / braindecode guidance).
  4. Positions from brain-bzh/reve-positions (preferred) or montages.json fallback.

Gated weights: brain-bzh/reve-base — prefer private Kaggle cache `muse-eeg-heads-cache/models/reve-*`; HF_TOKEN hub fallback only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .channel_map import MUSE4_MODEL_ORDER
except ImportError:  # flat Kaggle src pack
    from channel_map import MUSE4_MODEL_ORDER

PathLike = Union[str, Path]

REVE_SR = 200.0
REVE_PATCH = 200
REVE_OVERLAP = 20
REVE_EMB = 512
REVE_MODEL_ID = "brain-bzh/reve-base"
REVE_POS_ID = "brain-bzh/reve-positions"

# Prefer private Kaggle cache (offline) before HF hub.
_REVE_CACHE_MARKERS = ("config.json", "model.safetensors")
_REVE_CACHE_CANDIDATE_ROOTS = (
    Path("/kaggle/input/muse-eeg-heads-cache/models"),
    Path("/kaggle/input/muse-eeg-heads-cache/models/REVE"),
    Path("kaggle_datasets/muse-eeg-heads-cache/models"),
)


def _looks_like_hf_snapshot(d: Path) -> bool:
    return d.is_dir() and all((d / m).exists() for m in _REVE_CACHE_MARKERS)


def find_reve_cache_dir(name: str, *, extra_roots: Optional[Sequence[PathLike]] = None) -> Optional[Path]:
    """Find local `reve-base` / `reve-positions` dir under known cache roots or /kaggle/input glob."""
    roots: list[Path] = []
    for r in list(extra_roots or []) + list(_REVE_CACHE_CANDIDATE_ROOTS):
        roots.append(Path(r))
    # Direct children: models/reve-base, models/REVE/reve-base
    for root in roots:
        for cand in (root / name, root / "REVE" / name, root / name.replace("reve-", "REVE/")):
            if _looks_like_hf_snapshot(cand):
                return cand.resolve()
    # Glob under /kaggle/input
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for hit in sorted(input_root.glob(f"**/models/{name}")):
            if _looks_like_hf_snapshot(hit):
                return hit.resolve()
        for hit in sorted(input_root.glob(f"**/models/REVE/{name}")):
            if _looks_like_hf_snapshot(hit):
                return hit.resolve()
        for hit in sorted(input_root.glob(f"**/{name}")):
            if _looks_like_hf_snapshot(hit) and "models" in hit.parts:
                return hit.resolve()
    return None


def resolve_reve_pretrained(
    model_id: str = REVE_MODEL_ID,
    positions_id: str = REVE_POS_ID,
    *,
    local_model: Optional[PathLike] = None,
    local_positions: Optional[PathLike] = None,
    extra_roots: Optional[Sequence[PathLike]] = None,
) -> tuple[str, str, bool, bool]:
    """Return (model_path_or_id, positions_path_or_id, model_is_local, pos_is_local).

    Prefers explicit local_* paths, then private cache under /kaggle/input/.../muse-eeg-heads-cache,
    then HF hub ids (caller supplies HF_TOKEN as fallback).
    """
    model_local = False
    pos_local = False
    model_ref = model_id
    pos_ref = positions_id

    if local_model and _looks_like_hf_snapshot(Path(local_model)):
        model_ref = str(Path(local_model).resolve())
        model_local = True
    else:
        found = find_reve_cache_dir("reve-base", extra_roots=extra_roots)
        if found is not None:
            model_ref = str(found)
            model_local = True

    if local_positions and _looks_like_hf_snapshot(Path(local_positions)):
        pos_ref = str(Path(local_positions).resolve())
        pos_local = True
    else:
        found = find_reve_cache_dir("reve-positions", extra_roots=extra_roots)
        if found is not None:
            pos_ref = str(found)
            pos_local = True

    return model_ref, pos_ref, model_local, pos_local


# Official meters (stream order TP9,AF7,AF8,TP10) — same as datasets/common/montages.json
_FALLBACK_POS_M = {
    "TP9": [-0.08562, -0.04651, -0.04571],
    "AF7": [-0.05484, 0.06857, -0.01059],
    "AF8": [0.05574, 0.06966, -0.01075],
    "TP10": [0.08616, -0.04704, -0.04587],
}


def resample_bct(x: torch.Tensor, orig_sr: float, target_sr: float) -> torch.Tensor:
    """Resample (B, C, T) along time with linear interpolate.

    Always run F.interpolate on CPU then move back. Belt-and-suspenders for
    Pascal/P100 (sm_60) where newer CUDA wheels lack kernels for interpolate.
    """
    if abs(orig_sr - target_sr) < 1e-6:
        return x
    b, c, t = x.shape
    duration = t / float(orig_sr)
    t_new = max(1, int(round(duration * target_sr)))
    device = x.device
    dtype = x.dtype
    x_cpu = x.float().cpu()
    y = F.interpolate(x_cpu, size=t_new, mode="linear", align_corners=False)
    return y.to(device=device, dtype=dtype)


def fit_reve_length(
    x: torch.Tensor,
    patch_size: int = REVE_PATCH,
    overlap: int = REVE_OVERLAP,
) -> torch.Tensor:
    """Ensure (B,C,T) is long enough for at least one unfold patch; crop trailing."""
    if x.ndim != 3:
        raise ValueError(f"expected (B,C,T), got {tuple(x.shape)}")
    b, c, t = x.shape
    if t < patch_size:
        x = F.pad(x, (0, patch_size - t))
        t = patch_size
    step = patch_size - overlap
    # Keep as many full steps as possible; unfold needs last start + patch_size <= T
    n_patches = 1 + max(0, (t - patch_size) // step)
    t_keep = patch_size + (n_patches - 1) * step
    if t_keep < t:
        x = x[..., :t_keep]
    return x


def per_channel_zscore_clip(x: torch.Tensor, clip: float = 15.0) -> torch.Tensor:
    """Per-channel z-score over time, then clip ±clip σ (pretraining-matched)."""
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return ((x - mean) / std).clamp(-clip, clip)


def load_fallback_positions(
    channel_names: Sequence[str],
    montages_path: Optional[PathLike] = None,
) -> torch.Tensor:
    """(C, 3) float32 from montages.json or hardcoded official bank."""
    pos_map = dict(_FALLBACK_POS_M)
    if montages_path is not None and Path(montages_path).exists():
        data = json.loads(Path(montages_path).read_text())
        bank = data.get("muse4", {}).get("positions_m_stream_order") or {}
        for k, v in bank.items():
            pos_map[k] = list(v)
    rows = []
    for ch in channel_names:
        if ch not in pos_map:
            raise KeyError(f"no fallback position for {ch}; have {sorted(pos_map)}")
        rows.append(pos_map[ch])
    return torch.tensor(rows, dtype=torch.float32)


class FrozenREVEEncoder(nn.Module):
    """Load gated reve-base + positions, freeze, adapt Muse (B,C,T) → (B, 512)."""

    def __init__(
        self,
        *,
        model_id: str = REVE_MODEL_ID,
        positions_id: str = REVE_POS_ID,
        source_sr: float = 256.0,
        channel_names: Optional[Sequence[str]] = None,
        pool: str = "attention",  # attention | mean
        hf_token: Optional[str] = None,
        montages_path: Optional[PathLike] = None,
        device: Optional[Union[str, torch.device]] = None,
        trust_remote_code: bool = True,
        local_model: Optional[PathLike] = None,
        local_positions: Optional[PathLike] = None,
        cache_roots: Optional[Sequence[PathLike]] = None,
        prefer_local: bool = True,
    ):
        super().__init__()
        self.source_sr = float(source_sr)
        self.pool = pool
        self.channel_names: List[str] = list(channel_names or MUSE4_MODEL_ORDER)
        self.emb_dim = REVE_EMB

        if prefer_local:
            model_ref, pos_ref, model_local, pos_local = resolve_reve_pretrained(
                model_id,
                positions_id,
                local_model=local_model,
                local_positions=local_positions,
                extra_roots=cache_roots,
            )
        else:
            model_ref, pos_ref, model_local, pos_local = model_id, positions_id, False, False
            if local_model and _looks_like_hf_snapshot(Path(local_model)):
                model_ref, model_local = str(Path(local_model).resolve()), True
            if local_positions and _looks_like_hf_snapshot(Path(local_positions)):
                pos_ref, pos_local = str(Path(local_positions).resolve()), True

        self.model_id = model_ref
        self.positions_id = pos_ref
        self.model_from_local = model_local
        self.positions_from_local = pos_local

        # HF_TOKEN only needed when falling back to hub for gated weights
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError("transformers required for FrozenREVEEncoder") from e

        pos_kwargs = {"trust_remote_code": trust_remote_code}
        model_kwargs = {"trust_remote_code": trust_remote_code}
        if pos_local:
            pos_kwargs["local_files_only"] = True
        else:
            pos_kwargs["token"] = token
        if model_local:
            model_kwargs["local_files_only"] = True
        else:
            model_kwargs["token"] = token

        try:
            self.pos_bank = AutoModel.from_pretrained(pos_ref, **pos_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {pos_ref}. "
                f"{'Local cache incomplete?' if pos_local else 'Positions are public; check network.'} Err={e}"
            ) from e

        try:
            self.backbone = AutoModel.from_pretrained(model_ref, **model_kwargs)
        except Exception as e:
            msg = str(e).lower()
            gated = any(s in msg for s in ("gated", "401", "403", "authorized", "access"))
            hint = ""
            if gated and not model_local:
                hint = (
                    " GATED MODEL: prefer private Kaggle cache "
                    "`/kaggle/input/muse-eeg-heads-cache/models/reve-base`, or accept the license at "
                    "https://huggingface.co/brain-bzh/reve-base with the same HF account as HF_TOKEN / "
                    "Kaggle secret, then re-run. Do not hang waiting for access."
                )
            raise RuntimeError(f"Failed to load {model_ref}.{hint} Err={e}") from e

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # Resolve positions once (C, 3)
        try:
            pos_c3 = self.pos_bank(self.channel_names)
            if pos_c3.shape[0] != len(self.channel_names):
                raise RuntimeError(
                    f"pos_bank returned {pos_c3.shape[0]} rows for {self.channel_names}"
                )
        except Exception:
            pos_c3 = load_fallback_positions(self.channel_names, montages_path)
        self.register_buffer("positions_c3", pos_c3.float(), persistent=False)

        if device is not None:
            self.to(device)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self.backbone.eval()
        return self

    def prepare_eeg(self, x_bct: torch.Tensor) -> torch.Tensor:
        """(B,C,T) @ source_sr → (B,C,T200) z-scored for REVE."""
        x = resample_bct(x_bct.float(), self.source_sr, REVE_SR)
        x = fit_reve_length(x)
        x = per_channel_zscore_clip(x)
        return x

    @torch.no_grad()
    def encode_tokens(self, x_bct: torch.Tensor) -> torch.Tensor:
        """Return (B, C, n_patches, 512) token grid."""
        eeg = self.prepare_eeg(x_bct)
        b = eeg.shape[0]
        pos = self.positions_c3.unsqueeze(0).expand(b, -1, -1).to(eeg.device)
        return self.backbone(eeg, pos)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B,C,H,512) → (B,512)."""
        if self.pool == "attention" and hasattr(self.backbone, "attention_pooling"):
            return self.backbone.attention_pooling(tokens)
        if self.pool in ("mean", "attention"):
            return tokens.mean(dim=(1, 2))
        raise ValueError(f"unknown pool {self.pool!r}")

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.encode_tokens(x_bct)
            emb = self.pool_tokens(tokens)
        return emb.detach()

    def adapter_notes(self) -> dict:
        return {
            "encoder": "REVE-base",
            "model_id": self.model_id,
            "positions_id": self.positions_id,
            "emb_dim": self.emb_dim,
            "native_input": "(B, C, T) @ 200 Hz + (B, C, 3) positions",
            "fed_input": f"(B, C, T) @ {self.source_sr} Hz → resample 200 Hz → zscore±15",
            "window_policy": (
                "Keep native window duration after resample "
                "(2.0 s @ 256 Hz → 400 samples @ 200 Hz; patch=200 overlap=20 → 2 patches). "
                "Not inventing 4 s/5 s from 2 s data."
            ),
            "channel_order": list(self.channel_names),
            "pool": self.pool,
            "patch_size": REVE_PATCH,
            "patch_overlap": REVE_OVERLAP,
            "target_sr": REVE_SR,
            "gated": True,
            "model_from_local": self.model_from_local,
            "positions_from_local": self.positions_from_local,
            "license_note": (
                "Gated brain-bzh/reve-base: prefer private muse-eeg-heads-cache offline snapshot; "
                "HF_TOKEN hub fallback only. No weight redistribution."
            ),
        }

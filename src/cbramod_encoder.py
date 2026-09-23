"""Frozen CBraMod encoder + Muse-proxy input adapter.

CBraMod native input (upstream wjq-learning/CBraMod):
  x: (B, C, n_patches, patch_size) with patch_size==200 (== 1 s @ 200 Hz).
  Any C is accepted by the architecture (pretrained ~19 10-20 channels on TUEG).
  Output: (B, C, n_patches, 200) patch features (after proj_out).

Our Muse / Sleep-EDF-proxy windows:
  (B, 4, T) @ source_sr (typically 256 Hz), window_sec typically 2.0 → T=512.

Honest adapter (NOT silent physics invention):
  1. Resample last axis source_sr → 200 Hz (polyphase / linear fallback).
  2. Crop/pad so T_200 is divisible by 200.
  3. View as (B, 4, n_patches, 200).
  Channel count stays 4 (AF7,AF8,TP9,TP10). We do NOT invent extra 10-20
  electrodes; spatial attention runs over the 4 proxy channels only.
  Domain gap vs TUEG 19-ch pretrain is expected — this is a smoke / transfer probe.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbramod_backbone import CBraMod

PathLike = Union[str, Path]

CBRAMOD_SR = 200.0
CBRAMOD_PATCH = 200  # samples per patch @ 200 Hz
CBRAMOD_EMB = 200


def file_sha256(path: PathLike, max_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    path = Path(path)
    with path.open("rb") as f:
        if max_bytes is None:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(f.read(max_bytes))
    return h.hexdigest()


def resample_bt_ct(
    x: torch.Tensor,
    orig_sr: float,
    target_sr: float,
) -> torch.Tensor:
    """Resample (B, C, T) along time with linear interpolate (torch-only)."""
    if abs(orig_sr - target_sr) < 1e-6:
        return x
    b, c, t = x.shape
    duration = t / float(orig_sr)
    t_new = max(1, int(round(duration * target_sr)))
    # F.interpolate expects (N, C, L)
    return F.interpolate(x, size=t_new, mode="linear", align_corners=False)


def to_cbramod_patches(
    x: torch.Tensor,
    source_sr: float,
    patch_size: int = CBRAMOD_PATCH,
    target_sr: float = CBRAMOD_SR,
) -> torch.Tensor:
    """
    (B, C, T) @ source_sr → (B, C, n_patches, patch_size) @ target_sr.

    Crops trailing samples that do not fill a full patch (honest; no zero-pad
    inventing temporal content unless T_200 < patch_size, then right-pad zeros
    once so a single patch exists — documented edge case for short windows).
    """
    if x.ndim != 3:
        raise ValueError(f"expected (B,C,T), got {tuple(x.shape)}")
    x200 = resample_bt_ct(x, source_sr, target_sr)
    b, c, t = x200.shape
    if t < patch_size:
        pad = patch_size - t
        x200 = F.pad(x200, (0, pad))
        t = patch_size
    n_patches = t // patch_size
    t_keep = n_patches * patch_size
    if t_keep < t:
        x200 = x200[..., :t_keep]
    return x200.view(b, c, n_patches, patch_size)


def pool_patch_features(feats: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """(B, C, S, D) → (B, D)."""
    if mode == "mean":
        return feats.mean(dim=(1, 2))
    if mode == "flatten":
        b = feats.shape[0]
        return feats.reshape(b, -1)
    raise ValueError(f"unknown pool mode {mode!r}")


class FrozenCBraModEncoder(nn.Module):
    """Load CBraMod weights, freeze all params, adapt Muse-proxy (B,C,T)."""

    def __init__(
        self,
        weights_path: PathLike,
        *,
        source_sr: float = 256.0,
        pool: str = "mean",
        replace_proj_out_identity: bool = False,
        map_location: str = "cpu",
    ):
        super().__init__()
        self.source_sr = float(source_sr)
        self.pool = pool
        self.weights_path = str(weights_path)
        self.weights_sha256 = file_sha256(weights_path)

        self.backbone = CBraMod(
            in_dim=CBRAMOD_PATCH,
            out_dim=CBRAMOD_EMB,
            d_model=CBRAMOD_EMB,
            dim_feedforward=800,
            seq_len=30,
            n_layer=12,
            nhead=8,
        )
        sd = torch.load(weights_path, map_location=map_location, weights_only=True)
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys in CBraMod weights: {unexpected[:8]}...")
        # missing only OK if empty under strict architecture match
        if missing:
            raise RuntimeError(f"missing keys loading CBraMod: {missing[:8]}...")

        if replace_proj_out_identity:
            self.backbone.proj_out = nn.Identity()

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()
        self.emb_dim = CBRAMOD_EMB if pool == "mean" else None  # flatten dim depends on C,S

    def train(self, mode: bool = True):  # type: ignore[override]
        # Keep backbone in eval always (dropout/BN frozen behaviour)
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_patches(self, x_bct: torch.Tensor) -> torch.Tensor:
        patches = to_cbramod_patches(x_bct, source_sr=self.source_sr)
        return self.backbone(patches)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        """
        x_bct: (B, C, T) Muse-proxy @ self.source_sr.
        returns pooled embedding (B, D) with D=200 for pool='mean'.
        """
        with torch.no_grad():
            feats = self.encode_patches(x_bct)
        # Detach so head training never builds graph through encoder
        feats = feats.detach()
        return pool_patch_features(feats, mode=self.pool)

    def adapter_notes(self) -> dict:
        return {
            "encoder": "CBraMod",
            "weights_path": self.weights_path,
            "weights_sha256": self.weights_sha256,
            "native_input": "(B, C, n_patches, 200) @ 200 Hz",
            "fed_input": f"(B, C, T) @ {self.source_sr} Hz → resample → patches",
            "channel_policy": "keep Muse 4-ch; no fabricated 10-20 montage",
            "pool": self.pool,
            "patch_size": CBRAMOD_PATCH,
            "target_sr": CBRAMOD_SR,
            "license": "Apache-2.0 (HF weighting666/CBraMod; code wjq-learning/CBraMod)",
        }

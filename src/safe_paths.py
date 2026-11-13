"""Resolve paths and reject I/O that escapes an allowed root.

Used by training scripts so CLI arguments and paths taken from JSON/manifests
cannot traverse outside the repo or a configured data root.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

PathLike = Union[str, os.PathLike[str], Path]

ENV_DATA_ROOT = "NEUROFEED_DATA_ROOT"
ENV_CACHE = "NEUROFEED_CACHE"
ENV_KAGGLE_OUT = "KAGGLE_OUT3"
ENV_WEIGHTS = "CBRAMOD_WEIGHTS"

_CACHE_ENVS = (ENV_CACHE, ENV_KAGGLE_OUT)


def _resolve(path: PathLike) -> Path:
    return Path(path).expanduser().resolve()


def env_roots(*keys: str) -> list[Path]:
    """Optional extra roots from the environment."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for key in keys or (ENV_DATA_ROOT, ENV_CACHE, ENV_KAGGLE_OUT):
        value = os.environ.get(key)
        if not value:
            continue
        resolved = _resolve(value)
        if resolved not in seen:
            roots.append(resolved)
            seen.add(resolved)
    return roots


def allowed_bases(repo_root: PathLike) -> list[Path]:
    return [_resolve(repo_root), *env_roots()]


def resolve_under(path: PathLike, *bases: PathLike) -> Path:
    """Return the resolved path if it stays inside one of ``bases``.

    Relative paths are resolved against the first base. Raises ``ValueError``
    if the resolved location would escape every allowed root.
    """
    if not bases:
        raise ValueError("resolve_under requires at least one allowed base directory")
    raw = Path(path).expanduser()
    first = _resolve(bases[0])
    candidate = raw if raw.is_absolute() else (first / raw)
    resolved = candidate.resolve()
    allowed = [_resolve(base) for base in bases]
    for base in allowed:
        try:
            if resolved == base or resolved.is_relative_to(base):
                return resolved
        except ValueError:
            continue
    roots = ", ".join(str(base) for base in allowed)
    raise ValueError(f"path escapes allowed root(s) {roots}: {path}")


def cache_fallback_paths(relative: PathLike) -> list[Path]:
    """Lookup paths under ``NEUROFEED_CACHE`` / ``KAGGLE_OUT3`` (no hardcoded /tmp)."""
    rel = Path(relative)
    if rel.is_absolute():
        raise ValueError(f"cache relative path must not be absolute: {relative}")
    out: list[Path] = []
    seen: set[Path] = set()
    for key in _CACHE_ENVS:
        value = os.environ.get(key)
        if not value:
            continue
        resolved = _resolve(value) / rel
        if resolved not in seen:
            out.append(resolved)
            seen.add(resolved)
    return out


def cbramod_weight_candidates(repo_root: PathLike) -> list[Path]:
    """Candidate locations for CBraMod pretrained weights (repo cache + env)."""
    root = Path(repo_root)
    rel = Path("kaggle_datasets/muse-eeg-heads-cache/models/CBraMod/pretrained_weights.pth")
    cands: list[Path] = []
    weights = os.environ.get(ENV_WEIGHTS)
    if weights:
        cands.append(Path(weights).expanduser())
    cands.append(root / rel)
    cands.extend(cache_fallback_paths("models/CBraMod/pretrained_weights.pth"))
    out: list[Path] = []
    seen: set[str] = set()
    for path in cands:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out

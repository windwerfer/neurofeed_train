"""Public data/weights resolution for lab scripts (no private paths).

Windows
-------
``windows_dir(corpus)`` returns the folder holding ``*_windows.npz`` (+ optional ``*_qc.npz``):

1. ``--hf`` / ``use_hf=True`` or env ``NEUROFEED_USE_HF=1``: download that config's ``windows/`` from the public
   Hugging Face dataset ``windwerfer/neurofeed-eeg-windows`` (cached by ``huggingface_hub``; set ``HF_HOME`` to move
   the cache).
2. env ``HF_WINDOWS_ROOT``: an existing local snapshot of that dataset (``<root>/<hf_config>/windows``).
3. default: ``<repo>/datasets/<corpus>/windows`` (output of the lab export scripts).

CBraMod weights
---------------
``find_cbramod_weights()``: env ``CBRAMOD_WEIGHTS`` -> ``<repo>/models/CBraMod/pretrained_weights.pth`` ->
download ``pretrained_weights.pth`` from the public ``weighting666/CBraMod`` model repo (Apache-2.0).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_DATASET = "windwerfer/neurofeed-eeg-windows"
CBRAMOD_REPO = "weighting666/CBraMod"

# lab corpus name -> Hugging Face config name
CORPUS_TO_HF = {
    "vigilance_sleep_edf": "muse4_vigilance_sleep_edf",
    "vigilance_hmc_crown2": "crown2_vigilance_hmc",
    "vigilance_hmc_crown4": "crown4_vigilance_hmc",
    "attention_ds001787": "muse4_attention_ds001787",
    "attention_ds003969": "muse4_attention_ds003969",
    "attention_ds001787_crown8": "crown8_attention_ds001787",
    "attention_ds003969_crown8": "crown8_attention_ds003969",
    "engagement_a_eng": "muse4_engagement_a_eng",
}


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def windows_dir(corpus: str, use_hf: bool = False, revision: str | None = None) -> Path:
    cfg = CORPUS_TO_HF.get(corpus, corpus)
    if use_hf or _env_true("NEUROFEED_USE_HF"):
        from huggingface_hub import snapshot_download

        snap = snapshot_download(
            HF_DATASET,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[f"{cfg}/windows/*.npz"],
        )
        return Path(snap) / cfg / "windows"
    hf_root = os.environ.get("HF_WINDOWS_ROOT")
    if hf_root:
        return Path(hf_root).expanduser() / cfg / "windows"
    return ROOT / "datasets" / corpus / "windows"


def find_cbramod_weights() -> Path:
    env = os.environ.get("CBRAMOD_WEIGHTS")
    if env:
        p = Path(env).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"CBRAMOD_WEIGHTS points to a missing file: {p}")
        return p
    local = ROOT / "models" / "CBraMod" / "pretrained_weights.pth"
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(CBRAMOD_REPO, "pretrained_weights.pth"))

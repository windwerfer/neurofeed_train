#!/usr/bin/env python3
"""Light artifact QC on windows NPZ (obvious junk only).

Writes:
  - datasets/<corpus>/windows/<recording>_qc.npz  (mask + reasons; X not duplicated)
  - datasets/<corpus>/annotations/windows_qc.csv   (per-window qc_pass / qc_reason)
  - exports/artifact_qc_light/summary.json
  - docs/artifact_qc_light.md
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
EXPORTS = ROOT / "exports" / "artifact_qc_light"
DOCS = ROOT / "docs"

CORPORA = [
    "vigilance_sleep_edf",
    "attention_ds001787",
    "attention_ds003969",
    "attention_ds001787_crown8",
    "attention_ds003969_crown8",
]

# Units as stored in NPZ (document; do NOT silently rescale here).
UNITS_AS_STORED = {
    "vigilance_sleep_edf": "uV",
    "attention_ds001787": "V",
    "attention_ds003969": "V",
    "attention_ds001787_crown8": "V",
    "attention_ds003969_crown8": "V",
}

FLAT_STD = {
    "vigilance_sleep_edf": 0.1,  # uV
    "attention_ds001787": 1e-9,  # V
    "attention_ds003969": 1e-9,
    "attention_ds001787_crown8": 1e-9,
    "attention_ds003969_crown8": 1e-9,
}

PEAK_ABS = {
    "vigilance_sleep_edf": 350.0,  # uV
    "attention_ds001787": 0.15,  # V
    "attention_ds003969": 0.15,
    "attention_ds001787_crown8": 0.15,
    "attention_ds003969_crown8": 0.15,
}

Z_EXTREME = 25.0
LINE_NOISE_FRAC = 0.85
LINE_NOISE_ENABLE = True
SR_HZ = 256.0


def subject_for_recording(corpus: str, recording_id: str) -> str:
    if corpus == "vigilance_sleep_edf":
        return recording_id[:5]
    if corpus.startswith("attention_ds001787"):
        core = recording_id.split("_")[0]
        num = core.replace("sub", "")
        return f"sub-{num.zfill(3) if num.isdigit() else num}"
    if corpus.startswith("attention_ds003969"):
        num = recording_id.replace("sub", "")
        return f"sub-{num.zfill(3) if num.isdigit() else num}"
    return recording_id


def load_split_map(corpus_dir: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        p = corpus_dir / "splits" / f"{split}_subjects.json"
        if not p.exists():
            continue
        for sid in json.loads(p.read_text())["subjects"]:
            out[sid] = split
    return out


def line_noise_frac(x: np.ndarray, sr: float = SR_HZ) -> float:
    """Max over channels of (power in 48-52 U 58-62) / total power."""
    c, t = x.shape
    if t < 32:
        return 0.0
    win = np.hanning(t).astype(np.float64)
    freqs = np.fft.rfftfreq(t, d=1.0 / sr)
    band = ((freqs >= 48.0) & (freqs <= 52.0)) | ((freqs >= 58.0) & (freqs <= 62.0))
    if not band.any():
        return 0.0
    fracs = []
    for ch in range(c):
        sig = (x[ch].astype(np.float64) - float(x[ch].mean())) * win
        ps = np.abs(np.fft.rfft(sig)) ** 2
        tot = float(ps.sum())
        if tot <= 0.0:
            fracs.append(0.0)
        else:
            fracs.append(float(ps[band].sum() / tot))
    return float(max(fracs)) if fracs else 0.0


def qc_window(
    x: np.ndarray,
    *,
    flat_std: float,
    peak_abs: float,
    z_extreme: float = Z_EXTREME,
    line_noise_frac_thresh: float = LINE_NOISE_FRAC,
    check_line_noise: bool = LINE_NOISE_ENABLE,
) -> Tuple[bool, str]:
    if not np.isfinite(x).all():
        return False, "nan_inf"
    stds = x.std(axis=-1)
    if float(stds.min()) < flat_std:
        return False, "flat"
    if float(np.abs(x).max()) > peak_abs:
        return False, "peak_abs"
    means = x.mean(axis=-1, keepdims=True)
    std_safe = np.maximum(stds, flat_std)[:, None]
    z = np.abs((x - means) / std_safe)
    if float(z.max()) > z_extreme:
        return False, "peak_z"
    if check_line_noise:
        frac = line_noise_frac(x)
        if frac >= line_noise_frac_thresh:
            return False, "line_noise"
    return True, ""


def qc_file(npz_path: Path, corpus: str) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    label_names = [str(v) for v in data["label_names"].tolist()]
    n = int(X.shape[0])
    flat_std = FLAT_STD[corpus]
    peak_abs = PEAK_ABS[corpus]
    qc_pass = np.zeros(n, dtype=bool)
    qc_reason = np.array([""] * n, dtype=object)
    reason_counts: Dict[str, int] = defaultdict(int)
    for i in range(n):
        ok, reason = qc_window(X[i], flat_std=flat_std, peak_abs=peak_abs)
        qc_pass[i] = ok
        qc_reason[i] = reason
        reason_counts["pass" if ok else reason] += 1

    recording_id = npz_path.name.replace("_windows.npz", "")
    out_npz = npz_path.with_name(f"{recording_id}_qc.npz")
    payload = {
        "qc_pass": qc_pass,
        "qc_reason": qc_reason.astype(str),
        "y": y,
        "label_names": np.asarray(label_names),
        "n_total": np.int64(n),
        "n_pass": np.int64(int(qc_pass.sum())),
        "thresholds_flat_std": np.float64(flat_std),
        "thresholds_peak_abs": np.float64(peak_abs),
        "thresholds_z_extreme": np.float64(Z_EXTREME),
        "thresholds_line_noise_frac": np.float64(LINE_NOISE_FRAC),
        "units_as_stored": np.asarray(UNITS_AS_STORED[corpus]),
        "source_npz": np.asarray(str(npz_path.relative_to(ROOT))),
    }
    if "starts" in data.files:
        payload["starts"] = data["starts"].astype(np.int64)
    np.savez_compressed(out_npz, **payload)

    drop_by_label: Dict[str, Dict[str, int]] = {}
    for li, name in enumerate(label_names):
        mask = y == li
        n_lab = int(mask.sum())
        n_ok = int((mask & qc_pass).sum())
        drop_by_label[name] = {
            "n": n_lab,
            "pass": n_ok,
            "drop": n_lab - n_ok,
            "drop_rate": (n_lab - n_ok) / n_lab if n_lab else 0.0,
        }

    return {
        "corpus": corpus,
        "recording_id": recording_id,
        "npz_path": str(npz_path.relative_to(ROOT)),
        "qc_npz_path": str(out_npz.relative_to(ROOT)),
        "n_total": n,
        "n_pass": int(qc_pass.sum()),
        "n_drop": int((~qc_pass).sum()),
        "drop_rate": float((~qc_pass).sum()) / n if n else 0.0,
        "reason_counts": dict(reason_counts),
        "drop_by_label": drop_by_label,
        "qc_pass": qc_pass,
        "qc_reason": qc_reason.astype(str),
        "y": y,
        "label_names": label_names,
    }


def summarize_and_write(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    by_corpus: Dict[str, Any] = {}
    csv_rows_by_corpus: Dict[str, List[dict]] = defaultdict(list)

    for r in results:
        corpus = r["corpus"]
        corpus_dir = DATASETS / corpus
        split_map = load_split_map(corpus_dir)
        sid = subject_for_recording(corpus, r["recording_id"])
        split = split_map.get(sid, "")
        bucket = by_corpus.setdefault(
            corpus,
            {
                "units_as_stored": UNITS_AS_STORED[corpus],
                "thresholds": {
                    "flat_std": FLAT_STD[corpus],
                    "peak_abs": PEAK_ABS[corpus],
                    "z_extreme": Z_EXTREME,
                    "line_noise_frac": LINE_NOISE_FRAC if LINE_NOISE_ENABLE else None,
                },
                "n_total": 0,
                "n_pass": 0,
                "n_drop": 0,
                "reason_counts": defaultdict(int),
                "by_split": {},
                "by_label": {},
                "recordings": [],
            },
        )
        bucket["n_total"] += r["n_total"]
        bucket["n_pass"] += r["n_pass"]
        bucket["n_drop"] += r["n_drop"]
        for k, v in r["reason_counts"].items():
            bucket["reason_counts"][k] += v

        sp = bucket["by_split"].setdefault(
            split or "unknown", {"n_total": 0, "n_pass": 0, "n_drop": 0}
        )
        sp["n_total"] += r["n_total"]
        sp["n_pass"] += r["n_pass"]
        sp["n_drop"] += r["n_drop"]

        for lab, d in r["drop_by_label"].items():
            lb = bucket["by_label"].setdefault(lab, {"n": 0, "pass": 0, "drop": 0})
            lb["n"] += d["n"]
            lb["pass"] += d["pass"]
            lb["drop"] += d["drop"]

        rec_summary = {
            k: r[k]
            for k in (
                "recording_id",
                "npz_path",
                "qc_npz_path",
                "n_total",
                "n_pass",
                "n_drop",
                "drop_rate",
                "reason_counts",
                "drop_by_label",
            )
        }
        rec_summary["subject_id"] = sid
        rec_summary["split"] = split
        bucket["recordings"].append(rec_summary)

        for i in range(r["n_total"]):
            yi = int(r["y"][i])
            lab = r["label_names"][yi] if yi < len(r["label_names"]) else str(yi)
            csv_rows_by_corpus[corpus].append(
                {
                    "corpus": corpus,
                    "subject_id": sid,
                    "recording_id": r["recording_id"],
                    "window_index": i,
                    "split": split,
                    "y_head_a": lab,
                    "qc_pass": int(bool(r["qc_pass"][i])),
                    "qc_reason": r["qc_reason"][i],
                }
            )

    summary_out: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "light_obvious_junk_only",
        "corpora": {},
    }
    for corpus, bucket in by_corpus.items():
        bucket["drop_rate"] = (
            bucket["n_drop"] / bucket["n_total"] if bucket["n_total"] else 0.0
        )
        bucket["reason_counts"] = dict(bucket["reason_counts"])
        for sp, d in bucket["by_split"].items():
            d["drop_rate"] = d["n_drop"] / d["n_total"] if d["n_total"] else 0.0
        for lab, d in bucket["by_label"].items():
            d["drop_rate"] = d["drop"] / d["n"] if d["n"] else 0.0

        ann_dir = DATASETS / corpus / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        csv_path = ann_dir / "windows_qc.csv"
        rows = csv_rows_by_corpus[corpus]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "corpus",
                    "subject_id",
                    "recording_id",
                    "window_index",
                    "split",
                    "y_head_a",
                    "qc_pass",
                    "qc_reason",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        bucket["windows_qc_csv"] = str(csv_path.relative_to(ROOT))
        summary_out["corpora"][corpus] = bucket

    (EXPORTS / "summary.json").write_text(json.dumps(summary_out, indent=2))
    return summary_out


def write_docs(summary: Dict[str, Any]) -> Path:
    DOCS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Light artifact QC (windows)",
        "",
        "**Script:** `scripts/dataset/artifact_qc_windows.py`",
        "",
        "Conservative obvious-junk filters only — not a full EEG artifact pipeline.",
        "Does **not** rescale units; thresholds are corpus-specific for units as stored.",
        "",
        "## Units as stored",
        "",
        "| Corpus | Units | Notes |",
        "|--------|-------|-------|",
        "| `vigilance_sleep_edf` | approx uV | Sleep-EDF Muse-proxy windows |",
        "| `attention_ds001787` | approx V | OpenNeuro BioSemi to Muse-proxy, unscaled |",
        "| `attention_ds003969` | approx V | OpenNeuro Muse-native scale as exported |",
        "| `attention_ds001787_crown8` | approx V | Same BDFs, Crown8 montage |",
        "| `attention_ds003969_crown8` | approx V | Same BDFs, Crown8 montage |",
        "",
        "## Thresholds",
        "",
        "| Check | Vigilance (uV) | Attention (V) | Action |",
        "|-------|----------------|---------------|--------|",
        "| NaN / Inf | any | any | reject `nan_inf` |",
        f"| Flat / near-zero channel std | < {FLAT_STD['vigilance_sleep_edf']} | < {FLAT_STD['attention_ds001787']} | reject `flat` |",
        f"| Extreme peak abs(x) | > {PEAK_ABS['vigilance_sleep_edf']} | > {PEAK_ABS['attention_ds001787']} | reject `peak_abs` |",
        f"| Within-window z (per ch) | abs(z) > {Z_EXTREME} | same | reject `peak_z` |",
        f"| Line-noise proxy (48-52 U 58-62 Hz power frac) | >= {LINE_NOISE_FRAC} | same | reject `line_noise` |",
        "",
        "First-fail reason is stored; windows may fail multiple checks but only one reason is recorded.",
        "",
        "## Outputs",
        "",
        "- `datasets/<corpus>/windows/<recording>_qc.npz` — `qc_pass`, `qc_reason`, aligned `y`/`starts`",
        "- `datasets/<corpus>/annotations/windows_qc.csv` — per-window mask",
        "- `exports/artifact_qc_light/summary.json` — drop rates",
        "",
        "## Drop-rate summary",
        "",
    ]
    for corpus, bucket in summary["corpora"].items():
        lines.append(f"### {corpus}")
        lines.append("")
        lines.append(
            f"- total={bucket['n_total']} pass={bucket['n_pass']} "
            f"drop={bucket['n_drop']} drop_rate={bucket['drop_rate']:.4f}"
        )
        lines.append(f"- reasons: `{bucket['reason_counts']}`")
        lines.append("- by split:")
        for sp, d in sorted(bucket["by_split"].items()):
            lines.append(
                f"  - **{sp}**: n={d['n_total']} drop={d['n_drop']} "
                f"rate={d['drop_rate']:.4f}"
            )
        lines.append("- by label:")
        for lab, d in sorted(bucket["by_label"].items()):
            lines.append(
                f"  - **{lab}**: n={d['n']} drop={d['drop']} "
                f"rate={d['drop_rate']:.4f}"
            )
        lines.append("")
    lines.append(f"_Generated {summary['created_utc']}_")
    path = DOCS / "artifact_qc_light.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", nargs="*", default=CORPORA)
    args = ap.parse_args()
    results: List[Dict[str, Any]] = []
    for corpus in args.corpora:
        win_dir = DATASETS / corpus / "windows"
        if not win_dir.exists():
            print(f"SKIP missing {win_dir}")
            continue
        npzs = sorted(
            p
            for p in win_dir.glob("*_windows.npz")
            if not p.name.endswith("_qc.npz")
        )
        print(f"== {corpus}: {len(npzs)} npz")
        for p in npzs:
            r = qc_file(p, corpus)
            print(
                f"  {r['recording_id']}: n={r['n_total']} "
                f"drop={r['n_drop']} ({r['drop_rate']:.3%}) "
                f"reasons={r['reason_counts']}"
            )
            results.append(r)
    summary = summarize_and_write(results)
    docs = write_docs(summary)
    print("WROTE", EXPORTS / "summary.json")
    print("WROTE", docs)
    for corpus, b in summary["corpora"].items():
        print(
            f"SUMMARY {corpus}: drop_rate={b['drop_rate']:.4f} "
            f"({b['n_drop']}/{b['n_total']})"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Classical band-math vs frozen AI heads — same windows / same splits."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]  # muse-eeg-heads
BM = ROOT / "band_math"
sys.path.insert(0, str(BM))

from features.flutter_bands import (  # noqa: E402
    APP_FEATURE_IDS,
    compute_window_features_fast,
    feature_names,
)

RESULTS = BM / "results"
SUMMARIES = BM / "summaries"
RESULTS.mkdir(parents=True, exist_ok=True)
SUMMARIES.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_TRAIN_BALANCED = 80000  # match AI head train undersample cap


def undersample_balanced(X, y, cap=MAX_TRAIN_BALANCED, rng=None):
    rng = np.random.default_rng(SEED if rng is None else rng)
    y = np.asarray(y)
    classes = np.unique(y)
    per = min(cap // max(len(classes), 1), *(int((y == c).sum()) for c in classes))
    idx = []
    for c in classes:
        cand = np.where(y == c)[0]
        pick = rng.choice(cand, size=per, replace=False)
        idx.append(pick)
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return X[idx], y[idx]



def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def load_json(p: Path) -> Any:
    return json.loads(p.read_text())


def decision(ai: float, band: float, eps: float = 0.01) -> str:
    if band > ai + eps:
        return "win"
    if band < ai - eps:
        return "lose"
    return "tie"


def fit_eval_logistic(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    X_va: Optional[np.ndarray] = None,
    y_va: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Standardize + multinomial LR; pick C on val if provided else train."""
    # drop rows with nan
    def clean(X, y):
        m = np.isfinite(X).all(axis=1) & np.isfinite(y)
        return X[m], y[m]

    X_tr, y_tr = clean(X_tr, y_tr)
    X_te, y_te = clean(X_te, y_te)
    if len(y_tr) > MAX_TRAIN_BALANCED:
        X_tr, y_tr = undersample_balanced(X_tr, y_tr, MAX_TRAIN_BALANCED)
    if len(y_tr) < 10 or len(np.unique(y_tr)) < 2:
        return {
            "error": "insufficient_train",
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "macro_f1": 0.0,
            "accuracy": 0.0,
        }
    scaler = StandardScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_tes = scaler.transform(X_te)
    best = None
    best_score = -1.0
    Cs = [0.1, 1.0, 10.0]
    if X_va is not None and y_va is not None:
        X_va, y_va = clean(X_va, y_va)
        X_vas = scaler.transform(X_va) if len(y_va) else None
    else:
        X_vas, y_va = None, None
    for C in Cs:
        clf = LogisticRegression(
            C=C,
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
            solver="lbfgs",
        )
        clf.fit(X_trs, y_tr)
        if X_vas is not None and len(y_va) >= 2 and len(np.unique(y_va)) >= 2:
            score = macro_f1(y_va, clf.predict(X_vas))
        else:
            score = macro_f1(y_tr, clf.predict(X_trs))
        if score > best_score:
            best_score = score
            best = clf
    assert best is not None
    pred = best.predict(X_tes)
    return {
        "model": "logistic_balanced",
        "C": float(best.C),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_features": int(X_tr.shape[1]),
        "macro_f1": macro_f1(y_te, pred),
        "accuracy": float(accuracy_score(y_te, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
        "val_select_macro_f1": float(best_score),
    }


def fit_eval_threshold(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    higher_means_class1: bool = True,
) -> Dict[str, Any]:
    """Single-feature threshold: class1 if x > t (or <). Sweep percentiles on train."""
    mtr = np.isfinite(x_tr) & np.isfinite(y_tr)
    mte = np.isfinite(x_te) & np.isfinite(y_te)
    x_tr, y_tr = x_tr[mtr], y_tr[mtr]
    x_te, y_te = x_te[mte], y_te[mte]
    if len(y_tr) < 10 or len(np.unique(y_tr)) < 2:
        return {"error": "insufficient", "macro_f1": 0.0, "accuracy": 0.0}
    qs = np.linspace(5, 95, 37)
    thr_cands = np.unique(np.percentile(x_tr, qs))
    best_f1, best_t, best_flip = -1.0, float(np.median(x_tr)), False
    for t in thr_cands:
        for flip in (False, True):
            if flip:
                pred = (x_tr <= t).astype(int)
            else:
                pred = (x_tr > t).astype(int)
            f = macro_f1(y_tr, pred)
            if f > best_f1:
                best_f1, best_t, best_flip = f, float(t), flip
    if best_flip:
        pred_te = (x_te <= best_t).astype(int)
    else:
        pred_te = (x_te > best_t).astype(int)
    return {
        "model": "single_feature_threshold",
        "threshold": best_t,
        "flip": best_flip,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "train_macro_f1": float(best_f1),
        "macro_f1": macro_f1(y_te, pred_te),
        "accuracy": float(accuracy_score(y_te, pred_te)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, pred_te)),
    }


def stack_features(feat: Dict[str, np.ndarray], keys: List[str]) -> np.ndarray:
    return np.column_stack([feat[k] for k in keys])


def extract_npz_features(npz_path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Any]]:
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]
    y = z["y"].astype(np.int64)
    meta = {"label_names": [str(x) for x in z["label_names"]] if "label_names" in z.files else None}
    if "stage_coarse" in z.files:
        meta["stage_coarse"] = np.asarray([str(s) for s in z["stage_coarse"]])
    if "stage_raw" in z.files:
        meta["stage_raw"] = np.asarray([str(s) for s in z["stage_raw"]])
    feat = compute_window_features_fast(X)
    return feat, y, meta


# ---------- task loaders ----------


def vig_recording_splits():
    """Map recording_id -> split from fixed subject JSON (full 124-subj pool)."""
    corp = ROOT / "datasets/vigilance_sleep_edf"
    out = {}
    for sp in ("train", "val", "test"):
        doc = load_json(corp / f"splits/{sp}_subjects.json")
        for rid in doc.get("recordings", []):
            out[rid] = sp
    return out


def _resolve_vig_split(rid, rec_split):
    if rid in rec_split:
        return rec_split[rid]
    return "train"


def load_vig_a() -> Dict[str, Any]:
    """A-vig drowsy/hypnagogic on vigilance_sleep_edf (all on-disk windows)."""
    corp = ROOT / "datasets/vigilance_sleep_edf"
    rec_split = vig_recording_splits()
    cache = RESULTS / "vig_features_cache.npz"
    if cache.exists():
        print("[a_vig] loading feature cache", cache)
        c = np.load(cache, allow_pickle=True)
        return {
            "task": "a_vig",
            "dataset": "Sleep-EDF Expanded + HMC (2→4 muse4 proxy, professional)",
            "labels": ["drowsy", "hypnagogic"],
            "ai_macro_f1": 0.7469416004366879,
            "ai_source": "exports/head_a_vig_full_corpus/metrics_summary.json",
            "feat_keys": list(c["feat_keys"]),
            "F": c["F"],
            "y": c["y"],
            "split": c["split"],
            "rec_id": c["rec_id"],
        }

    feat_keys = feature_names()
    Fs, ys, splits, rids = [], [], [], []
    npzs = sorted((corp / "windows").glob("*_windows.npz"))
    for i, npz in enumerate(npzs):
        rid = npz.stem.replace("_windows", "")
        sp = _resolve_vig_split(rid, rec_split)
        feat, y, _ = extract_npz_features(npz)
        F = stack_features(feat, feat_keys)
        Fs.append(F)
        ys.append(y)
        splits.extend([sp] * len(y))
        rids.extend([rid] * len(y))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[a_vig] {i+1}/{len(npzs)} {rid} split={sp} n={len(y)}")
    F = np.concatenate(Fs, axis=0)
    y = np.concatenate(ys, axis=0)
    split = np.asarray(splits)
    rec_id = np.asarray(rids)
    np.savez_compressed(
        cache,
        F=F,
        y=y,
        split=split,
        rec_id=rec_id,
        feat_keys=np.asarray(feat_keys),
    )
    print("[a_vig] cached", cache, "N", len(y), {s: int((split == s).sum()) for s in ("train", "val", "test")})
    return {
        "task": "a_vig",
        "dataset": "Sleep-EDF Expanded + HMC (2→4 muse4 proxy, professional)",
        "labels": ["drowsy", "hypnagogic"],
        "ai_macro_f1": 0.7469416004366879,
        "ai_source": "exports/head_a_vig_full_corpus/metrics_summary.json",
        "feat_keys": feat_keys,
        "F": F,
        "y": y,
        "split": split,
        "rec_id": rec_id,
    }


def load_head_c() -> Dict[str, Any]:
    """Head C wake/light from same vig windows via stage_coarse."""
    corp = ROOT / "datasets/vigilance_sleep_edf"
    rec_split = vig_recording_splits()
    cache = RESULTS / "head_c_features_cache.npz"
    if cache.exists():
        print("[head_c] loading feature cache", cache)
        c = np.load(cache, allow_pickle=True)
        return {
            "task": "head_c_wake_light",
            "dataset": "Sleep-EDF Expanded + HMC (2→4 muse4 proxy, professional)",
            "labels": ["wake", "light"],
            "ai_macro_f1": 0.759732696722526,
            "ai_source": "exports/head_c_full_corpus/metrics_summary.json",
            "feat_keys": list(c["feat_keys"]),
            "F": c["F"],
            "y": c["y"],
            "split": c["split"],
            "rec_id": c["rec_id"],
        }
    feat_keys = feature_names()
    Fs, ys, splits, rids = [], [], [], []
    label_map = {"wake": 0, "light": 1}
    npzs = sorted((corp / "windows").glob("*_windows.npz"))
    for i, npz in enumerate(npzs):
        rid = npz.stem.replace("_windows", "")
        sp = _resolve_vig_split(rid, rec_split)
        feat, _, meta = extract_npz_features(npz)
        sc = meta.get("stage_coarse")
        if sc is None:
            continue
        keep = np.array([s in label_map for s in sc])
        if not keep.any():
            continue
        F = stack_features(feat, feat_keys)[keep]
        y = np.array([label_map[s] for s in sc[keep]], dtype=np.int64)
        Fs.append(F)
        ys.append(y)
        splits.extend([sp] * len(y))
        rids.extend([rid] * len(y))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[head_c] {i+1}/{len(npzs)} {rid} split={sp} n={len(y)}")
    F = np.concatenate(Fs, axis=0)
    y = np.concatenate(ys, axis=0)
    split = np.asarray(splits)
    rec_id = np.asarray(rids)
    np.savez_compressed(
        cache, F=F, y=y, split=split, rec_id=rec_id, feat_keys=np.asarray(feat_keys)
    )
    print("[head_c] cached N", len(y), {s: int((split == s).sum()) for s in ("train", "val", "test")})
    return {
        "task": "head_c_wake_light",
        "dataset": "Sleep-EDF Expanded + HMC (2→4 muse4 proxy, professional)",
        "labels": ["wake", "light"],
        "ai_macro_f1": 0.759732696722526,
        "ai_source": "exports/head_c_full_corpus/metrics_summary.json",
        "feat_keys": feat_keys,
        "F": F,
        "y": y,
        "split": split,
        "rec_id": rec_id,
    }


def subject_from_pack(name: str) -> str:
    # SC4001 -> SC400; sub001_ses01 -> sub001; ds007169_sub-001 -> barras style handled elsewhere
    stem = Path(name).stem.replace("_windows", "").replace("_aeng_windows", "")
    if stem.startswith("SC"):
        return stem[:5]  # SC400
    if stem.startswith("SN"):
        return stem[:5]
    if "sub-" in stem:
        # ds007169_sub-001
        parts = stem.split("sub-")
        return "sub-" + parts[-1].split("_")[0]
    if stem.startswith("sub"):
        return stem.split("_")[0]
    return stem


def load_aeng() -> Dict[str, Any]:
    corp = ROOT / "datasets/engagement_a_eng"
    splits_doc = load_json(corp / "splits/splits.json")
    # persons -> split
    person_split = {}
    for sp, persons in splits_doc["splits"].items():
        for p in persons:
            person_split[p] = sp
    # map pack -> person from splits_doc['persons']
    pack_to_person = {}
    for person, info in splits_doc.get("persons", {}).items():
        for pack in info.get("packs", []):
            pack_to_person[pack] = person

    feat_keys = feature_names()
    Fs, ys, splits, rids = [], [], [], []
    npzs = sorted((corp / "windows").rglob("*windows.npz"))
    # also check exports/windows_aeng
    if not npzs:
        npzs = sorted((ROOT / "exports/windows_aeng").rglob("*windows.npz"))
    for i, npz in enumerate(npzs):
        # pack id heuristics
        stem = npz.stem.replace("_aeng_windows", "").replace("_windows", "")
        # try match pack keys
        person = None
        for pack, pers in pack_to_person.items():
            if pack in stem or stem.startswith(pack) or pack in npz.name:
                person = pers
                break
        if person is None:
            # fallback: unique person from parent folder + stem
            person = f"{npz.parent.name}/{stem}"
        sp = person_split.get(person)
        if sp is None:
            # try barras_sub-XXX
            if "sub-" in stem:
                sid = "sub-" + stem.split("sub-")[-1].split("_")[0]
                for prefix in ("barras_", ""):
                    cand = prefix + sid if prefix else sid
                    # search
                    for p, info in splits_doc.get("persons", {}).items():
                        if sid in p or p.endswith(sid):
                            person = p
                            sp = person_split.get(p)
                            break
                    if sp:
                        break
        if sp is None:
            # last resort: put in train if unknown (shouldn't)
            print("[a_eng] unknown split for", npz.name, "person", person)
            sp = "train"
        feat, y, meta = extract_npz_features(npz)
        # ensure labels low=0 high=1
        names = meta.get("label_names") or ["low_engagement", "high_engagement"]
        if names[0] not in ("low_engagement", "low"):
            # remap if needed
            pass
        F = stack_features(feat, feat_keys)
        Fs.append(F)
        ys.append(y)
        splits.extend([sp] * len(y))
        rids.extend([stem] * len(y))
        if (i + 1) % 20 == 0:
            print(f"[a_eng] {i+1}/{len(npzs)}")
    F = np.concatenate(Fs, axis=0)
    y = np.concatenate(ys, axis=0)
    return {
        "task": "a_eng",
        "dataset": "engagement_a_eng union (mix 19/23/32-ch professional + STEW 14-ch Emotiv hobbyist)",
        "labels": ["low_engagement", "high_engagement"],
        "ai_macro_f1": 0.5481672654749781,
        "ai_macro_f1_reve": 0.590,
        "ai_source": "exports/head_a_eng_train_cbramod/metrics_summary.json",
        "feat_keys": feat_keys,
        "F": F,
        "y": y,
        "split": np.asarray(splits),
        "rec_id": np.asarray(rids),
    }


def load_attention(corp_name: str, ai_f1: float) -> Dict[str, Any]:
    corp = ROOT / "datasets" / corp_name
    feat_keys = feature_names()
    # subject splits
    subj_split = {}
    for sp in ("train", "val", "test"):
        doc = load_json(corp / f"splits/{sp}_subjects.json")
        for s in doc["subjects"]:
            # normalize sub-001 vs sub001
            subj_split[s] = sp
            subj_split[s.replace("-", "")] = sp
            if s.startswith("sub-"):
                subj_split["sub" + s[4:].zfill(3) if False else "sub" + s.split("-")[1]] = sp
    Fs, ys, splits, rids = [], [], [], []
    npzs = sorted((corp / "windows").glob("*windows.npz"))
    for npz in npzs:
        stem = npz.stem.replace("_windows", "")
        # sub001_ses01 or sub001
        if "_ses" in stem:
            sid_raw = stem.split("_ses")[0]
        else:
            sid_raw = stem
        # map to sub-XXX
        if sid_raw.startswith("sub") and not sid_raw.startswith("sub-"):
            num = sid_raw[3:]
            sid = f"sub-{num.zfill(3)}" if num.isdigit() else f"sub-{num}"
        else:
            sid = sid_raw
        sp = subj_split.get(sid) or subj_split.get(sid_raw) or subj_split.get(sid.replace("-", ""))
        if sp is None:
            # try subjects list keys
            for k, v in subj_split.items():
                if num_in(k) == num_in(sid):
                    sp = v
                    break
        if sp is None:
            print(f"[{corp_name}] unknown split", sid, npz.name)
            sp = "train"
        feat, y, _ = extract_npz_features(npz)
        F = stack_features(feat, feat_keys)
        Fs.append(F)
        ys.append(y)
        splits.extend([sp] * len(y))
        rids.extend([stem] * len(y))
    dataset_name = {
        "attention_ds001787": "ds001787 (64-ch BioSemi→muse4 proxy, professional)",
        "attention_ds003969": "ds003969 (64-ch Muse-proximal muse4, professional)",
    }[corp_name]
    return {
        "task": f"mw_{corp_name}",
        "dataset": dataset_name,
        "labels": ["concentration", "mind_wandering"],
        "ai_macro_f1": ai_f1,
        "ai_source": "docs/dataset_confidence_table.md (LOSO ~0.36)",
        "feat_keys": feat_keys,
        "F": np.concatenate(Fs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "split": np.asarray(splits),
        "rec_id": np.asarray(rids),
    }


def num_in(s: str) -> str:
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits.lstrip("0") or "0"


def load_med() -> Dict[str, Any]:
    win_dir = ROOT / "exports/head_a_med_smoke/windows"
    feat_keys = feature_names()
    Fs, ys, splits, rids = [], [], [], []
    npzs = sorted(win_dir.glob("*windows.npz"))
    # small n=6 subjects — LOSO-style: leave one subject out as test rotating, report mean; also fixed holdout if in metrics
    subjects = []
    for npz in npzs:
        stem = npz.stem.replace("_amed_windows", "").replace("_windows", "")
        subjects.append(stem)
        feat, y, _ = extract_npz_features(npz)
        F = stack_features(feat, feat_keys)
        Fs.append(F)
        ys.append(y)
        rids.append(np.array([stem] * len(y)))
    # Use LOSO: for each held-out subject, train on rest, avg test F1
    return {
        "task": "a_med_rest_meditation",
        "dataset": "ds003816 (Muse-native AF7/AF8/TP9/TP10, professional)",
        "labels": ["rest", "meditation"],
        "ai_macro_f1": 0.333,
        "ai_source": "exports/head_a_med_smoke/metrics_summary.json (~0.33)",
        "feat_keys": feat_keys,
        "F_list": Fs,
        "y_list": ys,
        "subjects": subjects,
        "loso": True,
    }


def load_depth() -> Dict[str, Any]:
    win_dir = ROOT / "exports/head_a_med_depth_smoke/windows"
    feat_keys = feature_names()
    meta = load_json(ROOT / "exports/head_a_med_depth_smoke/metrics_summary.json")
    holdout = meta.get("holdout_subject", "sub-020").replace("-", "")
    # holdout_tag sub020_ses01
    holdout_tag = meta.get("holdout_tag", "sub020_ses01")
    Fs, ys, splits, rids = [], [], [], []
    npzs = sorted(win_dir.glob("*windows.npz")) if win_dir.exists() else []
    if not npzs:
        return {"task": "med_depth_q1", "gap": f"missing windows at {win_dir}", "ai_macro_f1": 0.333}
    for npz in npzs:
        stem = npz.stem.replace("_windows", "")
        sp = "test" if holdout_tag in stem or holdout in stem.replace("-", "") else "train"
        feat, y, _ = extract_npz_features(npz)
        F = stack_features(feat, feat_keys)
        Fs.append(F)
        ys.append(y)
        splits.extend([sp] * len(y))
        rids.extend([stem] * len(y))
    return {
        "task": "med_depth_q1",
        "dataset": "ds001787 (64-ch BioSemi, professional) — muse4 proxy; Q1≤1 vs Q1≥2",
        "labels": ["meditation_depth_low", "meditation_depth_high"],
        "ai_macro_f1": 0.333,
        "ai_source": "exports/head_a_med_depth_smoke/metrics_summary.json",
        "feat_keys": feat_keys,
        "F": np.concatenate(Fs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "split": np.asarray(splits),
        "rec_id": np.asarray(rids),
    }


def eval_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    if bundle.get("gap"):
        return {
            "task": bundle["task"],
            "gap": bundle["gap"],
            "ai_macro_f1": bundle.get("ai_macro_f1"),
            "band_macro_f1": None,
            "decision": "gap",
        }
    if bundle.get("loso"):
        scores = []
        app_scores = {fid: [] for fid in APP_FEATURE_IDS}
        feat_keys = bundle["feat_keys"]
        for i, subj in enumerate(bundle["subjects"]):
            X_te = bundle["F_list"][i]
            y_te = bundle["y_list"][i]
            X_tr = np.concatenate([bundle["F_list"][j] for j in range(len(bundle["subjects"])) if j != i], axis=0)
            y_tr = np.concatenate([bundle["y_list"][j] for j in range(len(bundle["subjects"])) if j != i], axis=0)
            # logistic on app + extras
            keys_idx = list(range(len(feat_keys)))
            r = fit_eval_logistic(X_tr, y_tr, X_te, y_te)
            scores.append(r["macro_f1"])
            for fid in APP_FEATURE_IDS:
                j = feat_keys.index(fid)
                t = fit_eval_threshold(X_tr[:, j], y_tr, X_te[:, j], y_te)
                app_scores[fid].append(t["macro_f1"])
        band_f1 = float(np.mean(scores)) if scores else 0.0
        per_feat = {k: float(np.mean(v)) if v else 0.0 for k, v in app_scores.items()}
        best_single = max(per_feat, key=per_feat.get) if per_feat else None
        best_single_f1 = per_feat.get(best_single, 0.0) if best_single else 0.0
        use_f1 = max(band_f1, best_single_f1)
        method = "loso_logistic" if band_f1 >= best_single_f1 else f"loso_threshold:{best_single}"
        return {
            "task": bundle["task"],
            "dataset": bundle["dataset"],
            "labels": bundle["labels"],
            "ai_macro_f1": bundle["ai_macro_f1"],
            "ai_source": bundle.get("ai_source"),
            "band_macro_f1": use_f1,
            "band_logistic_macro_f1": band_f1,
            "band_best_single": best_single,
            "band_best_single_macro_f1": best_single_f1,
            "per_app_feature_macro_f1": per_feat,
            "method": method,
            "n_subjects_loso": len(bundle["subjects"]),
            "decision": decision(bundle["ai_macro_f1"], use_f1),
        }

    F, y, split = bundle["F"], bundle["y"], bundle["split"]
    feat_keys = bundle["feat_keys"]
    tr = split == "train"
    va = split == "val"
    te = split == "test"
    # if no val, use train for selection
    X_tr, y_tr = F[tr], y[tr]
    X_va, y_va = (F[va], y[va]) if va.any() else (None, None)
    X_te, y_te = F[te], y[te]
    if not te.any():
        return {
            "task": bundle["task"],
            "gap": "no test split windows",
            "ai_macro_f1": bundle.get("ai_macro_f1"),
            "decision": "gap",
        }

    print(f"[{bundle['task']}] fit train={len(y_tr)} val={0 if y_va is None else len(y_va)} test={len(y_te)}", flush=True)
    if len(y_tr) > MAX_TRAIN_BALANCED:
        X_tr, y_tr = undersample_balanced(X_tr, y_tr, MAX_TRAIN_BALANCED)
        print(f"[{bundle['task']}] undersampled train -> {len(y_tr)}", flush=True)
    if X_va is not None and len(y_va) > MAX_TRAIN_BALANCED:
        X_va, y_va = undersample_balanced(X_va, y_va, MAX_TRAIN_BALANCED // 2)

    log_r = fit_eval_logistic(X_tr, y_tr, X_te, y_te, X_va, y_va)
    print(f"[{bundle['task']}] logistic macro-F1={log_r['macro_f1']:.4f}", flush=True)
    per_feat = {}
    for fid in APP_FEATURE_IDS:
        j = feat_keys.index(fid)
        per_feat[fid] = fit_eval_threshold(X_tr[:, j], y_tr, X_te[:, j], y_te)
    # also try extras as single
    for fid in ("extra.theta_beta", "extra.delta_theta", "extra.alpha_beta"):
        if fid in feat_keys:
            j = feat_keys.index(fid)
            per_feat[fid] = fit_eval_threshold(X_tr[:, j], y_tr, X_te[:, j], y_te)

    best_single = max(per_feat.keys(), key=lambda k: per_feat[k]["macro_f1"])
    best_single_f1 = per_feat[best_single]["macro_f1"]
    band_f1 = max(log_r["macro_f1"], best_single_f1)
    method = "logistic_all_features" if log_r["macro_f1"] >= best_single_f1 else f"threshold:{best_single}"

    out = {
        "task": bundle["task"],
        "dataset": bundle["dataset"],
        "labels": bundle["labels"],
        "ai_macro_f1": bundle["ai_macro_f1"],
        "ai_source": bundle.get("ai_source"),
        "band_macro_f1": band_f1,
        "band_logistic_macro_f1": log_r["macro_f1"],
        "logistic_detail": {k: log_r[k] for k in log_r if k != "model"},
        "band_best_single": best_single,
        "band_best_single_macro_f1": best_single_f1,
        "per_app_feature_macro_f1": {k: v["macro_f1"] for k, v in per_feat.items()},
        "method": method,
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "decision": decision(bundle["ai_macro_f1"], band_f1),
    }
    if "ai_macro_f1_reve" in bundle:
        out["ai_macro_f1_reve"] = bundle["ai_macro_f1_reve"]
        out["decision_vs_reve"] = decision(bundle["ai_macro_f1_reve"], band_f1)
    return out


def main() -> None:
    print("band_math compare starting", utc_now())
    results = []
    gaps = []

    print("=== A-vig ===")
    vig = load_vig_a()
    r = eval_bundle(vig)
    results.append(r)
    (RESULTS / "a_vig.json").write_text(json.dumps(r, indent=2))
    print("A-vig band", r.get("band_macro_f1"), "AI", r.get("ai_macro_f1"), r.get("decision"))

    print("=== Head C ===")
    hc = load_head_c()
    r = eval_bundle(hc)
    results.append(r)
    (RESULTS / "head_c.json").write_text(json.dumps(r, indent=2))
    print("Head C band", r.get("band_macro_f1"), "AI", r.get("ai_macro_f1"), r.get("decision"))

    print("=== A-eng ===")
    try:
        eng = load_aeng()
        r = eval_bundle(eng)
        results.append(r)
        (RESULTS / "a_eng.json").write_text(json.dumps(r, indent=2))
        print("A-eng band", r.get("band_macro_f1"), "AI", r.get("ai_macro_f1"), r.get("decision"))
    except Exception as e:
        gaps.append({"task": "a_eng", "error": str(e)})
        print("A-eng ERROR", e)

    print("=== MW ds001787 ===")
    try:
        att = load_attention("attention_ds001787", 0.36)
        r = eval_bundle(att)
        results.append(r)
        (RESULTS / "mw_ds001787.json").write_text(json.dumps(r, indent=2))
        print("MW1787", r.get("band_macro_f1"), r.get("decision"))
    except Exception as e:
        gaps.append({"task": "mw_ds001787", "error": str(e)})
        print("MW1787 ERROR", e)

    print("=== MW ds003969 ===")
    try:
        att = load_attention("attention_ds003969", 0.36)
        r = eval_bundle(att)
        results.append(r)
        (RESULTS / "mw_ds003969.json").write_text(json.dumps(r, indent=2))
        print("MW3969", r.get("band_macro_f1"), r.get("decision"))
    except Exception as e:
        gaps.append({"task": "mw_ds003969", "error": str(e)})
        print("MW3969 ERROR", e)

    print("=== A-med ===")
    try:
        med = load_med()
        r = eval_bundle(med)
        results.append(r)
        (RESULTS / "a_med.json").write_text(json.dumps(r, indent=2))
        print("A-med", r.get("band_macro_f1"), r.get("decision"))
    except Exception as e:
        gaps.append({"task": "a_med", "error": str(e)})
        print("A-med ERROR", e)

    print("=== depth ===")
    try:
        depth = load_depth()
        r = eval_bundle(depth)
        results.append(r)
        (RESULTS / "med_depth.json").write_text(json.dumps(r, indent=2))
        print("depth", r.get("band_macro_f1"), r.get("decision"), r.get("gap"))
    except Exception as e:
        gaps.append({"task": "med_depth", "error": str(e)})
        print("depth ERROR", e)

    summary = {
        "created_utc": utc_now(),
        "protocol": {
            "fft": "256 @ 256 Hz (Flutter muse.rs)",
            "bands_hz": {"delta": "1-4", "theta": "4-8", "alpha": "8-13", "beta": "13-30", "gamma": "30-50"},
            "pads": ["AF7", "AF8"],
            "window": "2 s muse4 npz; average two 1 s FFT halves",
            "metric": "macro-F1 on AI test subjects (same splits)",
            "models": ["LogisticRegression balanced on all band features", "single-feature threshold sweep"],
            "ai_heads_not_retrained": True,
        },
        "tasks": results,
        "gaps": gaps,
    }
    (SUMMARIES / "compare_summary.json").write_text(json.dumps(summary, indent=2))
    (BM / "compare_summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", BM / "compare_summary.json")


if __name__ == "__main__":
    main()

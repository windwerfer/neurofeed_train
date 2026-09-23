#!/usr/bin/env python3
"""Grow Head A attention toward ~80: +40 Muse-proximal ds003969 (CC0).

Local-first; paced OpenNeuro; no Kaggle; no HF EEGMeditation.
New subjects → train (frozen test holdouts untouched).
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ds003969_attention_ingest as base  # noqa: E402

STEP = "expand_attention_toward80_ds003969"

# +11 htr (014–024) + 20 ctr (036–055) + 4 tm (056–059) + 5 vip (060–064) = 40
SUBJECTS: List[Dict[str, str]] = [
    {"sub": "014", "group": "htr", "first_session": "thinking", "tree": "fe8044d114bb473a9931eaa3d36d20d412d7a19c"},
    {"sub": "015", "group": "htr", "first_session": "meditation", "tree": "dec07e169adaa062cbeaefdbf4b07a16dcd9a602"},
    {"sub": "016", "group": "htr", "first_session": "thinking", "tree": "d15d6283df4cb06bdbccc9d6b3bbe0053e761998"},
    {"sub": "017", "group": "htr", "first_session": "meditation", "tree": "d109500c7979dffeafc20d4d0f3621c13e1e896f"},
    {"sub": "018", "group": "htr", "first_session": "thinking", "tree": "88a39324f5e37e2a179be8e1db8dec7bb6643e23"},
    {"sub": "019", "group": "htr", "first_session": "meditation", "tree": "50c4171df20ce61d0d84ebfa51826ef95066ba42"},
    {"sub": "020", "group": "htr", "first_session": "thinking", "tree": "edbe7793c41d02791172cbb857ed1c1be0bb81cc"},
    {"sub": "021", "group": "htr", "first_session": "meditation", "tree": "f44c345246c872209cc2b3e23f219730947ee70b"},
    {"sub": "022", "group": "htr", "first_session": "thinking", "tree": "21f5ac18c97850e096e7fa31da08a5a274ae76e8"},
    {"sub": "023", "group": "htr", "first_session": "meditation", "tree": "7a66b86e4bccc368946373e6f784c27a6f0af7a0"},
    {"sub": "024", "group": "htr", "first_session": "meditation", "tree": "b240a1cf53bac2d8be32aa0686cfe05ba7e5e844"},
    {"sub": "036", "group": "ctr", "first_session": "meditation", "tree": "b18937d208e0053170e7af06531b4c8bec1490ba"},
    {"sub": "037", "group": "ctr", "first_session": "thinking", "tree": "2cfa75454b332db306c42ca661d66f4cc45aa5ca"},
    {"sub": "038", "group": "ctr", "first_session": "meditation", "tree": "ab0d947b90df159b827dce48f992043c6987a93f"},
    {"sub": "039", "group": "ctr", "first_session": "thinking", "tree": "cd5b116ef0a1534d0ffe9a674b732a22224abde4"},
    {"sub": "040", "group": "ctr", "first_session": "meditation", "tree": "918d2c616aa306e779afbf8de88b6d893980d3be"},
    {"sub": "041", "group": "ctr", "first_session": "thinking", "tree": "e962f8309b39f17aa965a8932b1dc75ac78fd57f"},
    {"sub": "042", "group": "ctr", "first_session": "meditation", "tree": "9847db6a0c53a940b52a0c860720d88672ce8bcb"},
    {"sub": "043", "group": "ctr", "first_session": "thinking", "tree": "29927583dfa6f64e9daf6e8fe7b46034edda2914"},
    {"sub": "044", "group": "ctr", "first_session": "meditation", "tree": "3ed8d853c0f3e73630d60b7dcecb20ef559d16d1"},
    {"sub": "045", "group": "ctr", "first_session": "meditation", "tree": "717ba5c8b1ebe011d88fd0f26ccd97fb7105bd2d"},
    {"sub": "046", "group": "ctr", "first_session": "thinking", "tree": "5a66977c084e448c8fdb957e9d1f09cb90968bfd"},
    {"sub": "047", "group": "ctr", "first_session": "meditation", "tree": "dfae042f7911a18eba3f186fcb736aa3dad22543"},
    {"sub": "048", "group": "ctr", "first_session": "thinking", "tree": "9520b40809f646878158dcdd18eb35a5e45b6674"},
    {"sub": "049", "group": "ctr", "first_session": "meditation", "tree": "5ec7d6d921490318371428b542827ad1c047648c"},
    {"sub": "050", "group": "ctr", "first_session": "thinking", "tree": "35fb3e6038d0c885ff64245d6692a1fa206df338"},
    {"sub": "051", "group": "ctr", "first_session": "meditation", "tree": "231515072ff84ed884d91db4616f41c33e62812d"},
    {"sub": "052", "group": "ctr", "first_session": "thinking", "tree": "20369669b14e09c01b3062b57dcad2dbb4ea9773"},
    {"sub": "053", "group": "ctr", "first_session": "meditation", "tree": "a465198699e58cb7bb05735071bce30302823a43"},
    {"sub": "054", "group": "ctr", "first_session": "thinking", "tree": "0258ffa52bfb37f30de83259777301cfa8c81574"},
    {"sub": "055", "group": "ctr", "first_session": "meditation", "tree": "4cb62bc5ffccba7dbbd7ba345c1073ef6cfab0b7"},
    {"sub": "056", "group": "tm", "first_session": "meditation", "tree": "5940e09694d43175b53e7995f8ea7f0133da5bcc"},
    {"sub": "057", "group": "tm", "first_session": "thinking", "tree": "0186646cca8d59b7891185caa921843632554e9e"},
    {"sub": "058", "group": "tm", "first_session": "meditation", "tree": "db00774df58e3da3f541d0f353497a5634bc714e"},
    {"sub": "059", "group": "tm", "first_session": "thinking", "tree": "7cf991ead34074ada34d09e4234a4abaac950e14"},
    {"sub": "060", "group": "vip", "first_session": "meditation", "tree": "6775c4827d4214aef19c04e7b0267cfb67b470c0"},
    {"sub": "061", "group": "vip", "first_session": "thinking", "tree": "717169267d1a4a1f9d9a5215c78e469299053bde"},
    {"sub": "062", "group": "vip", "first_session": "meditation", "tree": "e5cf5e44138019cffe0996dc1a36d0207d2f3a5a"},
    {"sub": "063", "group": "vip", "first_session": "thinking", "tree": "7b3045c64c4b63b8794b351347f7b10ce24d108d"},
    {"sub": "064", "group": "vip", "first_session": "meditation", "tree": "9cd110e52b0861886a319bdc160412cf1d1ae909"},
]

PRIOR = [
    "sub001","sub002","sub003","sub004","sub005","sub006",
    "sub007","sub008","sub009","sub010","sub011","sub012","sub013",
    "sub025","sub026","sub027","sub028","sub029",
    "sub030","sub031","sub032","sub033","sub034","sub035",
]

FROZEN_TEST = ["sub-025", "sub-027"]
FROZEN_VAL = ["sub-026", "sub-028"]


def migrate_into_datasets(tag: str) -> None:
    win_dir = ROOT / "datasets/attention_ds003969/windows"
    win_dir.mkdir(parents=True, exist_ok=True)
    src_npz = base.EXPORT / tag / f"ds003969_{tag}_attention_windows.npz"
    src_man = base.EXPORT / tag / f"ds003969_{tag}_attention_manifest.json"
    for src, name in ((src_npz, f"{tag}_windows.npz"), (src_man, f"{tag}_manifest.json")):
        dst = win_dir / name
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())


def update_splits(new_tags: List[str]) -> None:
    split_dir = ROOT / "datasets/attention_ds003969/splits"
    subjects_path = split_dir / "subjects.json"
    subjects_doc = json.loads(subjects_path.read_text())
    subjects = subjects_doc.setdefault("subjects", {})
    for tag in new_tags:
        num = tag.replace("sub", "")
        sid = f"sub-{num.zfill(3)}"
        spec = next(s for s in SUBJECTS if s["sub"] == num.zfill(3) or s["sub"] == num)
        subjects[sid] = {
            "group": spec["group"],
            "recordings": [tag],
            "notes": f"toward80 {STEP}",
        }
    train = sorted(sid for sid in subjects if sid not in FROZEN_TEST and sid not in FROZEN_VAL)
    val = list(FROZEN_VAL)
    test = list(FROZEN_TEST)
    created = datetime.now(timezone.utc).isoformat()
    subjects_doc["created_utc"] = created
    subjects_path.write_text(json.dumps(subjects_doc, indent=2) + "\n")
    for split, ids in (("train", train), ("val", val), ("test", test)):
        (split_dir / f"{split}_subjects.json").write_text(
            json.dumps({
                "corpus": "attention_ds003969",
                "split": split,
                "policy": "fixed_subject_json",
                "subjects": ids,
                "created_utc": created,
                "note": f"updated by {STEP}; frozen test={FROZEN_TEST}",
            }, indent=2) + "\n"
        )
    policy = {
        "corpus": "attention_ds003969",
        "policy": "fixed_subject_json",
        "updated_utc": created,
        "updated_by": STEP,
        "splits": {"train": train, "val": val, "test": test},
        "frozen_test": FROZEN_TEST,
        "frozen_val": FROZEN_VAL,
    }
    (split_dir / "split_policy.json").write_text(json.dumps(policy, indent=2) + "\n")
    print(f"splits train={len(train)} val={len(val)} test={len(test)}", flush=True)


def main() -> None:
    base.CACHE.mkdir(parents=True, exist_ok=True)
    base.EXPORT.mkdir(parents=True, exist_ok=True)
    base.WINDOWS_PKG.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {
        "step": STEP,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prior_subjects": list(PRIOR),
        "subjects": [],
        "errors": [],
        "why": "Grow toward ~80 Head A attention subjects; Muse-proximal ds003969 CC0 only (no HF)",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "plan": "docs/head_a_40_subjects_plan.md",
        "target_attention": 80,
        "label_rule": "med*→concentration; think*→mind_wandering",
        "muse_proxy": "AF7/AF8 native; TP7/TP8→TP9/TP10",
        "id_collision_note": "Use corpus-qualified IDs for LOSO (ds001787/sub-XXX vs ds003969/sub-XXX).",
    }
    totals = Counter()
    provenance = {
        "step": STEP,
        "source": "https://openneuro.org/datasets/ds003969",
        "doi": "doi:10.18112/openneuro.ds003969.v1.0.0",
        "license_spdx": "CC0-1.0",
        "added_utc": datetime.now(timezone.utc).isoformat(),
        "recordings": [],
    }
    new_tags: List[str] = []
    for spec in SUBJECTS:
        try:
            recs = base.fetch_subject(spec)
            info = base.process_subject(spec, recs)
            man_path = Path(info["manifest"])
            man = json.loads(man_path.read_text())
            man["step"] = STEP
            man_path.write_text(json.dumps(man, indent=2) + "\n")
            shutil.copy2(man_path, base.WINDOWS_PKG / man_path.name)
            shutil.copy2(Path(info["npz"]), base.WINDOWS_PKG / Path(info["npz"]).name)
            migrate_into_datasets(info["tag"])
            results["subjects"].append(info)
            new_tags.append(info["tag"])
            for k, v in info["counts"].items():
                totals[k] += v
            for rec in recs:
                provenance["recordings"].append({
                    "name": rec["bdf"].name,
                    "subject": f"sub-{spec['sub']}",
                    "task": rec["task"],
                    "label": rec["label"],
                    "sha256": base.sha256(rec["bdf"]),
                    "bytes": rec["bdf_bytes"],
                    "url": rec["bdf_url"],
                })
            print(f"done {info['tag']}: {info['counts']} total={info['n_windows_total']}", flush=True)
        except Exception as e:
            err = {"sub": spec["sub"], "error": repr(e)}
            results["errors"].append(err)
            print(f"ERROR sub-{spec['sub']}: {e!r}", flush=True)

    results["totals_new"] = dict(totals)
    results["n_windows_new"] = int(sum(totals.values()))
    results["new_tags"] = new_tags
    if new_tags:
        update_splits(new_tags)

    pool_tags = PRIOR + new_tags
    pool_totals = Counter()
    for tag in pool_tags:
        man_p = base.EXPORT / tag / f"ds003969_{tag}_attention_manifest.json"
        if not man_p.exists():
            continue
        man = json.loads(man_p.read_text())
        for k, v in man["n_windows_per_label"].items():
            pool_totals[k] += int(v)
    results["pool_subjects"] = pool_tags
    results["pool_totals"] = dict(pool_totals)
    results["n_windows_pool"] = int(sum(pool_totals.values()))
    results["n_subjects_pool"] = len(pool_tags)
    results["attention_union_approx"] = 16 + len(pool_tags)
    results["gap_to_80_attention"] = max(0, 80 - (16 + len(pool_tags)))

    prov_path = base.CACHE / f"provenance_{STEP}.json"
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n")
    results["provenance_path"] = str(prov_path)
    summary_path = ROOT / "exports" / f"{STEP}_summary.json"
    summary_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({
        "totals_new": results["totals_new"],
        "n_windows_new": results["n_windows_new"],
        "new_tags": new_tags,
        "n_ok": len(new_tags),
        "n_err": len(results["errors"]),
        "n_subjects_pool_ds003969": results["n_subjects_pool"],
        "attention_union_approx": results["attention_union_approx"],
        "errors": results["errors"],
    }, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

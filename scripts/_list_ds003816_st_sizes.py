#!/usr/bin/env python3
"""List ds003816 st subjects PreResting+LKMSelf eeg sizes (paced)."""
from __future__ import annotations

import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DATASET = "ds003816"
TASKS = ("PreResting", "LKMSelf")
USED = {"sub-09st", "sub-22st", "sub-48st", "sub-26st", "sub-19st", "sub-23st"}
PAUSE = 1.5
OUT = Path("/tmp/ds003816_st_sizes.json")


def list_s3(prefix: str):
    url = (
        "https://s3.amazonaws.com/openneuro.org"
        f"?prefix={prefix}&delimiter=/&max-keys=1000"
    )
    data = urllib.request.urlopen(url, timeout=60).read()
    root = ET.fromstring(data)
    out = []
    for el in root:
        if el.tag.split("}")[-1] != "Contents":
            continue
        key = next(c for c in el if c.tag.split("}")[-1] == "Key").text
        size = int(next(c for c in el if c.tag.split("}")[-1] == "Size").text)
        out.append((key, size))
    return out


def main() -> None:
    url = (
        "https://s3.amazonaws.com/openneuro.org"
        "?prefix=ds003816/&delimiter=/&max-keys=1000"
    )
    data = urllib.request.urlopen(url, timeout=60).read()
    root = ET.fromstring(data)
    st = []
    for el in root:
        if el.tag.split("}")[-1] == "CommonPrefixes":
            p = next(c for c in el if c.tag.split("}")[-1] == "Prefix").text
            name = p.rstrip("/").split("/")[-1]
            if name.startswith("sub-") and name.endswith("st"):
                st.append(name)
    st = sorted(st)
    print("all st", len(st), flush=True)

    rows = []
    for sub in st:
        time.sleep(PAUSE)
        prefix = f"{DATASET}/{sub}/ses-01/eeg/"
        try:
            files = list_s3(prefix)
        except Exception as e:
            print("ERR", sub, e, flush=True)
            continue
        sizes = {}
        for key, size in files:
            fname = key.split("/")[-1]
            if not fname.endswith("_eeg.eeg"):
                continue
            for t in TASKS:
                if f"_task-{t}_" in fname:
                    sizes[t] = size
        if all(t in sizes for t in TASKS):
            total = sizes["PreResting"] + sizes["LKMSelf"]
            row = {
                "sub": sub,
                "PreResting_MB": round(sizes["PreResting"] / 1e6, 2),
                "LKMSelf_MB": round(sizes["LKMSelf"] / 1e6, 2),
                "total_MB": round(total / 1e6, 2),
                "used": sub in USED,
                "ok_5mb": sizes["PreResting"] >= 5e6 and sizes["LKMSelf"] >= 5e6,
            }
            rows.append(row)
            print(
                f"{sub}: Pre={row['PreResting_MB']} LKM={row['LKMSelf_MB']} "
                f"tot={row['total_MB']} used={row['used']}",
                flush=True,
            )
        else:
            miss = [t for t in TASKS if t not in sizes]
            print(f"{sub}: missing {miss}", flush=True)

    rows.sort(key=lambda r: r["total_MB"])
    cands = [r for r in rows if not r["used"] and r["ok_5mb"]]
    print("\nnext 6 lightest unused >=5MB:", [r["sub"] for r in cands[:6]], flush=True)
    print("their total MB:", sum(r["total_MB"] for r in cands[:6]), flush=True)
    OUT.write_text(json.dumps({"rows": rows, "next6": cands[:6]}, indent=2))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

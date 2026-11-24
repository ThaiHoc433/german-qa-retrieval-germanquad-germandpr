"""Xuất vài dòng đầu của mỗi dataset (.arrow) ra JSON để đọc bằng editor."""
import json
import sys
from pathlib import Path

from datasets import load_from_disk

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "samples"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5

OUT.mkdir(exist_ok=True)
for name in ("germanquad", "germandpr"):
    dsd = load_from_disk(str(RAW / name))
    for split, ds in dsd.items():
        path = OUT / f"{name}_{split}_sample.json"
        rows = [ds[i] for i in range(min(N, len(ds)))]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"{path}  ({len(ds)} dòng tổng, xuất {len(rows)})")

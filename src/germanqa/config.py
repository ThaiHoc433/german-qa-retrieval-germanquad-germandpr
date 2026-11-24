from dataclasses import dataclass
from pathlib import Path

# ---------- Đường dẫn: tất cả tính từ gốc dự án, không viết cứng ổ đĩa ----------
ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"

# ---------- Hằng số thí nghiệm: đổi ở đây thì đổi cho cả dự án ----------
SEED = 42          # R7: một seed duy nhất cho mọi phép ngẫu nhiên
DEV_RATIO = 0.10   # R5: tách 10% train làm dev (theo nhóm)


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    expected: dict


DATASETS = {
    "germanquad": DatasetSpec("deepset/germanquad", {"train": 11518, "test": 2204}),
    "germandpr":  DatasetSpec("deepset/germandpr",  {"train": 9275,  "test": 1025}),
}
"""GĐ0: tạo khung thư mục và in thông tin môi trường."""
import json

from germanqa.config import ROOT
from germanqa.experiment import environment_info

FOLDERS = [
    "data/raw", "data/processed", "configs", "reports", "results", "artifacts",
    "notebooks", "tests",
    "src/germanqa/data", "src/germanqa/retrieval", "src/germanqa/reader", "src/germanqa/evaluation",
    "scripts",
]


def main():
    for folder in FOLDERS:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
        print("ok  ", folder)
    print(json.dumps(environment_info(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
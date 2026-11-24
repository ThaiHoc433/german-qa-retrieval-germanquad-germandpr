"""Tìm thư mục snapshot của model trong cache HuggingFace theo commit của nhánh main (không cần mạng)."""
from pathlib import Path


def local_snapshot(repo):
    from huggingface_hub import constants
    root = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
    commit = (root / "refs" / "main").read_text().strip()
    path = root / "snapshots" / commit
    if not path.exists():
        raise FileNotFoundError(f"chưa có {repo} trong cache: {path}")
    return str(path)

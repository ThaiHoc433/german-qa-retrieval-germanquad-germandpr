"""Load GermanQuAD/GermanDPR từ HF Hub.

Repo gốc (deepset/germanquad, deepset/germandpr) chỉ còn loading script cũ mà
`datasets>=4` không hỗ trợ nữa ("Dataset scripts are no longer supported").
Dùng branch auto-convert `refs/convert/parquet` để đọc trực tiếp bằng pandas,
bỏ qua hoàn toàn cơ chế loading script.
"""
import pandas as pd
from huggingface_hub import hf_hub_download


def load_split(repo: str, split: str) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision="refs/convert/parquet",
        filename=f"plain_text/{split}/0000.parquet",
    )
    return pd.read_parquet(path)


def load_germanquad():
    return load_split("deepset/germanquad", "train"), load_split("deepset/germanquad", "test")


def load_germandpr():
    return load_split("deepset/germandpr", "train"), load_split("deepset/germandpr", "test")

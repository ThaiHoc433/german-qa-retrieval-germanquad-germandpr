"""Split D1 theo từng dòng; train/dev dùng chung pool question_key (spec V1.1 §2.3.e).

Hàm thuần: không đọc/ghi dataset, không lọc theo answer hay chunk coverage.
Quyết định bị loại dùng source_split=None; producer không ghi dòng đó vào split file.
"""
import math
import random
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

DEV_RATIO = 0.10
SEED = 42


class SplitDecision(TypedDict):
    query_id: str
    source_dataset: Literal["germanquad", "germandpr"]
    original_split: Literal["train", "test"]
    source_row_index: int
    question_key: str
    source_split: Literal["train", "dev", "test"] | None


def question_key(q: str) -> str:
    """Giữ whitespace nội bộ: thay đổi key có thể đổi tập dev dù counts không đổi."""
    if not isinstance(q, str) or not q.strip():
        raise ValueError("question phải là chuỗi khác rỗng/toàn khoảng trắng")
    return unicodedata.normalize("NFC", q).lower().strip()


def _counts(rows: Sequence[SplitDecision]) -> dict[str, int]:
    return {"rows": len(rows), "keys": len({r["question_key"] for r in rows})}


def assign_splits(
    gq_train_rows: Sequence[Mapping[str, object]],
    gq_test_rows: Sequence[Mapping[str, object]],
    gdpr_train_rows: Sequence[Mapping[str, object]],
    gdpr_test_rows: Sequence[Mapping[str, object]],
    *, dev_ratio: float = DEV_RATIO, seed: int = SEED,
) -> tuple[list[SplitDecision], dict]:
    """Nhận bốn split RAW còn nguyên thứ tự; trả mọi quyết định và report key/dòng.

    query_id luôn dùng split gốc. DPR dùng index trước lọc trong split gốc.
    API cũ nhận list câu hỏi/trả assignment theo key không còn được hỗ trợ.
    """
    if (not isinstance(dev_ratio, (int, float)) or isinstance(dev_ratio, bool)
            or not 0 <= dev_ratio <= 1 or not math.isfinite(dev_ratio)):
        raise ValueError("dev_ratio phải là số hữu hạn trong [0, 1], không nhận bool")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed phải là số nguyên, không nhận bool")

    inputs = (
        ("germanquad", "train", gq_train_rows),
        ("germanquad", "test", gq_test_rows),
        ("germandpr", "train", gdpr_train_rows),
        ("germandpr", "test", gdpr_test_rows),
    )
    decisions: list[SplitDecision] = []
    seen_ids: set[str] = set()
    for dataset, original_split, rows in inputs:
        where = f"{dataset}/{original_split}"
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise TypeError(f"{where}: rows phải là sequence record, không phải list câu hỏi cũ")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"{where}: row[{index}] phải là mapping có question")
            try:
                key = question_key(row.get("question"))
            except ValueError as exc:
                raise ValueError(f"{where} row[{index}]: {exc}") from exc
            if dataset == "germanquad":
                raw_id = row.get("id")
                if not isinstance(raw_id, int) or isinstance(raw_id, bool) or raw_id < 0:
                    raise ValueError(f"{where} row[{index}]: id phải là số nguyên không âm, không nhận bool")
                query_id = f"gq-{original_split}-{raw_id}"
            else:
                query_id = f"gdpr-{original_split}-{index}"
            if query_id in seen_ids:
                raise ValueError(f"{where} row[{index}]: query_id trùng {query_id!r}")
            seen_ids.add(query_id)
            decisions.append({
                "query_id": query_id,
                "source_dataset": dataset,
                "original_split": original_split,
                "source_row_index": index,
                "question_key": key,
                "source_split": None,
            })

    test_keys = {r["question_key"] for r in decisions if r["original_split"] == "test"}
    train_pool = {r["question_key"] for r in decisions if r["original_split"] == "train"} - test_keys
    dev_keys = set(random.Random(seed).sample(sorted(train_pool), int(len(train_pool) * dev_ratio)))
    train_keys = train_pool - dev_keys

    for row in decisions:
        key = row["question_key"]
        if row["original_split"] == "test":
            row["source_split"] = "test"
        elif key in test_keys:
            row["source_split"] = None
        else:
            row["source_split"] = "dev" if key in dev_keys else "train"

    report = {
        "seed": seed,
        "dev_ratio": dev_ratio,
        "key_counts": {
            "test_union": len(test_keys), "train_dev_pool": len(train_pool),
            "train": len(train_keys), "dev": len(dev_keys),
        },
        "datasets": {},
    }
    for dataset in ("germanquad", "germandpr"):
        rows = [r for r in decisions if r["source_dataset"] == dataset]
        removed = [r for r in rows if r["source_split"] is None]
        report["datasets"][dataset] = {
            "input": {split: _counts([r for r in rows if r["original_split"] == split])
                      for split in ("train", "test")},
            "assigned": {split: _counts([r for r in rows if r["source_split"] == split])
                         for split in ("train", "dev", "test")},
            "removed": {**_counts(removed), "query_ids": sorted(r["query_id"] for r in removed)},
        }
    return decisions, report

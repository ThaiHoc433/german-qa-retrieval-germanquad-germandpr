"""Orchestration D6/P4 cho base corpus V1.1 — module 2 (Data/Splits/Corpus).

Hàm thuần: nhận raw GQ/DPR train+test trước mọi D1/unmappable; gọi
CorpusBuilder hiện có để giữ mọi chunk từ nguồn được phép (GermanQuAD context,
GermanDPR positive context), đồng thời thu title trước dedup theo P4.
Không đọc answers/negative_ctxs/hard_negative_ctxs; không gọi Hub, không ghi
artifact (entrypoint thuộc T5.8); không nhận SplitDecision.
"""
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import TypedDict

from src.corpus import CorpusBuilder, passage_id


class BaseCorpusResult(TypedDict):
    passages: dict[str, dict]
    base_title_exclusions: list[str]
    source_report: dict


def normalize_title(title: str | None) -> str | None:
    """P4: NFC → `_` thành space → lowercase → gộp whitespace → strip.

    None hoặc kết quả rỗng đều trả None (title thiếu theo P4)."""
    if title is None:
        return None
    if not isinstance(title, str):
        raise TypeError("title phải là str hoặc None")
    key = re.sub(r"\s+", " ", unicodedata.normalize("NFC", title).replace("_", " ").lower()).strip()
    return key or None


def _require_sequence(value, where: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{where}: phải là sequence record")


def _gq_context(row: Mapping, where: str) -> str:
    if not isinstance(row, Mapping):
        raise TypeError(f"{where}: row phải là mapping")
    context = row.get("context")
    if not isinstance(context, str):
        raise ValueError(f"{where}: context phải là chuỗi")
    return context


def _dpr_positive_ctxs(row: Mapping, where: str) -> list[tuple[str | None, str]]:
    if not isinstance(row, Mapping):
        raise TypeError(f"{where}: row phải là mapping")
    group = row.get("positive_ctxs")
    if not isinstance(group, Mapping):
        raise ValueError(f"{where}: positive_ctxs phải là mapping có title/text/passage_id")
    titles, texts, pids = group.get("title"), group.get("text"), group.get("passage_id")
    for name, value in (("title", titles), ("text", texts), ("passage_id", pids)):
        if not isinstance(value, list):
            raise ValueError(f"{where}: positive_ctxs.{name} phải là list")
    if not (len(titles) == len(texts) == len(pids)):
        raise ValueError(f"{where}: positive_ctxs title/text/passage_id phải cùng độ dài")
    result = []
    for i, (title, text, pid) in enumerate(zip(titles, texts, pids)):
        label = f"{where}.positive_ctxs[{i}]"
        if not isinstance(text, str):
            raise ValueError(f"{label}: text phải là chuỗi")
        if title is not None and not isinstance(title, str):
            raise ValueError(f"{label}: title phải là chuỗi hoặc None")
        if not isinstance(pid, str):
            raise ValueError(f"{label}: passage_id raw phải là chuỗi")
        result.append((title, text))
    return result


def _new_group_counts() -> dict:
    return {"n_contexts": 0, "n_chunks_before_dedup": 0, "n_empty_contexts": 0,
            "n_missing_titles": 0, "n_unique_title_keys": 0}


def build_base_corpus(
    gq_train_rows: Sequence[Mapping[str, object]],
    gq_test_rows: Sequence[Mapping[str, object]],
    gdpr_train_rows: Sequence[Mapping[str, object]],
    gdpr_test_rows: Sequence[Mapping[str, object]],
) -> BaseCorpusResult:
    """D6 base = mọi context GQ + mọi positive context DPR, train+test, trước
    D1/unmappable. P4 title thu trước dedup từ mọi occurrence, kể cả context
    rỗng hoặc bị dedup mất. Đại diện giữ priority GQ > DPR positive của
    CorpusBuilder hiện có; cùng nguồn giữ occurrence đầu theo thứ tự input."""
    builder = CorpusBuilder()
    all_titles: set[str] = set()
    empty_meta: list[dict] = []
    missing_meta: list[dict] = []
    groups = {
        "germanquad": {"train": _new_group_counts(), "test": _new_group_counts()},
        "germandpr_positive": {"train": _new_group_counts(), "test": _new_group_counts()},
    }

    def process(source: str, split: str, occurrences_of, rows, row_label: str):
        _require_sequence(rows, row_label)
        counts = groups[source][split]
        title_keys: set[str] = set()
        for row_index, row in enumerate(rows):
            occs = occurrences_of(row, f"{row_label}[{row_index}]")
            for ctx_index, (raw_title, text) in enumerate(occs):
                counts["n_contexts"] += 1
                meta = {"source": source, "original_split": split, "source_row_index": row_index,
                        "source_context_index": ctx_index, "parent_document_id": passage_id(text)}
                key = normalize_title(raw_title)
                if key is None:
                    counts["n_missing_titles"] += 1
                    missing_meta.append(meta)
                else:
                    title_keys.add(key)
                    all_titles.add(key)
                chunks = builder.add_text(text, source, title=raw_title)
                counts["n_chunks_before_dedup"] += len(chunks)
                if not chunks:
                    counts["n_empty_contexts"] += 1
                    empty_meta.append(meta)
        counts["n_unique_title_keys"] = len(title_keys)

    def gq_occurrences(row, where):
        context = _gq_context(row, where)
        first_line = context.splitlines()[0] if context.splitlines() else None
        return [(first_line, context)]

    def dpr_occurrences(row, where):
        return _dpr_positive_ctxs(row, where)

    process("germanquad", "train", gq_occurrences, gq_train_rows, "germanquad/train")
    process("germanquad", "test", gq_occurrences, gq_test_rows, "germanquad/test")
    process("germandpr_positive", "train", dpr_occurrences, gdpr_train_rows, "germandpr/train")
    process("germandpr_positive", "test", dpr_occurrences, gdpr_test_rows, "germandpr/test")

    passages = {rec["passage_id"]: rec for rec in sorted(builder.records(), key=lambda r: r["passage_id"])}

    n_contexts_total = sum(g["n_contexts"] for s in groups.values() for g in s.values())
    n_chunks_total = sum(g["n_chunks_before_dedup"] for s in groups.values() for g in s.values())
    representative_sources = {"germanquad": 0, "germandpr_positive": 0}
    for rec in passages.values():
        representative_sources[rec["source"]] += 1

    report = {
        "input_sources": groups,
        "n_contexts_total": n_contexts_total,
        "n_chunks_before_dedup_total": n_chunks_total,
        "n_passages": len(passages),
        "representative_sources": representative_sources,
        "n_title_exclusions": len(all_titles),
        "empty_contexts": empty_meta,
        "missing_title_contexts": missing_meta,
    }
    return {
        "passages": passages,
        "base_title_exclusions": sorted(all_titles),
        "source_report": report,
    }

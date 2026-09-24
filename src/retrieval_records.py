"""DPR producer V1.1 §1.3/§2.3.f,g: positive occurrence union và hard-negative groups.

Hàm thuần: nhận split decisions T5.2 (phần GermanDPR) và corpus đại diện đã chọn xong.
Không chia split, sửa corpus, gọi Hub hoặc ghi artifact (entrypoint thuộc T5.8).
"""
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TypedDict

from src.corpus import chunk_text, find_span_chunks, passage_id
from src.schemas import SchemaError, validate_passage_record, validate_retrieval_training_record
from src.splits import SplitDecision, question_key


SPLITS = ("train", "dev", "test")
DECISION_FIELDS = ("query_id", "source_dataset", "original_split", "source_row_index",
                   "question_key", "source_split")
CTX_FIELDS = ("positive_ctxs", "hard_negative_ctxs", "negative_ctxs")


class RetrievalProductionResult(TypedDict):
    records: dict[str, list[dict]]
    germandpr_chunk_mapping_report: dict


def _require_sequence(value, where: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{where}: phải là sequence record")


def _decision_map(decisions: Sequence[SplitDecision]) -> dict[str, Mapping]:
    """Chỉ lấy GermanDPR, cho phép nguyên output T5.2 gồm cả GermanQuAD."""
    _require_sequence(decisions, "split_decisions")
    result = {}
    for index, decision in enumerate(decisions):
        where = f"decision[{index}]"
        if not isinstance(decision, Mapping):
            raise TypeError(f"{where}: phải là mapping")
        dataset = decision.get("source_dataset")
        if dataset not in ("germanquad", "germandpr"):
            raise ValueError(f"{where}: source_dataset không hợp lệ")
        if dataset == "germanquad":
            continue
        missing = [key for key in DECISION_FIELDS if key not in decision]
        if missing:
            raise ValueError(f"{where}: thiếu field decision {missing}")
        qid = decision["query_id"]
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"{where}: query_id phải là chuỗi khác rỗng")
        if qid in result:
            raise ValueError(f"{where}: decision query_id trùng {qid!r}")
        result[qid] = decision
    return result


def _ctx_group(row: Mapping, field: str, where: str) -> list[tuple[str | None, str]]:
    group = row.get(field)
    if not isinstance(group, Mapping):
        raise ValueError(f"{where}: {field} phải là mapping có title/text/passage_id")
    titles, texts, pids = group.get("title"), group.get("text"), group.get("passage_id")
    for name, value in (("title", titles), ("text", texts), ("passage_id", pids)):
        if not isinstance(value, list):
            raise ValueError(f"{where}: {field}.{name} phải là list")
    if not (len(titles) == len(texts) == len(pids)):
        raise ValueError(f"{where}: {field} title/text/passage_id phải cùng độ dài")
    result = []
    for i, (title, text, pid) in enumerate(zip(titles, texts, pids)):
        label = f"{where}.{field}[{i}]"
        if not isinstance(text, str):
            raise ValueError(f"{label}: text phải là chuỗi")
        if title is not None and not isinstance(title, str):
            raise ValueError(f"{label}: title phải là chuỗi hoặc None")
        if not isinstance(pid, str):
            raise ValueError(f"{label}: passage_id raw phải là chuỗi")
        result.append((title, text))
    return result


def _validate_raw(row: Mapping, where: str):
    if not isinstance(row, Mapping):
        raise TypeError(f"{where}: row phải là mapping")
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{where}: question phải là chuỗi không trắng")
    answers = row.get("answers")
    if not isinstance(answers, list):
        raise ValueError(f"{where}: answers phải là list chuỗi")
    for i, a in enumerate(answers):
        if not isinstance(a, str) or not a.strip():
            raise ValueError(f"{where}.answers[{i}]: answer phải là chuỗi không trắng")
    positive = _ctx_group(row, "positive_ctxs", where)
    hard_negative = _ctx_group(row, "hard_negative_ctxs", where)
    _ctx_group(row, "negative_ctxs", where)  # kiểm cùng dạng, không dùng tiếp
    return question, answers, positive, hard_negative


def _prepare_rows(gdpr_train_rows, gdpr_test_rows, split_decisions):
    remaining = _decision_map(split_decisions)
    prepared = []
    for original_split, rows in (("train", gdpr_train_rows), ("test", gdpr_test_rows)):
        _require_sequence(rows, f"germandpr/{original_split}")
        for index, row in enumerate(rows):
            where = f"germandpr/{original_split}[{index}]"
            question, answers, positive, hard_negative = _validate_raw(row, where)
            qid = f"gdpr-{original_split}-{index}"
            decision = remaining.pop(qid, None)
            if decision is None:
                raise ValueError(f"{qid}: thiếu decision GermanDPR")
            if decision["original_split"] != original_split:
                raise ValueError(f"{qid}: decision.original_split khác nguồn raw")
            row_index = decision["source_row_index"]
            if not isinstance(row_index, int) or isinstance(row_index, bool) or row_index != index:
                raise ValueError(f"{qid}: decision.source_row_index khác index raw hoặc không phải int")
            try:
                key = question_key(question)
            except ValueError as exc:
                raise ValueError(f"{where}: {exc}") from exc
            if decision["question_key"] != key:
                raise ValueError(f"{qid}: decision.question_key khác câu hỏi raw")
            allowed = ("test",) if original_split == "test" else ("train", "dev", None)
            if decision["source_split"] not in allowed:
                raise ValueError(f"{qid}: decision.source_split sai với nguồn {original_split}")
            prepared.append((qid, decision["source_split"], question, answers, positive, hard_negative))
    if remaining:
        raise ValueError(f"decision GermanDPR thừa: {sorted(remaining)}")
    return prepared


def _validate_positive_membership(pid: str, passages: Mapping, qid: str) -> None:
    rec = passages.get(pid)
    if not isinstance(rec, Mapping):
        raise ValueError(f"{qid}: passage thiếu hoặc không phải mapping: {pid}")
    if rec.get("passage_id") != pid:
        raise ValueError(f"{qid}: passage_id trong record khác key corpus {pid}")
    errors = validate_passage_record(dict(rec), where=f"PassageRecord[{qid}/{pid}]")
    if errors:
        raise SchemaError(errors)


def _positive_occurrences(positive_ctxs, answers, passages: Mapping, qid: str) -> dict[str, str]:
    """Union mọi chunk chứa trọn bất kỳ occurrence nào của bất kỳ answer nào
    (spec §2.3.f). Thứ tự: context raw rồi chunk index tăng dần; mỗi ID giữ
    text của chunk hợp lệ đầu tiên (dedup toàn query theo passage_id)."""
    positives: dict[str, str] = {}
    for _title, text in positive_ctxs:
        chunks = chunk_text(text)
        if not chunks:
            continue
        indices: set[int] = set()
        for answer in answers:
            start = text.find(answer)
            while start != -1:
                end = start + len(answer)
                indices.update(find_span_chunks(chunks, start, end))
                start = text.find(answer, start + 1)  # +1: giữ occurrence chồng nhau
        for idx in sorted(indices):
            chunk = chunks[idx]
            pid = passage_id(chunk.text)
            if pid not in positives:
                _validate_positive_membership(pid, passages, qid)
                positives[pid] = chunk.text
    return positives


def _hard_negative_groups(hard_negative_ctxs, gold_set: set[str], qid: str, filtering: list[dict]):
    """Một entry theo mỗi occurrence context raw (§2.3.g); không dedup theo
    source_context_id. Loại chunk trong gold_set; context rỗng hoặc group hết
    chunk vì mask đều bị bỏ, có log riêng."""
    groups: list[dict] = []
    n_masked = 0
    n_dropped = 0
    for context_index, (_title, text) in enumerate(hard_negative_ctxs):
        source_context_id = passage_id(text)
        chunks = chunk_text(text)
        if not chunks:
            filtering.append({
                "query_id": qid, "source_context_index": context_index,
                "source_context_id": source_context_id, "removed_passage_ids": [],
                "group_removed": True, "reason": "empty_context",
            })
            n_dropped += 1
            continue
        kept, removed_ids = [], []
        for c in chunks:
            pid = passage_id(c.text)
            if pid in gold_set:
                removed_ids.append(pid)
            else:
                kept.append(c)
        if not kept:
            filtering.append({
                "query_id": qid, "source_context_index": context_index,
                "source_context_id": source_context_id, "removed_passage_ids": removed_ids,
                "group_removed": True, "reason": "gold_overlap",
            })
            n_dropped += 1
            n_masked += len(removed_ids)
            continue
        groups.append({
            "source_context_id": source_context_id,
            "texts": [c.text for c in kept],
            "passage_ids": [passage_id(c.text) for c in kept],
        })
        if removed_ids:
            filtering.append({
                "query_id": qid, "source_context_index": context_index,
                "source_context_id": source_context_id, "removed_passage_ids": removed_ids,
                "group_removed": False, "reason": "gold_overlap",
            })
            n_masked += len(removed_ids)
    return groups, n_masked, n_dropped


def _histogram(counter: Counter) -> dict[str, int]:
    return {str(k): counter[k] for k in sorted(counter)}


def _new_stats() -> dict:
    return {
        "n_records": 0, "n_single_positive": 0, "n_multiple_positive": 0,
        "positive_chunks_per_record": Counter(), "n_positive_chunks": 0,
        "hard_negative_groups_per_record": Counter(), "hard_negative_chunks_per_group": Counter(),
        "n_hard_negative_groups": 0, "n_hard_negative_chunks": 0,
        "n_masked_hard_negative_chunks": 0, "n_dropped_hard_negative_groups": 0,
        "n_positive_text_not_raw_chunk": 0, "n_null_positive_ids": 0,
    }


def build_retrieval_training_records(
    gdpr_train_rows: Sequence[Mapping[str, object]],
    gdpr_test_rows: Sequence[Mapping[str, object]],
    split_decisions: Sequence[SplitDecision],
    passages: Mapping[str, Mapping[str, object]],
) -> RetrievalProductionResult:
    """Giữ mọi positive hợp lệ và mọi chunk còn lại của từng hard-negative
    group; không chọn positive/chunk theo epoch (thuộc sampler module 5).

    Input raw phải nguyên thứ tự để khớp source_row_index của T5.2. Record
    không map được positive nào bị loại toàn bộ khỏi output (khác QA: không
    có fallback context chưa chunk).
    """
    if not isinstance(passages, Mapping):
        raise TypeError("passages phải là map passage_id -> PassageRecord")
    prepared = _prepare_rows(gdpr_train_rows, gdpr_test_rows, split_decisions)

    records: dict[str, list[dict]] = {split: [] for split in SPLITS}
    unmappable: dict[str, list[str]] = {split: [] for split in SPLITS}
    stats = {split: _new_stats() for split in SPLITS}
    filtering: list[dict] = []
    excluded_ids: list[str] = []

    for qid, split, question, answers, positive, hard_negative in prepared:
        if split is None:
            excluded_ids.append(qid)
            continue
        positives = _positive_occurrences(positive, answers, passages, qid)
        if not positives:
            unmappable[split].append(qid)
            continue
        groups, n_masked, n_dropped = _hard_negative_groups(hard_negative, set(positives), qid, filtering)
        rec = {
            "query_id": qid, "question": question,
            "positive_texts": list(positives.values()),
            "positive_passage_ids": list(positives.keys()),
            "hard_negative_groups": groups,
            "source_split": split,
        }
        errors = validate_retrieval_training_record(rec)
        if errors:
            raise SchemaError(errors)
        records[split].append(rec)

        s = stats[split]
        n_pos = len(rec["positive_passage_ids"])
        s["n_records"] += 1
        if n_pos == 1:
            s["n_single_positive"] += 1
        else:
            s["n_multiple_positive"] += 1
        s["positive_chunks_per_record"][n_pos] += 1
        s["n_positive_chunks"] += n_pos
        s["hard_negative_groups_per_record"][len(groups)] += 1
        for g in groups:
            s["hard_negative_chunks_per_group"][len(g["passage_ids"])] += 1
        s["n_hard_negative_groups"] += len(groups)
        s["n_hard_negative_chunks"] += sum(len(g["passage_ids"]) for g in groups)
        s["n_masked_hard_negative_chunks"] += n_masked
        s["n_dropped_hard_negative_groups"] += n_dropped
        s["n_positive_text_not_raw_chunk"] += sum(
            passage_id(t) != pid for pid, t in zip(rec["positive_passage_ids"], rec["positive_texts"]))
        s["n_null_positive_ids"] += sum(pid is None for pid in rec["positive_passage_ids"])

    files = {}
    for split in SPLITS:
        s = stats[split]
        n_unmappable = len(unmappable[split])
        files[f"retrieval_germandpr_{split}.jsonl"] = {
            "n_input_after_d1": s["n_records"] + n_unmappable,
            "n_records": s["n_records"],
            "n_unmappable": n_unmappable,
            "unmappable_query_ids": sorted(unmappable[split]),
            "n_single_positive": s["n_single_positive"],
            "n_multiple_positive": s["n_multiple_positive"],
            "positive_chunks_per_record": _histogram(s["positive_chunks_per_record"]),
            "n_positive_chunks": s["n_positive_chunks"],
            "hard_negative_groups_per_record": _histogram(s["hard_negative_groups_per_record"]),
            "hard_negative_chunks_per_group": _histogram(s["hard_negative_chunks_per_group"]),
            "n_hard_negative_groups": s["n_hard_negative_groups"],
            "n_hard_negative_chunks": s["n_hard_negative_chunks"],
            "n_masked_hard_negative_chunks": s["n_masked_hard_negative_chunks"],
            "n_dropped_hard_negative_groups": s["n_dropped_hard_negative_groups"],
            "n_positive_text_not_raw_chunk": s["n_positive_text_not_raw_chunk"],
            "n_null_positive_ids": s["n_null_positive_ids"],
            "all_positive_texts_are_raw_chunks": s["n_positive_text_not_raw_chunk"] == 0,
            "no_null_positive_ids": s["n_null_positive_ids"] == 0,
        }

    report = {
        "input_rows": len(prepared),
        "excluded_leakage_rows": len(excluded_ids),
        "excluded_leakage_query_ids": sorted(excluded_ids),
        "files": files,
        "hard_negative_filtering": filtering,
    }
    return {"records": records, "germandpr_chunk_mapping_report": report}

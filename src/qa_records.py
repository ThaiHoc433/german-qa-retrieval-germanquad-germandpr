"""QA producer V1.1 §1.2/§2.3.a,c: char-offset, references và collision remap.

Hàm thuần: nhận split decisions T5.2 và corpus đại diện đã chọn xong.
Không chia split, sửa corpus, gọi Hub hoặc ghi artifact (entrypoint thuộc T5.8).
"""
from collections.abc import Mapping, Sequence
from typing import TypedDict

from src.corpus import Chunk, chunk_text, find_span_chunks, normalize_text, passage_id
from src.schemas import SchemaError, validate_passage_record, validate_qa_record
from src.splits import SplitDecision, question_key


SPLITS = ("train", "dev", "test")
DECISION_FIELDS = ("query_id", "source_dataset", "original_split", "source_row_index",
                   "question_key", "source_split")


class QAProductionResult(TypedDict):
    records: dict[str, list[dict]]
    coverage_report: dict
    ambiguous_offset_remap_report: list[dict]


def _require_sequence(value, where: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{where}: phải là sequence record")


def _decision_map(decisions: Sequence[SplitDecision]) -> dict[str, Mapping]:
    """Chỉ lấy GQ, cho phép nguyên output T5.2 gồm cả DPR."""
    _require_sequence(decisions, "split_decisions")
    result = {}
    for index, decision in enumerate(decisions):
        where = f"decision[{index}]"
        if not isinstance(decision, Mapping):
            raise TypeError(f"{where}: phải là mapping")
        dataset = decision.get("source_dataset")
        if dataset not in ("germanquad", "germandpr"):
            raise ValueError(f"{where}: source_dataset không hợp lệ")
        if dataset == "germandpr":
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


def _validate_raw(row: Mapping, where: str) -> tuple[int, str]:
    if not isinstance(row, Mapping):
        raise TypeError(f"{where}: row phải là mapping")
    raw_id = row.get("id")
    if not isinstance(raw_id, int) or isinstance(raw_id, bool) or raw_id < 0:
        raise ValueError(f"{where}: id phải là int không âm, không nhận bool")
    try:
        key = question_key(row.get("question"))
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    context = row.get("context")
    if not isinstance(context, str):
        raise ValueError(f"{where}: context phải là chuỗi")
    answers = row.get("answers")
    if not isinstance(answers, Mapping):
        raise ValueError(f"{where}: answers phải là mapping raw có text/answer_start")
    texts, starts = answers.get("text"), answers.get("answer_start")
    if not isinstance(texts, list) or not isinstance(starts, list) or len(texts) != len(starts):
        raise ValueError(f"{where}: answers.text/answer_start phải là hai list cùng độ dài")
    for index, (text, start) in enumerate(zip(texts, starts)):
        label = f"{where}.answers[{index}]"
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{label}: answer text phải là chuỗi không trắng")
        if not isinstance(start, int) or isinstance(start, bool) or start < 0:
            raise ValueError(f"{label}: answer_start phải là int không âm, không nhận bool")
        if context[start:start + len(text)] != text:
            raise ValueError(f"{label}: slice context gốc khác answer text")
    return raw_id, key


def _prepare_rows(gq_train_rows, gq_test_rows, split_decisions):
    remaining = _decision_map(split_decisions)
    prepared = []
    seen_ids = set()
    for original_split, rows in (("train", gq_train_rows), ("test", gq_test_rows)):
        _require_sequence(rows, f"germanquad/{original_split}")
        for index, row in enumerate(rows):
            raw_id, key = _validate_raw(row, f"germanquad/{original_split}[{index}]")
            qid = f"gq-{original_split}-{raw_id}"
            if qid in seen_ids:
                raise ValueError(f"query_id raw trùng {qid!r}")
            seen_ids.add(qid)
            decision = remaining.pop(qid, None)
            if decision is None:
                raise ValueError(f"{qid}: thiếu decision GermanQuAD")
            if decision["original_split"] != original_split:
                raise ValueError(f"{qid}: decision.original_split khác nguồn raw")
            row_index = decision["source_row_index"]
            if (not isinstance(row_index, int) or isinstance(row_index, bool) or row_index != index):
                raise ValueError(f"{qid}: decision.source_row_index khác index raw hoặc không phải int")
            if decision["question_key"] != key:
                raise ValueError(f"{qid}: decision.question_key khác câu hỏi raw")
            allowed = ("test",) if original_split == "test" else ("train", "dev", None)
            if decision["source_split"] not in allowed:
                raise ValueError(f"{qid}: decision.source_split sai với nguồn {original_split}")
            prepared.append((row, decision))
    if remaining:
        raise ValueError(f"decision GermanQuAD thừa: {sorted(remaining)}")
    return prepared


def _representative(pid: str, passages: Mapping, texts: dict[str, str], qid: str) -> str:
    if pid not in texts:
        rec = passages.get(pid)
        if not isinstance(rec, Mapping):
            raise ValueError(f"{qid}: passage thiếu hoặc không phải mapping: {pid}")
        if rec.get("passage_id") != pid:
            raise ValueError(f"{qid}: passage_id trong record khác key corpus {pid}")
        errors = validate_passage_record(dict(rec), where=f"PassageRecord[{qid}/{pid}]")
        if errors:
            raise SchemaError(errors)
        texts[pid] = rec["text"]
    return texts[pid]


def _remap_offset(*, representative: str, answer: str, raw_start: int, context_len: int,
                  chunk: Chunk, qid: str, pid: str, answer_index: int, report: list[dict]) -> int | None:
    """T5.4: chỉ gọi sau khi xác nhận collision và slice trực tiếp sai."""
    candidates = []
    start = representative.find(answer)
    while start != -1:
        candidates.append(start)
        # +1 để giữ occurrence chồng nhau; answer đã được kiểm khác rỗng.
        start = representative.find(answer, start + 1)
    selected = min(candidates, key=lambda offset: (
        abs(offset * context_len - raw_start * len(representative)), offset
    )) if candidates else None
    if len(candidates) != 1:
        report.append({
            "query_id": qid, "passage_id": pid, "answer_text": answer, "answer_index": answer_index,
            "source_answer_start": raw_start, "source_context_char_len": context_len,
            "source_chunk_start_char": chunk.char_start, "source_chunk_end_char": chunk.char_end,
            "representative_char_len": len(representative),
            "candidate_offsets": candidates, "selected_offset": selected,
        })
    return selected


def _gold_evidences(row: Mapping, qid: str, passages: Mapping, texts: dict[str, str], report: list[dict]) -> list[dict]:
    context, answers = row["context"], row["answers"]
    chunks = chunk_text(context)
    evidences, seen = [], set()
    for answer_index, (answer, raw_start) in enumerate(zip(answers["text"], answers["answer_start"])):
        for chunk_index in find_span_chunks(chunks, raw_start, raw_start + len(answer)):
            chunk = chunks[chunk_index]
            pid = passage_id(chunk.text)
            representative = _representative(pid, passages, texts, qid)
            if normalize_text(chunk.text) != normalize_text(representative):
                raise ValueError(f"{qid}/{pid}: passage hash trùng nhưng normalized_text khác nhau")
            local_start = raw_start - chunk.char_start
            if representative[local_start:local_start + len(answer)] != answer:
                if chunk.text == representative:
                    raise ValueError(f"{qid}/{pid}: slice local sai dù text giống nhau")
                local_start = _remap_offset(
                    representative=representative, answer=answer, raw_start=raw_start, context_len=len(context),
                    chunk=chunk, qid=qid, pid=pid, answer_index=answer_index, report=report,
                )
            if local_start is None:
                continue
            key = (pid, answer, local_start, local_start + len(answer))
            if key not in seen:
                seen.add(key)
                evidences.append({"passage_id": pid, "answer_text": answer,
                                  "answer_start": key[2], "answer_end": key[3]})
    return evidences


def build_qa_records(
    gq_train_rows: Sequence[Mapping[str, object]],
    gq_test_rows: Sequence[Mapping[str, object]],
    split_decisions: Sequence[SplitDecision],
    passages: Mapping[str, Mapping[str, object]],
) -> QAProductionResult:
    """Giữ mọi dòng GQ sau D1, kể cả gold rỗng; trả report trong bộ nhớ.

    Input raw phải nguyên thứ tự để khớp source_row_index của T5.2. Đại diện
    corpus phải đóng băng trước lời gọi. Chỉ kiểm PassageRecord cần cho evidence.
    Entry report là mỗi cặp annotation/chunk fallback có 0 hoặc >1 ứng viên.
    """
    if not isinstance(passages, Mapping):
        raise TypeError("passages phải là map passage_id -> PassageRecord")
    prepared = _prepare_rows(gq_train_rows, gq_test_rows, split_decisions)
    records = {split: [] for split in SPLITS}
    remap_report: list[dict] = []
    passage_texts: dict[str, str] = {}
    excluded = 0
    for row, decision in prepared:
        split = decision["source_split"]
        if split is None:
            excluded += 1
            continue
        qid = decision["query_id"]
        rec = {"query_id": qid, "question": row["question"],
               "gold_evidences": _gold_evidences(row, qid, passages, passage_texts, remap_report),
               "reference_answers": list(row["answers"]["text"]),
               "source_dataset": "germanquad", "source_split": split}
        errors = validate_qa_record(rec, passage_texts=passage_texts)
        if errors:
            raise SchemaError(errors)
        records[split].append(rec)

    files = {}
    for split, split_records in records.items():
        empty_ids = sorted(r["query_id"] for r in split_records if not r["gold_evidences"])
        n = len(split_records)
        files[f"qa_germanquad_{split}.jsonl"] = {
            "n_records": n, "n_empty_gold": len(empty_ids),
            "coverage_rate": (n - len(empty_ids)) / n if n else None,
            "empty_gold_query_ids": empty_ids,
        }
    return {"records": records, "ambiguous_offset_remap_report": remap_report,
            "coverage_report": {"input_rows": len(prepared), "excluded_leakage_rows": excluded, "files": files}}

"""Schema validation V1.1 — mục 1.1–1.5 của
docs/superpowers/specs/2026-09-10-german-qa-retrieval-module-specs.md.

Mỗi validator trả về list lỗi (rỗng = hợp lệ) để gom toàn bộ lỗi của cả file
rồi mới báo, thay vì dừng ở lỗi đầu tiên.
"""
import math
import re
from collections import Counter
from collections.abc import Mapping

from src.corpus import normalize_text, passage_id

SPLITS = ("train", "dev", "test")
CORPUS_IDS = ("base", "mid", "large")
RETRIEVER_NAMES = ("bm25", "dense", "hybrid", "cross_encoder")
STATUSES = ("ok", "error")
ERROR_STAGES = ("retrieval", "rerank", "read")
PASSAGE_SOURCES = ("germanquad", "germandpr_positive", "wikipedia_distractor")

PASSAGE_RECORD_FIELDS = (
    "passage_id", "text", "normalized_text", "source", "title", "char_len",
    "parent_document_id", "chunk_start_char", "chunk_end_char",
)
TRAINING_RECORD_FIELDS = (
    "query_id", "question", "positive_texts", "positive_passage_ids",
    "hard_negative_groups", "source_split",
)
HARD_NEGATIVE_GROUP_FIELDS = ("source_context_id", "texts", "passage_ids")

QA_RECORD_FIELDS = (
    "query_id", "question", "gold_evidences", "reference_answers", "source_dataset", "source_split",
)
GOLD_EVIDENCE_FIELDS = ("passage_id", "answer_text", "answer_start", "answer_end")
RETRIEVAL_RESULT_FIELDS = ("query_id", "passage_id", "rank", "score", "retriever_name")
ANSWER_FIELDS = (
    "predicted_answer_text", "predicted_passage_id",
    "predicted_answer_start", "predicted_answer_end", "reader_span_prob",
)
LATENCY_FIELDS = ("retrieval", "rerank", "read", "total")
PREDICTION_RECORD_FIELDS = (
    "run_id", "config_id", "corpus_id", "retrieval_k", "reader_k",
    "query_id", "status", "error_stage",
    "retrieved", "reranked", *ANSWER_FIELDS, "latency_ms",
)


class SchemaError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        shown = "\n".join(f"  - {e}" for e in self.errors[:20])
        more = f"\n  ... và {len(self.errors) - 20} lỗi khác" if len(self.errors) > 20 else ""
        super().__init__(f"{len(self.errors)} lỗi schema/nhất quán:\n{shown}{more}")


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _is_nonempty_str(x) -> bool:
    return isinstance(x, str) and x.strip() != ""


def _field_errors(rec, fields: tuple[str, ...], where: str) -> list[str]:
    if not isinstance(rec, dict):
        return [f"{where}: phải là JSON object, nhận {type(rec).__name__}"]
    missing = [f"{where}: thiếu field '{f}'" for f in fields if f not in rec]
    extra = [f"{where}: field '{f}' không có trong schema" for f in sorted(set(rec) - set(fields))]
    return missing + extra


def _has_all(rec, fields) -> bool:
    return isinstance(rec, dict) and all(f in rec for f in fields)


def _label(rec: dict, where: str) -> str:
    qid = rec.get("query_id")
    return f"{where}[{qid}]" if isinstance(qid, str) and qid else where


# ---------- PassageRecord (1.1) ----------

def _is_hash_id(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{16}", value) is not None


def validate_passage_record(rec: object, where: str = "PassageRecord", *,
                            parent_text: str | None = None) -> list[str]:
    """Kiểm schema; khi có parent_text, kiểm thêm hash parent và slice gốc."""
    errors = _field_errors(rec, PASSAGE_RECORD_FIELDS, where)
    if not _has_all(rec, PASSAGE_RECORD_FIELDS):
        return errors

    text = rec["text"]
    text_ok = _is_nonempty_str(text)
    if not text_ok:
        errors.append(f"{where}: text phải là chuỗi khác rỗng")
    if not _is_hash_id(rec["passage_id"]):
        errors.append(f"{where}: passage_id phải là 16 ký tự hex thường")
    if text_ok:
        if rec["normalized_text"] != normalize_text(text):
            errors.append(f"{where}: normalized_text khác normalize_text(text)")
        if rec["passage_id"] != passage_id(text):
            errors.append(f"{where}: passage_id khác hash của text đã normalize")
    if not _is_nonempty_str(rec["normalized_text"]):
        errors.append(f"{where}: normalized_text phải là chuỗi khác rỗng")
    if rec["source"] not in PASSAGE_SOURCES:
        errors.append(f"{where}: source phải thuộc {PASSAGE_SOURCES}")
    if rec["title"] is not None and not isinstance(rec["title"], str):
        errors.append(f"{where}: title phải là chuỗi hoặc None")
    if not (_is_int(rec["char_len"]) and rec["char_len"] > 0):
        errors.append(f"{where}: char_len phải là số nguyên dương")
    elif text_ok and rec["char_len"] != len(text):
        errors.append(f"{where}: char_len khác len(text)")
    if not _is_hash_id(rec["parent_document_id"]):
        errors.append(f"{where}: parent_document_id phải là 16 ký tự hex thường")

    start, end = rec["chunk_start_char"], rec["chunk_end_char"]
    offsets_ok = _is_int(start) and start >= 0 and _is_int(end) and end > start
    if not offsets_ok:
        errors.append(f"{where}: chunk_start_char/chunk_end_char phải là số nguyên, 0 ≤ start < end")
    elif text_ok and end - start != len(text):
        errors.append(f"{where}: chunk_end_char - chunk_start_char khác len(text)")
    if parent_text is not None:
        if not isinstance(parent_text, str):
            errors.append(f"{where}: parent_text phải là chuỗi")
        else:
            if rec["parent_document_id"] != passage_id(parent_text):
                errors.append(f"{where}: parent_document_id khác hash của parent_text")
            if offsets_ok and (end > len(parent_text) or parent_text[start:end] != text):
                errors.append(f"{where}: slice parent_text[chunk_start_char:chunk_end_char] khác text")
    return errors


# ---------- QARecord (1.2) ----------

def validate_qa_record(rec: object, where: str = "QARecord", *,
                       passage_texts: Mapping[str, str] | None = None) -> list[str]:
    errors = _field_errors(rec, QA_RECORD_FIELDS, where)
    if not _has_all(rec, QA_RECORD_FIELDS):
        return errors
    where = _label(rec, where)

    if not _is_nonempty_str(rec["query_id"]):
        errors.append(f"{where}: query_id phải là chuỗi khác rỗng")
    if not _is_nonempty_str(rec["question"]):
        errors.append(f"{where}: question phải là chuỗi khác rỗng")
    if rec["source_dataset"] != "germanquad":
        errors.append(f"{where}: source_dataset phải là 'germanquad', nhận {rec['source_dataset']!r}")
    if rec["source_split"] not in SPLITS:
        errors.append(f"{where}: source_split phải thuộc {SPLITS}, nhận {rec['source_split']!r}")

    # list rỗng hợp lệ (spec 3.5: gold answer rỗng → loại khỏi EM/F1, không chặn cả file); khi đó
    # gold_evidences cũng buộc phải rỗng vì mỗi evidence.answer_text phải thuộc reference_answers.
    refs = rec["reference_answers"]
    refs_ok = isinstance(refs, list) and all(_is_nonempty_str(a) for a in refs)
    if not refs_ok:
        errors.append(f"{where}: reference_answers phải là list gồm chuỗi khác rỗng")

    evidences = rec["gold_evidences"]
    if not isinstance(evidences, list):
        errors.append(f"{where}: gold_evidences phải là list (được phép rỗng)")
        return errors
    for i, ev in enumerate(evidences):
        ev_where = f"{where}.gold_evidences[{i}]"
        ev_errors = _field_errors(ev, GOLD_EVIDENCE_FIELDS, ev_where)
        errors += ev_errors
        if not _has_all(ev, GOLD_EVIDENCE_FIELDS):
            continue
        if not _is_nonempty_str(ev["passage_id"]):
            errors.append(f"{ev_where}: passage_id phải là chuỗi khác rỗng")
        if not _is_nonempty_str(ev["answer_text"]):
            errors.append(f"{ev_where}: answer_text phải là chuỗi khác rỗng")
            continue
        start, end = ev["answer_start"], ev["answer_end"]
        if not (_is_int(start) and start >= 0 and _is_int(end)):
            errors.append(f"{ev_where}: answer_start/answer_end phải là số nguyên, answer_start ≥ 0")
        elif end - start != len(ev["answer_text"]):
            errors.append(
                f"{ev_where}: answer_end - answer_start = {end - start} ≠ len(answer_text) = "
                f"{len(ev['answer_text'])} (answer_end là exclusive, text[start:end] phải == answer_text)"
            )
        if refs_ok and ev["answer_text"] not in refs:
            errors.append(f"{ev_where}: answer_text {ev['answer_text']!r} không có trong reference_answers")
        if passage_texts is not None and _is_nonempty_str(ev["passage_id"]):
            passage_text = passage_texts.get(ev["passage_id"])
            if not isinstance(passage_text, str):
                errors.append(f"{ev_where}: passage_id thiếu text trong passage_texts")
            elif _is_int(start) and start >= 0 and _is_int(end):
                if end > len(passage_text) or passage_text[start:end] != ev["answer_text"]:
                    errors.append(f"{ev_where}: slice passage[answer_start:answer_end] khác answer_text")
    return errors


# ---------- RetrievalTrainingRecord (1.3) ----------

def _text_id_errors(texts, ids, text_field: str, id_field: str, where: str) -> list[str]:
    """Hai list chunk/ID không rỗng, song song; không tra corpus cho hard negative."""
    errors = []
    if not isinstance(texts, list) or not texts:
        errors.append(f"{where}: {text_field} phải là list khác rỗng")
    if not isinstance(ids, list) or not ids:
        errors.append(f"{where}: {id_field} phải là list khác rỗng")
    if not isinstance(texts, list) or not isinstance(ids, list):
        return errors
    if len(texts) != len(ids):
        errors.append(f"{where}: {text_field}/{id_field} phải có cùng số phần tử")
    for i, text in enumerate(texts):
        if not _is_nonempty_str(text):
            errors.append(f"{where}.{text_field}[{i}]: phải là chuỗi khác rỗng")
    for i, pid in enumerate(ids):
        if not _is_hash_id(pid):
            errors.append(f"{where}.{id_field}[{i}]: phải là 16 ký tự hex thường")
        elif i < len(texts) and _is_nonempty_str(texts[i]) and pid != passage_id(texts[i]):
            errors.append(f"{where}.{id_field}[{i}]: khác hash của {text_field}[{i}]")
    return errors


def validate_retrieval_training_record(rec: object,
                                      where: str = "RetrievalTrainingRecord") -> list[str]:
    errors = _field_errors(rec, TRAINING_RECORD_FIELDS, where)
    if not _has_all(rec, TRAINING_RECORD_FIELDS):
        return errors
    where = _label(rec, where)
    for field in ("query_id", "question"):
        if not _is_nonempty_str(rec[field]):
            errors.append(f"{where}: {field} phải là chuỗi khác rỗng")
    if rec["source_split"] not in SPLITS:
        errors.append(f"{where}: source_split phải thuộc {SPLITS}")
    errors += _text_id_errors(rec["positive_texts"], rec["positive_passage_ids"],
                              "positive_texts", "positive_passage_ids", where)
    groups = rec["hard_negative_groups"]
    if not isinstance(groups, list):
        errors.append(f"{where}: hard_negative_groups phải là list (được phép rỗng)")
        return errors
    for i, group in enumerate(groups):
        group_where = f"{where}.hard_negative_groups[{i}]"
        errors += _field_errors(group, HARD_NEGATIVE_GROUP_FIELDS, group_where)
        if not _has_all(group, HARD_NEGATIVE_GROUP_FIELDS):
            continue
        if not _is_hash_id(group["source_context_id"]):
            errors.append(f"{group_where}: source_context_id phải là 16 ký tự hex thường")
        errors += _text_id_errors(group["texts"], group["passage_ids"], "texts", "passage_ids", group_where)
    return errors


# ---------- RetrievalResult (1.4) ----------

def validate_retrieval_result(rec, where: str = "RetrievalResult") -> list[str]:
    errors = _field_errors(rec, RETRIEVAL_RESULT_FIELDS, where)
    if not _has_all(rec, RETRIEVAL_RESULT_FIELDS):
        return errors
    if not _is_nonempty_str(rec["query_id"]):
        errors.append(f"{where}: query_id phải là chuỗi khác rỗng")
    if not _is_nonempty_str(rec["passage_id"]):
        errors.append(f"{where}: passage_id phải là chuỗi khác rỗng")
    if not (_is_int(rec["rank"]) and rec["rank"] >= 1):
        errors.append(f"{where}: rank phải là số nguyên ≥ 1 (rank bắt đầu từ 1), nhận {rec['rank']!r}")
    if not _is_number(rec["score"]):
        errors.append(f"{where}: score phải là số hữu hạn, nhận {rec['score']!r}")
    if rec["retriever_name"] not in RETRIEVER_NAMES:
        errors.append(f"{where}: retriever_name phải thuộc {RETRIEVER_NAMES}, nhận {rec['retriever_name']!r}")
    return errors


def validate_ranked_list(results, query_id: str, where: str, required_retriever: str | None = None) -> list[str]:
    """Danh sách kết quả của 1 query, đã theo thứ tự rank: rank 1..n liên tục,
    không trùng passage_id, score không tăng (lớn hơn = tốt hơn)."""
    if not isinstance(results, list):
        return [f"{where}: phải là list RetrievalResult"]
    errors = []
    for i, r in enumerate(results):
        errors += validate_retrieval_result(r, f"{where}[{i}]")
    if errors:
        return errors

    ranks = [r["rank"] for r in results]
    if ranks != list(range(1, len(results) + 1)):
        errors.append(f"{where}: rank phải là 1..{len(results)} liên tục theo thứ tự, nhận {ranks[:10]}")
    dup = sorted(p for p, c in Counter(r["passage_id"] for r in results).items() if c > 1)
    if dup:
        errors.append(f"{where}: passage_id trùng trong cùng 1 query: {dup[:5]}")
    for a, b in zip(results, results[1:]):
        if b["score"] > a["score"]:
            errors.append(
                f"{where}: score phải không tăng theo rank (rank {a['rank']}={a['score']} < "
                f"rank {b['rank']}={b['score']})"
            )
            break
    wrong_q = sorted({r["query_id"] for r in results if r["query_id"] != query_id})
    if wrong_q:
        errors.append(f"{where}: có kết quả mang query_id khác {query_id!r}: {wrong_q[:3]}")
    if required_retriever is not None:
        wrong = sorted({r["retriever_name"] for r in results if r["retriever_name"] != required_retriever})
        if wrong:
            errors.append(f"{where}: retriever_name phải là {required_retriever!r}, nhận {wrong}")
    return errors


# ---------- PredictionRecord (1.5) ----------

def validate_prediction_record(rec, where: str = "PredictionRecord") -> list[str]:
    errors = _field_errors(rec, PREDICTION_RECORD_FIELDS, where)
    if not _has_all(rec, PREDICTION_RECORD_FIELDS):
        return errors
    where = _label(rec, where)

    for f in ("run_id", "config_id", "query_id"):
        if not _is_nonempty_str(rec[f]):
            errors.append(f"{where}: {f} phải là chuỗi khác rỗng")
    if rec["corpus_id"] not in CORPUS_IDS:
        errors.append(f"{where}: corpus_id phải thuộc {CORPUS_IDS}, nhận {rec['corpus_id']!r}")

    for field in ("retrieval_k", "reader_k"):
        if not (_is_int(rec[field]) and rec[field] >= 1):
            errors.append(f"{where}: {field} phải là số nguyên ≥ 1")
    if (_is_int(rec["retrieval_k"]) and _is_int(rec["reader_k"])
            and rec["reader_k"] > rec["retrieval_k"]):
        errors.append(f"{where}: reader_k không được lớn hơn retrieval_k")

    status, stage = rec["status"], rec["error_stage"]
    if status not in STATUSES:
        errors.append(f"{where}: status phải thuộc {STATUSES}, nhận {status!r}")
    elif status == "ok" and stage is not None:
        errors.append(f"{where}: status='ok' thì error_stage phải là None, nhận {stage!r}")
    elif status == "error" and stage not in ERROR_STAGES:
        errors.append(f"{where}: status='error' thì error_stage phải thuộc {ERROR_STAGES}, nhận {stage!r}")

    query_id = rec["query_id"]
    errors += validate_ranked_list(rec["retrieved"], query_id, f"{where}.retrieved")
    if rec["reranked"] is not None:
        errors += validate_ranked_list(rec["reranked"], query_id, f"{where}.reranked", "cross_encoder")

    values = [rec[f] for f in ANSWER_FIELDS]
    n_none = sum(v is None for v in values)
    if 0 < n_none < len(values):
        errors.append(f"{where}: 5 field đáp án {ANSWER_FIELDS} phải cùng None (không có đáp án) hoặc cùng có giá trị")
    elif n_none == 0:
        if status == "error":
            errors.append(f"{where}: status='error' thì các field đáp án phải là None")
        text = rec["predicted_answer_text"]
        start, end = rec["predicted_answer_start"], rec["predicted_answer_end"]
        if not _is_nonempty_str(text):
            errors.append(f"{where}: predicted_answer_text phải khác rỗng — không có đáp án thì dùng None, không dùng ''")
        elif not (_is_int(start) and start >= 0 and _is_int(end)):
            errors.append(f"{where}: predicted_answer_start/end phải là số nguyên, start ≥ 0")
        elif end - start != len(text):
            errors.append(
                f"{where}: predicted_answer_end - predicted_answer_start = {end - start} ≠ "
                f"len(predicted_answer_text) = {len(text)} (end là exclusive)"
            )
        if not _is_nonempty_str(rec["predicted_passage_id"]):
            errors.append(f"{where}: predicted_passage_id phải là chuỗi khác rỗng")
        prob = rec["reader_span_prob"]
        if not (_is_number(prob) and 0.0 <= prob <= 1.0):
            errors.append(f"{where}: reader_span_prob phải thuộc [0, 1], nhận {prob!r}")

    latency = rec["latency_ms"]
    lat_errors = _field_errors(latency, LATENCY_FIELDS, f"{where}.latency_ms")
    errors += lat_errors
    if _has_all(latency, LATENCY_FIELDS) and status in STATUSES:
        for f in LATENCY_FIELDS:
            v = latency[f]
            if v is not None and not (_is_number(v) and v >= 0):
                errors.append(f"{where}.latency_ms.{f}: phải là số ≥ 0 hoặc None, nhận {v!r}")
        if status == "ok":
            for f in ("retrieval", "read", "total"):
                if latency[f] is None:
                    errors.append(f"{where}.latency_ms.{f}: status='ok' thì phải có giá trị")
            if rec["reranked"] is None and latency["rerank"] is not None:
                errors.append(f"{where}.latency_ms.rerank: không dùng reranker (reranked=None) thì phải là None")
            if rec["reranked"] is not None and latency["rerank"] is None:
                errors.append(f"{where}.latency_ms.rerank: có dùng reranker thì phải có giá trị")
    return errors

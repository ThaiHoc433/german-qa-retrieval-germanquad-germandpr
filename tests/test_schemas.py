import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.schemas import (
    validate_prediction_record,
    validate_qa_record,
    validate_ranked_list,
    validate_retrieval_result,
)


def qa_record(**overrides):
    rec = {
        "query_id": "gq-test-1",
        "question": "Wer entwickelte die Relativitätstheorie?",
        "gold_evidences": [
            {"passage_id": "p1", "answer_text": "Albert Einstein", "answer_start": 10, "answer_end": 25}
        ],
        "reference_answers": ["Albert Einstein", " Albert Einstein"],
        "source_dataset": "germanquad",
        "source_split": "test",
    }
    rec.update(overrides)
    return rec


def rr(qid, pid, rank, score, name="bm25"):
    return {"query_id": qid, "passage_id": pid, "rank": rank, "score": score, "retriever_name": name}


def prediction(**overrides):
    rec = {
        "run_id": "bm25_base_rerank-off__ckpt-none__seed42",
        "config_id": "bm25_base_rerank-off",
        "corpus_id": "base",
        "query_id": "gq-test-1",
        "status": "ok",
        "error_stage": None,
        "retrieved": [rr("gq-test-1", "p1", 1, 9.0), rr("gq-test-1", "p2", 2, 7.5)],
        "reranked": None,
        "predicted_answer_text": "Albert Einstein",
        "predicted_passage_id": "p1",
        "predicted_answer_start": 10,
        "predicted_answer_end": 25,
        "reader_span_prob": 0.8,
        "latency_ms": {"retrieval": 1.0, "rerank": None, "read": 5.0, "total": 6.5},
    }
    rec.update(overrides)
    return rec


def has(errors, fragment):
    return any(fragment in e for e in errors)


# ---------- QARecord ----------

def test_valid_qa_record_passes():
    assert validate_qa_record(qa_record()) == []


def test_qa_record_with_empty_gold_evidences_is_valid():
    assert validate_qa_record(qa_record(gold_evidences=[])) == []


def test_old_round1_schema_is_rejected_with_specific_fields():
    old = {
        "query_id": "gq-test-55538",
        "question": "Was ist die Hauptstadt der Marshallinseln?",
        "gold_passage_ids": ["7091e19c075e5aa6"],
        "answers": [{"text": "Majuro", "answer_start": 304}],
        "source_dataset": "germanquad",
        "source_split": "test",
    }
    errors = validate_qa_record(old)
    assert has(errors, "thiếu field 'gold_evidences'")
    assert has(errors, "thiếu field 'reference_answers'")
    assert has(errors, "field 'gold_passage_ids' không có trong schema")
    assert has(errors, "field 'answers' không có trong schema")


def test_evidence_offsets_must_match_answer_length_exclusive_end():
    bad = qa_record(gold_evidences=[
        {"passage_id": "p1", "answer_text": "Albert Einstein", "answer_start": 10, "answer_end": 24}
    ])
    assert has(validate_qa_record(bad), "≠ len(answer_text)")


def test_evidence_answer_text_must_be_a_reference_answer():
    bad = qa_record(gold_evidences=[
        {"passage_id": "p1", "answer_text": "Einstein", "answer_start": 0, "answer_end": 8}
    ])
    assert has(validate_qa_record(bad), "không có trong reference_answers")


def test_invalid_split_and_dataset():
    errors = validate_qa_record(qa_record(source_split="valid", source_dataset="germandpr"))
    assert has(errors, "source_split")
    assert has(errors, "source_dataset")


def test_empty_reference_answers_only_valid_without_evidences():
    # spec 3.5: gold answer rỗng → record hợp lệ nhưng bị loại khỏi eval (không chặn cả file)
    assert validate_qa_record(qa_record(gold_evidences=[], reference_answers=[])) == []
    assert has(validate_qa_record(qa_record(reference_answers=[])), "không có trong reference_answers")
    assert has(validate_qa_record(qa_record(reference_answers=["", "x"])), "reference_answers")


# ---------- RetrievalResult + ranked list ----------

def test_valid_retrieval_result_passes():
    assert validate_retrieval_result(rr("q", "p", 1, 3.2)) == []


def test_rank_must_start_at_1_and_not_be_bool():
    assert has(validate_retrieval_result(rr("q", "p", 0, 3.2)), "rank")
    assert has(validate_retrieval_result(rr("q", "p", True, 3.2)), "rank")


def test_score_must_be_finite_and_retriever_known():
    assert has(validate_retrieval_result(rr("q", "p", 1, float("nan"))), "score")
    assert has(validate_retrieval_result(rr("q", "p", 1, 1.0, name="tfidf")), "retriever_name")


def test_ranked_list_rules():
    good = [rr("q", "a", 1, 5.0), rr("q", "b", 2, 5.0), rr("q", "c", 3, 1.0)]  # đồng điểm hợp lệ
    assert validate_ranked_list(good, "q", "L") == []
    assert has(validate_ranked_list([rr("q", "a", 1, 5.0), rr("q", "b", 3, 4.0)], "q", "L"), "liên tục")
    assert has(validate_ranked_list([rr("q", "a", 1, 5.0), rr("q", "a", 2, 4.0)], "q", "L"), "trùng")
    assert has(validate_ranked_list([rr("q", "a", 1, 1.0), rr("q", "b", 2, 4.0)], "q", "L"), "không tăng")
    assert has(validate_ranked_list([rr("other", "a", 1, 1.0)], "q", "L"), "query_id khác")
    assert has(validate_ranked_list(good, "q", "L", required_retriever="cross_encoder"), "cross_encoder")


# ---------- PredictionRecord ----------

def test_valid_prediction_with_and_without_answer():
    assert validate_prediction_record(prediction()) == []
    no_answer = prediction(
        predicted_answer_text=None, predicted_passage_id=None, predicted_answer_start=None,
        predicted_answer_end=None, reader_span_prob=None,
    )
    assert validate_prediction_record(no_answer) == []


def test_valid_error_record():
    err = prediction(
        status="error", error_stage="read", predicted_answer_text=None, predicted_passage_id=None,
        predicted_answer_start=None, predicted_answer_end=None, reader_span_prob=None,
        latency_ms={"retrieval": 1.0, "rerank": None, "read": None, "total": None},
    )
    assert validate_prediction_record(err) == []


def test_status_error_stage_consistency():
    assert has(validate_prediction_record(prediction(error_stage="read")), "error_stage phải là None")
    errors = validate_prediction_record(prediction(status="error", error_stage=None))
    assert has(errors, "error_stage phải thuộc")
    assert has(validate_prediction_record(prediction(status="error", error_stage="read")), "phải là None")


def test_answer_fields_all_or_none():
    assert has(validate_prediction_record(prediction(reader_span_prob=None)), "cùng None")


def test_empty_string_answer_is_rejected_use_none():
    bad = prediction(predicted_answer_text="", predicted_answer_start=0, predicted_answer_end=0)
    assert has(validate_prediction_record(bad), "dùng None")


def test_answer_end_is_exclusive_and_matches_text_length():
    assert has(validate_prediction_record(prediction(predicted_answer_end=24)), "end là exclusive")


def test_reader_span_prob_range():
    assert has(validate_prediction_record(prediction(reader_span_prob=1.5)), "reader_span_prob")


def test_rerank_latency_consistent_with_reranked():
    assert has(
        validate_prediction_record(prediction(latency_ms={"retrieval": 1.0, "rerank": 2.0, "read": 5.0, "total": 8.0})),
        "reranked=None",
    )
    with_rerank = prediction(reranked=[rr("gq-test-1", "p1", 1, 0.9, name="cross_encoder")])
    assert has(validate_prediction_record(with_rerank), "có dùng reranker")


def test_reranked_items_must_be_cross_encoder():
    bad = prediction(
        reranked=[rr("gq-test-1", "p1", 1, 0.9, name="bm25")],
        latency_ms={"retrieval": 1.0, "rerank": 2.0, "read": 5.0, "total": 8.0},
    )
    assert has(validate_prediction_record(bad), "cross_encoder")


def test_unknown_corpus_id_and_missing_field():
    assert has(validate_prediction_record(prediction(corpus_id="huge")), "corpus_id")
    rec = prediction()
    del rec["latency_ms"]
    assert has(validate_prediction_record(rec), "thiếu field 'latency_ms'")

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import schemas
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
        "retrieval_k": 20,
        "reader_k": 5,
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


# V1.1: literal fixtures from the reviewed T5.0/T5.1 task card.
BERLIN_ID = "fb38a93fd89e1a5f"
ULM_ID = "37bfc8ba5ec56136"


def passage_record(**overrides):
    rec = {
        "passage_id": BERLIN_ID, "text": "Berlin", "normalized_text": "berlin",
        "source": "germanquad", "title": "Berlin", "char_len": 6,
        "parent_document_id": BERLIN_ID, "chunk_start_char": 2, "chunk_end_char": 8,
    }
    rec.update(overrides)
    return rec


def training_record(**overrides):
    rec = {
        "query_id": "gdpr-train-0", "question": "Welche Stadt?", "source_split": "train",
        "positive_texts": ["Berlin"], "positive_passage_ids": [BERLIN_ID],
        "hard_negative_groups": [
            {"source_context_id": ULM_ID, "texts": ["Ulm"], "passage_ids": [ULM_ID]}
        ],
    }
    rec.update(overrides)
    return rec


def test_passage_validation_checks_parent_slice_without_mutation():
    rec = passage_record()
    before = deepcopy(rec)
    assert schemas.validate_passage_record(rec, parent_text="  Berlin  ") == []
    assert rec == before


@pytest.mark.parametrize("field,value", [
    ("passage_id", ULM_ID), ("normalized_text", "Berlin"), ("text", ""),
    ("text", None), ("source", "germandpr_hard_negative"), ("source", "unknown"),
    ("title", 7), ("char_len", 7), ("char_len", True),
    ("parent_document_id", None), ("parent_document_id", "not-a-hash"),
    ("chunk_start_char", -1), ("chunk_start_char", True),
    ("chunk_end_char", 9), ("chunk_end_char", 8.0),
])
def test_passage_rejects_invalid_field(field, value):
    assert has(schemas.validate_passage_record(passage_record(**{field: value})), field)


@pytest.mark.parametrize("field", ["parent_document_id", "chunk_start_char", "chunk_end_char"])
def test_passage_requires_provenance_fields(field):
    rec = passage_record()
    del rec[field]
    assert has(schemas.validate_passage_record(rec), field)


def test_passage_checks_parent_identity_and_same_length_shift():
    assert has(schemas.validate_passage_record(passage_record(), parent_text="  Munich  "), "parent_document_id")
    shifted = passage_record(chunk_start_char=3, chunk_end_char=9)
    assert schemas.validate_passage_record(shifted) == []  # Structure alone cannot detect this.
    assert has(schemas.validate_passage_record(shifted, parent_text="  Berlin  "), "chunk_start_char")


def test_passage_allows_missing_title():
    assert schemas.validate_passage_record(passage_record(title=None)) == []


@pytest.mark.parametrize("name", ["validate_passage_record", "validate_retrieval_training_record"])
@pytest.mark.parametrize("rec", [None, [], "record", {}])
def test_new_validators_report_malformed_records_instead_of_crashing(name, rec):
    validate = getattr(schemas, name, None)
    assert callable(validate), f"Missing V1.1 validator: {name}"
    assert validate(rec)


def test_training_groups_do_not_require_negative_ids_in_a_corpus():
    rec = training_record()
    before = deepcopy(rec)
    assert schemas.validate_retrieval_training_record(rec) == []
    assert rec == before
    assert schemas.validate_retrieval_training_record(training_record(hard_negative_groups=[])) == []


def test_training_rejects_missing_last_positive_id_with_valid_prefix():
    rec = training_record(
        positive_texts=["Berlin", "Ulm"],
        positive_passage_ids=[BERLIN_ID, ULM_ID],
        hard_negative_groups=[],
    )
    assert schemas.validate_retrieval_training_record(rec) == []
    # The remaining pair still has the correct hash: only length validation catches this.
    rec["positive_passage_ids"].pop()
    errors = schemas.validate_retrieval_training_record(rec)
    assert has(errors, "positive_texts/positive_passage_ids")


def test_training_rejects_missing_last_group_id_with_valid_prefix():
    rec = training_record(
        hard_negative_groups=[{
            "source_context_id": ULM_ID,
            "texts": ["Ulm", "Berlin"],
            "passage_ids": [ULM_ID, BERLIN_ID],
        }],
    )
    assert schemas.validate_retrieval_training_record(rec) == []
    rec["hard_negative_groups"][0]["passage_ids"].pop()
    errors = schemas.validate_retrieval_training_record(rec)
    assert has(errors, "hard_negative_groups[0]: texts/passage_ids")


@pytest.mark.parametrize("overrides,field", [
    ({"positive_texts": [], "positive_passage_ids": []}, "positive_texts"),
    ({"positive_passage_ids": []}, "positive_passage_ids"),
    ({"positive_passage_ids": [None]}, "positive_passage_ids"),
    ({"positive_passage_ids": [ULM_ID]}, "positive_passage_ids"),
    ({"positive_texts": [42]}, "positive_texts"),
    ({"positive_texts": [""]}, "positive_texts"),
    ({"positive_texts": "Berlin"}, "positive_texts"),
    ({"query_id": ""}, "query_id"), ({"question": None}, "question"),
    ({"source_split": "valid"}, "source_split"),
    ({"hard_negative_groups": None}, "hard_negative_groups"),
])
def test_training_rejects_invalid_positive_pairs_and_fields(overrides, field):
    assert has(schemas.validate_retrieval_training_record(training_record(**overrides)), field)


@pytest.mark.parametrize("group,field", [
    (None, "hard_negative_groups"), ({}, "source_context_id"),
    ({"source_context_id": None, "texts": ["Ulm"], "passage_ids": [ULM_ID]}, "source_context_id"),
    ({"source_context_id": ULM_ID, "texts": ["Ulm"], "passage_ids": []}, "passage_ids"),
    ({"source_context_id": ULM_ID, "texts": ["Ulm"], "passage_ids": [BERLIN_ID]}, "passage_ids"),
    ({"source_context_id": ULM_ID, "texts": [], "passage_ids": []}, "texts"),
    ({"source_context_id": ULM_ID, "texts": [None], "passage_ids": [ULM_ID]}, "texts"),
])
def test_training_rejects_broken_context_groups(group, field):
    rec = training_record(hard_negative_groups=[group])
    assert has(schemas.validate_retrieval_training_record(rec), field)


def test_qa_checks_actual_slice_when_passage_texts_are_supplied():
    rec = qa_record(
        reference_answers=["Berlin"],
        gold_evidences=[{"passage_id": BERLIN_ID, "answer_text": "Berlin", "answer_start": 0, "answer_end": 6}],
    )
    assert validate_qa_record(rec, passage_texts={BERLIN_ID: "Berlin"}) == []
    before = deepcopy(rec)
    rec["gold_evidences"][0].update(answer_start=1, answer_end=7)
    assert validate_qa_record(rec) == []
    assert has(validate_qa_record(rec, passage_texts={BERLIN_ID: "Berlin"}), "answer_start")
    assert rec["gold_evidences"][0]["answer_start"] == 1  # No silent offset repair.
    assert has(validate_qa_record(before, passage_texts={}), "passage_id")
    assert validate_qa_record(qa_record(gold_evidences=[], reference_answers=[]), passage_texts={}) == []


@pytest.mark.parametrize("evidence", [
    {"passage_id": [], "answer_text": "Berlin", "answer_start": 0, "answer_end": 6},
    {"passage_id": BERLIN_ID, "answer_text": "Berlin", "answer_start": "0", "answer_end": 6},
    {"passage_id": BERLIN_ID, "answer_text": "Berlin", "answer_start": 0, "answer_end": True},
])
def test_qa_malformed_evidence_still_reports_errors_with_passage_map(evidence):
    assert validate_qa_record(qa_record(reference_answers=["Berlin"], gold_evidences=[evidence]),
                              passage_texts={BERLIN_ID: "Berlin"})


@pytest.mark.parametrize("field", ["retrieval_k", "reader_k"])
def test_prediction_requires_explicit_budget(field):
    rec = prediction()
    del rec[field]
    assert has(validate_prediction_record(rec), f"thiếu field '{field}'")


@pytest.mark.parametrize("field,value", [
    ("retrieval_k", True), ("reader_k", False), ("reader_k", 0),
    ("retrieval_k", -1), ("retrieval_k", 20.0), ("reader_k", "5"),
    ("retrieval_k", None),
])
def test_prediction_rejects_nonpositive_or_noninteger_budgets(field, value):
    # The paired valid record matters: rejecting all k fields would be a false positive.
    assert validate_prediction_record(prediction()) == []
    assert has(validate_prediction_record(prediction(**{field: value})), field)


def test_prediction_reader_budget_cannot_exceed_retrieval_budget():
    assert validate_prediction_record(prediction()) == []
    assert has(validate_prediction_record(prediction(reader_k=21)), "reader_k")
